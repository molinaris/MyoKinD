"""Tests for validation/phase_lag_analysis.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.phase_lag_analysis import (add_derived_columns, bimodality_report, icc1_one_way,
                                           mode_coherence_summary, ratio_test_h1,
                                           task_frequency_crosstab, wrapping_test_h2)


def test_add_derived_columns_maps_frequency_task_name_and_thumb():
    df = pd.DataFrame({
        "task_freq": [5, 75], "task": [1, 4], "phase_lag": [-0.5, -0.5]
    })
    out = add_derived_columns(df)

    assert list(out["frequency_hz"]) == [0.50, 0.75]
    assert list(out["task_name"]) == ["Index", "Thumb"]
    assert list(out["involves_thumb"]) == [False, True]
    # tau_ms = -phase_lag / (2*pi*f) * 1000
    expected_tau_050 = -(-0.5) / (2 * np.pi * 0.50) * 1000
    assert out["tau_ms"].iloc[0] == pytest.approx(expected_tau_050)


def _fixed_delay_table(tau_s: float) -> pd.DataFrame:
    """Synthetic quality table where phase_lag is exactly what a single fixed
    time delay `tau_s` would produce at each of the two task frequencies."""
    phase_050 = -2 * np.pi * 0.50 * tau_s
    phase_075 = -2 * np.pi * 0.75 * tau_s
    df = pd.DataFrame({
        "task_freq": [5] * 3 + [75] * 3, "task": ["1"] * 6,
        "phase_lag": [phase_050] * 3 + [phase_075] * 3
    })
    return add_derived_columns(df)


def test_ratio_test_h1_recovers_1_5_ratio_for_a_true_fixed_delay():
    # H1's own premise (phase proportional to frequency) built into the fixture -
    # confirms ratio_test_h1 correctly recovers the 1.5 ratio H1 predicts,
    # i.e. the test would (correctly) NOT reject H1 if it were true.
    df = _fixed_delay_table(tau_s=0.05)
    result = ratio_test_h1(df)
    assert result["phase_ratio_075_050"] == pytest.approx(1.5)


def test_ratio_test_h1_ratio_is_not_1_5_when_phase_is_frequency_independent():
    # The actual MyoKinD pattern: same phase at both frequencies (rejects H1).
    df = pd.DataFrame({
        "task_freq": [5] * 3 + [75] * 3, "task": ["1"] * 6,
        "phase_lag": [-0.54] * 3 + [-0.54] * 3
    })
    df = add_derived_columns(df)
    result = ratio_test_h1(df)
    assert result["phase_ratio_075_050"] == pytest.approx(1.0)


def test_wrapping_test_h2_converges_when_wrapping_is_the_true_mechanism():
    """Trials just below the 0.50 Hz wrap threshold (tau=1.0s) stay negative;
    trials just above it wrap into positive apparent phase. Unwrapping the
    positive group should land close to its true (>1s) delay, and clearly
    closer to the negative group's delay than the naive as-is delay is."""
    f = 0.50
    taus_neg_s = [0.85, 0.90, 0.95]
    taus_pos_true_s = [1.05, 1.10, 1.15]
    phases_neg = [-2 * np.pi * f * t for t in taus_neg_s]
    phases_pos_wrapped = [-2 * np.pi * f * t + 2 * np.pi for t in taus_pos_true_s]

    df = pd.DataFrame({
        "task_freq": [5] * 6, "task": ["1"] * 6, "phase_lag": phases_neg + phases_pos_wrapped
    })
    df = add_derived_columns(df)

    result = wrapping_test_h2(df)
    assert result["median_tau_ms_unwrapped_positive_group"] == pytest.approx(1100.0)
    assert result["median_tau_ms_asis_nonpositive_group"] == pytest.approx(900.0)
    # unwrapping brings the two groups within 200 ms of each other...
    gap_unwrapped = abs(result["median_tau_ms_unwrapped_positive_group"]
                        - result["median_tau_ms_asis_nonpositive_group"])
    # ...vastly closer than the as-is (not unwrapped) delay, which has the wrong sign entirely
    gap_asis = abs(result["median_tau_ms_asis_positive_group"]
                   - result["median_tau_ms_asis_nonpositive_group"])
    assert gap_unwrapped < gap_asis


def test_task_frequency_crosstab_shape_and_values():
    df = pd.DataFrame({
        "task_freq": [5, 75, 5, 75], "task": [1, 1, 2, 2], "phase_lag": [-0.1, -0.2, -0.3, -0.4]
    })
    df = add_derived_columns(df)
    cross = task_frequency_crosstab(df)

    assert set(cross.index) == {"Index", "Middle"}
    assert cross.loc["Index", 0.50] == pytest.approx(-0.1)
    assert cross.loc["Middle", 0.75] == pytest.approx(-0.4)


def test_bimodality_report_does_not_reject_unimodal_gaussian():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    result = bimodality_report(x)
    assert result["dip_pvalue"] > 0.05


def test_bimodality_report_rejects_unimodality_for_well_separated_mixture():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-5, 1, 1000), rng.normal(5, 1, 1000)])
    result = bimodality_report(x)
    assert result["dip_pvalue"] < 0.05
    assert result["delta_bic_k1_minus_k2"] > 0  # k=2 clearly preferred over k=1


def test_mode_coherence_summary_separates_the_two_coherence_levels():
    n = 20
    df = pd.DataFrame({
        "phase_lag": [-5.0] * n + [5.0] * n,
        "coh": [0.9] * n + [0.3] * n
    })
    summary = mode_coherence_summary(df)
    assert len(summary) == 2
    # mode 0 = lower phase_lag mean (-5, high coherence group)
    assert summary.loc[0, "mean"] == pytest.approx(0.9)
    assert summary.loc[1, "mean"] == pytest.approx(0.3)


def _grouped_table(group_means: list[float], n_per_group: int = 20, within_spread: float = 0.0) -> pd.DataFrame:
    rows = []
    for i, mean in enumerate(group_means):
        offsets = [within_spread, -within_spread] * (n_per_group // 2)
        rows += [{"participant": f"Sub{i}", "phase_lag": mean + o} for o in offsets[:n_per_group]]
    return pd.DataFrame(rows)


def test_icc1_one_way_is_zero_when_all_groups_share_the_same_mean():
    df = _grouped_table(group_means=[3.0, 3.0, 3.0], within_spread=1.0)
    result = icc1_one_way(df)
    assert result["icc1"] == pytest.approx(0.0, abs=1e-9)


def test_icc1_one_way_is_near_one_when_variance_is_almost_entirely_between_group():
    df = _grouped_table(group_means=[0.0, 10.0, 20.0], within_spread=0.01)
    result = icc1_one_way(df)
    assert result["icc1"] > 0.99
    assert result["p_anova"] < 0.05
