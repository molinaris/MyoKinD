"""Tests for figures/plot_quality_summary.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from figures.plot_quality_summary import METRIC_COLUMNS, channel_snr_rejection_summary, make_figure, summarize


def _quality_table(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "participant": ["Sub001"] * n,
        "task": ["1"] * n,
        "task_freq": ["05"] * n,
        "trial": list(range(n)),
        "snr": [14.89] * n,
        "smr": [25.18] * n,
        "spr": [20.99] * n,
        "thdn": [8.92] * n,
        "coh": [0.88] * n,
        "phase_lag": [-0.33] * n
    })


def test_summarize_reports_mean_std_n_per_metric():
    summary = summarize(_quality_table(n=10))

    assert set(summary["metric"]) == set(METRIC_COLUMNS)
    for _, row in summary.iterrows():
        assert row["n"] == 10
        assert row["std"] == pytest.approx(0.0, abs=1e-9)  # constant column in the fixture

    snr_row = summary[summary["metric"] == "snr"].iloc[0]
    assert snr_row["mean"] == pytest.approx(14.89)


def test_summarize_ignores_missing_values():
    table = _quality_table(n=10)
    table.loc[0, "smr"] = float("nan")

    summary = summarize(table)
    smr_row = summary[summary["metric"] == "smr"].iloc[0]
    assert smr_row["n"] == 9


def test_make_figure_returns_a_2x3_grid():
    fig = make_figure(_quality_table())
    assert len(fig.get_axes()) == 6


def _channel_table(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a minimal per-channel table: list of (participant, snr_db)."""
    return pd.DataFrame({
        "participant": [p for p, _ in rows],
        "task": ["1"] * len(rows), "frequency": ["05"] * len(rows), "repetition": [0] * len(rows),
        "muscle": ["EDC"] * len(rows), "channel": list(range(len(rows))),
        "snr_db": [s for _, s in rows]
    })


def test_channel_snr_rejection_summary_counts_instances_below_threshold():
    table = _channel_table([
        ("Sub001", 0.5), ("Sub001", 1.0), ("Sub001", 5.0),   # 2/3 below 1.8 dB
        ("Sub002", 10.0), ("Sub002", 0.1)                    # 1/2 below 1.8 dB
    ])

    summary = channel_snr_rejection_summary(table, threshold_db=1.8)

    assert summary["n_channel_trials"] == 5
    assert summary["n_below_threshold"] == 3
    assert summary["pct_below_threshold"] == pytest.approx(60.0)
    assert summary["max_below_threshold_per_participant"] == 2  # Sub001


def test_channel_snr_rejection_summary_none_below_threshold():
    table = _channel_table([("Sub001", 10.0), ("Sub001", 12.0)])

    summary = channel_snr_rejection_summary(table, threshold_db=1.8)

    assert summary["n_below_threshold"] == 0
    assert summary["pct_below_threshold"] == 0.0
    assert summary["max_below_threshold_per_participant"] == 0
