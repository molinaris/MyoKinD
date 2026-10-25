#!/usr/bin/env python3
"""Phase-lag (phase_lag) distribution analysis: reproduces the numbers behind
the manuscript's discussion of its elevated standard deviation (-0.52 +/- 1.01
rad, Fig. 5E), and the discrimination between three competing explanations,
using the per-trial quality table from validation/aggregate_quality_metrics.py.

Three artifact/instrumentation hypotheses are tested and, on this dataset,
all three are rejected/not supported:

- H1 (frequency-dependent fixed delay): a fixed time delay `tau` predicts
  phase proportional to frequency (phi = -2*pi*f*tau), i.e. a ~1.5x ratio
  between the mean phase at 0.75 Hz and at 0.50 Hz. Observed ratio is ~1.0.
- H2 (phase wrapping): large delays wrapped into (-pi, pi] would reappear
  with inverted sign; unwrapping (+2*pi) the positive-phase trials does not
  reconcile their delay with that of the negative-phase trials.
- H3 (task-specific sign-convention bug): no task, and in particular no
  thumb-involving-task subgroup, is offset by ~pi rad from the others.

Bimodality of `phase_lag` itself (not `tau_ms` - `tau_ms` differs
systematically between the two task frequencies, so testing bimodality on it
pooled across frequencies would reintroduce the very frequency-aggregation
artifact H1 rules out) is assessed with the Hartigan dip test alongside a
Gaussian-mixture BIC comparison (k=1,2,3), reported side by side rather than
selectively - the dip test never rejects unimodality on this dataset, even
though a 2-component mixture has consistently lower BIC, pooled and within
each frequency separately.

Finally, ICC(1) - the one-way random-effects intraclass correlation (Fisher,
1925; Searle, Casella & McCulloch, 1992, sec. 3.4, unbalanced case) - answers
whether phase_lag's spread is a between-participant or a within-participant
(trial-to-trial) effect: this is what actually determines whether a
participant-level covariate could ever explain it.

Usage:
    python phase_lag_analysis.py --input-csv quality_summary.csv
    python phase_lag_analysis.py --input-csv quality_summary.csv --output-csv phase_lag_analysis_summary.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import diptest
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.load_hdsemg import TASK_NAMES  # noqa: E402

FREQ_CODE_TO_HZ = {5: 0.50, 75: 0.75}  # quality_summary.csv's task_freq is read as int (05/075 -> 5/75)

# Measured one-way latency of the UDP trigger from the MPSoC to the Vicon
# setup (see manuscript); used to predict the phase a fixed instrumentation
# delay alone would produce, for comparison against the observed magnitude.
UDP_LATENCY_S = 0.07706

THUMB_TASKS = {"4", "5", "6", "7", "8"}  # Thumb, I. Pinch, M. Pinch, Tripod, 5-Finger (see TASK_NAMES)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add frequency_hz, task_name, involves_thumb and tau_ms to a quality table."""
    df = df.copy()
    df["frequency_hz"] = df["task_freq"].map(FREQ_CODE_TO_HZ)
    if df["frequency_hz"].isna().any():
        raise ValueError("task_freq has values outside the expected {5, 75} (i.e. 0.50/0.75 Hz)")
    df["task_name"] = df["task"].astype(str).map(TASK_NAMES)
    df["involves_thumb"] = df["task"].astype(str).isin(THUMB_TASKS)
    # tau_ms: apparent delay implied by phase_lag at each trial's own frequency.
    # phi = -2*pi*f*tau  =>  tau = -phi / (2*pi*f)
    df["tau_ms"] = -df["phase_lag"] / (2 * np.pi * df["frequency_hz"]) * 1000
    return df


def phase_by_frequency_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/median of phase_lag and tau_ms, grouped by task frequency."""
    return df.groupby("frequency_hz")[["phase_lag", "tau_ms"]].agg(["mean", "std", "median"])


def ratio_test_h1(df: pd.DataFrame, udp_latency_s: float = UDP_LATENCY_S) -> dict:
    """H1 (fixed delay): ratio of mean phase_lag between 0.75 Hz and 0.50 Hz,
    which a frequency-independent delay would put at ~1.5 (the frequency
    ratio), plus the phase a fixed delay of `udp_latency_s` alone would predict."""
    mean_phase = df.groupby("frequency_hz")["phase_lag"].mean()
    phase_ratio = mean_phase[0.75] / mean_phase[0.50]

    mean_tau = df.groupby("frequency_hz")["tau_ms"].mean()
    tau_ratio = mean_tau[0.50] / mean_tau[0.75]

    phi_pred_050 = -2 * np.pi * 0.50 * udp_latency_s
    phi_pred_075 = -2 * np.pi * 0.75 * udp_latency_s
    observed_magnitude = mean_phase.abs().mean()

    return {
        "mean_phase_050": mean_phase[0.50], "mean_phase_075": mean_phase[0.75],
        "phase_ratio_075_050": phase_ratio,  # H1 predicts ~1.5
        "mean_tau_ms_050": mean_tau[0.50], "mean_tau_ms_075": mean_tau[0.75],
        "tau_ratio_050_075": tau_ratio,
        "udp_predicted_phase_050": phi_pred_050, "udp_predicted_phase_075": phi_pred_075,
        "observed_vs_udp_predicted_ratio": observed_magnitude / abs(phi_pred_050)
    }


def wrapping_test_h2(df: pd.DataFrame) -> dict:
    """H2 (wrapping): median tau implied by unwrapping (+2*pi) the positive-phase
    trials, vs. the as-is median tau of the negative-phase trials. Convergence
    would indicate the positive-phase trials are just wrapped large delays."""
    pos = df[df["phase_lag"] > 0]
    neg = df[df["phase_lag"] <= 0]
    tau_unwrapped_pos = (-pos["phase_lag"] + 2 * np.pi) / (2 * np.pi * pos["frequency_hz"]) * 1000

    return {
        "n_positive_phase": len(pos), "n_nonpositive_phase": len(neg),
        "median_tau_ms_unwrapped_positive_group": tau_unwrapped_pos.median(),
        "median_tau_ms_asis_nonpositive_group": neg["tau_ms"].median(),
        "median_tau_ms_asis_positive_group": pos["tau_ms"].median()
    }


def task_frequency_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """H3 (task-specific bug): mean phase_lag per (task, frequency); a sign bug
    in one task's representative-finger convention would show as a ~pi rad
    offset for that task relative to the others."""
    return df.pivot_table(index="task_name", columns="frequency_hz", values="phase_lag", aggfunc="mean")


def thumb_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """H3 (task-specific bug): thumb-involving vs. non-thumb-involving tasks."""
    return df.groupby("involves_thumb")["phase_lag"].agg(["mean", "std", "median", "count"])


def bimodality_report(x: np.ndarray) -> dict:
    """Hartigan dip test (H0: unimodal) plus a 1/2/3-component Gaussian-mixture
    BIC comparison, reported together - the dip test is the direct test for
    this question; BIC alone tends to prefer more components on skewed/
    heavy-tailed data even when it is genuinely unimodal."""
    x = np.asarray(x)
    dip, pval = diptest.diptest(x, boot_pval=True, seed=1, n_threads=1)

    X = x.reshape(-1, 1)
    bics = {k: GaussianMixture(n_components=k, random_state=0, n_init=10).fit(X).bic(X) for k in (1, 2, 3)}

    gm2 = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(X)
    order = np.argsort(gm2.means_.ravel())
    means = gm2.means_.ravel()[order]
    stds = np.sqrt(gm2.covariances_.ravel())[order]
    weights = gm2.weights_[order]

    return {
        "n": len(x), "dip": dip, "dip_pvalue": pval,
        "bic_k1": bics[1], "bic_k2": bics[2], "bic_k3": bics[3],
        "delta_bic_k1_minus_k2": bics[1] - bics[2],  # positive favors k=2 over k=1
        "delta_bic_k3_minus_k2": bics[3] - bics[2],  # positive favors k=2 over k=3
        "component_means": means, "component_stds": stds, "component_weights": weights
    }


def gmm_2component_modes(x: np.ndarray) -> np.ndarray:
    """Hard-assign each value of `x` to one of 2 Gaussian-mixture components
    (0 = lower mean, 1 = higher mean). Used to split trials into "modes" for
    downstream per-mode comparisons (e.g. coherence)."""
    X = np.asarray(x).reshape(-1, 1)
    gm2 = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(X)
    order = np.argsort(gm2.means_.ravel())
    inv_order = np.argsort(order)
    return inv_order[gm2.predict(X)]


def mode_coherence_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/std coherence and trial count for each of the 2 phase_lag modes
    (assigned by gmm_2component_modes on the pooled, both-frequencies data) -
    a mode that is really just decoupled/noisy trials would show much lower
    coherence than the other."""
    df = df.assign(mode=gmm_2component_modes(df["phase_lag"].to_numpy()))
    return df.groupby("mode")["coh"].agg(["mean", "std", "count"])


def icc1_one_way(df: pd.DataFrame, value_col: str = "phase_lag", group_col: str = "participant") -> dict:
    """ICC(1): one-way random-effects intraclass correlation (Fisher, 1925;
    Searle, Casella & McCulloch, 1992, sec. 3.4 for the unbalanced-group-size
    case), i.e. the fraction of `value_col`'s total variance attributable to
    `group_col` identity rather than to within-group (here: trial-to-trial)
    variation. Also returns the one-way ANOVA F-test for a `group_col` effect."""
    groups = [g[value_col].to_numpy() for _, g in df.groupby(group_col)]
    k, n_total = len(groups), len(df)
    grand_mean = df[value_col].mean()

    f_stat, p_anova = stats.f_oneway(*groups)

    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (n_total - k)

    n_per_group = np.array([len(g) for g in groups])
    n0 = (n_total - (n_per_group ** 2).sum() / n_total) / (k - 1)  # Fisher's adjustment for unbalanced groups
    var_between = max((msb - msw) / n0, 0.0)
    var_within = msw
    icc1 = var_between / (var_between + var_within)

    return {
        "k_groups": k, "n_total": n_total, "f_stat": f_stat, "p_anova": p_anova,
        "msb": msb, "msw": msw, "n0": n0,
        "var_between": var_between, "var_within": var_within, "icc1": icc1
    }


def subject_mean_bimodality(df: pd.DataFrame, group_col: str = "participant",
                            value_col: str = "phase_lag") -> dict:
    """Bimodality report (see bimodality_report) on the per-`group_col` means
    of `value_col` - i.e. is the trial-level split (if any) actually a split
    between groups (e.g. participants), or does it wash out once averaged
    within each group? n is small here (one value per group), so the dip test
    has low power - report accordingly, not as a confident negative."""
    means = df.groupby(group_col)[value_col].mean().to_numpy()
    return bimodality_report(means)


def _print_report(df: pd.DataFrame) -> None:
    print("=== 1. phase_lag / tau_ms by frequency ===")
    print(phase_by_frequency_summary(df))

    print("\n=== 2. H1 - fixed-delay ratio test ===")
    h1 = ratio_test_h1(df)
    print(f"mean phase_lag: 0.50 Hz = {h1['mean_phase_050']:.4f} rad, 0.75 Hz = {h1['mean_phase_075']:.4f} rad")
    print(f"phase ratio (0.75/0.50) = {h1['phase_ratio_075_050']:.4f}  <- H1 predicts ~1.5")
    print(f"tau ratio (0.50/0.75) = {h1['tau_ratio_050_075']:.4f}")
    print(f"UDP-latency-predicted phase: {h1['udp_predicted_phase_050']:.3f} rad @ 0.50 Hz, "
          f"{h1['udp_predicted_phase_075']:.3f} rad @ 0.75 Hz")
    print(f"observed magnitude is {h1['observed_vs_udp_predicted_ratio']:.1f}x the 0.50 Hz UDP prediction")

    print("\n=== 3. H2 - phase-wrapping test ===")
    h2 = wrapping_test_h2(df)
    print(f"n(phase_lag > 0) = {h2['n_positive_phase']}, n(phase_lag <= 0) = {h2['n_nonpositive_phase']}")
    print(f"median tau_ms, unwrapped, phase>0 group:  {h2['median_tau_ms_unwrapped_positive_group']:.1f} ms")
    print(f"median tau_ms, as-is, phase<=0 group:      {h2['median_tau_ms_asis_nonpositive_group']:.1f} ms")

    print("\n=== 4. H3 - task-structure test ===")
    print(task_frequency_crosstab(df))
    print()
    print(thumb_task_summary(df))

    print("\n=== 5. bimodality of phase_lag (pooled and by frequency) ===")
    for label, sub in (("pooled", df), ("0.50 Hz", df[df["frequency_hz"] == 0.50]),
                       ("0.75 Hz", df[df["frequency_hz"] == 0.75])):
        r = bimodality_report(sub["phase_lag"].to_numpy())
        print(f"  {label} (n={r['n']}): dip={r['dip']:.4f}, p={r['dip_pvalue']:.4g}  "
              f"dBIC(k1-k2)={r['delta_bic_k1_minus_k2']:+.1f}, dBIC(k3-k2)={r['delta_bic_k3_minus_k2']:+.1f}")

    print("\n=== 5b. ICC(1): between- vs. within-participant variance of phase_lag ===")
    icc = icc1_one_way(df)
    print(f"ANOVA: F({icc['k_groups']-1},{icc['n_total']-icc['k_groups']}) = {icc['f_stat']:.3f}, "
          f"p = {icc['p_anova']:.3g}")
    print(f"ICC(1) = {icc['icc1']:.4f}  ({icc['icc1']*100:.1f}% of phase_lag variance is between-participant)")
    r_subj = subject_mean_bimodality(df)
    print(f"dip test on the {r_subj['n']} per-participant means: dip={r_subj['dip']:.4f}, "
          f"p={r_subj['dip_pvalue']:.4g} (low power at this n)")

    print("\n=== 6. coherence by phase_lag mode ===")
    print(mode_coherence_summary(df))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path,
                        help="Per-trial quality table from validation/aggregate_quality_metrics.py")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Save the H1, H2, ICC(1), and pooled bimodality results")
    args = parser.parse_args()

    df = add_derived_columns(pd.read_csv(args.input_csv))
    print(f"{len(df)} valid task trials loaded from {args.input_csv}\n")
    _print_report(df)

    if args.output_csv is not None:
        h1, h2, icc = ratio_test_h1(df), wrapping_test_h2(df), icc1_one_way(df)
        r_all = bimodality_report(df["phase_lag"].to_numpy())
        rows = [("h1", k, v) for k, v in h1.items()]
        rows += [("h2", k, v) for k, v in h2.items()]
        rows += [("icc", k, v) for k, v in icc.items()]
        rows += [("bimodality_pooled", k, v) for k, v in r_all.items()
                 if k not in ("component_means", "component_stds", "component_weights")]
        summary = pd.DataFrame(rows, columns=["section", "metric", "value"])
        summary.to_csv(args.output_csv, index=False)
        print(f"\nSummary table written to {args.output_csv}")


if __name__ == "__main__":
    main()
