#!/usr/bin/env python3
"""Plot HD sEMG and kinematic quality metric distributions from the per-trial
quality table produced by validation/aggregate_quality_metrics.py.

Prints each metric's mean +/- std, and draws a violin+box layout: top row
SNR/SMR/SPR (panels A-C), bottom row coherence/phase lag/THDN (panels D-F).

If ``--input-channels-csv`` is also given (the ``--output-csv-channels`` table
from ``validation/aggregate_quality_metrics.py``), also reports the pooled
SNR-threshold channel rejection rate: every (channel, trial) HD sEMG row is
treated as its own instance, and the reported percentage/max-per-participant
are pooled over all of them - not the trial-averaged ``snr`` column of the
main quality table, which is what the panels above plot.

Usage:
    python plot_quality_summary.py --input-csv quality_summary.csv --output quality_metrics.png
    python plot_quality_summary.py --input-csv quality_summary.csv --input-channels-csv quality_summary_channels.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validation.kinematic_quality import COHERENCE_WINDOW_S, TRIM_END_S, TRIM_START_S  # noqa: E402

# Minimum acceptable SNR/SMR from the sEMG-quality literature (Sinderby et al.,
# 1995; Chang et al., 2020). No universally accepted threshold exists for SPR,
# so it has none below.
SNR_MIN_DB = 1.8
SMR_MIN_DB = 12.0


def coherence_significance(trial_duration_s: float, window_s: float, alpha: float = 0.05) -> float:
    """95%-CI coherence significance threshold: 1 - alpha**(1/(L-1)), where L is the
    number of independent Welch segments (L = floor(trial_duration_s / window_s))."""
    n_segments = int(trial_duration_s // window_s)
    if n_segments < 2:
        raise ValueError("Need at least 2 segments to compute a significance threshold.")
    return 1 - alpha ** (1 / (n_segments - 1))


# Derived from this repository's own coherence analysis parameters (trimmed trial
# duration and Welch window, both from validation/kinematic_quality.py), not from a
# fixed/hardcoded value - so it always matches whatever window size is actually used.
COHERENCE_SIGNIFICANCE_THRESHOLD = coherence_significance(
    trial_duration_s=TRIM_END_S - TRIM_START_S, window_s=COHERENCE_WINDOW_S)


METRIC_COLUMNS = ["snr", "smr", "spr", "thdn", "coh", "phase_lag"]


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Per-metric mean/std/n from the quality table."""
    rows = []
    for column in METRIC_COLUMNS:
        values = table[column].dropna()
        rows.append((column, values.mean(), values.std(), len(values)))
    return pd.DataFrame(rows, columns=["metric", "mean", "std", "n"])


def channel_snr_rejection_summary(channel_table: pd.DataFrame, threshold_db: float = SNR_MIN_DB) -> dict:
    """Pooled SNR-threshold rejection rate over every (channel, trial) HD sEMG row.

    Each of the 128 channels in each valid task trial is treated as its own
    instance (as opposed to ``summarize``'s ``snr`` column, which is already
    averaged across the 128 channels of each trial). Returns the percentage of
    all such instances below ``threshold_db``, and the largest count of
    below-threshold instances found for any single participant.
    """
    below = channel_table["snr_db"] < threshold_db
    max_below_per_participant = below.groupby(channel_table["participant"]).sum().max()
    return {
        "threshold_db": threshold_db,
        "n_channel_trials": len(channel_table),
        "n_below_threshold": int(below.sum()),
        "pct_below_threshold": 100 * below.mean(),
        "max_below_threshold_per_participant": int(max_below_per_participant),
    }


def _print_channel_snr_summary(summary: dict) -> None:
    print(f"\nChannel-wise SNR rejection ({summary['n_channel_trials']} channel-trials, "
          f"threshold={summary['threshold_db']} dB):")
    print(f"  {summary['pct_below_threshold']:.1f}% of channel-trials fell below the threshold "
          f"({summary['n_below_threshold']}/{summary['n_channel_trials']})")
    print(f"  no participant exceeded {summary['max_below_threshold_per_participant']} such channel-trials")


def _style_violin(vp, color: str) -> None:
    for body in vp["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.25)


def make_figure(table: pd.DataFrame):
    import matplotlib.pyplot as plt

    vp_props = dict(showmeans=False, showextrema=False, showmedians=False, points=100)
    bp_props = dict(
        widths=0.15, showmeans=True, showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="white", color="k", linewidth=1, alpha=0.5),
        flierprops=dict(marker=".", mec="none", mfc="k", markersize=3),
        medianprops=dict(color="k", linewidth=1),
        meanprops=dict(marker="o", mec="k", mfc="none", markersize=3),
    )

    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    # (axis, column, y-label, reference line value, reference line label)
    panels = [
        (ax[0, 0], "snr", "SNR (dB)", SNR_MIN_DB, "min."),
        (ax[0, 1], "smr", "SMR (dB)", SMR_MIN_DB, "min."),
        (ax[0, 2], "spr", "SPR (dB)", None, None),
        (ax[1, 0], "coh", r"$\gamma^2_{R,\theta}$", COHERENCE_SIGNIFICANCE_THRESHOLD, r"$C_\alpha$"),
        (ax[1, 1], "phase_lag", r"$\phi_{R,\theta}$ (rad)", None, None),
        (ax[1, 2], "thdn", r"$\mathrm{THD}_\mathrm{N}$ (%)", None, None),
    ]
    for axis, column, ylabel, ref_line, ref_label in panels:
        values = table[column].dropna().to_numpy()
        vp = axis.violinplot([values], **vp_props)
        _style_violin(vp, color)
        axis.boxplot([values], **bp_props)
        axis.set_ylabel(ylabel)
        axis.set_xticks([])
        if ref_line is not None:
            axis.axhline(ref_line, color="red", linestyle="--", linewidth=1)
            axis.text(1.4, ref_line, ref_label, color="red", fontsize=10, va="bottom", ha="right")

    ax[1, 1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax[1, 1].set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    for axis, label in zip(ax.ravel(), "ABCDEF"):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(0.0, 1.05, f"{label})", transform=axis.transAxes, va="top", ha="left", weight="bold")

    fig.suptitle("HD sEMG and kinematic quality metrics")
    fig.align_ylabels()
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path,
                         help="Per-trial quality table from validation/aggregate_quality_metrics.py")
    parser.add_argument("--input-channels-csv", type=Path, default=None,
                         help="Per-channel HD sEMG table from validation/aggregate_quality_metrics.py's "
                              "--output-csv-channels; if given, also reports the pooled SNR-threshold "
                              "channel rejection rate")
    parser.add_argument("--output", type=Path, default=None,
                         help="Save the figure (e.g. quality_metrics.png); shown interactively if omitted")
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    summary = summarize(table)
    print(f"{len(table)} trial(s) loaded from {args.input_csv}\n")
    print(summary.to_string(index=False))

    if args.input_channels_csv is not None:
        channel_table = pd.read_csv(args.input_channels_csv)
        _print_channel_snr_summary(channel_snr_rejection_summary(channel_table))

    fig = make_figure(table)
    if args.output is not None:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"\nFigure written to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
