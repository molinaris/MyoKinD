#!/usr/bin/env python3
"""Run HD sEMG and kinematic quality metrics across participants and flatten the
per-channel / per-finger results into a single per-trial quality table.

Runs every one of the 48 task trials (8 tasks x 2 frequencies x 3 repetitions)
for each requested participant through ``validation.hdsemg_quality`` and
``validation.kinematic_quality``, then collapses each trial down to one row
with columns: participant, task, task_freq, trial, snr, smr, spr, thdn, coh,
phase_lag. Computing means/standard deviations and plotting the distributions
is done by ``figures/plot_quality_summary.py``, which reads the CSV this
script writes.

A small subset (a handful of participants) is enough to sanity-check the
pipeline; running all 21 covers the full population, but can take a while
(48 trials x 21 participants).

``--output-csv-channels`` optionally saves the raw per-channel HD sEMG table
(128 channel rows per trial, before it is averaged down into the per-trial
quality table) - needed for per-channel analyses such as
``figures/plot_quality_summary.py``'s SNR-threshold channel rejection rate.

Usage:
    python aggregate_quality_metrics.py --root /path/to/Dataset --participants Sub001 Sub002 Sub003 --output-csv quality_summary.csv
    python aggregate_quality_metrics.py --root /path/to/Dataset --participants all --output-csv quality_summary.csv --output-csv-channels quality_summary_channels.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validation.hdsemg_quality import FREQUENCIES, REPETITIONS, TASKS  # noqa: E402
from validation.hdsemg_quality import process_trial as process_hdsemg_trial  # noqa: E402
from validation.kinematic_quality import process_trial as process_kinematic_trial  # noqa: E402

ALL_PARTICIPANTS = [f"Sub{n:03d}" for n in range(1, 22)]


# Minimum time between progress redraws (besides the once-per-participant checkpoint),
# so a fast loop doesn't spend all its time writing to the terminal.
PROGRESS_INTERVAL_S = 0.2

# Wide enough to fully overwrite any previous, longer progress line via "\r" alone.
_PROGRESS_LINE_WIDTH = 110


def _format_progress(done: int, total: int, start_time: float, label: str, bar_width: int = 30) -> str:
    fraction = done / total if total else 1.0
    filled = int(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.monotonic() - start_time
    eta = elapsed / done * (total - done) if done else 0.0
    line = (
        f"[{bar}] {done}/{total} ({fraction * 100:5.1f}%) "
        f"elapsed={elapsed:6.0f}s eta={eta:6.0f}s  {label}"
    )
    return line.ljust(_PROGRESS_LINE_WIDTH)


def _print_progress(done: int, total: int, start_time: float, label: str) -> None:
    """Redraw the progress bar in place on the current terminal line."""
    end = "\n" if done == total else ""
    print("\r" + _format_progress(done, total, start_time, label), end=end, flush=True)


def _print_interrupting(message: str) -> None:
    """Print a message that isn't part of the progress bar (e.g. a skipped trial),
    always starting on a fresh line so it never gets appended to a pending
    in-place progress line."""
    print("\n" + message)


def collect(root: Path, participants: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    hdsemg_frames = []
    kinematic_frames = []
    combos = [
        (participant, task, freq, rep)
        for participant in participants
        for task in TASKS
        for freq in FREQUENCIES
        for rep in REPETITIONS
    ]
    total = len(combos)
    start_time = time.monotonic()
    last_print_time = start_time
    previous_participant = None
    for i, (participant, task, freq, rep) in enumerate(combos, start=1):
        try:
            hdsemg_frames.append(process_hdsemg_trial(root, participant, task, freq, rep))
        except FileNotFoundError as exc:
            _print_interrupting(f"  skipping HD sEMG {participant} task={task} freq={freq} rep={rep}: {exc}")
        try:
            kinematic_frames.append(process_kinematic_trial(root, participant, task, freq, rep))
        except FileNotFoundError as exc:
            _print_interrupting(f"  skipping kinematics {participant} task={task} freq={freq} rep={rep}: {exc}")

        now = time.monotonic()
        is_new_participant = participant != previous_participant
        is_due = now - last_print_time >= PROGRESS_INTERVAL_S
        if is_new_participant or is_due or i == total:
            _print_progress(i, total, start_time, label=f"{participant} task={task} freq={freq} rep={rep}")
            last_print_time = now
        previous_participant = participant
    hdsemg = pd.concat(hdsemg_frames, ignore_index=True)
    kinematic = pd.concat(kinematic_frames, ignore_index=True)
    return hdsemg, kinematic


def build_quality_table(hdsemg: pd.DataFrame, kinematic: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-channel HD sEMG rows and per-finger kinematic rows into one
    row per trial: participant, task, task_freq, trial, snr, smr, spr, thdn,
    coh, phase_lag.

    snr/smr/spr are averaged across the 128 HD sEMG channels of the trial.
    thdn is averaged across the 5 fingers, pooling THDN across all digits and
    conditions rather than only the task's representative finger. coh/phase_lag
    are already a single value per trial (computed only for the representative
    finger), taken here as-is.
    """
    group_cols = ["participant", "task", "frequency", "repetition"]

    hdsemg_trial = (
        hdsemg.groupby(group_cols)[["snr_db", "smr_db", "spr_db"]]
        .mean()
        .reset_index()
        .rename(columns={"snr_db": "snr", "smr_db": "smr", "spr_db": "spr"})
    )
    thdn_trial = (
        kinematic.groupby(group_cols)["thdn_percent"]
        .mean()
        .reset_index()
        .rename(columns={"thdn_percent": "thdn"})
    )
    reference_trial = (
        kinematic[kinematic["is_reference_finger"]][group_cols + ["coherence", "phase_lag_rad"]]
        .rename(columns={"coherence": "coh", "phase_lag_rad": "phase_lag"})
    )

    table = hdsemg_trial.merge(thdn_trial, on=group_cols).merge(reference_trial, on=group_cols)
    table = table.rename(columns={"frequency": "task_freq", "repetition": "trial"})
    return table.sort_values(["participant", "task", "task_freq", "trial"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participants", nargs="+", required=True,
                        help="Participant IDs (e.g. Sub001 Sub002), or 'all' for all 21")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Save the per-trial quality table (participant, task, task_freq, "
                             "trial, snr, smr, spr, thdn, coh, phase_lag)")
    parser.add_argument("--output-csv-channels", type=Path, default=None,
                        help="Save the raw per-channel HD sEMG table (participant, task, frequency, "
                             "repetition, muscle, channel, snr_db, smr_db, spr_db), before it is "
                             "averaged across the 128 channels into the quality table above - needed "
                             "for per-channel analyses such as figures/plot_quality_summary.py's "
                             "SNR-threshold channel rejection rate")
    args = parser.parse_args()

    participants = ALL_PARTICIPANTS if [p.lower() for p in args.participants] == ["all"] else args.participants

    hdsemg, kinematic = collect(args.root, participants)
    quality_table = build_quality_table(hdsemg, kinematic)

    print(f"\nAggregated over {len(participants)} participant(s): "
          f"{len(hdsemg)} HD sEMG channel-trials, {len(kinematic)} kinematic finger-trials, "
          f"{len(quality_table)} trial(s) in the quality table")
    print(quality_table.to_string(index=False))

    if args.output_csv is not None:
        quality_table.to_csv(args.output_csv, index=False)
        print(f"\nQuality table written to {args.output_csv}")
        print("Run figures/plot_quality_summary.py on this file to summarize and plot "
              "the quality metric distributions.")

    if args.output_csv_channels is not None:
        hdsemg.to_csv(args.output_csv_channels, index=False)
        print(f"\nPer-channel HD sEMG table written to {args.output_csv_channels}")


if __name__ == "__main__":
    main()
