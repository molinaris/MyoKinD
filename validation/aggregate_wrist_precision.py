#!/usr/bin/env python3
"""Aggregate validation/wrist_marker_precision.py across participants into a
population-level estimate of OMC reconstruction precision at the wrist
(within-trial standard deviation of the MW-LW marker distance).

Usage:
    python aggregate_wrist_precision.py --root /path/to/Dataset --participants Sub001 Sub002 Sub003 --output-csv wrist_precision.csv
    python aggregate_wrist_precision.py --root /path/to/Dataset --participants all --output-csv wrist_precision.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validation.wrist_marker_precision import FREQUENCIES, REPETITIONS, TASKS, process_trial  # noqa: E402

ALL_PARTICIPANTS = [f"Sub{n:03d}" for n in range(1, 22)]


def collect(root: Path, participants: list[str]) -> pd.DataFrame:
    combos = [(t, f, r) for t in TASKS for f in FREQUENCIES for r in REPETITIONS]
    combos += [("start_position", None, r) for r in REPETITIONS]

    rows = []
    for participant in participants:
        for task, freq, rep in combos:
            try:
                rows.append(process_trial(root, participant, task, freq, rep))
            except FileNotFoundError as exc:
                print(f"  skipping {participant} task={task} freq={freq} rep={rep}: {exc}")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> None:
    std = df["std_distance_mm"]
    n = len(df)

    print(f"\n{n} valid trial(s) over {df['participant'].nunique()} participant(s):")
    print(f"  within-trial std of MW-LW distance (mm): mean={std.mean():.3f}, median={std.median():.3f}, "
          f"std={std.std():.3f}, min={std.min():.3f}, p95={std.quantile(0.95):.3f}, max={std.max():.3f}")

    worst = df.loc[std.idxmax()]
    print(f"  worst-case trial: {worst['participant']} task={worst['task']} freq={worst['frequency']} "
          f"rep={worst['repetition']}, std={worst['std_distance_mm']:.3f} mm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participants", nargs="+", required=True,
                         help="Participant IDs (e.g. Sub001 Sub002), or 'all' for all 21")
    parser.add_argument("--output-csv", type=Path, default=None,
                         help="Save the per-trial table (participant, task, frequency, repetition, "
                              "mean_distance_mm, std_distance_mm, n_samples)")
    args = parser.parse_args()

    participants = ALL_PARTICIPANTS if args.participants == ["all"] else args.participants

    df = collect(args.root, participants)
    summarize(df)

    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    main()
