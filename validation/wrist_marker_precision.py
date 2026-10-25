#!/usr/bin/env python3
"""OMC reconstruction precision, estimated post hoc from the MW-LW wrist
marker distance.

Because the wrist remained stationary throughout the protocol, the distance
between the markers placed on the radial and ulnar styloid processes (MW and
LW) is nominally constant within a trial. Its within-trial standard deviation
therefore provides an empirical estimate of the combined effect of marker
reconstruction error and soft-tissue motion at the wrist.

Computed on every kinematic trial (the 48 numbered task trials plus the 3
`start_position` recordings - kinematics has no `rest` condition, see
README.md), trimmed to the same 4-44 s window used throughout this
repository (validation/kinematic_quality.py) to avoid transients at trial
onset/offset.

Usage:
    python wrist_marker_precision.py --root /path/to/Dataset --participant Sub001 --task 1 --freq 05 --rep 0
    python wrist_marker_precision.py --root /path/to/Dataset --participant Sub001 --all --output-csv sub001_wrist_precision.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.load_hdsemg import trial_filename  # noqa: E402
from examples.load_kinematics import KIN_FS, load_trajectories_csv  # noqa: E402
from validation.kinematic_quality import trim_signal  # noqa: E402

TASKS = [str(t) for t in range(1, 9)]
FREQUENCIES = ["05", "075"]
REPETITIONS = [0, 1, 2]

WRIST_MARKERS = ("MW", "LW")


def wrist_distance_mm(trajectories: pd.DataFrame) -> np.ndarray:
    """Frame-by-frame Euclidean distance (mm) between the MW and LW markers."""
    m0, m1 = WRIST_MARKERS
    p0 = trajectories[[f"{m0}_x", f"{m0}_y", f"{m0}_z"]].to_numpy()
    p1 = trajectories[[f"{m1}_x", f"{m1}_y", f"{m1}_z"]].to_numpy()
    return np.linalg.norm(p0 - p1, axis=1)


def process_trial(root: Path, participant: str, task: str, freq: str | None, rep: int) -> dict:
    """Within-trial mean/std of the MW-LW wrist distance for one kinematic trial."""
    path = root / participant / "HandKinematics" / "Trajectories" / trial_filename(participant, task, freq, rep)
    trajectories = load_trajectories_csv(path)
    distance = trim_signal(wrist_distance_mm(trajectories), KIN_FS)

    return {
        "participant": participant, "task": task, "frequency": freq, "repetition": rep,
        "mean_distance_mm": float(distance.mean()), "std_distance_mm": float(distance.std()),
        "n_samples": len(distance)
    }


def _print_summary(df: pd.DataFrame) -> None:
    std = df["std_distance_mm"]
    print(f"  std_distance_mm: mean={std.mean():.3f}, median={std.median():.3f}, "
          f"p95={std.quantile(0.95):.3f}, max={std.max():.3f} (n={len(df)} trials)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--task", default=None, help="1-8 or 'start_position' (ignored if --all)")
    parser.add_argument("--freq", default=None, help="'05' or '075' (ignored for start_position/--all)")
    parser.add_argument("--rep", type=int, default=None, help="0, 1, or 2 (ignored if --all)")
    parser.add_argument("--all", action="store_true",
                        help="process all 51 kinematic trials for the participant "
                             "(48 task trials + 3 start_position)")
    parser.add_argument("--output-csv", type=Path, default=None, help="Save the per-trial results table")
    args = parser.parse_args()

    if args.all:
        combos = [(t, f, r) for t in TASKS for f in FREQUENCIES for r in REPETITIONS]
        combos += [("start_position", None, r) for r in REPETITIONS]
    else:
        if args.task is None or args.rep is None:
            raise SystemExit("Provide --task/--rep (and --freq for numbered tasks), or use --all.")
        combos = [(args.task, args.freq, args.rep)]

    rows = []
    for task, freq, rep in combos:
        try:
            rows.append(process_trial(args.root, args.participant, task, freq, rep))
        except FileNotFoundError as exc:
            print(f"  skipping {args.participant} task={task} freq={freq} rep={rep}: {exc}")
    result = pd.DataFrame(rows)

    print(f"{args.participant}: {len(rows)}/{len(combos)} trial(s) processed")
    _print_summary(result)

    if args.output_csv is not None:
        result.to_csv(args.output_csv, index=False)
        print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
