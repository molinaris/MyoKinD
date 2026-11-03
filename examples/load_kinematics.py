#!/usr/bin/env python3
"""Load a matching pair of Angles + Trajectories CSVs for one kinematic trial.

Both files share the same 5-line Vicon metadata header (name/group, sampling
frequency, marker/finger group names, column header, units) followed by 4500
data rows at 100 Hz, plus a trailing blank line that pandas' default
``skip_blank_lines=True`` already discards.

Angles: only the Z column of each finger group is non-zero (X/Y are always 0
because the model's rotation axis is Z). Columns are selected in increasing
index order - thumb(4), index(7), middle(10), ring(13), little(16) - which is
the mapping confirmed against the dataset's own header (ring and little must
not be swapped).

Trajectories: 12 markers x (X, Y, Z), ordered MW, LW, PT, DT, PI, DI, PM, DM,
PR, DR, PL, DL (wrist markers first, then thumb..little, proximal/distal).

Missing values: blank cells (brief marker occlusion) are filled on load by
linear interpolation, or held at the nearest valid value at the trial edges.

Usage:
    python load_kinematics.py --file /path/to/Sub001_1_05_450_0.csv --plot
    python load_kinematics.py --root /path/to/Dataset --participant Sub001 --task 1 --freq 05 --rep 0 --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.load_hdsemg import TASK_NAMES, trial_filename  # noqa: E402

KIN_FS = 100.0  # Hz; also cross-checked from the file's own header (2nd line)
KIN_HEADER_LINES = 5

ANGLE_COLUMNS = [0, 1, 4, 7, 10, 13, 16]
ANGLE_COLUMN_NAMES = ["frame", "sub_frame", "thumb", "index", "middle", "ring", "little"]

MARKER_ORDER = ["MW", "LW", "PT", "DT", "PI", "DI", "PM", "DM", "PR", "DR", "PL", "DL"]
TRAJECTORY_COLUMN_NAMES = ["frame", "sub_frame"] + [
    f"{marker}_{axis}" for marker in MARKER_ORDER for axis in ("x", "y", "z")
]


def _contiguous_ranges(positions: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted integer positions into (start, end) runs of consecutive values."""
    ranges = []
    start = prev = positions[0]
    for pos in positions[1:]:
        if pos == prev + 1:
            prev = pos
            continue
        ranges.append((start, prev))
        start = prev = pos
    ranges.append((start, prev))
    return ranges


def _interpolate_missing(df: pd.DataFrame, data_columns: list[str], path: Path) -> pd.DataFrame:
    """Fill blank cells (loaded as NaN) in the data columns.

    inspect_dataset.py's MISSING_VALUES status documents these gaps across the
    real dataset - brief Vicon marker occlusion, not a data error - so they are
    filled here rather than left as NaN, which would otherwise crash detrend/
    Welch/coherence calls downstream. Interior gaps are linearly interpolated;
    a gap that runs all the way to the first or last row (no bounding sample on
    one side, e.g. the trial ending mid-occlusion) has no data to interpolate
    against, so it is held flat at the nearest valid value instead (ffill/bfill).
    """
    gap_mask = df[data_columns].isna().any(axis=1).to_numpy()
    n_missing = int(df[data_columns].isna().to_numpy().sum())
    if n_missing:
        frames = df["frame"].to_numpy()
        spans = []
        for start, end in _contiguous_ranges(np.flatnonzero(gap_mask)):
            t_start = (frames[start] - 1) / KIN_FS
            t_end = (frames[end] - 1) / KIN_FS
            spans.append(f"t={t_start:.2f}s" if start == end else f"t={t_start:.2f}-{t_end:.2f}s")
        # Leading "\n": callers such as aggregate_quality_metrics.py redraw a progress bar
        # in place on the current line ("\r", no trailing newline) - this guarantees the
        # note starts on its own fresh line instead of getting appended to it.
        print(f"\n  note: filled {n_missing} missing value(s) (marker occlusion) "
              f"in {path.name} at {', '.join(spans)}")
        df[data_columns] = df[data_columns].interpolate(method="linear", limit_direction="both")
        df[data_columns] = df[data_columns].ffill().bfill()  # covers gaps with no bounding sample
    return df


def load_angles_csv(path: Path) -> pd.DataFrame:
    """Load joint angles (degrees) for the 5 fingers: thumb, index, middle, ring, little."""
    df = pd.read_csv(path, skiprows=KIN_HEADER_LINES, header=None, usecols=ANGLE_COLUMNS)
    df.columns = ANGLE_COLUMN_NAMES
    return _interpolate_missing(df, ["thumb", "index", "middle", "ring", "little"], path)


def load_trajectories_csv(path: Path) -> pd.DataFrame:
    """Load 3D marker trajectories (mm) for all 12 markers."""
    df = pd.read_csv(path, skiprows=KIN_HEADER_LINES, header=None)
    df.columns = TRAJECTORY_COLUMN_NAMES
    marker_columns = [c for c in TRAJECTORY_COLUMN_NAMES if c not in ("frame", "sub_frame")]
    return _interpolate_missing(df, marker_columns, path)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.file is not None:
        angles_path = args.file
        # infer the matching Trajectories path by swapping the folder name
        parts = list(angles_path.parts)
        parts[parts.index("Angles")] = "Trajectories"
        return angles_path, Path(*parts)
    if not (args.root and args.participant and args.task is not None and args.rep is not None):
        raise SystemExit("Provide --file (Angles CSV), or --root/--participant/--task/--rep "
                         "(and --freq for numbered tasks).")
    name = trial_filename(args.participant, args.task, args.freq, args.rep)
    base = args.root / args.participant / "HandKinematics"
    return base / "Angles" / name, base / "Trajectories" / name


def _plot(angles: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    t = (angles["frame"] - 1) / KIN_FS
    fig, ax = plt.subplots(figsize=(9, 3))
    finger_columns = ["thumb", "index", "middle", "ring", "little"]
    for finger in finger_columns:
        ax.plot(t, angles[finger], label=finger)
    y_max = angles[finger_columns].max().max()
    y_min = angles[finger_columns].min().min()
    margin = 0.15 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (deg)")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(frameon=False, ncols=5, fontsize=8)
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=None, help="Direct path to an Angles CSV file")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", default=None, help="e.g. Sub001")
    parser.add_argument("--task", default=None,
                        help="1-8 (see TASK_NAMES: 1=Index, 2=Middle, 3=Ring-Little, 4=Thumb, "
                             "5=I.Pinch, 6=M.Pinch, 7=Tripod, 8=5-Finger), or 'start_position'")
    parser.add_argument("--freq", default=None, help="'05' or '075' (ignored for start_position)")
    parser.add_argument("--rep", type=int, default=None, help="0, 1, or 2")
    parser.add_argument("--plot", action="store_true", help="Show the 5 finger angle traces")
    args = parser.parse_args()

    angles_path, trajectories_path = _resolve_paths(args)
    angles = load_angles_csv(angles_path)
    trajectories = load_trajectories_csv(trajectories_path)

    if args.task in TASK_NAMES:
        print(f"Task {args.task}: {TASK_NAMES[args.task]}")
    print(f"Loaded angles:       {angles_path} -> shape {angles.shape}")
    print(f"Loaded trajectories: {trajectories_path} -> shape {trajectories.shape}")
    print(f"  duration: {len(angles) / KIN_FS:.2f} s")

    if args.plot:
        _plot(angles)


if __name__ == "__main__":
    main()
