"""Tests for examples/load_kinematics.py."""
from __future__ import annotations

from examples.load_kinematics import (
    ANGLE_COLUMN_NAMES, KIN_FS, TRAJECTORY_COLUMN_NAMES, load_angles_csv, load_trajectories_csv,
)


def test_angle_column_names_are_in_increasing_index_order():
    """Regression check: column order must be thumb, index, middle, ring,
    little - matching increasing column index order, not swapped."""
    assert ANGLE_COLUMN_NAMES == ["frame", "sub_frame", "thumb", "index", "middle", "ring", "little"]


def test_trajectory_column_names_marker_order():
    assert TRAJECTORY_COLUMN_NAMES[:5] == ["frame", "sub_frame", "MW_x", "MW_y", "MW_z"]
    assert TRAJECTORY_COLUMN_NAMES[-3:] == ["DL_x", "DL_y", "DL_z"]
    assert len(TRAJECTORY_COLUMN_NAMES) == 38


def test_load_angles_csv_real_file(dataset_root):
    angles = load_angles_csv(dataset_root / "Sub001" / "HandKinematics" / "Angles" / "Sub001_1_05_450_0.csv")
    assert angles.shape == (4500, 7)
    assert list(angles.columns) == ANGLE_COLUMN_NAMES
    duration_s = len(angles) / KIN_FS
    assert duration_s == 45.0


def test_load_trajectories_csv_real_file(dataset_root):
    trajectories = load_trajectories_csv(
        dataset_root / "Sub001" / "HandKinematics" / "Trajectories" / "Sub001_1_05_450_0.csv")
    assert trajectories.shape == (4500, 38)
    assert list(trajectories.columns) == TRAJECTORY_COLUMN_NAMES
