"""Tests for validation/wrist_marker_precision.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.wrist_marker_precision import wrist_distance_mm


def _trajectories_row(mw, lw):
    row = {f"{marker}_{axis}": 0.0 for marker in ("MW", "LW", "PT") for axis in ("x", "y", "z")}
    row["MW_x"], row["MW_y"], row["MW_z"] = mw
    row["LW_x"], row["LW_y"], row["LW_z"] = lw
    return row


def test_wrist_distance_mm_is_euclidean_distance_between_mw_and_lw():
    df = pd.DataFrame([
        _trajectories_row(mw=(0.0, 0.0, 0.0), lw=(3.0, 4.0, 0.0)),   # 3-4-5 triangle -> 5 mm
        _trajectories_row(mw=(1.0, 1.0, 1.0), lw=(1.0, 1.0, 1.0))   # coincident -> 0 mm
    ])
    np.testing.assert_allclose(wrist_distance_mm(df), [5.0, 0.0])


def test_wrist_distance_mm_constant_for_stationary_markers():
    df = pd.DataFrame([_trajectories_row(mw=(0.0, 0.0, 0.0), lw=(50.0, 0.0, 0.0))] * 10)
    distance = wrist_distance_mm(df)
    assert distance.std() == 0.0
    np.testing.assert_allclose(distance, 50.0)
