"""Tests for examples/load_hdsemg.py."""
from __future__ import annotations

import numpy as np

from examples.load_hdsemg import (
    HDSEMG_FS, TASK_NAMES, load_hdsemg_csv, moving_rms, split_edc_fds, trial_filename,
)


def test_trial_filename_task():
    assert trial_filename("Sub001", "1", "05", 0) == "Sub001_1_05_450_0.csv"


def test_trial_filename_rest_and_start_position():
    assert trial_filename("Sub019", "rest", None, 2) == "Sub019_rest_2.csv"
    assert trial_filename("Sub001", "start_position", None, 0) == "Sub001_start_position_0.csv"


def test_task_names():
    """Regression check: task codes 1-8 (the T field of SubXXX_T_F_450_R.csv)."""
    assert TASK_NAMES == {
        "1": "Index", "2": "Middle", "3": "Ring-Little", "4": "Thumb",
        "5": "I. Pinch", "6": "M. Pinch", "7": "Tripod", "8": "5-Finger",
    }


def test_split_edc_fds_shapes():
    x = np.zeros((100, 128))
    edc, fds = split_edc_fds(x)
    assert edc.shape == (100, 64)
    assert fds.shape == (100, 64)


def test_moving_rms_preserves_shape_and_is_non_negative():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 4))
    rms = moving_rms(x, window_samples=100)
    assert rms.shape == x.shape
    assert np.all(rms >= 0)


def test_moving_rms_of_constant_signal_equals_its_amplitude():
    x = np.full((500, 1), 3.0)
    rms = moving_rms(x, window_samples=50)
    np.testing.assert_allclose(rms[100:-100], 3.0, atol=1e-8)


def test_load_hdsemg_csv_real_file(dataset_root):
    data = load_hdsemg_csv(dataset_root / "Sub001" / "HD_sEMG" / "Sub001_1_05_450_0.csv")
    assert data.shape == (92160, 128)
    duration_s = data.shape[0] / HDSEMG_FS
    assert 44.8 < duration_s < 45.0

    edc, fds = split_edc_fds(data)
    assert edc.shape == (92160, 64)
    assert fds.shape == (92160, 64)
