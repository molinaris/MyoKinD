"""Tests for validation/inspect_dataset.py."""
from __future__ import annotations

from pathlib import Path

from validation import inspect_dataset as insp


def test_parse_filename_task_trial():
    meta = insp.parse_filename("Sub001_1_05_450_0.csv")
    assert meta == {"participant": "Sub001", "task": "1", "frequency": "05", "repetition": 0}


def test_parse_filename_rest():
    meta = insp.parse_filename("Sub019_rest_2.csv")
    assert meta == {"participant": "Sub019", "task": "rest", "frequency": None, "repetition": 2}


def test_parse_filename_start_position():
    meta = insp.parse_filename("Sub001_start_position_0.csv")
    assert meta == {"participant": "Sub001", "task": "start_position", "frequency": None, "repetition": 0}


def test_parse_filename_invalid():
    assert insp.parse_filename("not_a_dataset_file.csv") is None
    assert insp.parse_filename("Sub001_9_05_450_0.csv") is None  # task 9 does not exist


def test_classify_file():
    root = Path("/data/Dataset")
    assert insp.classify_file(root / "Sub001" / "HD_sEMG" / "x.csv", root) == "hdsemg"
    assert insp.classify_file(root / "Sub001" / "HandKinematics" / "Angles" / "x.csv", root) == "angles"
    assert insp.classify_file(root / "Sub001" / "HandKinematics" / "Trajectories" / "x.csv", root) == "trajectories"
    assert insp.classify_file(root / "Sub001" / "SomethingElse" / "x.csv", root) == "unknown"


def test_probe_hdsemg_empty(fixtures_dir):
    result = insp.probe_hdsemg(fixtures_dir / "empty_hdsemg.csv")
    assert result["status"] == "EMPTY"


def test_probe_hdsemg_wrong_columns(fixtures_dir):
    result = insp.probe_hdsemg(fixtures_dir / "wrong_columns_hdsemg.csv")
    assert result["status"] == "MISSING_COLUMNS"
    assert result["n_columns"] == 3


def test_probe_hdsemg_non_numeric(fixtures_dir):
    result = insp.probe_hdsemg(fixtures_dir / "non_numeric_hdsemg.csv")
    assert result["status"] == "NON_NUMERIC"


def test_probe_kinematics_missing_values_is_not_an_error(fixtures_dir):
    """A blank cell (marker occlusion) must be distinguished from real corruption."""
    result = insp.probe_kinematics(fixtures_dir / "missing_values_angles.csv", "angles")
    assert result["status"] == "MISSING_VALUES"
    assert "1 row(s)" in result["notes"]


def test_probe_kinematics_genuine_garbage(fixtures_dir):
    result = insp.probe_kinematics(fixtures_dir / "garbage_angles.csv", "angles")
    assert result["status"] == "NON_NUMERIC"


def test_probe_hdsemg_real_file(dataset_root):
    result = insp.probe_hdsemg(dataset_root / "Sub001" / "HD_sEMG" / "Sub001_1_05_450_0.csv")
    assert result["status"] == "OK"
    assert result["n_rows"] == insp.HDSEMG_N_ROWS
    assert result["n_columns"] == insp.HDSEMG_N_COLUMNS


def test_probe_kinematics_real_file(dataset_root):
    angles = insp.probe_kinematics(
        dataset_root / "Sub001" / "HandKinematics" / "Angles" / "Sub001_1_05_450_0.csv", "angles")
    assert angles["n_rows"] == insp.KIN_N_ROWS
    assert angles["n_columns"] == insp.ANGLES_N_COLUMNS
    assert angles["sampling_frequency_hz"] == 100.0

    trajectories = insp.probe_kinematics(
        dataset_root / "Sub001" / "HandKinematics" / "Trajectories" / "Sub001_1_05_450_0.csv", "trajectories")
    assert trajectories["n_columns"] == insp.TRAJECTORIES_N_COLUMNS


def test_sub017_documented_exceptions(dataset_root):
    """The dataset's two documented gaps (see dataset/README.md) must be exactly
    these files, and nothing else should be missing for this participant."""
    base = dataset_root / "Sub017"
    assert not (base / "HD_sEMG" / "Sub017_3_05_450_1.csv").exists()
    assert not (base / "HandKinematics" / "Angles" / "Sub017_start_position_0.csv").exists()
    assert not (base / "HandKinematics" / "Trajectories" / "Sub017_start_position_0.csv").exists()
    # the HD sEMG counterpart of start_position_0 is NOT excluded (per dataset/README.md)
    assert (base / "HD_sEMG" / "Sub017_start_position_0.csv").exists()

    result = insp.probe_hdsemg(base / "HD_sEMG" / "Sub017_1_05_450_0.csv")
    assert result["status"] == "OK"
