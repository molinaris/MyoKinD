"""Tests for biomechanical_model/estimate_kinematic_angles.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from biomechanical_model.estimate_kinematic_angles import (FINGERS, _project_perpendicular, _signed_angle,
                                                           calculate_mean_speed, calculate_rom, estimate_angles)
from examples.load_kinematics import load_angles_csv, load_trajectories_csv

FS = 100.0
F_TASK = 0.5
AMPLITUDES_DEG = np.array([10.0, 20.0, 30.0, 40.0, 50.0])


def _sine_cycle() -> np.ndarray:
    """One full sine cycle per finger (columns in FINGERS order), with amplitudes AMPLITUDES_DEG."""
    t = np.arange(int(FS / F_TASK)) / FS
    return np.sin(2 * np.pi * F_TASK * t)[:, None] * AMPLITUDES_DEG


def test_project_perpendicular_removes_the_component_along_c():
    c = np.array([[0.0, 0.0, 2.0]])
    v = np.array([[1.0, 2.0, 5.0]])
    np.testing.assert_allclose(_project_perpendicular(v, c), [[1.0, 2.0, 0.0]])


def test_signed_angle_recovers_a_known_angle_and_its_sign():
    c = np.array([[0.0, 0.0, 1.0]])
    a_perp = _project_perpendicular(np.array([[1.0, 0.0, 5.0]]), c)
    b_perp = _project_perpendicular(np.array([[np.cos(np.radians(30)), np.sin(np.radians(30)), -2.0]]), c)
    np.testing.assert_allclose(_signed_angle(a_perp, b_perp, c), [30.0])
    np.testing.assert_allclose(_signed_angle(a_perp, b_perp, -c), [-30.0])


@pytest.mark.parametrize("participant, trial", [("Sub001", "Sub001_1_05_450_0.csv"),
                                                ("Sub005", "Sub005_8_075_450_1.csv")])
def test_estimate_angles_matches_vicon_procalc(dataset_root, participant, trial):
    """Sub005 is left-handed (C vectors are flipped); both hands must match ProCalc within 1 mdeg."""
    base = dataset_root / participant / "HandKinematics"
    estimated, _ = estimate_angles(load_trajectories_csv(base / "Trajectories" / trial), participant)
    reference = load_angles_csv(base / "Angles" / trial)
    error = estimated[FINGERS].to_numpy() - reference[FINGERS].to_numpy()
    assert np.max(np.abs(error)) < 1e-3


def test_calculate_rom_of_a_sine_cycle_is_its_peak_to_peak_amplitude():
    rom = calculate_rom([_sine_cycle(), _sine_cycle()])
    assert list(rom["finger"]) == FINGERS
    np.testing.assert_allclose(rom["RoM"], 2 * AMPLITUDES_DEG)
    assert list(rom["n_cycles"]) == [2] * 5


def test_calculate_mean_speed_of_a_sine_cycle_matches_4_a_f():
    """Mean |d/dt A sin(2 pi f t)| over a full cycle is 4 A f (deg/s)."""
    speed = calculate_mean_speed([_sine_cycle(), _sine_cycle()], FS)
    np.testing.assert_allclose(speed["mean_speed"], 4 * AMPLITUDES_DEG * F_TASK, rtol=1e-3)


def test_calculate_mean_speed_uses_the_absolute_velocity():
    """The signed velocity averages to ~0 over a full cycle; the metric must not."""
    speed = calculate_mean_speed([_sine_cycle()], FS)
    assert (speed["mean_speed"] > 0.9 * 4 * AMPLITUDES_DEG * F_TASK).all()
