"""Tests for biomechanical_model/plot_interactive_kinematic.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from biomechanical_model.plot_interactive_kinematic import plot_interactive_kinematic
from examples.load_kinematics import KIN_FS


def test_plot_interactive_kinematic_real_file(dataset_root):
    fig = plot_interactive_kinematic(dataset_root, "Sub001", "7", "05", 0, fs=KIN_FS)
    axes = fig.get_axes()
    assert sum(type(ax).__name__ == "Axes3D" for ax in axes) == 6
    assert len(axes) == 14
