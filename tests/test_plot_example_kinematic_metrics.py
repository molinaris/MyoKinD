"""Tests for biomechanical_model/plot_example_kinematic_metrics.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from biomechanical_model.plot_example_kinematic_metrics import plot_example_kinematic_metrics
from examples.load_kinematics import KIN_FS


def test_plot_example_kinematic_metrics_real_file(dataset_root):
    fig = plot_example_kinematic_metrics(dataset_root, "Sub017", "6", "05", 0, fs=KIN_FS)
    assert len(fig.get_axes()) == 16
