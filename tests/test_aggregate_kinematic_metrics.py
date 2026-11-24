"""Tests for biomechanical_model/aggregate_kinematic_metrics.py."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from biomechanical_model import aggregate_kinematic_metrics
from biomechanical_model.aggregate_kinematic_metrics import (FREQUENCIES, TASKS, build_averaged_table,
                                                             build_descriptive_table,
                                                             plot_population_kinematic_metrics)

FINGERS = ("thumb", "index", "middle", "ring", "little")


def _trial_rows(repetition: int, rom: float, speed: float, thdn: float, participant: str = "Sub001",
                task: str = "1", frequency: str = "05") -> list[dict]:
    return [
        {
            "participant": participant, "task": task, "frequency": frequency, "repetition": repetition,
            "finger": finger,
            "rom_deg": rom, "mean_speed_deg_s": speed, "thdn_percent": thdn
        }
        for finger in FINGERS
    ]


def test_build_averaged_table_averages_the_three_repetitions_per_finger():
    metrics = pd.DataFrame(
        _trial_rows(0, rom=10, speed=20, thdn=3)
        + _trial_rows(1, rom=20, speed=40, thdn=6)
        + _trial_rows(2, rom=30, speed=60, thdn=9)
    )

    averaged = build_averaged_table(metrics)

    assert list(averaged.columns) == [
        "participant", "task", "frequency", "finger", "rom_deg", "mean_speed_deg_s", "thdn_percent"
    ]
    assert len(averaged) == 5

    thumb = averaged[averaged["finger"] == "thumb"].iloc[0]
    # repetitions 10, 20, 30 -> mean 20
    assert thumb["rom_deg"] == 20
    assert thumb["mean_speed_deg_s"] == 40
    assert thumb["thdn_percent"] == 6


def test_build_averaged_table_keeps_participants_separate():
    metrics = pd.DataFrame(
        _trial_rows(0, rom=10, speed=20, thdn=3, participant="Sub001")
        + _trial_rows(0, rom=50, speed=80, thdn=12, participant="Sub002")
    )

    averaged = build_averaged_table(metrics)

    assert len(averaged) == 10
    assert set(averaged[averaged["participant"] == "Sub001"]["rom_deg"]) == {10}
    assert set(averaged[averaged["participant"] == "Sub002"]["rom_deg"]) == {50}


def _averaged_table(n_participants: int = 2) -> pd.DataFrame:
    rows = []
    for participant in [f"Sub{n:03d}" for n in range(1, n_participants + 1)]:
        for task in TASKS:
            for frequency in FREQUENCIES:
                rows += _trial_rows(0, rom=62.5, speed=91.3, thdn=8.9, participant=participant,
                                    task=task, frequency=frequency)
    return build_averaged_table(pd.DataFrame(rows))


def test_build_descriptive_table_reports_the_mean_per_condition():
    descriptive = build_descriptive_table(_averaged_table(n_participants=2))

    assert len(descriptive) == len(TASKS) * len(FREQUENCIES) * len(FINGERS)
    assert descriptive["rom_deg_mean"].tolist() == pytest.approx([62.5] * len(descriptive))
    assert descriptive["mean_speed_deg_s_mean"].tolist() == pytest.approx([91.3] * len(descriptive))


def test_build_descriptive_table_ignores_missing_values():
    averaged = _averaged_table(n_participants=2)
    averaged.loc[0, "rom_deg"] = float("nan")

    descriptive = build_descriptive_table(averaged)
    assert not descriptive["rom_deg_mean"].isna().any()


def test_plot_population_kinematic_metrics_returns_a_5x3_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate_kinematic_metrics, "SIGNIFICANCE_CSV", tmp_path / "missing.csv")

    fig, _ = plot_population_kinematic_metrics(_averaged_table(), ["Sub001", "Sub002"])
    assert len(fig.get_axes()) == 15


def _significance_csv(path, participants: str) -> None:
    """Write a minimal R significance table for the given participant set ("Sub001|Sub002")."""
    pd.DataFrame({
        "Metric": ["rom_deg"], "Finger": ["index"], "Task_or_Factor": ["Task=1"],
        "A": ["05"], "B": ["075"], "p_bonf": [0.01], "Participants": [participants]
    }).to_csv(path, index=False)


def test_significance_annotations_are_used_when_participants_match(tmp_path, monkeypatch):
    csv = tmp_path / "kinematic_significant_interactions.csv"
    _significance_csv(csv, "Sub001|Sub002")
    monkeypatch.setattr(aggregate_kinematic_metrics, "SIGNIFICANCE_CSV", csv)

    significance, warning = aggregate_kinematic_metrics._load_significance_results(["Sub002", "Sub001"])

    assert warning is None
    assert len(significance) == 1


def test_significance_annotations_are_dropped_when_participants_differ(tmp_path, monkeypatch):
    csv = tmp_path / "kinematic_significant_interactions.csv"
    _significance_csv(csv, "Sub001|Sub002")
    monkeypatch.setattr(aggregate_kinematic_metrics, "SIGNIFICANCE_CSV", csv)

    significance, warning = aggregate_kinematic_metrics._load_significance_results(["Sub001"])

    assert significance.empty
    assert "do not match" in warning
