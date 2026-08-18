"""Tests for validation/aggregate_wrist_precision.py."""
from __future__ import annotations

import pandas as pd

from validation.aggregate_wrist_precision import summarize


def test_summarize_runs_on_synthetic_table_and_flags_the_worst_case(capsys):
    df = pd.DataFrame({
        "participant": ["Sub001", "Sub001", "Sub002", "Sub002"],
        "task": ["1", "2", "1", "start_position"],
        "frequency": ["05", "05", "05", None],
        "repetition": [0, 0, 1, 1],
        "mean_distance_mm": [55.0, 55.2, 54.8, 55.1],
        "std_distance_mm": [0.20, 0.45, 0.10, 0.15],
        "n_samples": [4000, 4000, 4000, 4000],
    })

    summarize(df)

    out = capsys.readouterr().out
    assert "4 valid trial(s) over 2 participant(s)" in out
    assert "worst-case trial: Sub001 task=2 freq=05 rep=0, std=0.450 mm" in out
