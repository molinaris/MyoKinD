"""Tests for validation/aggregate_rest_noise.py."""
from __future__ import annotations

import pandas as pd

from validation.aggregate_rest_noise import summarize


def test_summarize_runs_on_synthetic_table_and_flags_the_worst_case(capsys):
    df = pd.DataFrame({
        "participant": ["Sub001", "Sub001", "Sub002", "Sub002"],
        "repetition": [0, 0, 1, 1],
        "muscle": ["EDC", "FDS", "EDC", "FDS"],
        "channel": [0, 0, 1, 1],
        "rms_uv": [1.2, 0.4, 2.0, 1.5],
        "quant_penalty_db": [0.20, 0.45, 0.10, 0.15],
    })

    summarize(df, penalty_threshold_db=0.35)

    out = capsys.readouterr().out
    assert "4 channel-trials over 2 participant(s)" in out
    assert "channel-trials with penalty > 0.35 dB: 1/4" in out
    assert "worst-case channel-trial: Sub001 rep=0 FDS0" in out
