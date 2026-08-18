"""Tests for validation/aggregate_quality_metrics.py."""
from __future__ import annotations

import pandas as pd

from validation.aggregate_quality_metrics import build_quality_table

FINGERS = ("thumb", "index", "middle", "ring", "little")


def _hdsemg_rows(repetition: int, snr: float, smr: float, spr: float, n_channels: int = 4) -> list[dict]:
    return [
        {
            "participant": "Sub001", "task": "1", "frequency": "05", "repetition": repetition,
            "muscle": "EDC", "channel": channel,
            "snr_db": snr + channel, "smr_db": smr + channel, "spr_db": spr + channel,
        }
        for channel in range(n_channels)
    ]


def _kinematic_rows(repetition: int, thdn: float, coh: float, phase: float,
                     reference_finger: str = "index") -> list[dict]:
    return [
        {
            "participant": "Sub001", "task": "1", "frequency": "05", "repetition": repetition,
            "finger": finger, "is_reference_finger": finger == reference_finger,
            "thdn_percent": thdn,
            "coherence": coh if finger == reference_finger else float("nan"),
            "phase_lag_rad": phase if finger == reference_finger else float("nan"),
        }
        for finger in FINGERS
    ]


def test_build_quality_table_averages_channels_and_fingers_per_trial():
    hdsemg = pd.DataFrame(_hdsemg_rows(0, snr=10, smr=20, spr=21) + _hdsemg_rows(1, snr=5, smr=6, spr=7))
    kinematic = pd.DataFrame(
        _kinematic_rows(0, thdn=10, coh=0.9, phase=-0.5)
        + _kinematic_rows(1, thdn=20, coh=0.8, phase=-0.2)
    )

    table = build_quality_table(hdsemg, kinematic)

    assert list(table.columns) == [
        "participant", "task", "task_freq", "trial", "snr", "smr", "spr", "thdn", "coh", "phase_lag",
    ]
    assert len(table) == 2

    trial0 = table[table["trial"] == 0].iloc[0]
    # channels 0-3 with snr=10+channel -> mean of 10,11,12,13 = 11.5
    assert trial0["snr"] == 11.5
    assert trial0["smr"] == 21.5
    assert trial0["spr"] == 22.5
    assert trial0["thdn"] == 10  # same value on all 5 fingers -> mean unchanged
    assert trial0["coh"] == 0.9
    assert trial0["phase_lag"] == -0.5


def test_build_quality_table_keeps_only_reference_finger_coherence():
    hdsemg = pd.DataFrame(_hdsemg_rows(0, snr=10, smr=20, spr=21))
    kinematic = pd.DataFrame(_kinematic_rows(0, thdn=10, coh=0.95, phase=-1.0, reference_finger="middle"))

    table = build_quality_table(hdsemg, kinematic)

    assert len(table) == 1
    assert table.iloc[0]["coh"] == 0.95
    assert table.iloc[0]["phase_lag"] == -1.0
