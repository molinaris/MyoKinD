"""Tests for validation/kinematic_quality.py."""
from __future__ import annotations

import numpy as np
import pytest

from validation.kinematic_quality import (REFERENCE_FINGER_BY_TASK, compute_coherence_phase, compute_thdn,
                                          select_reference_finger, task_reference, trim_signal)


def test_reference_finger_by_task():
    assert REFERENCE_FINGER_BY_TASK == {
        "1": "index", "2": "middle", "3": "ring", "4": "thumb",
        "5": "index", "6": "middle", "7": "index", "8": "index"
    }
    assert select_reference_finger("3") == "ring"  # Ring-Little task


def test_task_reference_starts_at_minus_one():
    # cos(2*pi*f*0 + pi) = cos(pi) = -1
    assert task_reference(np.array([0.0]), f_task=0.5)[0] == pytest.approx(-1.0)


def test_trim_signal_keeps_40_seconds_at_100hz():
    x = np.arange(4500)
    trimmed = trim_signal(x, fs=100.0)
    assert len(trimmed) == 4000
    assert trimmed[0] == 400


def test_compute_thdn_pure_sine_has_low_distortion_and_harmonic_signal_has_more():
    fs = 100.0
    f_task = 0.5
    t = np.arange(40 * fs) / fs

    pure = np.sin(2 * np.pi * f_task * t) * 10
    with_harmonics = np.sin(2 * np.pi * f_task * t) * 10 + np.sin(2 * np.pi * 2 * f_task * t) * 5
    small_amplitude_pure = np.sin(2 * np.pi * f_task * t) * 0.5

    signals = {
        "thumb": pure, "index": with_harmonics, "middle": small_amplitude_pure,
        "ring": small_amplitude_pure, "little": small_amplitude_pure
    }
    thdn = compute_thdn(signals, fs=fs, f_task=f_task)

    assert thdn["thumb"] < thdn["index"]  # added 2nd harmonic increases distortion
    # small-amplitude finger's THDN is scaled down relative to its own (unscaled) ratio,
    # since its fundamental is much smaller than the largest one (thumb) in the same trial
    assert thdn["middle"] < 100 * (0.5 / 10)  # sanity bound from the amplitude scaling factor


def test_compute_coherence_phase_perfect_match_gives_high_coherence():
    fs = 100.0
    f_task = 0.5
    t = np.arange(40 * fs) / fs
    reference = task_reference(t, f_task)
    coherence_value, _ = compute_coherence_phase(reference, reference, fs, f_task)
    assert coherence_value == pytest.approx(1.0, abs=1e-6)
