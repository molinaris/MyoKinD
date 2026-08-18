"""Tests for validation/rest_noise_rms.py."""
from __future__ import annotations

import numpy as np

from validation.rest_noise_rms import QUANTIZATION_NOISE_RMS_UV, compute_rms, quantization_penalty_db


def test_quantization_noise_rms_is_one_over_sqrt12():
    np.testing.assert_allclose(QUANTIZATION_NOISE_RMS_UV, 1.0 / np.sqrt(12))


def test_compute_rms_of_constant_signal():
    x = np.full((100, 3), 2.0)
    np.testing.assert_allclose(compute_rms(x), [2.0, 2.0, 2.0])


def test_compute_rms_of_sine_wave():
    t = np.linspace(0, 10, 100_000, endpoint=False)
    x = (5.0 * np.sin(2 * np.pi * 3 * t))[:, None]
    np.testing.assert_allclose(compute_rms(x), 5.0 / np.sqrt(2), rtol=1e-3)


def test_quantization_penalty_shrinks_as_measured_rms_grows():
    small_noise_penalty = quantization_penalty_db(np.array([0.5]))
    large_noise_penalty = quantization_penalty_db(np.array([10.0]))
    assert small_noise_penalty > large_noise_penalty


def test_quantization_penalty_below_0_35_db_when_measured_rms_at_least_1uv():
    # The claim under test: if the analog/physiological baseline noise is >= 1 uV
    # RMS, the dataset's 1 uV storage quantization adds less than 0.35 dB SNR penalty.
    penalty = quantization_penalty_db(np.array([1.0]))
    assert penalty[0] < 0.35
