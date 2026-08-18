"""Tests for validation/hdsemg_quality.py."""
from __future__ import annotations

import numpy as np
from scipy.signal import freqz
from scipy.signal.windows import hann

from validation.hdsemg_quality import (
    HDSEMG_FS, bandpass_filter, compute_smr, compute_snr, compute_spr, periodic_hann, trim_signal,
)


def test_periodic_hann_matches_scipy_periodic_window():
    n = 4096
    np.testing.assert_array_equal(periodic_hann(n), hann(n, sym=False))


def test_periodic_hann_differs_from_symmetric_window():
    n = 16
    assert not np.allclose(periodic_hann(n), hann(n, sym=True))


def test_trim_signal_keeps_the_4_to_44_s_window():
    fs = 100.0
    x = np.arange(45 * 100)  # 1 sample per index, 45 s at 100 Hz
    trimmed = trim_signal(x, fs, start_s=4, end_s=44)
    assert len(trimmed) == 40 * 100
    assert trimmed[0] == 400  # first kept sample is at t=4s


def test_bandpass_filter_attenuates_outside_10_500_hz():
    fs = HDSEMG_FS
    sos = None
    from scipy.signal import butter
    sos = butter(4, [10, 500], btype="bandpass", fs=fs, output="sos")
    w, h = freqz(*_sos_to_ba(sos), fs=fs)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)

    def mag_at(f_hz):
        idx = np.argmin(np.abs(w - f_hz))
        return mag_db[idx]

    assert mag_at(200) > -3        # passband: little attenuation
    assert mag_at(1) < mag_at(200)  # well below the low cutoff: attenuated
    assert mag_at(900) < mag_at(200)  # well above the high cutoff: attenuated


def _sos_to_ba(sos):
    from scipy.signal import sos2tf
    return sos2tf(sos)


def test_bandpass_filter_shape_preserved():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5000, 4))
    y = bandpass_filter(x, HDSEMG_FS)
    assert y.shape == x.shape


def test_compute_snr_smr_spr_with_synthetic_psd():
    f = np.linspace(0, 1000, 501)
    psd_task = np.ones((501, 2))
    psd_rest = np.full((501, 2), 0.1)

    snr = compute_snr(psd_task, psd_rest)
    np.testing.assert_allclose(snr, 10 * np.log10(501 / (501 * 0.1)))

    smr = compute_smr(f, psd_task, cutoff_hz=10.0)
    assert np.all(smr > 0)  # low-frequency band is a small fraction of total power here

    spr = compute_spr(f, psd_task)
    assert np.all(np.isfinite(spr))
