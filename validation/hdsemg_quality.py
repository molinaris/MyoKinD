#!/usr/bin/env python3
"""HD sEMG quality metrics: SNR, SMR and SPR.

For each task trial, the matching ``rest`` recording (same participant and
repetition index) is used as the noise-floor reference for SNR. Metrics are
computed per channel on the raw (unfiltered) signal, only trimmed to its
useful 4-44 s window - a quality metric is meant to characterize the raw
acquisition itself, so it is deliberately computed *before* any band-pass
filtering (``bandpass_filter`` below is provided for other scripts, e.g.
``figures/plot_example_trial.py``, that do work on a filtered signal):

- Temporal trim: discard the first 4 s and the last 1 s of the 45 s trial, to
  avoid the transients of task onset/offset (same convention used for the
  kinematic recordings, for temporal correspondence between both modalities).
- PSD: Welch, 4 s window, no overlap, explicit periodic Hann window.
- SNR  = 10*log10(sum(psd_task) / sum(psd_rest))
- SMR  = 10*log10(sum(psd_task) / sum(psd_task[f <= 10 Hz]))
- SPR  = 10*log10(sum(psd_task) / sum(psd_task[60, 120, ..., 960 Hz bins]))

Usage:
    python hdsemg_quality.py --root /path/to/Dataset --participant Sub001 --task 1 --freq 05 --rep 0
    python hdsemg_quality.py --root /path/to/Dataset --participant Sub001 --all --output-csv sub001_emg_quality.csv
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal.windows import hann

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.load_hdsemg import HDSEMG_FS, load_hdsemg_csv, trial_filename  # noqa: E402

TASKS = [str(t) for t in range(1, 9)]
FREQUENCIES = ["05", "075"]
REPETITIONS = [0, 1, 2]

TRIM_START_S = 4.0
TRIM_END_S = 44.0
FILTER_LOW_HZ = 10.0
FILTER_HIGH_HZ = 500.0
FILTER_ORDER = 4
PSD_WINDOW_S = 4.0
SMR_CUTOFF_HZ = 10.0
POWERLINE_HARMONICS_HZ = np.arange(60, 1000, 60)


def periodic_hann(n: int) -> np.ndarray:
    """Explicit periodic ('hann_periodic') window, as opposed to the symmetric
    default some tools use. This is what scipy.signal.welch would already use
    internally when given the string 'hann' (fftbins=True), made explicit here
    for readability."""
    return hann(n, sym=False)


def bandpass_filter(x: np.ndarray, fs: float, low: float = FILTER_LOW_HZ,
                    high: float = FILTER_HIGH_HZ, order: int = FILTER_ORDER) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def trim_signal(x: np.ndarray, fs: float, start_s: float = TRIM_START_S,
                end_s: float = TRIM_END_S) -> np.ndarray:
    """Discard the first `start_s` seconds and everything after `end_s` seconds."""
    i0 = int(round(start_s * fs))
    i1 = int(round(end_s * fs))
    return x[i0:i1]


def compute_psd(x: np.ndarray, fs: float, window_s: float = PSD_WINDOW_S) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(round(window_s * fs))
    window = periodic_hann(nperseg)
    return welch(x, fs=fs, nperseg=nperseg, noverlap=0, window=window, axis=0)


def compute_snr(psd_task: np.ndarray, psd_rest: np.ndarray) -> np.ndarray:
    return 10 * np.log10(psd_task.sum(axis=0) / psd_rest.sum(axis=0))


def compute_smr(f: np.ndarray, psd: np.ndarray, cutoff_hz: float = SMR_CUTOFF_HZ) -> np.ndarray:
    mask = f <= cutoff_hz
    return 10 * np.log10(psd.sum(axis=0) / psd[mask].sum(axis=0))


def compute_spr(f: np.ndarray, psd: np.ndarray) -> np.ndarray:
    idx = [int(np.argmin(np.abs(f - h))) for h in POWERLINE_HARMONICS_HZ]
    return 10 * np.log10(psd.sum(axis=0) / psd[idx].sum(axis=0))


@lru_cache(maxsize=None)
def _rest_psd(root_str: str, participant: str, rep: int) -> np.ndarray:
    """PSD of the matching rest recording, cached per (participant, rep) since
    the same rest trial is reused as the SNR reference for 16 task trials."""
    rest_path = Path(root_str) / participant / "HD_sEMG" / trial_filename(participant, "rest", None, rep)
    rest_raw = trim_signal(load_hdsemg_csv(rest_path), HDSEMG_FS)
    _, psd_rest = compute_psd(rest_raw, HDSEMG_FS)
    return psd_rest


def process_trial(root: Path, participant: str, task: str, freq: str, rep: int) -> pd.DataFrame:
    """Compute SNR/SMR/SPR for every one of the 128 channels of one trial."""
    task_path = root / participant / "HD_sEMG" / trial_filename(participant, task, freq, rep)

    task_raw = trim_signal(load_hdsemg_csv(task_path), HDSEMG_FS)
    f, psd_task = compute_psd(task_raw, HDSEMG_FS)
    psd_rest = _rest_psd(str(root), participant, rep)

    snr = compute_snr(psd_task, psd_rest)
    smr = compute_smr(f, psd_task)
    spr = compute_spr(f, psd_task)

    n_channels = task_raw.shape[1]
    muscle = np.where(np.arange(n_channels) < 64, "EDC", "FDS")
    channel_in_muscle = np.where(np.arange(n_channels) < 64, np.arange(n_channels), np.arange(n_channels) - 64)

    return pd.DataFrame({
        "participant": participant, "task": task, "frequency": freq, "repetition": rep,
        "muscle": muscle, "channel": channel_in_muscle,
        "snr_db": snr, "smr_db": smr, "spr_db": spr
    })


def _print_summary(df: pd.DataFrame) -> None:
    for metric in ("snr_db", "smr_db", "spr_db"):
        print(f"  {metric}: mean={df[metric].mean():.2f}, std={df[metric].std():.2f} "
              f"(n={len(df)} channel-trials)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--task", default=None, help="1-8 (ignored if --all)")
    parser.add_argument("--freq", default=None, help="'05' or '075' (ignored if --all)")
    parser.add_argument("--rep", type=int, default=None, help="0, 1, or 2 (ignored if --all)")
    parser.add_argument("--all", action="store_true", help="process all 48 task trials for the participant")
    parser.add_argument("--output-csv", type=Path, default=None, help="Save the per-channel results table")
    args = parser.parse_args()

    if args.all:
        combos = [(t, f, r) for t in TASKS for f in FREQUENCIES for r in REPETITIONS]
    else:
        if args.task is None or args.freq is None or args.rep is None:
            raise SystemExit("Provide --task/--freq/--rep, or use --all.")
        combos = [(args.task, args.freq, args.rep)]

    frames = []
    for task, freq, rep in combos:
        try:
            frames.append(process_trial(args.root, args.participant, task, freq, rep))
        except FileNotFoundError as exc:
            print(f"  skipping {args.participant} task={task} freq={freq} rep={rep}: {exc}")
    result = pd.concat(frames, ignore_index=True)

    print(f"{args.participant}: {len(frames)}/{len(combos)} trial(s) processed, {len(result)} channel-trial rows")
    _print_summary(result)

    if args.output_csv is not None:
        result.to_csv(args.output_csv, index=False)
        print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
