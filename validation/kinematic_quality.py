#!/usr/bin/env python3
"""Kinematic quality metrics: spectral coherence, phase lag and THDN.

For each task trial, the joint-angle trajectory of the task's representative
finger is compared against a sinusoidal reference signal at the task
frequency. THDN is computed for all 5 fingers, while coherence/phase lag are
only meaningful for the representative finger.

Parameters:

- Temporal trim: discard the first 4 s and the last 1 s of the 45 s trial, to
  avoid the transients of task onset/offset (same convention used for the
  HD sEMG recordings, for temporal correspondence between both modalities).
- Reference signal: cos(2*pi*f_task*t + pi) - the +pi phase offset makes the
  reference start at its trough, matching the open-hand posture every task
  starts from.
- Coherence / phase lag: Welch/CSD-based, 4 s window, no overlap, explicit
  periodic Hann window; both evaluated at the task frequency bin.
- THDN: Welch PSD (4 s window, periodic Hann window), harmonic distortion
  ratio (fundamental + 4 harmonics, with the amplitude of each component
  estimated from the target frequency bin ± 1 bin) scaled by each
  finger's fundamental amplitude relative to the largest one across the 5
  fingers of the same trial - this scaled ratio (not the plain, unscaled THD)
  is the THDN metric used throughout this repository.

Usage:
    python kinematic_quality.py --root /path/to/Dataset --participant Sub001 --task 1 --freq 05 --rep 0
    python kinematic_quality.py --root /path/to/Dataset --participant Sub001 --all --output-csv sub001_kin_quality.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import coherence, csd, welch
from scipy.signal.windows import hann

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.load_hdsemg import trial_filename  # noqa: E402
from examples.load_kinematics import KIN_FS, load_angles_csv  # noqa: E402

TASKS = [str(t) for t in range(1, 9)]
FREQUENCIES = ["05", "075"]
REPETITIONS = [0, 1, 2]
FINGERS = ["thumb", "index", "middle", "ring", "little"]

TRIM_START_S = 4.0
TRIM_END_S = 44.0
COHERENCE_WINDOW_S = 4.0
THDN_WINDOW = "hann"
THDN_WINDOW_S = 4.0
THDN_N_HARMONICS = 5
THDN_N_BINS = 1

FREQ_CODE_TO_HZ = {"05": 0.5, "075": 0.75}

# Representative finger per task (Index/I.Pinch/Tripod/5-Finger -> index;
# Middle/M.Pinch -> middle; Ring-Little -> ring; Thumb -> thumb). See
# examples.load_hdsemg.TASK_NAMES for what each task number actually is
# (e.g. task "3" = Ring-Little).
REFERENCE_FINGER_BY_TASK = {
    "1": "index", "2": "middle", "3": "ring", "4": "thumb",
    "5": "index", "6": "middle", "7": "index", "8": "index"}


def periodic_hann(n: int) -> np.ndarray:
    """Explicit periodic Hann window, used in every Welch/CSD/coherence call."""
    return hann(n, sym=False)


def trim_signal(x: np.ndarray, fs: float, start_s: float = TRIM_START_S,
                end_s: float = TRIM_END_S) -> np.ndarray:
    i0 = int(round(start_s * fs))
    i1 = int(round(end_s * fs))
    return x[i0:i1]


def task_reference(t: np.ndarray, f_task: float) -> np.ndarray:
    """Sinusoidal reference signal presented by the visual interface."""
    return np.cos(2 * np.pi * f_task * t + np.pi)


def select_reference_finger(task: str) -> str:
    return REFERENCE_FINGER_BY_TASK[task]


def compute_coherence_phase(reference: np.ndarray, signal: np.ndarray, fs: float, f_task: float,
                            window_s: float = COHERENCE_WINDOW_S) -> tuple[float, float]:
    """Spectral coherence and phase lag (CSD angle) at the task frequency."""
    nperseg = int(round(window_s * fs))
    window = periodic_hann(nperseg)
    f, coh = coherence(reference, signal, fs=fs, nperseg=nperseg, noverlap=0, window=window)
    _, pxy = csd(reference, signal, fs=fs, nperseg=nperseg, noverlap=0, window=window)
    idx = int(np.argmin(np.abs(f - f_task)))
    return float(coh[idx]), float(np.angle(pxy[idx]))


def _band_amplitude(pxx: np.ndarray, idx_center: int, n_bins: int, delta_f: float) -> float:
    idx_min = max(0, idx_center - n_bins)
    idx_max = min(len(pxx) - 1, idx_center + n_bins)
    power = np.sum(pxx[idx_min:idx_max + 1]) * delta_f
    return float(np.sqrt(power))


def compute_thdn(finger_signals: dict[str, np.ndarray], fs: float, f_task: float,
                 window_s: float = THDN_WINDOW_S, n_harmonics: int = THDN_N_HARMONICS,
                 n_bins: int = THDN_N_BINS, window: str = THDN_WINDOW,
                 return_details: bool = False) -> dict[str, float] | tuple[dict[str, float], dict[str, dict]]:
    """Normalized Total Harmonic Distortion (THDN) for each finger.

    THDN = THD_ratio * (A_fundamental / A_fundamental_max_across_fingers) * 100,
    where THD_ratio = sqrt(sum(A_harmonic^2 for harmonics 2..n)) / A_fundamental.

    The amplitude of the fundamental and each harmonic is estimated by
    integrating the PSD over the target frequency bin ± `n_bins` bins.

    The THDN calculation uses the expected task frequency (`f_task`) as the
    fundamental frequency. When `return_details=True`, the dominant detected
    frequency of each finger is also returned from the PSD for quality-control
    and frequency-compliance analyses.

    The scaling by relative fundamental amplitude across the 5 fingers of the
    same trial is part of this metric's definition (as opposed to the plain,
    unscaled THD).
    """
    nperseg = int(round(window_s * fs))
    delta_f = fs / nperseg

    fundamental_amplitude = {}
    psd_by_finger = {}
    freq_by_finger = {}
    detected_frequency = {}
    for finger, signal in finger_signals.items():
        f, pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=0, detrend="linear", window=window)
        idx_fund = int(np.argmin(np.abs(f - f_task)))
        fundamental_amplitude[finger] = _band_amplitude(pxx, idx_fund, n_bins, delta_f)
        valid_idx = np.where(f > 0)[0]
        idx_detected = valid_idx[np.argmax(pxx[valid_idx])]
        detected_frequency[finger] = float(f[idx_detected])
        psd_by_finger[finger] = pxx
        freq_by_finger[finger] = f
    max_fundamental = max(fundamental_amplitude.values())

    thdn = {}
    for finger in finger_signals:
        f = freq_by_finger[finger]
        pxx = psd_by_finger[finger]
        amplitudes = [fundamental_amplitude[finger]]

        for harmonic in range(2, n_harmonics + 1):
            idx_center = int(np.argmin(np.abs(f - harmonic * f_task)))
            amplitudes.append(_band_amplitude(pxx, idx_center, n_bins, delta_f))

        amplitudes = np.asarray(amplitudes, dtype=float)
        thd_ratio = np.sqrt(np.sum(amplitudes[1:] ** 2)) / amplitudes[0]
        scale = fundamental_amplitude[finger] / max_fundamental
        thdn[finger] = float(thd_ratio * scale * 100.0)

    if not return_details:
        return thdn
    details = {finger: {"f": freq_by_finger[finger], "pxx": psd_by_finger[finger],
                        "fundamental_amplitude": fundamental_amplitude[finger],
                        "detected_frequency": detected_frequency[finger]} for finger in finger_signals}
    return thdn, details


def process_trial(root: Path, participant: str, task: str, freq: str, rep: int) -> pd.DataFrame:
    angles_path = root / participant / "HandKinematics" / "Angles" / trial_filename(participant, task, freq, rep)
    angles = load_angles_csv(angles_path)
    f_task = FREQ_CODE_TO_HZ[freq]

    signals = {finger: trim_signal(angles[finger].to_numpy(), KIN_FS) for finger in FINGERS}
    thdn = compute_thdn(signals, KIN_FS, f_task)

    reference_finger = select_reference_finger(task)
    n_samples = len(next(iter(signals.values())))
    t = np.arange(n_samples) / KIN_FS
    reference = task_reference(t, f_task)
    coherence_value, phase_lag = compute_coherence_phase(reference, signals[reference_finger], KIN_FS, f_task)

    rows = []
    for finger in FINGERS:
        is_reference = finger == reference_finger
        rows.append({
            "participant": participant, "task": task, "frequency": freq, "repetition": rep,
            "finger": finger, "is_reference_finger": is_reference,
            "thdn_percent": thdn[finger],
            "coherence": coherence_value if is_reference else np.nan,
            "phase_lag_rad": phase_lag if is_reference else np.nan
        })
    return pd.DataFrame(rows)


def _print_summary(df: pd.DataFrame) -> None:
    print(f"  thdn_percent: mean={df['thdn_percent'].mean():.2f}, std={df['thdn_percent'].std():.2f} "
          f"(n={len(df)} finger-trials)")
    ref = df[df["is_reference_finger"]]
    print(f"  coherence (reference finger): mean={ref['coherence'].mean():.3f}, std={ref['coherence'].std():.3f} "
          f"(n={len(ref)} trials)")
    print(f"  phase_lag_rad (reference finger): mean={ref['phase_lag_rad'].mean():.3f}, "
          f"std={ref['phase_lag_rad'].std():.3f} (n={len(ref)} trials)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--task", default=None, help="1-8 (ignored if --all)")
    parser.add_argument("--freq", default=None, help="'05' or '075' (ignored if --all)")
    parser.add_argument("--rep", type=int, default=None, help="0, 1, or 2 (ignored if --all)")
    parser.add_argument("--all", action="store_true", help="process all 48 task trials for the participant")
    parser.add_argument("--output-csv", type=Path, default=None, help="Save the per-finger results table")
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

    print(f"{args.participant}: {len(frames)}/{len(combos)} trial(s) processed, {len(result)} finger-trial rows")
    _print_summary(result)

    if args.output_csv is not None:
        result.to_csv(args.output_csv, index=False)
        print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
