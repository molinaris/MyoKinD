#!/usr/bin/env python3
"""Baseline noise-floor RMS of the HD sEMG ``rest`` recordings, and how much of
that floor is quantization vs. genuine (analog/physiological) noise.

The dataset's HD sEMG CSVs store integer microvolt (uV) values (see
dataset/README.md). The acquisition system's native ADC (Intan RHD2164) has a
0.195 uV/bit resolution, but the distributed values are rounded to whole
microvolts - a storage quantization step of 1 uV. That step alone adds a
quantization noise floor of 1/sqrt(12) = 0.29 uV RMS to every channel,
regardless of the signal it is applied to.

This script computes the *measured* RMS of each ``rest`` recording (same
4-44 s trim and raw/unfiltered convention as validation/hdsemg_quality.py's
SNR reference, since this characterizes the raw acquisition's noise floor
itself) and expresses the storage quantization noise as an SNR penalty
relative to it, treating the two as independent noise sources whose
variances add:

    penalty_db = 10*log10((sigma_measured^2 + sigma_q^2) / sigma_measured^2)

Usage:
    python rest_noise_rms.py --root /path/to/Dataset --participant Sub001 --all --output-csv sub001_rest_rms.csv
    python rest_noise_rms.py --root /path/to/Dataset --participant Sub001 --rep 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.load_hdsemg import HDSEMG_FS, load_hdsemg_csv, trial_filename  # noqa: E402
from validation.hdsemg_quality import REPETITIONS, trim_signal  # noqa: E402

STORAGE_LSB_UV = 1.0                                       # dataset's stored integer-uV rounding
QUANTIZATION_NOISE_RMS_UV = STORAGE_LSB_UV / np.sqrt(12)    # ~0.2887 uV


def compute_rms(x: np.ndarray) -> np.ndarray:
    """Per-channel RMS along axis 0."""
    return np.sqrt(np.mean(x**2, axis=0))


def quantization_penalty_db(measured_rms_uv: np.ndarray,
                             quant_noise_uv: float = QUANTIZATION_NOISE_RMS_UV) -> np.ndarray:
    """SNR penalty (dB) added by the storage quantization noise on top of the
    measured baseline noise (independent variances, so they add in power)."""
    return 10 * np.log10(1 + (quant_noise_uv / measured_rms_uv) ** 2)


def process_rest_trial(root: Path, participant: str, rep: int) -> pd.DataFrame:
    """RMS (and quantization penalty) for every one of the 128 channels of one ``rest`` recording."""
    rest_path = root / participant / "HD_sEMG" / trial_filename(participant, "rest", None, rep)
    rest_raw = trim_signal(load_hdsemg_csv(rest_path), HDSEMG_FS)

    rms = compute_rms(rest_raw)
    penalty = quantization_penalty_db(rms)

    n_channels = rest_raw.shape[1]
    muscle = np.where(np.arange(n_channels) < 64, "EDC", "FDS")
    channel_in_muscle = np.where(np.arange(n_channels) < 64, np.arange(n_channels), np.arange(n_channels) - 64)

    return pd.DataFrame({
        "participant": participant, "repetition": rep,
        "muscle": muscle, "channel": channel_in_muscle,
        "rms_uv": rms, "quant_penalty_db": penalty,
    })


def _print_summary(df: pd.DataFrame) -> None:
    rms, penalty = df["rms_uv"], df["quant_penalty_db"]
    print(f"  rms_uv: mean={rms.mean():.3f}, std={rms.std():.3f}, min={rms.min():.3f}, max={rms.max():.3f} "
          f"(n={len(df)} channel-trials)")
    print(f"  quant_penalty_db: mean={penalty.mean():.3f}, worst-case={penalty.max():.3f} "
          f"(quantization noise floor = {QUANTIZATION_NOISE_RMS_UV:.4f} uV RMS)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--rep", type=int, default=None, help="0, 1, or 2 (ignored if --all)")
    parser.add_argument("--all", action="store_true", help="process all 3 rest repetitions for the participant")
    parser.add_argument("--output-csv", type=Path, default=None, help="Save the per-channel results table")
    args = parser.parse_args()

    if args.all:
        reps = REPETITIONS
    else:
        if args.rep is None:
            raise SystemExit("Provide --rep, or use --all.")
        reps = [args.rep]

    frames = []
    for rep in reps:
        try:
            frames.append(process_rest_trial(args.root, args.participant, rep))
        except FileNotFoundError as exc:
            print(f"  skipping {args.participant} rest rep={rep}: {exc}")
    result = pd.concat(frames, ignore_index=True)

    print(f"{args.participant}: {len(frames)}/{len(reps)} rest repetition(s) processed, "
          f"{len(result)} channel-trial rows")
    _print_summary(result)

    if args.output_csv is not None:
        result.to_csv(args.output_csv, index=False)
        print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
