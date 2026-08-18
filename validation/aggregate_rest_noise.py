#!/usr/bin/env python3
"""Aggregate validation/rest_noise_rms.py across participants into one
population-level view of the HD sEMG baseline noise floor, and of how much
SNR penalty the dataset's 1 uV storage quantization adds on top of it.

Usage:
    python aggregate_rest_noise.py --root /path/to/Dataset --participants Sub001 Sub002 Sub003 --output-csv rest_noise.csv
    python aggregate_rest_noise.py --root /path/to/Dataset --participants all --output-csv rest_noise.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validation.rest_noise_rms import QUANTIZATION_NOISE_RMS_UV, REPETITIONS, process_rest_trial  # noqa: E402

ALL_PARTICIPANTS = [f"Sub{n:03d}" for n in range(1, 22)]


def collect(root: Path, participants: list[str]) -> pd.DataFrame:
    frames = []
    for participant in participants:
        for rep in REPETITIONS:
            try:
                frames.append(process_rest_trial(root, participant, rep))
            except FileNotFoundError as exc:
                print(f"  skipping {participant} rest rep={rep}: {exc}")
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, penalty_threshold_db: float = 0.35) -> None:
    rms, penalty = df["rms_uv"], df["quant_penalty_db"]
    n = len(df)
    over_threshold = int((penalty > penalty_threshold_db).sum())

    print(f"\n{n} channel-trials over {df['participant'].nunique()} participant(s):")
    print(f"  measured rest RMS (uV): mean={rms.mean():.3f}, std={rms.std():.3f}, "
          f"min={rms.min():.3f}, p5={rms.quantile(0.05):.3f}, max={rms.max():.3f}")
    print(f"  quantization noise floor (1 uV storage step): {QUANTIZATION_NOISE_RMS_UV:.4f} uV RMS")
    print(f"  quantization SNR penalty (dB): mean={penalty.mean():.3f}, worst-case={penalty.max():.3f}")
    print(f"  channel-trials with penalty > {penalty_threshold_db} dB: {over_threshold}/{n} "
          f"({100 * over_threshold / n:.1f}%)")

    worst = df.loc[penalty.idxmax()]
    print(f"  worst-case channel-trial: {worst['participant']} rep={worst['repetition']} "
          f"{worst['muscle']}{int(worst['channel'])}, rms_uv={worst['rms_uv']:.3f}, "
          f"penalty={worst['quant_penalty_db']:.3f} dB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participants", nargs="+", required=True,
                         help="Participant IDs (e.g. Sub001 Sub002), or 'all' for all 21")
    parser.add_argument("--penalty-threshold-db", type=float, default=0.35,
                         help="Report how many channel-trials exceed this quantization SNR penalty (dB)")
    parser.add_argument("--output-csv", type=Path, default=None,
                         help="Save the per-channel-trial table (participant, repetition, muscle, "
                              "channel, rms_uv, quant_penalty_db)")
    args = parser.parse_args()

    participants = ALL_PARTICIPANTS if args.participants == ["all"] else args.participants

    df = collect(args.root, participants)
    summarize(df, args.penalty_threshold_db)

    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
        print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    main()
