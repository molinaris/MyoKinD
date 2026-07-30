#!/usr/bin/env python3
"""Load a single HD sEMG trial, split it into EDC/FDS, and compute a moving RMS.

The CSV has no header: 128 columns (1-64 = Extensor Digitorum Communis / EDC,
65-128 = Flexor Digitorum Superficialis / FDS), 92160 rows sampled at
2052.5245535714284 Hz. Values are used directly as microvolts (uV).

Usage:
    python load_hdsemg.py --file /path/to/Sub001_1_05_450_0.csv --plot
    python load_hdsemg.py --root /path/to/Dataset --participant Sub001 --task 1 --freq 05 --rep 0 --plot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import oaconvolve

HDSEMG_FS = 2052.5245535714284      # Hz; not stored in the CSV
RMS_WINDOW_S = 0.3                  # 300 ms moving window

# Task code -> task name (the "T" field of SubXXX_T_F_450_R.csv). This is the
# single source of truth for this mapping - every other script/doc should
# reference it rather than repeating the names inline.
TASK_NAMES = {
    "1": "Index",
    "2": "Middle",
    "3": "Ring-Little",
    "4": "Thumb",
    "5": "I. Pinch",
    "6": "M. Pinch",
    "7": "Tripod",
    "8": "5-Finger",
}


def trial_filename(participant: str, task: str, freq: str | None, rep: int) -> str:
    """Build a dataset filename from its (participant, task, frequency, repetition).

    ``task`` is either a task number as a string ('1'..'8', see TASK_NAMES for
    what each one is) or one of the two special conditions ('rest',
    'start_position'), which have no frequency.
    """
    if task in ("rest", "start_position"):
        return f"{participant}_{task}_{rep}.csv"
    return f"{participant}_{task}_{freq}_450_{rep}.csv"


def load_hdsemg_csv(path: Path) -> np.ndarray:
    """Load a raw HD sEMG trial as a (92160, 128) array, in microvolts."""
    return np.loadtxt(path, delimiter=",")


def split_edc_fds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the 128 HD sEMG columns into (EDC, FDS), 64 columns each."""
    return x[:, :64], x[:, 64:]


def moving_rms(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Centered moving RMS along axis 0 (oaconvolve-based moving average of the
    squared signal, same shape as the input)."""
    x_squared = x**2
    if x.ndim == 1:
        kernel = np.ones(window_samples) / window_samples
        return np.sqrt(oaconvolve(x_squared, kernel, mode="same"))
    kernel = np.ones((window_samples, 1)) / window_samples
    return np.sqrt(oaconvolve(x_squared, kernel, mode="same", axes=0))


def _resolve_path(args: argparse.Namespace) -> Path:
    if args.file is not None:
        return args.file
    if not (args.root and args.participant and args.task is not None and args.rep is not None):
        raise SystemExit("Provide --file, or --root/--participant/--task/--rep "
                          "(and --freq for numbered tasks).")
    name = trial_filename(args.participant, args.task, args.freq, args.rep)
    return args.root / args.participant / "HD_sEMG" / name


def _plot(edc: np.ndarray, fds: np.ndarray, rms_window_samples: int) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(edc.shape[0]) / HDSEMG_FS
    rms_edc = moving_rms(edc, rms_window_samples)
    rms_fds = moving_rms(fds, rms_window_samples)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ax, rms, label in zip(axes, (rms_edc, rms_fds), ("EDC", "FDS")):
        ax.plot(t, rms, color="tab:blue", alpha=0.15, linewidth=0.5)
        ax.plot(t, rms.mean(axis=1), color="tab:blue", linewidth=1.5, label="channel mean")
        ax.set_ylabel(f"{label} RMS (uV)")
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=None, help="Direct path to an HD sEMG CSV file")
    parser.add_argument("--root", type=Path, default=None, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", default=None, help="e.g. Sub001")
    parser.add_argument("--task", default=None,
                         help="1-8 (see TASK_NAMES: 1=Index, 2=Middle, 3=Ring-Little, 4=Thumb, "
                              "5=I.Pinch, 6=M.Pinch, 7=Tripod, 8=5-Finger), 'rest', or 'start_position'")
    parser.add_argument("--freq", default=None, help="'05' or '075' (ignored for rest/start_position)")
    parser.add_argument("--rep", type=int, default=None, help="0, 1, or 2")
    parser.add_argument("--plot", action="store_true", help="Show a channel-wise RMS plot")
    args = parser.parse_args()

    path = _resolve_path(args)
    data = load_hdsemg_csv(path)
    edc, fds = split_edc_fds(data)

    print(f"Loaded {path}")
    if args.task in TASK_NAMES:
        print(f"  task {args.task}: {TASK_NAMES[args.task]}")
    print(f"  shape: {data.shape} (rows x channels), duration: {data.shape[0] / HDSEMG_FS:.2f} s")
    print(f"  EDC: {edc.shape}, FDS: {fds.shape}")

    if args.plot:
        rms_window_samples = int(round(RMS_WINDOW_S * HDSEMG_FS))
        _plot(edc, fds, rms_window_samples)


if __name__ == "__main__":
    main()
