#!/usr/bin/env python3
"""Plot one trial's HD sEMG activity and kinematics: channel-wise HD sEMG RMS
within the EDC and FDS electrode grids (light) with their across-channel mean
(bold), together with the joint-angle trajectories of all five fingers.

The trial is trimmed to its 4-44 s window (discarding the first 4 s and the
last 1 s of every 45 s trial, to avoid the transients of task onset/offset),
the same convention used throughout this repository. The HD sEMG signal is
band-pass filtered (10-500 Hz, matching validation/hdsemg_quality.py) before
RMS, since a clean RMS envelope for visual inspection benefits from removing
out-of-band noise - note this filtering is deliberately NOT applied when
computing the SNR/SMR/SPR quality metrics themselves (see
validation/hdsemg_quality.py), which are meant to characterize the raw
acquisition.

Usage:
    python plot_example_trial.py --root /path/to/Dataset --participant Sub001 --task 7 --freq 05 --rep 0 --output tripod_example.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.load_hdsemg import (HDSEMG_FS, load_hdsemg_csv, moving_rms,  # noqa: E402
                                  split_edc_fds, trial_filename)
from examples.load_kinematics import KIN_FS, load_angles_csv  # noqa: E402
from validation.hdsemg_quality import bandpass_filter  # noqa: E402
from validation.kinematic_quality import FREQ_CODE_TO_HZ, trim_signal  # noqa: E402

RMS_WINDOW_S = 0.3
FINGERS = ["thumb", "index", "middle", "ring", "little"]


def make_figure(root: Path, participant: str, task: str, freq: str, rep: int) -> Figure:
    import matplotlib.pyplot as plt

    name = trial_filename(participant, task, freq, rep)
    emg_path = root / participant / "HD_sEMG" / name
    angles_path = root / participant / "HandKinematics" / "Angles" / name

    emg_full = load_hdsemg_csv(emg_path)
    t_emg = trim_signal(np.arange(emg_full.shape[0]) / HDSEMG_FS, HDSEMG_FS)
    emg = bandpass_filter(trim_signal(emg_full, HDSEMG_FS), HDSEMG_FS)
    angles = load_angles_csv(angles_path)

    edc, fds = split_edc_fds(emg)
    rms_window_samples = int(round(RMS_WINDOW_S * HDSEMG_FS))
    rms_edc = moving_rms(edc, rms_window_samples)
    rms_fds = moving_rms(fds, rms_window_samples)

    fig, axes = plt.subplots(3, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(t_emg, rms_edc, color="tab:blue", alpha=0.15, linewidth=0.3)
    axes[0].plot(t_emg, rms_edc.mean(axis=1), color="tab:blue", linewidth=1.0, zorder=10)
    axes[0].set_ylabel("EDC RMS (uV)")
    axes[0].text(0.01, 1, "A)", transform=axes[0].transAxes, fontweight="bold", va="top")

    axes[1].plot(t_emg, rms_fds, color="tab:orange", alpha=0.15, linewidth=0.3)
    axes[1].plot(t_emg, rms_fds.mean(axis=1), color="tab:orange", linewidth=1.0, zorder=10)
    axes[1].set_ylabel("FDS RMS (uV)")
    axes[1].text(0.01, 1, "B)", transform=axes[1].transAxes, fontweight="bold", va="top")

    t_kin = trim_signal(angles["frame"].to_numpy() - 1, KIN_FS) / KIN_FS
    for finger in FINGERS:
        axes[2].plot(t_kin, trim_signal(angles[finger].to_numpy(), KIN_FS), label=finger)
    axes[2].set_ylabel("Position (deg)")
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.canvas.draw()
    angle_bottom, angle_top = axes[2].get_ylim()
    angle_ticks = axes[2].get_yticks()
    axes[2].set_ylim(angle_bottom, angle_top + 0.25 * (angle_top - angle_bottom))
    angle_ticks = angle_ticks[(angle_ticks >= angle_bottom) & (angle_ticks <= angle_top)]
    axes[2].set_yticks(angle_ticks)
    axes[2].legend(frameon=False, ncols=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.98))
    axes[2].text(0.01, 1, "C)", transform=axes[2].transAxes, fontweight="bold", va="top")

    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.suptitle(f"{participant} - task {task}, {FREQ_CODE_TO_HZ[freq]:.2f} Hz, rep {rep}")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--task", required=True, help="1-8")
    parser.add_argument("--freq", required=True, help="'05' or '075'")
    parser.add_argument("--rep", type=int, required=True, help="0, 1, or 2")
    parser.add_argument("--output", type=Path, default=None, help="Save the figure to this path (e.g. .png)")
    args = parser.parse_args()

    fig = make_figure(args.root, args.participant, args.task, args.freq, args.rep)

    if args.output is not None:
        fig.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
