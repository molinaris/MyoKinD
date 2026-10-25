#!/usr/bin/env python3
"""Demonstrate the HD sEMG <-> kinematics time alignment for one trial.

There is no timestamp or trigger column in either modality's CSV files: a
common hardware trigger is assumed to start both recordings at t=0, but that
is a design assumption, not something verifiable from the files themselves.
This script does **not** perform any real signal-based synchronization (e.g.
cross-correlation); it only reproduces the original analysis approach:
assume t=0 is shared, trim
both modalities to the same 4-44 s window, resample the kinematics (100 Hz)
onto the HD sEMG sample grid, and plot everything on a common time axis. As a
plausibility check (not a proof of synchronism), the coherence between the
representative finger's angle and the task reference signal is also printed.

Usage:
    python synchronize_modalities.py --root /path/to/Dataset --participant Sub001 --task 7 --freq 05 --rep 0 --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from examples.load_hdsemg import (HDSEMG_FS, load_hdsemg_csv, moving_rms,  # noqa: E402
                                  split_edc_fds, trial_filename)
from examples.load_kinematics import KIN_FS, load_angles_csv  # noqa: E402
from validation.kinematic_quality import (FREQ_CODE_TO_HZ, compute_coherence_phase,  # noqa: E402
                                          select_reference_finger, task_reference, trim_signal)

RMS_WINDOW_S = 0.3


def build_time_vectors(n_emg: int, n_kin: int) -> tuple[np.ndarray, np.ndarray]:
    """Time vectors for the (already trimmed) HD sEMG and kinematics segments,
    both starting at 0 s - i.e. assuming a shared t=0 across modalities."""
    return np.arange(n_emg) / HDSEMG_FS, np.arange(n_kin) / KIN_FS


def resample_kinematics_to_emg(kinematics_signal: np.ndarray, n_emg_samples: int) -> np.ndarray:
    """Resample a 100 Hz kinematics signal onto the HD sEMG sample grid via
    naive resampling (no cross-correlation alignment)."""
    return resample(kinematics_signal, n_emg_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--task", required=True, help="1-8")
    parser.add_argument("--freq", required=True, help="'05' or '075'")
    parser.add_argument("--rep", type=int, required=True, help="0, 1, or 2")
    parser.add_argument("--plot", action="store_true", help="Show HD sEMG RMS and kinematics on a common time axis")
    args = parser.parse_args()

    emg_path = args.root / args.participant / "HD_sEMG" / trial_filename(
        args.participant, args.task, args.freq, args.rep)
    angles_path = args.root / args.participant / "HandKinematics" / "Angles" / trial_filename(
        args.participant, args.task, args.freq, args.rep)

    emg_raw = trim_signal(load_hdsemg_csv(emg_path), HDSEMG_FS)
    angles = load_angles_csv(angles_path)

    reference_finger = select_reference_finger(args.task)
    finger_signal = trim_signal(angles[reference_finger].to_numpy(), KIN_FS)

    t_emg, t_kin = build_time_vectors(emg_raw.shape[0], finger_signal.shape[0])
    f_task = FREQ_CODE_TO_HZ[args.freq]

    # Plausibility check: coherence at the kinematics' native sampling rate
    # (this is NOT part of the alignment procedure itself, just a sanity check).
    reference_at_kin_fs = task_reference(t_kin, f_task)
    coherence_value, phase_lag = compute_coherence_phase(reference_at_kin_fs, finger_signal, KIN_FS, f_task)
    print(f"{args.participant} task={args.task} freq={args.freq} rep={args.rep}")
    print(f"  HD sEMG: {emg_raw.shape[0]} samples ({t_emg[-1]:.2f} s) at {HDSEMG_FS:.2f} Hz")
    print(f"  Kinematics ({reference_finger}): {finger_signal.shape[0]} samples "
          f"({t_kin[-1]:.2f} s) at {KIN_FS:.0f} Hz")
    print(f"  Plausibility check - coherence with task reference at {f_task} Hz: {coherence_value:.3f} "
          f"(phase lag: {phase_lag:.3f} rad)")

    if not args.plot:
        return

    import matplotlib.pyplot as plt

    edc, fds = split_edc_fds(emg_raw)
    rms_window_samples = int(round(RMS_WINDOW_S * HDSEMG_FS))
    rms_edc = moving_rms(edc, rms_window_samples).mean(axis=1)
    rms_fds = moving_rms(fds, rms_window_samples).mean(axis=1)

    finger_on_emg_grid = resample_kinematics_to_emg(finger_signal, emg_raw.shape[0])
    reference_on_emg_grid = task_reference(t_emg, f_task)

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t_emg, rms_edc, color="tab:blue")
    axes[0].set_ylabel("Mean EDC RMS (uV)")
    axes[1].plot(t_emg, rms_fds, color="tab:orange")
    axes[1].set_ylabel("Mean FDS RMS (uV)")
    axes[2].plot(t_emg, finger_on_emg_grid, color="tab:green", label=f"{reference_finger} finger angle")
    ax2 = axes[2].twinx()
    ax2.plot(t_emg, reference_on_emg_grid, color="gray", linestyle="--", alpha=0.7, label="task reference")
    axes[2].set_ylabel("Position (deg)")
    axes[2].set_xlabel("Time (s), relative to the 4-44 s trimmed window")
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    angle_bottom, angle_top = axes[2].get_ylim()
    reference_bottom, reference_top = ax2.get_ylim()
    angle_ticks = axes[2].get_yticks()
    reference_ticks = ax2.get_yticks()
    axes[2].set_ylim(angle_bottom, angle_top + 0.25 * (angle_top - angle_bottom))
    ax2.set_ylim(reference_bottom, reference_top + 0.25 * (reference_top - reference_bottom))
    angle_ticks = angle_ticks[(angle_ticks >= angle_bottom) & (angle_ticks <= angle_top)]
    reference_ticks = reference_ticks[(reference_ticks >= reference_bottom) & (reference_ticks <= reference_top)]
    axes[2].set_yticks(angle_ticks)
    ax2.set_yticks(reference_ticks)
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, frameon=False, ncols=2, fontsize=9,
                   loc="upper center", bbox_to_anchor=(0.5, 0.98))
    plt.show()


if __name__ == "__main__":
    main()
