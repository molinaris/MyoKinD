#!/usr/bin/env python3

"""Plot one trial's kinematic metrics: continuous finger-position trajectories,
per-cycle position and velocity, and frequency spectra for all five fingers.

The figure summarizes finger kinematics using the joint-angle trajectories
stored in the HandKinematics/Angles files. Continuous position trajectories
are displayed over a 15 s interval, while per-cycle angular position and
velocity profiles are computed from the complete trimmed trial. Frequency
spectra are shown for each finger, with the fundamental task frequency and
its harmonic components highlighted. The magnitude of the highlighted
components is scaled by each finger's fundamental amplitude relative to the
largest fundamental amplitude across the five fingers.

Parameters:

- Temporal trim: discard the first and last 5 s of each 45 s trial to avoid
  transients associated with task onset and offset.
- Movement cycles: identify cycles from the task-specific reference finger and
  apply the resulting boundaries to all five fingers.
- THDN: Welch PSD (8 s window, periodic Hamming window), with the fundamental
  defined by the prescribed task frequency (0.5 or 0.75 Hz) and four
  additional harmonics. Component amplitudes are estimated from the target
  frequency bin ± 1 bin, and the harmonic distortion ratio is scaled by each
  finger's fundamental amplitude relative to the largest across the five
  fingers.


Usage:
    python plot_example_kinematic_metrics.py --root /path/to/Dataset --participant Sub017 --task 6 --freq 05 --rep 0 --output Mpinch_example.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.load_hdsemg import trial_filename  # noqa: E402
from examples.load_kinematics import KIN_FS, load_angles_csv  # noqa: E402
from validation.kinematic_quality import compute_thdn, trim_signal  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent

FINGERS = ["thumb", "index", "middle", "ring", "little"]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
COLORS = ["#0073E6", "#E94252", "#FF8000", "#7BC235", "#A057E8"]

FREQ_CODE_TO_HZ = {"05": 0.5, "075": 0.75}
TRIM_START_S = 5.0
TRIM_END_S = 40.0

THDN_WINDOW = "hamming"
THDN_WINDOW_S = 8.0
THDN_N_HARMONICS = 5
THDN_N_BINS = 1


def plot_example_kinematic_metrics(root: Path, participant: str, task: str, freq: str,
                                   rep: int, fs: float, reference: pd.DataFrame | None = None) -> Figure:
    """Plot one trial's continuous and per-cycle kinematic metrics."""
    from biomechanical_model.estimate_kinematic_angles import get_cycle_segments # noqa: E402
    if reference is None:
        name = trial_filename(participant, task, freq, rep)
        angles_path = root / participant / "HandKinematics" / "Angles" / name
        reference = load_angles_csv(angles_path)
    trimmed_reference = pd.DataFrame(trim_signal(reference[FINGERS].to_numpy(dtype=float),
                                                 fs, TRIM_START_S, TRIM_END_S), columns=FINGERS)
    f_task = FREQ_CODE_TO_HZ[freq]
    cycle_segments = get_cycle_segments(trimmed_reference, task=task, f_task=f_task, fs=KIN_FS)

    if not cycle_segments:
        raise ValueError("No cycle segments were found.")

    signals = {finger: trimmed_reference[finger].to_numpy(dtype=float) for finger in FINGERS}
    _, thdn_details = compute_thdn(signals, fs=fs, f_task=f_task, window_s=THDN_WINDOW_S,
                                   n_harmonics=THDN_N_HARMONICS, n_bins=THDN_N_BINS, window=THDN_WINDOW, return_details=True)

    data = trimmed_reference[FINGERS].to_numpy(dtype=float)
    x_decimals = 1 if np.isclose(f_task, 0.5) else 2

    def clean_axis(ax: Axes) -> None:
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def panel_label(ax: Axes, label: str, x: float = 0.02) -> None:
        ax.text(x, 0.98, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")

    def vertical_label(ax: Axes, x: float, text: str, fontsize: float) -> None:
        ax.text(x, 0.5, text, transform=ax.transAxes, fontsize=fontsize, rotation=90, va="center", ha="center")

    def position_axis(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        min_tick = int(np.floor(min_value)) if min_value < 0 else 0
        max_tick = max(1, int(np.floor(max_value)))
        middle_tick = int(np.floor(max_tick / 2))
        ticks = np.array([min_tick, 0, middle_tick, max_tick])
        span = max_tick - min_tick

        return (ticks, (min(min_value, min_tick) - 0.02 * span, max_value + 0.15 * span))

    def velocity_axis(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        max_abs = np.nanmax(np.abs(values))
        max_tick = max(1, int(np.floor(max_abs)))
        middle_tick = int(np.floor(max_tick / 2))
        ticks = np.array([-max_tick, -middle_tick, 0, middle_tick, max_tick])
        margin = 0.07 * (2 * max_tick)

        return (ticks, (-max_abs - margin, max_abs + margin))

    def psd_axis(max_value: float) -> tuple[np.ndarray, tuple[float, float]]:
        max_tick = max(1, int(np.floor(max_value)))
        ticks = np.unique(np.floor(np.linspace(0, max_tick, 5)).astype(int))

        return (ticks, (0, max_value + 0.15 * max_tick))

    # Cycle data
    max_length = max(len(segment) for segment in cycle_segments)
    period = 1.0 / (2.0 * f_task)
    time_vector = np.linspace(-period, period, max_length)
    x_limit = period * 1.35
    cycle_velocities = [np.gradient(segment, 1.0 / fs, axis=0) for segment in cycle_segments]
    position_ticks, position_ylim = position_axis(np.concatenate([segment.ravel() for segment in cycle_segments]))
    velocity_ticks, velocity_ylim = velocity_axis(np.concatenate([velocity.ravel() for velocity in cycle_velocities]))
    cycle_ticks = [-period, 0, period]
    cycle_labels = [f"{x:.{x_decimals}f}" for x in cycle_ticks]

    # PSD data
    fundamental_amplitudes = [thdn_details[finger]["fundamental_amplitude"] for finger in FINGERS]
    a_fund_max = max(fundamental_amplitudes) or 1.0
    max_psd = max(np.nanmax(np.sqrt(thdn_details[finger]["pxx"])[(thdn_details[finger]["f"] >= 0)
                                                                 & (thdn_details[finger]["f"] <= 3.75)]) for finger in FINGERS)
    psd_ticks, psd_ylim = psd_axis(max_psd)
    max_frequency_tick = 3.75 if np.isclose(f_task, 0.75) else 3.00
    frequency_ticks = np.arange(0, max_frequency_tick + f_task / 2, f_task)
    frequency_ticks = frequency_ticks[frequency_ticks <= max_frequency_tick]
    frequency_labels = [f"{x:.{x_decimals}f}" for x in frequency_ticks]

    # Figure
    fig = plt.figure(figsize=(14, 11))

    # A) Continuous position
    ax_top = plt.subplot2grid((4, 5), (0, 0), colspan=5)
    time = TRIM_START_S + np.arange(len(data)) / fs
    mask = (time >= 5.0) & (time <= 20.0)
    time_plot = time[mask]
    data_plot = data[mask]
    top_ticks, top_ylim = position_axis(data_plot.ravel())
    top_span = top_ticks[-1] - top_ticks[0]
    top_ylim = (top_ylim[0], np.nanmax(data_plot) + 0.30 * top_span)

    for i, finger_name in enumerate(FINGER_NAMES):
        ax_top.plot(time_plot, data_plot[:, i], color=COLORS[i], linewidth=2.1, label=finger_name)

    ax_top.set(xlim=(5, 20), xticks=range(5, 21), ylim=top_ylim, yticks=top_ticks)
    ax_top.tick_params(labelsize=12)
    vertical_label(ax_top, -0.06, r"Position ($^\circ$)", 14)
    vertical_label(ax_top, -0.04, "Flexion/Opposition →", 10)
    legend = ax_top.legend(fontsize=12, loc="upper center", ncol=5)

    for line in legend.get_lines():
        line.set_linewidth(5)

    panel_label(ax_top, "A)", x=0.002)
    clean_axis(ax_top)

    # B-K) Position and velocity
    axs = np.empty((2, 5), dtype=object)
    for row in range(2):
        for col in range(5):
            ax = plt.subplot2grid((4, 5), (row + 1, col), sharex=(axs[row, 0] if col else None),
                                  sharey=(axs[row, 0] if col else None))
            axs[row, col] = ax
            if col:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.tick_params(labelbottom=False)

            label = f"{chr(66 + row * 5 + col)})"

            if row == 0:
                label += f" {FINGER_NAMES[col]}"

            panel_label(ax, label)
            clean_axis(ax)

    for finger_idx in range(5):
        angles = []
        velocities = []

        for segment, velocity_segment in zip(cycle_segments, cycle_velocities):
            angle = segment[:, finger_idx]
            velocity = velocity_segment[:, finger_idx]
            padded_angle = np.full(max_length, np.nan)
            padded_velocity = np.full(max_length, np.nan)
            padded_angle[:len(angle)] = angle
            padded_velocity[:len(velocity)] = velocity
            angles.append(padded_angle)
            velocities.append(padded_velocity)
            axs[0, finger_idx].plot(time_vector, padded_angle, color=COLORS[finger_idx], alpha=0.15)
            axs[1, finger_idx].plot(time_vector, padded_velocity, color=COLORS[finger_idx], alpha=0.15)

        axs[0, finger_idx].plot(time_vector, np.nanmean(angles, axis=0),
                                color=COLORS[finger_idx], linewidth=2.5)
        axs[1, finger_idx].plot(time_vector, np.nanmean(velocities, axis=0),
                                color=COLORS[finger_idx], linewidth=2.5)

        for row in range(2):
            axs[row, finger_idx].set_xlim(-x_limit, x_limit)
            axs[row, finger_idx].set_xticks(cycle_ticks)
            axs[row, finger_idx].set_xticklabels(cycle_labels, fontsize=9)

        axs[0, finger_idx].set_ylim(position_ylim)
        axs[0, finger_idx].set_yticks(position_ticks)
        axs[1, finger_idx].set_ylim(velocity_ylim)
        axs[1, finger_idx].set_yticks(velocity_ticks)
        axs[1, finger_idx].set_xlabel("Time (s)", fontsize=12)

        for row in range(2):
            axs[row, finger_idx].tick_params(axis="y", labelsize=12)

    vertical_label(axs[0, 0], -0.36, r"Position ($^\circ$)", 14)
    vertical_label(axs[0, 0], -0.26, "Flexion/Opposition →", 11)
    vertical_label(axs[1, 0], -0.36, r"Velocity ($^\circ$/s)", 14)
    vertical_label(axs[1, 0], -0.26, "Flexion/Opposition →", 11)

    # L-P) PSD
    axs_spectra = []

    for col in range(5):
        ax = plt.subplot2grid((4, 5), (3, col), sharey=(axs_spectra[0] if col else None))
        if col:
            ax.tick_params(labelleft=False)
        panel_label(ax, f"{chr(76 + col)})")
        clean_axis(ax)
        axs_spectra.append(ax)

    for finger_idx, (finger, ax) in enumerate(zip(FINGERS, axs_spectra)):
        f = thdn_details[finger]["f"]
        pxx = thdn_details[finger]["pxx"]
        a_fund = thdn_details[finger]["fundamental_amplitude"]
        delta_f = f[1] - f[0] if len(f) > 1 else 1.0
        fft_mag = np.sqrt(pxx)
        visible = (f >= 0) & (f <= 3.75)
        scale = a_fund / a_fund_max

        ax.plot(f[visible], fft_mag[visible], linewidth=1.5, color="k", zorder=2)
        ax.fill_between(f[visible], 0, fft_mag[visible], color="#F0F0F0F9", zorder=0)

        for harmonic in range(1, THDN_N_HARMONICS + 1):
            harmonic_frequency = harmonic * f_task
            if harmonic_frequency > 3.75:
                break
            idx_center = np.argmin(np.abs(f - harmonic_frequency))
            idx_harm = np.arange(max(0, idx_center - THDN_N_BINS),
                                 min(len(f) - 1, idx_center + THDN_N_BINS) + 1)
            if len(idx_harm) == 1:
                idx = idx_harm[0]
                ax.fill_between([f[idx] - delta_f / 2, f[idx] + delta_f / 2], [0, 0],
                                [fft_mag[idx] * scale, fft_mag[idx] * scale], color=COLORS[finger_idx],
                                alpha=0.28, step="mid", zorder=1)
            else:
                ax.fill_between(f[idx_harm], fft_mag[idx_harm] * scale,
                                color=COLORS[finger_idx], alpha=0.28, step="mid", zorder=1)

        ax.set_xlim(0, frequency_ticks[-1] + 0.25)
        ax.set_xticks(frequency_ticks)
        ax.set_xticklabels(frequency_labels, fontsize=9)
        ax.set_ylim(psd_ylim)
        ax.set_yticks(psd_ticks)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_xlabel("Frequency (Hz)", fontsize=12)

        if finger_idx == 0:
            ax.set_ylabel(r"PSD ($\circ^2$/Hz)", fontsize=14)

    plt.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.08, hspace=0.25, wspace=0.15)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--task", required=True, help="1-8")
    parser.add_argument("--freq", required=True, help="'05' or '075'")
    parser.add_argument("--rep", required=True, type=int, help="0, 1, or 2")
    parser.add_argument("--output", type=Path, default=None, help="Figure file name, e.g. .png (saved in biomechanical_model/)")
    args = parser.parse_args()

    fig = plot_example_kinematic_metrics(args.root, args.participant, args.task, args.freq, args.rep, fs=KIN_FS)

    if args.output is not None:
        output_path = SCRIPT_DIR / args.output.name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()