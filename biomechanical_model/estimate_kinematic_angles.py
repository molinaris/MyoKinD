#!/usr/bin/env python3

"""Estimate and evaluate finger kinematics from 3D marker trajectories.

This script provides a single-trial example of the biomechanical hand model
and kinematic analysis implemented for the MyoKinD dataset. Finger angles are
estimated from 3D hand-marker trajectories and characterized using range of
movement (RoM), mean angular speed, and normalized total harmonic distortion
(THDN). RoM and mean angular speed are calculated on a cycle-by-cycle basis
after segmenting the movement using the task-specific reference finger and
excluding cycles that are too short relative to the expected movement period.
THDN is computed from the trimmed continuous angle signals.

For each finger, vectors A and B define the directions used to estimate the
kinematic angle, while vector C defines the normal to the projection plane.
Vectors A and B are projected onto the plane perpendicular to C as:

    A_perp = A - ((A · C) / ||C||^2) C
    B_perp = B - ((B · C) / ||C||^2) C

The signed angle between the projected vectors is then computed as:

    theta = arccos((A_perp · B_perp) / (||A_perp|| ||B_perp||))
    theta_signed = sign((A_perp x B_perp) · C) * theta

The resulting angles are expressed in degrees. Marker trajectories are used
as returned by examples.load_kinematics.load_trajectories_csv(), without
additional filtering during angle estimation.

Three modes are available for inspecting the analysis of a selected trial:

- angles: compare the angles estimated from the 3D marker trajectories with
  those obtained using the Vicon ProCalc tool (provided in the dataset),
  reporting the estimation error for each finger.
- metrics: quantify finger kinematics using RoM, mean angular speed, and THDN,
  and visualize the corresponding movement cycles and spectral components.
- interactive: inspect the hand-marker configuration and the 3D-to-2D
  projection geometry underlying the angle-estimation procedure over time.

Parameters:

- Temporal trim: 5-40 s of each 45 s trial.
- Cycle segmentation: Hilbert phase of the task-specific reference finger
  (Index/I. Pinch -> index; Middle/M. Pinch/Tripod -> middle;
  Ring-Little/5-Finger -> ring; Thumb -> thumb); cycles shorter than 80% of
  the expected period are discarded.
- THDN: Welch PSD (8 s Hamming window, no overlap), fundamental + 4
  harmonics, each estimated from the target frequency bin ± 1 bin, computed
  with validation.kinematic_quality.compute_thdn.

These parameters belong to the biomechanical-model analysis and differ from
the dataset-validation settings in validation/kinematic_quality.py (4-44 s
trim, 4 s periodic Hann window).

The biomechanical model and kinematic analysis implemented in this example
are described in:

    Avilés-Carrillo et al. (2026), "A Biomechanical Hand Model to Quantify
    Finger Joint Kinematics Using a 3D Motion Capture System."
    DOI: 10.64898/2026.02.09.704796

Usage:
    python estimate_kinematic_angles.py --root /path/to/Dataset --participant Sub001 --task 8 --freq 05 --rep 0 --mode angles

    python estimate_kinematic_angles.py --root /path/to/Dataset --participant Sub001 --task 8 --freq 05 --rep 0 --mode metrics

    python estimate_kinematic_angles.py --root /path/to/Dataset --participant Sub001 --task 8 --freq 05 --rep 0 --mode interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.load_hdsemg import trial_filename  # noqa: E402
from examples.load_kinematics import KIN_FS, load_angles_csv, load_trajectories_csv  # noqa: E402
from biomechanical_model.plot_example_kinematic_metrics import plot_example_kinematic_metrics  # noqa: E402
from biomechanical_model.plot_interactive_kinematic import plot_interactive_kinematic  # noqa: E402
from validation.kinematic_quality import compute_thdn, trim_signal  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

LEFT_HANDED_PARTICIPANTS = {"Sub005"}
FINGERS = ["thumb", "index", "middle", "ring", "little"]
MARKERS = ["MW", "LW", "PT", "DT", "PI", "DI", "PM", "DM", "PR", "DR", "PL", "DL"]

FREQ_CODE_TO_HZ = {"05": 0.5, "075": 0.75}

TRIM_START_S = 5.0
TRIM_END_S = 40.0
MIN_CYCLE_FRACTION = 0.8

THDN_WINDOW = "hamming"
THDN_WINDOW_S = 8.0
THDN_N_HARMONICS = 5
THDN_N_BINS = 1

# Representative finger per task (Index/I.Pinch -> index;
# Middle/M.Pinch/Tripod -> middle; Ring-Little/5-Finger -> ring; Thumb -> thumb). See
# examples.load_hdsemg.TASK_NAMES for what each task number actually is
# (e.g. task "3" = Ring-Little).
REFERENCE_FINGER_BY_TASK = {"1": "index", "2": "middle", "3": "ring", "4": "thumb",
                            "5": "index", "6": "middle", "7": "middle", "8": "ring"}


def _marker_xyz(trajectories: pd.DataFrame, marker: str) -> np.ndarray:
    """Return one marker trajectory as an (n_samples, 3) XYZ array."""
    return trajectories[[f"{marker}_x", f"{marker}_y", f"{marker}_z"]].to_numpy(dtype=float)


def _rowwise_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the row-wise dot product of two (n_samples, 3) arrays."""
    return np.einsum("ij,ij->i", a, b)


def _project_perpendicular(vector: np.ndarray, axis_c: np.ndarray) -> np.ndarray:
    """Project a vector onto the plane perpendicular to C.

    Implements:

        v_perp = v - ((v · C) / ||C||^2) C

    corresponding to Eqs. (1) and (2).
    """
    c_norm_sq = np.linalg.norm(axis_c, axis=1, keepdims=True) ** 2
    dot_vc = _rowwise_dot(vector, axis_c)[:, None]
    projection_on_c = (dot_vc / c_norm_sq) * axis_c
    v_perp = vector - projection_on_c

    return v_perp


def _signed_angle(a_perp: np.ndarray, b_perp: np.ndarray, axis_c: np.ndarray) -> np.ndarray:
    """Return the signed angle between A_perp and B_perp around C, in degrees.

    Eq. (3):
        theta = arccos((A_perp · B_perp) / (||A_perp|| ||B_perp||))

    Eq. (4):
        theta_signed = sign((A_perp x B_perp) · C) * theta
    """
    theta = np.arccos(_rowwise_dot(a_perp, b_perp) /
                      (np.linalg.norm(a_perp, axis=1) * np.linalg.norm(b_perp, axis=1)))
    sign_theta = np.sign(_rowwise_dot(np.cross(a_perp, b_perp), axis_c))
    theta_signed = sign_theta * theta

    return np.degrees(theta_signed)


def _build_vectors(trajectories: pd.DataFrame, participant: str) -> dict:
    """Construct CW and the A, B, and C vectors for all five fingers."""

    markers = {marker: _marker_xyz(trajectories, marker) for marker in MARKERS}
    markers["CW"] = (markers["MW"] + markers["LW"]) / 2.0

    A = {"thumb": markers["PT"] - markers["MW"],
         "index": markers["PI"] - markers["CW"],
         "middle": markers["PM"] - markers["CW"],
         "ring": markers["PR"] - markers["CW"],
         "little": markers["PL"] - markers["CW"]}

    B = {"thumb": markers["MW"] - markers["LW"],
         "index": markers["DI"] - markers["PI"],
         "middle": markers["DM"] - markers["PM"],
         "ring": markers["DR"] - markers["PR"],
         "little": markers["DL"] - markers["PL"]}

    C = {"thumb": markers["PI"] - markers["MW"],
         "index": markers["PI"] - markers["PM"],
         "middle": markers["PM"] - markers["PR"],
         "ring": markers["PR"] - markers["PL"],
         "little": markers["PR"] - markers["PL"]}

    if participant in LEFT_HANDED_PARTICIPANTS:
        C = {finger: -vector for finger, vector in C.items()}

    return {"markers": markers, "A": A, "B": B, "C": C}


def estimate_angles(trajectories: pd.DataFrame, participant: str) -> tuple[pd.DataFrame, dict]:
    """Estimate finger angles from the 3D marker trajectories."""
    geometry = _build_vectors(trajectories, participant)
    estimated = pd.DataFrame({"frame": trajectories["frame"].to_numpy(),
                              "sub_frame": trajectories["sub_frame"].to_numpy()})
    details = {}

    for finger in FINGERS:
        vector_a = geometry["A"][finger]
        vector_b = geometry["B"][finger]
        vector_c = geometry["C"][finger]

        a_perp = _project_perpendicular(vector_a, vector_c)
        b_perp = _project_perpendicular(vector_b, vector_c)
        theta = _signed_angle(a_perp, b_perp, vector_c)
        estimated[finger] = theta

        details[finger] = {"A": vector_a, "B": vector_b, "C": vector_c,
                           "A_perp": a_perp, "B_perp": b_perp, "angle_deg": theta}

    return estimated, {"geometry": geometry, "details": details}


def get_cycle_segments(reference: pd.DataFrame, task: str, f_task: float, fs: float) -> list[np.ndarray]:
    """Segment cycles using the task-specific reference finger."""

    data = reference[FINGERS].to_numpy(dtype=float)
    reference_finger = REFERENCE_FINGER_BY_TASK[task]
    reference_idx = FINGERS.index(reference_finger)
    ref_signal = data[:, reference_idx]
    analytic_signal = hilbert(ref_signal)
    instantaneous_phase = np.angle(analytic_signal)
    phase_diff = np.diff(instantaneous_phase)
    cycle_starts = np.where(np.diff(np.sign(phase_diff)) < 0)[0]
    expected_cycle_samples = fs / f_task
    min_cycle_samples = int(MIN_CYCLE_FRACTION * expected_cycle_samples)

    cycle_segments = []
    for i in range(len(cycle_starts) - 1):
        start = cycle_starts[i]
        end = cycle_starts[i + 1]
        if end - start >= min_cycle_samples:
            cycle_segments.append(data[start:end, :])

    return cycle_segments


def mean_sd(values: list[float] | np.ndarray) -> tuple[float, float]:
    """Return the mean and standard deviation."""

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    mean_value = np.mean(values)

    if len(values) < 2:
        return float(mean_value), np.nan

    sd_value = np.std(values, ddof=1)

    return float(mean_value), float(sd_value)


def calculate_rom(cycle_segments: list[np.ndarray]) -> pd.DataFrame:
    """Calculate RoM for each finger across cycles."""
    results = []
    for finger_idx, finger in enumerate(FINGERS):
        values = [np.max(segment[:, finger_idx]) - np.min(segment[:, finger_idx]) for segment in cycle_segments]
        mean_rom, sd_rom = mean_sd(values)
        results.append({"finger": finger, "RoM": float(mean_rom), "SD": float(sd_rom), "n_cycles": len(values), "values": values})

    return pd.DataFrame(results)


def calculate_mean_speed(cycle_segments: list[np.ndarray], fs: float) -> pd.DataFrame:
    """Calculate mean angular speed for each finger across cycles."""

    dt = 1.0 / fs
    results = []
    for finger_idx, finger in enumerate(FINGERS):
        mean_speeds = []
        for segment in cycle_segments:
            signal = segment[:, finger_idx]
            velocity = np.gradient(signal, dt)
            mean_speeds.append(np.mean(np.abs(velocity)))
        mean_speed, mean_speed_sd = mean_sd(mean_speeds)
        results.append({"finger": finger, "mean_speed": float(mean_speed),
                        "mean_speed_SD": float(mean_speed_sd), "n_cycles": len(cycle_segments)})

    return pd.DataFrame(results)


def _plot(estimated: pd.DataFrame, reference: pd.DataFrame | None = None):
    """Plot estimated angles against the dataset reference angles."""

    t = (estimated["frame"].to_numpy() - 1) / KIN_FS
    fig, axes = plt.subplots(5, 1, figsize=(9, 8), sharex=True)

    for ax, finger in zip(axes, FINGERS):
        ax.plot(t, estimated[finger], color="tab:blue", linewidth=2.0,
                alpha=0.65, label="Python implementation", zorder=1)

        if reference is not None:
            ax.plot(t, reference[finger], color="black", linewidth=2.0,
                    linestyle=(0, (1, 2)), label="Vicon ProCalc (Dataset)", zorder=2)

        ax.set_ylabel(f"{finger.capitalize()}\n(deg)")
        ax.spines[["right", "top"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.canvas.draw()
    y_bottom, y_top = axes[0].get_ylim()
    y_ticks = axes[0].get_yticks()
    axes[0].set_ylim(y_bottom, y_top + 0.3 * (y_top - y_bottom))
    y_ticks = y_ticks[(y_ticks >= y_bottom) & (y_ticks <= y_top)]
    axes[0].set_yticks(y_ticks)
    axes[0].legend(frameon=False, ncols=2, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.98))

    return fig


def run_angles(root: Path, participant: str, task: str, freq: str, rep: int) -> None:
    """Estimate angles, report estimation error, and show the comparison plot."""
    name = trial_filename(participant, task, freq, rep)
    base = root / participant / "HandKinematics"
    trajectories_path = base / "Trajectories" / name
    angles_path = base / "Angles" / name
    trajectories = load_trajectories_csv(trajectories_path)
    reference = load_angles_csv(angles_path)
    estimated, _ = estimate_angles(trajectories, participant)
    print("\nAngle estimation error:")

    for finger in FINGERS:
        error = (estimated[finger].to_numpy() - reference[finger].to_numpy())
        mean_error = np.mean(error)
        std_error = np.std(error)
        magnitude = max(abs(mean_error), std_error)
        if magnitude >= 1e-3:
            scale, unit = 1e3, "mdeg"
        else:
            scale, unit = 1e6, "µdeg"

        print(f"{finger:6s}: {mean_error * scale:8.3f} ± {std_error * scale:.3f} {unit}")

    _plot(estimated, reference)
    plt.show()


def run_metrics(root: Path, participant: str, task: str, freq: str, rep: int) -> Figure:
    """Compute kinematic metrics and return the metrics figure."""
    name = trial_filename(participant, task, freq, rep)
    angles_path = root / participant / "HandKinematics" / "Angles" / name
    reference = load_angles_csv(angles_path)
    trimmed_reference = pd.DataFrame(trim_signal(reference[FINGERS].to_numpy(dtype=float),
                                                 KIN_FS, TRIM_START_S, TRIM_END_S), columns=FINGERS)
    cycle_segments = get_cycle_segments(trimmed_reference, task=task, f_task=FREQ_CODE_TO_HZ[freq], fs=KIN_FS)
    print(f"\nNumber of cycles: {len(cycle_segments)}")

    rom = calculate_rom(cycle_segments)
    print("\n=== Range of movement ===")
    for _, row in rom.iterrows():
        print(f"  {row['finger'].capitalize():6s}: {row['RoM']:.2f} ± {row['SD']:.2f} °")

    mean_speed = calculate_mean_speed(cycle_segments, fs=KIN_FS)
    print("\n=== Mean angular speed ===")
    for _, row in mean_speed.iterrows():
        print(f"  {row['finger'].capitalize():6s}: {row['mean_speed']:.2f} ± {row['mean_speed_SD']:.2f} °/s")

    f_task = FREQ_CODE_TO_HZ[freq]
    signals = {finger: trimmed_reference[finger].to_numpy(dtype=float) for finger in FINGERS}
    thdn = compute_thdn(signals, fs=KIN_FS, f_task=f_task, window_s=THDN_WINDOW_S,
                        n_harmonics=THDN_N_HARMONICS, n_bins=THDN_N_BINS, window=THDN_WINDOW)
    print("\n=== THD_N ===")
    for finger in FINGERS:
        print(f"  {finger.capitalize():6s}: {thdn[finger]:.2f} %")

    fig = plot_example_kinematic_metrics(root, participant, task, freq, rep, fs=KIN_FS, reference=reference)

    return fig


def run_interactive(root: Path, participant: str, task: str, freq: str, rep: int) -> None:
    """Show the interactive angle-estimation geometry."""
    plot_interactive_kinematic(root, participant, task, freq, rep, fs=KIN_FS)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--task", required=True, help="1-8")
    parser.add_argument("--freq", required=True, choices=["05", "075"], help="'05' or '075'")
    parser.add_argument("--rep", required=True, type=int, help="0, 1, or 2")
    parser.add_argument("--mode", required=True, choices=["angles", "metrics", "interactive"],
                        help="Run angle validation, kinematic metrics, or interactive visualization")
    parser.add_argument("--output", type=Path, default=None, help="Optional figure file name (saved in biomechanical_model/)")
    args = parser.parse_args()

    if args.mode == "angles":
        run_angles(args.root, args.participant, args.task, args.freq, args.rep)

    elif args.mode == "metrics":
        fig = run_metrics(args.root, args.participant, args.task, args.freq, args.rep)

        if args.output is not None:
            output_path = SCRIPT_DIR / args.output.name
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to {output_path}")
        else:
            plt.show()

    elif args.mode == "interactive":
        run_interactive(args.root, args.participant, args.task, args.freq, args.rep)

if __name__ == "__main__":
    main()