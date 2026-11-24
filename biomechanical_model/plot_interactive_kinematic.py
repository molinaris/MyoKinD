#!/usr/bin/env python3

"""Interactively visualize one trial's hand kinematics and angle-estimation
geometry for all five fingers.

The figure is organized in three rows. The top row shows the complete hand
in 3D: all recorded marker positions are displayed together with the finger
chains, formed by vectors A and B, giving an overview of the hand
configuration at each time point.

The middle row shows one 3D panel per finger. Each panel displays the
finger's marker chain, the projection plane perpendicular to C rendered as
a surface, and the three vectors involved in the angle estimation: A_perp
and B_perp, the projections of vectors A and B onto that plane, and C, the
normal axis that defines the plane. This view shows how the angle-estimation
geometry is constructed directly from the recorded marker trajectories.

The bottom row shows one 2D panel per finger. Each panel expresses A_perp
and B_perp in the local coordinate system of the projection plane, and
displays the signed angle theta computed between them. This is the geometry
from which the kinematic angle is calculated at each time step.

A time slider and playback controls allow all three rows to be inspected
throughout the trial. The visualization is generated from the marker
trajectories stored in the HandKinematics/Trajectories files using the
angle-estimation procedure implemented in
biomechanical_model/estimate_kinematic_angles.py.

Usage:
    python plot_interactive_kinematic.py --root /path/to/Dataset --participant Sub001 --task 7 --freq 05 --rep 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.load_hdsemg import trial_filename  # noqa: E402
from examples.load_kinematics import KIN_FS, load_trajectories_csv  # noqa: E402


FINGERS = ["thumb", "index", "middle", "ring", "little"]

FINGER_COLORS = {
    "thumb":  "tab:blue",
    "index":  "tab:orange",
    "middle": "tab:green",
    "ring":   "tab:red",
    "little": "tab:purple"}

FINGER_CHAINS = {
    "thumb":  ["CW", "MW", "PT", "DT"],
    "index":  ["CW", "PI", "DI"],
    "middle": ["CW", "PM", "DM"],
    "ring":   ["CW", "PR", "DR"],
    "little": ["CW", "PL", "DL"]}

PROJECTION_MARKER = {
    "thumb":  "MW",
    "index":  "PI",
    "middle": "PM",
    "ring":   "PR",
    "little": "PL"}

VECTOR_STYLE = {
    "A": ("black", r"$A_{\perp}$", 1, 0.3),
    "B": ("red", r"$B_{\perp}$", 1, 0.3),
    "C": ("darkblue", r"$C$", 1, 0.3)}


def _rowwise_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the dot product between corresponding rows of two arrays."""
    return np.sum(a * b, axis=1)


def plot_interactive_kinematic(root: Path, participant: str, task: str, freq: str, rep: int, fs: float) -> Figure:
    """Interactively visualize hand geometry and angle estimation."""

    from biomechanical_model.estimate_kinematic_angles import estimate_angles  # noqa: E402

    name = trial_filename(participant, task, freq, rep)

    trajectories_path = root / participant / "HandKinematics" / "Trajectories" / name
    trajectories = load_trajectories_csv(trajectories_path)
    estimated, extra = estimate_angles(trajectories, participant)

    markers = extra["geometry"]["markers"]
    details = extra["details"]
    marker_names = list(markers)
    t = (estimated["frame"].to_numpy() - 1) / fs

    # Pre-compute display geometry
    relative = {marker: markers[marker] - markers["CW"] for marker in marker_names}
    relative["CW"] = np.zeros_like(markers["CW"])
    hand_points = np.stack([relative[marker] for marker in marker_names], axis=1)
    visual = {}

    for finger in FINGERS:
        d = details[finger]

        norm_A = np.linalg.norm(d["A_perp"], axis=1)
        norm_B = np.linalg.norm(d["B_perp"], axis=1)
        norm_C = np.linalg.norm(d["C"], axis=1)

        A_unit = d["A_perp"] / norm_A[:, None]
        B_unit = d["B_perp"] / norm_B[:, None]
        C_unit = d["C"] / norm_C[:, None]
        e2 = np.cross(C_unit, A_unit)

        points = np.stack(
            [relative[marker] for marker in FINGER_CHAINS[finger]], axis=1)
        origin = relative[PROJECTION_MARKER[finger]]

        visual[finger] = {"points": points, "origin": origin, "A_unit": A_unit,
                          "B_unit": B_unit, "C_unit": C_unit, "e1": A_unit, "e2": e2,
                          "A_2D": np.column_stack([norm_A, np.zeros_like(norm_A)]),
                          "B_2D": np.column_stack([_rowwise_dot(d["B_perp"], A_unit), _rowwise_dot(d["B_perp"], e2)]),
                          "size": np.linalg.norm(points - origin[:, None, :], axis=2).max(axis=1),
                          "limit": 1.20 * max(norm_A.max(), norm_B.max())}

    # Figure layout
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 5, height_ratios=[2.5, 1.5, 1.1])
    ax_general = fig.add_subplot(gs[0, :], projection="3d")
    vector_axes = [fig.add_subplot(gs[1, i], projection="3d") for i in range(5)]
    angle_axes = [fig.add_subplot(gs[2, i]) for i in range(5)]
    fig.subplots_adjust(left=0.05, right=0.98, top=0.97, bottom=0.10, hspace=-0.15, wspace=0.18)

    for ax in angle_axes:
        pos = ax.get_position()
        scale = 0.75
        width = pos.width * scale
        height = pos.height * scale

        ax.set_position([pos.x0 + (pos.width - width) / 2, pos.y0 + (pos.height - height) / 2, width, height])

    fig.text(0.015, 0.195, "Projection onto\n" r"plane $\perp C$", rotation=90, va="center", ha="center", fontsize=10)
    slider = Slider(fig.add_axes([0.18, 0.025, 0.40, 0.025]),
                    "Time (s)", t[0], t[-1], valinit=t[0], valstep=1 / fs, valfmt="%.2f s")
    start_button = Button(fig.add_axes([0.72, 0.015, 0.08, 0.045]), "Start")
    stop_button = Button(fig.add_axes([0.82, 0.015, 0.08, 0.045]), "Stop")

    # Plot helpers
    def clean_3d(ax: Axes) -> None:
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        ax.xaxis.labelpad = -15
        ax.yaxis.labelpad = -15
        ax.zaxis.labelpad = -15
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.xaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))
        ax.yaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))
        ax.zaxis.pane.set_facecolor((0.99, 0.99, 0.99, 1.0))

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1.0)
            axis._axinfo["grid"]["linewidth"] = 0.5

    def draw_vector(ax: Axes, origin: np.ndarray, vector: np.ndarray, style: tuple[str, str, float, float]) -> None:
        color, label, linewidth, ratio = style
        ax.quiver(*origin, *vector, color=color, linewidth=linewidth, arrow_length_ratio=ratio, label=label)

    def draw(time_value: float) -> None:
        idx = min(int(round(time_value * fs)), len(t) - 1)

        ax_general.clear()
        points = hand_points[idx]
        ax_general.scatter(*points.T, color="black", s=22)

        for marker, point in zip(marker_names, points):
            ax_general.text(*(point + 2), marker, fontsize=8)

        ax_general.scatter(0, 0, 0, color="black", marker="x", s=40)
        ax_general.text(2, 2, 2, "CW", fontsize=8)

        for finger in FINGERS:
            p = visual[finger]["points"][idx]
            ax_general.plot(*p.T, color=FINGER_COLORS[finger], linewidth=3)

        xy = 1.15 * np.abs(points[:, :2]).max()
        z_min = points[:, 2].min()
        z_max = points[:, 2].max()
        z_margin = 0.15 * (z_max - z_min)
        ax_general.set_xlim(-xy, xy)
        ax_general.set_ylim(-xy, xy)
        ax_general.set_zlim(z_min - z_margin, z_max + z_margin)

        clean_3d(ax_general)
        ax_general.set_title("Hand marker geometry", fontsize=11)
        ax_general.legend(
            handles=[Line2D([0], [0], color=FINGER_COLORS[finger],
                            linewidth=3, label=finger.capitalize()) for finger in FINGERS],
            frameon=False, ncols=5, fontsize=8, loc="upper center")


        # Individual 3D views
        for i, (ax, finger) in enumerate(zip(vector_axes, FINGERS)):
            ax.clear()
            v = visual[finger]
            points = v["points"][idx]
            origin = v["origin"][idx]
            length = 0.45 * v["size"][idx]
            A = v["A_unit"][idx] * length
            B = v["B_unit"][idx] * length
            C = v["C_unit"][idx] * 0.75 * length

            ax.plot(*points.T, color=FINGER_COLORS[finger], linewidth=1.5)
            ax.scatter(*points.T, color="black", s=7)
            ax.scatter(*origin, color="black", s=15)
            plane_size = 0.45 * length
            u, w = np.meshgrid([-plane_size, plane_size], [-plane_size, plane_size])
            e1 = v["e1"][idx]
            e2 = v["e2"][idx]
            plane = (origin[:, None, None] + e1[:, None, None] * u + e2[:, None, None] * w)

            ax.plot_surface(*plane, color="lightsteelblue",
                            alpha=0.55, shade=False, edgecolor="gray", linewidth=0.6)
            draw_vector(ax, origin, A, VECTOR_STYLE["A"])
            draw_vector(ax, origin, B, VECTOR_STYLE["B"])
            draw_vector(ax, origin, C, VECTOR_STYLE["C"])

            lower = np.minimum(points.min(axis=0), origin - length)
            upper = np.maximum(points.max(axis=0), origin + length)
            margin = 0.15 * (upper - lower)
            ax.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
            ax.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
            ax.set_zlim(lower[2] - margin[2], upper[2] + margin[2])
            ax.set_box_aspect((1, 1, 0.8))
            ax.view_init(elev=20, azim=-60)
            clean_3d(ax)
            ax.set_title(finger.capitalize(), fontsize=10, y=0.92)

            if i == 0:
                ax.legend(
                    handles=[
                        Line2D([0], [0], color="black", lw=3, label=r"$A_{\perp}$"),
                        Line2D([0], [0], color="red", lw=3, label=r"$B_{\perp}$"),
                        Line2D([0], [0], color="darkblue", lw=3, label=r"$C$")],
                    frameon=True, facecolor="white", framealpha=1, edgecolor="none", fontsize=9,
                    loc="center right", bbox_to_anchor=(0.05, 0.50))


        # 2D projections
        for ax, finger in zip(angle_axes, FINGERS):
            ax.clear()
            v = visual[finger]
            A = v["A_2D"][idx]
            B = v["B_2D"][idx]
            theta = details[finger]["angle_deg"][idx]
            for vector, color, label in ((A, "black", r"$A_{\perp}$"),
                                         (B, "red", r"$B_{\perp}$")):
                ax.quiver(0, 0, *vector, color=color, angles="xy", scale_units="xy", scale=1, width=0.012)
                ax.text(*(1.04 * vector), label, color=color, fontsize=9, fontweight="bold")
            radius = 0.40 * min(np.linalg.norm(A), np.linalg.norm(B))

            arc = np.linspace(0, np.radians(theta), 80)
            ax.plot(radius * np.cos(arc), radius * np.sin(arc), color="darkblue", linewidth=2.5)
            limit = v["limit"]
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
            ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
            ax.set_xlabel(r"$X'$")
            ax.set_ylabel(r"$Y'$")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect("equal", adjustable="box")
            ax.spines[["right", "top"]].set_visible(False)
            ax.text(0.02, 0.3, rf"$\theta = {theta:.1f}^\circ$", transform=ax.transAxes, fontsize=8, color="darkblue", va="bottom", ha="left")

        fig.canvas.draw_idle()


    # Playback
    slider_callback = slider.on_changed(draw)
    timer = fig.canvas.new_timer(interval=10)

    def advance() -> None:
        next_time = slider.val + 20 / fs
        slider.set_val(t[0] if next_time > t[-1] else next_time)

    timer.add_callback(advance)
    start_callback = start_button.on_clicked(lambda _: timer.start())
    stop_callback = stop_button.on_clicked(lambda _: timer.stop())

    # Keep widget references alive while the figure exists.
    fig._slider = slider
    fig._slider_callback = slider_callback
    fig._start_button = start_button
    fig._start_callback = start_callback
    fig._stop_button = stop_button
    fig._stop_callback = stop_callback
    fig._timer = timer

    draw(t[0])

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participant", required=True, help="e.g. Sub001")
    parser.add_argument("--task", required=True, help="1-8")
    parser.add_argument("--freq", required=True, help="'05' or '075'")
    parser.add_argument("--rep", required=True, type=int, help="0, 1, or 2")
    args = parser.parse_args()

    print("Starting interactive plot...")

    plot_interactive_kinematic(args.root, args.participant, args.task, args.freq, args.rep, fs=KIN_FS)

    plt.show()


if __name__ == "__main__":
    main()