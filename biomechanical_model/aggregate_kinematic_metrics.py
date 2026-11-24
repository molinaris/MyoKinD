#!/usr/bin/env python3

"""Aggregate kinematic metrics across participants, tasks, frequencies, and
repetitions, and produce descriptive statistics, frequency-mismatch summaries,
and population-level distribution plots.

Processes every requested trial (up to 8 tasks × 2 frequencies × 3
repetitions) for each participant. Range of movement (RoM) and mean angular
speed are computed using the angle-estimation functions in
``biomechanical_model/estimate_kinematic_angles.py``; THDN (normalized total
harmonic distortion) and the detected fundamental frequency are computed with
``compute_thdn`` from ``validation/kinematic_quality.py``, using the
biomechanical-model parameters defined in ``estimate_kinematic_angles.py``
(5-40 s trim, 8 s Hamming window).

Results are saved at two levels of aggregation. The trial-level CSV stores one
row per participant × task × frequency × repetition × finger, with columns
participant, task, frequency, repetition, finger, rom_deg, mean_speed_deg_s,
thdn_percent, target_frequency_hz, detected_frequency_hz, and frequency_match.
The participant-level CSV averages the three repetitions within each
participant × task × frequency × finger condition, yielding one value per
participant per condition; this file is read by ``statistical_analysis.R``.

Descriptive statistics (mean ± SD across participants) are printed to the
terminal for each metric and condition. Trials whose detected fundamental
frequency does not match the target metronome frequency are flagged and
summarised in a separate mismatch report.

Population-level distribution plots are generated for RoM, mean angular speed,
and THDN for each finger across tasks and frequencies. Statistical annotations
marking significant Task × Frequency interactions are read from
``biomechanical_model/kinematic_significant_interactions.csv``, which is
produced by ``statistical_analysis.R``. That script fits an aligned rank
transform (ART) mixed-effects model for each metric and finger and applies
Bonferroni-corrected post-hoc comparisons. Annotations are included only when
the participant set in the CSV matches the current run.

Usage:
    python aggregate_kinematic_metrics.py --root /path/to/Dataset --participants Sub001 Sub002
    python aggregate_kinematic_metrics.py --root /path/to/Dataset --participants all \
        --output-csv kinematic_metrics_trials.csv \
        --output-csv-averaged kinematic_metrics_participants.csv \
        --output population_kinematic_metrics.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biomechanical_model.estimate_kinematic_angles import (FINGERS, FREQ_CODE_TO_HZ,
                                                           THDN_N_BINS, THDN_N_HARMONICS,
                                                           THDN_WINDOW, THDN_WINDOW_S, TRIM_END_S,
                                                           TRIM_START_S, calculate_mean_speed, calculate_rom,
                                                           get_cycle_segments, mean_sd) # noqa: E402
from examples.load_hdsemg import trial_filename  # noqa: E402
from examples.load_kinematics import KIN_FS, load_angles_csv  # noqa: E402
from validation.hdsemg_quality import FREQUENCIES, REPETITIONS, TASKS  # noqa: E402
from validation.kinematic_quality import compute_thdn, trim_signal  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SIGNIFICANCE_CSV = SCRIPT_DIR / "kinematic_significant_interactions.csv"
ALL_PARTICIPANTS = [f"Sub{n:03d}" for n in range(1, 22)]

TASK_LABELS = {"1": "Index", "2": "Middle", "3": "Ring-Little", "4": "Thumb",
               "5": "I. Pinch", "6": "M. Pinch", "7": "Tripod", "8": "5-Finger"}
TASK_FINGERS = {"1": ["index"], "2": ["middle"], "3": ["ring", "little"], "4": ["thumb"], "5": ["thumb", "index"],
                "6": ["thumb", "middle"], "7": ["thumb", "index", "middle"], "8": FINGERS}
FREQUENCY_LABELS = {"05": "0.50 Hz", "075": "0.75 Hz"}

PLOT_METRICS = ["rom_deg", "mean_speed_deg_s", "thdn_percent"]
PLOT_METRIC_LABELS = ["RoM (deg)", "Mean Speed (deg/s)", r"$\mathrm{THD}_{\mathrm{N}}$ (%)"]

PLOT_COLORS = {"05": "#FF8000", "075": "#0073E6"}

PROGRESS_INTERVAL_S = 0.2
_PROGRESS_LINE_WIDTH = 110


def _progress(done: int, total: int, start: float, label: str) -> None:
    fraction = done / total
    elapsed = time.monotonic() - start
    eta = elapsed / done * (total - done) if done else 0
    filled = int(30 * fraction)
    bar = "#" * filled + "-" * (30 - filled)
    line = (f"[{bar}] {done}/{total} ({fraction * 100:5.1f}%) "
            f"elapsed={elapsed:6.0f}s eta={eta:6.0f}s  {label}").ljust(_PROGRESS_LINE_WIDTH)
    print("\r" + line, end="\n" if done == total else "", flush=True)


def process_trial(root: Path, participant: str, task: str, freq: str, rep: int) -> pd.DataFrame:
    """Compute kinematic metrics for one trial."""

    path = root / participant / "HandKinematics" / "Angles" / trial_filename(participant, task, freq, rep)
    reference = load_angles_csv(path)
    trimmed = pd.DataFrame(trim_signal(reference[FINGERS].to_numpy(float),
                                       KIN_FS, TRIM_START_S, TRIM_END_S), columns=FINGERS)
    cycles = get_cycle_segments(trimmed, task=task, f_task=FREQ_CODE_TO_HZ[freq], fs=KIN_FS)

    if not cycles:
        raise ValueError("No valid movement cycles were found.")

    rom = calculate_rom(cycles).set_index("finger")
    speed = calculate_mean_speed(cycles, KIN_FS).set_index("finger")
    signals = {finger: trimmed[finger].to_numpy(float) for finger in FINGERS}
    thdn, details = compute_thdn(signals, fs=KIN_FS, f_task=FREQ_CODE_TO_HZ[freq],
                                 window_s=THDN_WINDOW_S, n_harmonics=THDN_N_HARMONICS,
                                 n_bins=THDN_N_BINS, window=THDN_WINDOW, return_details=True)
    target = FREQ_CODE_TO_HZ[freq]

    return pd.DataFrame([{"participant": participant, "task": task, "frequency": freq, "repetition": rep,
                          "finger": finger, "rom_deg": rom.loc[finger, "RoM"],
                          "mean_speed_deg_s": speed.loc[finger, "mean_speed"],
                          "thdn_percent": thdn[finger], "target_frequency_hz": target,
                          "detected_frequency_hz": details[finger]["detected_frequency"],
                          "frequency_match": details[finger]["detected_frequency"] == target} for finger in FINGERS])


def collect(root: Path, participants: list[str]) -> pd.DataFrame:
    """Collect metrics for all requested trials."""

    trial_combinations = [(participant, task, freq, rep) for participant in participants
                          for task in TASKS for freq in FREQUENCIES for rep in REPETITIONS]
    frames = []
    start = last_print = time.monotonic()

    for i, (participant, task, freq, rep) in enumerate(trial_combinations, 1):
        try:
            frames.append(process_trial(root, participant, task, freq, rep))
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n  skipping {participant} task={task} freq={freq} rep={rep}: {exc}")
        now = time.monotonic()
        if i == 1 or now - last_print >= PROGRESS_INTERVAL_S or i == len(trial_combinations):
            _progress(i, len(trial_combinations), start, f"{participant} task={task} freq={freq} rep={rep}")
            last_print = now

    if not frames:
        return pd.DataFrame()
    metrics = pd.concat(frames, ignore_index=True)
    metrics["finger"] = pd.Categorical(metrics["finger"], FINGERS, ordered=True)

    return metrics.sort_values(["participant", "task", "frequency", "repetition", "finger"]).reset_index(drop=True)


def build_averaged_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average the three repetitions within participant."""
    groups = ["participant", "task", "frequency", "finger"]
    averaged = metrics.groupby(groups, as_index=False, observed=True)[PLOT_METRICS].mean().sort_values(groups).reset_index(drop=True)
    averaged["finger"] = pd.Categorical(averaged["finger"], FINGERS, ordered=True)

    return averaged


def build_descriptive_table(averaged: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean and standard deviation across participants."""
    rows = []
    for (task, freq, finger), group in averaged.groupby(["task", "frequency", "finger"], observed=True):
        row = {"task": str(task), "frequency": str(freq), "finger": str(finger)}
        for metric in PLOT_METRICS:
            mean, sd = mean_sd(group[metric].to_numpy(float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
        rows.append(row)

    return pd.DataFrame(rows)


def _format_mean_sd(mean: float, sd: float) -> str:
    return "NA" if np.isnan(mean) else f"{mean:.2f} ± {'NA' if np.isnan(sd) else f'{sd:.2f}'}"


def _build_paper_table(descriptive: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Build a task × finger × frequency descriptive table."""
    rows = []
    for task in map(str, TASKS):
        row = {}

        for finger in FINGERS:
            for freq in map(str, FREQUENCIES):
                subset = descriptive[(descriptive["task"] == task) &
                                     (descriptive["frequency"] == freq) & (descriptive["finger"] == finger)]
                value = "NA"
                if not subset.empty:
                    result = subset.iloc[0]
                    value = _format_mean_sd(result[f"{metric}_mean"], result[f"{metric}_sd"])
                row[finger.capitalize(), FREQUENCY_LABELS[freq]] = value
        rows.append(row)
    table = pd.DataFrame(rows, index=[TASK_LABELS[str(task)] for task in TASKS])
    table.index.name = "Task"

    return table


def print_descriptive_tables(descriptive: pd.DataFrame) -> None:
    """Print descriptive statistics."""

    print("\nDescriptive data (mean ± SD).\n"
          "Values are calculated after averaging the three repetitions within each participant.")

    for title, metric in zip(PLOT_METRIC_LABELS, PLOT_METRICS):
        title = title.replace(r"$\mathrm{THD}_{\mathrm{N}}$", "THDN")
        print(f"\n{title} — mean ± SD")
        print("=" * 150)
        print(_build_paper_table(descriptive, metric).to_string())


def print_frequency_mismatch_summary(metrics: pd.DataFrame) -> None:
    """Print fundamental-frequency mismatch summaries."""

    bold, reset = "\033[1m", "\033[0m"
    data = metrics.copy()
    data["task"] = data["task"].astype(str)
    data["frequency"] = data["frequency"].astype(str)
    data["finger"] = data["finger"].astype(str)
    mismatches = data[~data["frequency_match"]]

    print("\nFundamental Frequency Mismatches\n"
          "Mean ± SD (N) of detected fundamental frequencies for fingers that did not match the target frequency.\n"
          "Task-relevant fingers are shown in bold.")

    if mismatches.empty:
        print("\nAll detected fundamental frequencies matched the target frequencies.")
        return

    # Summary tables: all fingers
    for freq in map(str, FREQUENCIES):
        print(f"\nTarget frequency: {FREQUENCY_LABELS[freq]}")
        print("=" * 110)
        headers = ["Task"] + [finger.capitalize() for finger in FINGERS]
        rows = []

        for task in map(str, TASKS):
            row = [TASK_LABELS[task]]

            for finger in FINGERS:
                values = mismatches.loc[(mismatches["frequency"] == freq) & (mismatches["task"] == task)
                                        & (mismatches["finger"] == finger), "detected_frequency_hz"].to_numpy(float)

                if len(values):
                    mean, sd = mean_sd(values)
                    row.append(f"{mean:.2f} ± {sd:.2f} ({len(values)})")
                else:
                    row.append("-")

            rows.append(row)

        widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)))

        for task, row in zip(map(str, TASKS), rows):
            cells = []

            for i, value in enumerate(row):
                cell = value.ljust(widths[i])
                if i and FINGERS[i - 1] in TASK_FINGERS[task] and value != "-":
                    cell = f"{bold}{cell}{reset}"
                cells.append(cell)
            print("  ".join(cells))

    # Exact problematic trials: task-relevant fingers only
    relevant = data[[finger in TASK_FINGERS[task] for task, finger in zip(data["task"], data["finger"])]]
    rows = []
    for (participant, task, freq, rep), group in relevant.groupby(["participant", "task", "frequency", "repetition"]):
        failed = group[~group["frequency_match"]]
        if failed.empty:
            continue
        rows.append({"Participant": participant, "Task": TASK_LABELS[task], "Target": FREQUENCY_LABELS[freq],
                     "Rep": rep, "Finger(s)": ", ".join(finger.capitalize() for finger in failed["finger"]),
                     "Detected Hz": ", ".join(f"{value:.3f}" for value in failed["detected_frequency_hz"])})
    print("\nTrials with Fundamental-Frequency Mismatch")
    print("=" * 110)

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("All task-relevant fingers matched the target frequencies.")


def _load_significance_results(participants: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Load R significance results when participant sets match."""
    columns = ["Metric", "Finger", "Task_or_Factor", "p_bonf"]
    empty = pd.DataFrame(columns=columns)
    if not SIGNIFICANCE_CSV.exists():
        return empty, f"{SIGNIFICANCE_CSV.name} was not found."

    significance = pd.read_csv(SIGNIFICANCE_CSV)
    required = set(columns + ["Participants"])
    if not required.issubset(significance.columns):
        return empty, "The R results do not contain the required participant metadata."
    participant_sets = significance["Participants"].dropna().unique()
    if len(participant_sets) != 1:
        return empty, "Participant metadata in the R results is missing or inconsistent."
    stats_participants = set(participant_sets[0].split("|"))
    current_participants = set(participants)

    if stats_participants != current_participants:
        return empty, ("Participants used in R do not match this Python run.\n"
                       f"  Python: {', '.join(sorted(current_participants))}\n"
                       f"  R:      {', '.join(sorted(stats_participants))}")

    return significance, None


def plot_population_kinematic_metrics(averaged: pd.DataFrame, participants: list[str]) -> tuple[Figure, str | None]:
    """Plot population RoM, mean speed, and THDN."""
    significance, warning = _load_significance_results(participants)
    data = averaged.copy()
    for column in ["task", "frequency", "finger"]:
        data[column] = data[column].astype(str)
    tasks = list(map(str, TASKS))
    freqs = list(map(str, FREQUENCIES))
    x = np.arange(len(tasks))
    width = 0.3

    def values(metric: str, freq: str, task: str, finger: str) -> list[float]:
        return data.loc[(data["frequency"] == freq) & (data["task"] == task)
                        & (data["finger"] == finger), metric].dropna().tolist()

    ymax = {metric: data[metric].max() for metric in PLOT_METRICS}
    fig, axes = plt.subplots(len(FINGERS), len(PLOT_METRICS), figsize=(12, 10), sharex=True)
    plt.subplots_adjust(wspace=0.12, hspace=0.15)
    letters = [f"{chr(65 + i)})" for i in range(len(FINGERS) * len(PLOT_METRICS))]

    for row, finger in enumerate(FINGERS):
        for col, metric in enumerate(PLOT_METRICS):
            ax = axes[row, col]
            axis_max = ymax[metric] * 1.45
            ax.set_ylim(0, axis_max)
            ax.set_yticks(np.linspace(0, ymax[metric], 5).astype(int))

            for i, freq in enumerate(freqs):
                box_values = [values(metric, freq, task, finger) for task in tasks]
                positions = x + (i - 0.5) * width

                ax.boxplot(box_values, positions=positions, widths=width, patch_artist=True,
                           boxprops={"facecolor": PLOT_COLORS[freq], "color": PLOT_COLORS[freq], "alpha": 0.6},
                           medianprops={"color": "black"}, whiskerprops={"color": "black"}, capprops={"color": "black"},
                           flierprops={"markerfacecolor": "gray", "markersize": 1.5, "linestyle": "none", "marker": "o", "alpha": 0.3}, manage_ticks=False)
                ax.plot(positions, [np.mean(v) if v else np.nan for v in box_values], "ko", markersize=2)
            sig = significance[(significance["Metric"] == metric) & (significance["Finger"] == finger)]

            for _, result in sig.iterrows():
                task_value = str(result["Task_or_Factor"])
                if not task_value.startswith("Task="):
                    continue
                task = task_value.split("=")[1]
                if task not in tasks:
                    continue
                idx = tasks.index(task)
                pos_05 = x[idx] - width / 2
                pos_075 = x[idx] + width / 2
                task_max = max(values(metric, "05", task, finger) + values(metric, "075", task, finger) or [0])
                y_end = task_max + 0.05 * axis_max
                y_start = y_end - 0.01 * axis_max
                p = float(result["p_bonf"])
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                if stars:
                    ax.plot([pos_05, pos_05, pos_075, pos_075], [y_start, y_end, y_end, y_start], color="black", lw=1)
                    ax.text((pos_05 + pos_075) / 2, task_max + 0.015 * axis_max, stars, ha="center", va="bottom", fontsize=10)
            if row == 0:
                ax.set_title(PLOT_METRIC_LABELS[col], fontsize=15, pad=10)
            if col == 0:
                ax.set_ylabel(finger.capitalize(), fontsize=12, rotation=90, labelpad=8, va="center")
            if row == len(FINGERS) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels([TASK_LABELS[task] for task in tasks], rotation=35, ha="right", fontsize=10)
            else:
                ax.set_xticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="y", labelsize=10)
            ax.tick_params(axis="x", labelsize=11)
            letter_idx = col * len(FINGERS) + row
            ax.text(0.02, 0.96, letters[letter_idx], transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    handles = [plt.Line2D([0], [0], color=PLOT_COLORS[freq], lw=5, alpha=0.7) for freq in freqs]
    axes[0, 0].legend(handles, ["0.50 Hz", "0.75 Hz"], loc="upper left", bbox_to_anchor=(0.05, 0.8), fontsize=11, frameon=True, handlelength=1)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, warning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Dataset root (contains SubXXX/)")
    parser.add_argument("--participants", nargs="+", required=True, help="Participant IDs, or 'all'")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Trial-level metrics CSV file name (saved in biomechanical_model/)")
    parser.add_argument("--output-csv-averaged", type=Path, default=None,
                        help="Participant-level averaged metrics CSV file name (saved in biomechanical_model/, "
                             "where statistical_analysis.R reads it)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Population figure file name (saved in biomechanical_model/)")
    args = parser.parse_args()

    participants = ALL_PARTICIPANTS if len(args.participants) == 1 and args.participants[0].lower() == "all" else args.participants
    metrics = collect(args.root, participants)
    if metrics.empty:
        print("\nNo valid kinematic trials were found.")
        return

    averaged = build_averaged_table(metrics)
    descriptive = build_descriptive_table(averaged)

    print(f"\nAggregated over {len(participants)} participant(s): {len(metrics)} finger-trial rows, "
          f"{len(averaged)} averaged participant-level rows")

    outputs = [(args.output_csv, metrics, "Trial-level kinematic metrics", "participant × task × frequency × repetition × finger"),
               (args.output_csv_averaged, averaged, "Averaged participant-level kinematic metrics", "participant × task × frequency × finger")]

    for filename, table, label, structure in outputs:
        if filename is None:
            continue
        path = SCRIPT_DIR / filename.name
        table.to_csv(path, index=False)
        print(f"\n{label} written to: \n  {path} \n  {structure}")

    print_descriptive_tables(descriptive)
    print_frequency_mismatch_summary(metrics)
    print("\nGenerating population kinematic metrics figure..."
          f"\nSignificance annotations: {SIGNIFICANCE_CSV}")

    fig, warning = plot_population_kinematic_metrics(averaged, participants)
    if warning:
        print(f"\nWARNING: {warning}\n"
              "The population figure was generated WITHOUT statistical significance annotations.\n"
              "Run the R statistical analysis using the same participant set to update kinematic_significant_interactions.csv.")
    if args.output is not None:
        output_path = SCRIPT_DIR / args.output.name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nFigure saved to:\n  {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()