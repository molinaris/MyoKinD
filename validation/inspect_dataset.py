#!/usr/bin/env python3
"""Read-only structural inspection of a MyoKinD dataset copy.

Walks a ``Dataset/`` folder (the one directly containing ``Sub001/`` ...
``Sub021/``), classifies every CSV file by modality (HD sEMG, joint angles,
marker trajectories), extracts participant/task/frequency/repetition from the
filename, checks basic structural integrity (row/column counts, numeric
content, expected file pairs across modalities) and prints a summary. A full
manifest can optionally be written to CSV.

This script never modifies the dataset and only uses the standard library
plus pandas (to build and summarize the manifest, and for its optional CSV
export).

Usage:
    python inspect_dataset.py --root /path/to/Dataset [--manifest-csv out.csv] [--quiet]
"""
from __future__ import annotations

import argparse
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

# --- Known, fixed properties of the dataset, confirmed directly against real sample
# files (see also README.md) ---
HDSEMG_FS = 2052.5245535714284      # Hz; not stored in the CSV itself
HDSEMG_N_ROWS = 92160
HDSEMG_N_COLUMNS = 128

KIN_HEADER_LINES = 5                # Vicon metadata lines before the data starts
KIN_N_ROWS = 4500
ANGLES_N_COLUMNS = 17
TRAJECTORIES_N_COLUMNS = 38

PARTICIPANTS = [f"Sub{n:03d}" for n in range(1, 22)]   # Sub001 .. Sub021
TASKS = [str(t) for t in range(1, 9)]                   # '1' .. '8'
FREQUENCIES = ["05", "075"]                             # literal strings, not decimals
REPETITIONS = [0, 1, 2]

# Documented dataset exceptions (see README.md): a (participant,
# modality, task, frequency, repetition) key expected to be ABSENT. Anything
# else missing is reported as an anomaly instead of an expected exclusion.
KNOWN_EXCEPTIONS = {
    ("Sub017", "hdsemg", "3", "05", 1),
    ("Sub017", "angles", "3", "05", 1),
    ("Sub017", "trajectories", "3", "05", 1),
    ("Sub017", "angles", "start_position", None, 0),
    ("Sub017", "trajectories", "start_position", None, 0)
}

TASK_RE = re.compile(r"^(Sub\d{3})_([1-8])_(05|075)_450_([0-2])\.csv$")
REST_RE = re.compile(r"^(Sub\d{3})_rest_([0-2])\.csv$")
START_RE = re.compile(r"^(Sub\d{3})_start_position_([0-2])\.csv$")


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def parse_filename(name: str) -> dict | None:
    """Extract participant/task/frequency/repetition from a dataset filename.

    Returns None if the name matches none of the two expected patterns
    (task trial, or the rest/start_position special conditions).
    """
    m = TASK_RE.match(name)
    if m:
        participant, task, frequency, repetition = m.groups()
        return {"participant": participant, "task": task, "frequency": frequency,
                "repetition": int(repetition)}
    m = REST_RE.match(name)
    if m:
        participant, repetition = m.groups()
        return {"participant": participant, "task": "rest", "frequency": None,
                "repetition": int(repetition)}
    m = START_RE.match(name)
    if m:
        participant, repetition = m.groups()
        return {"participant": participant, "task": "start_position", "frequency": None,
                "repetition": int(repetition)}
    return None


def classify_file(path: Path, root: Path) -> str:
    """Classify a CSV file by modality, based on its parent directory names."""
    parts = path.relative_to(root).parts
    if len(parts) >= 3 and parts[-2] == "HD_sEMG":
        return "hdsemg"
    if len(parts) >= 4 and parts[-3] == "HandKinematics" and parts[-2] == "Angles":
        return "angles"
    if len(parts) >= 4 and parts[-3] == "HandKinematics" and parts[-2] == "Trajectories":
        return "trajectories"
    return "unknown"


def iter_csv_files(root: Path) -> Iterator[Path]:
    """Yield every real CSV file under root, skipping macOS zip artifacts."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__MACOSX"]
        for filename in filenames:
            if filename.startswith(".") or not filename.lower().endswith(".csv"):
                continue
            yield Path(dirpath) / filename


def _count_lines_fast(path: Path, chunk_size: int = 1 << 20) -> int:
    """Count lines by scanning raw bytes in chunks (fast, O(1) memory).

    Counts '\\n' occurrences and additionally counts a final line that has no
    trailing newline (defensive: a truncated/corrupted file may lack one).
    """
    count = 0
    last_byte = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        count += 1
    return count


def probe_hdsemg(path: Path) -> dict:
    """Read only what is needed to check an HD sEMG file's shape/content."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline().rstrip("\r\n")

    if first_line == "":
        return {"n_rows": 0, "n_columns": 0, "sampling_frequency_hz": HDSEMG_FS,
                "duration_s": 0.0, "status": "EMPTY", "notes": "file has no content"}

    fields = first_line.split(",")
    n_columns = len(fields)
    non_numeric = not all(_is_float(x) for x in fields)
    n_rows = _count_lines_fast(path)

    status = "OK"
    notes = []
    if n_columns != HDSEMG_N_COLUMNS:
        status = "MISSING_COLUMNS"
        notes.append(f"expected {HDSEMG_N_COLUMNS} columns, found {n_columns}")
    if non_numeric:
        status = "NON_NUMERIC"
        notes.append("first row contains non-numeric values")
    if n_rows != HDSEMG_N_ROWS:
        status = "ROW_COUNT_MISMATCH" if status == "OK" else status
        notes.append(f"expected {HDSEMG_N_ROWS} rows, found {n_rows}")

    return {
        "n_rows": n_rows, "n_columns": n_columns,
        "sampling_frequency_hz": HDSEMG_FS, "duration_s": n_rows / HDSEMG_FS,
        "status": status, "notes": "; ".join(notes)
    }


def probe_kinematics(path: Path, modality: str) -> dict:
    """Check an Angles/Trajectories file's shape/content.

    Unlike HD sEMG, these files are small (4500 data rows), so every data row
    is scanned - not just the first one. This lets us tell apart two very
    different situations: a MISSING value (blank cell, e.g. ``1,0,,,,0,0,...``)
    is an expected artifact of marker occlusion near a trial's boundaries
    (documented in README.md), whereas a genuine NON_NUMERIC token (anything
    that is neither blank nor a valid float) would indicate real corruption.
    """
    expected_columns = ANGLES_N_COLUMNS if modality == "angles" else TRAJECTORIES_N_COLUMNS

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header_lines = [f.readline() for _ in range(KIN_HEADER_LINES)]
        if any(line == "" for line in header_lines):
            return {"n_rows": 0, "n_columns": 0, "sampling_frequency_hz": None,
                    "duration_s": 0.0, "status": "EMPTY",
                    "notes": f"file shorter than the expected {KIN_HEADER_LINES}-line Vicon header"}
        fs_text = header_lines[1].strip()
        sampling_frequency_hz = float(fs_text) if _is_float(fs_text) else None

        n_rows = 0
        n_columns = None
        rows_with_missing_values = 0
        non_numeric_found = False
        blank_lines_skipped = 0
        for line in f:
            line = line.rstrip("\r\n")
            if line == "":
                blank_lines_skipped += 1
                continue
            fields = line.split(",")
            if n_columns is None:
                n_columns = len(fields)
            row_has_missing = False
            for value in fields:
                if value == "":
                    row_has_missing = True
                elif not _is_float(value):
                    non_numeric_found = True
            if row_has_missing:
                rows_with_missing_values += 1
            n_rows += 1

    if n_columns is None:
        return {"n_rows": 0, "n_columns": 0, "sampling_frequency_hz": sampling_frequency_hz,
                "duration_s": 0.0, "status": "EMPTY", "notes": "no data rows after header"}

    status = "OK"
    notes = []
    if n_columns != expected_columns:
        status = "MISSING_COLUMNS"
        notes.append(f"expected {expected_columns} columns, found {n_columns}")
    if n_rows != KIN_N_ROWS:
        status = "ROW_COUNT_MISMATCH" if status == "OK" else status
        notes.append(f"expected {KIN_N_ROWS} data rows, found {n_rows}")
    if non_numeric_found:
        status = "NON_NUMERIC" if status == "OK" else status
        notes.append("at least one non-blank, non-numeric value found")
    if rows_with_missing_values > 0:
        if status == "OK":
            status = "MISSING_VALUES"
        notes.append(f"{rows_with_missing_values} row(s) contain missing (blank) values "
                     "- likely marker occlusion near the trial boundaries, see README.md")
    if blank_lines_skipped:
        notes.append(f"{blank_lines_skipped} blank line(s) skipped")

    duration_s = n_rows / sampling_frequency_hz if sampling_frequency_hz else None
    return {
        "n_rows": n_rows, "n_columns": n_columns,
        "sampling_frequency_hz": sampling_frequency_hz, "duration_s": duration_s,
        "status": status, "notes": "; ".join(notes)
    }


# Minimum time between progress redraws (besides the once-per-participant checkpoint),
# so a fast loop doesn't spend all its time writing to the terminal.
PROGRESS_INTERVAL_S = 0.2

# Wide enough to fully overwrite any previous, longer progress line via "\r" alone.
_PROGRESS_LINE_WIDTH = 110


def _format_progress(done: int, total: int, start_time: float, label: str, bar_width: int = 30) -> str:
    fraction = done / total if total else 1.0
    filled = int(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.monotonic() - start_time
    eta = elapsed / done * (total - done) if done else 0.0
    line = (
        f"[{bar}] {done}/{total} ({fraction * 100:5.1f}%) "
        f"elapsed={elapsed:6.0f}s eta={eta:6.0f}s  {label}"
    )
    return line.ljust(_PROGRESS_LINE_WIDTH)


def _print_progress(done: int, total: int, start_time: float, label: str) -> None:
    """Redraw the progress bar in place on the current terminal line."""
    end = "\n" if done == total else ""
    print("\r" + _format_progress(done, total, start_time, label), end=end, flush=True)


def build_manifest(root: Path, quiet: bool = False) -> pd.DataFrame:
    """Walk the dataset root and build one manifest row per CSV file found."""
    rows = []
    files = sorted(iter_csv_files(root))
    total = len(files)
    start_time = time.monotonic()
    last_print_time = start_time
    previous_participant = None
    for i, path in enumerate(files, start=1):
        modality = classify_file(path, root)
        meta = parse_filename(path.name)

        row = {
            "file_path": str(path.relative_to(root)),
            "modality": modality,
            "participant": meta["participant"] if meta else None,
            "task": meta["task"] if meta else None,
            "frequency": meta["frequency"] if meta else None,
            "repetition": meta["repetition"] if meta else None,
            "n_rows": None, "n_columns": None,
            "sampling_frequency_hz": None, "duration_s": None,
            "status": None, "notes": ""
        }

        if meta is None:
            row["status"] = "NAME_MISMATCH"
            row["notes"] = "filename does not match any expected naming pattern"
        elif modality == "hdsemg":
            row.update(probe_hdsemg(path))
        elif modality in ("angles", "trajectories"):
            row.update(probe_kinematics(path, modality))
        else:
            row["status"] = "UNKNOWN_LOCATION"
            row["notes"] = "file is not under HD_sEMG or HandKinematics/{Angles,Trajectories}"

        rows.append(row)

        if not quiet:
            participant = meta["participant"] if meta else path.relative_to(root).parts[0]
            now = time.monotonic()
            is_new_participant = participant != previous_participant
            is_due = now - last_print_time >= PROGRESS_INTERVAL_S
            if is_new_participant or is_due or i == total:
                _print_progress(i, total, start_time, label=f"{participant}: {path.name}")
                last_print_time = now
            previous_participant = participant

    return pd.DataFrame(rows)


def _expected_keys_for_modality(modality: str) -> set[tuple[str, str | None, int]]:
    """The set of (task, frequency, repetition) keys expected for a modality."""
    keys = {(task, freq, rep) for task in TASKS for freq in FREQUENCIES for rep in REPETITIONS}
    keys |= {("start_position", None, rep) for rep in REPETITIONS}
    if modality == "hdsemg":
        keys |= {("rest", None, rep) for rep in REPETITIONS}
    return keys


def summarize(manifest: pd.DataFrame) -> None:
    """Print a human-readable summary of the manifest to the terminal."""
    print("\n=== File counts by modality ===")
    print(manifest["modality"].value_counts().to_string())

    print("\n=== Participants ===")
    found = sorted(set(manifest["participant"].dropna()))
    missing_participants = [p for p in PARTICIPANTS if p not in found]
    print(f"Found: {len(found)}/{len(PARTICIPANTS)}")
    if missing_participants:
        print(f"MISSING participants: {missing_participants}")

    print("\n=== Non-OK file status ===")
    bad = manifest[manifest["status"] != "OK"]
    if bad.empty:
        print("None - every recognized file matches the expected shape/content.")
    else:
        for _, r in bad.iterrows():
            print(f"  [{r['status']}] {r['file_path']} - {r['notes']}")

    print("\n=== Duplicate (participant, modality, task, frequency, repetition) keys ===")
    keyed = manifest.dropna(subset=["participant", "task"])
    dup_counts = keyed.groupby(["participant", "modality", "task", "frequency", "repetition"]).size()
    dups = dup_counts[dup_counts > 1]
    if dups.empty:
        print("None.")
    else:
        print(dups.to_string())

    print("\n=== Missing expected file combinations ===")
    n_anomalies = 0
    n_known = 0
    for participant in found:
        for modality in ("hdsemg", "angles", "trajectories"):
            present = set()
            subset = manifest[
                (manifest["participant"] == participant) & (manifest["modality"] == modality)]

            for row in subset.itertuples():
                frequency = None if pd.isna(row.frequency) else row.frequency
                present.add((row.task, frequency, row.repetition))
            for task, frequency, repetition in sorted(_expected_keys_for_modality(modality)):
                if (task, frequency, repetition) in present:
                    continue
                key = (participant, modality, task, frequency, repetition)
                if key in KNOWN_EXCEPTIONS:
                    print(f"  [INFO, documented exclusion] {participant}/{modality}: "
                          f"task={task} freq={frequency} rep={repetition}")
                    n_known += 1
                else:
                    print(f"  [WARNING] {participant}/{modality}: "
                          f"task={task} freq={frequency} rep={repetition} missing")
                    n_anomalies += 1
    if n_anomalies == 0:
        print(f"No unexpected missing combinations ({n_known} documented exclusion(s) found, as expected).")
    else:
        print(f"{n_anomalies} unexpected missing combination(s), {n_known} documented exclusion(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Path to the Dataset/ folder (containing Sub001/ ... Sub021/)")
    parser.add_argument("--manifest-csv", type=Path, default=None,
                        help="Optional path to save the full manifest as CSV")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-batch progress output")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"--root does not exist or is not a directory: {args.root}")

    start = time.time()
    manifest = build_manifest(args.root, quiet=args.quiet)

    summarize(manifest)
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f} s. {len(manifest)} files scanned under {args.root}")

    if args.manifest_csv is not None:
        manifest.to_csv(args.manifest_csv, index=False)
        print(f"Manifest written to {args.manifest_csv}")


if __name__ == "__main__":
    main()
