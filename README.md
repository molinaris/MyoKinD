# MyoKinD toolkit

Minimal, reproducible scripts for the **MyoKinD** dataset: *HD sEMG and 3D Hand Kinematics During 
Sinusoidally Modulated Dynamic Finger Movements*, described in the IEEE Data Descriptor by 
Avilés-Carrillo, Molinari, and Elias. Molinari and Avilés-Carrillo contributed 
equally to the development of this repository. Dataset DOI: `10.6084/m9.figshare.31032934`.

This repository does **not** ship the dataset itself. Download it from Figshare, unzip it, and point the
scripts below at the resulting `Dataset/` folder (the one that directly contains `Sub001/ … Sub021/`).

## What is in the dataset

- **HD sEMG** (`Sub*/HD_sEMG/*.csv`): 128 channels x 92160 samples, 2052.5245535714284 Hz, no header.
  Columns 1-64 = Extensor Digitorum Communis (EDC), columns 65-128 = Flexor Digitorum Superficialis (FDS).
  Values are treated as microvolts (µV), with no additional conversion factor applied.
- **Joint angles** (`Sub*/HandKinematics/Angles/*.csv`): 17 columns x 4500 rows, 100 Hz, 5 Vicon header
  lines + 1 trailing blank line. Only the Z column of each finger group is non-zero (columns 4, 7, 10, 13,
  16, in increasing order = thumb, index, middle, ring, little).
- **Marker trajectories** (`Sub*/HandKinematics/Trajectories/*.csv`): 38 columns x 4500 rows, same header
  format, X/Y/Z (mm) for 12 markers ordered `MW, LW, PT, DT, PI, DI, PM, DM, PR, DR, PL, DL`.
- **`AddInfo/Task_randomization.ods`** and **`AddInfo/Kinematic_review.ods`**: auxiliary documentation
  (per-participant randomization order, and a color-coded manual quality review of the kinematic
  recordings). These are reference documents, not parsed by any script here.

Files follow the naming convention `SubXXX_T_F_450_R.csv` (task, frequency, fixed duration, repetition), or
`SubXXX_rest_R.csv` / `SubXXX_start_position_R.csv` for the two supplementary conditions. The `T` field is
one of 8 task codes, defined in `examples/load_hdsemg.py` (`TASK_NAMES`) and printed by `load_hdsemg.py`/`load_kinematics.py` alongside the loaded trial:

| T | Task name | Description |
|---|---|---|
| 1 | Index | Index finger flexion-extension |
| 2 | Middle | Middle finger flexion-extension |
| 3 | Ring-Little | Coupled flexion-extension of the ring and little fingers |
| 4 | Thumb | Thumb opposition-reposition |
| 5 | I. Pinch | Index-thumb pinch opening and closing |
| 6 | M. Pinch | Middle-thumb pinch opening and closing |
| 7 | Tripod | Tripod pinch (thumb, index, middle) |
| 8 | 5-Finger | Full-hand opening and closing (all five fingers) |

Known exception: participant Sub017 is missing the `3_05_450_1` trial in all three modalities (excluded by
the authors for incorrect task execution) and the `start_position_0` kinematic files (kinematic
acquisition failure; the matching HD sEMG file is present) - both are handled (skipped, not an error) by
every script here.

## Repository layout

```
MyoKinD/
├── examples/
│   ├── load_hdsemg.py             # load one HD sEMG trial, split EDC/FDS, moving RMS
│   ├── load_kinematics.py         # load matching Angles + Trajectories
│   └── synchronize_modalities.py  # demonstrate the assumed HD sEMG <-> kinematics alignment
├── validation/
│   ├── inspect_dataset.py         # read-only structural/integrity check + manifest
│   ├── hdsemg_quality.py          # SNR / SMR / SPR quality metrics
│   ├── kinematic_quality.py       # coherence / phase lag / THDN quality metrics
│   └── aggregate_quality_metrics.py  # runs both across participants, writes a per-trial quality table
└── figures/
    ├── plot_example_trial.py      # plots HD sEMG + kinematics for any chosen trial
    └── plot_quality_summary.py    # plots quality metric distributions from the quality table above
```

## Installation

```
pip install -r requirements.txt
```

or, with conda:

```
conda env create -f environment.yml
conda activate myokind
```

Only `numpy`, `scipy`, `pandas` and `matplotlib` are required — see `requirements.txt`.

## Usage

All scripts take `--root` pointing at the dataset's `Dataset/` folder (the one containing `SubXXX/`
subfolders) and are meant to be run from the `MyoKinD/` folder.

```bash
# Structural inspection + manifest
python validation/inspect_dataset.py --root /path/to/Dataset --manifest-csv manifest.csv

# Load and plot one HD sEMG trial
python examples/load_hdsemg.py --root /path/to/Dataset --participant Sub001 --task 1 --freq 05 --rep 0 --plot

# Load and plot matching kinematics (Angles + Trajectories)
python examples/load_kinematics.py --root /path/to/Dataset --participant Sub001 --task 1 --freq 05 --rep 0 --plot

# Demonstrate the assumed HD sEMG <-> kinematics time alignment
python examples/synchronize_modalities.py --root /path/to/Dataset --participant Sub001 --task 7 --freq 05 --rep 0

# HD sEMG quality metrics (SNR/SMR/SPR)
python validation/hdsemg_quality.py --root /path/to/Dataset --participant Sub001 --all --output-csv sub001_emg_quality.csv

# Kinematic quality metrics (coherence, phase lag, THDN)
python validation/kinematic_quality.py --root /path/to/Dataset --participant Sub001 --all --output-csv sub001_kin_quality.csv

# Aggregate across participants into one row per trial (snr, smr, spr, thdn, coh, phase_lag)
python validation/aggregate_quality_metrics.py --root /path/to/Dataset --participants Sub001 Sub002 Sub003 --output-csv quality_summary.csv
python validation/aggregate_quality_metrics.py --root /path/to/Dataset --participants all --output-csv quality_summary.csv

# Plot one trial's HD sEMG + kinematics
python figures/plot_example_trial.py --root /path/to/Dataset --participant Sub001 --task 7 --freq 05 --rep 0 --output tripod_example.png

# Plot quality metric distributions from the table above
python figures/plot_quality_summary.py --input-csv quality_summary.csv --output quality_metrics.png
```

`aggregate_quality_metrics.py` only collects the raw per-trial quality table (one row per participant/task/
frequency/repetition, with `snr`/`smr`/`spr` averaged across the 128 HD sEMG channels and `thdn` averaged
across the 5 fingers). Computing means/standard deviations and plotting the distributions is
`plot_quality_summary.py`'s job, so the same table can be reused for both without re-running the (slow)
per-trial processing.

## Processing conventions used throughout this repository

- **Spectral window:** periodic Hann window is used explicitly in every Welch/CSD/coherence call.
- **HD sEMG band-pass filter:** 4th-order Butterworth, 10-500 Hz, zero-phase (`sosfiltfilt`). Applied in
  `figures/plot_example_trial.py`, but deliberately **not** applied by `validation/hdsemg_quality.py`'s 
  SNR/SMR/SPR metrics - those characterize the raw acquisition itself, computed on the trimmed but unfiltered signal.
- **Temporal trimming:** a single convention is used everywhere — discard the first 4 s and the last 1 s of
  every 45 s trial, keeping the 4-44 s window (40 s of data), to avoid the transients of task onset/offset.
  Applied to both HD sEMG and kinematics, for temporal correspondence between both modalities.
- **Representative finger per task:** Index -> index; Middle -> middle; Ring-Little -> ring; Thumb -> thumb;
  I. Pinch -> index; M. Pinch -> middle; Tripod -> index; 5-Finger -> index.
- **THDN:** computed as `THD * (A_fundamental / A_fundamental_max_across_fingers) * 100`, i.e. the harmonic
  distortion ratio scaled by each finger's relative fundamental amplitude within the same trial.
- **HD sEMG units:** raw CSV integers are used directly as microvolts (µV).
- **Missing kinematic samples:** a small number of Angles/Trajectories files have blank cells (flagged as `MISSING_VALUES`, not an error, by `inspect_dataset.py`). These load as
  NaN and are linearly interpolated in `examples/load_kinematics.py` right after loading (with a printed note of where and
how many values were filled per file), so every downstream script sees a complete, finite signal.

## License

See `LICENSE`. The MyoKinD **dataset** itself is distributed separately under CC BY 4.0 by the original
authors — this repository only contains code.
