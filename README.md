# MyoKinD toolkit
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21721906.svg)](https://doi.org/10.5281/zenodo.21721906)

Minimal, reproducible scripts for the **MyoKinD** dataset: *HD sEMG and 3D Hand Kinematics During
Sinusoidally Modulated Dynamic Finger Movements*, described in the IEEE Data Descriptor by
Avilés-Carrillo, Molinari, and Elias. Molinari and Avilés-Carrillo contributed
equally to the development of this repository. Dataset DOI: `10.6084/m9.figshare.31032934`.

The repository also contains the implementation of the biomechanical hand model used to obtain the
dataset's joint angles from the marker trajectories, together with the kinematic analysis built on it
(see [Biomechanical hand model](#biomechanical-hand-model)). This model is described in
Avilés-Carrillo et al. (2026), *A Biomechanical Hand Model to Quantify Finger Joint Kinematics Using a
3D Motion Capture System*, DOI: `10.64898/2026.02.09.704796`.

This repository does **not** ship the dataset itself. Download it from Figshare, unzip it, and point the
scripts below at the resulting `Dataset/` folder (the one that directly contains `Sub001/ … Sub021/`).

## What is in the dataset

- **HD sEMG** (`Sub*/HD_sEMG/*.csv`): 128 channels x 92160 samples, 2052.5245535714284 Hz, no header.
  Columns 1-64 = Extensor Digitorum Communis (EDC), columns 65-128 = Flexor Digitorum Superficialis (FDS).
  Values are treated as microvolts (µV), with no additional conversion factor applied.
- **Joint angles** (`Sub*/HandKinematics/Angles/*.csv`): 17 columns x 4500 rows, 100 Hz, 5 Vicon header
  lines + 1 trailing blank line. Only the Z column of each finger group is non-zero (columns 4, 7, 10, 13,
  16, in increasing order = thumb, index, middle, ring, little). These angles are not an independent
  measurement: Vicon ProCalc computes them from the marker trajectories using the biomechanical hand
  model, which `biomechanical_model/estimate_kinematic_angles.py` reproduces within 1 mdeg.
- **Marker trajectories** (`Sub*/HandKinematics/Trajectories/*.csv`): 38 columns x 4500 rows, same header
  format, X/Y/Z (mm) for 12 markers ordered `MW, LW, PT, DT, PI, DI, PM, DM, PR, DR, PL, DL`.
- **`AddInfo/Task_randomization.ods`** and **`AddInfo/Kinematic_review.ods`**: auxiliary documentation
  (per-participant randomization order, and a color-coded manual quality review of the kinematic
  recordings). These are reference documents, not parsed by any script here.

Files follow the naming convention `SubXXX_T_F_450_R.csv` (task, frequency, fixed duration, repetition), or
`SubXXX_rest_R.csv` (HD sEMG only) / `SubXXX_start_position_R.csv` (all three modalities) for the two
supplementary conditions. The `T` field is one of 8 task codes, defined in `examples/load_hdsemg.py`
(`TASK_NAMES`) and printed by `load_hdsemg.py`/`load_kinematics.py` alongside the loaded trial:

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
│   ├── aggregate_quality_metrics.py  # runs both across participants, writes a per-trial quality table
│   ├── rest_noise_rms.py          # baseline RMS of `rest` recordings vs. the 1 uV storage quantization noise
│   ├── aggregate_rest_noise.py    # runs the above across participants, writes a per-channel-trial table
│   ├── wrist_marker_precision.py  # within-trial std of the MW-LW wrist marker distance (OMC precision)
│   ├── aggregate_wrist_precision.py  # runs the above across participants, writes a per-trial table
│   ├── phase_lag_analysis.py      # phase_lag distribution: fixed-delay/wrapping/task-bug tests, ICC(1), bimodality
│   └── quality_summary.csv        # per-trial quality table for all 21 participants (input of phase_lag_analysis.py)
├── figures/
│   ├── plot_example_trial.py      # plots HD sEMG + kinematics for any chosen trial
│   └── plot_quality_summary.py    # plots quality metric distributions from the quality table above
├── biomechanical_model/
│   ├── estimate_kinematic_angles.py  # estimates joint angles from marker trajectories, computes RoM, mean speed, THDN
│   ├── plot_example_kinematic_metrics.py  # plots one trial's finger positions, per-cycle position/velocity, and spectra
│   ├── plot_interactive_kinematic.py  # interactive 3D view of the angle-estimation geometry for any chosen trial
│   ├── aggregate_kinematic_metrics.py  # runs the above across participants, writes the metric tables + population figure
│   ├── statistical_analysis.R     # ART mixed-effects models with Bonferroni post hoc comparisons
│   ├── kinematic_metrics_participants.csv  # participant-level metric table (input of statistical_analysis.R)
│   ├── kinematic_significant_interactions.csv  # significant Task x Frequency interactions from R (figure annotations)
│   └── kinematic_statistical_analysis.txt  # full R statistical report
└── tests/                         # pytest suite, one test_<script>.py per script (see Tests)
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

Only `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn` and `diptest` are required — see
`requirements.txt`. The last two are only used by `validation/phase_lag_analysis.py`. `pytest` is
also listed, only to run the test suite (see [Tests](#tests)).

The statistical analysis of the biomechanical model (`biomechanical_model/statistical_analysis.R`) runs in
R and requires the packages `tidyverse`, `lmerTest`, `pbkrtest`, `ARTool`, and `emmeans`:

```r
install.packages(c("tidyverse", "lmerTest", "pbkrtest", "ARTool", "emmeans"))
```

## Usage

The commands below cover the scripts of the dataset paper (`examples/`, `validation/`, and `figures/`).
The biomechanical model has its own commands in [Biomechanical hand model](#biomechanical-hand-model).

All scripts are meant to be run from the `MyoKinD/` folder. Scripts that read the raw recordings take
`--root` pointing at the dataset's `Dataset/` folder (the one containing `SubXXX/` subfolders), while
`phase_lag_analysis.py` and `plot_quality_summary.py` read an already computed quality table through
`--input-csv`.

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

# Optional: also save the per-channel HD sEMG quality table (used for the channel-wise SNR-threshold rejection rate)
python validation/aggregate_quality_metrics.py --root /path/to/Dataset --participants all --output-csv quality_summary.csv --output-csv-channels quality_summary_channels.csv

# Baseline noise RMS of the `rest` recordings vs. the dataset's 1 uV storage quantization noise
python validation/rest_noise_rms.py --root /path/to/Dataset --participant Sub001 --all --output-csv sub001_rest_rms.csv

# Aggregate the above across participants into one row per channel-trial (rms_uv, quant_penalty_db)
python validation/aggregate_rest_noise.py --root /path/to/Dataset --participants Sub001 Sub002 Sub003 --output-csv rest_noise_summary.csv
python validation/aggregate_rest_noise.py --root /path/to/Dataset --participants all --output-csv rest_noise_summary.csv

# OMC reconstruction precision from the within-trial std of the MW-LW wrist marker distance
python validation/wrist_marker_precision.py --root /path/to/Dataset --participant Sub001 --all --output-csv sub001_wrist_precision.csv

# Aggregate the above across participants into one row per trial (mean_distance_mm, std_distance_mm)
python validation/aggregate_wrist_precision.py --root /path/to/Dataset --participants Sub001 Sub002 Sub003 --output-csv wrist_precision_summary.csv
python validation/aggregate_wrist_precision.py --root /path/to/Dataset --participants all --output-csv wrist_precision_summary.csv

# phase_lag distribution analysis: H1/H2/H3 discrimination, ICC(1) (between- vs. within-participant), bimodality
python validation/phase_lag_analysis.py --input-csv validation/quality_summary.csv --output-csv phase_lag_analysis_summary.csv

# Plot one trial's HD sEMG + kinematics
python figures/plot_example_trial.py --root /path/to/Dataset --participant Sub001 --task 7 --freq 05 --rep 0 --output tripod_example.png

# Plot quality metric distributions from the table above
python figures/plot_quality_summary.py --input-csv quality_summary.csv --output quality_metrics.png
# Optional: also report the channel-wise SNR-threshold rejection rate (needs the per-channel table above)
python figures/plot_quality_summary.py --input-csv quality_summary.csv --input-channels-csv quality_summary_channels.csv --output quality_metrics.png
```

`aggregate_quality_metrics.py` only collects the raw per-trial quality table (one row per participant/task/
frequency/repetition, with `snr`/`smr`/`spr` averaged across the 128 HD sEMG channels and `thdn` averaged
across the 5 fingers). Computing means/standard deviations and plotting the distributions is
`plot_quality_summary.py`'s job, so the same table can be reused for both without re-running the (slow)
per-trial processing.

## Processing conventions

- **Spectral window:** periodic Hann window is used explicitly in every Welch/CSD/coherence call.
- **HD sEMG band-pass filter:** 4th-order Butterworth, 10-500 Hz, zero-phase (`sosfiltfilt`). Applied in
  `figures/plot_example_trial.py`, but deliberately **not** applied by `validation/hdsemg_quality.py`'s
  SNR/SMR/SPR metrics - those characterize the raw acquisition itself, computed on the trimmed but unfiltered signal.
- **Temporal trimming:** discard the first 4 s and the last 1 s of every 45 s trial, keeping the 4-44 s
  window (40 s of data), to avoid the transients of task onset/offset. Applied to both HD sEMG and
  kinematics, for temporal correspondence between both modalities. The biomechanical model uses its own
  window (see [Processing parameters](#processing-parameters)).
- **Representative finger per task:** Index -> index; Middle -> middle; Ring-Little -> ring; Thumb -> thumb;
  I. Pinch -> index; M. Pinch -> middle; Tripod -> index; 5-Finger -> index.
- **THDN (normalized total harmonic distortion):** computed as
  `THD * (A_fundamental / A_fundamental_max_across_fingers) * 100`, i.e. the harmonic distortion ratio
  scaled by each finger's relative fundamental amplitude within the same trial. The PSD is estimated with
  Welch's method (4 s periodic Hann window), and the amplitude of the fundamental and of harmonics 2-5 is
  taken from the target frequency bin ± 1 bin. The biomechanical model uses a different Welch window
  (see [Processing parameters](#processing-parameters)).
- **HD sEMG units:** raw CSV integers are used directly as microvolts (µV).
- **Storage quantization is negligible:** values are rounded to whole µV before distribution (the native
  Intan RHD2164 ADC resolution is 0.195 µV), adding a quantization noise floor of 1/√12 ≈ 0.29 µV RMS to
  every channel. `validation/rest_noise_rms.py` measures the actual baseline RMS of every `rest` recording
  and expresses this quantization floor as an SNR penalty relative to it. Across all 21 participants (8064
  channel-trials, computed with `validation/aggregate_rest_noise.py`), measured rest RMS is 15.68 ± 10.87 µV
  (min 2.66 µV), giving a worst-case quantization penalty of 0.051 dB (mean 0.005 dB) — well under the
  0.35 dB penalty a 1 µV noise floor would imply, so the 1 µV rounding is not a meaningful resolution loss
  in practice.
- **OMC reconstruction precision:** the wrist stayed stationary throughout the protocol, so the distance
  between the MW and LW markers (radial/ulnar styloid processes) is nominally constant within a trial;
  `validation/wrist_marker_precision.py` uses its within-trial standard deviation as an empirical, post hoc
  estimate of the combined effect of reconstruction error and soft-tissue motion at the wrist. Across all
  21 participants (1069 valid trials - the 48 task trials plus the 3 `start_position` recordings per
  participant, minus Sub017's two documented exclusions - computed with `validation/aggregate_wrist_precision.py`),
  the within-trial standard deviation of this distance was 0.31 mm on average (95th percentile, 0.67 mm).
- **`phase_lag`'s elevated standard deviation (-0.52 ± 1.01 rad) is trial-to-trial, not between-participant,
  and not explained by a fixed instrumentation delay, phase wrapping, or a task-specific sign bug:**
  `validation/phase_lag_analysis.py` tests and rejects all three (see its module docstring), and quantifies
  the spread with ICC(1) - the one-way random-effects intraclass correlation (Fisher, 1925; Searle, Casella &
  McCulloch, 1992, sec. 3.4 for unbalanced groups) - by participant: only ~5.6% of the variance is
  between-participant (ANOVA F(20,986)=3.86, p≈3×10⁻⁸ - detectable, but small - and all 21 per-participant
  means are negative, i.e. no subgroup of participants with positive mean phase). A 2-component Gaussian
  mixture fits `phase_lag` better than 1 by BIC (consistently so pooled and within each frequency
  separately, ruling out a frequency-aggregation artifact), but the Hartigan dip test never rejects
  unimodality (p > 0.5 in every subset tested) - reported together rather than selectively, the bimodality
  itself is suggestive, not statistically confirmed.
- **Missing kinematic samples:** a small number of Angles/Trajectories files have blank cells (flagged as
  `MISSING_VALUES`, not an error, by `inspect_dataset.py`). These load as NaN and are linearly
  interpolated in `examples/load_kinematics.py` right after loading (with a printed note of where and how
  many values were filled per file), so every downstream script sees a complete, finite signal.

## Biomechanical hand model

`biomechanical_model/` contains the implementation of the hand model described in Avilés-Carrillo et al.
(2026). The model quantifies thumb opposition-reposition (O-R) and the flexion-extension (F-E) of the
index, middle, ring, and little fingers through projected joint angles derived from the motion capture
markers.

Each joint angle is computed as the rotation between two vectors, A and B, around a third vector, C,
that serves as the reference axis. Vicon ProCalc applies the same procedure to generate the dataset's
`Angles` files, and `estimate_kinematic_angles.py` reproduces them from the `Trajectories` files with a
maximum difference below 0.001° (verified by `tests/test_estimate_kinematic_angles.py`).

From the estimated angles, the analysis characterizes each finger with range of movement (RoM), mean
angular speed, and THDN, aggregates these metrics across participants, and compares tasks and
frequencies in R (`statistical_analysis.R`).

### Processing parameters

The biomechanical analysis uses its own processing parameters, which differ on purpose from the
general conventions listed in [Processing conventions](#processing-conventions):

- **Temporal trimming:** 5-40 s of each 45 s trial.
- **Cycle segmentation:** RoM and mean angular speed are computed cycle by cycle. Cycles are detected
  from the Hilbert phase of a task-specific reference finger (Index and I. Pinch -> index; Middle,
  M. Pinch, and Tripod -> middle; Ring-Little and 5-Finger -> ring; Thumb -> thumb), and cycles shorter
  than 80% of the expected period are discarded.
- **THDN:** same definition as in the dataset paper, but with an 8 s Hamming window for the Welch PSD.
- **Left-handed participant:** Sub005 is left-handed, so the sign of the reference axis C is reversed.

### Usage

All commands are run from the `MyoKinD/` folder. Output files are saved in `biomechanical_model/`
(only the file name given in `--output*` is used), where `statistical_analysis.R` reads its input.

```bash
# Estimate the joint angles of one trial and compare them with the dataset's Angles files
python biomechanical_model/estimate_kinematic_angles.py --root /path/to/Dataset --participant Sub017 --task 6 --freq 075 --rep 2 --mode angles

# Kinematic metrics (RoM, mean angular speed, THDN) of one trial
python biomechanical_model/estimate_kinematic_angles.py --root /path/to/Dataset --participant Sub017 --task 6 --freq 075 --rep 2 --mode metrics --output example_kinematic_metrics.png

# Interactive view of the angle-estimation geometry for one trial
python biomechanical_model/estimate_kinematic_angles.py --root /path/to/Dataset --participant Sub017 --task 6 --freq 075 --rep 2 --mode interactive

# Aggregate across participants into one row per participant/task/frequency/finger (repetitions averaged)
python biomechanical_model/aggregate_kinematic_metrics.py --root /path/to/Dataset --participants Sub001 Sub002 Sub003 --output-csv-averaged kinematic_metrics_participants.csv
python biomechanical_model/aggregate_kinematic_metrics.py --root /path/to/Dataset --participants all --output-csv-averaged kinematic_metrics_participants.csv --output population_kinematic_metrics.png

# Optional: also save the trial-level table (one row per repetition)
python biomechanical_model/aggregate_kinematic_metrics.py --root /path/to/Dataset --participants all --output-csv kinematic_metrics_trials.csv --output-csv-averaged kinematic_metrics_participants.csv

# Statistical analysis of the participant-level table (ART mixed-effects models, Bonferroni post hoc comparisons)
Rscript biomechanical_model/statistical_analysis.R
```

## Tests

Each script has a matching test file in `tests/` (for example, `tests/test_load_hdsemg.py` tests
`examples/load_hdsemg.py`). Some tests use small synthetic signals and always run. Others load real
recordings and only run when the location of the `Dataset/` folder is given in the
`MYOKIND_DATASET_ROOT` environment variable; without it, those tests are reported as skipped.

### Usage

All commands are run from the `MyoKinD/` folder.

```bash
# Run the tests that use synthetic signals (real-recording tests are skipped)
pytest tests/

# Run all tests, including those that load real recordings
MYOKIND_DATASET_ROOT=/path/to/Dataset pytest tests/

# Run the tests of a single script
pytest tests/test_estimate_kinematic_angles.py
```

On Windows (PowerShell), set the variable first with `$env:MYOKIND_DATASET_ROOT = "C:\path\to\Dataset"`
and then run `pytest tests/`. Five `probe_*` tests in `test_inspect_dataset.py` read small synthetic files
from `tests/fixtures/`, which is not included in this repository, so they fail with `FileNotFoundError`.

## License

See `LICENSE`. The MyoKinD **dataset** itself is distributed separately under CC BY 4.0 by the original
authors — this repository only contains code.
