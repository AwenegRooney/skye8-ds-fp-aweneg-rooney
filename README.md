# Phonify

On-device model serving layer for converting TensorFlow/Keras models to TFLite and benchmarking them for low-cost Android deployment.

## Overview

Phonify is a lightweight Python toolchain for:

- loading Keras or SavedModel artifacts
- resolving input contracts and dynamic shapes
- freezing model variables before TFLite conversion
- running INT8/PTQ calibration and host-side benchmarking
- exporting benchmark reports and metadata for Android device testing

This project is designed for workflows where the model must be optimized for offline inference on lower-end Android devices without shipping a full cloud pipeline.

## Repository layout

- `src/tools/convert_model.py` — CLI conversion entry point
- `src/utils/data_loader.py` — dataset loading, validation, and dynamic-shape resolution
- `src/utils/report.py` — metric calculation and benchmark report generation
- `stage_a_test_app/` — Flutter benchmark app for Android-side testing
- `.github/workflows/` — CI and APK build automation

## Features

- Supports `.keras`, `.h5`, `.hdf5`, and TensorFlow SavedModel directories
- Accepts representative `.npy`/`.npz` calibration and evaluation datasets
- Handles single/multi-input models with explicit and inferred dynamic shapes
- Runs optional TFLite optimization modes: `none`, `dynamic`, and `int8`
- Produces human-readable benchmark markdown reports and JSON metadata

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## CLI usage

```bash
convert-model path/to/model.keras \
  --optimize int8 \
  --calib-data path/to/calib.npz \
  --eval-data path/to/eval.npz \
  --runs 30 \
  --calib-samples 100
```

Typical arguments:

- `--optimize {none,dynamic,int8}`
- `--calib-data` for representative calibration tensors
- `--eval-data` for evaluation data with targets
- `--input-shape NAME=D0,D1,...` to resolve dynamic dimensions explicitly
- `--out-dir` to choose where TFLite and reports are written

## Example conversion workflow

```bash
convert-model ./models/mobilenet_v2.keras \
  --optimize int8 \
  --calib-data ./data/demo_assets/data/calibration.npz \
  --eval-data ./data/demo_assets/data/evaluation.npz \
  --metric accuracy \
  --out-dir ./phone_models
```

## CI and quality gates

This repo includes a GitHub Actions CI workflow that runs on pushes and pull requests to `main`.

The workflow checks:

- formatting with Black
- linting with Ruff
- static typing with mypy
- test execution with pytest

The CI config lives in [.github/workflows/ci.yaml](.github/workflows/ci.yaml).

## Development notes

- The project targets Python 3.10+
- TensorFlow/TFLite conversion is intentionally strict: datasets must already match the model contract
- Dynamic shape resolution avoids guessing modality-specific preprocessing

## Benchmarking

Phonify generates:

- a TFLite model artifact
- a markdown benchmark report
- a JSON file containing model metadata for downstream Android benchmarking

The Flutter app in `stage_a_test_app/` is intended for real-device Android performance validation.
