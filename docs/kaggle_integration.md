## Kaggle Integration

This project now includes Kaggle-ready import commands for:

- respiratory wheeze datasets,
- face-image pain datasets.

## Recommended datasets

- Respiratory Sound Database:
  [https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database](https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database)
- 5-Class Lung Sounds Dataset:
  [https://www.kaggle.com/datasets/annisahmrn/5-class-lung-sounds-dataset-from-several-sources](https://www.kaggle.com/datasets/annisahmrn/5-class-lung-sounds-dataset-from-several-sources)
- Pain face dataset:
  [https://www.kaggle.com/datasets/sammay594/pain-detection-face-expressions](https://www.kaggle.com/datasets/sammay594/pain-detection-face-expressions)

## Current machine status

Kaggle CLI is installed, but it still needs authentication before downloads can start.

The simplest setup for this project is:

1. Create `C:\Users\elaiy\OneDrive\Documents\New project\.kaggle`
2. Put your Kaggle token file there as `kaggle.json`
3. Run commands with `KAGGLE_CONFIG_DIR` pointing to that folder

Example:

```powershell
$env:KAGGLE_CONFIG_DIR = (Resolve-Path .kaggle).Path
```

## Import commands

### Respiratory dataset to wheeze CSV

This expects a folder containing `.wav` files and matching annotation `.txt` files.

```powershell
py -3 pain_main.py import-kaggle-respiratory --dataset-dir C:\path\to\respiratory-dataset --out-csv artifacts\kaggle_wheeze.csv
```

### Face dataset to pain CSV

This expects a folder containing face images arranged in label-like folder names such as `no_pain`, `neutral`, `mild`, `moderate`, `severe`, or `pain`.

```powershell
py -3 pain_main.py import-kaggle-face --dataset-dir C:\path\to\face-dataset --out-csv artifacts\kaggle_face_pain.csv
```

If the folder names do not contain a clear label, use a fallback label:

```powershell
py -3 pain_main.py import-kaggle-face --dataset-dir C:\path\to\face-dataset --out-csv artifacts\kaggle_face_pain.csv --default-label 6.0
```

## Merge and train

After importing both datasets:

```powershell
py -3 pain_main.py prepare-dataset --csv artifacts\kaggle_face_pain.csv artifacts\kaggle_wheeze.csv --out-csv artifacts\kaggle_combined.csv --augment-factor 10
py -3 pain_main.py train --csv artifacts\kaggle_combined.csv --model-out artifacts\pain_model_kaggle.json --metrics-out artifacts\pain_model_kaggle_metrics.json
py -3 pain_main.py evaluate --model artifacts\pain_model_kaggle.json --csv artifacts\kaggle_combined.csv
```

## What these importers do

- The respiratory importer converts annotated breathing cycles into wheeze-feature rows.
- The face importer extracts face-expression features from labeled images and converts them into pain-feature rows.
- The trainer can now learn pain and wheeze from separate rows in the same merged CSV.
