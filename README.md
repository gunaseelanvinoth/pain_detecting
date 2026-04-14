# Patient Pain and Wheeze Monitoring

This upgraded project now supports a richer multimodal pipeline:

- More facial-expression features for pain estimation
- Optional wheeze analysis from WAV audio or live microphone input
- Dataset preparation and synthetic expansion for larger training sets
- Multimodal model training and evaluation
- Live logging, summaries, and dashboard views for pain plus wheeze

## Important note

This is still research or demo software, not a certified medical device. It can help with monitoring and experimentation, but real hospital deployment still needs clinically labeled datasets, validation, and safety review.

## Setup

```powershell
py -3 -m pip install -r requirements.txt
```

If `streamlit` is not on your PATH, run it like this:

```powershell
py -3 -m streamlit run streamlit_pain_app.py
```

## Main workflow

### 1. Prepare a larger training CSV

Merge one or more labeled CSV files and expand them with controlled synthetic augmentation:

```powershell
py -3 pain_main.py prepare-dataset --csv sample_data\pain_labels\pain_labeled_template.csv --out-csv artifacts\prepared_training.csv --augment-factor 20
```

You can pass multiple CSV files after `--csv`.

### 2. Train the multimodal pain model

```powershell
py -3 pain_main.py train --csv artifacts\prepared_training.csv --model-out artifacts\pain_model.json --metrics-out artifacts\pain_training_metrics.json
```

### 3. Evaluate the model

```powershell
py -3 pain_main.py evaluate --model artifacts\pain_model.json --csv artifacts\prepared_training.csv
```

### 4. Extract multimodal features from a video

Pain-only extraction:

```powershell
py -3 pain_main.py extract-features --video sample_data\sample_face_video.mp4 --out-csv artifacts\video_features.csv --fixed-label 6.0
```

Pain + wheeze extraction with aligned WAV audio:

```powershell
py -3 pain_main.py extract-features --video sample_data\sample_face_video.mp4 --audio sample_data\respiratory_audio.wav --out-csv artifacts\video_features_with_audio.csv --fixed-label 6.0 --fixed-wheeze-label 0.7
```

### 5. Run live monitoring

Camera with automatic microphone wheeze input when available:

```powershell
py -3 pain_main.py --config configs\default_config.json live --model artifacts\pain_model.json
```

Video plus WAV audio:

```powershell
py -3 pain_main.py --config configs\default_config.json live --model artifacts\pain_model.json --video sample_data\sample_face_video.mp4 --audio sample_data\respiratory_audio.wav
```

### 6. Open the dashboard

```powershell
py -3 -m streamlit run streamlit_pain_app.py
```

## Camera-side improvements in this version

- The live panel is now smaller and moves to the side so it does not cover the full face.
- Pain and wheeze values are now tuned with stronger display stabilization.
- When the face is mostly still, the values move slowly to reduce random variation.
- When you make a real expression change, the values react faster.

These controls are configured in [default_config.json](C:\Users\elaiy\OneDrive\Documents\New project\configs\default_config.json):

- `overlay_scale`
- `overlay_anchor`
- `pain_display_alpha_still`
- `pain_display_alpha_change`
- `wheeze_display_alpha`
- `expression_change_threshold`

## Parameter help

For a simple explanation of every parameter and every JSON output, read [parameter_reference.md](C:\Users\elaiy\OneDrive\Documents\New project\docs\parameter_reference.md).

## Kaggle datasets

I also added Kaggle-ready import support. The guide is here: [kaggle_integration.md](C:\Users\elaiy\OneDrive\Documents\New project\docs\kaggle_integration.md).

New commands:

```powershell
py -3 pain_main.py import-kaggle-respiratory --dataset-dir C:\path\to\respiratory-dataset --out-csv artifacts\kaggle_wheeze.csv
py -3 pain_main.py import-kaggle-face --dataset-dir C:\path\to\face-dataset --out-csv artifacts\kaggle_face_pain.csv
py -3 pain_main.py prepare-dataset --csv artifacts\kaggle_face_pain.csv artifacts\kaggle_wheeze.csv --out-csv artifacts\kaggle_combined.csv --augment-factor 10
py -3 pain_main.py train --csv artifacts\kaggle_combined.csv --model-out artifacts\pain_model_kaggle.json --metrics-out artifacts\pain_model_kaggle_metrics.json
```

Before Kaggle download commands will work, place your Kaggle token in `.kaggle\kaggle.json` inside this project and set:

```powershell
$env:KAGGLE_CONFIG_DIR = (Resolve-Path .kaggle).Path
```

## Why commands print JSON

The JSON shown after commands is not an error. It is the structured result of the command.

Example:

```json
{
  "model_path": "artifacts\\pain_model.json",
  "rows_evaluated": 50,
  "mae": 0.13
}
```

This means:

- which file was used,
- how many rows were checked,
- how accurate the model was on that dataset.

## Dataset guidance

For real improvement, add more real labeled CSVs built from your own annotated patient videos or public pain-expression datasets that you have permission to use. The new pipeline is designed to merge many CSVs into one training set instead of relying on a tiny template.

Recommended label columns:

- `pain_label_0_10`
- `wheeze_label_0_1`

The model will automatically use richer face features and audio-derived wheeze features when those columns are present.

## Outputs

- Model: `artifacts/pain_model.json`
- Training metrics: `artifacts/pain_training_metrics.json`
- Prepared dataset: `artifacts/prepared_training.csv`
- Live session logs: `sample_data/pain_session_*.csv`
- Episode events: `sample_data/pain_episode_events_*.csv`
