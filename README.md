# Patient Pain and Wheeze Monitoring

This project detects visible pain-related facial changes and also estimates wheezing from respiratory audio or microphone input.

It includes:

- More facial-expression features for pain estimation
- Landmark-based eyebrow contraction detection using MediaPipe FaceMesh when available
- Optional wheeze analysis from WAV audio or live microphone input
- Dataset preparation and synthetic expansion for larger training sets
- Multimodal model training and evaluation
- Live logging, summaries, and dashboard views for pain plus wheeze

## Important note

This is still research or demo software, not a certified medical device. It can help with monitoring and experimentation, but real hospital deployment still needs clinically labeled datasets, validation, and safety review.

## Quick start

Open a terminal inside this project folder first.

### Install dependencies

Windows:

```powershell
py -3 -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m pip install -r requirements.txt
```

If `python3` is not available, use:

```bash
python -m pip install -r requirements.txt
```

### Run the live camera monitor

Windows:

```powershell
py -3 pain_main.py --config configs\default_config.json live
```

macOS or Linux:

```bash
python3 pain_main.py --config configs/default_config.json live
```

If you already trained a model and want to use it:

Windows:

```powershell
py -3 pain_main.py --config configs\default_config.json live --model artifacts\pain_model.json
```

macOS or Linux:

```bash
python3 pain_main.py --config configs/default_config.json live --model artifacts/pain_model.json
```

Keep your face relaxed and normal during the first calibration seconds. The live monitor now learns your personal neutral eyebrow, mouth, nose, and model-score baseline, then suppresses false `Pain Detected` output when the current face still matches that baseline.

### Run the Streamlit dashboard

Windows:

```powershell
py -3 -m streamlit run streamlit_pain_app.py
```

macOS or Linux:

```bash
python3 -m streamlit run streamlit_pain_app.py
```

If `python3 -m streamlit` does not work on your laptop, try:

```bash
python -m streamlit run streamlit_pain_app.py
```

## Main workflow

### 1. Prepare a larger training CSV

Merge one or more labeled CSV files and expand them with controlled synthetic augmentation:

Windows:

```powershell
py -3 pain_main.py prepare-dataset --csv sample_data\pain_labels\pain_labeled_template.csv --out-csv artifacts\prepared_training.csv --augment-factor 20
```

macOS or Linux:

```bash
python3 pain_main.py prepare-dataset --csv sample_data/pain_labels/pain_labeled_template.csv --out-csv artifacts/prepared_training.csv --augment-factor 20
```

You can pass multiple CSV files after `--csv`.

### 2. Train the multimodal pain model

Windows:

```powershell
py -3 pain_main.py train --csv artifacts\prepared_training.csv --model-out artifacts\pain_model.json --metrics-out artifacts\pain_training_metrics.json
```

macOS or Linux:

```bash
python3 pain_main.py train --csv artifacts/prepared_training.csv --model-out artifacts/pain_model.json --metrics-out artifacts/pain_training_metrics.json
```

### 3. Evaluate the model

Windows:

```powershell
py -3 pain_main.py evaluate --model artifacts\pain_model.json --csv artifacts\prepared_training.csv
```

macOS or Linux:

```bash
python3 pain_main.py evaluate --model artifacts/pain_model.json --csv artifacts/prepared_training.csv
```

### 4. Extract multimodal features from a video

Pain-only extraction:

Windows:

```powershell
py -3 pain_main.py extract-features --video sample_data\sample_face_video.mp4 --out-csv artifacts\video_features.csv --fixed-label 6.0
```

Pain + wheeze extraction with aligned WAV audio:

Windows:

```powershell
py -3 pain_main.py extract-features --video sample_data\sample_face_video.mp4 --audio sample_data\respiratory_audio.wav --out-csv artifacts\video_features_with_audio.csv --fixed-label 6.0 --fixed-wheeze-label 0.7
```

### 5. Run live monitoring

Camera with automatic microphone wheeze input when available:

```powershell
py -3 pain_main.py --config configs\default_config.json live
```

Video plus WAV audio:

```powershell
py -3 pain_main.py --config configs\default_config.json live --video sample_data\sample_face_video.mp4 --audio sample_data\respiratory_audio.wav
```

## Camera-side improvements in this version

- The live panel is now smaller and moves to the side so it does not cover the full face.
- Pain and wheeze values are now tuned to react faster to smaller but real changes.
- Eyebrow position, eyebrow motion, nose contrast, and mouth micro-motion are now measured separately for more attentive pain detection.
- Eyebrow contraction is measured from facial landmarks using both inner-eyebrow distance and inner-brow angle/lowering. If the brows pull together strongly enough, the overlay shows `Eyebrow: Pain` with a confidence score.
- During calibration the overlay keeps pain display suppressed, then uses your relaxed eyebrow distance as the normal baseline to avoid false pain output.
- If the same pain-like face is held for several seconds or longer, the monitor now keeps building sustained evidence instead of treating it like a one-frame change.
- A neutral-face guard now keeps calm normal faces below the pain-detected range after calibration.
- Trained-model output is corrected against your calibrated neutral face, so a model that scores your normal face too high is pulled back to `No Pain`.
- Wheeze detection now also uses sustained audio evidence, so repeated/held wheeze-like breathing can raise confidence more reliably.
- When the face is mostly still, the values stay calmer to reduce random variation.
- When you make even a lighter facial-expression change, the pain score can rise sooner.
- The camera now shows clear labels: `Pain Detected` or `No Pain`.
- The camera also shows clear wheeze labels: `Wheezing Detected` or `No Wheezing`.
- Optional Gmail notifications can send instant alerts and the patient session report.

These controls are configured in `configs/default_config.json`:

- `overlay_scale`
- `overlay_anchor`
- `pain_display_alpha_still`
- `pain_display_alpha_change`
- `wheeze_display_alpha`
- `expression_change_threshold`
- `pain_expression_boost`
- `micro_expression_trigger_threshold`
- `enable_eyebrow_landmarks`
- `eyebrow_distance_pain_threshold`
- `eyebrow_contraction_drop_threshold`
- `eyebrow_angle_pain_threshold`
- `eyebrow_pain_confidence_threshold`
- `eyebrow_pain_score_boost`
- `brow_edge_pain_boost`
- `sustained_pain_signal_threshold`
- `sustained_pain_boost`
- `sustained_pain_seconds_to_full_boost`
- `neutral_expression_signal_threshold`
- `neutral_motion_threshold`
- `neutral_face_score_cap`
- `wheeze_support_boost`
- `sustained_wheeze_signal_threshold`
- `sustained_wheeze_boost`
- `sustained_wheeze_seconds_to_full_boost`
- `wheeze_alert_threshold`

## Gmail notifications

To send alerts and the patient report to Gmail, set these values in `configs/default_config.json` or through environment variables:

- `email_notifications_enabled`
- `notification_email_to`
- `notification_email_from`
- `notification_email_password`
- `smtp_host`
- `smtp_port`

This project is configured to notify:

- `gunaseelanv58@gmail.com`
- `kavipreethirathna@gmail.com`

You can also keep the password out of the JSON file and set:

```powershell
$env:PAIN_MONITOR_EMAIL_FROM = "yourgmail@gmail.com"
$env:PAIN_MONITOR_EMAIL_TO = "receiver@gmail.com"
$env:PAIN_MONITOR_EMAIL_PASSWORD = "your-gmail-app-password"
```

Gmail requires an App Password for SMTP access.

## Command guide

If you want a simple explanation of why each command line is used and what output it gives, read `docs/command_reference.md`.

## Parameter help

For a simple explanation of every parameter and every JSON output, read `docs/parameter_reference.md`.

## Kaggle datasets

I also added Kaggle-ready import support. The guide is in `docs/kaggle_integration.md`.

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

New camera feature columns include:

- `brow_position`
- `brow_motion`
- `eyebrow_distance_ratio`
- `eyebrow_angle_score`
- `eyebrow_contraction`
- `eyebrow_pain_confidence`
- `mouth_micro_motion`
- `nose_contrast`

## Outputs

- Model: `artifacts/pain_model.json`
- Training metrics: `artifacts/pain_training_metrics.json`
- Prepared dataset: `artifacts/prepared_training.csv`
- Live session logs: `sample_data/pain_session_*.csv`
- Episode events: `sample_data/pain_episode_events_*.csv`
