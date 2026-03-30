# Patient Pain Detection and Duration Monitoring

This is a complete project package for pain-expression monitoring using face video, with a full workflow:

- Feature extraction from raw videos
- Label-based model training
- Model evaluation
- Live pain + duration monitoring
- Session summarization
- Dashboard analytics
- Unit tests

## Project Structure

```text
pain-monitoring-project/
  pain_main.py
  streamlit_pain_app.py
  requirements.txt
  .gitignore
  configs/
    default_config.json
  src/pain_monitoring/
    __init__.py
    config.py
    episode_tracker.py
    features.py
    io_utils.py
    logger.py
    model.py
    overlay.py
    reporting.py
    runner.py
    types.py
  tests/
    test_episode_tracker.py
    test_model.py
  docs/
    pain_monitoring.md
  sample_data/
    pain_labels/
      pain_labeled_template.csv
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Tools Used

- Python 3.13
- OpenCV (`opencv-python`) for face and expression-region analysis
- NumPy for numerical operations
- Pandas for dataset and log processing
- Streamlit for dashboard visualization
- Haar Cascade models (OpenCV bundled): frontal face, eyes, smile
- `unittest` for test automation

## Command Prompt (Step By Step)

```powershell
cd "C:\Users\HP\OneDrive\Documents\New project\pain-monitoring-project"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pain_main.py train --csv sample_data\pain_labels\pain_labeled_template.csv --model-out artifacts\pain_model.json --metrics-out artifacts\pain_training_metrics.json
python pain_main.py evaluate --model artifacts\pain_model.json --csv sample_data\pain_labels\pain_labeled_template.csv
python pain_main.py --config configs/default_config.json live --model artifacts\pain_model.json
streamlit run streamlit_pain_app.py
```

## Commands

### 1. Live monitoring

```powershell
python pain_main.py --config configs/default_config.json live
```

Important for accuracy: during the first `6` seconds, keep a neutral face so the system calibrates your personal baseline.

### 2. Live monitoring with trained model

```powershell
python pain_main.py --config configs/default_config.json live --model artifacts/pain_model.json
```

### 3. Extract features from a video (for dataset building)

```powershell
python pain_main.py --config configs/default_config.json extract-features --video input.mp4 --out-csv sample_data/generated_features.csv --sample-every 3
```

Attach a fixed label if this video segment has one known pain level:

```powershell
python pain_main.py --config configs/default_config.json extract-features --video input.mp4 --out-csv sample_data/generated_features_labeled.csv --sample-every 3 --fixed-label 6.0
```

### 4. Train model

```powershell
python pain_main.py train --csv sample_data/pain_labels/pain_labeled_template.csv --model-out artifacts/pain_model.json --metrics-out artifacts/pain_training_metrics.json
```

### 5. Evaluate model

```powershell
python pain_main.py evaluate --model artifacts/pain_model.json --csv sample_data/pain_labels/pain_labeled_template.csv
```

### 6. Summarize one live session

```powershell
python pain_main.py summarize-session --csv sample_data/pain_session_YYYYMMDD_HHMMSS.csv --out artifacts/session_summary.csv
```

### 7. Dashboard

```powershell
streamlit run streamlit_pain_app.py
```

### 8. Unit tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- This is a project-grade engineering package for demos/academics.
- It is still not a certified medical device.
- For real clinical deployment, you need hospital-approved datasets, calibration studies, bias audits, and regulatory clearance.

## Cleanup Generated Logs

If you want to remove generated run files:

```powershell
Remove-Item sample_data\pain_session_*.csv -Force
Remove-Item sample_data\pain_episode_events_*.csv -Force
Remove-Item artifacts\session_summary.csv -Force
```

If access is denied, first close Streamlit/OpenCV windows and retry.
