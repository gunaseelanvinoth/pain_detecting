# Pain Monitoring System Design

## Goal

Detect facial pain-expression proxies and estimate pain duration episodes over time.

## Pipeline

1. Input video frame stream.
2. Face and region analysis (eyes, brow, mouth, smile, motion).
3. Frame-level pain score prediction (`0-10`) via linear regression model.
4. Personal baseline calibration for improved pain-score accuracy.
5. Temporal smoothing and hysteresis-based episode detection.
6. Live frame logging + episode event logging.
7. Session summary and dashboard analytics.

## Dataset Workflow

1. Extract frame features from annotated videos using `extract-features`.
2. Add/verify `pain_label_0_10` labels.
3. Train with `train`.
4. Evaluate with `evaluate`.
5. Deploy with `live --model`.

## Artifacts

- Model: `artifacts/pain_model.json`
- Training metrics: `artifacts/pain_training_metrics.json`
- Live frame logs: `sample_data/pain_session_*.csv`
- Episode event logs: `sample_data/pain_episode_events_*.csv`
- Optional session summary: `artifacts/session_summary.csv`

## Configuration

System behavior is controlled by JSON config (default `configs/default_config.json`):

- camera/source sizing
- minimum face area
- start/end pain thresholds
- start/end hold durations
- score smoothing alpha
- baseline calibration window and sample count
- logging controls

## Clinical Reminder

The current system is decision-support software, not diagnostic software.
Clinical adoption needs prospective validation, bias evaluation, and safety governance.
