# Pain Monitoring System Design

## Goal

Estimate patient pain from facial expression dynamics and track possible wheeze burden from respiratory audio features.

## Upgraded pipeline

1. Input video frames and optional WAV or microphone audio.
2. Face analysis using richer handcrafted expression features.
3. Audio analysis using respiratory motion, tonal wheeze energy, and spectral concentration.
4. Multimodal ridge model predicts pain score `0-10` and wheeze probability `0-1`.
5. Personal baseline calibration stabilizes the pain score.
6. Temporal smoothing and hysteresis detect pain episodes.
7. Live logs, summaries, and dashboard views expose both pain and wheeze measurements.

## Dataset workflow

1. Extract multimodal features from labeled videos with `extract-features`.
2. Prepare a larger training CSV with `prepare-dataset` by merging multiple CSVs and augmenting small sets.
3. Train with `train`.
4. Evaluate with `evaluate`.
5. Deploy with `live --model`.

## Key files

- Model: `artifacts/pain_model.json`
- Training metrics: `artifacts/pain_training_metrics.json`
- Prepared dataset: `artifacts/prepared_training.csv`
- Live frame logs: `sample_data/pain_session_*.csv`
- Episode event logs: `sample_data/pain_episode_events_*.csv`

## Clinical reminder

This remains decision-support research software. True hospital-grade accuracy still depends on real clinical datasets, annotation quality, bias testing, and safety validation.
