## Parameter Reference

This file explains the parameters used in `configs/default_config.json` and the JSON outputs printed by the commands.

### Why this file exists

The project uses a JSON config file because it is an easy way to change behavior without editing Python code.

`configs/default_config.json` means:

- It is a settings file.
- Each key changes one part of the monitoring or training pipeline.
- You can create another config JSON for a different camera, hospital room, or patient setup.

## Config parameters

### Camera and frame parameters

- `camera_index`: Which camera OpenCV opens. Usually `0` is the default webcam.
- `frame_width`: Camera preview width in pixels.
- `frame_height`: Camera preview height in pixels.
- `min_face_area`: Minimum allowed face box area. Small noisy detections below this size are ignored.

### Pain episode parameters

- `pain_start_threshold`: Pain score above this value starts a pain episode.
- `pain_end_threshold`: Pain score below this value ends a pain episode.
- `start_hold_seconds`: Score must stay above the start threshold for this long before starting an episode.
- `end_hold_seconds`: Score must stay below the end threshold for this long before ending an episode.

### Stability and smoothing parameters

- `smoothing_alpha`: Reserved general smoothing control used by the tracking pipeline.
- `prediction_smoothing_alpha`: Base smoothing for raw prediction flow.
- `pain_display_alpha_still`: Strong stabilization when the face is not changing much.
- `pain_display_alpha_change`: Faster update speed when expression changes clearly.
- `wheeze_display_alpha`: Smoothing strength for wheeze display values.
- `expression_change_threshold`: Minimum detected expression change before the UI updates faster.

### Calibration parameters

- `calibration_seconds`: Neutral-face calibration time at the beginning of live monitoring.
- `calibration_min_samples`: Minimum number of valid face samples needed before calibration completes.
- `neutral_anchor_score`: After calibration, the neutral face is shifted near this pain score.

### Logging and patient parameters

- `save_live_data`: If `true`, session CSV logs are saved.
- `log_every_n_frames`: Writes one log row every N frames.
- `patient_id`: Patient identifier stored in logs.

### Audio and wheeze parameters

- `enable_audio_monitoring`: Turns respiratory and wheeze analysis on or off.
- `audio_window_seconds`: Audio window size used for wheeze feature extraction.
- `audio_sample_rate`: Microphone sample rate for live audio capture.
- `audio_channels`: Number of microphone channels.

### Training parameters

- `dataset_augmentation_factor`: How much synthetic expansion is applied in `prepare-dataset`.
- `training_ridge_alpha`: Regularization strength used while training the multimodal ridge model.

### Camera overlay parameters

- `overlay_scale`: Makes the camera information panel smaller or larger.
- `overlay_anchor`: Places the panel in the `top_left` or `top_right` corner.

## Why command output is JSON

Many commands print JSON because JSON is structured and easy to:

- read by humans,
- save to a file,
- pass to dashboards or APIs,
- use later in automation.

## What the JSON outputs represent

### `prepare-dataset`

Example meanings:

- `output_csv`: New merged training CSV path.
- `input_files`: Source CSV files used.
- `base_rows`: Original row count before augmentation.
- `prepared_rows`: Final row count after augmentation.
- `augment_factor`: Synthetic expansion amount used.
- `pain_labels_present`: How many rows have pain labels.
- `wheeze_labels_present`: How many rows have wheeze labels.

### `train`

- `source_csv`: Training CSV used.
- `model_path`: Saved model file path.
- `rows_used`: Number of rows used for training.
- `mae`: Mean absolute error for pain prediction on the training CSV.
- `rmse`: Root mean squared error for pain prediction on the training CSV.
- `wheeze_mae`: Mean absolute error for wheeze prediction if wheeze labels exist.
- `feature_columns`: Feature names used by the model.
- `design_feature_count`: Number of expanded model inputs after polynomial and interaction features.

### `evaluate`

- `model_path`: Model file that was loaded.
- `csv_path`: CSV used for evaluation.
- `rows_evaluated`: Number of rows checked.
- `mae`: Average absolute pain prediction error.
- `rmse`: Average squared pain error summarized as RMSE.
- `within_1_point_percent`: Percent of pain predictions within 1 point of the label.
- `within_2_points_percent`: Percent of pain predictions within 2 points of the label.
- `wheeze_mae`: Average wheeze prediction error if wheeze labels are present.

### `live`

- `frames_processed`: Number of frames processed.
- `elapsed_seconds`: Total runtime in seconds.
- `fps`: Effective processing speed.
- `total_pain_duration_seconds`: Total time classified as pain-active.
- `latest_wheeze_probability`: Final smoothed wheeze estimate.
- `log_file`: Session CSV path.
- `episode_event_file`: Episode start/end CSV path.
- `calibration_done`: Whether neutral calibration completed.
- `baseline_score`: Learned neutral baseline score.

## Which parameters to adjust first

If the display feels unstable:

- Lower `pain_display_alpha_change`
- Lower `wheeze_display_alpha`
- Increase `expression_change_threshold`

If the overlay hides too much of the face:

- Lower `overlay_scale`
- Change `overlay_anchor` to the other corner

If pain episodes start too easily:

- Increase `pain_start_threshold`
- Increase `start_hold_seconds`

If pain episodes end too slowly:

- Increase `pain_end_threshold`
- Lower `end_hold_seconds`
