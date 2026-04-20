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
- `pain_expression_boost`: Extra pain score added when real micro-expression changes are detected.
- `micro_expression_trigger_threshold`: Minimum expression-change signal required before the extra pain boost is applied.
- `enable_eyebrow_landmarks`: Turns MediaPipe FaceMesh eyebrow landmark detection on or off.
- `eyebrow_distance_pain_threshold`: Inner-eyebrow distance ratio below this value is treated as eyebrow contraction pain when no personal baseline is available.
- `eyebrow_contraction_drop_threshold`: Relative drop from the calibrated neutral eyebrow distance needed for full eyebrow-contraction confidence.
- `eyebrow_angle_pain_threshold`: Inner-brow downward angle/lowering needed for full eyebrow angle confidence. This helps catch furrowed brows, not only brows that move closer together.
- `eyebrow_pain_confidence_threshold`: Minimum eyebrow confidence required before eyebrow contraction can affect the final pain decision.
- `eyebrow_pain_score_boost`: Extra pain score added when eyebrow landmarks show contracted brows.
- `brow_edge_pain_boost`: Extra eyebrow and face-edge emphasis used to improve pain sensitivity from brow tightening and edge changes.
- `calm_brow_threshold`: Brow signal level treated as calm so eyebrow boosting does not fire unnecessarily.
- `calm_edge_threshold`: Face-edge and nose-contrast level treated as calm so edge boosting does not fire unnecessarily.
- `sustained_pain_signal_threshold`: Minimum steady eyebrow, mouth, and nose evidence needed before held-expression confidence starts building.
- `sustained_pain_boost`: Maximum extra score added when a pain-like expression is held long enough.
- `sustained_pain_seconds_to_full_boost`: Number of seconds a pain-like expression should be held before the full sustained-expression boost is available.
- `neutral_expression_signal_threshold`: Maximum calm-face evidence allowed before the normal-face guard stops applying.
- `neutral_motion_threshold`: Maximum movement allowed for a face to be treated as normal/calm.
- `neutral_face_score_cap`: Highest pain score allowed when the face looks calm and neutral.
- `neutral_relative_evidence_margin`: Extra facial evidence above the calibrated neutral face required before calm-face suppression stops applying.
- `minimum_pain_regions_required`: Minimum number of active facial regions required before the score can rise freely into pain-detected range.
- `single_region_score_cap`: Highest score allowed when only one region, such as eyebrow-only activity, is active.

### Calibration parameters

- `calibration_seconds`: Neutral-face calibration time at the beginning of live monitoring.
- `calibration_min_samples`: Minimum number of valid face samples needed before calibration completes.
- `neutral_anchor_score`: After calibration, the neutral face is shifted near this pain score.

### Logging and patient parameters

- `save_live_data`: If `true`, session CSV logs are saved.
- `log_every_n_frames`: Writes one log row every N frames.
- `patient_id`: Patient identifier stored in logs.
- `notification_cooldown_seconds`: Minimum wait time between repeated email alerts.

### Audio and wheeze parameters

- `enable_audio_monitoring`: Turns respiratory and wheeze analysis on or off.
- `audio_window_seconds`: Audio window size used for wheeze feature extraction.
- `audio_sample_rate`: Microphone sample rate for live audio capture.
- `audio_channels`: Number of microphone channels.
- `wheeze_support_boost`: Extra support added when respiratory motion and wheeze-band audio agree with the model.
- `sustained_wheeze_signal_threshold`: Minimum steady wheeze evidence needed before held/repeated wheeze confidence starts building.
- `sustained_wheeze_boost`: Maximum extra wheeze probability added when wheeze evidence is sustained.
- `sustained_wheeze_seconds_to_full_boost`: Number of seconds wheeze evidence should persist before the full sustained-wheeze boost is available.
- `wheeze_alert_threshold`: Wheeze probability level that can trigger a live wheeze alert.

### Training parameters

- `dataset_augmentation_factor`: How much synthetic expansion is applied in `prepare-dataset`.
- `training_ridge_alpha`: Regularization strength used while training the multimodal ridge model.

### Camera overlay parameters

- `overlay_scale`: Makes the camera information panel smaller or larger.
- `overlay_anchor`: Places the panel in the `top_left` or `top_right` corner.

### Email notification parameters

- `email_notifications_enabled`: Turns patient email notifications on or off.
- `notification_email_to`: One or more receiver email IDs separated by commas.
- `notification_email_from`: Gmail sender account used for SMTP delivery.
- `notification_email_password`: Gmail App Password for the sender account.
- `smtp_host`: SMTP server host. Gmail uses `smtp.gmail.com`.
- `smtp_port`: SMTP server port. Gmail TLS uses `587`.
- `smtp_use_tls`: Uses TLS encryption when `true`.
- `email_send_instant_alerts`: Sends live pain and wheeze alerts while monitoring.
- `email_send_session_report`: Sends the patient report and CSV files when the session ends.

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

Important face feature columns:

- `brow_position`: Tracks where strong eyebrow-region edges sit vertically, so lowered or tightened brows can influence the score.
- `brow_motion`: Tracks frame-to-frame eyebrow-region movement for small eyebrow position changes.
- `eyebrow_distance_ratio`: Landmark-based normalized gap between the inner eyebrows.
- `eyebrow_angle_score`: Landmark-based score for inner eyebrows lowering relative to the outer eyebrows.
- `eyebrow_contraction`: Landmark-based eyebrow contraction strength.
- `eyebrow_pain_confidence`: Confidence that contracted eyebrows indicate `Pain`.
- `mouth_micro_motion`: Tracks small mouth-region movement separately from larger lower-face motion.
- `nose_contrast`: Tracks nose-region contrast and edges to make nose tension/contrast changes more reliable.

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

## Why each command line is used

### `python pain_main.py live`

- Use this when you want real-time camera monitoring.
- It opens the webcam, extracts face and audio features, predicts pain and wheeze, shows the overlay, writes session logs, and can send patient email notifications.
- Output: JSON with runtime details such as processed frames, total pain duration, log file path, and session report summary if email reporting is enabled.

### `python pain_main.py extract-features --video ... --out-csv ...`

- Use this when you want to convert a recorded video into a labeled feature CSV for training or analysis.
- Output: JSON with the video path, output CSV path, row count, and label attachment status.

### `python pain_main.py prepare-dataset --csv ... --out-csv ...`

- Use this when you want to merge CSV files and increase the training rows using synthetic augmentation.
- Output: JSON with source files, base rows, prepared rows, and augmentation details.

### `python pain_main.py train --csv ... --model-out ...`

- Use this when you want to train or retrain the pain and wheeze model.
- Output: JSON with the saved model path and accuracy-style metrics such as `mae`, `rmse`, and `wheeze_mae`.

### `python pain_main.py evaluate --model ... --csv ...`

- Use this when you want to check how well a trained model performs on labeled data.
- Output: JSON with evaluation row count and error metrics such as `mae`, `rmse`, `within_1_point_percent`, and `within_2_points_percent`.

### `python pain_main.py summarize-session --csv ...`

- Use this when you want a short report from one patient session log.
- Output: JSON with rows logged, episodes detected, total pain duration, mean pain score, and max wheeze probability.
