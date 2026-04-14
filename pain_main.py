import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.kaggle_import import import_kaggle_face_dataset, import_kaggle_respiratory_dataset
from pain_monitoring.runner import (
    build_training_dataset,
    evaluate_from_csv,
    extract_features_from_video,
    run_live_monitor,
    summarize_session,
    train_from_labeled_csv,
)


def _load_config(config_path: str | None) -> PainMonitoringConfig:
    return PainMonitoringConfig.from_json(Path(config_path)) if config_path else PainMonitoringConfig()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient Pain Detection and Wheeze Monitoring")
    parser.add_argument("--config", default=None, help="Optional JSON config file")

    subparsers = parser.add_subparsers(dest="command")

    live_parser = subparsers.add_parser("live", help="Run live camera pain monitoring")
    live_parser.add_argument("--model", default=None, help="Optional model json path")
    live_parser.add_argument("--video", default=None, help="Optional video file path instead of camera")
    live_parser.add_argument("--audio", default=None, help="Optional WAV audio file for wheeze monitoring")

    extract_parser = subparsers.add_parser("extract-features", help="Extract multimodal feature CSV from a video")
    extract_parser.add_argument("--video", required=True, help="Video path")
    extract_parser.add_argument("--audio", default=None, help="Optional WAV audio path aligned to the video")
    extract_parser.add_argument("--out-csv", required=True, help="Output feature CSV path")
    extract_parser.add_argument("--sample-every", type=int, default=3, help="Sample every N frames")
    extract_parser.add_argument("--fixed-label", type=float, default=None, help="Optional fixed pain label for all rows")
    extract_parser.add_argument("--fixed-wheeze-label", type=float, default=None, help="Optional fixed wheeze label 0..1")
    extract_parser.add_argument("--no-preview", action="store_true", help="Disable extraction preview window")

    prepare_parser = subparsers.add_parser("prepare-dataset", help="Merge and augment labeled CSVs for training")
    prepare_parser.add_argument("--csv", nargs="+", required=True, help="One or more labeled CSV files")
    prepare_parser.add_argument("--out-csv", required=True, help="Prepared dataset output path")
    prepare_parser.add_argument("--augment-factor", type=int, default=None, help="Synthetic augmentation multiplier")

    kaggle_resp_parser = subparsers.add_parser("import-kaggle-respiratory", help="Convert a Kaggle respiratory dataset into wheeze training CSV")
    kaggle_resp_parser.add_argument("--dataset-dir", required=True, help="Folder containing Kaggle respiratory WAV and TXT files")
    kaggle_resp_parser.add_argument("--out-csv", required=True, help="Output CSV path")

    kaggle_face_parser = subparsers.add_parser("import-kaggle-face", help="Convert a Kaggle face dataset into pain training CSV")
    kaggle_face_parser.add_argument("--dataset-dir", required=True, help="Folder containing labeled face images")
    kaggle_face_parser.add_argument("--out-csv", required=True, help="Output CSV path")
    kaggle_face_parser.add_argument("--default-label", type=float, default=None, help="Fallback pain label if folder names do not contain pain keywords")

    train_parser = subparsers.add_parser("train", help="Train from labeled CSV")
    train_parser.add_argument("--csv", required=True, help="Labeled training CSV path")
    train_parser.add_argument("--model-out", default="artifacts/pain_model.json", help="Output model path")
    train_parser.add_argument("--metrics-out", default="artifacts/pain_training_metrics.json", help="Training metrics output path")
    train_parser.add_argument("--ridge-alpha", type=float, default=None, help="Optional ridge regularization strength")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model on labeled CSV")
    eval_parser.add_argument("--model", required=True, help="Trained model path")
    eval_parser.add_argument("--csv", required=True, help="Labeled evaluation CSV path")

    summary_parser = subparsers.add_parser("summarize-session", help="Create summary from a session CSV")
    summary_parser.add_argument("--csv", required=True, help="Session CSV path")
    summary_parser.add_argument("--out", default=None, help="Optional summary CSV output path")

    args = parser.parse_args()
    config = _load_config(args.config)

    if args.command in {None, "live"}:
        summary = run_live_monitor(
            model_path=Path(args.model) if args.model else None,
            video_path=Path(args.video) if args.video else None,
            audio_path=Path(args.audio) if args.audio else None,
            config=config,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "extract-features":
        payload = extract_features_from_video(
            video_path=Path(args.video),
            audio_path=Path(args.audio) if args.audio else None,
            output_csv_path=Path(args.out_csv),
            config=config,
            sample_every_n_frames=args.sample_every,
            fixed_label=args.fixed_label,
            fixed_wheeze_label=args.fixed_wheeze_label,
            show_preview=not args.no_preview,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "prepare-dataset":
        payload = build_training_dataset(
            csv_paths=[Path(item) for item in args.csv],
            output_csv=Path(args.out_csv),
            config=config,
            augment_factor=args.augment_factor,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "import-kaggle-respiratory":
        payload = import_kaggle_respiratory_dataset(
            dataset_dir=Path(args.dataset_dir),
            output_csv_path=Path(args.out_csv),
            config=config,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "import-kaggle-face":
        payload = import_kaggle_face_dataset(
            dataset_dir=Path(args.dataset_dir),
            output_csv_path=Path(args.out_csv),
            config=config,
            default_label=args.default_label,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "train":
        metrics = train_from_labeled_csv(
            csv_path=Path(args.csv),
            model_out=Path(args.model_out),
            metrics_out=Path(args.metrics_out),
            ridge_alpha=args.ridge_alpha if args.ridge_alpha is not None else config.training_ridge_alpha,
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "evaluate":
        metrics = evaluate_from_csv(model_path=Path(args.model), csv_path=Path(args.csv))
        print(json.dumps(metrics, indent=2))
    elif args.command == "summarize-session":
        payload = summarize_session(session_csv=Path(args.csv), summary_out=Path(args.out) if args.out else None)
        print(json.dumps(payload, indent=2))
