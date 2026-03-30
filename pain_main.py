import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pain_monitoring.config import PainMonitoringConfig
from pain_monitoring.runner import (
    evaluate_from_csv,
    extract_features_from_video,
    run_live_monitor,
    summarize_session,
    train_from_labeled_csv,
)


def _load_config(config_path: str | None) -> PainMonitoringConfig:
    return PainMonitoringConfig.from_json(Path(config_path)) if config_path else PainMonitoringConfig()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient Pain Detection and Duration Monitoring")
    parser.add_argument("--config", default=None, help="Optional JSON config file")

    subparsers = parser.add_subparsers(dest="command")

    live_parser = subparsers.add_parser("live", help="Run live camera pain monitoring")
    live_parser.add_argument("--model", default=None, help="Optional model json path")
    live_parser.add_argument("--video", default=None, help="Optional video file path instead of camera")

    extract_parser = subparsers.add_parser("extract-features", help="Extract feature CSV from a video")
    extract_parser.add_argument("--video", required=True, help="Video path")
    extract_parser.add_argument("--out-csv", required=True, help="Output feature CSV path")
    extract_parser.add_argument("--sample-every", type=int, default=3, help="Sample every N frames")
    extract_parser.add_argument("--fixed-label", type=float, default=None, help="Optional fixed pain label for all rows")

    train_parser = subparsers.add_parser("train", help="Train from labeled CSV")
    train_parser.add_argument("--csv", required=True, help="Labeled training CSV path")
    train_parser.add_argument("--model-out", default="artifacts/pain_model.json", help="Output model path")
    train_parser.add_argument("--metrics-out", default="artifacts/pain_training_metrics.json", help="Training metrics output path")

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
            config=config,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "extract-features":
        payload = extract_features_from_video(
            video_path=Path(args.video),
            output_csv_path=Path(args.out_csv),
            config=config,
            sample_every_n_frames=args.sample_every,
            fixed_label=args.fixed_label,
        )
        print(json.dumps(payload, indent=2))
    elif args.command == "train":
        metrics = train_from_labeled_csv(
            csv_path=Path(args.csv),
            model_out=Path(args.model_out),
            metrics_out=Path(args.metrics_out),
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "evaluate":
        metrics = evaluate_from_csv(model_path=Path(args.model), csv_path=Path(args.csv))
        print(json.dumps(metrics, indent=2))
    elif args.command == "summarize-session":
        payload = summarize_session(session_csv=Path(args.csv), summary_out=Path(args.out) if args.out else None)
        print(json.dumps(payload, indent=2))
