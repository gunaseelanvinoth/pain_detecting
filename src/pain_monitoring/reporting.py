from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_session_csv(session_csv_path: Path, summary_out: Path | None = None) -> dict:
    frame = pd.read_csv(session_csv_path)
    if frame.empty:
        summary = {
            "session_csv": str(session_csv_path),
            "rows": 0,
            "total_pain_duration_s": 0.0,
            "mean_pain_score": 0.0,
            "max_pain_score": 0.0,
            "episodes_detected": 0,
        }
        return summary

    frame["pain_score_0_10"] = pd.to_numeric(frame.get("pain_score_0_10"), errors="coerce")
    frame["total_pain_duration_s"] = pd.to_numeric(frame.get("total_pain_duration_s"), errors="coerce")

    total_pain_duration = float(frame["total_pain_duration_s"].dropna().iloc[-1]) if not frame["total_pain_duration_s"].dropna().empty else 0.0
    mean_pain = float(frame["pain_score_0_10"].mean()) if "pain_score_0_10" in frame else 0.0
    max_pain = float(frame["pain_score_0_10"].max()) if "pain_score_0_10" in frame else 0.0

    episodes = 0
    if "active_episode_id" in frame.columns:
        episodes = int(frame["active_episode_id"].dropna().nunique())

    summary = {
        "session_csv": str(session_csv_path),
        "rows": int(frame.shape[0]),
        "total_pain_duration_s": total_pain_duration,
        "mean_pain_score": mean_pain,
        "max_pain_score": max_pain,
        "episodes_detected": episodes,
    }

    if summary_out is not None:
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(summary_out, index=False)

    return summary
