from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pain_monitoring.model import FEATURE_COLUMNS, TARGET_COLUMN, WHEEZE_TARGET_COLUMN


DEFAULT_SEED = 42


def _ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = 0.0
    if TARGET_COLUMN not in enriched.columns:
        enriched[TARGET_COLUMN] = np.nan
    if WHEEZE_TARGET_COLUMN not in enriched.columns:
        enriched[WHEEZE_TARGET_COLUMN] = np.nan
    return enriched


def _augment_rows(frame: pd.DataFrame, augment_factor: int, seed: int) -> pd.DataFrame:
    if augment_factor <= 0 or frame.empty:
        return frame

    rng = np.random.default_rng(seed)
    base = _ensure_columns(frame)
    augmented_rows: list[dict] = []
    rows = base.to_dict(orient="records")
    for _ in range(augment_factor):
        for idx, row in enumerate(rows):
            peer = rows[(idx + 1) % len(rows)]
            blend = float(rng.uniform(0.15, 0.85))
            new_row = dict(row)
            for column in FEATURE_COLUMNS:
                source = float(row.get(column, 0.0) or 0.0)
                target = float(peer.get(column, source) or 0.0)
                jitter = float(rng.normal(0.0, 0.03))
                new_row[column] = float(np.clip(source * (1.0 - blend) + target * blend + jitter, 0.0, 1.0))

            pain_source = row.get(TARGET_COLUMN)
            pain_target = peer.get(TARGET_COLUMN)
            if pd.notna(pain_source) and pd.notna(pain_target):
                new_row[TARGET_COLUMN] = float(
                    np.clip(float(pain_source) * (1.0 - blend) + float(pain_target) * blend + rng.normal(0.0, 0.2), 0.0, 10.0)
                )
            else:
                new_row[TARGET_COLUMN] = np.nan

            wheeze_value = row.get(WHEEZE_TARGET_COLUMN)
            peer_wheeze = peer.get(WHEEZE_TARGET_COLUMN)
            if pd.notna(wheeze_value) and pd.notna(peer_wheeze):
                mixed = float(wheeze_value) * (1.0 - blend) + float(peer_wheeze) * blend + rng.normal(0.0, 0.04)
                new_row[WHEEZE_TARGET_COLUMN] = float(np.clip(mixed, 0.0, 1.0))

            new_row["row_origin"] = "synthetic_augmented"
            augmented_rows.append(new_row)

    return pd.concat([base, pd.DataFrame(augmented_rows)], ignore_index=True)


def prepare_training_dataset(
    csv_paths: list[Path],
    output_csv_path: Path,
    augment_factor: int = 20,
    seed: int = DEFAULT_SEED,
) -> dict:
    if not csv_paths:
        raise ValueError("At least one CSV path is required.")

    frames = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frame["row_origin"] = frame.get("row_origin", "labeled")
        frames.append(_ensure_columns(frame))

    merged = pd.concat(frames, ignore_index=True)
    if not merged[TARGET_COLUMN].notna().any() and not merged[WHEEZE_TARGET_COLUMN].notna().any():
        raise ValueError(f"Training CSVs must include {TARGET_COLUMN} or {WHEEZE_TARGET_COLUMN}.")

    prepared = _augment_rows(merged, augment_factor=augment_factor, seed=seed)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_csv_path, index=False)

    return {
        "output_csv": str(output_csv_path),
        "input_files": [str(path) for path in csv_paths],
        "base_rows": int(merged.shape[0]),
        "prepared_rows": int(prepared.shape[0]),
        "augment_factor": int(augment_factor),
        "pain_labels_present": int(prepared[TARGET_COLUMN].notna().sum()),
        "wheeze_labels_present": int(prepared[WHEEZE_TARGET_COLUMN].notna().sum()),
    }
