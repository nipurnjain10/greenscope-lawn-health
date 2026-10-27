"""Small I/O helpers shared by scripts and the application."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert NumPy and pandas scalars so they can be written as JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def assert_no_leakage(columns: list[str]) -> None:
    leaked = [column for column in columns if column in LEAKAGE_COLUMNS or column == TARGET_COLUMN]
    if leaked:
        raise AssertionError(f"Target leakage columns are present in model inputs: {leaked}")


def feature_frame(records: dict | pd.DataFrame) -> pd.DataFrame:
    """Build a one-row or multi-row frame containing exactly the model features."""
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame([records])
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    model_frame = frame.loc[:, FEATURE_COLUMNS].copy()
    assert_no_leakage(list(model_frame.columns))
    return model_frame


def package_versions() -> dict[str, str]:
    import importlib.metadata as metadata

    names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "streamlit",
        "plotly",
        "joblib",
        "matplotlib",
        "python-docx",
        "python-pptx",
        "pytest",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions
