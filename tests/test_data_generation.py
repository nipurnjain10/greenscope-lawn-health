"""Generator reproducibility and schema checks."""

from __future__ import annotations

import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    N_MAIN,
    SEED_MAIN,
    SEED_ROBUSTNESS,
    TARGET_COLUMN,
)
from src.synthetic_data import generate_inspections


def test_generation_is_reproducible():
    first, _ = generate_inspections(400, seed=7, scenario="main")
    second, _ = generate_inspections(400, seed=7, scenario="main")
    pd.testing.assert_frame_equal(first, second)


def test_identifiers_are_unique_and_target_is_not_a_feature():
    frame, _ = generate_inspections(400, seed=7, scenario="main")
    assert frame["inspection_id"].is_unique
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert not set(LEAKAGE_COLUMNS).intersection(FEATURE_COLUMNS)
    for column in LEAKAGE_COLUMNS:
        if column == TARGET_COLUMN:
            continue
        assert column not in frame.columns


def test_robustness_uses_frozen_thresholds_and_distinct_ids():
    primary, summary = generate_inspections(500, seed=SEED_MAIN, scenario="main")
    cuts = (summary["thresholds"]["low_lt"], summary["thresholds"]["high_ge"])
    robust, _ = generate_inspections(300, seed=SEED_ROBUSTNESS, scenario="robustness", thresholds=cuts)
    assert set(primary["inspection_id"]).isdisjoint(set(robust["inspection_id"]))
    assert robust["inspection_id"].str.startswith("RB-").all()


def test_saved_primary_file_has_the_required_row_count():
    frame = pd.read_csv("data/synthetic_lawn_inspections.csv")
    assert len(frame) == N_MAIN
    assert frame["inspection_id"].is_unique
    assert set(frame[TARGET_COLUMN]) <= {"Low", "Moderate", "High"}
