"""Preprocessing, splits, and a small end-to-end fit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLASS_ORDER, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_COLUMN
from src.ensemble import soft_voting_proba
from src.preprocessing import build_estimators
from src.splits import make_split_table, subset_by_split
from src.synthetic_data import generate_inspections
from src.utilities import feature_frame


def _sample():
    frame, _ = generate_inspections(360, seed=19, scenario="main")
    splits = make_split_table(frame, seed=19)
    return frame, splits


def test_splits_do_not_overlap():
    frame, splits = _sample()
    groups = [set(splits.loc[splits["split"] == name, "inspection_id"]) for name in ("train", "validation", "test")]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert set().union(*groups) == set(frame["inspection_id"])


def test_feature_frame_drops_the_target():
    frame, _ = _sample()
    model_input = feature_frame(frame)
    assert list(model_input.columns) == FEATURE_COLUMNS
    assert TARGET_COLUMN not in model_input.columns


def test_imputer_learns_training_medians_only():
    frame, splits = _sample()
    train = subset_by_split(frame, splits, "train")
    validation = subset_by_split(frame, splits, "validation")
    model = build_estimators()["logistic_regression"]
    model.fit(feature_frame(train), train[TARGET_COLUMN].astype(str))
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    training_median = feature_frame(train)[NUMERIC_FEATURES].median().to_numpy()
    combined_median = feature_frame(pd.concat([train, validation]))[NUMERIC_FEATURES].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, training_median)
    assert not np.allclose(training_median, combined_median)


def test_each_model_predicts_probabilities_that_sum_to_one():
    frame, splits = _sample()
    train = subset_by_split(frame, splits, "train")
    test = subset_by_split(frame, splits, "test")
    models = build_estimators()
    features = feature_frame(train)
    target = train[TARGET_COLUMN].astype(str)
    held_out = feature_frame(test)
    for name, model in models.items():
        model.fit(features, target)
        probabilities = model.predict_proba(held_out)
        assert probabilities.shape[1] == len(CLASS_ORDER)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert set(model.classes_) == set(CLASS_ORDER)
    averaged = soft_voting_proba(models, held_out)
    np.testing.assert_allclose(averaged.sum(axis=1), 1.0, atol=1e-6)
