"""Permutation importance and local baseline explanations.

Both methods describe how the fitted models use the supplied features.
They do not establish that a feature caused a lawn to be unhealthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import CATEGORICAL_FEATURES, CLASS_ORDER, FEATURE_COLUMNS, NUMERIC_FEATURES, SPLIT_SEED
from src.ensemble import soft_voting_proba


def training_baselines(fitting_frame: pd.DataFrame) -> dict:
    """Medians and modes from the rows the final models were fit on."""
    baselines = {}
    for column in NUMERIC_FEATURES:
        baselines[column] = float(pd.to_numeric(fitting_frame[column], errors="coerce").median())
    for column in CATEGORICAL_FEATURES:
        modes = fitting_frame[column].astype(str).mode()
        baselines[column] = str(modes.iloc[0])
    baselines["source"] = "median or mode of the train-plus-validation fitting partition"
    return baselines


def permutation_table(models: dict, features: pd.DataFrame, target: pd.Series, *, repeats: int = 5) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        result = permutation_importance(
            model,
            features,
            target,
            n_repeats=repeats,
            random_state=SPLIT_SEED,
            scoring="accuracy",
            n_jobs=1,
        )
        for index, column in enumerate(features.columns):
            rows.append(
                {
                    "model": name,
                    "feature": column,
                    "importance_mean": float(result.importances_mean[index]),
                    "importance_std": float(result.importances_std[index]),
                }
            )
    table = pd.DataFrame(rows)
    return table.sort_values(["model", "importance_mean"], ascending=[True, False]).reset_index(drop=True)


def local_baseline_contributions(models: dict, row: pd.DataFrame, baselines: dict, predicted_class: str) -> list[dict]:
    """Change in soft-voting probability when one observed value is replaced by the fitting-set baseline."""
    if predicted_class not in CLASS_ORDER:
        raise ValueError(predicted_class)
    class_index = CLASS_ORDER.index(predicted_class)
    base_probability = float(soft_voting_proba(models, row)[0, class_index])
    contributions = []
    for feature in FEATURE_COLUMNS:
        observed = row.iloc[0][feature]
        if pd.isna(observed):
            contributions.append(
                {
                    "feature": feature,
                    "status": "missing",
                    "observed": None,
                    "baseline": baselines.get(feature),
                    "contribution": None,
                    "note": "No leave-one-out contrast is shown because the measurement was not supplied. The model imputed it.",
                }
            )
            continue
        modified = row.copy()
        modified.iloc[0, modified.columns.get_loc(feature)] = baselines[feature]
        replaced = float(soft_voting_proba(models, modified)[0, class_index])
        observed_value = observed.item() if hasattr(observed, "item") else observed
        if isinstance(observed_value, (np.floating, float)):
            observed_value = float(observed_value)
        elif isinstance(observed_value, (np.integer, int)):
            observed_value = int(observed_value)
        else:
            observed_value = str(observed_value)
        contributions.append(
            {
                "feature": feature,
                "status": "measured",
                "observed": observed_value,
                "baseline": baselines.get(feature),
                "contribution": float(base_probability - replaced),
                "note": (
                    "Positive means the observed value raises the soft-voting probability of the predicted class "
                    "relative to the fitting-set baseline."
                ),
            }
        )
    contributions.sort(key=lambda item: abs(item["contribution"] or 0.0), reverse=True)
    return contributions
