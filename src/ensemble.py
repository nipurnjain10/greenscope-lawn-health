"""Equal-weight soft voting over the four saved pipelines.

Soft voting is the mean of the four class-probability vectors after the columns
are aligned to Low, Moderate, High. Weights were not tuned on the test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config import CLASS_ORDER


def align_proba(model, probabilities: np.ndarray) -> np.ndarray:
    order = [str(label) for label in model.classes_]
    missing = [label for label in CLASS_ORDER if label not in order]
    if missing:
        raise ValueError(f"Model is missing classes {missing}; it has {order}.")
    index = [order.index(label) for label in CLASS_ORDER]
    return np.asarray(probabilities)[:, index]


def soft_voting_proba(models: dict, features: pd.DataFrame) -> np.ndarray:
    if len(models) < 2:
        raise ValueError("Soft voting needs at least two models.")
    aligned = []
    for model in models.values():
        aligned.append(align_proba(model, model.predict_proba(features)))
    mean = np.mean(np.stack(aligned, axis=0), axis=0)
    # Guard against tiny numerical drift.
    total = mean.sum(axis=1, keepdims=True)
    return mean / total


def soft_voting_predict(models: dict, features: pd.DataFrame) -> np.ndarray:
    probabilities = soft_voting_proba(models, features)
    return np.array(CLASS_ORDER, dtype=object)[np.argmax(probabilities, axis=1)]


class SoftVotingModel(ClassifierMixin, BaseEstimator):
    """Thin estimator so permutation importance can call the equal-weight average.

    ``fit`` does not refit the base models. They must already be fitted.
    """

    def __init__(self, models=None):
        self.models = models

    def fit(self, features=None, target=None):
        self.classes_ = np.array(CLASS_ORDER)
        return self

    def predict(self, features):
        return soft_voting_predict(self.models, features)

    def predict_proba(self, features):
        return soft_voting_proba(self.models, features)


def leave_one_out_soft_predict(models: dict, features: pd.DataFrame) -> dict[str, np.ndarray]:
    """Soft vote after dropping one base model. The remaining models are not retrained."""
    predictions = {}
    for held_out in models:
        subset = {name: model for name, model in models.items() if name != held_out}
        predictions[held_out] = soft_voting_predict(subset, features)
    return predictions
