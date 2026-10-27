"""Soft voting is the mean of aligned probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

import src.ensemble as ensemble_module
from src.ensemble import soft_voting_predict, soft_voting_proba


class _Stub:
    def __init__(self, probabilities, order):
        self._probabilities = np.asarray(probabilities)
        self.classes_ = np.array(order)

    def predict_proba(self, features):
        return self._probabilities

    def predict(self, features):
        index = np.argmax(self.predict_proba(features), axis=1)
        return self.classes_[index]


def test_soft_voting_averages_and_selects_the_largest_mean():
    frame = pd.DataFrame({"x": [0, 1]})
    # Deliberately different class orders.
    first = _Stub([[0.2, 0.5, 0.3], [0.6, 0.2, 0.2]], ["Low", "Moderate", "High"])
    second = _Stub([[0.1, 0.7, 0.2], [0.2, 0.2, 0.6]], ["High", "Low", "Moderate"])
    models = {"first": first, "second": second}
    averaged = soft_voting_proba(models, frame)
    # Row 0, class order Low, Moderate, High.
    # first: 0.2, 0.5, 0.3
    # second aligned from High, Low, Moderate -> Low 0.7, Moderate 0.2, High 0.1
    expected_row = np.array([(0.2 + 0.7) / 2, (0.5 + 0.2) / 2, (0.3 + 0.1) / 2])
    np.testing.assert_allclose(averaged[0], expected_row)
    np.testing.assert_allclose(averaged.sum(axis=1), 1.0)
    assert soft_voting_predict(models, frame)[0] == "Low"


def test_soft_voting_is_the_only_ensemble_rule():
    assert not hasattr(ensemble_module, "hard_voting_predict")
