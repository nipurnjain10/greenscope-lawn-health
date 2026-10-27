"""Metric calculations used for the held-out evaluation.

Nothing in this module fits a model or chooses a threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import CLASS_ORDER


def _as_array(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def score_predictions(y_true, y_pred) -> dict:
    truth = _as_array(y_true)
    pred = _as_array(y_pred)
    matrix = confusion_matrix(truth, pred, labels=CLASS_ORDER)
    per_class = {}
    precision = precision_score(truth, pred, labels=CLASS_ORDER, average=None, zero_division=0)
    recall = recall_score(truth, pred, labels=CLASS_ORDER, average=None, zero_division=0)
    f1 = f1_score(truth, pred, labels=CLASS_ORDER, average=None, zero_division=0)
    supports = matrix.sum(axis=1)
    for index, label in enumerate(CLASS_ORDER):
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(supports[index]),
        }
    high_index = CLASS_ORDER.index("High")
    low_index = CLASS_ORDER.index("Low")
    high_support = int(supports[high_index])
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_precision": float(precision_score(truth, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, pred, labels=CLASS_ORDER, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "class_order": list(CLASS_ORDER),
        "support_total": int(len(truth)),
        "high_risk_predicted_low": int(matrix[high_index, low_index]),
        "high_risk_missed": int(high_support - matrix[high_index, high_index]),
        "high_risk_support": high_support,
    }


def bootstrap_intervals(y_true, y_pred, *, resamples: int = 400, seed: int = 42) -> dict:
    """Resample fixed predictions. Models are not refit."""
    truth = _as_array(y_true)
    pred = _as_array(y_pred)
    rng = np.random.default_rng(seed)
    accuracies = []
    f1_scores = []
    n_rows = len(truth)
    for _ in range(resamples):
        index = rng.integers(0, n_rows, n_rows)
        accuracies.append(float(accuracy_score(truth[index], pred[index])))
        f1_scores.append(
            float(f1_score(truth[index], pred[index], labels=CLASS_ORDER, average="macro", zero_division=0))
        )
    return {
        "resamples": resamples,
        "accuracy_ci95": [float(np.quantile(accuracies, 0.025)), float(np.quantile(accuracies, 0.975))],
        "macro_f1_ci95": [float(np.quantile(f1_scores, 0.025)), float(np.quantile(f1_scores, 0.975))],
        "interpretation": (
            "Percentile interval from resampling the stored test-set predictions. "
            "The models are not refitted. The interval describes sampling variability "
            "of this synthetic test estimate, not uncertainty on real lawns."
        ),
    }


def calibration_report(y_true, probabilities: np.ndarray, *, n_bins: int = 10) -> dict:
    """One-versus-rest reliability bins and a multiclass Brier score.

    This is an evaluation of the probabilities already produced. It does not
    fit a calibrator, and it must not be used to change the reported classes.
    """
    truth = _as_array(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    one_hot = np.zeros_like(probabilities)
    report = {}
    ece_values = []
    for index, label in enumerate(CLASS_ORDER):
        observed = (truth == label).astype(float)
        one_hot[:, index] = observed
        scores = probabilities[:, index]
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = []
        ece = 0.0
        for bin_index in range(n_bins):
            lower, upper = float(edges[bin_index]), float(edges[bin_index + 1])
            if bin_index == n_bins - 1:
                mask = (scores >= lower) & (scores <= upper)
            else:
                mask = (scores >= lower) & (scores < upper)
            count = int(mask.sum())
            if count == 0:
                continue
            mean_predicted = float(scores[mask].mean())
            observed_frequency = float(observed[mask].mean())
            ece += (count / len(scores)) * abs(observed_frequency - mean_predicted)
            bins.append(
                {
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": count,
                    "mean_predicted": mean_predicted,
                    "observed_frequency": observed_frequency,
                }
            )
        report[label] = {"bins": bins, "ece": float(ece)}
        ece_values.append(float(ece))
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    return {
        "method": "One-versus-rest equal-width bins on the evaluation predictions. No calibrator was fit on these rows.",
        "n_bins": n_bins,
        "one_vs_rest": report,
        "macro_ece": float(np.mean(ece_values)),
        "multiclass_brier": brier,
    }


def metrics_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, payload in metrics["models"].items():
        for split_name in ("validation_before_refit", "test", "robustness"):
            block = payload.get(split_name)
            if not block:
                continue
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": block["accuracy"],
                    "macro_precision": block["macro_precision"],
                    "macro_recall": block["macro_recall"],
                    "macro_f1": block["macro_f1"],
                    "weighted_f1": block["weighted_f1"],
                    "high_risk_missed": block["high_risk_missed"],
                    "high_risk_predicted_low": block["high_risk_predicted_low"],
                    "support_total": block["support_total"],
                }
            )
    return pd.DataFrame(rows)


def disagreement_summary(prediction_frame: pd.DataFrame, model_columns: list[str]) -> dict:
    votes = prediction_frame.loc[:, model_columns].astype(str)
    n_unique = votes.nunique(axis=1)
    unanimous = int((n_unique == 1).sum())
    majority = 0
    split = 0
    for _, row in votes.iterrows():
        counts = row.value_counts()
        if counts.iloc[0] == len(model_columns):
            continue
        if counts.iloc[0] >= 3:
            majority += 1
        else:
            split += 1
    n_rows = int(len(votes))
    return {
        "n": n_rows,
        "unanimous": unanimous,
        "majority_with_dissent": majority,
        "split": split,
        "unanimous_fraction": unanimous / n_rows if n_rows else None,
        "model_columns": model_columns,
    }
