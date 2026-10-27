"""Static figures for the report and the presentation.

The application draws its own charts from the same saved metrics. These files
exist so the documents can embed the experimental results.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASS_ORDER, FIGURE_DIR, TARGET_COLUMN
from src.utilities import ensure_dir

INK = "#1C2833"
MUTED = "#52606D"
GREEN = "#2F6F4E"
LINE = "#E4E0D8"
CLASS_COLORS = {"Low": "#2F6F4E", "Moderate": "#B5812F", "High": "#8C3A3A"}
MODEL_ORDER = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm_rbf",
    "soft_voting",
]
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "svm_rbf": "SVM (RBF)",
    "soft_voting": "Soft voting",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_f1": "Macro-F1",
    "weighted_f1": "Weighted-F1",
    "macro_precision": "Macro-precision",
    "macro_recall": "Macro-recall",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": LINE,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def _class_distribution(primary: pd.DataFrame, folder: Path) -> None:
    counts = primary[TARGET_COLUMN].astype(str).value_counts().reindex(CLASS_ORDER)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    bars = axis.bar(CLASS_ORDER, counts.to_numpy(), color=[CLASS_COLORS[label] for label in CLASS_ORDER], width=0.62)
    axis.set_ylabel("Inspections")
    axis.set_xlabel("Simulated lawn-health risk")
    axis.set_title("Primary dataset: simulated risk classes")
    for bar, count in zip(bars, counts.to_numpy()):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40, f"{int(count):,}", ha="center", color=INK)
    axis.set_ylim(0, max(counts.to_numpy()) * 1.15)
    _save(figure, folder / "fig_class_distribution.png")


def _model_comparison(metrics: dict, folder: Path) -> None:
    keys = list(METRIC_LABELS)
    figure, axis = plt.subplots(figsize=(11.2, 5.6))
    y = np.arange(len(MODEL_ORDER))
    height = 0.14
    colors = ["#1F4E79", "#2F6F4E", "#6B5B4B", "#3D6B8C", "#8C6A3A"]
    for index, key in enumerate(keys):
        values = [metrics["models"][name]["test"][key] for name in MODEL_ORDER]
        offset = (index - 2) * height
        axis.barh(y + offset, values, height=height, color=colors[index], label=METRIC_LABELS[key])
    axis.set_yticks(y)
    axis.set_yticklabels([MODEL_LABELS[name] for name in MODEL_ORDER])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Score on the held-out synthetic test set")
    axis.set_title("Test-set performance of the four models and soft voting")
    axis.legend(frameon=False, ncol=3, loc="lower right")
    _save(figure, folder / "fig_model_comparison.png")


def _confusion(metrics: dict, folder: Path) -> None:
    for name in MODEL_ORDER:
        matrix = np.array(metrics["models"][name]["test"]["confusion_matrix"], dtype=float)
        figure, axis = plt.subplots(figsize=(5.4, 4.6))
        image = axis.imshow(matrix, cmap="Greens", vmin=0)
        axis.set_xticks(range(3), CLASS_ORDER)
        axis.set_yticks(range(3), CLASS_ORDER)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Simulated reference class")
        axis.set_title(MODEL_LABELS[name])
        for row in range(3):
            for column in range(3):
                axis.text(column, row, f"{int(matrix[row, column])}", ha="center", va="center", color=INK)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        _save(figure, folder / f"fig_confusion_{name}.png")


def _per_class(metrics: dict, folder: Path) -> None:
    soft = metrics["models"]["soft_voting"]["test"]["per_class"]
    keys = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(CLASS_ORDER))
    width = 0.22
    colors = ["#1F4E79", "#2F6F4E", "#B5812F"]
    for index, key in enumerate(keys):
        values = [soft[label][key] for label in CLASS_ORDER]
        axis.bar(x + (index - 1) * width, values, width=width, color=colors[index], label=labels[index])
    axis.set_xticks(x, CLASS_ORDER)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xlabel("Simulated risk class")
    axis.set_title("Soft voting: class-wise test performance")
    axis.legend(frameon=False)
    _save(figure, folder / "fig_per_class_soft_voting.png")

    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(MODEL_ORDER))
    width = 0.22
    for index, label in enumerate(CLASS_ORDER):
        values = [metrics["models"][name]["test"]["per_class"][label]["recall"] for name in MODEL_ORDER]
        axis.bar(x + (index - 1) * width, values, width=width, color=CLASS_COLORS[label], label=f"{label} recall")
    axis.set_xticks(x, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recall")
    axis.set_title("Test recall by class and model")
    axis.legend(frameon=False, ncol=3)
    _save(figure, folder / "fig_recall_by_class.png")


def _importance(importance: pd.DataFrame, folder: Path) -> None:
    subset = (
        importance.loc[importance["model"] == "soft_voting"]
        .sort_values("importance_mean", ascending=True)
        .tail(12)
    )
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.barh(subset["feature"].str.replace("_", " "), subset["importance_mean"], color=GREEN, xerr=subset["importance_std"], ecolor=MUTED)
    axis.set_xlabel("Mean drop in test accuracy when the feature is shuffled")
    axis.set_title("Permutation importance for soft voting")
    _save(figure, folder / "fig_feature_importance.png")


def _calibration(metrics: dict, folder: Path) -> None:
    block = metrics["calibration"]["soft_voting"]["one_vs_rest"]
    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    axis.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="Matches the bin frequency")
    for label in CLASS_ORDER:
        bins = block[label]["bins"]
        if not bins:
            continue
        axis.plot(
            [item["mean_predicted"] for item in bins],
            [item["observed_frequency"] for item in bins],
            marker="o",
            color=CLASS_COLORS[label],
            label=label,
        )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Mean predicted probability in the bin")
    axis.set_ylabel("Observed frequency of the simulated class")
    axis.set_title("Soft voting: one-versus-rest reliability")
    axis.legend(frameon=False)
    _save(figure, folder / "fig_calibration_soft_voting.png")


def _robustness(metrics: dict, folder: Path) -> None:
    figure, axis = plt.subplots(figsize=(10.4, 4.8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.18
    series = [
        ("Test accuracy", "accuracy", "test", "#2F6F4E"),
        ("Test macro-F1", "macro_f1", "test", "#1F4E79"),
        ("Robustness accuracy", "accuracy", "robustness", "#C4B49A"),
        ("Robustness macro-F1", "macro_f1", "robustness", "#8C6A3A"),
    ]
    for index, (label, key, split, color) in enumerate(series):
        values = [metrics["models"][name][split][key] for name in MODEL_ORDER]
        axis.bar(x + (index - 1.5) * width, values, width=width, color=color, label=label)
    axis.set_xticks(x, [MODEL_LABELS[name] for name in MODEL_ORDER], rotation=20, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Accuracy and macro-F1 on the test set and the shifted scenarios")
    axis.legend(frameon=False, ncol=2)
    _save(figure, folder / "fig_robustness.png")


def _ablation(metrics: dict, folder: Path) -> None:
    full = metrics["models"]["soft_voting"]["test"]["macro_f1"]
    names = list(metrics["ablation"])
    values = [full] + [metrics["ablation"][name]["macro_f1"] for name in names]
    labels = ["All four"] + [f"Without {MODEL_LABELS[name]}" for name in names]
    figure, axis = plt.subplots(figsize=(8.4, 4.6))
    colors = [GREEN] + ["#8AA399"] * len(names)
    axis.barh(labels[::-1], values[::-1], color=colors[::-1])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Macro-F1 on the held-out test set")
    axis.set_title("Soft-voting ablation: one model removed, others not retrained")
    _save(figure, folder / "fig_ablation.png")


def _disagreement(metrics: dict, folder: Path) -> None:
    block = metrics["disagreement"]
    labels = ["Unanimous", "Majority with one dissent", "No three-vote majority"]
    values = [block["unanimous"], block["majority_with_dissent"], block["split"]]
    figure, axis = plt.subplots(figsize=(7.4, 4.2))
    axis.bar(labels, values, color=["#2F6F4E", "#C4B49A", "#8C3A3A"])
    axis.set_ylabel("Test inspections")
    axis.set_title("Agreement among the four base models")
    _save(figure, folder / "fig_disagreement.png")


def write_report_figures(metrics: dict, importance: pd.DataFrame, primary: pd.DataFrame) -> None:
    _style()
    folder = ensure_dir(FIGURE_DIR)
    _class_distribution(primary, folder)
    _model_comparison(metrics, folder)
    _confusion(metrics, folder)
    _per_class(metrics, folder)
    _importance(importance, folder)
    _calibration(metrics, folder)
    _robustness(metrics, folder)
    _ablation(metrics, folder)
    _disagreement(metrics, folder)
