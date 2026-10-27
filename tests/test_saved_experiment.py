"""Checks against the saved dataset, models, and evaluation files."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import CLASS_ORDER, DATA_DIR, MODEL_DIR, REPORT_DIR, ROOT, SUBMISSION_DIR
from src.ensemble import soft_voting_predict
from src.utilities import feature_frame, read_json

REQUIRED = [
    ROOT / "README.md",
    ROOT / "HELP.md",
    ROOT / "PROJECT_OVERVIEW.md",
    ROOT / "DATA_SOURCES.md",
    ROOT / "PUBLICATION_CHECKLIST.md",
    ROOT / "GROUP_INFO_TEMPLATE.md",
    DATA_DIR / "synthetic_lawn_inspections.csv",
    DATA_DIR / "robustness_scenarios.csv",
    DATA_DIR / "data_dictionary.md",
    DATA_DIR / "source_register.md",
    DATA_DIR / "metadata.json",
    REPORT_DIR / "data_validation_report.md",
    REPORT_DIR / "model_metrics.json",
    REPORT_DIR / "model_metrics.csv",
    REPORT_DIR / "test_predictions.csv",
    REPORT_DIR / "experiment_analysis.md",
    REPORT_DIR / "figures" / "fig_model_comparison.png",
    ROOT / "PROJECT_REPORT.md",
    SUBMISSION_DIR / "HELP.md",
    SUBMISSION_DIR / "DATA_SOURCES.md",
    SUBMISSION_DIR / "DATA_DICTIONARY.md",
    SUBMISSION_DIR / "DATASET_PROVENANCE.md",
    SUBMISSION_DIR / "MODEL_METHODOLOGY.md",
    SUBMISSION_DIR / "CONTRIBUTIONS_TEMPLATE.md",
]


def test_required_deliverables_exist_and_are_not_empty():
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists() or path.stat().st_size < 200]
    assert missing == []


def test_saved_metrics_match_the_prediction_file():
    metrics = read_json(REPORT_DIR / "model_metrics.json")
    predictions = pd.read_csv(REPORT_DIR / "test_predictions.csv")
    assert len(predictions) == metrics["split_counts"]["test"]
    for name in (
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "svm_rbf",
        "soft_voting",
    ):
        assert f"pred_{name}" in predictions.columns
        accuracy = (predictions[f"pred_{name}"] == predictions["simulated_reference_label"]).mean()
        assert abs(accuracy - metrics["models"][name]["test"]["accuracy"]) < 1e-12
    probability_columns = [f"p_soft_voting_{label}" for label in CLASS_ORDER]
    totals = predictions[probability_columns].sum(axis=1)
    assert ((totals - 1).abs() < 1e-6).all()
    assert "hard_voting" not in metrics["models"]
    assert "pred_hard_voting" not in predictions.columns
    for label in CLASS_ORDER:
        member_mean = sum(predictions[f"p_{name}_{label}"] for name in (
            "logistic_regression",
            "random_forest",
            "gradient_boosting",
            "svm_rbf",
        )) / 4
        assert ((predictions[f"p_soft_voting_{label}"] - member_mean).abs() < 1e-6).all()
    for name in ("logistic_regression", "random_forest", "gradient_boosting", "svm_rbf", "soft_voting"):
        block = metrics["models"][name]["robustness"]
        assert "accuracy" in block and "macro_f1" in block


def test_demo_cases_are_held_out_test_rows():
    scenarios = read_json(REPORT_DIR / "demo_scenarios.json")
    splits = pd.read_csv(DATA_DIR / "split_assignments.csv")
    test_ids = set(splits.loc[splits["split"] == "test", "inspection_id"].astype(str))
    train_ids = set(splits.loc[splits["split"] == "train", "inspection_id"].astype(str))
    lists = {name: identifiers for name, identifiers in scenarios.items() if isinstance(identifiers, list)}
    assert lists
    for name, identifiers in lists.items():
        assert identifiers, name
        assert set(identifiers).issubset(test_ids), name
        assert set(identifiers).isdisjoint(train_ids)


def test_loaded_models_reproduce_one_stored_soft_vote():
    metrics = read_json(REPORT_DIR / "model_metrics.json")
    run = read_json(MODEL_DIR / "run_metadata.json")
    data = read_json(DATA_DIR / "metadata.json")
    assert run["dataset_fingerprint_main"] == data["fingerprint_sha256_main"]
    assert metrics["dataset_fingerprint_main"] == data["fingerprint_sha256_main"]
    predictions = pd.read_csv(REPORT_DIR / "test_predictions.csv")
    row = predictions.iloc[0]
    primary = pd.read_csv(DATA_DIR / "synthetic_lawn_inspections.csv")
    sample = primary.loc[primary["inspection_id"] == row["inspection_id"]]
    features = feature_frame(sample)
    models = {name: joblib.load(MODEL_DIR / f"{name}.joblib") for name in run["model_files"]}
    predicted = soft_voting_predict(models, features)[0]
    assert predicted == row["pred_soft_voting"]


def test_report_quotes_the_saved_soft_voting_accuracy():
    metrics = read_json(REPORT_DIR / "model_metrics.json")
    accuracy = f"{metrics['models']['soft_voting']['test']['accuracy'] * 100:.1f}%"
    report = Path(ROOT / "PROJECT_REPORT.md").read_text(encoding="utf-8")
    analysis = Path(REPORT_DIR / "experiment_analysis.md").read_text(encoding="utf-8")
    assert accuracy in report
    assert accuracy in analysis
    assert "The Grounds Guys" in report
    assert "not affiliated" in report.lower() or "not software of The Grounds Guys" in report
