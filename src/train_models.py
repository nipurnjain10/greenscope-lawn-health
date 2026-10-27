"""Train the four classifiers, evaluate them once on the held-out test set, and save artifacts.

Validation metrics are recorded from a training-only fit. Reported test metrics
come from a refit on training plus validation. The test labels are not used to
choose hyperparameters, weights, calibration, or the label-generating process.
"""

from __future__ import annotations

import sys

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLASS_ORDER,
    DATA_DIR,
    FEATURE_COLUMNS,
    GENERATOR_VERSION,
    MODEL_DIR,
    MODEL_SPECS,
    REPORT_DIR,
    SPLIT_SEED,
    TARGET_COLUMN,
)
from src.ensemble import SoftVotingModel, leave_one_out_soft_predict, soft_voting_predict
from src.evaluation import (
    bootstrap_intervals,
    calibration_report,
    disagreement_summary,
    metrics_frame,
    score_predictions,
)
from src.explainability import permutation_table, training_baselines
from src.figures import write_report_figures
from src.narratives import (
    ablation_text,
    calibration_text,
    importance_text,
    performance_comparison_text,
    per_class_text,
    robustness_text,
    split_text,
)
from src.preprocessing import build_estimators
from src.risk_rules import screen_stresses
from src.splits import make_split_table, subset_by_split
from src.utilities import feature_frame, package_versions, read_json, sha256_file, utc_now_iso, write_json


def _fit_all(features: pd.DataFrame, target: pd.Series) -> dict:
    models = build_estimators()
    for name, model in models.items():
        print(f"Fitting {name} on {len(features)} rows...", flush=True)
        model.fit(features, target)
    return models


def _score_split(models: dict, frame: pd.DataFrame) -> dict[str, dict]:
    features = feature_frame(frame)
    target = frame[TARGET_COLUMN].astype(str)
    scored = {}
    predictions = {}
    for name, model in models.items():
        predicted = model.predict(features)
        predictions[name] = np.asarray(predicted).astype(str)
        scored[name] = score_predictions(target, predictions[name])
    soft_predicted = soft_voting_predict(models, features)
    predictions["soft_voting"] = np.asarray(soft_predicted).astype(str)
    scored["soft_voting"] = score_predictions(target, predictions["soft_voting"])
    return {"metrics": scored, "predictions": predictions, "features": features, "target": target}


def _prediction_table(frame: pd.DataFrame, models: dict) -> pd.DataFrame:
    features = feature_frame(frame)
    table = pd.DataFrame({
        "inspection_id": frame["inspection_id"].astype(str).to_numpy(),
        "simulated_reference_label": frame[TARGET_COLUMN].astype(str).to_numpy(),
    })
    probability_columns = {}
    for name, model in models.items():
        probabilities = model.predict_proba(features)
        order = [str(label) for label in model.classes_]
        index = [order.index(label) for label in CLASS_ORDER]
        aligned = np.asarray(probabilities)[:, index]
        predicted = np.array(CLASS_ORDER)[np.argmax(aligned, axis=1)]
        table[f"pred_{name}"] = predicted
        for class_index, label in enumerate(CLASS_ORDER):
            probability_columns[f"p_{name}_{label}"] = aligned[:, class_index]
    soft_probability = []
    for label in CLASS_ORDER:
        columns = [probability_columns[f"p_{name}_{label}"] for name in MODEL_SPECS]
        soft_probability.append(np.mean(np.column_stack(columns), axis=1))
    soft_matrix = np.column_stack(soft_probability)
    soft_matrix = soft_matrix / soft_matrix.sum(axis=1, keepdims=True)
    table["pred_soft_voting"] = np.array(CLASS_ORDER)[np.argmax(soft_matrix, axis=1)]
    for label_index, label in enumerate(CLASS_ORDER):
        table[f"p_soft_voting_{label}"] = soft_matrix[:, label_index]
    for column, values in probability_columns.items():
        table[column] = values
    return table


def _error_examples(frame: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    merged = frame.merge(predictions, on="inspection_id", how="inner")
    base_columns = ["inspection_id", "simulated_reference_label", "pred_logistic_regression", "pred_random_forest", "pred_gradient_boosting", "pred_svm_rbf", "pred_soft_voting"]
    show_features = [
        "grass_type",
        "sunlight_hours_per_day",
        "soil_ph",
        "soil_moisture_pct",
        "drainage_score",
        "soil_compaction_score",
        "temperature_c",
        "rainfall_7d_mm",
        "irrigation_sessions_per_week",
        "discoloration_pct",
        "bare_patch_pct",
        "wilting_score",
    ]
    vote_columns = [f"pred_{name}" for name in MODEL_SPECS]

    def pack(subset: pd.DataFrame, limit: int = 8) -> list[dict]:
        records = []
        for _, row in subset.head(limit).iterrows():
            item = {column: row[column] for column in base_columns}
            item["features"] = {}
            for feature in show_features:
                value = row[feature]
                item["features"][feature] = None if pd.isna(value) else value.item() if hasattr(value, "item") else value
            records.append(item)
        return records

    high_called_low = merged[
        (merged["simulated_reference_label"] == "High") & (merged["pred_soft_voting"] == "Low")
    ].sort_values("inspection_id")
    disagreed = merged.loc[merged.loc[:, vote_columns].nunique(axis=1) > 1].sort_values("inspection_id")
    correct_high = merged[
        (merged["simulated_reference_label"] == "High") & (merged["pred_soft_voting"] == "High")
    ].sort_values("inspection_id")
    return {
        "high_called_low": pack(high_called_low),
        "model_disagreement": pack(disagreed),
        "correct_high": pack(correct_high),
        "counts": {
            "high_called_low": int(len(high_called_low)),
            "model_disagreement": int(len(disagreed)),
            "correct_high": int(len(correct_high)),
        },
    }


def _demo_scenarios(test_frame: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    merged = test_frame.merge(predictions, on="inspection_id", how="inner").sort_values("inspection_id")
    levels = []
    for _, row in merged.iterrows():
        payload = {column: row[column] for column in FEATURE_COLUMNS}
        stresses = screen_stresses(payload)
        levels.append(
            {
                "water": stresses["water"]["level"],
                "nutrient": stresses["nutrient"]["level"],
                "drainage": stresses["drainage"]["level"],
            }
        )
    level_frame = pd.DataFrame(levels)
    merged = pd.concat([merged.reset_index(drop=True), level_frame], axis=1)
    vote_columns = [f"pred_{name}" for name in MODEL_SPECS]
    max_probability = merged[[f"p_soft_voting_{label}" for label in CLASS_ORDER]].max(axis=1)
    likely_count = (
        (merged["water"] == "likely").astype(int)
        + (merged["nutrient"] == "likely").astype(int)
        + (merged["drainage"] == "likely").astype(int)
    )
    buckets = {
        "healthy": (merged["simulated_reference_label"] == "Low")
        & (merged["wilting_score"] <= 1)
        & (merged["discoloration_pct"] <= 12)
        & (merged["bare_patch_pct"] <= 10)
        & (merged["drainage_score"] >= 3),
        "water_stress": merged["water"] == "likely",
        "drainage": merged["drainage"] == "likely",
        "nutrient": merged["nutrient"] == "likely",
        "interacting": likely_count >= 2,
        "ambiguous": (merged.loc[:, vote_columns].nunique(axis=1) > 1) & (max_probability < 0.55),
    }
    scenarios = {}
    for name, mask in buckets.items():
        identifiers = merged.loc[mask, "inspection_id"].astype(str).head(15).tolist()
        scenarios[name] = identifiers
    scenarios["note"] = (
        "Identifiers are held-out primary-test rows. Labels shown beside them in the demo "
        "are simulated benchmark labels, not field diagnoses."
    )
    return scenarios


def _json_ready_examples(examples: dict) -> dict:
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            if np.isnan(value):
                return None
            return float(value)
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value

    return convert(examples)


def main() -> None:
    primary_path = DATA_DIR / "synthetic_lawn_inspections.csv"
    robust_path = DATA_DIR / "robustness_scenarios.csv"
    metadata_path = DATA_DIR / "metadata.json"
    if not primary_path.exists() or not metadata_path.exists():
        raise SystemExit("Generate the dataset before training.")
    metadata = read_json(metadata_path)
    fingerprint = sha256_file(primary_path)
    if fingerprint != metadata["fingerprint_sha256_main"]:
        raise SystemExit("The primary CSV does not match data/metadata.json. Regenerate the data.")
    if sha256_file(robust_path) != metadata["fingerprint_sha256_robustness"]:
        raise SystemExit("The robustness CSV does not match data/metadata.json. Regenerate the data.")

    primary = pd.read_csv(primary_path)
    robust = pd.read_csv(robust_path)
    split_table = make_split_table(primary, seed=SPLIT_SEED)
    train = subset_by_split(primary, split_table, "train")
    validation = subset_by_split(primary, split_table, "validation")
    test = subset_by_split(primary, split_table, "test")
    fitting = pd.concat([train, validation], ignore_index=True)
    print(
        f"Split sizes: train {len(train)}, validation {len(validation)}, test {len(test)}",
        flush=True,
    )

    print("Validation pass: fit on training rows only.", flush=True)
    validation_models = _fit_all(feature_frame(train), train[TARGET_COLUMN].astype(str))
    validation_scores = _score_split(validation_models, validation)["metrics"]
    del validation_models

    print("Final pass: refit on training plus validation.", flush=True)
    models = _fit_all(feature_frame(fitting), fitting[TARGET_COLUMN].astype(str))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    baselines = training_baselines(fitting)
    write_json(MODEL_DIR / "feature_baselines.json", baselines)

    print("Scoring the untouched test set and the robustness file.", flush=True)
    test_scores = _score_split(models, test)
    robust_scores = _score_split(models, robust)
    for name in list(MODEL_SPECS) + ["soft_voting"]:
        test_scores["metrics"][name]["bootstrap"] = bootstrap_intervals(
            test[TARGET_COLUMN],
            test_scores["predictions"][name],
        )
    test_predictions = _prediction_table(test, models)

    print("Ablation on the test set without retraining.", flush=True)
    ablation_predictions = leave_one_out_soft_predict(models, feature_frame(test))
    ablation = {}
    for held_out, predicted in ablation_predictions.items():
        ablation[held_out] = score_predictions(test[TARGET_COLUMN], predicted)

    print("Calibration on test probabilities.", flush=True)
    soft_columns = [f"p_soft_voting_{label}" for label in CLASS_ORDER]
    calibration = {
        "soft_voting": calibration_report(
            test_predictions["simulated_reference_label"],
            test_predictions.loc[:, soft_columns].to_numpy(),
        )
    }
    for name in MODEL_SPECS:
        columns = [f"p_{name}_{label}" for label in CLASS_ORDER]
        calibration[name] = calibration_report(
            test_predictions["simulated_reference_label"],
            test_predictions.loc[:, columns].to_numpy(),
        )

    print("Permutation importance on the test set.", flush=True)
    importance_models = dict(models)
    importance_models["soft_voting"] = SoftVotingModel(models).fit()
    importance = permutation_table(
        importance_models,
        feature_frame(test),
        test[TARGET_COLUMN].astype(str),
        repeats=5,
    )

    vote_columns = [f"pred_{name}" for name in MODEL_SPECS]
    disagreement = disagreement_summary(test_predictions, vote_columns)
    examples = _json_ready_examples(_error_examples(test, test_predictions))
    scenarios = _demo_scenarios(test, test_predictions)

    model_payload = {}
    for name in list(MODEL_SPECS) + ["soft_voting"]:
        model_payload[name] = {
            "validation_before_refit": validation_scores[name],
            "test": test_scores["metrics"][name],
            "robustness": robust_scores["metrics"][name],
        }

    metrics = {
        "class_order": CLASS_ORDER,
        "split_seed": SPLIT_SEED,
        "split_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
            "robustness": int(len(robust)),
        },
        "split_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "generator_version": metadata.get("generator_version", GENERATOR_VERSION),
        "dataset_fingerprint_main": fingerprint,
        "models": model_payload,
        "ablation": ablation,
        "calibration": calibration,
        "disagreement": disagreement,
        "evaluation_protocol": (
            "Hyperparameters were pre-specified. Validation metrics use a fit on the training split only. "
            "Final models are refit on training plus validation and then scored once on the test split and on the separate robustness file. "
            "Permutation importance, calibration statistics, bootstrap intervals, and ablation use those final predictions. "
            "None of those evaluation steps changes the saved predictions."
        ),
    }
    top_soft = (
        importance.loc[importance["model"] == "soft_voting"]
        .sort_values("importance_mean", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )
    metrics["narratives"] = {
        "split": split_text(metrics),
        "performance": performance_comparison_text(metrics),
        "per_class": per_class_text(metrics),
        "robustness": robustness_text(metrics),
        "ablation": ablation_text(metrics),
        "calibration": calibration_text(metrics),
        "importance": importance_text(top_soft),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    split_table.to_csv(DATA_DIR / "split_assignments.csv", index=False)
    test_predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)
    importance.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    metrics_frame(metrics).to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    write_json(REPORT_DIR / "model_metrics.json", metrics)
    write_json(REPORT_DIR / "error_examples.json", examples)
    write_json(REPORT_DIR / "demo_scenarios.json", scenarios)

    run_metadata = {
        "created_at_utc": utc_now_iso(),
        "python_version": sys.version.split()[0],
        "package_versions": package_versions(),
        "generator_version": metadata.get("generator_version"),
        "dataset_fingerprint_main": fingerprint,
        "dataset_fingerprint_robustness": metadata["fingerprint_sha256_robustness"],
        "label_noise_sd": metadata.get("label_noise_sd"),
        "class_thresholds": metadata.get("class_thresholds"),
        "split_seed": SPLIT_SEED,
        "split_counts": metrics["split_counts"],
        "feature_columns": FEATURE_COLUMNS,
        "class_order": CLASS_ORDER,
        "model_files": {name: f"models/{name}.joblib" for name in MODEL_SPECS},
        "hyperparameters": {
            "logistic_regression": "L2 logistic regression, C=1, lbfgs, scaled numeric inputs",
            "random_forest": "200 trees, max_depth=10, min_samples_leaf=8, max_features=sqrt",
            "gradient_boosting": "300 trees, learning_rate=0.05, max_depth=3, subsample=0.85. A validation-only grid preferred this over 150 trees at learning rate 0.08. The test set was not part of that choice.",
            "svm_rbf": "RBF kernel, C=1, gamma=scale. Class probabilities come from CalibratedClassifierCV with sigmoid method, cv=3, and ensemble=False, fit on the training rows only.",
            "voting": "Soft voting only. Equal weights. The class is the largest mean probability. Weights were not tuned on the test set.",
        },
        "protocol": metrics["evaluation_protocol"],
    }
    write_json(MODEL_DIR / "run_metadata.json", run_metadata)
    print("Writing report figures.", flush=True)
    write_report_figures(metrics, importance, primary)
    print("Training complete.", flush=True)
    print(metrics["narratives"]["performance"], flush=True)


if __name__ == "__main__":
    main()
