"""Single-inspection prediction used by the application and the tests.

The function loads nothing and fits nothing. Callers pass already fitted pipelines.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CLASS_ORDER,
    FEATURE_SPECS,
    MODEL_SPECS,
    UI_LIMITS,
)
from src.cost_estimation import default_assumptions, plans_from_recommendations
from src.ensemble import align_proba, soft_voting_proba
from src.explainability import local_baseline_contributions
from src.recommendations import build_recommendations
from src.utilities import feature_frame


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return True
    return False


def validate_user_inputs(raw: dict) -> tuple[dict, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    cleaned = {}
    required = ["lawn_area_m2", "grass_type", "soil_texture", "foot_traffic_level"]
    for name in required:
        if _is_blank(raw.get(name)):
            errors.append(f"{name.replace('_', ' ')} is required. It is observed on site and is not imputed.")

    unknown_count = 0
    for name, spec in FEATURE_SPECS.items():
        value = raw.get(name)
        if _is_blank(value):
            cleaned[name] = np.nan
            if name not in required:
                unknown_count += 1
            continue
        if spec["kind"] == "categorical":
            text = str(value).strip()
            if text not in spec["levels"]:
                errors.append(f"{name.replace('_', ' ')} must be one of: {', '.join(spec['levels'])}.")
                cleaned[name] = text
            else:
                cleaned[name] = text
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name.replace('_', ' ')} must be a number or marked unknown.")
            cleaned[name] = np.nan
            continue
        lower, upper = UI_LIMITS[name]
        if number < lower or number > upper:
            errors.append(
                f"{name.replace('_', ' ')} must be between {lower:g} and {upper:g} {spec['unit']}."
            )
        elif number < spec["minimum"] or number > spec["maximum"]:
            warnings.append(
                f"{name.replace('_', ' ')} is outside the {spec['minimum']:g}–{spec['maximum']:g} range "
                "used to generate the training data. The model has little evidence in this region."
            )
        if spec["decimals"] == 0:
            cleaned[name] = int(round(number))
        else:
            cleaned[name] = number

    if unknown_count >= 6:
        warnings.append(
            f"{unknown_count} inputs were marked unknown. Imputed values are training statistics, not measurements, "
            "so the assessment is weakly supported."
        )
    elif unknown_count:
        warnings.append(
            f"{unknown_count} input(s) were marked unknown and will be imputed with the training-only pipeline. "
            "Unknown does not mean a normal or healthy value."
        )
    return cleaned, errors, warnings


def _agreement(predictions: dict[str, str]) -> dict:
    labels = list(predictions.values())
    counts = {label: labels.count(label) for label in set(labels)}
    top = max(counts.values())
    if top == len(labels):
        status = "unanimous"
        detail = "All four models predict the same class."
    elif top >= 3:
        status = "majority"
        winner = max(counts, key=counts.get)
        dissenters = [name for name, label in predictions.items() if label != winner]
        detail = (
            f"Three models predict {winner}. "
            + ", ".join(f"{MODEL_SPECS[name]} predicts {predictions[name]}" for name in dissenters)
            + "."
        )
    else:
        status = "disagreement"
        detail = "The models do not share a three-vote majority. Treat the class as unsettled and inspect the lawn again."
    return {"status": status, "votes": predictions, "detail": detail}


def assess(raw_features: dict, models: dict, baselines: dict | None = None, assumptions: dict | None = None) -> dict:
    cleaned, errors, warnings = validate_user_inputs(raw_features)
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}
    missing_models = [name for name in MODEL_SPECS if name not in models]
    if missing_models:
        return {
            "ok": False,
            "errors": [f"Fitted models are missing: {', '.join(missing_models)}. Train the project before assessing a lawn."],
            "warnings": warnings,
        }

    frame = feature_frame(cleaned)
    individual = {}
    for name, model in models.items():
        if name not in MODEL_SPECS:
            continue
        probabilities = align_proba(model, model.predict_proba(frame))[0]
        predicted = CLASS_ORDER[int(np.argmax(probabilities))]
        individual[name] = {
            "display_name": MODEL_SPECS[name],
            "prediction": predicted,
            "probabilities": {label: float(value) for label, value in zip(CLASS_ORDER, probabilities)},
        }
    soft = soft_voting_proba({name: models[name] for name in MODEL_SPECS}, frame)[0]
    soft_prediction = CLASS_ORDER[int(np.argmax(soft))]
    probability_map = {label: float(value) for label, value in zip(CLASS_ORDER, soft)}
    agreement = _agreement({name: payload["prediction"] for name, payload in individual.items()})
    recommendations = build_recommendations(cleaned, soft_prediction, probability_map)
    price_book = assumptions or default_assumptions("INR")
    costs = plans_from_recommendations(float(cleaned["lawn_area_m2"]), recommendations, price_book)
    explanations = []
    if baselines:
        explanations = local_baseline_contributions(
            {name: models[name] for name in MODEL_SPECS},
            frame,
            baselines,
            soft_prediction,
        )
    serialisable_inputs = {}
    for key, value in cleaned.items():
        if _is_blank(value):
            serialisable_inputs[key] = None
        elif isinstance(value, (np.integer, int)):
            serialisable_inputs[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            serialisable_inputs[key] = float(value)
        else:
            serialisable_inputs[key] = str(value)
    return {
        "ok": True,
        "errors": [],
        "warnings": warnings,
        "inputs": serialisable_inputs,
        "prediction": soft_prediction,
        "probabilities": probability_map,
        "individual": individual,
        "agreement": agreement,
        "explanations": explanations,
        "recommendations": recommendations,
        "costs": costs,
        "probability_note": (
            "These are model-estimated class probabilities from the fitted pipelines. "
            "They have not been shown to match frequencies of real lawn outcomes. "
            "See the calibration results on the research page before reading them as confidence."
        ),
        "explanation_note": (
            "A local contribution is the change in the soft-voting probability of the predicted class "
            "when that measurement is replaced by the fitting-set median or mode. "
            "It describes the model's dependence on the value, not a biological cause."
        ),
    }


def compare_assessments(older: dict, newer: dict) -> dict:
    """Describe input and output differences. This is not a treatment-effect estimate."""
    changes = []
    for name, spec in FEATURE_SPECS.items():
        before = older["inputs"].get(name)
        after = newer["inputs"].get(name)
        if before == after:
            continue
        changes.append(
            {
                "feature": name,
                "unit": spec["unit"],
                "before": before,
                "after": after,
            }
        )
    return {
        "risk_before": older.get("prediction"),
        "risk_after": newer.get("prediction"),
        "input_changes": changes,
        "stress_before": {key: older["recommendations"]["stresses"][key]["level"] for key in ("water", "nutrient", "drainage")},
        "stress_after": {key: newer["recommendations"]["stresses"][key]["level"] for key in ("water", "nutrient", "drainage")},
        "cost_before": older.get("costs", {}).get("comprehensive", {}).get("display_total"),
        "cost_after": newer.get("costs", {}).get("comprehensive", {}).get("display_total"),
        "limitation": (
            "The comparison shows that the recorded inputs and the resulting model output changed. "
            "It does not show that a service caused the lawn to improve or decline."
        ),
    }
