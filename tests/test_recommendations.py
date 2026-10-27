"""Screens, recommendations, and illustrative costs."""

from __future__ import annotations

from src.config import FEATURE_SPECS
from src.cost_estimation import default_assumptions, estimate_services
from src.recommendations import build_recommendations
from src.risk_rules import screen_stresses


def _base_inputs() -> dict:
    values = {}
    for name, spec in FEATURE_SPECS.items():
        if spec["kind"] == "categorical":
            values[name] = spec["levels"][0]
        else:
            values[name] = (float(spec["minimum"]) + float(spec["maximum"])) / 2
    values["grass_type"] = "tall_fescue"
    values["soil_texture"] = "loam"
    values["foot_traffic_level"] = "low"
    values["soil_ph"] = 6.4
    values["soil_moisture_pct"] = 22
    values["temperature_c"] = 21
    values["rainfall_7d_mm"] = 25
    values["irrigation_sessions_per_week"] = 2
    values["wilting_score"] = 0
    values["drainage_score"] = 4
    values["soil_compaction_score"] = 2
    values["lawn_area_m2"] = 200
    return values


def test_hot_dry_lawn_is_screened_for_water_not_called_a_model():
    features = _base_inputs()
    features.update(
        {
            "soil_moisture_pct": 8,
            "temperature_c": 34,
            "rainfall_7d_mm": 2,
            "irrigation_sessions_per_week": 0,
            "wilting_score": 4,
        }
    )
    screen = screen_stresses(features)
    assert screen["water"]["level"] == "likely"
    assert "not independently trained" in screen["method"]
    assert screen["water"]["reasons"]


def test_missing_soil_tests_do_not_become_healthy_defaults():
    features = _base_inputs()
    for name in ("soil_ph", "nitrogen_mg_kg", "phosphorus_mg_kg", "potassium_mg_kg"):
        features[name] = None
    screen = screen_stresses(features)
    assert screen["nutrient"]["level"] == "unavailable"
    recommendation = build_recommendations(features, "High")
    actions = {item["action_id"]: item for item in recommendation["actions"]}
    assert "confirm_soil_test" in actions
    assert "fertiliser rate" not in actions["review_nutrients"]["action"].lower() if "review_nutrients" in actions else True
    assert "prescription" in recommendation["disclaimer"] or "not a diagnosis" in recommendation["disclaimer"]


def test_cost_uses_area_and_edited_assumptions():
    inr = default_assumptions("INR")
    estimate = estimate_services(100, ["aeration"], inr)
    assert estimate["total"] == 400
    assert "100" in estimate["lines"][0]["formula"]
    inr["items"]["aeration"]["per_m2"] = 10
    revised = estimate_services(100, ["aeration"], inr)
    assert revised["total"] == 1000
    usd = default_assumptions("USD")
    assert usd["symbol"] == "$"
    assert estimate_services(100, ["aeration"], usd)["total"] != estimate["total"]


def test_high_risk_alone_does_not_invent_a_nutrient_action_without_evidence():
    features = _base_inputs()
    recommendation = build_recommendations(features, "High")
    nutrient_actions = [item for item in recommendation["actions"] if item["action_id"] == "review_nutrients"]
    assert nutrient_actions == []
    assert recommendation["follow_up"]["interval_days"] == 14
