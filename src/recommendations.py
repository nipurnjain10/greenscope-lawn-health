"""Rule-based maintenance priorities.

The priority score is an explained ranking of screening actions. It is not a
model prediction and it does not prescribe chemical rates.
"""

from __future__ import annotations

from src.config import WARM_SEASON_GRASSES
from src.risk_rules import screen_stresses


URGENCY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}


def _action(
    *,
    action_id: str,
    concern: str,
    evidence: list[str],
    action: str,
    urgency: str,
    rationale: str,
    limitation: str,
    service_key: str | None,
    evidence_strength: float,
) -> dict:
    priority = URGENCY_WEIGHT[urgency] * evidence_strength
    return {
        "action_id": action_id,
        "concern": concern,
        "evidence": evidence,
        "action": action,
        "urgency": urgency,
        "rationale": rationale,
        "limitation": limitation,
        "service_key": service_key,
        "priority_score": round(priority, 2),
        "scoring": "priority = urgency weight (high 3, medium 2, low 1) × evidence strength (0.5 to 1). Cost is not part of the rank; it is shown separately.",
    }


def follow_up_plan(predicted_risk: str, stresses: dict) -> dict:
    levels = [stresses["water"]["level"], stresses["nutrient"]["level"], stresses["drainage"]["level"]]
    if predicted_risk == "High" or "likely" in levels:
        days = 14
        reason = "The ensemble risk is high or at least one screen is marked likely."
    elif predicted_risk == "Moderate" or "possible" in levels:
        days = 28
        reason = "The ensemble risk is moderate or a screen is only possible."
    else:
        days = 56
        reason = "The ensemble risk is low and no screen is marked possible or likely."
    return {
        "interval_days": days,
        "reason": reason,
        "limitation": (
            "The interval is an operational heuristic for this prototype. "
            "A later change in the predicted class would show that the inputs changed, "
            "not that a treatment caused the lawn to improve."
        ),
        "checklist": [
            "Re-measure soil moisture and record rainfall and irrigation since the last visit.",
            "Look again at wilt, colour, and bare ground, using the same scales as the first visit.",
            "Confirm whether any missing soil-test values have been obtained.",
            "Check whether water still ponds and whether traffic has worsened compaction.",
            "Do not judge success from the model score alone.",
        ],
    }


def build_recommendations(features: dict, predicted_risk: str, probabilities: dict | None = None) -> dict:
    stresses = screen_stresses(features)
    actions = []
    lab_fields = ["soil_ph", "phosphorus_mg_kg", "potassium_mg_kg", "nitrogen_mg_kg"]
    missing_labs = [name for name in lab_fields if features.get(name) in (None, "")]
    # NaN check
    missing_labs = []
    for name in lab_fields:
        value = features.get(name)
        if value is None or (isinstance(value, float) and value != value):
            missing_labs.append(name)

    nutrient_level = stresses["nutrient"]["level"]
    if missing_labs or nutrient_level in {"possible", "likely"}:
        evidence = []
        if missing_labs:
            evidence.append("Missing laboratory fields: " + ", ".join(missing_labs) + ".")
        evidence.extend(stresses["nutrient"]["reasons"][:3])
        urgency = "high" if nutrient_level == "likely" or len(missing_labs) >= 3 else "medium"
        actions.append(
            _action(
                action_id="confirm_soil_test",
                concern="Soil-test evidence is incomplete or outside the ranges discussed by the cited extension guidance.",
                evidence=evidence or ["A soil measurement used by the nutrient screen needs confirmation."],
                action="Obtain or review a laboratory soil test before any lime or nutrient decision. Sample the lawn separately from garden beds.",
                urgency=urgency,
                rationale="Rutgers and Maryland Extension both treat a soil test as the basis for phosphorus, potassium, and lime decisions. Rutgers also reports the test results that a routine turf analysis returns, including pH, phosphorus, and potassium.",
                limitation="This prototype does not turn the risk class into a fertiliser grade or an application rate.",
                service_key="soil_test_review",
                evidence_strength=1.0 if missing_labs or nutrient_level == "likely" else 0.7,
            )
        )

    if stresses["water"]["level"] in {"possible", "likely"}:
        actions.append(
            _action(
                action_id="inspect_irrigation",
                concern="The water screen found dry, hot, or wilt conditions, or frequent irrigation that has not raised moisture.",
                evidence=stresses["water"]["reasons"],
                action="Inspect irrigation coverage and how deeply water enters the soil. Compare the last week of rain with the visit notes before changing the schedule.",
                urgency="high" if stresses["water"]["level"] == "likely" else "medium",
                rationale="Maryland Extension distinguishes deep, infrequent watering of established lawns from light, frequent watering, and it describes visible drought signs.",
                limitation="The screen cannot see sprinkler uniformity, slope, or whether the lawn is dormant.",
                service_key="irrigation_inspection",
                evidence_strength=1.0 if stresses["water"]["level"] == "likely" else 0.7,
            )
        )

    if stresses["drainage"]["level"] in {"possible", "likely"}:
        actions.append(
            _action(
                action_id="investigate_drainage",
                concern="Drainage or compaction conditions match the project screen for slow water movement.",
                evidence=stresses["drainage"]["reasons"],
                action="Walk the lawn after rain or irrigation and see whether water stands. Where compaction is the issue, consider core aeration rather than a solid-tine tool. Chronically saturated areas may be unsuitable for turf.",
                urgency="high" if stresses["drainage"]["level"] == "likely" else "medium",
                rationale="Maryland Extension links slow drainage and compaction with turf stress, and it says aeration does not cure a site that stays saturated.",
                limitation="A drainage score is a worker rating, not a percolation test.",
                service_key="drainage_investigation",
                evidence_strength=1.0 if stresses["drainage"]["level"] == "likely" else 0.7,
            )
        )

    compaction = features.get("soil_compaction_score")
    traffic = features.get("foot_traffic_level")
    try:
        compaction_value = float(compaction)
    except (TypeError, ValueError):
        compaction_value = None
    if compaction_value is not None and (compaction_value >= 4 or (traffic == "high" and compaction_value >= 3)):
        actions.append(
            _action(
                action_id="assess_compaction",
                concern="Compaction or traffic may be limiting roots and drainage.",
                evidence=[
                    f"Compaction score is {int(compaction_value)} of 5.",
                    f"Foot traffic is recorded as {traffic}.",
                ],
                action="Confirm compaction in the field and, if the lawn is actively growing, schedule core aeration. Keep traffic off very wet soil.",
                urgency="high" if compaction_value >= 4 else "medium",
                rationale="Maryland Extension describes compaction from traffic, especially on wet soil, and recommends core aeration while turf is growing.",
                limitation="The score is ordinal. Equal numeric steps are a convenience, not a measured physical density.",
                service_key="aeration",
                evidence_strength=0.9 if compaction_value >= 4 else 0.6,
            )
        )

    sunlight = features.get("sunlight_hours_per_day")
    grass = features.get("grass_type")
    try:
        sun = float(sunlight)
    except (TypeError, ValueError):
        sun = None
    if sun is not None and (sun < 4 or (sun < 6 and grass in WARM_SEASON_GRASSES)):
        actions.append(
            _action(
                action_id="review_sunlight",
                concern="Direct sun may be too limited for a durable lawn in this location.",
                evidence=[
                    f"Direct sun is about {sun:.1f} hours per day.",
                    f"Recorded grass type: {str(grass).replace('_', ' ')}.",
                ],
                action="Review whether the grass type suits the light. Maryland Extension says full sun is best at about 6 or more hours and that lawn grasses do not thrive in little direct sun. A lawn alternative may be more realistic in deep shade.",
                urgency="medium" if sun < 4 else "low",
                rationale="The source statement is about site suitability, not about a fertiliser or pesticide.",
                limitation="Hours of sun are an estimate. Species-level shade tolerance was not taken from a downloaded EcoCrop record.",
                service_key="sunlight_review",
                evidence_strength=0.8 if sun < 4 else 0.5,
            )
        )

    if nutrient_level in {"possible", "likely"}:
        actions.append(
            _action(
                action_id="review_nutrients",
                concern="One or more project nutrient or pH screens fired.",
                evidence=stresses["nutrient"]["reasons"],
                action="Review nutrient management against a current soil test. Do not choose a product rate from the overall risk class.",
                urgency="medium" if nutrient_level == "likely" else "low",
                rationale="Rutgers Extension says phosphorus, potassium, and lime for an established lawn should follow a soil test, and that nitrogen need is not established by a routine soil test.",
                limitation="Project mg/kg bands are not transferable laboratory thresholds.",
                service_key="nutrient_review",
                evidence_strength=0.8 if nutrient_level == "likely" else 0.5,
            )
        )

    thatch = features.get("thatch_thickness_mm")
    try:
        thatch_mm = float(thatch)
    except (TypeError, ValueError):
        thatch_mm = None
    if thatch_mm is not None and thatch_mm > 12.7:
        actions.append(
            _action(
                action_id="review_thatch",
                concern="Thatch is thicker than about one-half inch.",
                evidence=[f"Thatch thickness is {thatch_mm:.1f} mm. Maryland Extension discusses dethatching once thatch exceeds about one-half inch (12.7 mm)."],
                action="Confirm the thatch depth in the field. If it is still above one-half inch, plan dethatching when the lawn is growing, and review whether heavy watering or nitrogen is contributing.",
                urgency="medium",
                rationale="The half-inch figure is the extension management discussion used as a screen, not as a measured biological threshold from this dataset.",
                limitation="The thickness is a simulated or worker estimate.",
                service_key="thatch_service",
                evidence_strength=0.7,
            )
        )

    mowing = features.get("mowing_sessions_per_month")
    try:
        mowing_n = float(mowing)
    except (TypeError, ValueError):
        mowing_n = None
    if mowing_n is not None and mowing_n <= 1:
        actions.append(
            _action(
                action_id="review_mowing",
                concern="Very few mowing sessions were recorded.",
                evidence=[f"Mowing sessions in the recent month: {int(mowing_n)}."],
                action="If the lawn is actively growing, a long gap can force too large a cut at the next mow. Confirm growth before changing the schedule. Rutgers and Maryland Extension both say not to remove more than about one-third of the leaf at a mowing.",
                urgency="low",
                rationale="Low mowing frequency can mean neglect or seasonal dormancy. The recommendation asks for that distinction.",
                limitation="The prototype does not know the mowing height or whether the lawn is dormant.",
                service_key="mowing_visit",
                evidence_strength=0.5,
            )
        )

    weeds = features.get("weed_cover_pct")
    try:
        weed_pct = float(weeds)
    except (TypeError, ValueError):
        weed_pct = None
    if weed_pct is not None and weed_pct >= 25:
        actions.append(
            _action(
                action_id="review_weeds",
                concern="Weeds occupy a substantial share of the lawn.",
                evidence=[f"Weed cover is estimated at {weed_pct:.1f}% of the area. Species were not identified."],
                action="Identify the weeds before any control decision. Restore turf density with suitable maintenance. This prototype does not recommend a herbicide or a rate.",
                urgency="medium" if weed_pct >= 40 else "low",
                rationale="Rutgers Extension emphasises identification and a dense lawn, and it warns that herbicide choice depends on the weed.",
                limitation="Cover is a visual estimate, not a weed survey.",
                service_key="weed_review",
                evidence_strength=0.7 if weed_pct >= 40 else 0.5,
            )
        )

    follow = follow_up_plan(predicted_risk, stresses)
    actions.append(
        _action(
            action_id="schedule_follow_up",
            concern="The lawn needs a planned return visit so changes are observed rather than assumed.",
            evidence=[
                f"Ensemble risk class: {predicted_risk}.",
                follow["reason"],
            ],
            action=f"Schedule a follow-up in about {follow['interval_days']} days and use the checklist.",
            urgency="high" if follow["interval_days"] == 14 else "medium" if follow["interval_days"] == 28 else "low",
            rationale="A second visit checks whether the recorded conditions persist. It is not an estimate of treatment effect.",
            limitation=follow["limitation"],
            service_key="follow_up_visit",
            evidence_strength=0.8,
        )
    )

    actions = sorted(actions, key=lambda item: (-item["priority_score"], item["action_id"]))
    minimal = [item for item in actions if item["urgency"] == "high"]
    if not minimal:
        minimal = [item for item in actions if item["action_id"] == "schedule_follow_up"]
    comprehensive = [item for item in actions if item["urgency"] in {"high", "medium"}]
    return {
        "stresses": stresses,
        "actions": actions,
        "minimal_action_ids": [item["action_id"] for item in minimal],
        "comprehensive_action_ids": [item["action_id"] for item in comprehensive],
        "follow_up": follow,
        "disclaimer": (
            "Suggestions are cautious next steps. They are not a diagnosis, a fertiliser prescription, "
            "or a pesticide recommendation. Overall risk does not by itself select a treatment."
        ),
    }
