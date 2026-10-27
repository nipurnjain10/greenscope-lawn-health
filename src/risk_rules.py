"""Transparent water, nutrient, and drainage screens.

These are documented rules. They are not separate trained classifiers and they
do not emit probabilities.
"""

from __future__ import annotations

from src.config import WARM_SEASON_GRASSES


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)  # NaN
    except Exception:
        return False


def _num(value):
    if _missing(value):
        return None
    return float(value)


def screen_stresses(features: dict) -> dict:
    """Return three screening indicators with the conditions that fired."""
    return {
        "water": _water(features),
        "nutrient": _nutrient(features),
        "drainage": _drainage(features),
        "method": (
            "Source-informed screening rules. These are not independently trained "
            "or evaluated classifiers, and they are not probabilities."
        ),
    }


def _level(points: int, unavailable: bool) -> str:
    if unavailable:
        return "unavailable"
    if points <= 0:
        return "none"
    if points == 1:
        return "possible"
    return "likely"


def _water(features: dict) -> dict:
    reasons = []
    missing = []
    points = 0
    moisture = _num(features.get("soil_moisture_pct"))
    temperature = _num(features.get("temperature_c"))
    rainfall = _num(features.get("rainfall_7d_mm"))
    irrigation = _num(features.get("irrigation_sessions_per_week"))
    wilting = _num(features.get("wilting_score"))
    grass = features.get("grass_type")
    if _missing(grass):
        missing.append("grass_type")

    if moisture is None:
        missing.append("soil_moisture_pct")
    if temperature is None:
        missing.append("temperature_c")
    if rainfall is None:
        missing.append("rainfall_7d_mm")
    if irrigation is None:
        missing.append("irrigation_sessions_per_week")
    if wilting is None:
        missing.append("wilting_score")

    if moisture is not None and temperature is not None and moisture < 12 and temperature >= 29:
        points += 1
        reasons.append(
            f"Soil moisture is {moisture:.1f}% while air temperature is {temperature:.1f} °C."
        )
    if rainfall is not None and irrigation is not None and temperature is not None:
        if rainfall < 8 and irrigation <= 1 and temperature >= 27:
            points += 1
            reasons.append(
                f"Only {rainfall:.1f} mm of rain fell in seven days, irrigation was "
                f"{int(irrigation)} session(s) per week, and temperature is {temperature:.1f} °C. "
                "University of Maryland Extension discusses about 25 mm of water per week for turf in active growth; "
                "this rule uses a much drier project cutoff and still requires heat and little irrigation."
            )
    if wilting is not None and wilting >= 3:
        points += 1
        reasons.append(
            f"Wilting score is {int(wilting)} on a 0–4 project scale. "
            "Extension notes describe footprinting and a dull colour as drought signs; this score is not itself a measurement of leaf water potential."
        )
    if (
        moisture is not None
        and temperature is not None
        and grass
        and grass not in WARM_SEASON_GRASSES
        and temperature >= 33
        and moisture < 18
    ):
        points += 1
        reasons.append(
            f"{grass.replace('_', ' ')} is treated as cool-season in this prototype, "
            f"temperature is {temperature:.1f} °C, and moisture is {moisture:.1f}%."
        )
    if irrigation is not None and moisture is not None and irrigation >= 6 and moisture < 14:
        points += 1
        reasons.append(
            f"Irrigation is frequent ({int(irrigation)} sessions per week) but moisture is still {moisture:.1f}%. "
            "That pattern can mean poor coverage or shallow watering; it is not proof of either."
        )
    unavailable = moisture is None and rainfall is None and wilting is None
    return {
        "indicator": "water",
        "level": _level(points, unavailable),
        "points": points,
        "reasons": reasons,
        "missing_fields": missing,
        "limitation": (
            "A water-stress flag is a screening hint from the entered measurements. "
            "It does not establish drought injury or an irrigation schedule."
        ),
    }


def _nutrient(features: dict) -> dict:
    reasons = []
    missing = []
    points = 0
    ph = _num(features.get("soil_ph"))
    nitrogen = _num(features.get("nitrogen_mg_kg"))
    phosphorus = _num(features.get("phosphorus_mg_kg"))
    potassium = _num(features.get("potassium_mg_kg"))
    grass = features.get("grass_type")
    for name, value in (
        ("soil_ph", ph),
        ("nitrogen_mg_kg", nitrogen),
        ("phosphorus_mg_kg", phosphorus),
        ("potassium_mg_kg", potassium),
    ):
        if value is None:
            missing.append(name)

    if ph is not None:
        warm = grass in WARM_SEASON_GRASSES
        upper = 7.7 if warm else 7.0
        if ph < 5.5:
            points += 1
            reasons.append(
                f"Soil pH is {ph:.2f}. University of Maryland Extension says to review lime when pH is below 5.5. "
                "This prototype does not calculate a lime rate."
            )
        elif ph > upper:
            points += 1
            reasons.append(
                f"Soil pH is {ph:.2f}, above the project screen of {upper:.1f} for this grass group. "
                "Maryland Extension discusses alkaline conditions above 7.0 for many lawns and notes a higher upper tolerance for zoysia and bermudagrass. "
                "No sulfur rate is calculated here."
            )
        elif ph < 6.0 or (not warm and ph > 6.8):
            points += 1
            reasons.append(
                f"Soil pH is {ph:.2f}, outside the about 6.0–6.8 band Maryland Extension describes as favourable for turf. "
                "The departure is recorded as a review item, not as a diagnosis."
            )
    if phosphorus is not None and phosphorus < 12:
        points += 1
        reasons.append(
            f"Simulated extractable phosphorus is {phosphorus:.1f} mg/kg, inside this project's low band (below 12 mg/kg). "
            "That band is not a Rutgers or Maryland sufficiency category."
        )
    if potassium is not None and potassium < 50:
        points += 1
        reasons.append(
            f"Simulated extractable potassium is {potassium:.1f} mg/kg, inside this project's low band (below 50 mg/kg). "
            "That band is not a laboratory sufficiency category."
        )
    if nitrogen is not None and nitrogen < 8:
        points += 1
        reasons.append(
            f"Simulated mineral nitrogen is {nitrogen:.1f} mg/kg, inside this project's low band (below 8 mg/kg). "
            "Rutgers Extension states that a routine soil test has not been developed to measure turfgrass nitrogen need, "
            "so this value only prompts a review of the fertilisation programme."
        )
    if phosphorus is not None and phosphorus > 80:
        reasons.append(
            f"Simulated phosphorus is {phosphorus:.1f} mg/kg, in the upper part of the generated range. "
            "Maryland lawn-fertiliser guidance says phosphorus additions are not appropriate when a real soil test is already optimum or excessive. "
            "Confirm the laboratory result before adding phosphorus. This note does not add a stress point by itself."
        )
    unavailable = ph is None and phosphorus is None and potassium is None
    return {
        "indicator": "nutrient",
        "level": _level(points, unavailable),
        "points": points,
        "reasons": reasons or (["No project nutrient-screen condition fired for the measurements that were supplied."] if not unavailable else []),
        "missing_fields": missing,
        "limitation": (
            "Nutrient screening uses project bands and published pH discussion. "
            "It does not prescribe fertiliser, lime, or sulfur rates."
        ),
    }


def _drainage(features: dict) -> dict:
    reasons = []
    missing = []
    points = 0
    drainage = _num(features.get("drainage_score"))
    compaction = _num(features.get("soil_compaction_score"))
    moisture = _num(features.get("soil_moisture_pct"))
    rainfall = _num(features.get("rainfall_7d_mm"))
    texture = features.get("soil_texture")
    for name, value in (
        ("drainage_score", drainage),
        ("soil_compaction_score", compaction),
        ("soil_moisture_pct", moisture),
        ("rainfall_7d_mm", rainfall),
    ):
        if value is None:
            missing.append(name)
    if _missing(texture):
        missing.append("soil_texture")

    if drainage is not None and drainage <= 2:
        points += 2
        reasons.append(
            f"Drainage score is {int(drainage)} on a 1–5 scale (1 is very poor). "
            "University of Maryland Extension advises avoiding sites that hold water and drain slowly."
        )
    elif drainage is not None and rainfall is not None and drainage == 3 and rainfall >= 40:
        points += 1
        reasons.append(
            f"Drainage is only moderate (score 3) and seven-day rainfall is {rainfall:.1f} mm."
        )
    if compaction is not None and compaction >= 4:
        points += 1
        reasons.append(
            f"Compaction score is {int(compaction)} of 5. Compacted soil drains more slowly and can be hard to re-wet, as described by Maryland Extension."
        )
    if texture in {"clay", "clay_loam"} and compaction is not None and compaction >= 4:
        points += 1
        reasons.append(
            f"Texture is {texture.replace('_', ' ')} and compaction is high. Finer soils are treated as more compaction-prone in this simulator."
        )
    if moisture is not None and drainage is not None and moisture >= 38 and drainage <= 3:
        points += 1
        reasons.append(
            f"Soil moisture is {moisture:.1f}% while drainage score is {int(drainage)}."
        )
    unavailable = drainage is None and compaction is None
    return {
        "indicator": "drainage",
        "level": _level(points, unavailable),
        "points": points,
        "reasons": reasons,
        "missing_fields": missing,
        "limitation": (
            "A drainage flag means the entered ratings deserve a site look. "
            "It does not design a drain, and aeration alone does not fix a saturated site."
        ),
    }
