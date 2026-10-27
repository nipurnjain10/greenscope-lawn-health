"""Source-informed synthetic lawn-inspection generator.

Each row is a simulated inspection, not a customer, property, or company record.
Latent variables and the continuous stress score are used only to create the
features and the three-class label. They are not written to the CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    GENERATOR_VERSION,
    GRASS_PROB_MAIN,
    GRASS_PROB_ROBUSTNESS,
    GRASS_TYPES,
    ID_COLUMN,
    LABEL_NOISE_SD,
    MISSING_RATES_MAIN,
    MISSING_RATES_ROBUSTNESS,
    N_MAIN,
    N_ROBUSTNESS,
    SEED_MAIN,
    SEED_ROBUSTNESS,
    SOIL_TEXTURES,
    TARGET_COLUMN,
    TEXTURE_PROB,
    WARM_SEASON_GRASSES,
)
from src.utilities import utc_now_iso

DRAINAGE_BASE = {
    "sand": 4.2,
    "sandy_loam": 3.9,
    "loam": 3.3,
    "silt_loam": 3.0,
    "clay_loam": 2.5,
    "clay": 2.0,
}
TEXTURE_MOISTURE = {
    "sand": -4.0,
    "sandy_loam": -2.0,
    "loam": 0.5,
    "silt_loam": 1.5,
    "clay_loam": 3.0,
    "clay": 4.5,
}
CLAY_COMPACTION = {"sand": 0.0, "sandy_loam": 0.05, "loam": 0.15, "silt_loam": 0.25, "clay_loam": 0.45, "clay": 0.75}


def _choice(rng: np.random.Generator, levels: list[str], probabilities: dict[str, float], n: int) -> np.ndarray:
    weights = np.array([probabilities[level] for level in levels], dtype=float)
    weights = weights / weights.sum()
    return rng.choice(np.array(levels, dtype=object), size=n, p=weights)


def _round_spec(values: np.ndarray, name: str) -> np.ndarray:
    spec = FEATURE_SPECS[name]
    decimals = int(spec["decimals"])
    lower = spec["minimum"]
    upper = spec["maximum"]
    clipped = np.clip(values, lower, upper)
    if decimals == 0:
        return np.rint(clipped).astype(int)
    return np.round(clipped, decimals)


def _apply_missing(
    rng: np.random.Generator,
    frame: pd.DataFrame,
    days: np.ndarray,
    rates: dict[str, float],
) -> dict[str, float]:
    observed = {}
    for column, base in rates.items():
        probability = np.clip(base * np.where(days >= 100, 1.7, 1.0), 0.0, 0.55)
        mask = rng.random(len(frame)) < probability
        frame.loc[mask, column] = np.nan
        observed[column] = float(mask.mean())
    return observed


def generate_inspections(
    n: int,
    seed: int,
    scenario: str = "main",
    thresholds: tuple[float, float] | None = None,
    id_prefix: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Generate ``n`` inspections and the class thresholds used to label them.

    For the main scenario, thresholds are the tertiles of this sample's latent
    score. For the robustness scenario, pass the frozen main-sample thresholds
    so the class definition does not get rebalanced.
    """
    if n < 30:
        raise ValueError("Generate at least 30 rows so class tertiles are defined.")
    if scenario not in {"main", "robustness"}:
        raise ValueError(f"Unknown scenario: {scenario}")
    if scenario != "main" and thresholds is None:
        raise ValueError("Robustness generation requires the frozen primary-sample thresholds.")

    rng = np.random.default_rng(seed)
    if scenario == "main":
        heat = rng.normal(0.0, 1.0, n)
        water = rng.normal(0.0, 1.0, n)
        grass_prob = GRASS_PROB_MAIN
        rain_scale = 1.0
        traffic_bias = 0.0
        missing_rates = MISSING_RATES_MAIN
        prefix = id_prefix or "GS"
    else:
        heat = rng.normal(0.85, 1.05, n)
        water = rng.normal(-0.55, 1.05, n)
        grass_prob = GRASS_PROB_ROBUSTNESS
        rain_scale = 0.62
        traffic_bias = 0.35
        missing_rates = MISSING_RATES_ROBUSTNESS
        prefix = id_prefix or "RB"

    chem = rng.normal(0.0, 1.0, n)
    care = rng.normal(0.0, 1.0, n)
    site = rng.normal(0.0, 1.0, n)

    grass = _choice(rng, GRASS_TYPES, grass_prob, n)
    texture = _choice(rng, SOIL_TEXTURES, TEXTURE_PROB, n)
    warm = np.isin(grass, WARM_SEASON_GRASSES)

    sunlight = np.clip(rng.normal(7.0, 2.3, n), 1.5, 13.0)
    area = np.clip(rng.lognormal(5.45, 0.48, n), 40.0, 2200.0)

    traffic_score = (
        rng.normal(0.0, 1.0, n)
        + np.where(area < 120, 0.55, 0.0)
        + np.where(area > 800, -0.40, 0.0)
        + traffic_bias
    )
    traffic = np.where(traffic_score < -0.35, "low", np.where(traffic_score < 0.55, "moderate", "high"))
    traffic_num = np.where(traffic == "low", 0.0, np.where(traffic == "moderate", 1.0, 2.0))

    clay_bonus = np.array([CLAY_COMPACTION[value] for value in texture])
    compaction_cont = 2.15 + 0.82 * traffic_num + clay_bonus - 0.22 * care + rng.normal(0.0, 0.48, n)
    compaction = _round_spec(compaction_cont, "soil_compaction_score")

    drain_base = np.array([DRAINAGE_BASE[value] for value in texture])
    drainage_cont = drain_base - 0.30 * (compaction - 2.0) + 0.12 * site + rng.normal(0.0, 0.42, n)
    drainage = _round_spec(drainage_cont, "drainage_score")

    temperature = np.clip(21.5 + 6.2 * heat + rng.normal(0.0, 1.8, n), 7.0, 38.5)
    humidity = np.clip(72.0 - 0.85 * (temperature - 21.0) + rng.normal(0.0, 6.5, n), 30.0, 96.0)
    rainfall = rng.gamma(1.7, 10.5, n) * rain_scale
    rainfall = np.clip(rainfall * (1.0 - 0.10 * np.maximum(heat, 0.0)), 0.0, 125.0)

    irrigation_cont = (
        1.55
        + 0.80 * care
        + 0.50 * np.maximum(heat, 0.0)
        - 0.022 * rainfall
        + rng.normal(0.0, 0.65, n)
    )
    irrigation = _round_spec(irrigation_cont, "irrigation_sessions_per_week")

    mowing_cont = 3.3 + 1.05 * care + 0.28 * np.maximum(heat, -0.4) + rng.normal(0.0, 0.75, n)
    mowing_cont = np.where((~warm) & (temperature > 33.0), mowing_cont - 1.4, mowing_cont)
    mowing = _round_spec(mowing_cont, "mowing_sessions_per_month")

    days = np.clip(np.rint(42 - 16 * care + rng.normal(0.0, 14.0, n)), 2, 170)

    ph = np.clip(6.35 + rng.normal(0.0, 0.48, n) - 0.10 * chem, 4.7, 8.1)
    nitrogen = np.exp(rng.normal(2.55 + 0.28 * chem + 0.12 * care, 0.42, n))
    phosphorus = np.exp(rng.normal(3.05 + 0.32 * chem + 0.08 * care, 0.40, n))
    potassium = np.exp(rng.normal(4.65 + 0.28 * chem + 0.08 * care, 0.35, n))

    texture_moisture = np.array([TEXTURE_MOISTURE[value] for value in texture])
    moisture = (
        16.5
        + 3.1 * water
        + 0.15 * rainfall
        + 1.10 * irrigation
        + texture_moisture
        - 1.25 * (drainage - 3.0)
        - 0.16 * (temperature - 22.0)
        + rng.normal(0.0, 2.6, n)
    )

    # The exported label is a noisy function of the simulated measurements.
    # Hidden latents create dependence among those measurements, but they are
    # not terms in the score. Generator 1.0.0 did the reverse and a validation
    # pilot could not recover the label (about 0.62–0.64). These component
    # variables are not written to the CSV.
    thatch = (
        4.0
        + 2.0 * np.maximum(chem, 0.0)
        + 1.2 * np.maximum(water, 0.0)
        + 0.45 * irrigation
        + rng.normal(0.0, 3.2, n)
    )
    moisture_term = (22.0 - moisture) / 7.0
    heat_term = (temperature - 22.0) / 7.0
    heat_drought = np.maximum(heat_term, 0.0) * np.maximum(moisture_term, 0.0)
    cool_heat = np.where(warm, 0.25, 1.0) * heat_term
    low_rain = (16.0 - rainfall) / 18.0 * np.maximum(2.5 - irrigation, 0.0) / 2.5
    water_stress = 0.85 * moisture_term + 0.95 * heat_drought + 0.40 * cool_heat + 0.30 * low_rain

    ph_center = np.where(warm, 6.6, 6.4)
    ph_term = np.abs(ph - ph_center) / 0.55
    low_p = (26.0 - phosphorus) / 16.0
    low_k = (115.0 - potassium) / 55.0
    low_n = (16.0 - nitrogen) / 9.0
    nutrient_stress = (
        0.75 * ph_term
        + 0.40 * low_p
        + 0.30 * low_k
        + 0.22 * low_n
        + 0.35 * ph_term * np.maximum(low_p, 0.0)
    )
    drainage_stress = (
        0.70 * (3.2 - drainage)
        + 0.35 * (compaction - 2.5)
        + 0.45 * np.maximum(moisture - 28.0, 0.0) / 8.0 * np.maximum(3.2 - drainage, 0.0)
        + 0.20 * np.maximum(rainfall - 30.0, 0.0) / 25.0 * np.maximum(3.2 - drainage, 0.0)
    )
    shade_stress = (6.5 - sunlight) / 3.5
    neglect_stress = (
        (days - 40.0) / 45.0
        + 0.28 * (3.0 - mowing)
        + 0.22 * np.maximum(thatch - 12.7, 0.0) / 8.0
    )
    symptom_drive = (
        0.45 * water_stress
        + 0.28 * nutrient_stress
        + 0.28 * drainage_stress
        + 0.16 * neglect_stress
        + 0.16 * shade_stress
    )
    discoloration = 6.0 + 6.5 * symptom_drive + rng.normal(0.0, 4.5, n)
    bare = 3.0 + 3.6 * symptom_drive + rng.normal(0.0, 3.5, n)
    wilting = 1.1 + 0.55 * moisture_term + 0.70 * heat_drought + rng.normal(0.0, 0.55, n)
    weed = (
        8.0
        + 4.0 * np.maximum(-care, 0.0)
        + 3.0 * np.maximum(shade_stress, 0.0)
        + 0.08 * np.maximum(bare, 0.0)
        + rng.normal(0.0, 5.0, n)
    )
    symptom_term = (
        np.clip(discoloration, 0, 82) / 25.0
        + np.clip(bare, 0, 68) / 20.0
        + np.clip(wilting, 0, 4) / 2.2
    )
    score = (
        water_stress
        + 0.85 * nutrient_stress
        + 0.80 * drainage_stress
        + 0.45 * shade_stress
        + 0.40 * neglect_stress
        + 0.30 * symptom_term
        + rng.normal(0.0, LABEL_NOISE_SD, n)
    )
    if thresholds is None:
        low_cut, high_cut = [float(value) for value in np.quantile(score, [1.0 / 3.0, 2.0 / 3.0])]
    else:
        low_cut, high_cut = float(thresholds[0]), float(thresholds[1])
    labels = np.where(score < low_cut, "Low", np.where(score >= high_cut, "High", "Moderate"))

    frame = pd.DataFrame(
        {
            ID_COLUMN: [f"{prefix}-{index + 1:06d}" for index in range(n)],
            "lawn_area_m2": _round_spec(area, "lawn_area_m2"),
            "grass_type": grass.astype(str),
            "sunlight_hours_per_day": _round_spec(sunlight, "sunlight_hours_per_day"),
            "soil_texture": texture.astype(str),
            "foot_traffic_level": traffic.astype(str),
            "soil_ph": _round_spec(ph, "soil_ph"),
            "nitrogen_mg_kg": _round_spec(nitrogen, "nitrogen_mg_kg"),
            "phosphorus_mg_kg": _round_spec(phosphorus, "phosphorus_mg_kg"),
            "potassium_mg_kg": _round_spec(potassium, "potassium_mg_kg"),
            "soil_moisture_pct": _round_spec(moisture, "soil_moisture_pct"),
            "drainage_score": drainage.astype(int),
            "soil_compaction_score": compaction.astype(int),
            "temperature_c": _round_spec(temperature, "temperature_c"),
            "humidity_pct": _round_spec(humidity, "humidity_pct"),
            "rainfall_7d_mm": _round_spec(rainfall, "rainfall_7d_mm"),
            "irrigation_sessions_per_week": irrigation.astype(int),
            "mowing_sessions_per_month": mowing.astype(int),
            "days_since_last_service": _round_spec(days, "days_since_last_service").astype(int),
            "discoloration_pct": _round_spec(discoloration, "discoloration_pct"),
            "bare_patch_pct": _round_spec(bare, "bare_patch_pct"),
            "wilting_score": _round_spec(wilting, "wilting_score").astype(int),
            "thatch_thickness_mm": _round_spec(thatch, "thatch_thickness_mm"),
            "weed_cover_pct": _round_spec(weed, "weed_cover_pct"),
            TARGET_COLUMN: labels.astype(str),
        }
    )
    missingness = _apply_missing(rng, frame, days, missing_rates)
    frame = frame.loc[:, [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]]

    summary = {
        "scenario": scenario,
        "n": int(n),
        "seed": int(seed),
        "generator_version": GENERATOR_VERSION,
        "label_noise_sd": LABEL_NOISE_SD,
        "thresholds": {"low_lt": low_cut, "high_ge": high_cut},
        "class_counts": {key: int(value) for key, value in frame[TARGET_COLUMN].value_counts().items()},
        "missingness_rate": missingness,
        "component_std": {
            "water_stress": float(np.std(water_stress)),
            "nutrient_stress": float(np.std(nutrient_stress)),
            "drainage_stress": float(np.std(drainage_stress)),
            "shade_stress": float(np.std(shade_stress)),
            "neglect_stress": float(np.std(neglect_stress)),
            "score": float(np.std(score)),
        },
        "created_at_utc": utc_now_iso(),
    }
    return frame, summary


def write_data_dictionary(path: Path) -> None:
    lines = [
        "# Data dictionary",
        "",
        "Primary file: `data/synthetic_lawn_inspections.csv`.",
        "Robustness file: `data/robustness_scenarios.csv`.",
        "",
        "Every row is a **simulated lawn inspection** created by `data/generate_synthetic_data.py`.",
        "The files do not contain customers, addresses, names, or company records.",
        "Nutrient columns use a project-defined mg/kg scale. They are not Rutgers",
        "sufficiency categories and they are not fertilizer grades.",
        "",
        f"Generator version: `{GENERATOR_VERSION}`. Label noise standard deviation: `{LABEL_NOISE_SD}`.",
        "",
        "The target `overall_lawn_health_risk` is a tertile of a noisy latent stress score",
        "on the primary sample. `Low` means the score is below the primary-sample 33rd",
        "percentile. `High` means the score is at or above the 67th percentile.",
        "`Moderate` is the remainder. The same numeric cutoffs are applied to the",
        "robustness sample. These classes are simulated risk bins, not a verified",
        "biological diagnosis.",
        "",
        "Laboratory fields may be missing. Missingness is more likely when",
        "`days_since_last_service` is at least 100. The label is computed before",
        "those values are hidden, and the hidden values are not stored.",
        "",
        "## Identifier",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| `inspection_id` | Synthetic identifier. Primary rows use `GS-######`. Robustness rows use `RB-######`. |",
        "",
        "## Target",
        "",
        "| Column | Values | Meaning |",
        "| --- | --- | --- |",
        "| `overall_lawn_health_risk` | `Low`, `Moderate`, `High` | Simulated overall lawn-health risk bin. Not an input feature. |",
        "",
        "## Input features",
        "",
        "| Column | Group | Kind | Unit | Simulation bounds | How it is obtained in the app |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, spec in FEATURE_SPECS.items():
        if spec["kind"] == "categorical":
            bounds = ", ".join(spec["levels"])
        else:
            bounds = f"{spec['minimum']} to {spec['maximum']}"
        lines.append(
            f"| `{name}` | {spec['group']} | {spec['kind']} | {spec['unit']} | {bounds} | {spec['source']} |"
        )
    lines.extend(["", "## Feature notes", ""])
    for name, spec in FEATURE_SPECS.items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(spec["description"])
        lines.append("")
        lines.append(spec["role"])
        lines.append("")
    lines.extend(
        [
            "## What is deliberately absent",
            "",
            "The CSV does not contain the latent variables (`water_latent`, `heat_latent`,",
            "`chem_latent`, `care_latent`, `site_latent`), the component stresses, or the",
            "continuous risk score. Those quantities would leak the label-generating process.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
