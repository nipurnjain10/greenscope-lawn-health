"""Portable project configuration.

Paths are resolved from this file so the project can be moved to another
directory. No personal machine paths are stored here.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
SUBMISSION_DIR = ROOT / "submission"
OUTPUT_DIR = ROOT / "output"

GENERATOR_VERSION = "1.1.0"
SEED_MAIN = 42
SEED_ROBUSTNESS = 20261022
N_MAIN = 10_000
N_ROBUSTNESS = 2_000

# Gaussian noise added to the latent stress score before tertile labelling.
# Chosen during generator development (see data/metadata.json notes).
LABEL_NOISE_SD = 0.15

TARGET_COLUMN = "overall_lawn_health_risk"
ID_COLUMN = "inspection_id"
CLASS_ORDER = ["Low", "Moderate", "High"]

# Columns that must never be presented to a classifier.
LEAKAGE_COLUMNS = [
    TARGET_COLUMN,
    "risk_score",
    "latent_score",
    "water_stress",
    "nutrient_stress",
    "drainage_stress",
    "shade_stress",
    "neglect_stress",
    "water_latent",
    "heat_latent",
    "chem_latent",
    "care_latent",
    "site_latent",
]

CATEGORICAL_FEATURES = ["grass_type", "soil_texture", "foot_traffic_level"]

GRASS_TYPES = [
    "tall_fescue",
    "cool_season_blend",
    "kentucky_bluegrass",
    "fine_fescue",
    "perennial_ryegrass",
    "zoysiagrass",
    "bermudagrass",
]
WARM_SEASON_GRASSES = ["bermudagrass", "zoysiagrass"]
SOIL_TEXTURES = ["sand", "sandy_loam", "loam", "silt_loam", "clay_loam", "clay"]
TRAFFIC_LEVELS = ["low", "moderate", "high"]

# Grass mix for the primary sample. Tall fescue is the largest share because
# University of Maryland Extension states that most Maryland lawns are
# dominated by tall fescue. The remaining shares are author assumptions.
GRASS_PROB_MAIN = {
    "tall_fescue": 0.34,
    "cool_season_blend": 0.18,
    "kentucky_bluegrass": 0.14,
    "fine_fescue": 0.12,
    "perennial_ryegrass": 0.10,
    "zoysiagrass": 0.07,
    "bermudagrass": 0.05,
}
# Robustness scenario: a deliberately warmer species mix, not a climate projection.
GRASS_PROB_ROBUSTNESS = {
    "bermudagrass": 0.22,
    "zoysiagrass": 0.18,
    "tall_fescue": 0.20,
    "cool_season_blend": 0.12,
    "kentucky_bluegrass": 0.10,
    "perennial_ryegrass": 0.10,
    "fine_fescue": 0.08,
}

TEXTURE_PROB = {
    "loam": 0.28,
    "sandy_loam": 0.22,
    "silt_loam": 0.16,
    "clay_loam": 0.16,
    "sand": 0.10,
    "clay": 0.08,
}

# Base probability that a laboratory field is hidden. The probability is
# multiplied by 1.7 when days_since_last_service >= 100 (missingness depends
# on maintenance recency, not on the label).
MISSING_RATES_MAIN = {
    "nitrogen_mg_kg": 0.09,
    "phosphorus_mg_kg": 0.045,
    "potassium_mg_kg": 0.045,
    "soil_ph": 0.03,
}
MISSING_RATES_ROBUSTNESS = {
    "nitrogen_mg_kg": 0.14,
    "phosphorus_mg_kg": 0.08,
    "potassium_mg_kg": 0.08,
    "soil_ph": 0.06,
}

FEATURE_SPECS: dict[str, dict] = {
    "lawn_area_m2": {
        "group": "Lawn and site",
        "kind": "numeric",
        "unit": "square metres",
        "minimum": 40.0,
        "maximum": 2200.0,
        "decimals": 1,
        "source": "field",
        "description": "Estimated turf area of the inspected lawn.",
        "role": "Scales service quantities. Not a biological risk driver in the label.",
    },
    "grass_type": {
        "group": "Lawn and site",
        "kind": "categorical",
        "unit": "category",
        "levels": GRASS_TYPES,
        "source": "field",
        "description": "Dominant turfgrass group reported for the lawn.",
        "role": "Modifies how heat enters the latent water-stress term. Warm-season types are bermudagrass and zoysiagrass.",
    },
    "sunlight_hours_per_day": {
        "group": "Lawn and site",
        "kind": "numeric",
        "unit": "hours per day of direct sun",
        "minimum": 1.5,
        "maximum": 13.0,
        "decimals": 1,
        "source": "field",
        "description": "Typical direct sunlight reaching the turf after trees have leafed out.",
        "role": "Shade stress rises as hours fall below 6, following the extension statement that full sun is about 6 or more hours and that lawn grasses do not thrive in little direct sun.",
    },
    "soil_texture": {
        "group": "Lawn and site",
        "kind": "categorical",
        "unit": "category",
        "levels": SOIL_TEXTURES,
        "source": "field",
        "description": "Dominant mineral texture class of the root-zone soil.",
        "role": "Shifts drainage and how much water the soil holds. Author-assigned texture effects, not a copied soil-survey table.",
    },
    "foot_traffic_level": {
        "group": "Lawn and site",
        "kind": "categorical",
        "unit": "category",
        "levels": TRAFFIC_LEVELS,
        "source": "worker",
        "description": "Worker judgement of routine foot traffic.",
        "role": "Increases the chance of compaction. Smaller lawns are more often coded as higher traffic.",
    },
    "soil_ph": {
        "group": "Soil measurements",
        "kind": "numeric",
        "unit": "pH",
        "minimum": 4.7,
        "maximum": 8.1,
        "decimals": 2,
        "source": "soil_test",
        "description": "Simulated laboratory soil pH. May be missing when no recent test is available.",
        "role": "Departure from mildly acidic turf conditions contributes to nutrient stress. UMD Extension treats about 6.0–6.8 as a favourable band and flags pH below 5.5 for liming review.",
    },
    "nitrogen_mg_kg": {
        "group": "Soil measurements",
        "kind": "numeric",
        "unit": "mg/kg, simulated mineral nitrogen on a project scale",
        "minimum": 1.0,
        "maximum": 80.0,
        "decimals": 1,
        "source": "soil_test",
        "description": "Synthetic proxy for relative mineral-nitrogen supply. It is not a Rutgers or Maryland fertilizer recommendation.",
        "role": "A noisy view of the soil-chemistry latent variable. Rutgers Extension states that a routine soil test has not been developed to measure turfgrass nitrogen need.",
    },
    "phosphorus_mg_kg": {
        "group": "Soil measurements",
        "kind": "numeric",
        "unit": "mg/kg, simulated extractable phosphorus on a project scale",
        "minimum": 2.0,
        "maximum": 150.0,
        "decimals": 1,
        "source": "soil_test",
        "description": "Synthetic extractable phosphorus. Not a Mehlich-3 or Rutgers sufficiency category.",
        "role": "Noisy view of the soil-chemistry latent variable and an input to the transparent nutrient screen.",
    },
    "potassium_mg_kg": {
        "group": "Soil measurements",
        "kind": "numeric",
        "unit": "mg/kg, simulated extractable potassium on a project scale",
        "minimum": 20.0,
        "maximum": 400.0,
        "decimals": 1,
        "source": "soil_test",
        "description": "Synthetic extractable potassium. Not a laboratory sufficiency category.",
        "role": "Noisy view of the soil-chemistry latent variable and an input to the transparent nutrient screen.",
    },
    "soil_moisture_pct": {
        "group": "Soil measurements",
        "kind": "numeric",
        "unit": "percent volumetric water, simulated",
        "minimum": 4.0,
        "maximum": 48.0,
        "decimals": 1,
        "source": "field",
        "description": "Simulated root-zone moisture at inspection.",
        "role": "Generated from rainfall, irrigation, drainage, texture, temperature, and a water-supply latent variable.",
    },
    "drainage_score": {
        "group": "Soil measurements",
        "kind": "numeric",
        "unit": "ordinal score 1 (very poor) to 5 (excellent)",
        "minimum": 1,
        "maximum": 5,
        "decimals": 0,
        "source": "worker",
        "description": "Worker rating of how freely water leaves the root zone.",
        "role": "Lower scores raise drainage stress, especially together with compaction, high moisture, or heavy recent rain.",
    },
    "soil_compaction_score": {
        "group": "Soil measurements",
        "kind": "numeric",
        "unit": "ordinal score 1 (none) to 5 (severe)",
        "minimum": 1,
        "maximum": 5,
        "decimals": 0,
        "source": "worker",
        "description": "Worker rating of surface compaction.",
        "role": "Rises with traffic and finer texture. Compaction lowers the drainage score and adds to water and drainage stress.",
    },
    "temperature_c": {
        "group": "Environment",
        "kind": "numeric",
        "unit": "degrees Celsius, air temperature at inspection",
        "minimum": 7.0,
        "maximum": 38.5,
        "decimals": 1,
        "source": "field",
        "description": "Air temperature during the inspection.",
        "role": "A noisy observation of the heat latent variable. Cool-season grasses receive additional latent heat stress; warm-season grasses receive less.",
    },
    "humidity_pct": {
        "group": "Environment",
        "kind": "numeric",
        "unit": "percent relative humidity",
        "minimum": 30.0,
        "maximum": 96.0,
        "decimals": 0,
        "source": "field",
        "description": "Relative humidity at inspection.",
        "role": "Generated with a negative association with temperature. It is context, not a dominant label term.",
    },
    "rainfall_7d_mm": {
        "group": "Environment",
        "kind": "numeric",
        "unit": "millimetres over the preceding 7 days",
        "minimum": 0.0,
        "maximum": 125.0,
        "decimals": 1,
        "source": "field",
        "description": "Rainfall total for the seven days before inspection.",
        "role": "Raises moisture and can raise drainage stress when drainage is already limited. UMD Extension cites about 1 inch (25 mm) of water per week for actively growing turf; that figure is background, not a label threshold.",
    },
    "irrigation_sessions_per_week": {
        "group": "Maintenance",
        "kind": "numeric",
        "unit": "sessions per week",
        "minimum": 0,
        "maximum": 7,
        "decimals": 0,
        "source": "worker",
        "description": "How many times the lawn was irrigated in a typical recent week.",
        "role": "Rises with caretaker effort and heat, and falls after heavier rain. Frequent sessions do not guarantee deep moisture.",
    },
    "mowing_sessions_per_month": {
        "group": "Maintenance",
        "kind": "numeric",
        "unit": "sessions per month",
        "minimum": 0,
        "maximum": 12,
        "decimals": 0,
        "source": "worker",
        "description": "Mowing visits in the recent month.",
        "role": "A partial view of maintenance effort. Very few sessions increase neglect stress unless growth has paused; the screen therefore asks the user to confirm that the lawn is actively growing.",
    },
    "days_since_last_service": {
        "group": "Maintenance",
        "kind": "numeric",
        "unit": "days",
        "minimum": 2,
        "maximum": 170,
        "decimals": 0,
        "source": "worker",
        "description": "Days since the last organised lawn service or equivalent care visit.",
        "role": "Longer gaps increase neglect stress and the chance that laboratory measurements are missing.",
    },
    "discoloration_pct": {
        "group": "Visible observations",
        "kind": "numeric",
        "unit": "percent of turf area",
        "minimum": 0.0,
        "maximum": 82.0,
        "decimals": 1,
        "source": "worker",
        "description": "Share of the lawn with off-colour turf. Not a disease diagnosis.",
        "role": "Noisy visible consequence of the latent stresses, plus independent observation noise.",
    },
    "bare_patch_pct": {
        "group": "Visible observations",
        "kind": "numeric",
        "unit": "percent of turf area",
        "minimum": 0.0,
        "maximum": 68.0,
        "decimals": 1,
        "source": "worker",
        "description": "Share of the lawn with little or no turf cover.",
        "role": "Noisy visible consequence of latent stress. A single bare-patch value does not set the label.",
    },
    "wilting_score": {
        "group": "Visible observations",
        "kind": "numeric",
        "unit": "ordinal score 0 (none) to 4 (severe)",
        "minimum": 0,
        "maximum": 4,
        "decimals": 0,
        "source": "worker",
        "description": "Worker rating of wilt, footprinting, or folded leaves.",
        "role": "More strongly tied to the water-stress latent variable than the other symptoms, but still noisy. UMD Extension describes dull colour and footprinting as drought signs; this score is a project scale for those kinds of observations.",
    },
    "thatch_thickness_mm": {
        "group": "Visible observations",
        "kind": "numeric",
        "unit": "millimetres",
        "minimum": 0.0,
        "maximum": 36.0,
        "decimals": 1,
        "source": "worker",
        "description": "Estimated thatch thickness at the soil surface.",
        "role": "Tends to be higher when nutrient supply and watering are high. Thickness above about 12.7 mm (one-half inch) adds to neglect stress. That half-inch figure is the UMD Extension dethatching discussion, used here only as a screen.",
    },
    "weed_cover_pct": {
        "group": "Visible observations",
        "kind": "numeric",
        "unit": "percent of turf area",
        "minimum": 0.0,
        "maximum": 72.0,
        "decimals": 1,
        "source": "worker",
        "description": "Estimated area where weeds are visually prominent. Species are not identified.",
        "role": "Rises with low maintenance effort, shade, and bare ground. It does not trigger a herbicide rate.",
    },
}

FEATURE_COLUMNS = list(FEATURE_SPECS.keys())
NUMERIC_FEATURES = [
    name for name, spec in FEATURE_SPECS.items() if spec["kind"] == "numeric"
]

# Wider than the simulation bounds. Values inside these limits but outside
# the simulation bounds are accepted with a warning. Values outside are rejected.
UI_LIMITS = {
    "lawn_area_m2": (20.0, 5000.0),
    "sunlight_hours_per_day": (0.0, 16.0),
    "soil_ph": (3.5, 10.0),
    "nitrogen_mg_kg": (0.0, 150.0),
    "phosphorus_mg_kg": (0.0, 300.0),
    "potassium_mg_kg": (0.0, 800.0),
    "soil_moisture_pct": (0.0, 70.0),
    "drainage_score": (1, 5),
    "soil_compaction_score": (1, 5),
    "temperature_c": (-5.0, 48.0),
    "humidity_pct": (5.0, 100.0),
    "rainfall_7d_mm": (0.0, 300.0),
    "irrigation_sessions_per_week": (0, 14),
    "mowing_sessions_per_month": (0, 20),
    "days_since_last_service": (0, 400),
    "discoloration_pct": (0.0, 100.0),
    "bare_patch_pct": (0.0, 100.0),
    "wilting_score": (0, 4),
    "thatch_thickness_mm": (0.0, 80.0),
    "weed_cover_pct": (0.0, 100.0),
}

MODEL_SPECS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "svm_rbf": "SVM (RBF)",
}
ENSEMBLE_SPECS = {
    "soft_voting": "Soft voting",
}
ALL_MODEL_SPECS = {**MODEL_SPECS, **ENSEMBLE_SPECS}

SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_SEED = 42

DISCLOSURE = (
    "Independent academic prototype. Model development and evaluation use synthetic data; "
    "performance has not been validated on actual customer lawns."
)

# Illustrative planning prices. These are not market quotations.
DEFAULT_CURRENCY = "INR"
PRICE_BOOKS = {
    "INR": {
        "currency": "INR",
        "symbol": "₹",
        "items": {
            "follow_up_visit": {"label": "Follow-up inspection visit", "flat": 700.0, "per_m2": 0.0},
            "soil_test_review": {"label": "Soil-test review visit (excludes laboratory fees)", "flat": 650.0, "per_m2": 0.0},
            "irrigation_inspection": {"label": "Irrigation coverage inspection", "flat": 600.0, "per_m2": 2.0},
            "drainage_investigation": {"label": "Drainage investigation", "flat": 1200.0, "per_m2": 3.5},
            "aeration": {"label": "Core aeration", "flat": 0.0, "per_m2": 4.0},
            "thatch_service": {"label": "Thatch assessment and dethatching", "flat": 500.0, "per_m2": 6.0},
            "sunlight_review": {"label": "Sunlight and grass-suitability review", "flat": 500.0, "per_m2": 0.0},
            "nutrient_review": {"label": "Nutrient-management review (no product or rate)", "flat": 800.0, "per_m2": 0.0},
            "mowing_visit": {"label": "One mowing visit", "flat": 0.0, "per_m2": 2.2},
            "weed_review": {"label": "Weed identification and density review", "flat": 450.0, "per_m2": 0.0},
        },
    },
    "USD": {
        "currency": "USD",
        "symbol": "$",
        "items": {
            "follow_up_visit": {"label": "Follow-up inspection visit", "flat": 28.0, "per_m2": 0.0},
            "soil_test_review": {"label": "Soil-test review visit (excludes laboratory fees)", "flat": 25.0, "per_m2": 0.0},
            "irrigation_inspection": {"label": "Irrigation coverage inspection", "flat": 24.0, "per_m2": 0.08},
            "drainage_investigation": {"label": "Drainage investigation", "flat": 48.0, "per_m2": 0.14},
            "aeration": {"label": "Core aeration", "flat": 0.0, "per_m2": 0.16},
            "thatch_service": {"label": "Thatch assessment and dethatching", "flat": 20.0, "per_m2": 0.24},
            "sunlight_review": {"label": "Sunlight and grass-suitability review", "flat": 20.0, "per_m2": 0.0},
            "nutrient_review": {"label": "Nutrient-management review (no product or rate)", "flat": 32.0, "per_m2": 0.0},
            "mowing_visit": {"label": "One mowing visit", "flat": 0.0, "per_m2": 0.09},
            "weed_review": {"label": "Weed identification and density review", "flat": 18.0, "per_m2": 0.0},
        },
    },
}
