# Data dictionary

Primary file: `data/synthetic_lawn_inspections.csv`.
Robustness file: `data/robustness_scenarios.csv`.

Every row is a **simulated lawn inspection** created by `data/generate_synthetic_data.py`.
The files do not contain customers, addresses, names, or company records.
Nutrient columns use a project-defined mg/kg scale. They are not Rutgers
sufficiency categories and they are not fertilizer grades.

Generator version: `1.1.0`. Label noise standard deviation: `0.15`.

The target `overall_lawn_health_risk` is a tertile of a noisy latent stress score
on the primary sample. `Low` means the score is below the primary-sample 33rd
percentile. `High` means the score is at or above the 67th percentile.
`Moderate` is the remainder. The same numeric cutoffs are applied to the
robustness sample. These classes are simulated risk bins, not a verified
biological diagnosis.

Laboratory fields may be missing. Missingness is more likely when
`days_since_last_service` is at least 100. The label is computed before
those values are hidden, and the hidden values are not stored.

## Identifier

| Column | Meaning |
| --- | --- |
| `inspection_id` | Synthetic identifier. Primary rows use `GS-######`. Robustness rows use `RB-######`. |

## Target

| Column | Values | Meaning |
| --- | --- | --- |
| `overall_lawn_health_risk` | `Low`, `Moderate`, `High` | Simulated overall lawn-health risk bin. Not an input feature. |

## Input features

| Column | Group | Kind | Unit | Simulation bounds | How it is obtained in the app |
| --- | --- | --- | --- | --- | --- |
| `lawn_area_m2` | Lawn and site | numeric | square metres | 40.0 to 2200.0 | field |
| `grass_type` | Lawn and site | categorical | category | tall_fescue, cool_season_blend, kentucky_bluegrass, fine_fescue, perennial_ryegrass, zoysiagrass, bermudagrass | field |
| `sunlight_hours_per_day` | Lawn and site | numeric | hours per day of direct sun | 1.5 to 13.0 | field |
| `soil_texture` | Lawn and site | categorical | category | sand, sandy_loam, loam, silt_loam, clay_loam, clay | field |
| `foot_traffic_level` | Lawn and site | categorical | category | low, moderate, high | worker |
| `soil_ph` | Soil measurements | numeric | pH | 4.7 to 8.1 | soil_test |
| `nitrogen_mg_kg` | Soil measurements | numeric | mg/kg, simulated mineral nitrogen on a project scale | 1.0 to 80.0 | soil_test |
| `phosphorus_mg_kg` | Soil measurements | numeric | mg/kg, simulated extractable phosphorus on a project scale | 2.0 to 150.0 | soil_test |
| `potassium_mg_kg` | Soil measurements | numeric | mg/kg, simulated extractable potassium on a project scale | 20.0 to 400.0 | soil_test |
| `soil_moisture_pct` | Soil measurements | numeric | percent volumetric water, simulated | 4.0 to 48.0 | field |
| `drainage_score` | Soil measurements | numeric | ordinal score 1 (very poor) to 5 (excellent) | 1 to 5 | worker |
| `soil_compaction_score` | Soil measurements | numeric | ordinal score 1 (none) to 5 (severe) | 1 to 5 | worker |
| `temperature_c` | Environment | numeric | degrees Celsius, air temperature at inspection | 7.0 to 38.5 | field |
| `humidity_pct` | Environment | numeric | percent relative humidity | 30.0 to 96.0 | field |
| `rainfall_7d_mm` | Environment | numeric | millimetres over the preceding 7 days | 0.0 to 125.0 | field |
| `irrigation_sessions_per_week` | Maintenance | numeric | sessions per week | 0 to 7 | worker |
| `mowing_sessions_per_month` | Maintenance | numeric | sessions per month | 0 to 12 | worker |
| `days_since_last_service` | Maintenance | numeric | days | 2 to 170 | worker |
| `discoloration_pct` | Visible observations | numeric | percent of turf area | 0.0 to 82.0 | worker |
| `bare_patch_pct` | Visible observations | numeric | percent of turf area | 0.0 to 68.0 | worker |
| `wilting_score` | Visible observations | numeric | ordinal score 0 (none) to 4 (severe) | 0 to 4 | worker |
| `thatch_thickness_mm` | Visible observations | numeric | millimetres | 0.0 to 36.0 | worker |
| `weed_cover_pct` | Visible observations | numeric | percent of turf area | 0.0 to 72.0 | worker |

## Feature notes

### `lawn_area_m2`

Estimated turf area of the inspected lawn.

Scales service quantities. Not a biological risk driver in the label.

### `grass_type`

Dominant turfgrass group reported for the lawn.

Modifies how heat enters the latent water-stress term. Warm-season types are bermudagrass and zoysiagrass.

### `sunlight_hours_per_day`

Typical direct sunlight reaching the turf after trees have leafed out.

Shade stress rises as hours fall below 6, following the extension statement that full sun is about 6 or more hours and that lawn grasses do not thrive in little direct sun.

### `soil_texture`

Dominant mineral texture class of the root-zone soil.

Shifts drainage and how much water the soil holds. Author-assigned texture effects, not a copied soil-survey table.

### `foot_traffic_level`

Worker judgement of routine foot traffic.

Increases the chance of compaction. Smaller lawns are more often coded as higher traffic.

### `soil_ph`

Simulated laboratory soil pH. May be missing when no recent test is available.

Departure from mildly acidic turf conditions contributes to nutrient stress. UMD Extension treats about 6.0–6.8 as a favourable band and flags pH below 5.5 for liming review.

### `nitrogen_mg_kg`

Synthetic proxy for relative mineral-nitrogen supply. It is not a Rutgers or Maryland fertilizer recommendation.

A noisy view of the soil-chemistry latent variable. Rutgers Extension states that a routine soil test has not been developed to measure turfgrass nitrogen need.

### `phosphorus_mg_kg`

Synthetic extractable phosphorus. Not a Mehlich-3 or Rutgers sufficiency category.

Noisy view of the soil-chemistry latent variable and an input to the transparent nutrient screen.

### `potassium_mg_kg`

Synthetic extractable potassium. Not a laboratory sufficiency category.

Noisy view of the soil-chemistry latent variable and an input to the transparent nutrient screen.

### `soil_moisture_pct`

Simulated root-zone moisture at inspection.

Generated from rainfall, irrigation, drainage, texture, temperature, and a water-supply latent variable.

### `drainage_score`

Worker rating of how freely water leaves the root zone.

Lower scores raise drainage stress, especially together with compaction, high moisture, or heavy recent rain.

### `soil_compaction_score`

Worker rating of surface compaction.

Rises with traffic and finer texture. Compaction lowers the drainage score and adds to water and drainage stress.

### `temperature_c`

Air temperature during the inspection.

A noisy observation of the heat latent variable. Cool-season grasses receive additional latent heat stress; warm-season grasses receive less.

### `humidity_pct`

Relative humidity at inspection.

Generated with a negative association with temperature. It is context, not a dominant label term.

### `rainfall_7d_mm`

Rainfall total for the seven days before inspection.

Raises moisture and can raise drainage stress when drainage is already limited. UMD Extension cites about 1 inch (25 mm) of water per week for actively growing turf; that figure is background, not a label threshold.

### `irrigation_sessions_per_week`

How many times the lawn was irrigated in a typical recent week.

Rises with caretaker effort and heat, and falls after heavier rain. Frequent sessions do not guarantee deep moisture.

### `mowing_sessions_per_month`

Mowing visits in the recent month.

A partial view of maintenance effort. Very few sessions increase neglect stress unless growth has paused; the screen therefore asks the user to confirm that the lawn is actively growing.

### `days_since_last_service`

Days since the last organised lawn service or equivalent care visit.

Longer gaps increase neglect stress and the chance that laboratory measurements are missing.

### `discoloration_pct`

Share of the lawn with off-colour turf. Not a disease diagnosis.

Noisy visible consequence of the latent stresses, plus independent observation noise.

### `bare_patch_pct`

Share of the lawn with little or no turf cover.

Noisy visible consequence of latent stress. A single bare-patch value does not set the label.

### `wilting_score`

Worker rating of wilt, footprinting, or folded leaves.

More strongly tied to the water-stress latent variable than the other symptoms, but still noisy. UMD Extension describes dull colour and footprinting as drought signs; this score is a project scale for those kinds of observations.

### `thatch_thickness_mm`

Estimated thatch thickness at the soil surface.

Tends to be higher when nutrient supply and watering are high. Thickness above about 12.7 mm (one-half inch) adds to neglect stress. That half-inch figure is the UMD Extension dethatching discussion, used here only as a screen.

### `weed_cover_pct`

Estimated area where weeds are visually prominent. Species are not identified.

Rises with low maintenance effort, shade, and bare ground. It does not trigger a herbicide rate.

## What is deliberately absent

The CSV does not contain the latent variables (`water_latent`, `heat_latent`,
`chem_latent`, `care_latent`, `site_latent`), the component stresses, or the
continuous risk score. Those quantities would leak the label-generating process.
