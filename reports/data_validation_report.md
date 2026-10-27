# Data validation report

This report is produced by `src/data_validation.py` from the CSV files on disk.
It describes synthetic inspections only.

Unusual combinations are counted and retained. The validator does not delete
stress cases and does not clip values to obtain a pass.

## Generator notes

- Primary sample size is 10,000 and the robustness sample size is 2,000.
- Both files are produced by the same generator function with different seeds and documented distribution shifts.
- Robustness labels use the primary sample's frozen score cutoffs.

## Primary synthetic inspections — passed

- Rows: 10000
- Columns: 25
- Duplicate identifiers: 0
- Duplicate feature rows: 0
- Class counts: {'Low': 3333, 'Moderate': 3333, 'High': 3334}

### Missingness

| Column | Missing fraction |
| --- | ---: |
| `soil_ph` | 0.0307 |
| `nitrogen_mg_kg` | 0.0865 |
| `phosphorus_mg_kg` | 0.0453 |
| `potassium_mg_kg` | 0.0462 |

### Selected correlations

| Pair | Pearson correlation |
| --- | ---: |
| rainfall_7d_mm and soil_moisture_pct | 0.309 |
| discoloration_pct and bare_patch_pct | 0.331 |
| drainage_score and soil_moisture_pct | -0.507 |
| nitrogen_mg_kg and phosphorus_mg_kg | 0.345 |
| temperature_c and humidity_pct | -0.628 |

### Numeric ranges that were checked

A range failure is a value outside the documented simulation bounds.

No numeric values fell outside the documented simulation bounds.

### Unusual combinations retained

- severe_wilt_and_wet_soil: 0

### Errors

None.

### Warnings

None.

## Robustness scenarios — passed

- Rows: 2000
- Columns: 25
- Duplicate identifiers: 0
- Duplicate feature rows: 0
- Class counts: {'Low': 266, 'Moderate': 513, 'High': 1221}

### Missingness

| Column | Missing fraction |
| --- | ---: |
| `soil_ph` | 0.0530 |
| `nitrogen_mg_kg` | 0.1430 |
| `phosphorus_mg_kg` | 0.0790 |
| `potassium_mg_kg` | 0.0810 |

### Selected correlations

| Pair | Pearson correlation |
| --- | ---: |
| rainfall_7d_mm and soil_moisture_pct | 0.213 |
| discoloration_pct and bare_patch_pct | 0.462 |
| drainage_score and soil_moisture_pct | -0.517 |
| nitrogen_mg_kg and phosphorus_mg_kg | 0.315 |
| temperature_c and humidity_pct | -0.637 |

### Numeric ranges that were checked

A range failure is a value outside the documented simulation bounds.

No numeric values fell outside the documented simulation bounds.

### Unusual combinations retained

- severe_wilt_and_wet_soil: 0

### Errors

None.

### Warnings

None.

## Known limitations

- The rows are synthetic. Passing validation does not make them field observations.
- Nutrient units are a project scale in mg/kg, not a laboratory method such as Mehlich-3.
- Designed correlations are a property of the generator. They are not estimates from a customer population.
- Class labels are tertiles of a noisy score computed from recorded measurements, not diagnoses.
- Some laboratory fields are intentionally missing. Models must impute them from training data only.
