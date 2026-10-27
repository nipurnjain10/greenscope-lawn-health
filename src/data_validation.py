"""Checks for the synthetic lawn-inspection tables.

Validation reports problems. It does not delete unusual but legal stress cases,
and it does not clip values to force a pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASS_ORDER,
    DATA_DIR,
    FEATURE_COLUMNS,
    FEATURE_SPECS,
    ID_COLUMN,
    LEAKAGE_COLUMNS,
    REPORT_DIR,
    TARGET_COLUMN,
)
from src.utilities import ensure_dir


EXPECTED_RELATIONSHIPS = [
    ("rainfall_7d_mm", "soil_moisture_pct", "positive", 0.15),
    ("discoloration_pct", "bare_patch_pct", "positive", 0.10),
    ("drainage_score", "soil_moisture_pct", "negative", -0.05),
    ("nitrogen_mg_kg", "phosphorus_mg_kg", "positive", 0.10),
    ("temperature_c", "humidity_pct", "negative", -0.10),
]


def _is_missing_column(series: pd.Series) -> pd.Series:
    return series.isna()


def validate_frame(frame: pd.DataFrame, *, expected_n: int | None, dataset_name: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if ID_COLUMN not in frame.columns:
        errors.append("Missing inspection_id column.")
    if TARGET_COLUMN not in frame.columns:
        errors.append("Missing target column.")
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            errors.append(f"Missing feature column: {column}")

    leaked = [column for column in frame.columns if column in LEAKAGE_COLUMNS and column != TARGET_COLUMN]
    if leaked:
        errors.append(f"Latent or score columns were exported: {leaked}")
    if TARGET_COLUMN in FEATURE_COLUMNS:
        errors.append("The target is listed as an input feature.")

    n_rows = int(len(frame))
    if expected_n is not None and n_rows != expected_n:
        errors.append(f"Expected {expected_n} rows and found {n_rows}.")

    duplicate_ids = 0
    if ID_COLUMN in frame.columns:
        duplicate_ids = int(frame[ID_COLUMN].duplicated().sum())
        if duplicate_ids:
            errors.append(f"Found {duplicate_ids} duplicated inspection identifiers.")

    duplicate_feature_rows = 0
    if all(column in frame.columns for column in FEATURE_COLUMNS):
        duplicate_feature_rows = int(frame.loc[:, FEATURE_COLUMNS].duplicated().sum())
        if duplicate_feature_rows:
            warnings.append(
                f"Found {duplicate_feature_rows} rows with identical input features. "
                "A small number can occur by chance; a large number would mean the generator collapsed."
            )
            if duplicate_feature_rows > max(10, int(0.005 * max(n_rows, 1))):
                errors.append("Too many exact duplicate feature rows.")

    class_counts: dict[str, int] = {}
    if TARGET_COLUMN in frame.columns:
        unknown = sorted(set(frame[TARGET_COLUMN].dropna().astype(str)) - set(CLASS_ORDER))
        if unknown:
            errors.append(f"Unexpected risk labels: {unknown}")
        if int(frame[TARGET_COLUMN].isna().sum()):
            errors.append("Target contains missing labels.")
        class_counts = {
            label: int((frame[TARGET_COLUMN].astype(str) == label).sum()) for label in CLASS_ORDER
        }
        if n_rows and min(class_counts.values()) / n_rows < 0.12:
            warnings.append(
                f"A risk class is below 12% of {dataset_name}: {class_counts}. "
                "The primary benchmark is expected to stay broadly balanced."
            )

    range_failures: dict[str, int] = {}
    category_failures: dict[str, int] = {}
    missingness: dict[str, float] = {}
    numeric_summary: dict[str, dict] = {}
    allowed_missing = {"nitrogen_mg_kg", "phosphorus_mg_kg", "potassium_mg_kg", "soil_ph"}

    for name, spec in FEATURE_SPECS.items():
        if name not in frame.columns:
            continue
        series = frame[name]
        missing_rate = float(_is_missing_column(series).mean()) if n_rows else 0.0
        missingness[name] = missing_rate
        if missing_rate > 0 and name not in allowed_missing:
            errors.append(f"{name} should not contain missing values (rate {missing_rate:.3f}).")
        if name in allowed_missing and missing_rate > 0.40:
            errors.append(f"{name} missingness is unexpectedly high ({missing_rate:.3f}).")
        observed = series.dropna()
        if spec["kind"] == "categorical":
            bad = ~observed.astype(str).isin(spec["levels"])
            category_failures[name] = int(bad.sum())
            if int(bad.sum()):
                errors.append(f"{name} has {int(bad.sum())} values outside the permitted categories.")
        else:
            numeric = pd.to_numeric(observed, errors="coerce")
            if int(numeric.isna().sum()):
                errors.append(f"{name} has non-numeric values.")
            below = int((numeric < spec["minimum"]).sum())
            above = int((numeric > spec["maximum"]).sum())
            range_failures[name] = below + above
            if below or above:
                errors.append(
                    f"{name} has {below} values below {spec['minimum']} and {above} above {spec['maximum']}."
                )
            if len(numeric):
                numeric_summary[name] = {
                    "mean": float(numeric.mean()),
                    "std": float(numeric.std(ddof=0)),
                    "min": float(numeric.min()),
                    "p50": float(numeric.median()),
                    "max": float(numeric.max()),
                }

    correlations: dict[str, float | None] = {}
    for left, right, direction, bound in EXPECTED_RELATIONSHIPS:
        if left not in frame.columns or right not in frame.columns:
            continue
        pair = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 30:
            correlations[f"{left}__{right}"] = None
            warnings.append(f"Not enough paired rows to correlate {left} and {right}.")
            continue
        value = float(pair[left].corr(pair[right]))
        correlations[f"{left}__{right}"] = value
        if direction == "positive" and value < bound:
            warnings.append(
                f"Expected a positive association between {left} and {right} "
                f"(correlation {value:.3f}, design check > {bound:.2f})."
            )
        if direction == "negative" and value > bound:
            warnings.append(
                f"Expected a negative association between {left} and {right} "
                f"(correlation {value:.3f}, design check < {bound:.2f})."
            )

    unusual = {}
    if {"wilting_score", "soil_moisture_pct"}.issubset(frame.columns):
        wilt = pd.to_numeric(frame["wilting_score"], errors="coerce")
        moist = pd.to_numeric(frame["soil_moisture_pct"], errors="coerce")
        unusual["severe_wilt_and_wet_soil"] = int(((wilt >= 3) & (moist >= 35)).sum())

    personal_columns = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("name", "phone", "email", "address", "customer"))
    ]
    if personal_columns:
        errors.append(f"Unexpected personal-information columns: {personal_columns}")

    return {
        "dataset": dataset_name,
        "n_rows": n_rows,
        "n_columns": int(frame.shape[1]),
        "errors": errors,
        "warnings": warnings,
        "class_counts": class_counts,
        "missingness": missingness,
        "range_failures": range_failures,
        "category_failures": category_failures,
        "numeric_summary": numeric_summary,
        "correlations": correlations,
        "duplicate_ids": duplicate_ids,
        "duplicate_feature_rows": duplicate_feature_rows,
        "unusual_but_retained": unusual,
        "passed": not errors,
    }


def render_validation_markdown(results: list[dict], notes: list[str] | None = None) -> str:
    lines = [
        "# Data validation report",
        "",
        "This report is produced by `src/data_validation.py` from the CSV files on disk.",
        "It describes synthetic inspections only.",
        "",
        "Unusual combinations are counted and retained. The validator does not delete",
        "stress cases and does not clip values to obtain a pass.",
        "",
    ]
    if notes:
        lines.append("## Generator notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    for result in results:
        status = "passed" if result["passed"] else "failed"
        lines.append(f"## {result['dataset']} — {status}")
        lines.append("")
        lines.append(f"- Rows: {result['n_rows']}")
        lines.append(f"- Columns: {result['n_columns']}")
        lines.append(f"- Duplicate identifiers: {result['duplicate_ids']}")
        lines.append(f"- Duplicate feature rows: {result['duplicate_feature_rows']}")
        lines.append(f"- Class counts: {result['class_counts']}")
        lines.append("")
        lines.append("### Missingness")
        lines.append("")
        lines.append("| Column | Missing fraction |")
        lines.append("| --- | ---: |")
        for column, rate in result["missingness"].items():
            if rate > 0:
                lines.append(f"| `{column}` | {rate:.4f} |")
        lines.append("")
        lines.append("### Selected correlations")
        lines.append("")
        lines.append("| Pair | Pearson correlation |")
        lines.append("| --- | ---: |")
        for pair, value in result["correlations"].items():
            shown = "not computed" if value is None else f"{value:.3f}"
            lines.append(f"| {pair.replace('__', ' and ')} | {shown} |")
        lines.append("")
        lines.append("### Numeric ranges that were checked")
        lines.append("")
        lines.append("A range failure is a value outside the documented simulation bounds.")
        lines.append("")
        failed = {key: value for key, value in result["range_failures"].items() if value}
        if failed:
            for key, value in failed.items():
                lines.append(f"- `{key}`: {value} out-of-range values")
        else:
            lines.append("No numeric values fell outside the documented simulation bounds.")
        lines.append("")
        if result["unusual_but_retained"]:
            lines.append("### Unusual combinations retained")
            lines.append("")
            for key, value in result["unusual_but_retained"].items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        lines.append("### Errors")
        lines.append("")
        if result["errors"]:
            for error in result["errors"]:
                lines.append(f"- {error}")
        else:
            lines.append("None.")
        lines.append("")
        lines.append("### Warnings")
        lines.append("")
        if result["warnings"]:
            for warning in result["warnings"]:
                lines.append(f"- {warning}")
        else:
            lines.append("None.")
        lines.append("")
    lines.extend(
        [
            "## Known limitations",
            "",
            "- The rows are synthetic. Passing validation does not make them field observations.",
            "- Nutrient units are a project scale in mg/kg, not a laboratory method such as Mehlich-3.",
            "- Designed correlations are a property of the generator. They are not estimates from a customer population.",
            "- Class labels are tertiles of a noisy score computed from recorded measurements, not diagnoses.",
            "- Some laboratory fields are intentionally missing. Models must impute them from training data only.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_saved_datasets(data_dir: Path | None = None, report_path: Path | None = None) -> list[dict]:
    data_dir = data_dir or DATA_DIR
    report_path = report_path or (REPORT_DIR / "data_validation_report.md")
    primary_path = data_dir / "synthetic_lawn_inspections.csv"
    robust_path = data_dir / "robustness_scenarios.csv"
    if not primary_path.exists() or not robust_path.exists():
        raise FileNotFoundError("Generate the CSV files before validation.")
    primary = pd.read_csv(primary_path)
    robust = pd.read_csv(robust_path)
    results = [
        validate_frame(primary, expected_n=10_000, dataset_name="Primary synthetic inspections"),
        validate_frame(robust, expected_n=2_000, dataset_name="Robustness scenarios"),
    ]
    shared = set(primary[ID_COLUMN]).intersection(set(robust[ID_COLUMN]))
    if shared:
        results[0]["errors"].append(f"Primary and robustness identifiers overlap ({len(shared)} ids).")
        results[0]["passed"] = False
    notes = [
        "Primary sample size is 10,000 and the robustness sample size is 2,000.",
        "Both files are produced by the same generator function with different seeds and documented distribution shifts.",
        "Robustness labels use the primary sample's frozen score cutoffs.",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text(render_validation_markdown(results, notes), encoding="utf-8")
    return results


def main() -> None:
    results = validate_saved_datasets()
    for result in results:
        state = "PASS" if result["passed"] else "FAIL"
        print(f"{state}: {result['dataset']} ({result['n_rows']} rows)")
        for error in result["errors"]:
            print(f"  error: {error}")
        for warning in result["warnings"]:
            print(f"  warning: {warning}")
    if not all(result["passed"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
