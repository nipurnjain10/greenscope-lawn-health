"""Shared layout, artifact loading, and chart styling for the Streamlit app."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    CLASS_ORDER,
    DATA_DIR,
    DISCLOSURE,
    ENSEMBLE_SPECS,
    MODEL_DIR,
    MODEL_SPECS,
    REPORT_DIR,
)
from src.utilities import read_json

CLASS_COLORS = {"Low": "#2F6F4E", "Moderate": "#B5812F", "High": "#8C3A3A"}
MODEL_COLORS = {
    "logistic_regression": "#1F4E79",
    "random_forest": "#2F6F4E",
    "gradient_boosting": "#6B5B4B",
    "svm_rbf": "#3D6B8C",
    "soft_voting": "#1E4D36",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1120px; padding-top: 1.4rem; padding-bottom: 3rem;}
        [data-testid="stSidebar"] {background: #FBF9F6; border-right: 1px solid #E4E0D8;}
        h1 {font-size: 1.75rem; font-weight: 560; letter-spacing: -0.015em; margin-bottom: 0.2rem;}
        h2 {font-size: 1.22rem; font-weight: 560; margin-top: 1.4rem;}
        h3 {font-size: 1.02rem; font-weight: 560;}
        p, li {line-height: 1.5;}
        .gs-kicker {color: #2F6F4E; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.15rem;}
        .gs-sub {color: #3E4C59; font-size: 1.02rem; margin-bottom: 0.8rem;}
        .gs-note {color: #3E4C59; font-size: 0.95rem; line-height: 1.5; margin: 0.15rem 0 0.8rem 0;}
        .gs-disclosure {font-size: 0.92rem; line-height: 1.45; color: #243038; background: #F3F1EC; border: 1px solid #E4E0D8; padding: 0.65rem 0.75rem; border-radius: 2px;}
        .risk-banner {border: 1px solid #E4E0D8; border-left-width: 6px; background: white; padding: 0.8rem 1rem; margin: 0.4rem 0 0.8rem 0;}
        .risk-low {border-left-color: #2F6F4E;}
        .risk-moderate {border-left-color: #B5812F;}
        .risk-high {border-left-color: #8C3A3A;}
        .risk-label {font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; color: #52606D;}
        .risk-value {font-size: 1.7rem; font-weight: 560; color: #1C2833;}
        div[data-testid="stMetric"] {background: white; border: 1px solid #E4E0D8; padding: 0.45rem 0.7rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_intro(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='gs-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f"<div class='gs-sub'>{subtitle}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<p class='gs-note'>{text}</p>", unsafe_allow_html=True)


def risk_banner(label: str, kicker: str = "Ensemble risk") -> None:
    css = {"Low": "risk-low", "Moderate": "risk-moderate", "High": "risk-high"}.get(label, "risk-low")
    st.markdown(
        f"<div class='risk-banner {css}'><div class='risk-label'>{kicker}</div>"
        f"<div class='risk-value'>{label}</div></div>",
        unsafe_allow_html=True,
    )


def apply_chart(figure, title: str, x_title: str = "", y_title: str = ""):
    figure.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1C2833")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", color="#1C2833", size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.22),
        margin=dict(l=48, r=16, t=52, b=48),
        xaxis=dict(title=x_title, gridcolor="#EFEBE4", linecolor="#E4E0D8"),
        yaxis=dict(title=y_title, gridcolor="#EFEBE4", linecolor="#E4E0D8"),
    )
    return figure


def artifact_report() -> dict:
    data_meta_path = DATA_DIR / "metadata.json"
    run_path = MODEL_DIR / "run_metadata.json"
    metrics_path = REPORT_DIR / "model_metrics.json"
    primary_path = DATA_DIR / "synthetic_lawn_inspections.csv"
    robust_path = DATA_DIR / "robustness_scenarios.csv"
    model_paths = {name: MODEL_DIR / f"{name}.joblib" for name in MODEL_SPECS}
    report = {
        "primary_csv": primary_path.exists(),
        "robust_csv": robust_path.exists(),
        "data_metadata": data_meta_path.exists(),
        "models": {name: path.exists() for name, path in model_paths.items()},
        "metrics": metrics_path.exists(),
        "run_metadata": run_path.exists(),
        "baselines": (MODEL_DIR / "feature_baselines.json").exists(),
        "fingerprint_match": None,
        "data_meta": None,
        "run_meta": None,
    }
    if data_meta_path.exists():
        report["data_meta"] = read_json(data_meta_path)
    if run_path.exists() and report["data_meta"]:
        report["run_meta"] = read_json(run_path)
        report["fingerprint_match"] = (
            report["run_meta"].get("dataset_fingerprint_main")
            == report["data_meta"].get("fingerprint_sha256_main")
        )
    report["ready"] = (
        report["primary_csv"]
        and report["robust_csv"]
        and all(report["models"].values())
        and report["metrics"]
        and report["baselines"]
        and report["fingerprint_match"] is True
    )
    return report


@st.cache_resource
def _load_fitted_models(fingerprint: str) -> dict:
    models = {name: joblib.load(MODEL_DIR / f"{name}.joblib") for name in MODEL_SPECS}
    baselines = read_json(MODEL_DIR / "feature_baselines.json")
    return {"models": models, "baselines": baselines, "fingerprint": fingerprint}


def load_bundle():
    status = artifact_report()
    if not status["ready"]:
        return None, status
    bundle = _load_fitted_models(status["run_meta"]["dataset_fingerprint_main"])
    bundle["run"] = status["run_meta"]
    bundle["data"] = status["data_meta"]
    return bundle, status


@st.cache_data
def load_metrics() -> dict:
    return read_json(REPORT_DIR / "model_metrics.json")


@st.cache_data
def load_primary() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "synthetic_lawn_inspections.csv")


@st.cache_data
def load_robustness() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "robustness_scenarios.csv")


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_csv(REPORT_DIR / "permutation_importance.csv")


@st.cache_data
def load_test_frame() -> pd.DataFrame:
    primary = load_primary()
    splits = pd.read_csv(DATA_DIR / "split_assignments.csv")
    test_ids = set(splits.loc[splits["split"] == "test", "inspection_id"].astype(str))
    frame = primary.loc[primary["inspection_id"].astype(str).isin(test_ids)].copy()
    return frame.sort_values("inspection_id").reset_index(drop=True)


def setup_message(status: dict) -> None:
    st.warning(
        "The trained models or the dataset metadata are not ready. "
        "From the project root, with the virtual environment active, run: "
        "`python data/generate_synthetic_data.py` and then `python -m src.train_models`."
    )
    if status.get("fingerprint_match") is False:
        st.error(
            "The saved models were trained on a different dataset file than the one present now. "
            "Retrain before trusting any number in the application."
        )
    missing = [name for name, present in status["models"].items() if not present]
    if missing:
        st.write("Missing model files:", ", ".join(missing))


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def pretty_feature(name: str) -> str:
    return name.replace("_", " ")


def pretty_model(name: str) -> str:
    return {**MODEL_SPECS, **ENSEMBLE_SPECS}.get(name, name)


def json_dump(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


DISCLOSURE_TEXT = DISCLOSURE
CLASS_ORDER_UI = CLASS_ORDER
