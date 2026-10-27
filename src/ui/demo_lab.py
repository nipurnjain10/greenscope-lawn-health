"""Held-out demonstration. Every probability comes from the saved models."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import FEATURE_COLUMNS, FEATURE_SPECS, REPORT_DIR, TARGET_COLUMN
from src.cost_estimation import default_assumptions
from src.prediction_service import assess
from src.ui.common import (
    CLASS_COLORS,
    CLASS_ORDER_UI,
    apply_chart,
    load_bundle,
    load_test_frame,
    note,
    page_intro,
    percent,
    pretty_feature,
    pretty_model,
    risk_banner,
    setup_message,
)
from src.utilities import read_json

SCENARIOS = {
    "healthy": "A generally healthy simulated lawn",
    "water_stress": "A lawn with a likely water-stress screen",
    "drainage": "A lawn with a likely drainage screen",
    "nutrient": "A lawn with a likely nutrient screen",
    "interacting": "Several screens marked likely",
    "ambiguous": "Models disagree or the probability is not decisive",
}


def _raw_from_row(row: pd.Series) -> dict:
    raw = {}
    for column in FEATURE_COLUMNS:
        value = row[column]
        if pd.isna(value):
            raw[column] = None
        elif hasattr(value, "item"):
            raw[column] = value.item()
        else:
            raw[column] = value
    return raw


def render_demo() -> None:
    page_intro(
        "Research demonstration",
        "Demo test lab",
        "Each run scores one untouched held-out synthetic inspection with the saved models. Nothing is retrained.",
    )
    bundle, status = load_bundle()
    if bundle is None:
        setup_message(status)
        return
    scenario_path = REPORT_DIR / "demo_scenarios.json"
    if not scenario_path.exists():
        st.warning("demo_scenarios.json is missing. Retrain so the demonstration can offer predefined cases.")
        scenarios = {}
    else:
        scenarios = read_json(scenario_path)
    test_frame = load_test_frame()
    st.write(f"Held-out test rows available: {len(test_frame):,}.")

    choice = st.selectbox(
        "Case source",
        options=["random_test_row", *SCENARIOS.keys()],
        format_func=lambda key: "Any held-out test row" if key == "random_test_row" else SCENARIOS[key],
    )
    identifiers = scenarios.get(choice, []) if choice != "random_test_row" else test_frame["inspection_id"].astype(str).tolist()
    if choice != "random_test_row" and not identifiers:
        st.info("No held-out row currently matches this scenario filter.")
    selected_id = None
    if identifiers:
        selected_id = st.selectbox("Inspection identifier", options=identifiers)
    run = st.button("Run Demonstration Assessment", type="primary")
    if run and selected_id:
        _run_case(test_frame, selected_id, bundle)
    elif run:
        st.error("There is no inspection identifier to run.")

    history = st.session_state.get("demo_history", [])
    if history:
        st.subheader("Cases run in this session")
        st.dataframe(pd.DataFrame(history), hide_index=True, width="stretch")


def _run_case(test_frame: pd.DataFrame, inspection_id: str, bundle: dict) -> None:
    row = test_frame.loc[test_frame["inspection_id"].astype(str) == str(inspection_id)].iloc[0]
    reference = str(row[TARGET_COLUMN])
    result = assess(
        _raw_from_row(row),
        bundle["models"],
        bundle["baselines"],
        st.session_state.setdefault("price_assumptions", default_assumptions("INR")),
    )
    if not result["ok"]:
        for message in result["errors"]:
            st.error(message)
        return
    st.subheader(f"Inspection {inspection_id}")
    note("The inputs below are the stored synthetic measurements, including fields the generator marked missing.")
    display_rows = []
    for column in FEATURE_COLUMNS:
        value = row[column]
        display_rows.append(
            {
                "Feature": pretty_feature(column),
                "Group": FEATURE_SPECS[column]["group"],
                "Value": "unknown" if pd.isna(value) else str(value),
                "Unit": FEATURE_SPECS[column]["unit"],
            }
        )
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, width="stretch")

    risk_banner(result["prediction"], "Soft-voting prediction")
    st.markdown(
        f"<div class='risk-banner'><div class='risk-label'>Simulated benchmark label</div>"
        f"<div class='risk-value'>{reference}</div></div>",
        unsafe_allow_html=True,
    )
    if result["prediction"] == reference:
        note(
            f"The soft-voting class matches the simulated benchmark label ({reference}). "
            "That is agreement with the generator, not confirmation from a real lawn."
        )
    else:
        note(
            f"The soft-voting class is {result['prediction']} and the simulated benchmark label is {reference}. "
            "The models missed this synthetic case. The label is not a field diagnosis, and the miss is still a real error on the benchmark."
        )
    note("The simulated benchmark label is shown only on this research page. The operational assessment page does not have one.")

    vote_rows = []
    for name, payload in result["individual"].items():
        vote_rows.append({"Model": payload["display_name"], "Class": payload["prediction"], **{label: payload["probabilities"][label] for label in CLASS_ORDER_UI}})
    vote_rows.append({"Model": "Soft voting", "Class": result["prediction"], **result["probabilities"]})
    figure = go.Figure()
    for label in CLASS_ORDER_UI:
        figure.add_bar(
            name=label,
            x=[row["Model"] for row in vote_rows],
            y=[row[label] for row in vote_rows],
            marker_color=CLASS_COLORS[label],
        )
    apply_chart(figure, "Probabilities for this held-out inspection", "", "Estimated probability")
    figure.update_layout(barmode="group", yaxis_range=[0, 1], yaxis_tickformat=".0%")
    st.plotly_chart(figure, width="stretch")
    parts = []
    for label in CLASS_ORDER_UI:
        values = [payload["probabilities"][label] for payload in result["individual"].values()]
        parts.append(
            f"{label}: ({' + '.join(f'{value:.3f}' for value in values)}) / 4 = {result['probabilities'][label]:.3f}"
        )
    note(
        f"Agreement: {result['agreement']['detail']} "
        "Soft voting is the mean of the four class probabilities, and the class is the largest mean. "
        + " ".join(parts)
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": row["Model"],
                    "Class": row["Class"],
                    **{label: percent(row[label]) if label in row else "—" for label in CLASS_ORDER_UI},
                }
                for row in vote_rows
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Screens, actions, and illustrative cost")
    for key in ("water", "nutrient", "drainage"):
        screen = result["recommendations"]["stresses"][key]
        st.markdown(f"**{key.capitalize()} screen: {screen['level']}**")
        for reason in screen["reasons"][:3]:
            st.write(reason)
    for action in result["recommendations"]["actions"]:
        if action["urgency"] == "low" and action["action_id"] not in result["recommendations"]["minimal_action_ids"]:
            continue
        st.write(f"{action['urgency'].capitalize()}: {action['action']}")
    plan = result["costs"]["comprehensive"]
    st.write(f"Comprehensive illustrative total: {plan['display_total']}.")
    note(result["recommendations"]["disclaimer"])
    follow = result["recommendations"]["follow_up"]
    st.write(f"Suggested return in about {follow['interval_days']} days. {follow['reason']}")
    for item in follow["checklist"]:
        st.write(f"- {item}")
    note(follow["limitation"])
    history = st.session_state.setdefault("demo_history", [])
    history.append(
        {
            "inspection_id": inspection_id,
            "simulated_benchmark_label": reference,
            "soft_voting": result["prediction"],
            "match": result["prediction"] == reference,
        }
    )
    st.session_state["latest_assessment"] = result
