"""Operational lawn assessment. No benchmark labels are shown here."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import FEATURE_SPECS, OUTPUT_DIR, UI_LIMITS
from src.cost_estimation import default_assumptions, plans_from_recommendations
from src.prediction_service import assess, compare_assessments
from src.ui.common import (
    CLASS_COLORS,
    CLASS_ORDER_UI,
    apply_chart,
    json_dump,
    load_bundle,
    note,
    page_intro,
    percent,
    pretty_feature,
    pretty_model,
    risk_banner,
    setup_message,
)
from src.ui.planner import edit_price_assumptions

ESSENTIAL_FIELDS = [
    "lawn_area_m2",
    "grass_type",
    "sunlight_hours_per_day",
    "soil_texture",
    "soil_ph",
    "soil_moisture_pct",
    "drainage_score",
    "soil_compaction_score",
    "temperature_c",
    "rainfall_7d_mm",
    "irrigation_sessions_per_week",
    "foot_traffic_level",
]
ADVANCED_FIELDS = [name for name in FEATURE_SPECS if name not in ESSENTIAL_FIELDS]


def render_assessment() -> None:
    page_intro(
        "Operational decision support",
        "Lawn assessment",
        "Record an inspection, then review the risk, the reasons, and a service plan.",
    )
    bundle, status = load_bundle()
    if bundle is None:
        setup_message(status)
        return
    st.session_state.setdefault("price_assumptions", default_assumptions("INR"))

    with st.form("lawn_form"):
        st.subheader("Inspection")
        note("Leave a numeric field empty when it was not measured. An empty field is unknown. It is not treated as a healthy value.")
        widgets = {}
        for left_name, right_name in zip(ESSENTIAL_FIELDS[0::2], ESSENTIAL_FIELDS[1::2]):
            left, right = st.columns(2)
            widgets[left_name] = _input(left, left_name)
            widgets[right_name] = _input(right, right_name)
        with st.expander("Advanced measurements (optional)"):
            note("Nutrients, humidity, mowing history, and visible symptoms. Empty fields stay missing and are imputed inside the saved pipelines.")
            for left_name, right_name in zip(ADVANCED_FIELDS[0::2], ADVANCED_FIELDS[1::2]):
                left, right = st.columns(2)
                widgets[left_name] = _input(left, left_name)
                widgets[right_name] = _input(right, right_name)
            if len(ADVANCED_FIELDS) % 2:
                widgets[ADVANCED_FIELDS[-1]] = _input(st, ADVANCED_FIELDS[-1])
        submitted = st.form_submit_button("Assess this lawn")

    if submitted:
        raw = {}
        form_errors = []
        for name, widget in widgets.items():
            if widget["kind"] == "categorical":
                raw[name] = widget["value"]
                continue
            if widget["value"] is None:
                if name == "lawn_area_m2":
                    form_errors.append("Enter lawn area. It is required and is not imputed.")
                else:
                    raw[name] = None
            else:
                raw[name] = widget["value"]
        if form_errors:
            for message in form_errors:
                st.error(message)
            return
        result = assess(
            raw,
            bundle["models"],
            bundle["baselines"],
            st.session_state["price_assumptions"],
        )
        if not result["ok"]:
            for message in result["errors"]:
                st.error(message)
            return
        result["saved_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        history = st.session_state.setdefault("assessment_history", [])
        history.append(result)
        json_dump(OUTPUT_DIR / "assessment_history.json", history)
        st.session_state["latest_assessment"] = result

    result = st.session_state.get("latest_assessment")
    if not result:
        note("Submit an inspection to see the risk class, the reasons, the screens, and a cost estimate.")
        return
    _show_result(result)
    _show_history()


def _input(container, name: str) -> dict:
    spec = FEATURE_SPECS[name]
    label = f"{pretty_feature(name)} ({spec['unit']})"
    key = f"field_{name}"
    if spec["kind"] == "categorical":
        kwargs = {
            "options": spec["levels"],
            "placeholder": "Select",
            "key": key,
            "help": spec["description"],
        }
        if key not in st.session_state:
            kwargs["index"] = None
        return {"kind": "categorical", "value": container.selectbox(label, **kwargs)}
    lower, upper = UI_LIMITS[name]
    kwargs = {
        "min_value": float(lower),
        "max_value": float(upper),
        "step": 1.0 if spec["decimals"] == 0 else 0.1,
        "key": key,
        "help": spec["description"],
    }
    if key not in st.session_state:
        kwargs["value"] = None
    return {"kind": "numeric", "value": container.number_input(label, **kwargs)}


def _show_result(result: dict) -> None:
    for message in result["warnings"]:
        st.warning(message)
    risk_banner(result["prediction"], "Soft-voting risk")
    probability = result["probabilities"]
    columns = st.columns(3)
    for column, label in zip(columns, CLASS_ORDER_UI):
        column.metric(f"{label} probability", percent(probability[label]))
    note(result["probability_note"])

    with st.expander("Model prediction details"):
        _model_details(result, probability)

    st.subheader("Why this class")
    note(result["explanation_note"])
    measured = [item for item in result["explanations"] if item["status"] == "measured"][:6]
    if measured:
        explain = go.Figure(
            go.Bar(
                x=[item["contribution"] for item in measured][::-1],
                y=[pretty_feature(item["feature"]) for item in measured][::-1],
                orientation="h",
                marker_color="#2F6F4E",
            )
        )
        apply_chart(explain, "Change in the predicted-class probability", "Probability points versus the fitting-set baseline", "")
        st.plotly_chart(explain, width="stretch")
        top = measured[0]
        direction = "raises" if top["contribution"] >= 0 else "lowers"
        note(
            f"{pretty_feature(top['feature'])} {direction} the predicted-class probability by "
            f"{abs(top['contribution']) * 100:.1f} points relative to the fitting-set baseline. "
            "That is model dependence, not a biological cause."
        )
    missing = [pretty_feature(item["feature"]) for item in result["explanations"] if item["status"] == "missing"]
    if missing:
        note("No contrast is shown for unknown fields: " + ", ".join(missing) + ". Those fields were imputed inside the pipelines.")

    st.subheader("Screening indicators")
    note(result["recommendations"]["stresses"]["method"])
    for key in ("water", "nutrient", "drainage"):
        screen = result["recommendations"]["stresses"][key]
        st.markdown(f"**{key.capitalize()} screen: {screen['level']}**")
        for reason in screen["reasons"]:
            st.write(reason)
        if screen["missing_fields"]:
            st.write("Not used because they were unknown: " + ", ".join(pretty_feature(name) for name in screen["missing_fields"]) + ".")
        note(screen["limitation"])

    st.subheader("Maintenance actions")
    note(result["recommendations"]["disclaimer"])
    for action in result["recommendations"]["actions"]:
        st.markdown(f"**{action['urgency'].capitalize()} · {action['concern']}**")
        st.write(action["action"])
        if action["evidence"]:
            st.write(action["evidence"][0])
        note(action["rationale"] + " " + action["limitation"])

    st.subheader("Illustrative cost")
    assumptions = edit_price_assumptions()
    result["costs"] = plans_from_recommendations(float(result["inputs"]["lawn_area_m2"]), result["recommendations"], assumptions)
    st.session_state["latest_assessment"] = result
    _show_costs(result["costs"])

    follow = result["recommendations"]["follow_up"]
    st.subheader("Follow-up")
    st.write(f"Suggested return in about {follow['interval_days']} days. {follow['reason']}")
    for item in follow["checklist"]:
        st.write(f"- {item}")
    note(follow["limitation"])


def _model_details(result: dict, probability: dict) -> None:
    agreement = result["agreement"]
    if agreement["status"] == "disagreement":
        st.warning(agreement["detail"])
    elif agreement["status"] == "majority":
        st.info(agreement["detail"])
    else:
        st.success(agreement["detail"])
    vote_rows = []
    for name, payload in result["individual"].items():
        vote_rows.append(
            {
                "Model": payload["display_name"],
                "Predicted class": payload["prediction"],
                **{label: percent(payload["probabilities"][label]) for label in CLASS_ORDER_UI},
            }
        )
    vote_rows.append(
        {
            "Model": "Soft voting",
            "Predicted class": result["prediction"],
            **{label: percent(probability[label]) for label in CLASS_ORDER_UI},
        }
    )
    st.dataframe(pd.DataFrame(vote_rows), hide_index=True, width="stretch")
    figure = go.Figure()
    names = list(result["individual"]) + ["soft_voting"]
    for label in CLASS_ORDER_UI:
        figure.add_bar(
            name=label,
            x=[pretty_model(name) for name in names],
            y=[
                result["individual"][name]["probabilities"][label] if name != "soft_voting" else probability[label]
                for name in names
            ],
            marker_color=CLASS_COLORS[label],
        )
    apply_chart(figure, "Class probabilities by model", "", "Estimated probability")
    figure.update_layout(barmode="group", yaxis_range=[0, 1], yaxis_tickformat=".0%")
    st.plotly_chart(figure, width="stretch")
    note("Each group is one saved pipeline. Soft voting is the equal-weight average of those four probability vectors.")


def _show_costs(costs: dict) -> None:
    for plan_name in ("minimal", "comprehensive"):
        plan = costs[plan_name]
        st.markdown(f"**{plan_name.capitalize()} plan: {plan['display_total']}**")
        if not plan["lines"]:
            st.write("No priced line was selected.")
            continue
        st.dataframe(
            pd.DataFrame(
                [
                    {"Service": line["label"], "Calculation": line["formula"], "Amount": line["display"]}
                    for line in plan["lines"]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    note(costs["minimal"]["note"] + " " + costs["minimal"]["what_this_is_not"])


def _show_history() -> None:
    history = st.session_state.get("assessment_history", [])
    if len(history) < 2:
        return
    st.subheader("Compare saved assessments")
    note("Stored in this browser session and in output/assessment_history.json. No customer name or contact details are kept.")
    labels = [f"{index + 1}. {item['saved_at']} — {item['prediction']}" for index, item in enumerate(history)]
    first = st.selectbox("Earlier assessment", options=range(len(history)), format_func=lambda index: labels[index], key="compare_first")
    second = st.selectbox(
        "Later assessment",
        options=range(len(history)),
        index=len(history) - 1,
        format_func=lambda index: labels[index],
        key="compare_second",
    )
    if first == second:
        st.write("Choose two different assessments.")
        return
    comparison = compare_assessments(history[first], history[second])
    st.write(
        f"Risk: {comparison['risk_before']} then {comparison['risk_after']}. "
        f"Screens: {comparison['stress_before']} then {comparison['stress_after']}. "
        f"Comprehensive estimate: {comparison['cost_before']} then {comparison['cost_after']}."
    )
    if comparison["input_changes"]:
        st.dataframe(pd.DataFrame(comparison["input_changes"]), hide_index=True, width="stretch")
    else:
        st.write("The recorded inputs are the same.")
    note(comparison["limitation"])
