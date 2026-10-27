"""Landing page. Counts and scores are read from saved files."""

from __future__ import annotations

import streamlit as st

from src.config import CLASS_ORDER, DISCLOSURE
from src.ui.common import artifact_report, load_metrics, note, page_intro, percent, pretty_model
from src.ui.navstate import PAGES_BY_PATH


def render_overview() -> None:
    page_intro(
        "Academic prototype",
        "GreenScope",
        "An explainable machine-learning decision-support prototype for landscaping operations.",
    )
    st.markdown(
        "A landscaping crew records a lawn inspection. GreenScope estimates whether that lawn "
        "is at low, moderate, or high health risk, explains the measurements the model used, "
        "and turns the result into maintenance priorities and an illustrative service budget. "
        "It is an independent semester project, not a product of a commercial lawn-care company."
    )
    st.subheader("Workflow")
    st.markdown(
        "1. Record the lawn inspection.\n"
        "2. Score it with logistic regression, a random forest, gradient boosting, and an RBF support-vector machine.\n"
        "3. Soft voting averages the four class probabilities and selects the largest.\n"
        "4. Explain the measurements that moved the prediction.\n"
        "5. Apply water, nutrient, and drainage screens, then list maintenance actions and a cost estimate."
    )
    note("The screens are written rules, not extra classifiers. Prices are editable illustrations, not evidence of profit.")

    status = artifact_report()
    data_meta = status.get("data_meta") or {}
    if data_meta:
        counts = data_meta.get("class_counts_main", {})
        st.write(
            f"Primary dataset: {int(data_meta.get('n_main', 0)):,} synthetic inspections. "
            "Class counts: "
            + ", ".join(f"{label} {counts.get(label, 0):,}" for label in CLASS_ORDER)
            + "."
        )
    else:
        st.info("Dataset metadata has not been written yet.")
    if status.get("fingerprint_match") is False:
        st.error("Model files do not match the current dataset fingerprint. Retrain before quoting results.")

    st.subheader("Latest evaluation")
    if status["metrics"] and status.get("fingerprint_match") is not False:
        metrics = load_metrics()
        soft = metrics["models"]["soft_voting"]["test"]
        individuals = {
            name: metrics["models"][name]["test"]["accuracy"]
            for name in ("logistic_regression", "random_forest", "gradient_boosting", "svm_rbf")
        }
        best_name = max(individuals, key=individuals.get)
        c1, c2, c3 = st.columns(3)
        c1.metric("Soft-voting test accuracy", percent(soft["accuracy"]))
        c2.metric("Soft-voting test macro-F1", percent(soft["macro_f1"]))
        c3.metric("Best individual test accuracy", f"{pretty_model(best_name)} {percent(individuals[best_name])}")
        note(metrics["narratives"]["performance"])
    else:
        st.info("Evaluation metrics will appear here after training finishes. No placeholder accuracy is shown.")

    st.subheader("Continue")
    for path, label in (
        ("assessment", "Start a lawn assessment"),
        ("demo", "Run a demonstration on a held-out synthetic case"),
        ("results", "Read the model results and dataset"),
    ):
        page = PAGES_BY_PATH.get(path)
        if page is not None:
            st.page_link(page, label=label)
    st.caption(DISCLOSURE)
