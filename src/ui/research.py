"""Academic evaluation page. Every number is read from the saved test evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import CLASS_ORDER, MODEL_SPECS, REPORT_DIR
from src.narratives import confusion_text, importance_text
from src.ui.common import (
    CLASS_COLORS,
    MODEL_COLORS,
    apply_chart,
    load_importance,
    load_metrics,
    note,
    page_intro,
    percent,
    pretty_feature,
    pretty_model,
    setup_message,
    artifact_report,
)
from src.ui.explorer import render_dataset_details, render_dataset_summary
from src.ui.methodology import render_methodology_brief
from src.utilities import read_json

METRIC_KEYS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
    ("macro_precision", "Macro-precision"),
    ("macro_recall", "Macro-recall"),
]
COMPARISON_MODELS = list(MODEL_SPECS) + ["soft_voting"]


def render_research() -> None:
    page_intro(
        "Evaluation",
        "Model results and data",
        "Held-out results for the four classifiers and soft voting. Training scores are not mixed into these charts.",
    )
    status = artifact_report()
    if not status["metrics"] or status.get("fingerprint_match") is False:
        setup_message(status)
        return
    metrics = load_metrics()
    note(metrics["narratives"]["split"])
    note(metrics["evaluation_protocol"])

    st.subheader("Model performance")
    figure = go.Figure()
    for key, label in METRIC_KEYS:
        figure.add_bar(
            name=label,
            y=[pretty_model(name) for name in COMPARISON_MODELS],
            x=[metrics["models"][name]["test"][key] for name in COMPARISON_MODELS],
            orientation="h",
        )
    apply_chart(figure, "Test-set scores", "Score", "")
    figure.update_layout(barmode="group", xaxis_range=[0, 1], xaxis_tickformat=".0%")
    st.plotly_chart(figure, width="stretch")
    note(metrics["narratives"]["performance"])
    _interval_table(metrics)

    st.subheader("Confusion matrices")
    selected = st.selectbox("Model", options=COMPARISON_MODELS, format_func=pretty_model)
    block = metrics["models"][selected]["test"]
    matrix = np.array(block["confusion_matrix"])
    heat = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=CLASS_ORDER,
            y=CLASS_ORDER,
            colorscale=[[0, "#F7F5F2"], [1, "#2F6F4E"]],
            text=matrix.astype(int),
            texttemplate="%{text}",
            showscale=False,
            hovertemplate="Simulated %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    apply_chart(heat, f"{pretty_model(selected)} on the test set", "Predicted class", "Simulated reference class")
    heat.update_yaxes(autorange="reversed")
    st.plotly_chart(heat, width="stretch")
    note(confusion_text(selected, block))

    st.subheader("Per-class performance")
    class_figure = go.Figure()
    soft = metrics["models"]["soft_voting"]["test"]["per_class"]
    for key, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
        class_figure.add_bar(name=label, x=CLASS_ORDER, y=[soft[name][key] for name in CLASS_ORDER], marker_color={"Precision": "#1F4E79", "Recall": "#2F6F4E", "F1": "#B5812F"}[label])
    apply_chart(class_figure, "Soft voting by simulated class", "Class", "Score")
    class_figure.update_layout(barmode="group", yaxis_range=[0, 1], yaxis_tickformat=".0%")
    st.plotly_chart(class_figure, width="stretch")
    note(metrics["narratives"]["per_class"])

    st.subheader("How soft voting combines the models")
    _soft_vote_example()
    note(
        "For each class, soft voting is the arithmetic mean of the four model probabilities. "
        "The predicted class is the largest mean. Weights are equal and were not chosen from the test set. "
        "Call the values model-estimated probabilities. The calibration section shows whether those probabilities "
        "matched simulated-label frequencies in probability bins."
    )

    st.subheader("Explainability")
    importance = load_importance()
    model_for_importance = st.selectbox(
        "Importance model",
        options=["soft_voting", *MODEL_SPECS.keys()],
        format_func=pretty_model,
    )
    subset = importance.loc[importance["model"] == model_for_importance].sort_values("importance_mean", ascending=True).tail(12)
    importance_figure = go.Figure(
        go.Bar(
            x=subset["importance_mean"],
            y=subset["feature"].map(pretty_feature),
            orientation="h",
            error_x=dict(type="data", array=subset["importance_std"]),
            marker_color=MODEL_COLORS.get(model_for_importance, "#2F6F4E"),
        )
    )
    apply_chart(importance_figure, f"Permutation importance: {pretty_model(model_for_importance)}", "Mean drop in test accuracy", "")
    st.plotly_chart(importance_figure, width="stretch")
    top = subset.sort_values("importance_mean", ascending=False).head(5).to_dict(orient="records")
    note(importance_text(top))

    st.subheader("Robustness")
    robust = go.Figure()
    series = [
        ("Test accuracy", "test", "accuracy", "#2F6F4E"),
        ("Test macro-F1", "test", "macro_f1", "#1F4E79"),
        ("Robustness accuracy", "robustness", "accuracy", "#C4B49A"),
        ("Robustness macro-F1", "robustness", "macro_f1", "#8C6A3A"),
    ]
    for label, split, key, color in series:
        robust.add_bar(
            name=label,
            x=[pretty_model(name) for name in COMPARISON_MODELS],
            y=[metrics["models"][name][split][key] for name in COMPARISON_MODELS],
            marker_color=color,
        )
    apply_chart(robust, "Accuracy and macro-F1 under the designed distribution shift", "", "Score")
    robust.update_layout(barmode="group", yaxis_range=[0, 1], yaxis_tickformat=".0%")
    st.plotly_chart(robust, width="stretch")
    note(metrics["narratives"]["robustness"])

    st.subheader("Dataset")
    render_dataset_summary()

    with st.expander("Additional experiments"):
        st.subheader("Recall by model")
        recall_figure = go.Figure()
        for label in CLASS_ORDER:
            recall_figure.add_bar(
                name=f"{label} recall",
                x=[pretty_model(name) for name in COMPARISON_MODELS],
                y=[metrics["models"][name]["test"]["per_class"][label]["recall"] for name in COMPARISON_MODELS],
                marker_color=CLASS_COLORS[label],
            )
        apply_chart(recall_figure, "Test recall by class", "", "Recall")
        recall_figure.update_layout(barmode="group", yaxis_range=[0, 1], yaxis_tickformat=".0%")
        st.plotly_chart(recall_figure, width="stretch")
        note("Recall is the share of each simulated class that the model recovered. Low recall for High means missed high-risk cases.")

        st.subheader("Calibration")
        _calibration(metrics)
        note(metrics["narratives"]["calibration"])

        st.subheader("Ablation")
        full = metrics["models"]["soft_voting"]["test"]["macro_f1"]
        names = ["All four"] + [f"Without {pretty_model(name)}" for name in metrics["ablation"]]
        values = [full] + [metrics["ablation"][name]["macro_f1"] for name in metrics["ablation"]]
        ablation = go.Figure(go.Bar(y=names[::-1], x=values[::-1], orientation="h", marker_color="#2F6F4E"))
        apply_chart(ablation, "Test macro-F1 when one model is dropped from the average", "Macro-F1", "")
        ablation.update_layout(xaxis_range=[0, 1], xaxis_tickformat=".0%")
        st.plotly_chart(ablation, width="stretch")
        note(metrics["narratives"]["ablation"])

        st.subheader("Agreement and errors")
        disagreement = metrics["disagreement"]
        agree = go.Figure(
            go.Bar(
                x=["Unanimous", "Majority with one dissent", "No three-vote majority"],
                y=[disagreement["unanimous"], disagreement["majority_with_dissent"], disagreement["split"]],
                marker_color=["#2F6F4E", "#C4B49A", "#8C3A3A"],
                text=["Unanimous", "Majority", "Split"],
            )
        )
        apply_chart(agree, "Do the four models agree on the test set?", "", "Test inspections")
        st.plotly_chart(agree, width="stretch")
        note(
            f"Of {disagreement['n']} test rows, {disagreement['unanimous']} are unanimous "
            f"({percent(disagreement['unanimous_fraction'])}), {disagreement['majority_with_dissent']} have a 3–1 split, "
            f"and {disagreement['split']} have no three-vote majority."
        )
        examples = read_json(REPORT_DIR / "error_examples.json")
        st.markdown("**High simulated risk predicted as low**")
        _example_table(examples["high_called_low"])
        note(
            f"There are {examples['counts']['high_called_low']} such test rows. "
            "A high-to-low error is the miss that most affects a follow-up decision on this benchmark."
        )
        st.markdown("**Rows where the four models do not all agree**")
        _example_table(examples["model_disagreement"])
        note(f"{examples['counts']['model_disagreement']} test rows have at least one dissenting model. Eight examples are listed.")
        render_dataset_details()

    st.subheader("What the metrics mean")
    st.markdown(
        """
        **Accuracy** is the share of test rows whose predicted class matches the simulated label.

        **Precision** for a class is the share of rows predicted as that class whose simulated label really is that class. **Recall** is the share of rows with that simulated label which the model found. **F1** is their harmonic mean.

        **Macro-F1** averages the three class F1 scores equally, so a rare class counts as much as a common one. **Weighted-F1** weights each class by its number of test rows.

        A **confusion matrix** counts simulated label against predicted class. The diagonal is correct. A high-risk row in the low-risk column is the most serious miss in this prototype.

        **Soft voting** averages the four class probabilities with equal weights and selects the largest average.

        **Calibration** compares a probability bin with the frequency of the simulated class inside that bin. A point near the diagonal is well matched on this synthetic test set only.

        **Permutation importance** is the drop in accuracy after a column is shuffled. It is not a causal effect.
        """
    )
    st.subheader("Methodology and sources")
    render_methodology_brief()


def _interval_table(metrics: dict) -> None:
    rows = []
    for name in COMPARISON_MODELS:
        block = metrics["models"][name]["test"]
        low, high = block["bootstrap"]["accuracy_ci95"]
        rows.append(
            {
                "Model": pretty_model(name),
                "Accuracy": percent(block["accuracy"]),
                "Accuracy interval": f"{percent(low)} to {percent(high)}",
                "Macro-F1": percent(block["macro_f1"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    note(metrics["models"]["soft_voting"]["test"]["bootstrap"]["interpretation"])


def _soft_vote_example() -> None:
    predictions = pd.read_csv(REPORT_DIR / "test_predictions.csv")
    choice = st.selectbox("Test inspection used to show the average", options=predictions["inspection_id"].astype(str).tolist())
    row = predictions.loc[predictions["inspection_id"].astype(str) == choice].iloc[0]
    figure = go.Figure()
    model_names = list(MODEL_SPECS) + ["soft_voting"]
    for label in CLASS_ORDER:
        figure.add_bar(
            name=label,
            x=[pretty_model(name) for name in model_names],
            y=[row[f"p_{name}_{label}"] for name in model_names],
            marker_color=CLASS_COLORS[label],
        )
    apply_chart(figure, f"Probability average for {choice}", "", "Probability")
    figure.update_layout(barmode="group", yaxis_range=[0, 1], yaxis_tickformat=".0%")
    st.plotly_chart(figure, width="stretch")
    parts = []
    for label in CLASS_ORDER:
        values = [float(row[f"p_{name}_{label}"]) for name in MODEL_SPECS]
        parts.append(
            f"{label}: ({' + '.join(f'{value:.3f}' for value in values)}) / 4 = {float(row[f'p_soft_voting_{label}']):.3f}"
        )
    note(
        "Worked average for this row. "
        + " ".join(parts)
        + f" Soft voting therefore predicts {row['pred_soft_voting']}. "
        f"The simulated benchmark label is {row['simulated_reference_label']}."
    )


def _calibration(metrics: dict) -> None:
    model_name = st.selectbox("Calibration model", options=["soft_voting", *MODEL_SPECS.keys()], format_func=pretty_model, key="calibration_model")
    block = metrics["calibration"][model_name]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Bin matches the simulated frequency", line=dict(color="#52606D", dash="dash")))
    for label in CLASS_ORDER:
        bins = block["one_vs_rest"][label]["bins"]
        figure.add_trace(
            go.Scatter(
                x=[item["mean_predicted"] for item in bins],
                y=[item["observed_frequency"] for item in bins],
                mode="lines+markers",
                name=label,
                marker_color=CLASS_COLORS[label],
                text=[f"n={item['count']}" for item in bins],
                hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<br>%{text}<extra></extra>",
            )
        )
    apply_chart(figure, f"One-versus-rest reliability: {pretty_model(model_name)}", "Mean predicted probability", "Observed frequency of the simulated class")
    figure.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(figure, width="stretch")
    note(
        f"Macro expected calibration error {block['macro_ece']:.3f}. "
        f"Multiclass Brier score {block['multiclass_brier']:.3f}. "
        "Bins with no test rows are omitted. No calibrator was fit on these rows."
    )


def _example_table(rows: list[dict]) -> None:
    if not rows:
        st.write("None in the saved sample.")
        return
    flat = []
    for row in rows:
        flat.append(
            {
                "Inspection": row["inspection_id"],
                "Simulated label": row["simulated_reference_label"],
                "Soft voting": row["pred_soft_voting"],
                "Logistic regression": row["pred_logistic_regression"],
                "Random forest": row["pred_random_forest"],
                "Gradient boosting": row["pred_gradient_boosting"],
                "SVM": row["pred_svm_rbf"],
            }
        )
    st.dataframe(pd.DataFrame(flat), hide_index=True, width="stretch")
