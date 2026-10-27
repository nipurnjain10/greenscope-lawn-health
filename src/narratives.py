"""Interpretations computed from saved metrics.

The wording follows the numbers. It does not assume that soft voting won.
"""

from __future__ import annotations

from src.config import ALL_MODEL_SPECS, CLASS_ORDER, MODEL_SPECS


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _model(name: str) -> str:
    return ALL_MODEL_SPECS.get(name, name)


def performance_comparison_text(metrics: dict) -> str:
    test = {name: payload["test"] for name, payload in metrics["models"].items()}
    ranking = sorted(test.items(), key=lambda item: item[1]["accuracy"], reverse=True)
    best_name, best = ranking[0]
    soft = test["soft_voting"]
    individuals = {name: row for name, row in test.items() if name in MODEL_SPECS}
    best_individual_name, best_individual = max(individuals.items(), key=lambda item: item[1]["accuracy"])
    gap = soft["accuracy"] - best_individual["accuracy"]
    if gap > 0.005:
        ensemble_sentence = (
            f"Soft voting accuracy is {_pct(soft['accuracy'])}, which is {_pct(gap)} higher than "
            f"{_model(best_individual_name)} ({_pct(best_individual['accuracy'])}), the best individual model on this test set."
        )
    elif gap < -0.005:
        ensemble_sentence = (
            f"Soft voting accuracy is {_pct(soft['accuracy'])}. It does not beat {_model(best_individual_name)} "
            f"({_pct(best_individual['accuracy'])}) on this test set. Equal-weight averaging was still reported because "
            "it was the pre-specified ensemble, not because the test set selected it."
        )
    else:
        ensemble_sentence = (
            f"Soft voting ({_pct(soft['accuracy'])}) and {_model(best_individual_name)} "
            f"({_pct(best_individual['accuracy'])}) are within half a percentage point on this test set."
        )
    high_miss = soft["high_risk_missed"]
    high_support = soft["high_risk_support"]
    return (
        f"All figures in this comparison are from the same untouched test partition "
        f"(n = {best['support_total']}), after the models were refit on training plus validation. "
        f"The highest test accuracy is {_model(best_name)} at {_pct(best['accuracy'])} "
        f"(macro-F1 {_pct(best['macro_f1'])}). {ensemble_sentence} "
        f"Soft voting misses {high_miss} of {high_support} simulated high-risk cases, "
        f"including {soft['high_risk_predicted_low']} that it calls low risk. "
        "Accuracy alone hides that difference. These percentages describe the synthetic benchmark only."
    )


def confusion_text(model_name: str, block: dict) -> str:
    matrix = block["confusion_matrix"]
    # Rows are true class in CLASS_ORDER.
    lines = []
    for index, label in enumerate(CLASS_ORDER):
        correct = matrix[index][index]
        support = block["per_class"][label]["support"]
        lines.append(f"{label}: {correct} of {support} correct")
    return (
        f"For {_model(model_name)}, the matrix counts test rows. "
        + "; ".join(lines)
        + f". The most serious operational error in this prototype is a high-risk lawn called low risk: "
        f"{block['high_risk_predicted_low']} row(s). "
        "Off-diagonal counts are mistakes relative to the simulated label, not confirmed field mistakes."
    )


def per_class_text(metrics: dict) -> str:
    soft = metrics["models"]["soft_voting"]["test"]["per_class"]
    recalls = {label: soft[label]["recall"] for label in CLASS_ORDER}
    weakest = min(recalls, key=recalls.get)
    strongest = max(recalls, key=recalls.get)
    uneven = max(recalls.values()) - min(recalls.values()) > 0.08
    balance = (
        f"Recall is uneven. The weakest class for soft voting is {weakest} "
        f"({_pct(recalls[weakest])}) and the strongest is {strongest} ({_pct(recalls[strongest])})."
        if uneven
        else (
            f"Soft-voting recall is relatively even across classes, from {_pct(min(recalls.values()))} "
            f"to {_pct(max(recalls.values()))}."
        )
    )
    high = soft["High"]
    return (
        f"{balance} High-risk precision is {_pct(high['precision'])} and high-risk recall is {_pct(high['recall'])} "
        f"on {high['support']} test rows. A missed high-risk case is the error a follow-up plan should not ignore. "
        "Class-wise scores use the simulated labels."
    )


def robustness_text(metrics: dict) -> str:
    parts = []
    for name in ["logistic_regression", "random_forest", "gradient_boosting", "svm_rbf", "soft_voting"]:
        test = metrics["models"][name]["test"]
        shifted = metrics["models"][name]["robustness"]
        parts.append(
            f"{_model(name)} accuracy {_pct(test['accuracy'])} to {_pct(shifted['accuracy'])} "
            f"({(shifted['accuracy'] - test['accuracy']) * 100:+.1f} points), "
            f"macro-F1 {_pct(test['macro_f1'])} to {_pct(shifted['macro_f1'])} "
            f"({(shifted['macro_f1'] - test['macro_f1']) * 100:+.1f} points)"
        )
    soft_gap = metrics["models"]["soft_voting"]["robustness"]["accuracy"] - metrics["models"]["soft_voting"]["test"]["accuracy"]
    direction = "lower" if soft_gap < -0.01 else "higher" if soft_gap > 0.01 else "about the same"
    return (
        "The robustness file was generated with a different seed, a hotter and drier latent climate, "
        "a warmer grass mix, more traffic, less rain, higher laboratory missingness, and the original score cutoffs. "
        "It does not reuse test rows, and its class mix differs from the balanced test set, so accuracy and macro-F1 need not move together. "
        + "; ".join(parts)
        + f". On accuracy, soft voting is {direction} on the shifted sample. "
        "The comparison describes this synthetic shift. It is not an estimate of performance under real climate change."
    )


def ablation_text(metrics: dict) -> str:
    full = metrics["models"]["soft_voting"]["test"]["macro_f1"]
    sentences = []
    helpful = []
    for name, block in metrics["ablation"].items():
        delta = block["macro_f1"] - full
        sentences.append(f"without {_model(name)}: macro-F1 {_pct(block['macro_f1'])} ({delta * 100:+.1f} points)")
        if delta < -0.005:
            helpful.append(name)
    if helpful:
        judgement = (
            "Removing "
            + ", ".join(_model(name) for name in helpful)
            + " lowers macro-F1 by more than half a point, so those models contribute something the others do not fully replace on this test set."
        )
    else:
        judgement = (
            "Removing any one model changes macro-F1 by less than half a point. "
            "On this benchmark the four models are not adding large complementary gains beyond the best subset."
        )
    return (
        f"Full soft voting has test macro-F1 {_pct(full)}. Leave-one-out averages use the same fitted models and do not retrain the others. "
        + "; ".join(sentences)
        + ". "
        + judgement
    )


def calibration_text(metrics: dict) -> str:
    block = metrics["calibration"]["soft_voting"]
    return (
        f"For soft voting, the one-versus-rest macro expected calibration error is {block['macro_ece']:.3f} "
        f"and the multiclass Brier score is {block['multiclass_brier']:.3f}, both computed on the test predictions. "
        "A reliability point near the diagonal means that, inside that probability bin, the simulated label occurred about as often as the probability suggested. "
        "Empty bins are omitted. These quantities were not used to refit or recalibrate the models. "
        "They do not make the probabilities into real-world confidence."
    )


def importance_text(top_features: list[dict]) -> str:
    if not top_features:
        return "Permutation importance was not available."
    rendered = ", ".join(
        f"{item['feature'].replace('_', ' ')} ({item['importance_mean'] * 100:.1f} accuracy points)"
        for item in top_features[:5]
    )
    return (
        "Permutation importance is the drop in test accuracy when a column is shuffled. "
        f"The largest mean drops for this model are {rendered}. "
        "A large drop means the fitted model depends on that column in this synthetic test set. "
        "It does not mean the column caused the lawn condition, and it can give credit to a proxy for a latent variable the model never saw."
    )


def split_text(metrics: dict) -> str:
    counts = metrics["split_counts"]
    return (
        f"The primary 10,000 rows were stratified by the simulated risk label into "
        f"{counts['train']} training, {counts['validation']} validation, and {counts['test']} test rows "
        f"(seed {metrics['split_seed']}). Validation metrics come from models fit on training rows only. "
        "Test and robustness metrics come from the refit on training plus validation. "
        "The same row identifiers were used for every model."
    )
