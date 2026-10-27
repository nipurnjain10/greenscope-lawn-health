"""Methodology, interpretation limits, and links to the project documents."""

from __future__ import annotations

import streamlit as st

from src.ui.common import note, page_intro


def render_methodology() -> None:
    page_intro(
        "Documentation",
        "Methodology and help",
        "What the prototype does, what the numbers mean, and what they do not mean.",
    )
    st.subheader("Objective and scope")
    st.markdown(
        """
        The research question is whether an ensemble of diverse supervised classifiers can produce a useful and explainable lawn-health risk assessment on a source-informed synthetic benchmark, and how that assessment can support transparent, cost-aware landscaping decisions.

        The operational output is one of three simulated classes: Low, Moderate, or High. Those classes are bins of a noisy stress score. They are not a laboratory diagnosis and they have not been checked on customer lawns.

        The prototype does not prescribe fertiliser rates, lime rates, or pesticides. It does not estimate profit.
        """
    )
    st.subheader("Where the rows come from")
    st.markdown(
        """
        `data/synthetic_lawn_inspections.csv` has 10,000 generated inspections. `data/robustness_scenarios.csv` has 2,000 more, from a different seed and a hotter, drier, more warm-season scenario. The same score cutoffs are applied to both.

        Hidden variables create dependence among rainfall, moisture, nutrients, traffic, and compaction. The risk score itself is a function of the simulated measurements — heat with dryness, pH away from the mildly acidic band together with low phosphorus, poor drainage with wet soil — plus Gaussian noise. The hidden variables and the score are not columns in the CSV and are not given to the models.

        Some pH and nutrient values are then hidden, more often when the last service was at least 100 days earlier. The model imputes those cells from the training partition only.
        """
    )
    note("Read data/source_register.md and submission/DATASET_PROVENANCE.md before describing the file as an external dataset. It is not one.")
    st.subheader("Models and voting")
    st.markdown(
        """
        Logistic regression is the linear baseline, with scaled numeric inputs. The random forest and gradient boosting use unscaled inputs because their splits do not need it. The support-vector machine uses an RBF kernel and the probability estimates produced during its training fit. Those probabilities were not recalibrated on the test set.

        The primary 10,000 rows are stratified 70% / 15% / 15% into train, validation, and test. Validation metrics come from a fit on the training rows. The saved models are refit on training plus validation. Test metrics, robustness metrics, permutation importance, calibration, and ablation all use that refit and do not change it.

        Soft voting is the unweighted average of the four probability vectors. The predicted class is the largest average. There is no second voting rule.
        """
    )
    st.subheader("Explanations, screens, and money")
    st.markdown(
        """
        Permutation importance shuffles one test-set column at a time and records the drop in accuracy. A local explanation replaces one observed value with the fitting-set median or mode and records the change in the soft-voting probability of the predicted class. Both describe the model. Neither proves a biological cause.

        Water, nutrient, and drainage indicators are written rules. A rule can fire when the overall class is low, and a high overall class does not by itself select a nutrient action.

        Service totals are flat fees plus a per-square-metre rate times the entered lawn area. The default currency schedules are illustrative. They are not a price survey, and the US dollar schedule is not an exchange-rate conversion.
        """
    )
    st.subheader("How to read a result")
    st.markdown(
        """
        Use the class, the probability, and the agreement together. If the four models disagree, the class is unsettled. A probability is a model score; the research page shows whether similar scores matched simulated-label frequencies.

        A later assessment can be compared with an earlier one. A change in the predicted class means the inputs changed. It does not mean a treatment worked.

        Accuracy on this benchmark is not accuracy on a company's customers, and it is not evidence of higher profit.
        """
    )
    st.subheader("Domain references")
    st.markdown(
        """
        - [University of Maryland Extension — Starting a New Lawn](https://extension.umd.edu/resource/starting-new-lawn)
        - [University of Maryland Extension — Maintaining an Established Lawn](https://extension.umd.edu/resource/maintaining-established-lawn)
        - [Rutgers Cooperative Extension — Soil Testing for Home Lawns and Gardens (FS797)](https://njaes.rutgers.edu/fs797/)
        - [Rutgers Cooperative Extension — Fertilizing the Home Lawn (FS633)](https://njaes.rutgers.edu/FS633/)
        - [Rutgers Cooperative Extension — Your Lawn and Its Care (FS102)](https://njaes.rutgers.edu/fs102/)
        - [FAO EcoCrop](https://ecocrop.apps.fao.org/ecocrop/srv/en/home)
        """
    )
    st.subheader("Project documents")
    st.markdown(
        """
        - `HELP.md` — install, train, and use the application
        - `README.md` — short map of the repository
        - `data/data_dictionary.md` — columns, units, and bounds
        - `data/source_register.md` — what each public page contributed
        - `reports/data_validation_report.md` — checks on the generated files
        - `reports/model_metrics.json` — the numbers behind the research page
        - `PROJECT_REPORT.md` — the written experiment note already in the project folder
        """
    )


def render_methodology_brief() -> None:
    st.markdown(
        """
        Logistic regression, a random forest, gradient boosting, and a calibrated RBF support-vector machine predict one simulated class: Low, Moderate, or High. Soft voting averages their probabilities with equal weights. The SVM probabilities come from training-fold calibration and were not refit on the test set.

        Water, nutrient, and drainage indicators are written rules. A high overall class does not by itself select a fertiliser. Service totals use the lawn area and the editable price book. They are not a market quotation or a proven saving.

        The 10,000-row file and the 2,000-row robustness file were generated by this project. Extension pages informed the feature meanings. They did not supply the rows.
        """
    )
    st.markdown(
        """
        - [University of Maryland Extension — Starting a New Lawn](https://extension.umd.edu/resource/starting-new-lawn)
        - [University of Maryland Extension — Maintaining an Established Lawn](https://extension.umd.edu/resource/maintaining-established-lawn)
        - [Rutgers Cooperative Extension — Soil Testing for Home Lawns and Gardens (FS797)](https://njaes.rutgers.edu/fs797/)
        - [Rutgers Cooperative Extension — Fertilizing the Home Lawn (FS633)](https://njaes.rutgers.edu/FS633/)
        - [Rutgers Cooperative Extension — Your Lawn and Its Care (FS102)](https://njaes.rutgers.edu/fs102/)
        - [FAO EcoCrop](https://ecocrop.apps.fao.org/ecocrop/srv/en/home)
        - `HELP.md`, `data/data_dictionary.md`, `data/source_register.md`, `reports/model_metrics.json`
        """
    )
