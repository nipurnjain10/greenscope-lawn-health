# Help — GreenScope

This guide is for a student who has the project folder and has not read the source. Commands are run from the project root, the folder that contains `app.py`.

## Prerequisites

- Python. This copy was tested with Python 3.14.7. Python 3.11 was not installed here, so it was not tested.
- Enough disk space for a virtual environment, two CSV files, four model files, and the report figures. A student laptop is sufficient.
- No database, no account, and no internet connection after the packages have been installed. The generator does not download the 10,000 rows.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions that were installed for the saved experiment: NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, SciPy 1.18.1, Streamlit 1.66.1, Plotly 7.1.0, joblib 1.6.0, pytest 9.1.1, python-docx 1.2.0, python-pptx 1.0.2, and Matplotlib 3.11.2. SHAP is not used.

## Dataset generation

```bash
python data/generate_synthetic_data.py
```

This writes:

- `data/synthetic_lawn_inspections.csv` — 10,000 inspections, seed 42, generator 1.1.0
- `data/robustness_scenarios.csv` — 2,000 inspections, seed 20261022, with a hotter and drier scenario and the primary score cutoffs
- `data/data_dictionary.md`
- `data/metadata.json`
- `reports/data_validation_report.md`

The same command validates the files. A second validation-only run is:

```bash
python -m src.data_validation
```

The rows are synthetic. The extension pages listed in `DATA_SOURCES.md` informed the meaning of sunlight, drainage, pH, soil tests, water, mowing, and thatch. They did not supply the table.

## Model training

```bash
export MPLCONFIGDIR="$PWD/.matplotlib"
python -m src.train_models
```

The script refuses to train if the CSV fingerprint does not match `data/metadata.json`. It fits logistic regression, a random forest, gradient boosting, and a calibrated RBF support-vector machine on the same stratified 70/15/15 split. Validation scores come from a fit on the training rows. The saved models are then refit on training plus validation and scored once on the test rows and on the robustness file.

Outputs:

- `models/*.joblib` and `models/run_metadata.json`
- `reports/model_metrics.json` and `reports/model_metrics.csv`
- `reports/test_predictions.csv`
- `reports/permutation_importance.csv`
- `reports/demo_scenarios.json`
- `reports/figures/`
- `data/split_assignments.csv`

Training takes several minutes because the SVM uses three-fold probability calibration and permutation importance runs in one process. Do not stop it and then quote a partial metrics file.

Rebuild the report after training:

```bash
python -m src.build_documents
```

## Running the application

```bash
python -m streamlit run app.py
```

Open the local URL Streamlit prints. The sidebar states that this is an independent academic prototype and that the predictions have not been validated on customer lawns.

If the overview says the models are missing or the fingerprint does not match, generate the data and train again. The app will not invent accuracy numbers.

## How to complete a lawn assessment

Open **Lawn assessment**.

1. Enter the lawn area, grass type, soil texture, and foot-traffic level. Those four are required. They are not imputed.
2. Enter the other measurements you actually have. Mark a measurement **unknown** when it was not taken. Unknown is stored as missing and imputed inside the saved training pipeline. It is not replaced with a healthy value.
3. Read the unit and the help text. Values outside the allowed range are rejected. Values inside the form limits but outside the range used to generate the training data produce a warning.
4. Submit the inspection.

The page then shows:

- the soft-voting class and the three class probabilities
- each model's class and probabilities
- whether the four models agree
- a local explanation: the change in the predicted-class probability when one observed value is replaced by the fitting-set median or mode
- water, nutrient, and drainage screens, labelled as rules
- actions, a follow-up interval, and an illustrative cost

The simulated benchmark label is not shown here. A new lawn does not have one.

You can save the assessment locally in `output/assessment_history.json`. Do not type a customer name, phone number, or address. The history compares inputs and predictions. A change in class is not evidence that a treatment worked.

## How to run the demonstration

Open **Demo test lab** and use **Run Demonstration Assessment**.

The page selects a row from the held-out test split, shows the inputs, and runs the four saved models. It does not fit a new model. The simulated benchmark label is shown on this page only, and it is labelled as a simulated benchmark label. A match means the model agreed with the generator. It does not mean the lawn was diagnosed.

You can also pick a stored scenario: a generally healthy lawn, likely water stress, drainage concerns, nutrient-related concerns, several interacting factors, or a borderline case. Those rows are generated test cases, not hand-written predictions.

The session keeps a short list of the demonstration cases already run.

## How to interpret probabilities and risk indicators

The three classes are Low, Moderate, and High. They are tertiles of a noisy score on the primary file. Moderate is the class the models confuse most often. Read the research page before treating a probability as confidence.

A probability is the model's score for a class. Soft voting averages the four scores with equal weights. If the models disagree, say so: there is no three-vote majority, and the class should be treated as unsettled.

The calibration chart asks a narrower question: among test rows whose score fell in a bin, how often was the simulated label that class? A point near the diagonal is agreement with the synthetic label inside that bin. It is not a confidence interval for a real lawn.

Water, nutrient, and drainage levels are `none`, `possible`, `likely`, or `unavailable`. They are written screens. They are not separate trained classifiers, and they do not have their own probabilities. A missing soil test makes the nutrient screen unavailable. It does not become "healthy."

## How to understand the plots

**Model results and data**

- The comparison chart uses the same 1,500 test rows for every model. Do not compare a training score with a test score.
- A confusion matrix has the simulated class in the rows and the prediction in the columns, in the order Low, Moderate, High. The operational miss emphasised in the report is a high-risk row predicted as low risk.
- Permutation importance is the drop in test accuracy when one column is shuffled. A large drop means the model depends on that column in this file. It does not mean the column caused the condition.
- The robustness chart compares the standard test set with a file that was generated under a shifted climate and species mix. An accuracy change describes that designed shift.
- The ablation removes one model's probabilities from the average. The other models are not retrained. If removing a model raises the score, the report says so.

**Dataset explorer**

Every chart is computed from the saved CSV. The page states that the rows are synthetic.

## How costs and recommendations are calculated

Each recommendation names the concern, the inputs that triggered it, the next action, an urgency, a rationale, and a limitation. Priority is urgency weight times evidence strength. Cost does not change the rank. The follow-up interval is 14, 28, or 56 days. It is an operational heuristic.

A priced line is a flat amount plus a per-square-metre amount times the lawn area. The default books are illustrative Indian rupees and a separate illustrative US dollar schedule. The dollar figures are not an exchange-rate conversion of the rupee figures. Edit the amounts on **Recommendations and costs** before discussing a budget. The result is not an invoice and not a measured saving.

The actions do not state a fertiliser, lime, sulfur, or pesticide rate. A high overall class does not by itself add a nutrient action.

## Where files are saved

| What | Where |
| --- | --- |
| Inspections | `data/synthetic_lawn_inspections.csv`, `data/robustness_scenarios.csv` |
| Models | `models/` |
| Metrics, predictions, figures | `reports/` |
| Report and slides | `submission/` |
| Local assessment history | `output/assessment_history.json` (not a customer database) |

## Group details

Edit `group_info.json` with the real group number, names, and enrollment numbers. Replace each contribution sentence only after that member confirms it. Then run:

```bash
python -m src.build_documents
```

Check the printed page numbers in the Word file. The template records section locations because pagination depends on the word processor.

## Troubleshooting

- **Fingerprint mismatch.** The CSV changed after training. Run `python -m src.train_models` again. Do not mix metrics from an old run with a new CSV.
- **Matplotlib cannot create a directory in your home folder.** Set `MPLCONFIGDIR` to a folder inside the project, as in the training command.
- **A joblib error about semaphores.** Permutation importance is set to one process for this reason. Do not switch it back to parallel workers inside a restricted sandbox.
- **The overview has no accuracy.** Training has not finished, or `reports/model_metrics.json` is missing. The page leaves the number blank on purpose.
- **A form value is rejected.** It is outside the documented unit range. Correct it or mark it unknown if the measurement was not taken. Required site fields cannot be unknown.
- **The demonstration label disagrees with the prediction.** That is a real error on the synthetic benchmark. Read the probabilities and the confusion matrix. Do not edit the displayed class.
- **Import error for `src.ui.common`.** The helpers live in `src/ui/common.py`. Run the app from the project root so `src` is importable.

## Limitations

- Agreement with a simulated label is not agronomic accuracy.
- Maryland, New Jersey, and FAO pages are domain context. They are not a sample of the lawns a local company services, and the prices are not a market survey.
- The nitrogen, phosphorus, and potassium columns use a project milligram-per-kilogram scale. They are not a laboratory sufficiency class, and a routine soil test does not establish turfgrass nitrogen need.
- Imputed laboratory values are training statistics, not measurements.
- The prototype does not claim a profit, a retention gain, or a treatment effect.
