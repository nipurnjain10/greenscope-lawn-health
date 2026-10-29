# GreenScope — Lawn Health & Service Planning

An explainable machine-learning decision-support prototype for landscaping operations.

**Independent academic prototype. Model development and evaluation use synthetic data; performance has not been validated on actual customer lawns.**

## What it does

A user enters a lawn inspection. Four saved classifiers predict one simulated risk class: Low, Moderate, or High. Soft voting averages their class probabilities. The application then shows agreement among the models, a local explanation, written water, nutrient, and drainage screens, maintenance actions, and an editable illustrative cost.

The results page shows the same models on one held-out test set, a separately generated robustness file, confusion matrices, feature importance, and the dataset. Calibration, ablation, and extra distributions sit in an additional-experiments section.

## Environment that was actually tested

The packages in `requirements.txt` were installed and run with **Python 3.14.7** on macOS. Python 3.11 was requested and was not available on this machine, so compatibility with 3.11 was not checked. Create the virtual environment with the same interpreter you intend to use.

## Setup

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## Reproduce the experiment

The CSV files and the trained models in this folder were produced by the commands below. Run them again if you move the project or change the generator.

```bash
python data/generate_synthetic_data.py
python -m src.data_validation
export MPLCONFIGDIR="$PWD/.matplotlib"
python -m src.train_models
python -m src.build_documents
pytest
```

`MPLCONFIGDIR` keeps Matplotlib from writing its cache outside the project. The training script also sets it.

Generation writes 10,000 primary rows and 2,000 robustness rows, then validates them. Training fits the four pipelines, scores the untouched test set once, and writes `reports/model_metrics.json`.

## Run the application

```bash
python -m streamlit run app.py
```

The app loads the saved models. It does not retrain when you change pages or run a demonstration.

## Main files

| Path | Role |
| --- | --- |
| `app.py` | Streamlit navigation |
| `data/generate_synthetic_data.py` | Reproducible synthetic inspections |
| `src/train_models.py` | Fit, evaluate, and save the models |
| `src/build_documents.py` | Report, presentation, and submission copies |
| `HELP.md` | How to use the prototype |
| `submission/` | Report, slides, dictionary, provenance, and contribution template |
