# Model methodology

This note matches the saved run `2026-10-22T12:35:19+00:00` and generator `1.1.0`.

## Target

`overall_lawn_health_risk` with classes Low, Moderate, and High, in that order. All four models and the soft-voting ensemble use this target.

## Split

The primary 10,000 rows were stratified by the simulated risk label into 7000 training, 1500 validation, and 1500 test rows (seed 42). Validation metrics come from models fit on training rows only. Test and robustness metrics come from the refit on training plus validation. The same row identifiers were used for every model.

## Models

- logistic_regression: L2 logistic regression, C=1, lbfgs, scaled numeric inputs
- random_forest: 200 trees, max_depth=10, min_samples_leaf=8, max_features=sqrt
- gradient_boosting: 300 trees, learning_rate=0.05, max_depth=3, subsample=0.85. A validation-only grid preferred this over 150 trees at learning rate 0.08. The test set was not part of that choice.
- svm_rbf: RBF kernel, C=1, gamma=scale. Class probabilities come from CalibratedClassifierCV with sigmoid method, cv=3, and ensemble=False, fit on the training rows only.
- voting: Soft voting only. Equal weights. The class is the largest mean probability. Weights were not tuned on the test set.

## Protocol

Hyperparameters were pre-specified. Validation metrics use a fit on the training split only. Final models are refit on training plus validation and then scored once on the test split and on the separate robustness file. Permutation importance, calibration statistics, bootstrap intervals, and ablation use those final predictions. None of those evaluation steps changes the saved predictions.

## What is not claimed

The test metrics are not field accuracy. The probabilities are not customer confidence. The cost sheet is not a saving.
