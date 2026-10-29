"""Training-only preprocessing pipelines.

Imputation, encoding, and scaling statistics are learned inside each pipeline's
``fit`` call. Callers must fit on the training partition only.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SPLIT_SEED


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def make_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _one_hot()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_estimators() -> dict[str, Pipeline]:
    logistic = Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=SPLIT_SEED,
                ),
            ),
        ]
    )
    forest = Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=8,
                    min_samples_split=16,
                    max_features="sqrt",
                    random_state=SPLIT_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    boosting = Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric=False)),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.85,
                    random_state=SPLIT_SEED,
                ),
            ),
        ]
    )
    svm = Pipeline(
        [
            ("preprocess", make_preprocessor(scale_numeric=True)),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        cache_size=512,
                        random_state=SPLIT_SEED,
                    ),
                    method="sigmoid",
                    cv=3,
                    ensemble=False,
                ),
            ),
        ]
    )
    return {
        "logistic_regression": logistic,
        "random_forest": forest,
        "gradient_boosting": boosting,
        "svm_rbf": svm,
    }
