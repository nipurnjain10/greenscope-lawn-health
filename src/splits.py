"""Reproducible stratified split of the primary synthetic sample."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ID_COLUMN, SPLIT_FRACTIONS, SPLIT_SEED, TARGET_COLUMN


def make_split_table(frame: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Return inspection_id and split. Partitions do not overlap.

    The validation fraction is taken from the non-test remainder so the nominal
    70/15/15 shares refer to the full sample.
    """
    identifiers = frame[ID_COLUMN].astype(str)
    target = frame[TARGET_COLUMN].astype(str)
    id_trainval, id_test, y_trainval, _ = train_test_split(
        identifiers,
        target,
        test_size=SPLIT_FRACTIONS["test"],
        stratify=target,
        random_state=seed,
    )
    relative_validation = SPLIT_FRACTIONS["validation"] / (
        SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["validation"]
    )
    id_train, id_validation, _, _ = train_test_split(
        id_trainval,
        y_trainval,
        test_size=relative_validation,
        stratify=y_trainval,
        random_state=seed,
    )
    parts = [
        pd.DataFrame({"inspection_id": id_train.astype(str), "split": "train"}),
        pd.DataFrame({"inspection_id": id_validation.astype(str), "split": "validation"}),
        pd.DataFrame({"inspection_id": id_test.astype(str), "split": "test"}),
    ]
    table = pd.concat(parts, ignore_index=True)
    if table["inspection_id"].duplicated().any():
        raise RuntimeError("A row was assigned to more than one split.")
    if set(table["inspection_id"]) != set(identifiers):
        raise RuntimeError("The split does not cover every inspection.")
    return table


def subset_by_split(frame: pd.DataFrame, split_table: pd.DataFrame, split_name: str) -> pd.DataFrame:
    ids = set(split_table.loc[split_table["split"] == split_name, "inspection_id"].astype(str))
    selected = frame.loc[frame[ID_COLUMN].astype(str).isin(ids)].copy()
    if len(selected) != len(ids):
        raise RuntimeError(f"Could not recover every {split_name} row from the feature table.")
    return selected
