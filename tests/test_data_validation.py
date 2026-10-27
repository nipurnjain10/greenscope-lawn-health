"""Validation must fail closed on impossible values and must not clip them."""

from __future__ import annotations

from src.data_validation import validate_frame
from src.synthetic_data import generate_inspections


def test_valid_sample_passes():
    frame, _ = generate_inspections(300, seed=11, scenario="main")
    result = validate_frame(frame, expected_n=300, dataset_name="unit")
    assert result["passed"], result["errors"]


def test_impossible_ph_is_reported_and_not_rewritten():
    frame, _ = generate_inspections(300, seed=11, scenario="main")
    original = frame["soil_ph"].iloc[0]
    frame.loc[frame.index[0], "soil_ph"] = 12.5
    result = validate_frame(frame, expected_n=300, dataset_name="unit")
    assert not result["passed"]
    assert any("soil_ph" in error for error in result["errors"])
    assert frame["soil_ph"].iloc[0] == 12.5
    assert original != 12.5


def test_unexpected_missing_field_fails():
    frame, _ = generate_inspections(300, seed=11, scenario="main")
    frame.loc[frame.index[0], "temperature_c"] = None
    result = validate_frame(frame, expected_n=300, dataset_name="unit")
    assert not result["passed"]
    assert any("temperature_c" in error for error in result["errors"])


def test_latent_column_is_rejected():
    frame, _ = generate_inspections(300, seed=11, scenario="main")
    frame["water_latent"] = 0.0
    result = validate_frame(frame, expected_n=300, dataset_name="unit")
    assert not result["passed"]
