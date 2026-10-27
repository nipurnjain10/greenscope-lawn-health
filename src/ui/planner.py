"""Editable illustrative prices used by the lawn assessment."""

from __future__ import annotations

import streamlit as st

from src.config import PRICE_BOOKS
from src.cost_estimation import default_assumptions
from src.ui.common import note


def edit_price_assumptions() -> dict:
    """Return the session price book after the user edits flat fees and per-square-metre rates."""
    currency = st.selectbox("Currency schedule", options=list(PRICE_BOOKS), index=0)
    if st.session_state.get("price_currency") != currency:
        st.session_state["price_assumptions"] = default_assumptions(currency)
        st.session_state["price_currency"] = currency
    assumptions = st.session_state.setdefault("price_assumptions", default_assumptions(currency))
    note(assumptions["note"])
    edited = {}
    for key, spec in assumptions["items"].items():
        columns = st.columns([3, 1, 1])
        columns[0].write(spec["label"])
        flat_key = f"flat_{currency}_{key}"
        rate_key = f"perm2_{currency}_{key}"
        flat_kwargs = {
            "min_value": 0.0,
            "step": 10.0 if currency == "INR" else 1.0,
            "key": flat_key,
            "label_visibility": "collapsed",
        }
        rate_kwargs = {
            "min_value": 0.0,
            "step": 0.5 if currency == "INR" else 0.01,
            "key": rate_key,
            "label_visibility": "collapsed",
        }
        if flat_key not in st.session_state:
            flat_kwargs["value"] = float(spec["flat"])
        if rate_key not in st.session_state:
            rate_kwargs["value"] = float(spec["per_m2"])
        flat = columns[1].number_input(f"Flat fee, {spec['label']}", **flat_kwargs)
        per_m2 = columns[2].number_input(f"Per square metre, {spec['label']}", **rate_kwargs)
        edited[key] = {"label": spec["label"], "flat": float(flat), "per_m2": float(per_m2)}
    assumptions["items"] = edited
    st.session_state["price_assumptions"] = assumptions
    note(
        "Each line is a flat fee plus the per-square-metre rate times lawn area. "
        "These figures are illustrative planning assumptions, not a market survey or a proven saving."
    )
    return assumptions
