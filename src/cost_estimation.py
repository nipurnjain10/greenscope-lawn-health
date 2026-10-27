"""Illustrative service-cost estimates from editable assumptions.

Prices are project defaults for planning demonstrations. They are not sourced
market quotations, and the estimates are not evidence of profit or savings.
"""

from __future__ import annotations

from copy import deepcopy

from src.config import DEFAULT_CURRENCY, PRICE_BOOKS


def default_assumptions(currency: str = DEFAULT_CURRENCY) -> dict:
    if currency not in PRICE_BOOKS:
        raise KeyError(f"Unknown currency '{currency}'. Choose one of {list(PRICE_BOOKS)}.")
    book = deepcopy(PRICE_BOOKS[currency])
    book["note"] = (
        f"Illustrative {currency} defaults chosen for this academic prototype. "
        "They are not a survey of local prices. The USD schedule is a separate illustration, "
        "not a converted exchange rate. Actual quotations need local costs and a site visit."
    )
    return book


def estimate_services(area_m2: float, service_keys: list[str], assumptions: dict) -> dict:
    if area_m2 is None or area_m2 != area_m2 or area_m2 <= 0:
        raise ValueError("Lawn area must be a positive number of square metres.")
    items = assumptions["items"]
    symbol = assumptions["symbol"]
    lines = []
    total = 0.0
    for key in service_keys:
        if key not in items:
            raise KeyError(f"Unknown service '{key}'.")
        spec = items[key]
        flat = float(spec["flat"])
        per_m2 = float(spec["per_m2"])
        amount = flat + per_m2 * float(area_m2)
        total += amount
        if per_m2 and flat:
            formula = f"{flat:g} + ({per_m2:g} × {float(area_m2):.1f} m²)"
        elif per_m2:
            formula = f"{per_m2:g} × {float(area_m2):.1f} m²"
        else:
            formula = f"{flat:g} flat"
        lines.append(
            {
                "service_key": key,
                "label": spec["label"],
                "flat": flat,
                "per_m2": per_m2,
                "area_m2": float(area_m2),
                "amount": round(amount, 2),
                "formula": formula,
                "display": f"{symbol}{amount:,.2f}",
            }
        )
    return {
        "currency": assumptions["currency"],
        "symbol": symbol,
        "lines": lines,
        "total": round(total, 2),
        "display_total": f"{symbol}{total:,.2f}",
        "area_m2": float(area_m2),
        "note": assumptions.get("note", ""),
        "what_this_is": (
            "A calculated illustration from the lawn area and the price assumptions shown above."
        ),
        "what_this_is_not": (
            "Not an observed invoice, not a replacement-cost study, and not evidence that the model saves money."
        ),
    }


def plans_from_recommendations(area_m2: float, recommendation: dict, assumptions: dict) -> dict:
    def keys_for(id_list: list[str]) -> list[str]:
        by_id = {item["action_id"]: item for item in recommendation["actions"]}
        ordered = []
        for action_id in id_list:
            service = by_id[action_id].get("service_key")
            if service and service not in ordered:
                ordered.append(service)
        return ordered

    minimal_keys = keys_for(recommendation["minimal_action_ids"])
    comprehensive_keys = keys_for(recommendation["comprehensive_action_ids"])
    return {
        "minimal": estimate_services(area_m2, minimal_keys, assumptions) if minimal_keys else _empty(assumptions, area_m2),
        "comprehensive": estimate_services(area_m2, comprehensive_keys, assumptions) if comprehensive_keys else _empty(assumptions, area_m2),
    }


def _empty(assumptions: dict, area_m2: float) -> dict:
    return {
        "currency": assumptions["currency"],
        "symbol": assumptions["symbol"],
        "lines": [],
        "total": 0.0,
        "display_total": f"{assumptions['symbol']}0.00",
        "area_m2": float(area_m2),
        "note": assumptions.get("note", ""),
        "what_this_is": "No priced service was selected.",
        "what_this_is_not": "Not evidence of a zero real-world cost.",
    }
