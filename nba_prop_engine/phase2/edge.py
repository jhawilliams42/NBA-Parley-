"""
Phase 2 — Execution Price, Break-Even, Shrinkage, and Edge
Sections 2.7, 2.8, 2.9, 2.10
"""

from __future__ import annotations

import logging
from typing import Optional

from ..phase0.constants import SAMPLE_N_STRONG_THRESHOLD, SHRINKAGE_TABLE
from ..phase1.distribution import american_to_decimal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2.7 Execution price and break-even
# ---------------------------------------------------------------------------

def compute_execution_price_and_break_even(
    fd_execution_odds_american: Optional[float],
) -> dict:
    """
    Compute decimal execution price and break-even from FanDuel American odds.
    Section 2.7.

    Returns dict with fd_execution_odds_decimal, break_even, integrity_state.
    """
    if fd_execution_odds_american is None or fd_execution_odds_american == 0:
        return {
            "fd_execution_odds_decimal": None,
            "break_even": None,
            "execution_price_integrity_state": "INVALID",
        }

    try:
        decimal = american_to_decimal(fd_execution_odds_american)
    except ValueError as exc:
        return {
            "fd_execution_odds_decimal": None,
            "break_even": None,
            "execution_price_integrity_state": "INVALID",
            "error": str(exc),
        }

    break_even = 1.0 / decimal

    return {
        "fd_execution_odds_decimal": decimal,
        "break_even": break_even,
        "execution_price_integrity_state": "DERIVED_BY_APPROVED_RULE",
    }


# ---------------------------------------------------------------------------
# 2.8 Shrinkage
# ---------------------------------------------------------------------------

def _get_shrinkage_weight(
    repeatability_class: Optional[str],
    sample_n: int,
) -> float:
    """
    Look up shrinkage weight from the default shrinkage table. Section 2.8.
    """
    if repeatability_class == "HIGH":
        return SHRINKAGE_TABLE.get(("HIGH", "strong"), 0.00)
    if repeatability_class == "MODERATE":
        strength = "strong" if sample_n >= SAMPLE_N_STRONG_THRESHOLD else "weak"
        return SHRINKAGE_TABLE.get(("MODERATE", strength), 0.10)
    if repeatability_class == "LOW":
        return SHRINKAGE_TABLE.get(("LOW", None), 0.25)
    # UNKNOWN or missing
    return SHRINKAGE_TABLE.get(("UNKNOWN", None), 0.20)


def compute_model_prob(
    raw_event_prob: Optional[float],
    val_fair_p: Optional[float],
    repeatability_class: Optional[str],
    sample_n: int,
) -> dict:
    """
    Compute model_prob applying shrinkage toward val_fair_p. Section 2.8.

    Returns dict with model_prob, shrinkage_method_id, shrink_w.
    """
    if raw_event_prob is None:
        return {
            "model_prob": None,
            "shrinkage_method_id": "NONE_RAW_PROB_MISSING",
            "shrink_w": None,
        }

    if val_fair_p is None:
        return {
            "model_prob": raw_event_prob,
            "shrinkage_method_id": "NONE_NO_VALUATION",
            "shrink_w": 0.0,
        }

    shrink_w = _get_shrinkage_weight(repeatability_class, sample_n)
    model_prob = (1.0 - shrink_w) * raw_event_prob + shrink_w * val_fair_p

    return {
        "model_prob": model_prob,
        "shrinkage_method_id": "SHRINK_TO_VALUATION",
        "shrink_w": shrink_w,
    }


# ---------------------------------------------------------------------------
# 2.9 Edge computation
# ---------------------------------------------------------------------------

def compute_edge(
    model_prob: Optional[float],
    break_even: Optional[float],
) -> dict:
    """
    Compute current_edge_pct = (model_prob - break_even) / break_even.
    Section 2.9. Canonical edge field name: current_edge_pct (Section 0.6).
    """
    if model_prob is None or break_even is None or break_even == 0:
        return {
            "current_edge_pct": None,
            "edge_integrity_state": "INVALID",
        }

    current_edge_pct = (model_prob - break_even) / break_even

    return {
        "current_edge_pct": current_edge_pct,
        "edge_integrity_state": "DERIVED_BY_APPROVED_RULE",
    }


# ---------------------------------------------------------------------------
# 2.10 Predictive lower bound
# ---------------------------------------------------------------------------

def compute_predictive_lower_bound(obj: dict) -> dict:
    """
    Compute predictive_lower_bound if an approved method exists.
    Section 2.10.

    In v15.1, no approved method is mandated; returns UNASSESSED if not
    explicitly computed.
    """
    # Placeholder: production implementation would apply an approved statistical
    # lower-bound method (e.g., bootstrap confidence interval).
    approved_method = obj.get("predictive_lower_bound_method")

    if not approved_method:
        return {
            "predictive_lower_bound": None,
            "predictive_lower_bound_integrity_state": "UNASSESSED",
        }

    # If a method ID is provided but not implemented, mark unassessed
    return {
        "predictive_lower_bound": None,
        "predictive_lower_bound_integrity_state": "UNASSESSED",
    }
