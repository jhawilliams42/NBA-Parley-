"""
Phase 2 — Fair Probability / De-Vig with Method Governance
Section 2.5
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..phase0.constants import DEVIG_METHOD_REGISTRY
from ..phase1.distribution import american_to_implied_prob

logger = logging.getLogger(__name__)

MAX_SHIN_ITERATIONS = 1000
SHIN_EPSILON = 1e-8


# ---------------------------------------------------------------------------
# Market structure validation
# ---------------------------------------------------------------------------

def validate_market_structure(
    odds_over_american: float,
    odds_under_american: float,
) -> dict:
    """
    Validate that a two-way market has acceptable structure for de-vig.
    Returns dict with 'valid', 'reason', 'p_over_implied', 'p_under_implied',
    'total_implied_prob'.
    """
    try:
        p_over = american_to_implied_prob(odds_over_american)
        p_under = american_to_implied_prob(odds_under_american)
    except (ValueError, ZeroDivisionError) as exc:
        return {
            "valid": False,
            "reason": f"INVALID_ODDS: {exc}",
            "p_over_implied": None,
            "p_under_implied": None,
            "total_implied_prob": None,
        }

    total = p_over + p_under

    if total <= 1.0:
        return {
            "valid": False,
            "reason": f"NO_VIG_OR_NEGATIVE_VIG: total_implied_prob={total:.4f}",
            "p_over_implied": p_over,
            "p_under_implied": p_under,
            "total_implied_prob": total,
        }

    if total > 1.30:
        return {
            "valid": False,
            "reason": f"EXCESSIVE_VIG: total_implied_prob={total:.4f}",
            "p_over_implied": p_over,
            "p_under_implied": p_under,
            "total_implied_prob": total,
        }

    return {
        "valid": True,
        "reason": None,
        "p_over_implied": p_over,
        "p_under_implied": p_under,
        "total_implied_prob": total,
    }


# ---------------------------------------------------------------------------
# 2.5 Proportional de-vig (PROPORTIONAL_V1) — production-authorized
# ---------------------------------------------------------------------------

def proportional_devig(
    p_over_implied: float,
    p_under_implied: float,
) -> tuple[float, float, float]:
    """
    PROPORTIONAL_V1 — Proportional Normalization de-vig.
    Production-authorized per Section 2.5.

    Returns: (fair_p_over, fair_p_under, vig_pct)
    """
    total_implied_prob = p_over_implied + p_under_implied
    fair_p_over = p_over_implied / total_implied_prob
    fair_p_under = p_under_implied / total_implied_prob
    vig_pct = (total_implied_prob - 1.0) * 100
    return fair_p_over, fair_p_under, vig_pct


# ---------------------------------------------------------------------------
# 2.5 Shin de-vig (SHIN_V1) — specified, not production-authorized
# ---------------------------------------------------------------------------

def shin_devig(
    p_over_implied: float,
    p_under_implied: float,
    max_iter: int = MAX_SHIN_ITERATIONS,
    epsilon: float = SHIN_EPSILON,
) -> tuple[float, float, float, int]:
    """
    SHIN_V1 — Shin Implicit Vig Model.
    NOT production-authorized in v15.1. Only for validation testing.
    Section 2.5.

    Returns: (fair_p_over, fair_p_under, vig_pct, iterations)
    Raises: ValueError if convergence fails.
    """
    # Shin model: solve for z (insider share) such that implied probs are
    # consistent with the Shin (1993) model.
    # p_i_implied = p_i_fair * (1 - z) + z * p_i_fair^2 / sum(p_j_fair^2)
    # Iterative solution using two-outcome case.

    # Initial estimate from proportional
    total = p_over_implied + p_under_implied
    p1 = p_over_implied / total
    p2 = p_under_implied / total

    z = 0.0
    for i in range(max_iter):
        # Compute denominator term
        denom = p1 ** 2 + p2 ** 2
        if denom <= 0:
            raise ValueError("SHIN_DEVIG_DENOM_ZERO")

        # Updated z estimate
        # For 2-outcome: z = (total - 1) / (1 - sum(p_fair^2))
        z_new = (total - 1.0) / (1.0 - denom)

        if abs(z_new - z) < epsilon:
            z = z_new
            break

        z = z_new

        # Update fair probabilities
        if abs(1.0 - z) < 1e-12:
            raise ValueError("SHIN_DEVIG_DIVISION_BY_ZERO_IN_UPDATE")

        p1 = (p_over_implied - z * p1 ** 2 / denom) / (1.0 - z)
        p2 = (p_under_implied - z * p2 ** 2 / denom) / (1.0 - z)

        # Clamp to [0, 1]
        p1 = max(0.0, min(1.0, p1))
        p2 = max(0.0, min(1.0, p2))

        # Renormalize
        norm = p1 + p2
        if norm > 0:
            p1 /= norm
            p2 /= norm
    else:
        raise ValueError(
            f"SHIN_DEVIG_DID_NOT_CONVERGE after {max_iter} iterations"
        )

    fair_p_over = p1
    fair_p_under = p2
    vig_pct = (total - 1.0) * 100
    return fair_p_over, fair_p_under, vig_pct, i + 1


# ---------------------------------------------------------------------------
# 2.5 Per-book processing dispatcher
# ---------------------------------------------------------------------------

def process_valuation_book(
    book_name: str,
    odds_over_american: float,
    odds_under_american: float,
    preferred_method: str = "PROPORTIONAL_V1",
) -> dict:
    """
    Process a single valuation book. Section 2.5.

    Args:
        book_name: Name of the book (DraftKings, BetMGM, Caesars).
        odds_over_american: Over price in American odds.
        odds_under_american: Under price in American odds.
        preferred_method: 'PROPORTIONAL_V1' or 'SHIN_V1'.

    Returns:
        Dict with val_fair_p, vig_pct, devig_method, book_status, etc.
    """
    validation = validate_market_structure(odds_over_american, odds_under_american)

    if not validation["valid"]:
        return {
            "book_name": book_name,
            "book_status": "EXCLUDED",
            "exclusion_reason": validation["reason"],
            "total_implied_prob": validation["total_implied_prob"],
        }

    p_over = validation["p_over_implied"]
    p_under = validation["p_under_implied"]

    # Guard against unauthorized SHIN_V1 in production
    if preferred_method == "SHIN_V1":
        if not DEVIG_METHOD_REGISTRY["SHIN_V1"]["production_authorized"]:
            return {
                "book_name": book_name,
                "book_status": "EXCLUDED",
                "exclusion_reason": "SHIN_V1_NOT_PRODUCTION_AUTHORIZED",
            }

    if preferred_method == "SHIN_V1":
        try:
            fair_p_over, fair_p_under, vig_pct, iterations = shin_devig(
                p_over, p_under,
                max_iter=MAX_SHIN_ITERATIONS,
                epsilon=SHIN_EPSILON,
            )
            return {
                "book_name": book_name,
                "devig_method": "SHIN_V1",
                "shin_iterations": iterations,
                "val_fair_p": fair_p_over,
                "vig_pct": vig_pct,
                "book_status": "SUCCESS",
            }
        except ValueError as shin_error:
            # Fallback to PROPORTIONAL_V1
            fair_p_over, fair_p_under, vig_pct = proportional_devig(p_over, p_under)
            return {
                "book_name": book_name,
                "devig_method": "PROPORTIONAL_V1",
                "shin_status": "FALLBACK_FROM_SHIN",
                "shin_error": str(shin_error),
                "val_fair_p": fair_p_over,
                "vig_pct": vig_pct,
                "book_status": "FALLBACK_SUCCESS",
            }

    # Default: PROPORTIONAL_V1
    fair_p_over, fair_p_under, vig_pct = proportional_devig(p_over, p_under)
    return {
        "book_name": book_name,
        "devig_method": "PROPORTIONAL_V1",
        "val_fair_p": fair_p_over,
        "vig_pct": vig_pct,
        "book_status": "SUCCESS",
    }


# ---------------------------------------------------------------------------
# 2.5 Consensus construction
# ---------------------------------------------------------------------------

def build_fair_probability_consensus(valuation_books_data: list[dict]) -> dict:
    """
    Aggregate per-book fair probabilities into a consensus. Section 2.5.

    Returns dict with val_fair_p, val_book_count, valuation_integrity_state,
    val_disagreement_score, valuation_books.
    """
    valid_books = [
        b for b in valuation_books_data
        if b.get("book_status") in ("SUCCESS", "FALLBACK_SUCCESS")
    ]

    if not valid_books:
        return {
            "val_fair_p": None,
            "val_book_count": 0,
            "valuation_integrity_state": "NO_VALID_BOOKS",
            "val_disagreement_score": None,
            "valuation_books": valuation_books_data,
        }

    fair_probs = [b["val_fair_p"] for b in valid_books]
    val_fair_p = float(np.median(fair_probs))
    val_disagreement_score = float(np.std(fair_probs))

    return {
        "val_fair_p": val_fair_p,
        "val_book_count": len(valid_books),
        "valuation_integrity_state": "VERIFIED",
        "val_disagreement_score": val_disagreement_score,
        "valuation_books": valuation_books_data,
    }
