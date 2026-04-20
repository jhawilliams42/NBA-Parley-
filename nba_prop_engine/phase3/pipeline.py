"""
Phase 3 — Ticket Construction, Portfolio Build
Sections 3.1–3.13
"""

from __future__ import annotations

import itertools
import logging
import uuid
from typing import Optional

from ..phase0.constants import (
    PHASE2_ALLOWED_STATUSES,
    TICKET_FAMILY_TYPES,
)
from ..phase0.governance import emit_circuit_breaker
from ..phase0.hash_utils import freeze_object_with_hash, verify_hash
from ..phase1.distribution import american_to_decimal
from .joint_prob import compute_joint_probability

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 3.2 Entry requirement check
# ---------------------------------------------------------------------------

def check_leg_phase3_eligible(leg: dict) -> tuple[bool, list[str]]:
    """Section 3.2 eligibility check for a single leg."""
    failures: list[str] = []

    if not verify_hash(leg, "phase2_frozen_hash", phase=2)["valid"]:
        failures.append("PHASE2_HASH_INVALID")

    if leg.get("leg_approval_status") != "APPROVED":
        failures.append(f"NOT_APPROVED: {leg.get('leg_approval_status')}")

    if leg.get("bucket") == "INELIGIBLE":
        failures.append("BUCKET_INELIGIBLE")

    if leg.get("scope_violation"):
        failures.append("SCOPE_VIOLATION")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# 3.4 Unique player rule
# ---------------------------------------------------------------------------

def check_unique_player_rule(legs: list[dict], family_id: str) -> Optional[str]:
    """Section 3.4: no duplicate player in standard tickets."""
    is_sgp = family_id.startswith("SGP_")
    if is_sgp:
        return None  # Same-player multi-prop allowed in SGP under approved rules

    player_ids = [leg.get("player_id") for leg in legs]
    if len(player_ids) != len(set(player_ids)):
        return "DUPLICATE_PLAYER_IN_STANDARD_TICKET"

    return None


# ---------------------------------------------------------------------------
# 3.5 Same-game rule
# ---------------------------------------------------------------------------

def check_same_game_rule(legs: list[dict], family_id: str) -> Optional[str]:
    """Section 3.5: max one leg per game in standard tickets."""
    is_sgp = family_id.startswith("SGP_")
    if is_sgp:
        return None

    game_ids = [leg.get("game_id") for leg in legs]
    if len(game_ids) != len(set(game_ids)):
        # More than one leg from same game without approved adjustment
        for leg in legs:
            if leg.get("same_game_dependence_method_id") in (None, "NONE_INDEPENDENCE_ALLOWED"):
                return "SAME_GAME_MULTIPLE_LEGS_WITHOUT_APPROVED_ADJUSTMENT"

    return None


# ---------------------------------------------------------------------------
# 3.6 Same-player SGP rule
# ---------------------------------------------------------------------------

def check_same_player_sgp_rule(legs: list[dict]) -> Optional[str]:
    """Section 3.6: same-player SGP requires FanDuel SGP support."""
    player_ids = [leg.get("player_id") for leg in legs]
    if len(player_ids) != len(set(player_ids)):
        # Same player — check SGP support
        for leg in legs:
            if not leg.get("fd_sgp_supported"):
                return f"SAME_PLAYER_SGP_NOT_SUPPORTED: player_id={leg.get('player_id')}"
        for leg in legs:
            if leg.get("dependence_method_id") in (None, "NONE_INDEPENDENCE_ALLOWED"):
                return "SAME_PLAYER_SGP_NO_APPROVED_DEPENDENCE_METHOD"
    return None


# ---------------------------------------------------------------------------
# 3.11 Ticket pricing
# ---------------------------------------------------------------------------

def compute_ticket_price(legs: list[dict], family_id: str) -> dict:
    """
    Compute ticket execution price. Section 3.11.
    For SGP tickets with FanDuel SGP price available, use that directly.
    For standard tickets, multiply FanDuel leg decimals.
    """
    is_sgp = family_id.startswith("SGP_")

    # Check if exact FanDuel SGP price is available for SGP tickets
    if is_sgp:
        sgp_price_american = None
        for leg in legs:
            if leg.get("fd_sgp_price_american") is not None:
                sgp_price_american = leg["fd_sgp_price_american"]
                break

        if sgp_price_american is not None:
            try:
                sgp_decimal = american_to_decimal(sgp_price_american)
                return {
                    "fd_ticket_price_american": sgp_price_american,
                    "fd_ticket_price_decimal": sgp_decimal,
                    "ticket_price_method": "FANDUEL_SGP_EXACT",
                    "ticket_price_integrity_state": "VERIFIED",
                }
            except ValueError as exc:
                pass

    # Standard ticket: product of FanDuel leg decimals
    product = 1.0
    for leg in legs:
        decimal = leg.get("fd_execution_odds_decimal")
        if decimal is None:
            return {
                "fd_ticket_price_american": None,
                "fd_ticket_price_decimal": None,
                "ticket_price_method": "PRODUCT_OF_LEG_DECIMALS",
                "ticket_price_integrity_state": "INVALID",
                "failure_reason": f"MISSING_DECIMAL_ODDS for player_id={leg.get('player_id')}",
            }
        product *= decimal

    return {
        "fd_ticket_price_american": None,  # Not computed from parlay for display
        "fd_ticket_price_decimal": product,
        "ticket_price_method": "PRODUCT_OF_LEG_DECIMALS",
        "ticket_price_integrity_state": "DERIVED_BY_APPROVED_RULE",
    }


# ---------------------------------------------------------------------------
# 3.12 Ticket EV
# ---------------------------------------------------------------------------

def compute_ticket_ev(joint_prob: Optional[float], fd_ticket_price_decimal: Optional[float]) -> Optional[float]:
    """ticket_ev_pct = (joint_prob * fd_ticket_price_decimal) - 1. Section 3.12."""
    if joint_prob is None or fd_ticket_price_decimal is None:
        return None
    return joint_prob * fd_ticket_price_decimal - 1.0


# ---------------------------------------------------------------------------
# Ticket construction
# ---------------------------------------------------------------------------

def build_ticket(
    legs: list[dict],
    family_id: str,
) -> dict:
    """
    Construct a single approved ticket from a list of approved legs.
    Sections 3.1–3.13.
    """
    ticket = {
        "ticket_id": f"TKT_{uuid.uuid4().hex[:8].upper()}",
        "family_id": family_id,
        "leg_ids": [leg.get("player_id", leg.get("id", "")) for leg in legs],
        "same_game_flag": _has_same_game(legs),
        "same_player_flag": _has_same_player(legs),
    }

    # Validation rules
    violation = check_unique_player_rule(legs, family_id)
    if violation:
        ticket["ticket_integrity_status"] = "FAIL"
        ticket["failure_reason"] = violation
        return ticket

    violation = check_same_game_rule(legs, family_id)
    if violation:
        ticket["ticket_integrity_status"] = "FAIL"
        ticket["failure_reason"] = violation
        return ticket

    if ticket["same_player_flag"] and family_id.startswith("SGP_"):
        violation = check_same_player_sgp_rule(legs)
        if violation:
            ticket["ticket_integrity_status"] = "FAIL"
            ticket["failure_reason"] = violation
            return ticket

    # Joint probability
    jp_result = compute_joint_probability(legs)
    ticket.update(jp_result)

    if jp_result.get("joint_prob_integrity_status") != "PASS":
        ticket["ticket_integrity_status"] = "FAIL"
        ticket["failure_reason"] = jp_result.get("failure_reason", "JOINT_PROB_FAILED")
        return ticket

    # Ticket pricing
    price_result = compute_ticket_price(legs, family_id)
    ticket.update(price_result)

    if price_result.get("ticket_price_integrity_state") == "INVALID":
        ticket["ticket_integrity_status"] = "FAIL"
        ticket["failure_reason"] = price_result.get("failure_reason", "PRICE_INVALID")
        return ticket

    # Ticket EV
    ticket["ticket_ev_pct"] = compute_ticket_ev(
        ticket.get("joint_prob"), ticket.get("fd_ticket_price_decimal")
    )

    # Scoring
    ticket["ticket_score"] = _compute_ticket_score(ticket, legs)
    ticket["corridor_fit_class"] = _classify_corridor_fit(ticket)

    ticket["ticket_integrity_status"] = "PASS"

    # Freeze
    ticket = freeze_object_with_hash(ticket, "phase3_frozen_hash")
    return ticket


def _has_same_game(legs: list[dict]) -> bool:
    game_ids = [leg.get("game_id") for leg in legs if leg.get("game_id")]
    return len(game_ids) != len(set(game_ids))


def _has_same_player(legs: list[dict]) -> bool:
    player_ids = [leg.get("player_id") for leg in legs if leg.get("player_id")]
    return len(player_ids) != len(set(player_ids))


def _compute_ticket_score(ticket: dict, legs: list[dict]) -> float:
    """
    Compute composite ticket score from EV, joint prob, and leg quality.
    Higher is better.
    """
    ev = ticket.get("ticket_ev_pct") or 0.0
    jp = ticket.get("joint_prob") or 0.0
    n = len(legs)

    # Weighted composite: EV is primary, joint prob is secondary
    # Penalize very long tickets (>4 legs) slightly
    n_penalty = 1.0 - max(0, (n - 4) * 0.05)
    return float((ev * 0.6 + jp * 0.4) * n_penalty)


def _classify_corridor_fit(ticket: dict) -> str:
    """Classify ticket into a corridor fit class based on EV and joint prob."""
    ev = ticket.get("ticket_ev_pct") or 0.0
    jp = ticket.get("joint_prob") or 0.0

    if ev >= 0.10 and jp >= 0.40:
        return "PREMIUM"
    if ev >= 0.05 and jp >= 0.25:
        return "STANDARD"
    if ev >= 0.01:
        return "MARGINAL"
    return "BELOW_THRESHOLD"


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def build_portfolio(
    approved_legs: list[dict],
    target_tickets: int = 21,
    max_legs_per_ticket: int = 5,
) -> list[dict]:
    """
    Build portfolio of up to target_tickets approved tickets from approved legs.
    Section 3.1, 5.2.
    """
    # Filter Phase 3 eligible legs
    eligible_legs = []
    for leg in approved_legs:
        ok, _ = check_leg_phase3_eligible(leg)
        if ok:
            eligible_legs.append(leg)

    tickets: list[dict] = []

    # Generate single-leg tickets first (guaranteed feasible)
    for leg in eligible_legs:
        if len(tickets) >= target_tickets:
            break
        t = build_ticket([leg], _select_family([leg]))
        if t.get("ticket_integrity_status") == "PASS":
            tickets.append(t)

    # Generate 2-leg combinations from cross-game, different-player legs
    if len(tickets) < target_tickets:
        two_leg_combos = list(itertools.combinations(eligible_legs, 2))
        for combo in two_leg_combos:
            if len(tickets) >= target_tickets:
                break
            legs = list(combo)
            family = _select_family(legs)
            t = build_ticket(legs, family)
            if t.get("ticket_integrity_status") == "PASS":
                tickets.append(t)

    # Generate 3-leg combinations
    if len(tickets) < target_tickets:
        three_leg_combos = list(itertools.combinations(eligible_legs, 3))
        for combo in three_leg_combos:
            if len(tickets) >= target_tickets:
                break
            legs = list(combo)
            family = _select_family(legs)
            t = build_ticket(legs, family)
            if t.get("ticket_integrity_status") == "PASS":
                tickets.append(t)

    # Sort by ticket_score descending
    tickets.sort(key=lambda t: t.get("ticket_score", 0), reverse=True)

    return tickets[:target_tickets]


def _select_family(legs: list[dict]) -> str:
    """Select appropriate ticket family based on legs composition."""
    n = len(legs)
    has_same_player = _has_same_player(legs)

    if has_same_player:
        if n <= 2:
            return "SGP_SHORT"
        if n <= 4:
            return "SGP_MID"
        return "SGP_LONG"
    else:
        if n <= 2:
            return "STANDARD_SHORT"
        if n <= 4:
            return "STANDARD_MID"
        return "STANDARD_LONG"
