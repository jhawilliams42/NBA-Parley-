"""
Phase 4 — Ticket-Level Kelly Sizing with Numerical Stability
Sections 4.1–4.6
"""

from __future__ import annotations

import logging
from typing import Optional

from ..phase0.constants import EPSILON_B, EPSILON_P_HIGH, EPSILON_P_LOW
from ..phase0.hash_utils import freeze_object_with_hash, verify_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 4.4 Kelly input validation
# ---------------------------------------------------------------------------

def validate_kelly_inputs(ticket: dict) -> tuple[bool, dict]:
    """
    Validate all Kelly sizing inputs with epsilon guards. Section 4.4.

    Returns (is_valid, validation_dict).
    """
    joint_prob = ticket.get("joint_prob")
    joint_prob_status = ticket.get("joint_prob_integrity_status")
    fd_price_decimal = ticket.get("fd_ticket_price_decimal")
    bankroll = ticket.get("bankroll_input")
    bankroll_state = ticket.get("bankroll_integrity_state")

    failures: list[str] = []

    if joint_prob is None:
        failures.append("MISSING_JOINT_PROB")
    elif joint_prob_status != "PASS":
        failures.append(f"INVALID_JOINT_PROB_STATUS: {joint_prob_status}")
    elif joint_prob < EPSILON_P_LOW:
        failures.append(f"JOINT_PROB_BELOW_EPSILON: {joint_prob} < {EPSILON_P_LOW}")
    elif joint_prob > EPSILON_P_HIGH:
        failures.append(f"JOINT_PROB_ABOVE_EPSILON: {joint_prob} > {EPSILON_P_HIGH}")

    if fd_price_decimal is None:
        failures.append("MISSING_FD_TICKET_PRICE_DECIMAL")
        b = None
    else:
        b = fd_price_decimal - 1.0  # Net payout per unit wagered
        if b < EPSILON_B:
            failures.append(f"PAYOUT_BELOW_EPSILON: b={b:.4f} < {EPSILON_B}")

    if bankroll is None or bankroll <= 0:
        failures.append(f"INVALID_BANKROLL: {bankroll}")

    if bankroll_state not in ("VERIFIED", "DERIVED_BY_APPROVED_RULE", None):
        failures.append(f"BANKROLL_INTEGRITY_STATE_INVALID: {bankroll_state}")

    if failures:
        return False, {
            "kelly_eligibility_status": "BLOCKED_INPUT_VALIDATION",
            "failures": failures,
            "b": b,
        }

    return True, {
        "kelly_eligibility_status": "ELIGIBLE",
        "b": b,
    }


# ---------------------------------------------------------------------------
# 4.5 Kelly formula computation
# ---------------------------------------------------------------------------

def compute_kelly_stake(
    ticket: dict,
    kelly_fraction: float = 1 / 8,
) -> tuple[float, dict]:
    """
    Compute Kelly stake with numerical stability guards. Section 4.5.

    Per Section 0.24, this is the authoritative implementation function;
    all binding examples must be generated or verified against it.

    Args:
        ticket: Sized ticket dict with all required fields.
        kelly_fraction: Fractional Kelly multiplier (default 1/8).

    Returns:
        (kelly_stake_dollars, detail_dict)
    """
    is_valid, validation = validate_kelly_inputs(ticket)

    if not is_valid:
        return 0.0, validation

    p = float(ticket["joint_prob"])
    q = 1.0 - p
    b = float(validation["b"])
    bankroll = float(ticket["bankroll_input"])

    # Kelly formula: f* = (b*p - q) / b
    numerator = b * p - q
    f_star_raw = numerator / b
    f_star = max(0.0, f_star_raw)

    # Guard: f* > 1.0 indicates model error
    if f_star > 1.0:
        return 0.0, {
            "kelly_eligibility_status": "BLOCKED_KELLY_EXCEEDS_UNITY",
            "f_star": f_star,
            "failure_reason": f"f_star = {f_star:.4f} > 1.0 indicates model error",
        }

    # Apply fractional Kelly
    kelly_fractional = f_star * kelly_fraction
    kelly_cap = float(ticket.get("kelly_cap_fraction_of_bankroll", 0.02))
    kelly_fractional_capped = min(kelly_fractional, kelly_cap)

    # Convert to dollar amount
    kelly_stake_dollars = bankroll * kelly_fractional_capped

    # Stake floor check
    min_op_dollar = float(ticket.get("minimum_operational_dollar_threshold", 1.0))
    min_op_frac = float(ticket.get("min_operational_fraction", 0.001))

    stake_floor_eligible = (
        f_star > 0
        and (ticket.get("ticket_ev_pct") or 0) > 0
        and kelly_fractional >= min_op_frac
    )

    if kelly_stake_dollars < min_op_dollar and not stake_floor_eligible:
        kelly_stake_dollars = 0.0

    stake_cap_applied = kelly_fractional > kelly_cap

    return kelly_stake_dollars, {
        "kelly_eligibility_status": "ELIGIBLE",
        "f_star": f_star,
        "kelly_fraction": kelly_fraction,
        "kelly_fractional": kelly_fractional,
        "kelly_fractional_capped": kelly_fractional_capped,
        "kelly_stake_dollars": kelly_stake_dollars,
        "stake_cap_applied": stake_cap_applied,
        "stake_floor_eligible": stake_floor_eligible,
        "b": b,
        "p": p,
        "q": q,
        "numerator": numerator,
    }


# ---------------------------------------------------------------------------
# Phase 4 pipeline
# ---------------------------------------------------------------------------

def run_phase4_pipeline(
    approved_tickets: list[dict],
    run_context: dict,
    kelly_fraction: float = 1 / 8,
) -> list[dict]:
    """
    Run Phase 4 Kelly sizing for all approved tickets. Section 4.1–4.6.

    Args:
        approved_tickets: List of phase3-frozen ticket objects.
        run_context: run_context dict with bankroll and Kelly config.
        kelly_fraction: Fractional Kelly multiplier.

    Returns:
        List of sized_ticket_objects with phase4_frozen_hash.
    """
    sized: list[dict] = []

    for ticket in approved_tickets:
        # 4.2 Entry requirements
        if not verify_hash(ticket, "phase3_frozen_hash", phase=3)["valid"]:
            logger.warning(
                "PHASE4_ENTRY_FAIL: invalid phase3_frozen_hash for ticket_id=%s",
                ticket.get("ticket_id"),
            )
            continue

        if ticket.get("joint_prob_integrity_status") != "PASS":
            logger.warning(
                "PHASE4_ENTRY_FAIL: joint_prob_integrity_status != PASS for ticket_id=%s",
                ticket.get("ticket_id"),
            )
            continue

        if not ticket.get("fd_ticket_price_decimal"):
            logger.warning(
                "PHASE4_ENTRY_FAIL: missing fd_ticket_price_decimal for ticket_id=%s",
                ticket.get("ticket_id"),
            )
            continue

        # Inject run_context fields needed for sizing
        sized_ticket = dict(ticket)
        sized_ticket["bankroll_input"] = run_context.get("bankroll_input")
        sized_ticket["bankroll_integrity_state"] = run_context.get("bankroll_integrity_state")
        sized_ticket["kelly_cap_fraction_of_bankroll"] = run_context.get(
            "kelly_cap_fraction_of_bankroll", 0.02
        )
        sized_ticket["min_operational_fraction"] = run_context.get(
            "min_operational_fraction", 0.001
        )
        sized_ticket["minimum_operational_dollar_threshold"] = run_context.get(
            "minimum_operational_dollar_threshold", 1.0
        )

        # 4.5 Kelly computation
        kelly_dollars, kelly_detail = compute_kelly_stake(
            sized_ticket, kelly_fraction=kelly_fraction
        )

        sized_ticket["kelly_stake"] = kelly_dollars
        sized_ticket["f_star"] = kelly_detail.get("f_star")
        sized_ticket["kelly_fraction"] = kelly_detail.get("kelly_fraction")
        sized_ticket["stake_floor_eligibility"] = kelly_detail.get("stake_floor_eligible")
        sized_ticket["stake_cap_applied_flag"] = kelly_detail.get("stake_cap_applied")
        sized_ticket["kelly_eligibility_status"] = kelly_detail.get("kelly_eligibility_status")
        sized_ticket["kelly_detail"] = kelly_detail

        # 4.6 Freeze
        sized_ticket = freeze_object_with_hash(sized_ticket, "phase4_frozen_hash")
        sized.append(sized_ticket)

    return sized
