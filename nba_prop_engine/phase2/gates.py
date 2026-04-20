"""
Phase 2 — Hard Gates A, B, C, D
Section 2.12
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from ..phase0.constants import (
    GATE_B_MODEL_PROB_FLOOR,
    GATE_C_BRANCH1_EDGE_MIN,
    GATE_C_BRANCH2_BOOK_COUNT_MIN,
    GATE_C_BRANCH2_DISAGREEMENT_MAX,
    GATE_C_BRANCH2_EDGE_MIN,
    GATE_C_BRANCH2_MODEL_PROB_MIN,
    GATE_C_BRANCH3_EDGE_MIN,
    GATE_C_BRANCH3_MODEL_PROB_MIN,
    GATE_C_BRANCH3_SAMPLE_N_MIN,
    GATE_D_GTD_PLAY_RATE_MIN,
    GATE_D_GTD_TIP_SECONDS_THRESHOLD,
    PHASE2_ALLOWED_STATUSES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate A — Source and Data Quality
# ---------------------------------------------------------------------------

def evaluate_gate_a(leg: dict, hash_verified: bool) -> tuple[str, list[str]]:
    """
    Gate A: Source and data quality. Section 2.12.
    Returns ('PASS' | 'FAIL', list_of_failure_reasons).
    """
    failures: list[str] = []

    if not hash_verified:
        failures.append("PHASE1_HASH_NOT_VERIFIED")

    raw_prob = leg.get("raw_event_prob_over_current_line")
    if raw_prob is None:
        failures.append("MISSING_RAW_EVENT_PROB")

    raw_integrity = leg.get("raw_event_prob_integrity_state")
    if raw_integrity not in ("VERIFIED", "DERIVED_BY_APPROVED_RULE"):
        failures.append(f"INVALID_RAW_PROB_INTEGRITY: {raw_integrity}")

    if leg.get("fd_current_line") is None:
        failures.append("MISSING_FD_CURRENT_LINE")

    fd_odds = leg.get("fd_execution_odds_american")
    if fd_odds is None or fd_odds == 0:
        failures.append("MISSING_OR_ZERO_FD_EXECUTION_ODDS")

    normalized = leg.get("normalized_status", "")
    if normalized not in PHASE2_ALLOWED_STATUSES:
        failures.append(f"BLOCKING_NORMALIZED_STATUS: {normalized}")

    if leg.get("phase1_integrity_status") != "PASS":
        failures.append(
            f"PHASE1_CIRCUIT_BREAKER: {leg.get('phase1_integrity_status')}"
        )

    # Check snapshot bundle valid
    if leg.get("snapshot_bundle_status") not in ("VALID", None):
        failures.append(f"SNAPSHOT_BUNDLE_STATUS: {leg.get('snapshot_bundle_status')}")

    if failures:
        return "FAIL", failures
    return "PASS", []


# ---------------------------------------------------------------------------
# Gate B — Probability Floor
# ---------------------------------------------------------------------------

def evaluate_gate_b(leg: dict) -> tuple[str, list[str]]:
    """
    Gate B: model_prob >= 0.50. Section 2.12.
    """
    model_prob = leg.get("model_prob")
    if model_prob is None:
        return "FAIL", ["MODEL_PROB_MISSING"]
    if model_prob < GATE_B_MODEL_PROB_FLOOR:
        return "FAIL", [f"MODEL_PROB_BELOW_FLOOR: {model_prob:.4f} < {GATE_B_MODEL_PROB_FLOOR}"]
    return "PASS", []


# ---------------------------------------------------------------------------
# Gate C — Edge and Predictive Bound with Valuation Mandate
# ---------------------------------------------------------------------------

def evaluate_gate_c(leg: dict) -> tuple[str, list[str]]:
    """
    Gate C with three branches. Section 2.12.
    Returns ('PASS' | 'PROVISIONAL_PASS_PENDING_MANUAL_REVIEW' | 'FAIL', reasons).
    """
    current_edge_pct = leg.get("current_edge_pct")
    predictive_lb = leg.get("predictive_lower_bound")
    predictive_lb_state = leg.get("predictive_lower_bound_integrity_state")
    val_fair_p = leg.get("val_fair_p")
    val_book_count = leg.get("val_book_count", 0)
    val_disagreement = leg.get("val_disagreement_score")
    val_stale = leg.get("val_stale_flag", False)
    val_inflation = leg.get("val_inflation_flag", False)
    model_prob = leg.get("model_prob")
    break_even = leg.get("break_even")
    sample_n = leg.get("sample_n", 0)

    # Branch 1: Predictive lower bound available
    if predictive_lb is not None and predictive_lb_state == "PASS":
        branch1_pass = (
            predictive_lb >= (break_even or 0)
            and val_fair_p is not None
            and val_book_count >= 2
            and (current_edge_pct or 0) >= GATE_C_BRANCH1_EDGE_MIN
        )
        if branch1_pass:
            return "PASS", []
        reasons = []
        if predictive_lb < (break_even or 0):
            reasons.append("PREDICTIVE_LB_BELOW_BREAK_EVEN")
        if val_fair_p is None:
            reasons.append("VAL_FAIR_P_MISSING")
        if val_book_count < 2:
            reasons.append(f"INSUFFICIENT_BOOK_COUNT: {val_book_count} < 2")
        if (current_edge_pct or 0) < GATE_C_BRANCH1_EDGE_MIN:
            reasons.append(
                f"EDGE_BELOW_THRESHOLD: {current_edge_pct} < {GATE_C_BRANCH1_EDGE_MIN}"
            )
        return "FAIL", reasons

    # Branch 2: Strong valuation consensus
    if val_fair_p is not None:
        disagreement = 1.0 if val_disagreement is None else val_disagreement
        branch2_pass = (
            val_book_count >= GATE_C_BRANCH2_BOOK_COUNT_MIN
            and disagreement < GATE_C_BRANCH2_DISAGREEMENT_MAX
            and not val_stale
            and not val_inflation
            and (current_edge_pct or 0) >= GATE_C_BRANCH2_EDGE_MIN
            and (model_prob or 0) >= GATE_C_BRANCH2_MODEL_PROB_MIN
        )
        if branch2_pass:
            return "PASS", []
        reasons = []
        if val_book_count < GATE_C_BRANCH2_BOOK_COUNT_MIN:
            reasons.append(
                f"INSUFFICIENT_BOOK_COUNT: {val_book_count} < {GATE_C_BRANCH2_BOOK_COUNT_MIN}"
            )
        if disagreement >= GATE_C_BRANCH2_DISAGREEMENT_MAX:
            reasons.append(
                f"HIGH_DISAGREEMENT: {val_disagreement} >= {GATE_C_BRANCH2_DISAGREEMENT_MAX}"
            )
        if val_stale:
            reasons.append("VAL_STALE_FLAG_TRUE")
        if val_inflation:
            reasons.append("VAL_INFLATION_FLAG_TRUE")
        if (current_edge_pct or 0) < GATE_C_BRANCH2_EDGE_MIN:
            reasons.append(
                f"EDGE_BELOW_THRESHOLD: {current_edge_pct} < {GATE_C_BRANCH2_EDGE_MIN}"
            )
        if (model_prob or 0) < GATE_C_BRANCH2_MODEL_PROB_MIN:
            reasons.append(
                f"MODEL_PROB_BELOW_BRANCH2_MIN: {model_prob} < {GATE_C_BRANCH2_MODEL_PROB_MIN}"
            )
        return "FAIL", reasons

    # Branch 3: Emergency bypass (no valuation)
    repeatability = leg.get("repeatability_class")
    fragility = leg.get("minutes_fragility_class")
    functional = leg.get("functional_status_class")
    fd_market_status = leg.get("fd_prop_market_status")

    branch3_pass = (
        repeatability == "HIGH"
        and fragility == "LOW"
        and functional == "CLEAN"
        and sample_n >= GATE_C_BRANCH3_SAMPLE_N_MIN
        and (model_prob or 0) >= GATE_C_BRANCH3_MODEL_PROB_MIN
        and (current_edge_pct or 0) >= GATE_C_BRANCH3_EDGE_MIN
        and fd_market_status == "ACTIVE"
    )
    if branch3_pass:
        return "PROVISIONAL_PASS_PENDING_MANUAL_REVIEW", []

    reasons = []
    if repeatability != "HIGH":
        reasons.append(f"REPEATABILITY_NOT_HIGH: {repeatability}")
    if fragility != "LOW":
        reasons.append(f"FRAGILITY_NOT_LOW: {fragility}")
    if functional != "CLEAN":
        reasons.append(f"FUNCTIONAL_NOT_CLEAN: {functional}")
    if sample_n < GATE_C_BRANCH3_SAMPLE_N_MIN:
        reasons.append(f"SAMPLE_N_TOO_LOW: {sample_n} < {GATE_C_BRANCH3_SAMPLE_N_MIN}")
    if (model_prob or 0) < GATE_C_BRANCH3_MODEL_PROB_MIN:
        reasons.append(
            f"MODEL_PROB_BELOW_BRANCH3_MIN: {model_prob} < {GATE_C_BRANCH3_MODEL_PROB_MIN}"
        )
    if (current_edge_pct or 0) < GATE_C_BRANCH3_EDGE_MIN:
        reasons.append(
            f"EDGE_BELOW_THRESHOLD: {current_edge_pct} < {GATE_C_BRANCH3_EDGE_MIN}"
        )
    if fd_market_status != "ACTIVE":
        reasons.append(f"FD_MARKET_NOT_ACTIVE: {fd_market_status}")
    return "FAIL", reasons


# ---------------------------------------------------------------------------
# Gate D — Governance Hard Block with Exact Boolean Grouping [v15.1]
# ---------------------------------------------------------------------------

def evaluate_gate_d(
    leg: dict,
    now_utc: Optional[datetime] = None,
) -> tuple[str, list[str]]:
    """
    Gate D: Governance hard block. Section 2.12.
    Authoritative implementation per Section 0.23 Prose-Code Semantic Identity.

    Returns ('PASS' | 'FAIL', list_of_failure_reasons).
    """
    if now_utc is None:
        now_utc = datetime.now(tz=timezone.utc)

    failure_reasons: list[str] = []

    # Condition 1: (HIGH or HIGH_UNCERTAINTY fragility) AND non-CLEAN status
    if (
        leg.get("minutes_fragility_class") in ("HIGH", "HIGH_UNCERTAINTY")
        and leg.get("functional_status_class") != "CLEAN"
    ):
        failure_reasons.append("HIGH_FRAGILITY_NON_CLEAN_STATUS")

    # Condition 2: epistemic disqualifier
    if leg.get("epistemic_disqualifier", False):
        failure_reasons.append("EPISTEMIC_UNCERTAINTY_TOO_HIGH")

    # Condition 3: high ramp risk
    if leg.get("ramp_risk_class") == "HIGH":
        failure_reasons.append("HIGH_RAMP_RISK")

    # Condition 4: GTD with known low play rate
    if (
        leg.get("normalized_status") == "GTD"
        and leg.get("gtd_play_rate") is not None
        and leg["gtd_play_rate"] < GATE_D_GTD_PLAY_RATE_MIN
    ):
        failure_reasons.append("GTD_LOW_PLAY_RATE")

    # Condition 5: GTD with unknown play rate close to tip
    tip_time_utc = leg.get("tip_time_utc")
    if tip_time_utc is not None:
        if isinstance(tip_time_utc, str):
            if tip_time_utc.endswith("Z"):
                tip_time_utc = tip_time_utc[:-1] + "+00:00"
            tip_time_utc = datetime.fromisoformat(tip_time_utc)
            if tip_time_utc.tzinfo is None:
                tip_time_utc = tip_time_utc.replace(tzinfo=timezone.utc)

        time_to_tip_seconds = (tip_time_utc - now_utc).total_seconds()
        if (
            leg.get("normalized_status") == "GTD"
            and leg.get("gtd_play_rate") is None
            and time_to_tip_seconds < GATE_D_GTD_TIP_SECONDS_THRESHOLD
        ):
            failure_reasons.append("GTD_PLAY_RATE_MISSING_NEAR_TIP")

    # Condition 6: fragile or unresolvable functional status
    if leg.get("functional_status_class") in ("FRAGILE", "UNRESOLVABLE"):
        failure_reasons.append("FRAGILE_OR_UNRESOLVABLE_STATUS")

    # Condition 7: high restriction severity
    if leg.get("restriction_severity_class") == "HIGH":
        failure_reasons.append("HIGH_RESTRICTION_SEVERITY")

    if failure_reasons:
        return "FAIL", failure_reasons
    return "PASS", []
