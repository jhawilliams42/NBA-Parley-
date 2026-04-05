"""
Phase 2 — Full Pipeline: Valuation, Gates, Classification, Leg Approval
Sections 2.1, 2.2, 2.13, 2.14, 2.15
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from ..phase0.constants import GATE_C_BRANCH3_MAX_PER_PORTFOLIO
from ..phase0.governance import emit_circuit_breaker
from ..phase0.hash_utils import freeze_object_with_hash, verify_hash
from .edge import (
    compute_edge,
    compute_execution_price_and_break_even,
    compute_model_prob,
    compute_predictive_lower_bound,
)
from .gates import evaluate_gate_a, evaluate_gate_b, evaluate_gate_c, evaluate_gate_d
from .valuation import build_fair_probability_consensus, process_valuation_book

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2.2 Entry requirements check
# ---------------------------------------------------------------------------

def check_phase2_entry_requirements(obj: dict) -> tuple[bool, list[str]]:
    """
    Verify Phase 2 entry requirements. Section 2.2.
    Returns (ok, list_of_failures).
    """
    failures: list[str] = []

    if not obj.get("phase1_frozen_hash"):
        failures.append("MISSING_PHASE1_HASH")

    if obj.get("phase1_integrity_status") != "PASS":
        failures.append(f"PHASE1_INTEGRITY_FAIL: {obj.get('phase1_integrity_status')}")

    normalized = obj.get("normalized_status", "")
    from ..phase0.constants import BLOCKING_STATUSES
    if normalized in BLOCKING_STATUSES or normalized not in (
        "ACTIVE", "GTD", "QUESTIONABLE", "ACTIVE_PENDING_VERIFICATION"
    ):
        failures.append(f"BLOCKING_STATUS: {normalized}")

    if obj.get("fd_current_line") is None:
        failures.append("MISSING_FD_CURRENT_LINE")

    execution_side = obj.get("fd_execution_side", "OVER")
    if execution_side == "OVER":
        if obj.get("fd_current_odds_american_over") is None:
            failures.append("MISSING_FD_EXECUTION_ODDS")
    else:
        if obj.get("fd_current_odds_american_under") is None:
            failures.append("MISSING_FD_EXECUTION_ODDS")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# 2.13 Bucket assignment
# ---------------------------------------------------------------------------

def assign_bucket(
    gate_c_result: str,
    current_edge_pct: Optional[float],
    model_prob: Optional[float],
    minutes_fragility_class: Optional[str],
    repeatability_class: Optional[str],
) -> str:
    """
    Assign governance bucket from gate results and metrics. Section 2.13.
    """
    if gate_c_result == "PROVISIONAL_PASS_PENDING_MANUAL_REVIEW":
        return "GOVERNANCE_LIMITED"

    if current_edge_pct is None or model_prob is None:
        return "INELIGIBLE"

    edge = current_edge_pct
    prob = model_prob
    fragility = minutes_fragility_class or "UNKNOWN"
    repeat = repeatability_class or "UNKNOWN"

    if edge >= 0.12 and prob >= 0.65 and fragility == "LOW" and repeat == "HIGH":
        return "STAR"
    if edge >= 0.08 and prob >= 0.60 and fragility in ("LOW", "MODERATE") and repeat in (
        "HIGH", "MODERATE"
    ):
        return "ELITE"
    if edge >= 0.06 and prob >= 0.57:
        return "CATALYST"
    if edge >= 0.03 and prob >= 0.52:
        return "SOLID"
    return "INELIGIBLE"


# ---------------------------------------------------------------------------
# Phase 2 single-object processing
# ---------------------------------------------------------------------------

def process_leg_phase2(
    obj: dict,
    valuation_books_data: Optional[list[dict]] = None,
    now_utc: Optional[datetime] = None,
) -> dict:
    """
    Run full Phase 2 processing for a single player-game object.
    Returns the processed object with gate outcomes and leg_approval_status.
    """
    if now_utc is None:
        now_utc = datetime.now(tz=timezone.utc)

    # 2.2 Entry check
    entry_ok, entry_failures = check_phase2_entry_requirements(obj)
    if not entry_ok:
        obj["leg_approval_status"] = "REJECTED"
        obj["rejection_reason"] = entry_failures
        obj["rejection_phase"] = "PHASE2_ENTRY"
        return obj

    # Verify Phase 1 hash
    hash_result = verify_hash(obj, "phase1_frozen_hash", phase=1)
    if not hash_result["valid"]:
        obj["leg_approval_status"] = "REJECTED"
        obj["rejection_reason"] = [hash_result["error"]]
        obj["rejection_phase"] = "HASH_VERIFICATION"
        emit_circuit_breaker("HASH_MISMATCH", {"player_id": obj.get("player_id")})
        return obj

    # Determine execution odds based on side
    execution_side = obj.get("fd_execution_side", "OVER")
    if execution_side == "OVER":
        fd_execution_odds_american = obj.get("fd_current_odds_american_over")
    else:
        fd_execution_odds_american = obj.get("fd_current_odds_american_under")

    obj["fd_execution_odds_american"] = fd_execution_odds_american

    # 2.7 Execution price and break-even
    price_result = compute_execution_price_and_break_even(fd_execution_odds_american)
    obj.update(price_result)

    # 2.5 Valuation / de-vig
    if valuation_books_data:
        consensus = build_fair_probability_consensus(valuation_books_data)
    else:
        consensus = {
            "val_fair_p": None,
            "val_book_count": 0,
            "valuation_integrity_state": "NO_VALID_BOOKS",
            "val_disagreement_score": None,
            "valuation_books": [],
        }
    obj.update(consensus)

    # 2.8 Shrinkage → model_prob
    shrink_result = compute_model_prob(
        raw_event_prob=obj.get("raw_event_prob_over_current_line"),
        val_fair_p=obj.get("val_fair_p"),
        repeatability_class=obj.get("repeatability_class"),
        sample_n=obj.get("sample_n", 0),
    )
    obj.update(shrink_result)

    # 2.9 Edge
    edge_result = compute_edge(obj.get("model_prob"), obj.get("break_even"))
    obj.update(edge_result)

    # 2.10 Predictive lower bound
    plb_result = compute_predictive_lower_bound(obj)
    obj.update(plb_result)

    # 2.12 Gate A
    gate_a, gate_a_reasons = evaluate_gate_a(
        obj, hash_verified=hash_result["valid"]
    )
    obj["gate_a_result"] = gate_a
    obj["gate_a_reasons"] = gate_a_reasons

    if gate_a == "FAIL":
        obj["leg_approval_status"] = "REJECTED"
        obj["rejection_reason"] = gate_a_reasons
        obj["rejection_phase"] = "GATE_A"
        return obj

    # 2.12 Gate B
    gate_b, gate_b_reasons = evaluate_gate_b(obj)
    obj["gate_b_result"] = gate_b
    obj["gate_b_reasons"] = gate_b_reasons

    if gate_b == "FAIL":
        obj["leg_approval_status"] = "REJECTED"
        obj["rejection_reason"] = gate_b_reasons
        obj["rejection_phase"] = "GATE_B"
        return obj

    # 2.12 Gate C
    gate_c, gate_c_reasons = evaluate_gate_c(obj)
    obj["gate_c_result"] = gate_c
    obj["gate_c_reasons"] = gate_c_reasons

    if gate_c == "FAIL":
        obj["leg_approval_status"] = "REJECTED"
        obj["rejection_reason"] = gate_c_reasons
        obj["rejection_phase"] = "GATE_C"
        return obj

    # Branch 3 provisional pass annotations
    if gate_c == "PROVISIONAL_PASS_PENDING_MANUAL_REVIEW":
        obj["no_valuation_flag"] = True
        obj["sgp_eligible"] = False  # Cannot enter SGP tickets

    # 2.12 Gate D
    gate_d, gate_d_reasons = evaluate_gate_d(obj, now_utc=now_utc)
    obj["gate_d_result"] = gate_d
    obj["gate_d_reasons"] = gate_d_reasons

    if gate_d == "FAIL":
        obj["leg_approval_status"] = "REJECTED"
        obj["rejection_reason"] = gate_d_reasons
        obj["rejection_phase"] = "GATE_D"
        return obj

    # 2.13 Bucket
    bucket = assign_bucket(
        gate_c_result=gate_c,
        current_edge_pct=obj.get("current_edge_pct"),
        model_prob=obj.get("model_prob"),
        minutes_fragility_class=obj.get("minutes_fragility_class"),
        repeatability_class=obj.get("repeatability_class"),
    )
    obj["bucket"] = bucket

    if bucket == "INELIGIBLE":
        obj["leg_approval_status"] = "REJECTED"
        obj["rejection_reason"] = ["BUCKET_INELIGIBLE"]
        obj["rejection_phase"] = "BUCKET_CLASSIFICATION"
        return obj

    # 2.14 Reuse freeze
    obj.setdefault("sgp_eligibility_class", "ELIGIBLE" if not obj.get("no_valuation_flag") else "INELIGIBLE")
    obj.setdefault("short_family_reuse_cap", 1)
    obj.setdefault("long_family_reuse_cap", 2)

    # Approved
    obj["leg_approval_status"] = "APPROVED"

    # Phase 2 freeze
    obj = freeze_object_with_hash(obj, "phase2_frozen_hash")
    return obj


# ---------------------------------------------------------------------------
# Phase 2 batch pipeline
# ---------------------------------------------------------------------------

def run_phase2_pipeline(
    player_game_objects: list[dict],
    valuation_books_map: Optional[dict[str, list[dict]]] = None,
    now_utc: Optional[datetime] = None,
    max_governance_limited: int = GATE_C_BRANCH3_MAX_PER_PORTFOLIO,
) -> tuple[list[dict], list[dict]]:
    """
    Run Phase 2 for all objects. Returns (approved_legs, rejected_legs).
    Enforces max 2 GOVERNANCE_LIMITED legs per portfolio. Section 2.12 Branch 3.
    """
    if valuation_books_map is None:
        valuation_books_map = {}

    approved: list[dict] = []
    rejected: list[dict] = []
    governance_limited_count = 0

    for obj in player_game_objects:
        player_id = obj.get("player_id", "")
        books_data = valuation_books_map.get(player_id) or valuation_books_map.get("default")

        processed = process_leg_phase2(obj, books_data, now_utc=now_utc)

        if processed.get("leg_approval_status") == "APPROVED":
            if processed.get("bucket") == "GOVERNANCE_LIMITED":
                if governance_limited_count >= max_governance_limited:
                    processed["leg_approval_status"] = "REJECTED"
                    processed["rejection_reason"] = ["GOVERNANCE_LIMITED_PORTFOLIO_CAP_EXCEEDED"]
                    processed["rejection_phase"] = "PORTFOLIO_CAP"
                    rejected.append(processed)
                    continue
                governance_limited_count += 1
            approved.append(processed)
        else:
            rejected.append(processed)

    return approved, rejected
