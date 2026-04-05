"""
Phase 1 — Audit Steps: P1-S11A and P1-S24A
Sections 1.22 (P1-S11A) and 1.23 (P1-S24A) of v15.1
"""

from __future__ import annotations

import logging

from ..phase0.governance import emit_circuit_breaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# P1-S11A: Input Dependency Audit — Raw Source Integrity Only
# ---------------------------------------------------------------------------

RAW_INPUT_REQUIREMENTS = [
    "lineup_context_loaded",
    "injury_status_loaded",
    "player_stats_loaded",
    "fanduel_market_loaded",
]


def execute_P1_S11A_input_audit(player_game_objects: list[dict]) -> list[dict]:
    """
    Validate raw external inputs before derivation begins.
    Audits only source-level dependencies, never derived fields.
    Section 1.22 (P1-S11A in sequence) / v15.1 Section 1.22.

    Args:
        player_game_objects: List of player-game objects post-load steps.

    Returns:
        List of objects that passed the audit.
    """
    audited: list[dict] = []

    for obj in player_game_objects:
        missing = [f for f in RAW_INPUT_REQUIREMENTS if not obj.get(f, False)]

        if missing:
            obj["phase1_integrity_status"] = "INPUT_DEPENDENCY_FAILURE"
            emit_circuit_breaker(
                "CB-P1-11A: RAW_INPUT_MISSING",
                {
                    "player_id": obj.get("player_id"),
                    "missing_inputs": missing,
                },
            )
            logger.error(
                "P1-S11A FAIL | player_id=%s | missing=%s",
                obj.get("player_id"),
                missing,
            )
            continue

        audited.append(obj)

    return audited


# ---------------------------------------------------------------------------
# P1-S24A: Derived Field Integrity Audit — Post-Derivation Binding Check
# ---------------------------------------------------------------------------

BINDING_DERIVED_FIELDS = [
    "opportunity_context_class",
    "minutes_fragility_class",
    "role_lock_class",
    "repeatability_class",
    "functional_status_class",
]

VALID_POST_DERIVATION_STATES = frozenset({
    "VERIFIED",
    "DERIVED_BY_APPROVED_RULE",
    "RECOMPUTED_POST_FREEZE",
})


def execute_P1_S24A_derived_audit(player_game_objects: list[dict]) -> list[dict]:
    """
    After all Phase 1 derivations complete, audit that all binding derived
    fields are either validly computed or explicitly failed.
    Section 1.23 (P1-S24A).

    Args:
        player_game_objects: List of player-game objects post-derivation.

    Returns:
        List of objects that passed the audit.
    """
    passed: list[dict] = []

    for obj in player_game_objects:
        failures: list[dict] = []

        for field in BINDING_DERIVED_FIELDS:
            integrity = obj.get(f"{field}_integrity_state")
            value = obj.get(field)

            if integrity not in VALID_POST_DERIVATION_STATES:
                failures.append({"field": field, "integrity_state": integrity})

            if value is None:
                failures.append({"field": field, "reason": "NULL_BINDING_VALUE"})

        if failures:
            obj["phase1_integrity_status"] = "DERIVED_FIELD_AUDIT_FAILURE"
            emit_circuit_breaker(
                "CB-P1-24A: BINDING_DERIVATION_FAILURE",
                {
                    "player_id": obj.get("player_id"),
                    "failures": failures,
                },
            )
            logger.error(
                "P1-S24A FAIL | player_id=%s | failures=%s",
                obj.get("player_id"),
                failures,
            )
            continue

        passed.append(obj)

    return passed
