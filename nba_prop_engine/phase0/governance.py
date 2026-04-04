"""
Phase 0 — Governance Rules: Scope Violations, Field Integrity, Kill Switches
Sections 0.5, 0.7, 0.11, 0.20
"""

from __future__ import annotations

import logging
from typing import Any

from .constants import (
    FANDUEL_EXECUTION_FIELDS,
    FIELD_INTEGRITY_STATES,
    KILL_SWITCH_STATES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error emission helpers
# ---------------------------------------------------------------------------

def emit_error(code: str, context: dict | None = None) -> None:
    """Log a structured error emission. Section 0.11."""
    payload = {"error_code": code}
    if context:
        payload.update(context)
    logger.error("EMIT_ERROR: %s | context=%s", code, context or {})


def emit_circuit_breaker(code: str, context: dict | None = None) -> None:
    """Log a circuit-breaker emission. Sections 1.3, 2.12."""
    payload = {"circuit_breaker": code}
    if context:
        payload.update(context)
    logger.error("CIRCUIT_BREAKER: %s | context=%s", code, context or {})


# ---------------------------------------------------------------------------
# 0.7 Field integrity state validation
# ---------------------------------------------------------------------------

def assert_valid_integrity_state(state: str, field_name: str) -> None:
    """Raise ValueError if state is not a recognized integrity state."""
    if state not in FIELD_INTEGRITY_STATES:
        raise ValueError(
            f"Invalid integrity state '{state}' for field '{field_name}'. "
            f"Allowed: {sorted(FIELD_INTEGRITY_STATES)}"
        )


# ---------------------------------------------------------------------------
# 0.16 FanDuel execution field protection
# ---------------------------------------------------------------------------

def check_fanduel_field_contamination(
    obj: dict[str, Any], source_name: str
) -> list[str]:
    """
    Return list of FanDuel execution fields that were populated from a
    non-FanDuel source. Caller should set field_integrity_state = INVALID
    and fire kill switch if any violations found.
    Section 0.16.
    """
    if source_name.lower() in ("fanduel", "fd", "fanduel_sportsbook"):
        return []

    violations = []
    for field in FANDUEL_EXECUTION_FIELDS:
        if obj.get(field) is not None:
            violations.append(field)
    return violations


# ---------------------------------------------------------------------------
# 0.11 Scope violation detection
# ---------------------------------------------------------------------------

class ScopeViolation(RuntimeError):
    """Raised when a scope violation is detected. Section 0.11."""

    def __init__(self, reason: str, obj_id: str | None = None) -> None:
        self.reason = reason
        self.obj_id = obj_id
        super().__init__(f"SCOPE_VIOLATION | obj={obj_id} | {reason}")


def check_and_raise_scope_violation(
    condition: bool, reason: str, obj: dict | None = None
) -> None:
    """If condition is True, set scope_violation=True on obj and raise."""
    if condition:
        obj_id = obj.get("id") if obj else None
        if obj is not None:
            obj["scope_violation"] = True
        emit_error("MANUAL_OVERRIDE_PROHIBITED", {"reason": reason, "obj_id": obj_id})
        raise ScopeViolation(reason, obj_id)


# ---------------------------------------------------------------------------
# 0.20 Kill switch state enforcement
# ---------------------------------------------------------------------------

def fire_kill_switch(run_context: dict, switch: str, context: dict | None = None) -> None:
    """
    Fire a run-level kill switch: halt run, block downstream phases.
    Section 0.20.
    """
    if switch not in KILL_SWITCH_STATES:
        raise ValueError(f"Unknown kill switch '{switch}'")
    run_context["run_status"] = f"HALTED_{switch}"
    run_context["kill_switch"] = switch
    emit_error(switch, context)
    logger.critical("KILL_SWITCH_FIRED: %s | run_id=%s", switch, run_context.get("run_id"))
