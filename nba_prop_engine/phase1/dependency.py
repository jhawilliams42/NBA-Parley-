"""
Phase 1 — Dependency Tracking, Null Propagation, and Topological Recomputation
Sections 1.10A, 1.15A, 1.22, 1.22A, 1.24
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable, Optional

from ..phase0.constants import BINDING_FIELDS, NON_BINDING_FIELDS, PROVISIONAL_DERIVED_FIELDS
from ..phase0.governance import emit_circuit_breaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1.10A Field dependency declarations
# ---------------------------------------------------------------------------

FIELD_DEPENDENCIES: dict[str, dict] = {
    "opportunity_context_class": {
        "depends_on": ["usage_rate", "field_goal_attempts", "touches", "drives"],
        "binding": True,
    },
    "minutes_fragility_class": {
        "depends_on": [
            "opportunity_context_class",
            "rotation_volatility_class",
            "role_lock_class",
            "blowout_risk_class",
        ],
        "binding": True,
    },
    "role_lock_class": {
        "depends_on": [
            "starter_rate_last_n",
            "substitution_pattern_stability",
            "coaching_volatility_class",
        ],
        "binding": True,
    },
    "repeatability_class": {
        "depends_on": [
            "minutes_fragility_class",
            "role_lock_class",
            "sample_n",
        ],
        "binding": True,
    },
    "blowout_fragility": {
        "depends_on": ["blowout_risk_class", "spread_context_class"],
        "binding": False,
    },
    "role_fragility": {
        "depends_on": ["role_lock_class", "rotation_volatility_class"],
        "binding": False,
    },
    "rotation_volatility_derived": {
        "depends_on": ["substitution_pattern_stability", "coaching_volatility_class"],
        "binding": False,
    },
    "coaching_volatility_class": {
        "depends_on": ["lineup_continuity_score"],
        "binding": False,
    },
    "lineup_continuity_score": {
        "depends_on": ["lineup_combinations"],
        "binding": False,
    },
}

# Recompute functions registry — callables populated at runtime by phase1 pipeline
FIELD_RECOMPUTE_FUNCTIONS: dict[str, Callable] = {}


def register_recompute_function(field_name: str, func: Callable) -> None:
    """Register a recompute function for a binding field. Section 1.22."""
    FIELD_RECOMPUTE_FUNCTIONS[field_name] = func


# ---------------------------------------------------------------------------
# 1.10A Dependency validation at compute time
# ---------------------------------------------------------------------------

def validate_dependencies_at_compute_time(
    field_name: str, obj: dict
) -> tuple[bool, list[str]]:
    """
    Validate that all declared dependencies of field_name are VALID.
    Returns (is_valid, list_of_invalid_deps). Section 1.10A.
    """
    deps = FIELD_DEPENDENCIES.get(field_name, {}).get("depends_on", [])
    invalid: list[str] = []

    for dep in deps:
        dep_state = obj.get(f"{dep}_dependency_state", "NOT_COMPUTED")
        dep_integrity = obj.get(f"{dep}_integrity_state")

        if dep_state != "VALID":
            invalid.append(dep)
        elif dep_integrity in ("MISSING", "INVALID", "UNRESOLVABLE", "DEPENDENCY_FAILURE"):
            invalid.append(dep)

    if invalid:
        return False, invalid
    return True, []


def mark_dependency_failure(
    obj: dict, field_name: str, invalid_deps: list[str], step_id: str
) -> None:
    """
    Mark a field as dependency failure and log it. Section 1.10A.
    """
    obj[field_name] = None
    obj[f"{field_name}_integrity_state"] = "DEPENDENCY_FAILURE"
    obj[f"{field_name}_dependency_state"] = "FAILED"
    logger.warning(
        "DEPENDENCY_FAILURE | field=%s | step=%s | invalid_deps=%s",
        field_name,
        step_id,
        invalid_deps,
    )


def compute_field_with_dependency_check(
    obj: dict,
    field_name: str,
    compute_func: Callable,
    step_id: str,
) -> None:
    """
    Validate dependencies then compute field, updating state/integrity.
    Section 1.10A.
    """
    is_valid, invalid_deps = validate_dependencies_at_compute_time(field_name, obj)
    if not is_valid:
        mark_dependency_failure(obj, field_name, invalid_deps, step_id)
        return

    obj[f"{field_name}_dependency_state"] = "COMPUTING"
    try:
        value = compute_func(obj)
        obj[field_name] = value
        obj[f"{field_name}_integrity_state"] = "DERIVED_BY_APPROVED_RULE"
        obj[f"{field_name}_dependency_state"] = "VALID"
    except Exception as exc:  # noqa: BLE001
        obj[field_name] = None
        obj[f"{field_name}_integrity_state"] = "DEPENDENCY_FAILURE"
        obj[f"{field_name}_dependency_state"] = "FAILED"
        logger.error(
            "COMPUTE_ERROR | field=%s | step=%s | error=%s", field_name, step_id, exc
        )


# ---------------------------------------------------------------------------
# 1.22 Topological recomputation infrastructure
# ---------------------------------------------------------------------------

def build_reverse_dependency_graph(
    field_dependencies: dict,
) -> dict[str, set[str]]:
    """Build reverse adjacency map for dependency graph. Section 1.22."""
    reverse: dict[str, set[str]] = {}
    for field, config in field_dependencies.items():
        for dep in config.get("depends_on", []):
            reverse.setdefault(dep, set()).add(field)
    return reverse


def collect_transitive_dependents(
    start_fields: list[str],
    reverse_graph: dict[str, set[str]],
) -> set[str]:
    """BFS to find all transitive dependents of start_fields. Section 1.22."""
    affected = set(start_fields)
    queue: deque[str] = deque(start_fields)

    while queue:
        current = queue.popleft()
        for dependent in reverse_graph.get(current, set()):
            if dependent not in affected:
                affected.add(dependent)
                queue.append(dependent)

    return affected


def topological_sort_subset(
    fields_subset: set[str],
    field_dependencies: dict,
) -> list[str]:
    """
    Kahn's algorithm topological sort on subset of the dependency graph.
    Raises ValueError if cycle detected. Section 1.22.
    """
    in_degree: dict[str, int] = {f: 0 for f in fields_subset}

    for field in fields_subset:
        for dep in field_dependencies.get(field, {}).get("depends_on", []):
            if dep in fields_subset:
                in_degree[field] += 1

    ordered: list[str] = []
    queue: deque[str] = deque(
        [f for f, deg in in_degree.items() if deg == 0]
    )

    while queue:
        node = queue.popleft()
        ordered.append(node)
        for candidate in fields_subset:
            if node in field_dependencies.get(candidate, {}).get("depends_on", []):
                in_degree[candidate] -= 1
                if in_degree[candidate] == 0:
                    queue.append(candidate)

    if len(ordered) != len(fields_subset):
        raise ValueError("Dependency cycle detected in recompute graph")

    return ordered


# ---------------------------------------------------------------------------
# 1.22 State change detection
# ---------------------------------------------------------------------------

def detect_injury_state_change(obj: dict) -> tuple[bool, list[dict]]:
    """
    Detect if final frozen injury state differs from provisional cache.
    Section 1.22.
    """
    state_changed = False
    changes: list[dict] = []

    pairs = [
        ("functional_status_class_pre", "functional_status_class"),
        ("ramp_risk_class_pre", "ramp_risk_class"),
        ("soft_cap_risk_pre", "soft_cap_risk"),
    ]
    for field_pre, field_final in pairs:
        pre_val = obj.get(field_pre)
        final_val = obj.get(field_final)
        if pre_val != final_val:
            state_changed = True
            changes.append(
                {"field": field_final, "pre": pre_val, "frozen": final_val}
            )

    return state_changed, changes


def identify_primary_affected_fields(
    obj: dict, changes: list[dict]
) -> list[str]:
    """
    Identify provisionally-derived fields directly affected by injury state
    changes. Section 1.22.
    """
    changed_fields = {c["field"] for c in changes}
    affected: list[str] = []

    for field_name in PROVISIONAL_DERIVED_FIELDS:
        metadata = obj.get(f"{field_name}_metadata", {})
        provisional_deps = {
            dep.replace("_pre", "")
            for dep in metadata.get("provisional_dependencies", [])
        }
        if metadata.get("derived_using_provisional_state") and provisional_deps & changed_fields:
            affected.append(field_name)

    return affected


# ---------------------------------------------------------------------------
# 1.22A Recomputation failure handling and routing
# ---------------------------------------------------------------------------

def classify_and_handle_recompute_failure(
    obj: dict, field_name: str, error: Exception
) -> str:
    """
    Tier-based routing for recompute failures. Section 1.22A.
    Returns 'CONTINUE' or 'HALT'.
    """
    provisional_value = obj.get(f"{field_name}_metadata", {}).get("provisional_value")

    if field_name not in BINDING_FIELDS:
        obj[field_name] = None
        obj[f"{field_name}_integrity_state"] = "RECOMPUTE_FAILED_NON_BINDING"
        logger.warning(
            "RECOMPUTE_FAIL_NON_BINDING | tier=TIER_1 | field=%s | error=%s",
            field_name,
            error,
        )
        return "CONTINUE"

    if provisional_value is not None:
        obj["object_integrity_status"] = "RECOMPUTE_FAILURE_BINDING"
        obj["phase1_integrity_status"] = "FAIL"
        emit_circuit_breaker(
            "CB-P1-22: BINDING_FIELD_RECOMPUTE_FAILED",
            {
                "field_name": field_name,
                "provisional_value": provisional_value,
                "error": str(error),
                "action": "OBJECT_HALTED",
            },
        )
        logger.error(
            "RECOMPUTE_FAIL_BINDING | tier=TIER_2 | field=%s | prov=%s | error=%s",
            field_name,
            provisional_value,
            error,
        )
        return "HALT"

    obj["object_integrity_status"] = "UNRECOVERABLE_RECOMPUTE_FAILURE"
    obj["phase1_integrity_status"] = "FAIL"
    emit_circuit_breaker(
        "CB-P1-22: UNRECOVERABLE_BINDING_FIELD_FAILURE",
        {
            "field_name": field_name,
            "error": str(error),
            "action": "OBJECT_HALTED",
        },
    )
    logger.error(
        "RECOMPUTE_FAIL_BINDING_NO_PROV | tier=TIER_3 | field=%s | error=%s",
        field_name,
        error,
    )
    return "HALT"


def recompute_field_with_tier_routing(obj: dict, field_name: str) -> str:
    """
    Execute recompute for a single field using registered function.
    Section 1.22.
    """
    try:
        recompute_func = FIELD_RECOMPUTE_FUNCTIONS.get(field_name)
        if recompute_func is None:
            raise KeyError(f"No recompute function registered for '{field_name}'")

        new_value = recompute_func(obj, use_frozen_state=True)
        obj[field_name] = new_value
        obj[f"{field_name}_integrity_state"] = "RECOMPUTED_POST_FREEZE"
        return "CONTINUE"
    except Exception as exc:  # noqa: BLE001
        return classify_and_handle_recompute_failure(obj, field_name, exc)


# ---------------------------------------------------------------------------
# 1.22 Full opportunity invalidation and recompute execution
# ---------------------------------------------------------------------------

def execute_P1_S22_opportunity_invalidation_and_recompute(
    player_game_objects: list[dict],
) -> list[dict]:
    """
    Full execution of P1-S22 step. Section 1.22 / 1.24.
    """
    reverse_graph = build_reverse_dependency_graph(FIELD_DEPENDENCIES)
    passed: list[dict] = []

    for obj in player_game_objects:
        state_changed, changes = detect_injury_state_change(obj)

        if not state_changed:
            obj["injury_state_consistency"] = "PROVISIONAL_MATCHES_FROZEN"
            passed.append(obj)
            continue

        obj["injury_state_consistency"] = "PROVISIONAL_DIFFERS_FROM_FROZEN"
        obj["injury_state_changes"] = changes

        primary = identify_primary_affected_fields(obj, changes)
        full_affected = collect_transitive_dependents(primary, reverse_graph)

        try:
            recompute_order = topological_sort_subset(full_affected, FIELD_DEPENDENCIES)
        except ValueError as exc:
            obj["phase1_status"] = "HALTED_DEPENDENCY_CYCLE"
            emit_circuit_breaker(
                "CB-P1-22: DEPENDENCY_CYCLE",
                {"player_id": obj.get("player_id"), "error": str(exc)},
            )
            continue

        obj["recompute_order"] = recompute_order
        obj["recompute_failures"] = []

        halted = False
        for field_name in recompute_order:
            action = recompute_field_with_tier_routing(obj, field_name)
            if action == "HALT":
                obj["phase1_status"] = "HALTED_RECOMPUTE_FAILURE"
                halted = True
                break

        if not halted:
            obj["object_integrity_status"] = "RECOMPUTE_SUCCESS"
            passed.append(obj)

    return passed
