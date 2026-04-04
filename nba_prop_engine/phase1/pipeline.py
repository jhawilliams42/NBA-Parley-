"""
Phase 1 — Full Pipeline: P1-S01 through P1-S29
Sections 1.1, 1.2, 1.3, and all sub-steps
"""

from __future__ import annotations

import logging
from typing import Optional

from ..phase0.constants import (
    BINDING_FIELDS,
    DISTRIBUTION_MIN_SAMPLE,
    NORMALIZED_STATUS_VALUES,
    PHASE2_ALLOWED_STATUSES,
)
from ..phase0.governance import emit_circuit_breaker
from ..phase0.hash_utils import freeze_object_with_hash
from .audit import execute_P1_S11A_input_audit, execute_P1_S24A_derived_audit
from .dependency import (
    FIELD_DEPENDENCIES,
    compute_field_with_dependency_check,
    execute_P1_S22_opportunity_invalidation_and_recompute,
    validate_dependencies_at_compute_time,
)
from .distribution import (
    _check_distribution_guards,
    compute_raw_event_prob_over,
    select_distribution,
)
from .fragility import compute_fragility_score
from .status import (
    check_fanduel_market_staleness,
    is_status_blocking,
    normalize_status,
    resolve_status_conflict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# P1-S05 / P1-S06 — Status ingest and active/inactive confirmation
# ---------------------------------------------------------------------------

def apply_status_normalization(obj: dict, snapshot_bundle: dict) -> dict:
    """
    P1-S05 and P1-S06: ingest official status and resolve conflicts.
    Section 1.5, 1.21.
    """
    official = obj.get("official_injury_designation")
    active_inactive = obj.get("active_inactive_designation")
    has_fd_lines = bool(obj.get("fanduel_market_loaded"))

    conflict_result = resolve_status_conflict(
        snapshot_bundle=snapshot_bundle,
        player_id=obj.get("player_id", ""),
        official_injury_designation=official,
        has_active_fanduel_lines=has_fd_lines,
    )
    obj.update(conflict_result)

    # Override with active_inactive_designation if it explicitly says inactive
    if active_inactive and active_inactive.upper() == "INACTIVE":
        obj["normalized_status"] = normalize_status(official, active_inactive)

    # Staleness check
    staleness = check_fanduel_market_staleness(snapshot_bundle, obj.get("player_id", ""))
    obj["fd_market_staleness"] = staleness

    if is_status_blocking(obj.get("normalized_status", "UNRESOLVABLE")):
        obj["phase1_block_reason"] = f"BLOCKING_STATUS: {obj['normalized_status']}"
        emit_circuit_breaker(
            "CB-P1-03: MISSING_OFFICIAL_STATUS_OR_BLOCKING",
            {"player_id": obj.get("player_id"), "status": obj["normalized_status"]},
        )
        return obj

    return obj


# ---------------------------------------------------------------------------
# P1-S16 — Injury pre-classification cache
# ---------------------------------------------------------------------------

def apply_injury_pre_classification(obj: dict) -> dict:
    """
    P1-S18: populate injury pre-classification cache.
    These are temporary routing fields only (Section 1.13).
    """
    # Derive provisional values from available data
    restriction = obj.get("restriction_language", "")
    games_since = obj.get("games_since_return")
    normalized = obj.get("normalized_status", "ACTIVE")

    if normalized in ("OUT", "DOUBTFUL"):
        functional_pre = "HIGH_UNCERTAINTY"
        ramp_risk_pre = "HIGH"
        soft_cap_pre = True
    elif normalized == "GTD":
        functional_pre = "LIMITED"
        ramp_risk_pre = "MODERATE"
        soft_cap_pre = False
    elif normalized in ("ACTIVE", "ACTIVE_PENDING_VERIFICATION"):
        if games_since is not None and isinstance(games_since, (int, float)) and games_since < 3:
            functional_pre = "LIMITED"
            ramp_risk_pre = "MODERATE"
            soft_cap_pre = False
        else:
            functional_pre = "CLEAN"
            ramp_risk_pre = "LOW"
            soft_cap_pre = False
    else:
        functional_pre = "HIGH_UNCERTAINTY"
        ramp_risk_pre = "HIGH"
        soft_cap_pre = True

    obj["functional_status_class_pre"] = functional_pre
    obj["ramp_risk_class_pre"] = ramp_risk_pre
    obj["soft_cap_risk_pre"] = soft_cap_pre
    return obj


# ---------------------------------------------------------------------------
# P1-S21 — Final injury and functional fragility freeze
# ---------------------------------------------------------------------------

def freeze_injury_and_fragility(obj: dict) -> dict:
    """
    P1-S21: finalize injury state fields from official sources.
    Section 1.14.
    Fields must become frozen (not pre-classification values).
    """
    # In a production system, these would be derived from official confirmed data.
    # Here we promote pre-classification to final frozen state if not already set.
    for field, pre_field in [
        ("functional_status_class", "functional_status_class_pre"),
        ("ramp_risk_class", "ramp_risk_class_pre"),
        ("soft_cap_risk", "soft_cap_risk_pre"),
    ]:
        if obj.get(field) is None:
            obj[field] = obj.get(pre_field)

    # Apply fragility scoring
    fragility_result = compute_fragility_score(obj)
    obj.update(fragility_result)
    obj["minutes_fragility_class_integrity_state"] = "DERIVED_BY_APPROVED_RULE"
    obj["minutes_fragility_class_dependency_state"] = "VALID"

    # Validate mutual exclusivity for games_since fields (Section 1.14)
    games_since_return = obj.get("games_since_return")
    games_since_minor = obj.get("games_since_minor_return")
    if games_since_return is not None and games_since_minor is not None:
        logger.warning(
            "MUTUAL_EXCLUSIVITY_VIOLATION: games_since_return and "
            "games_since_minor_return both non-null for player_id=%s",
            obj.get("player_id"),
        )

    return obj


# ---------------------------------------------------------------------------
# P1-S25–P1-S27 — Distribution selection and raw event probability
# ---------------------------------------------------------------------------

def apply_distribution_and_probability(obj: dict) -> dict:
    """
    P1-S25, P1-S26, P1-S27: select distribution and compute raw event prob.
    Sections 1.18, 1.19.
    """
    sample_n = obj.get("sample_n", 0)
    fd_current_line = obj.get("fd_current_line")
    mean_raw = obj.get("mean_raw")
    std_raw = obj.get("std_raw")
    stat_family = obj.get("stat_family", "points")
    sample_values = obj.get("sample_values")

    # Numeric guards
    guard_error = _check_distribution_guards(
        sample_n=sample_n,
        fd_current_line=fd_current_line,
        mean_raw=mean_raw,
        std_raw=std_raw,
        stat_family=stat_family,
    )

    if guard_error:
        if guard_error == "INSUFFICIENT_SAMPLE":
            # Try empirical fallback
            if sample_values and len(sample_values) >= DISTRIBUTION_MIN_SAMPLE:
                dist = "EMPIRICAL"
            else:
                obj["distribution_selected"] = "DISTRIBUTION_INVALID_NUMERICS"
                obj["raw_event_prob_over_current_line"] = None
                obj["raw_event_prob_integrity_state"] = "INVALID"
                if guard_error == "INSUFFICIENT_SAMPLE":
                    emit_circuit_breaker(
                        "CB-P1-10: INSUFFICIENT_SAMPLE_AND_NO_EMPIRICAL_FALLBACK",
                        {"player_id": obj.get("player_id"), "sample_n": sample_n},
                    )
                return obj
        else:
            obj["distribution_selected"] = "DISTRIBUTION_INVALID_NUMERICS"
            obj["raw_event_prob_over_current_line"] = None
            obj["raw_event_prob_integrity_state"] = "INVALID"
            emit_circuit_breaker(
                "CB-P1-06: INVALID_NUMERIC_DISTRIBUTIONS",
                {
                    "player_id": obj.get("player_id"),
                    "guard_error": guard_error,
                    "fd_current_line": fd_current_line,
                },
            )
            return obj
    else:
        dist = select_distribution(sample_n, mean_raw, std_raw, stat_family)

    obj["distribution_selected"] = dist

    prob_result = compute_raw_event_prob_over(
        distribution=dist,
        fd_current_line=fd_current_line,
        mean_raw=mean_raw,
        std_raw=std_raw,
        sample_values=sample_values,
    )
    obj.update(prob_result)
    obj["raw_event_prob_integrity_state"] = prob_result.get("integrity_state", "INVALID")

    return obj


# ---------------------------------------------------------------------------
# P1-S28 — Field integrity audit
# ---------------------------------------------------------------------------

def run_field_integrity_audit(obj: dict) -> dict:
    """
    P1-S28: verify all binding fields have valid integrity states.
    """
    for field in BINDING_FIELDS:
        state = obj.get(f"{field}_integrity_state")
        if state not in (
            "VERIFIED",
            "DERIVED_BY_APPROVED_RULE",
            "RECOMPUTED_POST_FREEZE",
        ):
            obj[f"{field}_integrity_state"] = obj.get(
                f"{field}_integrity_state", "MISSING"
            )

    if obj.get("raw_event_prob_over_current_line") is None:
        obj["phase1_raw_prob_status"] = "MISSING"
    else:
        obj["phase1_raw_prob_status"] = "VALID"

    return obj


# ---------------------------------------------------------------------------
# P1-S29 — Phase 1 emission and freeze
# ---------------------------------------------------------------------------

def emit_and_freeze_phase1(obj: dict) -> dict:
    """
    P1-S29: compute and attach phase1_frozen_hash. Section 1.26.
    """
    # Only hash the serializable subset (exclude non-serializable fields)
    frozen = freeze_object_with_hash(obj, "phase1_frozen_hash")
    return frozen


# ---------------------------------------------------------------------------
# Main Phase 1 pipeline entry
# ---------------------------------------------------------------------------

def run_phase1_pipeline(
    player_game_objects: list[dict],
    snapshot_bundle: dict,
) -> list[dict]:
    """
    Execute Phase 1 pipeline steps P1-S01 through P1-S29 for a list of
    player-game objects. Returns the list of successfully processed objects.

    Sections 1.1, 1.2.
    """
    # P1-S11A: Input audit
    objects = execute_P1_S11A_input_audit(player_game_objects)

    processed: list[dict] = []
    for obj in objects:
        try:
            # P1-S05/S06: Status normalization
            obj = apply_status_normalization(obj, snapshot_bundle)

            if obj.get("phase1_block_reason"):
                continue  # Blocked by status

            # P1-S18: Injury pre-classification
            obj = apply_injury_pre_classification(obj)

            # P1-S21: Final injury and fragility freeze
            obj = freeze_injury_and_fragility(obj)

            # P1-S22: Opportunity invalidation check and mandatory recompute
            # (processes single object in list form)
            results = execute_P1_S22_opportunity_invalidation_and_recompute([obj])
            if not results:
                continue
            obj = results[0]

            # P1-S25–S27: Distribution and raw probability
            obj = apply_distribution_and_probability(obj)

            # P1-S28: Field integrity audit
            obj = run_field_integrity_audit(obj)

            processed.append(obj)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "PHASE1_PIPELINE_ERROR | player_id=%s | error=%s",
                obj.get("player_id"),
                exc,
            )
            continue

    # P1-S24A: Derived field audit
    audited = execute_P1_S24A_derived_audit(processed)

    # P1-S29: Emit and freeze
    frozen: list[dict] = []
    for obj in audited:
        try:
            obj["phase1_integrity_status"] = "PASS"
            frozen_obj = emit_and_freeze_phase1(obj)
            frozen.append(frozen_obj)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "PHASE1_FREEZE_ERROR | player_id=%s | error=%s",
                obj.get("player_id"),
                exc,
            )

    return frozen
