"""
Phase 1 — Status Normalization and Timestamp Conflict Resolution
Sections 1.5, 1.21
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..phase0.constants import (
    BLOCKING_STATUSES,
    FUNCTIONAL_STATUS_CLASS_VALUES,
    NORMALIZED_STATUS_VALUES,
    STALENESS_THRESHOLD_SECONDS,
)
from ..phase0.governance import emit_circuit_breaker


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO timestamp string to timezone-aware datetime."""
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def normalize_status(
    official_injury_designation: Optional[str],
    active_inactive_designation: Optional[str],
    restriction_language: Optional[str] = None,
) -> str:
    """
    Derive normalized_status from official sources. Section 1.5.

    Returns one of the allowed normalized_status values.
    If conflict persists or data is missing, returns UNRESOLVABLE.
    """
    # If inactive designation is explicit, cannot be ACTIVE
    if active_inactive_designation and active_inactive_designation.upper() == "INACTIVE":
        if official_injury_designation and official_injury_designation.upper() == "OUT":
            return "OUT"
        return "INACTIVE_OTHER"

    if not official_injury_designation and not active_inactive_designation:
        return "UNRESOLVABLE"

    designation = (official_injury_designation or "").upper().strip()

    mapping = {
        "ACTIVE": "ACTIVE",
        "OUT": "OUT",
        "GTD": "GTD",
        "GAME TIME DECISION": "GTD",
        "QUESTIONABLE": "QUESTIONABLE",
        "DOUBTFUL": "DOUBTFUL",
        "INACTIVE": "INACTIVE_OTHER",
    }

    normalized = mapping.get(designation)
    if normalized:
        return normalized

    # Partial match fallback
    for key, val in mapping.items():
        if key in designation:
            return val

    return "UNRESOLVABLE"


def is_status_blocking(normalized_status: str) -> bool:
    """Return True if this status blocks Phase 1 processing. Section 1.5."""
    return normalized_status in BLOCKING_STATUSES


# ---------------------------------------------------------------------------
# 1.21 P1-S02A: Snapshot Timestamp Precedence and Conflict Resolution
# ---------------------------------------------------------------------------

def resolve_status_conflict(
    snapshot_bundle: dict,
    player_id: str,
    official_injury_designation: Optional[str],
    has_active_fanduel_lines: bool,
) -> dict:
    """
    Apply deterministic precedence rules when timestamp conflicts exist
    between snapshot components. Section 1.21.

    Args:
        snapshot_bundle: Dict with timestamp fields from SnapshotBundle.
        player_id: Player identifier for logging.
        official_injury_designation: Raw injury designation from official source.
        has_active_fanduel_lines: Whether FanDuel has active lines for this player.

    Returns:
        Dict with normalized_status, fd_current_line_validity, and metadata.
    """
    fanduel_ts = _parse_ts(snapshot_bundle["fanduel_market_snapshot_ts_utc"])
    nba_status_ts = _parse_ts(snapshot_bundle["nba_status_snapshot_ts_utc"])

    time_delta = (nba_status_ts - fanduel_ts).total_seconds()

    if time_delta > 0:
        # Official status is newer
        if (
            official_injury_designation
            and official_injury_designation.upper() == "OUT"
            and has_active_fanduel_lines
        ):
            return {
                "normalized_status": "OUT",
                "fd_current_line_validity": "STALE_SUPERSEDED",
                "fd_prop_market_status": "DELISTED_POST_SNAPSHOT",
                "conflict_resolution_method": "OFFICIAL_STATUS_PRECEDENCE",
                "time_delta_seconds": time_delta,
            }
        return {
            "normalized_status": normalize_status(official_injury_designation, None),
            "fd_current_line_validity": "VALID_STATUS_NEWER",
            "time_delta_seconds": time_delta,
        }

    if time_delta < 0:
        # FanDuel market is newer
        if has_active_fanduel_lines:
            return {
                "normalized_status": "ACTIVE_PENDING_VERIFICATION",
                "fd_current_line_validity": "VALID_NEWER_THAN_STATUS",
                "fd_market_precedence_flag": True,
                "time_delta_seconds": abs(time_delta),
            }

    # Synchronized
    return {
        "normalized_status": normalize_status(official_injury_designation, None),
        "fd_current_line_validity": "SYNCHRONIZED",
        "time_delta_seconds": 0,
    }


def check_fanduel_market_staleness(
    snapshot_bundle: dict,
    player_id: str,
) -> dict:
    """
    Execution Market Staleness Check. Section 1.21.

    Returns action REQUIRE_MANUAL_VERIFICATION_OR_REJECT or PROCEED.
    """
    fanduel_ts = _parse_ts(snapshot_bundle["fanduel_market_snapshot_ts_utc"])
    nba_status_ts = _parse_ts(snapshot_bundle["nba_status_snapshot_ts_utc"])
    time_delta = (nba_status_ts - fanduel_ts).total_seconds()

    if time_delta > STALENESS_THRESHOLD_SECONDS:
        return {
            "fd_market_staleness_risk": True,
            "fd_market_age_seconds": time_delta,
            "action": "REQUIRE_MANUAL_VERIFICATION_OR_REJECT",
        }
    return {
        "fd_market_staleness_risk": False,
        "fd_market_age_seconds": time_delta,
        "action": "PROCEED",
    }
