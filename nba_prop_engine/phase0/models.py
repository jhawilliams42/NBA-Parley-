"""
Phase 0 — Root Object Models: run_context, snapshot_bundle
Sections 0.9, 0.21, 0.22
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .constants import (
    EXECUTION_BOOK,
    MAX_SNAPSHOT_DELTA_SECONDS,
    SCHEMA_VERSION,
)
from .governance import emit_circuit_breaker


# ---------------------------------------------------------------------------
# 0.22 run_context
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """
    run_context root object. Section 0.22.
    All required fields must be provided at construction.
    """
    run_id: str
    schema_version: str
    objective_id: str
    target_date_utc: str  # ISO date string e.g. "2026-03-10"
    execution_book: str
    bankroll_input: float
    bankroll_input_currency: str
    bankroll_input_ts_utc: str
    bankroll_integrity_state: str
    kelly_fraction_config_id: str
    kelly_cap_fraction_of_bankroll: float
    min_operational_fraction: float
    minimum_operational_dollar_threshold: float
    run_created_ts_utc: str
    run_status: str = "INITIALIZED"
    kill_switch: Optional[str] = None
    scope_violation: bool = False

    def __post_init__(self) -> None:
        if self.execution_book != EXECUTION_BOOK:
            raise ValueError(
                f"execution_book must be '{EXECUTION_BOOK}', got '{self.execution_book}'"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be '{SCHEMA_VERSION}', got '{self.schema_version}'"
            )
        if self.bankroll_input <= 0:
            raise ValueError(
                f"bankroll_input must be > 0, got {self.bankroll_input}"
            )
        if not (0 < self.kelly_cap_fraction_of_bankroll <= 1):
            raise ValueError(
                f"kelly_cap_fraction_of_bankroll must be in (0, 1], "
                f"got {self.kelly_cap_fraction_of_bankroll}"
            )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "objective_id": self.objective_id,
            "target_date_utc": self.target_date_utc,
            "execution_book": self.execution_book,
            "bankroll_input": self.bankroll_input,
            "bankroll_input_currency": self.bankroll_input_currency,
            "bankroll_input_ts_utc": self.bankroll_input_ts_utc,
            "bankroll_integrity_state": self.bankroll_integrity_state,
            "kelly_fraction_config_id": self.kelly_fraction_config_id,
            "kelly_cap_fraction_of_bankroll": self.kelly_cap_fraction_of_bankroll,
            "min_operational_fraction": self.min_operational_fraction,
            "minimum_operational_dollar_threshold": self.minimum_operational_dollar_threshold,
            "run_created_ts_utc": self.run_created_ts_utc,
            "run_status": self.run_status,
        }


# ---------------------------------------------------------------------------
# 0.21 snapshot_bundle
# ---------------------------------------------------------------------------

@dataclass
class SnapshotBundle:
    """
    snapshot_bundle root object. Section 0.21.
    Atomicity constraint enforced at construction.
    """
    snapshot_bundle_id: str
    bundle_frozen_ts_utc: str
    nba_status_snapshot_ts_utc: str
    nba_schedule_snapshot_ts_utc: str
    nba_stats_snapshot_ts_utc: str
    team_context_snapshot_ts_utc: str
    referee_snapshot_ts_utc: str
    fanduel_market_snapshot_ts_utc: str
    valuation_market_snapshot_ts_utc: str
    snapshot_sequence_log: list = field(default_factory=list)
    bundle_integrity_hash: Optional[str] = None
    bundle_status: str = "PENDING"
    max_component_time_delta_seconds: float = 0.0

    def _parse_ts(self, ts_str: str) -> datetime:
        """Parse an ISO timestamp string into a timezone-aware datetime."""
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def validate_atomicity(self) -> bool:
        """
        Check that all component timestamps are within MAX_SNAPSHOT_DELTA_SECONDS
        of each other. Section 0.21 Atomicity Constraint.

        Returns True if valid, False if failed (and fires circuit breaker).
        """
        ts_fields = [
            self.nba_status_snapshot_ts_utc,
            self.nba_schedule_snapshot_ts_utc,
            self.nba_stats_snapshot_ts_utc,
            self.team_context_snapshot_ts_utc,
            self.referee_snapshot_ts_utc,
            self.fanduel_market_snapshot_ts_utc,
            self.valuation_market_snapshot_ts_utc,
        ]
        parsed = [self._parse_ts(ts) for ts in ts_fields]
        delta = (max(parsed) - min(parsed)).total_seconds()
        self.max_component_time_delta_seconds = delta

        if delta > MAX_SNAPSHOT_DELTA_SECONDS:
            self.bundle_status = "STALE_SYNCHRONIZATION_FAILURE"
            emit_circuit_breaker(
                "CB-P1-02: UNSYNCHRONIZED_CRITICAL_SOURCE_TIMESTAMPS",
                {
                    "snapshot_bundle_id": self.snapshot_bundle_id,
                    "max_delta_seconds": delta,
                    "threshold_seconds": MAX_SNAPSHOT_DELTA_SECONDS,
                },
            )
            return False

        self.bundle_status = "VALID"
        return True

    def to_hash_input(self) -> dict:
        """Build the canonical dict for bundle_integrity_hash computation."""
        return {
            "snapshot_bundle_id": self.snapshot_bundle_id,
            "bundle_frozen_ts_utc": self.bundle_frozen_ts_utc,
            "nba_status_snapshot_ts_utc": self.nba_status_snapshot_ts_utc,
            "nba_schedule_snapshot_ts_utc": self.nba_schedule_snapshot_ts_utc,
            "nba_stats_snapshot_ts_utc": self.nba_stats_snapshot_ts_utc,
            "team_context_snapshot_ts_utc": self.team_context_snapshot_ts_utc,
            "referee_snapshot_ts_utc": self.referee_snapshot_ts_utc,
            "fanduel_market_snapshot_ts_utc": self.fanduel_market_snapshot_ts_utc,
            "valuation_market_snapshot_ts_utc": self.valuation_market_snapshot_ts_utc,
        }

    def freeze(self) -> None:
        """
        Validate atomicity and compute bundle_integrity_hash.
        Sets bundle_status to VALID or STALE_SYNCHRONIZATION_FAILURE.
        """
        from .hash_utils import compute_hash

        if not self.validate_atomicity():
            return

        self.bundle_integrity_hash = compute_hash(self.to_hash_input())
        self.bundle_status = "VALID"

    def to_dict(self) -> dict:
        return {
            "snapshot_bundle_id": self.snapshot_bundle_id,
            "bundle_frozen_ts_utc": self.bundle_frozen_ts_utc,
            "nba_status_snapshot_ts_utc": self.nba_status_snapshot_ts_utc,
            "nba_schedule_snapshot_ts_utc": self.nba_schedule_snapshot_ts_utc,
            "nba_stats_snapshot_ts_utc": self.nba_stats_snapshot_ts_utc,
            "team_context_snapshot_ts_utc": self.team_context_snapshot_ts_utc,
            "referee_snapshot_ts_utc": self.referee_snapshot_ts_utc,
            "fanduel_market_snapshot_ts_utc": self.fanduel_market_snapshot_ts_utc,
            "valuation_market_snapshot_ts_utc": self.valuation_market_snapshot_ts_utc,
            "bundle_integrity_hash": self.bundle_integrity_hash,
            "bundle_status": self.bundle_status,
            "max_component_time_delta_seconds": self.max_component_time_delta_seconds,
            "snapshot_sequence_log": self.snapshot_sequence_log,
        }
