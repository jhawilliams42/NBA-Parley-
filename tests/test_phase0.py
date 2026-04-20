"""
Tests for Phase 0 — Hash utils, constants, models, governance.
"""

import math
import pytest

from nba_prop_engine.phase0.constants import (
    DEVIG_METHOD_REGISTRY,
    EXECUTION_BOOK,
    FIELD_INTEGRITY_STATES,
    SCHEMA_VERSION,
)
from nba_prop_engine.phase0.governance import (
    ScopeViolation,
    check_and_raise_scope_violation,
    check_fanduel_field_contamination,
    emit_error,
)
from nba_prop_engine.phase0.hash_utils import (
    compute_hash,
    freeze_object_with_hash,
    verify_hash,
)
from nba_prop_engine.phase0.models import RunContext, SnapshotBundle


# ---------------------------------------------------------------------------
# Hash utils
# ---------------------------------------------------------------------------

class TestComputeHash:
    def test_deterministic_same_input(self):
        obj = {"a": 1, "b": "hello", "c": None}
        assert compute_hash(obj) == compute_hash(obj)

    def test_key_order_independent(self):
        """RFC 8785 mandates lexicographic key ordering — hash must match."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"b": 2, "a": 1}
        assert compute_hash(obj1) == compute_hash(obj2)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            compute_hash({"x": float("nan")})

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="Infinity"):
            compute_hash({"x": float("inf")})

    def test_neg_inf_raises(self):
        with pytest.raises(ValueError, match="Infinity"):
            compute_hash({"x": float("-inf")})

    def test_nested_nan_raises(self):
        with pytest.raises(ValueError):
            compute_hash({"outer": {"inner": float("nan")}})

    def test_returns_hex_string(self):
        digest = compute_hash({"x": 1})
        assert isinstance(digest, str)
        assert len(digest) == 64
        int(digest, 16)  # valid hex

    def test_null_serialized_explicitly(self):
        """None must serialize as JSON null, not be omitted."""
        obj_with_null = {"a": None, "b": 1}
        obj_without_null = {"b": 1}
        # These have different canonical forms
        assert compute_hash(obj_with_null) != compute_hash(obj_without_null)


class TestFreezeAndVerify:
    def test_freeze_adds_hash_field(self):
        obj = {"ticket_id": "TKT_001", "joint_prob": 0.5}
        frozen = freeze_object_with_hash(obj, "phase3_frozen_hash")
        assert "phase3_frozen_hash" in frozen
        assert isinstance(frozen["phase3_frozen_hash"], str)

    def test_verify_valid_hash(self):
        obj = {"x": 42, "y": "hello"}
        frozen = freeze_object_with_hash(obj, "phase1_frozen_hash")
        result = verify_hash(frozen, "phase1_frozen_hash", phase=1)
        assert result["valid"] is True
        assert result["error"] is None

    def test_verify_tampered_hash_fails(self):
        obj = {"x": 42}
        frozen = freeze_object_with_hash(obj, "phase1_frozen_hash")
        frozen["x"] = 99  # tamper
        result = verify_hash(frozen, "phase1_frozen_hash", phase=1)
        assert result["valid"] is False
        assert "HASH_MISMATCH" in result["error"]

    def test_verify_missing_hash_field(self):
        obj = {"x": 1}
        result = verify_hash(obj, "phase1_frozen_hash", phase=1)
        assert result["valid"] is False


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------

class TestRunContext:
    def _valid_kwargs(self):
        return {
            "run_id": "RUN_001",
            "schema_version": SCHEMA_VERSION,
            "objective_id": "OBJ_01",
            "target_date_utc": "2026-03-10",
            "execution_book": EXECUTION_BOOK,
            "bankroll_input": 10000.0,
            "bankroll_input_currency": "USD",
            "bankroll_input_ts_utc": "2026-03-10T12:00:00Z",
            "bankroll_integrity_state": "VERIFIED",
            "kelly_fraction_config_id": "DEFAULT_1_8",
            "kelly_cap_fraction_of_bankroll": 0.02,
            "min_operational_fraction": 0.001,
            "minimum_operational_dollar_threshold": 1.0,
            "run_created_ts_utc": "2026-03-10T12:00:00Z",
        }

    def test_valid_construction(self):
        ctx = RunContext(**self._valid_kwargs())
        assert ctx.execution_book == EXECUTION_BOOK
        assert ctx.bankroll_input == 10000.0

    def test_wrong_execution_book_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["execution_book"] = "DRAFTKINGS"
        with pytest.raises(ValueError, match="FANDUEL"):
            RunContext(**kwargs)

    def test_wrong_schema_version_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["schema_version"] = "v99"
        with pytest.raises(ValueError, match="v15.1"):
            RunContext(**kwargs)

    def test_negative_bankroll_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["bankroll_input"] = -100.0
        with pytest.raises(ValueError, match="bankroll"):
            RunContext(**kwargs)

    def test_to_dict_roundtrip(self):
        ctx = RunContext(**self._valid_kwargs())
        d = ctx.to_dict()
        assert d["run_id"] == "RUN_001"
        assert d["execution_book"] == EXECUTION_BOOK


# ---------------------------------------------------------------------------
# SnapshotBundle
# ---------------------------------------------------------------------------

class TestSnapshotBundle:
    def _valid_bundle_kwargs(self):
        return {
            "snapshot_bundle_id": "20260310_143215_UTC",
            "bundle_frozen_ts_utc": "2026-03-10T14:32:55Z",
            "nba_status_snapshot_ts_utc": "2026-03-10T14:32:45Z",
            "nba_schedule_snapshot_ts_utc": "2026-03-10T14:30:05Z",
            "nba_stats_snapshot_ts_utc": "2026-03-10T14:30:00Z",
            "team_context_snapshot_ts_utc": "2026-03-10T14:30:15Z",
            "referee_snapshot_ts_utc": "2026-03-10T14:30:30Z",
            "fanduel_market_snapshot_ts_utc": "2026-03-10T14:32:15Z",
            "valuation_market_snapshot_ts_utc": "2026-03-10T14:32:50Z",
        }

    def test_valid_bundle_passes_atomicity(self):
        bundle = SnapshotBundle(**self._valid_bundle_kwargs())
        assert bundle.validate_atomicity() is True
        assert bundle.bundle_status == "VALID"
        assert bundle.max_component_time_delta_seconds == pytest.approx(170, abs=1)

    def test_stale_bundle_fails_atomicity(self):
        kwargs = self._valid_bundle_kwargs()
        # Push one timestamp 10 minutes ahead (600s)
        kwargs["valuation_market_snapshot_ts_utc"] = "2026-03-10T14:40:00Z"
        bundle = SnapshotBundle(**kwargs)
        assert bundle.validate_atomicity() is False
        assert bundle.bundle_status == "STALE_SYNCHRONIZATION_FAILURE"

    def test_freeze_computes_hash(self):
        bundle = SnapshotBundle(**self._valid_bundle_kwargs())
        bundle.freeze()
        assert bundle.bundle_status == "VALID"
        assert isinstance(bundle.bundle_integrity_hash, str)
        assert len(bundle.bundle_integrity_hash) == 64

    def test_to_dict_has_all_required_fields(self):
        bundle = SnapshotBundle(**self._valid_bundle_kwargs())
        bundle.freeze()
        d = bundle.to_dict()
        required = [
            "snapshot_bundle_id", "bundle_frozen_ts_utc",
            "nba_status_snapshot_ts_utc", "nba_schedule_snapshot_ts_utc",
            "nba_stats_snapshot_ts_utc", "team_context_snapshot_ts_utc",
            "referee_snapshot_ts_utc", "fanduel_market_snapshot_ts_utc",
            "valuation_market_snapshot_ts_utc", "bundle_integrity_hash",
            "bundle_status", "max_component_time_delta_seconds",
            "snapshot_sequence_log",
        ]
        for field in required:
            assert field in d, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

class TestGovernance:
    def test_fanduel_contamination_non_fd_source(self):
        obj = {"fd_current_line": 25.5, "other_field": 1}
        violations = check_fanduel_field_contamination(obj, "draftkings")
        assert "fd_current_line" in violations

    def test_fanduel_contamination_fd_source_clean(self):
        obj = {"fd_current_line": 25.5}
        violations = check_fanduel_field_contamination(obj, "fanduel")
        assert violations == []

    def test_scope_violation_raises(self):
        with pytest.raises(ScopeViolation):
            check_and_raise_scope_violation(True, "test violation")

    def test_scope_violation_not_raised_when_false(self):
        check_and_raise_scope_violation(False, "should not raise")


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

class TestConstants:
    def test_devig_proportional_authorized(self):
        assert DEVIG_METHOD_REGISTRY["PROPORTIONAL_V1"]["production_authorized"] is True

    def test_devig_shin_not_authorized(self):
        assert DEVIG_METHOD_REGISTRY["SHIN_V1"]["production_authorized"] is False

    def test_schema_version(self):
        assert SCHEMA_VERSION == "v15.1"
