"""
Tests for Phase 1 — Status, Fragility, Dependency, Distribution, Audit.
"""

import pytest

from nba_prop_engine.phase1.audit import (
    execute_P1_S11A_input_audit,
    execute_P1_S24A_derived_audit,
)
from nba_prop_engine.phase1.dependency import (
    FIELD_DEPENDENCIES,
    build_reverse_dependency_graph,
    classify_and_handle_recompute_failure,
    collect_transitive_dependents,
    detect_injury_state_change,
    identify_primary_affected_fields,
    topological_sort_subset,
    validate_dependencies_at_compute_time,
)
from nba_prop_engine.phase1.distribution import (
    american_to_decimal,
    american_to_implied_prob,
    compute_raw_event_prob_over,
    select_distribution,
)
from nba_prop_engine.phase1.fragility import compute_fragility_score
from nba_prop_engine.phase1.status import normalize_status, is_status_blocking


# ---------------------------------------------------------------------------
# Status normalization
# ---------------------------------------------------------------------------

class TestNormalizeStatus:
    def test_active(self):
        assert normalize_status("ACTIVE", None) == "ACTIVE"

    def test_out(self):
        assert normalize_status("OUT", None) == "OUT"

    def test_gtd(self):
        assert normalize_status("GTD", None) == "GTD"

    def test_game_time_decision(self):
        assert normalize_status("GAME TIME DECISION", None) == "GTD"

    def test_questionable(self):
        assert normalize_status("QUESTIONABLE", None) == "QUESTIONABLE"

    def test_doubtful(self):
        assert normalize_status("DOUBTFUL", None) == "DOUBTFUL"

    def test_inactive_designation_overrides(self):
        result = normalize_status("ACTIVE", "INACTIVE")
        assert result == "INACTIVE_OTHER"

    def test_inactive_with_out_designation(self):
        result = normalize_status("OUT", "INACTIVE")
        assert result == "OUT"

    def test_missing_both_returns_unresolvable(self):
        assert normalize_status(None, None) == "UNRESOLVABLE"

    def test_blocking_statuses(self):
        for s in ("OUT", "INACTIVE_OTHER", "UNRESOLVABLE"):
            assert is_status_blocking(s) is True

    def test_non_blocking_statuses(self):
        for s in ("ACTIVE", "GTD", "QUESTIONABLE"):
            assert is_status_blocking(s) is False


# ---------------------------------------------------------------------------
# Fragility model
# ---------------------------------------------------------------------------

class TestFragilityModel:
    def _base_obj(self, override=None):
        obj = {
            "minutes_fragility_component": "LOW",
            "injury_fragility_component": "LOW",
            "foul_fragility_component": "LOW",
            "role_fragility_component": "LOW",
            "rotation_fragility_component": "LOW",
            "blowout_fragility_component": "LOW",
            "dependency_fragility_component": "LOW",
            "uncertainty_fragility_component": "LOW",
        }
        if override:
            obj.update(override)
        return obj

    def test_all_low_gives_low_class(self):
        result = compute_fragility_score(self._base_obj())
        assert result["minutes_fragility_class"] == "LOW"
        assert result["epistemic_disqualifier"] is False

    def test_all_high_gives_high_class(self):
        obj = {k: "HIGH" for k in self._base_obj()}
        result = compute_fragility_score(obj)
        assert result["minutes_fragility_class"] == "HIGH"

    def test_three_unknowns_triggers_epistemic_disqualifier(self):
        obj = self._base_obj({
            "minutes_fragility_component": "UNKNOWN",
            "injury_fragility_component": "UNKNOWN",
            "foul_fragility_component": "UNKNOWN",
        })
        result = compute_fragility_score(obj)
        assert result["epistemic_disqualifier"] is True
        assert result["minutes_fragility_class"] == "HIGH_UNCERTAINTY"
        assert result["fragility_score"] is None

    def test_two_unknowns_does_not_trigger_disqualifier(self):
        obj = self._base_obj({
            "minutes_fragility_component": "UNKNOWN",
            "injury_fragility_component": "UNKNOWN",
        })
        result = compute_fragility_score(obj)
        assert result["epistemic_disqualifier"] is False

    def test_moderate_score_gives_moderate_class(self):
        obj = self._base_obj({
            "minutes_fragility_component": "MODERATE",
            "injury_fragility_component": "MODERATE",
        })
        result = compute_fragility_score(obj)
        assert result["minutes_fragility_class"] in ("LOW", "MODERATE")


# ---------------------------------------------------------------------------
# Dependency validation
# ---------------------------------------------------------------------------

class TestDependencyValidation:
    def test_valid_dependencies_pass(self):
        obj = {
            "usage_rate": 0.28,
            "usage_rate_dependency_state": "VALID",
            "usage_rate_integrity_state": "VERIFIED",
            "field_goal_attempts": 12,
            "field_goal_attempts_dependency_state": "VALID",
            "field_goal_attempts_integrity_state": "VERIFIED",
            "touches": 40,
            "touches_dependency_state": "VALID",
            "touches_integrity_state": "DERIVED_BY_APPROVED_RULE",
            "drives": 8,
            "drives_dependency_state": "VALID",
            "drives_integrity_state": "DERIVED_BY_APPROVED_RULE",
        }
        is_valid, invalid = validate_dependencies_at_compute_time("opportunity_context_class", obj)
        assert is_valid is True
        assert invalid == []

    def test_missing_dependency_fails(self):
        obj = {
            "usage_rate_dependency_state": "NOT_COMPUTED",
        }
        is_valid, invalid = validate_dependencies_at_compute_time("opportunity_context_class", obj)
        assert is_valid is False
        assert "usage_rate" in invalid

    def test_failed_dependency_state_fails(self):
        obj = {
            "usage_rate_dependency_state": "FAILED",
            "usage_rate_integrity_state": "DEPENDENCY_FAILURE",
        }
        is_valid, invalid = validate_dependencies_at_compute_time("opportunity_context_class", obj)
        assert is_valid is False


class TestTopologicalSort:
    def test_simple_chain(self):
        deps = {
            "A": {"depends_on": []},
            "B": {"depends_on": ["A"]},
            "C": {"depends_on": ["B"]},
        }
        order = topological_sort_subset({"A", "B", "C"}, deps)
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_cycle_raises(self):
        deps = {
            "A": {"depends_on": ["B"]},
            "B": {"depends_on": ["A"]},
        }
        with pytest.raises(ValueError, match="cycle"):
            topological_sort_subset({"A", "B"}, deps)

    def test_reverse_graph_builds_correctly(self):
        deps = {
            "minutes_fragility_class": {
                "depends_on": ["opportunity_context_class", "role_lock_class"],
            }
        }
        reverse = build_reverse_dependency_graph(deps)
        assert "minutes_fragility_class" in reverse.get("opportunity_context_class", set())
        assert "minutes_fragility_class" in reverse.get("role_lock_class", set())

    def test_transitive_dependents(self):
        deps = {
            "A": {"depends_on": []},
            "B": {"depends_on": ["A"]},
            "C": {"depends_on": ["B"]},
            "D": {"depends_on": ["C"]},
        }
        reverse = build_reverse_dependency_graph(deps)
        affected = collect_transitive_dependents(["A"], reverse)
        assert affected == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# Distribution selection and probability
# ---------------------------------------------------------------------------

class TestDistributionSelection:
    def test_insufficient_sample_invalid(self):
        result = select_distribution(5, 20.0, 5.0, "points")
        assert result == "DISTRIBUTION_INVALID_NUMERICS"

    def test_positive_family_uses_gamma(self):
        result = select_distribution(30, 20.0, 5.0, "points")
        assert result == "GAMMA"

    def test_invalid_std_returns_invalid(self):
        result = select_distribution(20, 20.0, 0.0, "points")
        assert result == "DISTRIBUTION_INVALID_NUMERICS"

    def test_missing_mean_returns_invalid(self):
        result = select_distribution(20, None, 5.0, "points")
        assert result == "DISTRIBUTION_INVALID_NUMERICS"


class TestRawEventProb:
    def test_normal_distribution_over_line(self):
        result = compute_raw_event_prob_over("NORMAL", 20.0, 22.0, 5.0)
        prob = result["raw_event_prob_over_current_line"]
        assert prob is not None
        assert 0 < prob < 1

    def test_gamma_distribution_over_line(self):
        result = compute_raw_event_prob_over("GAMMA", 20.0, 22.0, 5.0)
        prob = result["raw_event_prob_over_current_line"]
        assert prob is not None
        assert 0 < prob < 1

    def test_empirical_distribution(self):
        import numpy as np
        values = list(np.random.default_rng(42).normal(22, 5, 30))
        result = compute_raw_event_prob_over("EMPIRICAL", 20.0, 22.0, 5.0, values)
        prob = result["raw_event_prob_over_current_line"]
        assert prob is not None
        assert 0 <= prob <= 1

    def test_invalid_distribution_returns_invalid(self):
        result = compute_raw_event_prob_over("DISTRIBUTION_INVALID_NUMERICS", 20.0, 22.0, 5.0)
        assert result["raw_event_prob_over_current_line"] is None
        assert result["integrity_state"] == "INVALID"

    def test_poisson_over_line(self):
        result = compute_raw_event_prob_over("POISSON", 1.5, 2.5, 1.5)
        prob = result["raw_event_prob_over_current_line"]
        assert prob is not None
        assert 0 <= prob <= 1


class TestAmericanOdds:
    def test_favorite_to_decimal(self):
        # -110 → (100/110) + 1 = 1.909...
        assert american_to_decimal(-110) == pytest.approx(100 / 110 + 1, rel=1e-6)

    def test_underdog_to_decimal(self):
        # +200 → (200/100) + 1 = 3.0
        assert american_to_decimal(200) == pytest.approx(3.0)

    def test_implied_prob_favorite(self):
        prob = american_to_implied_prob(-110)
        assert prob == pytest.approx(110 / 210, rel=1e-6)

    def test_implied_prob_underdog(self):
        prob = american_to_implied_prob(200)
        assert prob == pytest.approx(1 / 3, rel=1e-5)


# ---------------------------------------------------------------------------
# P1-S11A Input Audit
# ---------------------------------------------------------------------------

class TestP1S11AAudit:
    def _valid_obj(self):
        return {
            "player_id": "P001",
            "lineup_context_loaded": True,
            "injury_status_loaded": True,
            "player_stats_loaded": True,
            "fanduel_market_loaded": True,
        }

    def test_valid_object_passes(self):
        result = execute_P1_S11A_input_audit([self._valid_obj()])
        assert len(result) == 1

    def test_missing_fanduel_market_fails(self):
        obj = self._valid_obj()
        obj["fanduel_market_loaded"] = False
        result = execute_P1_S11A_input_audit([obj])
        assert len(result) == 0
        assert obj["phase1_integrity_status"] == "INPUT_DEPENDENCY_FAILURE"

    def test_multiple_missing_inputs_all_listed(self):
        obj = {
            "player_id": "P002",
            "lineup_context_loaded": False,
            "injury_status_loaded": False,
            "player_stats_loaded": True,
            "fanduel_market_loaded": True,
        }
        execute_P1_S11A_input_audit([obj])
        assert obj["phase1_integrity_status"] == "INPUT_DEPENDENCY_FAILURE"


# ---------------------------------------------------------------------------
# P1-S24A Derived Audit
# ---------------------------------------------------------------------------

class TestP1S24AAudit:
    def _valid_derived_obj(self):
        obj = {"player_id": "P001"}
        for field in [
            "opportunity_context_class",
            "minutes_fragility_class",
            "role_lock_class",
            "repeatability_class",
            "functional_status_class",
        ]:
            obj[field] = "MODERATE"
            obj[f"{field}_integrity_state"] = "DERIVED_BY_APPROVED_RULE"
        return obj

    def test_valid_derived_fields_pass(self):
        result = execute_P1_S24A_derived_audit([self._valid_derived_obj()])
        assert len(result) == 1

    def test_null_binding_value_fails(self):
        obj = self._valid_derived_obj()
        obj["minutes_fragility_class"] = None
        result = execute_P1_S24A_derived_audit([obj])
        assert len(result) == 0
        assert obj["phase1_integrity_status"] == "DERIVED_FIELD_AUDIT_FAILURE"

    def test_invalid_integrity_state_fails(self):
        obj = self._valid_derived_obj()
        obj["role_lock_class_integrity_state"] = "MISSING"
        result = execute_P1_S24A_derived_audit([obj])
        assert len(result) == 0

    def test_recomputed_post_freeze_passes(self):
        obj = self._valid_derived_obj()
        obj["repeatability_class_integrity_state"] = "RECOMPUTED_POST_FREEZE"
        result = execute_P1_S24A_derived_audit([obj])
        assert len(result) == 1
