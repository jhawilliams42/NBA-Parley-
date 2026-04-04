"""
Tests for Phase 2 — Valuation, Edge, Gates A/B/C/D, Leg Approval.
"""

from datetime import datetime, timezone

import pytest

from nba_prop_engine.phase2.edge import (
    compute_edge,
    compute_execution_price_and_break_even,
    compute_model_prob,
)
from nba_prop_engine.phase2.gates import (
    evaluate_gate_a,
    evaluate_gate_b,
    evaluate_gate_c,
    evaluate_gate_d,
)
from nba_prop_engine.phase2.valuation import (
    build_fair_probability_consensus,
    process_valuation_book,
    proportional_devig,
    validate_market_structure,
)
from nba_prop_engine.phase0.hash_utils import freeze_object_with_hash


# ---------------------------------------------------------------------------
# Valuation / De-vig
# ---------------------------------------------------------------------------

class TestMarketValidation:
    def test_valid_market(self):
        result = validate_market_structure(-110, -110)
        assert result["valid"] is True
        assert result["total_implied_prob"] > 1.0

    def test_no_vig_invalid(self):
        # +100 / +100 → total = 1.0 (no vig)
        result = validate_market_structure(100, 100)
        assert result["valid"] is False

    def test_excessive_vig_invalid(self):
        # Very short odds → huge vig
        result = validate_market_structure(-1000, -1000)
        assert result["valid"] is False

    def test_invalid_odds_zero(self):
        # american_to_decimal(0) raises ValueError, caught internally → valid=False
        result = validate_market_structure(0, -110)
        assert result["valid"] is False


class TestProportionalDevig:
    def test_proportional_sums_to_one(self):
        fair_over, fair_under, vig = proportional_devig(0.55, 0.52)
        assert fair_over + fair_under == pytest.approx(1.0, abs=1e-9)

    def test_vig_positive(self):
        _, _, vig = proportional_devig(0.55, 0.52)
        assert vig > 0

    def test_symmetric_market(self):
        fair_over, fair_under, _ = proportional_devig(0.52381, 0.52381)
        assert fair_over == pytest.approx(0.5, abs=1e-3)
        assert fair_under == pytest.approx(0.5, abs=1e-3)


class TestProcessValuationBook:
    def test_valid_book_proportional(self):
        result = process_valuation_book("DraftKings", -110, -110)
        assert result["book_status"] == "SUCCESS"
        assert result["devig_method"] == "PROPORTIONAL_V1"
        assert 0 < result["val_fair_p"] < 1

    def test_shin_v1_blocked_in_production(self):
        result = process_valuation_book("DraftKings", -110, -110, preferred_method="SHIN_V1")
        assert result["book_status"] == "EXCLUDED"
        assert "NOT_PRODUCTION_AUTHORIZED" in result["exclusion_reason"]

    def test_invalid_market_excluded(self):
        result = process_valuation_book("BetMGM", 100, 100)
        assert result["book_status"] == "EXCLUDED"


class TestBuildConsensus:
    def test_single_valid_book(self):
        books = [
            {"book_name": "DraftKings", "val_fair_p": 0.55, "book_status": "SUCCESS"}
        ]
        result = build_fair_probability_consensus(books)
        assert result["val_fair_p"] == pytest.approx(0.55)
        assert result["val_book_count"] == 1

    def test_multiple_valid_books_median(self):
        books = [
            {"book_name": "DK", "val_fair_p": 0.50, "book_status": "SUCCESS"},
            {"book_name": "MGM", "val_fair_p": 0.54, "book_status": "SUCCESS"},
            {"book_name": "CZ", "val_fair_p": 0.58, "book_status": "SUCCESS"},
        ]
        result = build_fair_probability_consensus(books)
        assert result["val_fair_p"] == pytest.approx(0.54, abs=1e-6)
        assert result["val_book_count"] == 3

    def test_no_valid_books(self):
        books = [{"book_name": "DK", "book_status": "EXCLUDED"}]
        result = build_fair_probability_consensus(books)
        assert result["val_fair_p"] is None
        assert result["val_book_count"] == 0
        assert result["valuation_integrity_state"] == "NO_VALID_BOOKS"


# ---------------------------------------------------------------------------
# Execution price and edge
# ---------------------------------------------------------------------------

class TestExecutionPriceAndBreakEven:
    def test_negative_american_odds(self):
        result = compute_execution_price_and_break_even(-110)
        assert result["fd_execution_odds_decimal"] == pytest.approx(100/110 + 1, rel=1e-6)
        assert result["break_even"] == pytest.approx(110/210, rel=1e-6)

    def test_positive_american_odds(self):
        result = compute_execution_price_and_break_even(200)
        assert result["fd_execution_odds_decimal"] == pytest.approx(3.0)
        assert result["break_even"] == pytest.approx(1/3, rel=1e-5)

    def test_none_odds_returns_invalid(self):
        result = compute_execution_price_and_break_even(None)
        assert result["fd_execution_odds_decimal"] is None
        assert result["execution_price_integrity_state"] == "INVALID"

    def test_zero_odds_returns_invalid(self):
        result = compute_execution_price_and_break_even(0)
        assert result["execution_price_integrity_state"] == "INVALID"


class TestComputeEdge:
    def test_positive_edge(self):
        result = compute_edge(0.60, 0.50)
        # edge = (0.60 - 0.50) / 0.50 = 0.20
        assert result["current_edge_pct"] == pytest.approx(0.20)

    def test_negative_edge(self):
        result = compute_edge(0.45, 0.50)
        assert result["current_edge_pct"] < 0

    def test_missing_model_prob(self):
        result = compute_edge(None, 0.50)
        assert result["current_edge_pct"] is None
        assert result["edge_integrity_state"] == "INVALID"

    def test_zero_break_even(self):
        result = compute_edge(0.6, 0.0)
        assert result["current_edge_pct"] is None


class TestModelProb:
    def test_no_val_fair_p_returns_raw(self):
        result = compute_model_prob(0.60, None, "HIGH", 25)
        assert result["model_prob"] == pytest.approx(0.60)
        assert result["shrinkage_method_id"] == "NONE_NO_VALUATION"

    def test_high_repeatability_zero_shrinkage(self):
        result = compute_model_prob(0.60, 0.50, "HIGH", 25)
        # shrink_w=0 → model_prob = raw = 0.60
        assert result["model_prob"] == pytest.approx(0.60)
        assert result["shrink_w"] == 0.0

    def test_low_repeatability_high_shrinkage(self):
        result = compute_model_prob(0.60, 0.50, "LOW", 25)
        # shrink_w=0.25 → 0.75*0.60 + 0.25*0.50 = 0.45 + 0.125 = 0.575
        assert result["model_prob"] == pytest.approx(0.575)


# ---------------------------------------------------------------------------
# Gate A
# ---------------------------------------------------------------------------

class TestGateA:
    def _valid_leg(self):
        obj = {
            "player_id": "P001",
            "phase1_frozen_hash": "abc",
            "phase1_integrity_status": "PASS",
            "raw_event_prob_over_current_line": 0.55,
            "raw_event_prob_integrity_state": "DERIVED_BY_APPROVED_RULE",
            "fd_current_line": 25.5,
            "fd_execution_odds_american": -110,
            "normalized_status": "ACTIVE",
        }
        return obj

    def test_valid_leg_passes_gate_a(self):
        result, reasons = evaluate_gate_a(self._valid_leg(), hash_verified=True)
        assert result == "PASS"

    def test_missing_hash_fails(self):
        leg = self._valid_leg()
        result, reasons = evaluate_gate_a(leg, hash_verified=False)
        assert result == "FAIL"
        assert any("HASH" in r for r in reasons)

    def test_blocking_status_fails(self):
        leg = self._valid_leg()
        leg["normalized_status"] = "OUT"
        result, reasons = evaluate_gate_a(leg, hash_verified=True)
        assert result == "FAIL"

    def test_missing_line_fails(self):
        leg = self._valid_leg()
        leg["fd_current_line"] = None
        result, reasons = evaluate_gate_a(leg, hash_verified=True)
        assert result == "FAIL"


# ---------------------------------------------------------------------------
# Gate B
# ---------------------------------------------------------------------------

class TestGateB:
    def test_above_floor_passes(self):
        leg = {"model_prob": 0.55}
        result, _ = evaluate_gate_b(leg)
        assert result == "PASS"

    def test_at_floor_passes(self):
        leg = {"model_prob": 0.50}
        result, _ = evaluate_gate_b(leg)
        assert result == "PASS"

    def test_below_floor_fails(self):
        leg = {"model_prob": 0.49}
        result, reasons = evaluate_gate_b(leg)
        assert result == "FAIL"

    def test_missing_prob_fails(self):
        result, _ = evaluate_gate_b({})
        assert result == "FAIL"


# ---------------------------------------------------------------------------
# Gate C
# ---------------------------------------------------------------------------

class TestGateC:
    def _branch2_leg(self, edge=0.06, model_prob=0.56, book_count=3, disagreement=0.02):
        return {
            "val_fair_p": 0.55,
            "val_book_count": book_count,
            "val_disagreement_score": disagreement,
            "val_stale_flag": False,
            "val_inflation_flag": False,
            "current_edge_pct": edge,
            "model_prob": model_prob,
            "predictive_lower_bound": None,
            "predictive_lower_bound_integrity_state": "UNASSESSED",
            "break_even": 0.50,
        }

    def test_branch2_passes_all_conditions_met(self):
        result, _ = evaluate_gate_c(self._branch2_leg())
        assert result == "PASS"

    def test_branch2_fails_insufficient_books(self):
        result, reasons = evaluate_gate_c(self._branch2_leg(book_count=2))
        assert result == "FAIL"
        assert any("BOOK_COUNT" in r for r in reasons)

    def test_branch2_fails_high_disagreement(self):
        result, reasons = evaluate_gate_c(self._branch2_leg(disagreement=0.06))
        assert result == "FAIL"

    def test_branch2_fails_low_edge(self):
        result, reasons = evaluate_gate_c(self._branch2_leg(edge=0.04))
        assert result == "FAIL"

    def test_branch3_provisional_pass(self):
        leg = {
            "val_fair_p": None,
            "val_book_count": 0,
            "predictive_lower_bound": None,
            "predictive_lower_bound_integrity_state": "UNASSESSED",
            "repeatability_class": "HIGH",
            "minutes_fragility_class": "LOW",
            "functional_status_class": "CLEAN",
            "sample_n": 25,
            "model_prob": 0.62,
            "current_edge_pct": 0.09,
            "fd_prop_market_status": "ACTIVE",
            "break_even": 0.55,
        }
        result, _ = evaluate_gate_c(leg)
        assert result == "PROVISIONAL_PASS_PENDING_MANUAL_REVIEW"


# ---------------------------------------------------------------------------
# Gate D (Section 2.12, 0.23 authoritative implementation)
# ---------------------------------------------------------------------------

class TestGateD:
    def _clean_leg(self, now=None):
        tip = datetime(2026, 3, 10, 20, 0, tzinfo=timezone.utc)
        return {
            "minutes_fragility_class": "LOW",
            "functional_status_class": "CLEAN",
            "epistemic_disqualifier": False,
            "ramp_risk_class": "LOW",
            "normalized_status": "ACTIVE",
            "gtd_play_rate": None,
            "tip_time_utc": tip.isoformat(),
            "restriction_severity_class": "LOW",
        }

    def _now(self):
        return datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc)

    def test_clean_leg_passes(self):
        result, reasons = evaluate_gate_d(self._clean_leg(), self._now())
        assert result == "PASS"
        assert reasons == []

    def test_condition1_high_fragility_non_clean(self):
        leg = self._clean_leg()
        leg["minutes_fragility_class"] = "HIGH"
        leg["functional_status_class"] = "LIMITED"
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert "HIGH_FRAGILITY_NON_CLEAN_STATUS" in reasons

    def test_condition1_high_uncertainty_also_triggers(self):
        leg = self._clean_leg()
        leg["minutes_fragility_class"] = "HIGH_UNCERTAINTY"
        leg["functional_status_class"] = "LIMITED"
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert "HIGH_FRAGILITY_NON_CLEAN_STATUS" in reasons

    def test_condition1_high_fragility_clean_does_not_trigger(self):
        """HIGH fragility with CLEAN status should NOT trigger Condition 1."""
        leg = self._clean_leg()
        leg["minutes_fragility_class"] = "HIGH"
        # functional_status_class stays "CLEAN"
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "PASS"

    def test_condition2_epistemic_disqualifier(self):
        leg = self._clean_leg()
        leg["epistemic_disqualifier"] = True
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert "EPISTEMIC_UNCERTAINTY_TOO_HIGH" in reasons

    def test_condition3_high_ramp_risk(self):
        leg = self._clean_leg()
        leg["ramp_risk_class"] = "HIGH"
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert "HIGH_RAMP_RISK" in reasons

    def test_condition4_gtd_low_play_rate(self):
        leg = self._clean_leg()
        leg["normalized_status"] = "GTD"
        leg["gtd_play_rate"] = 0.30  # below 0.35
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert "GTD_LOW_PLAY_RATE" in reasons

    def test_condition4_gtd_acceptable_play_rate_passes(self):
        leg = self._clean_leg()
        leg["normalized_status"] = "GTD"
        leg["gtd_play_rate"] = 0.80
        result, _ = evaluate_gate_d(leg, self._now())
        assert result == "PASS"

    def test_condition5_gtd_no_play_rate_near_tip(self):
        now = datetime(2026, 3, 10, 19, 30, tzinfo=timezone.utc)  # 30min before tip
        leg = self._clean_leg()
        leg["normalized_status"] = "GTD"
        leg["gtd_play_rate"] = None
        result, reasons = evaluate_gate_d(leg, now)
        assert result == "FAIL"
        assert "GTD_PLAY_RATE_MISSING_NEAR_TIP" in reasons

    def test_condition5_gtd_no_play_rate_far_from_tip_passes(self):
        now = datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc)  # 10h before tip
        leg = self._clean_leg()
        leg["normalized_status"] = "GTD"
        leg["gtd_play_rate"] = None
        result, _ = evaluate_gate_d(leg, now)
        assert result == "PASS"

    def test_condition6_fragile_status(self):
        leg = self._clean_leg()
        leg["functional_status_class"] = "FRAGILE"
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert "FRAGILE_OR_UNRESOLVABLE_STATUS" in reasons

    def test_condition6_unresolvable_status(self):
        leg = self._clean_leg()
        leg["functional_status_class"] = "UNRESOLVABLE"
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert "FRAGILE_OR_UNRESOLVABLE_STATUS" in reasons

    def test_condition7_high_restriction(self):
        leg = self._clean_leg()
        leg["restriction_severity_class"] = "HIGH"
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert "HIGH_RESTRICTION_SEVERITY" in reasons

    def test_multiple_conditions_all_reported(self):
        leg = self._clean_leg()
        leg["epistemic_disqualifier"] = True
        leg["ramp_risk_class"] = "HIGH"
        result, reasons = evaluate_gate_d(leg, self._now())
        assert result == "FAIL"
        assert len(reasons) >= 2
