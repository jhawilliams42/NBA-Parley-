"""
Tests for Phase 3 — Correlation, Joint Probability, Ticket Construction.
"""

import numpy as np
import pytest

from nba_prop_engine.phase3.correlation import (
    build_correlation_matrix,
    higham_nearest_corr,
    validate_and_repair_correlation_matrix,
    validate_psd,
)
from nba_prop_engine.phase3.joint_prob import (
    compute_joint_prob_independent,
    compute_joint_probability,
)
from nba_prop_engine.phase3.pipeline import (
    build_ticket,
    check_leg_phase3_eligible,
    check_same_game_rule,
    check_unique_player_rule,
    compute_ticket_ev,
    compute_ticket_price,
)
from nba_prop_engine.phase0.hash_utils import freeze_object_with_hash


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def _approved_leg(player_id="P1", game_id="G1", model_prob=0.60):
    return {
        "player_id": player_id,
        "game_id": game_id,
        "model_prob": model_prob,
        "dependence_method_id": "NONE_INDEPENDENCE_ALLOWED",
    }


class TestCorrelationMatrix:
    def test_independent_legs_identity_matrix(self):
        legs = [_approved_leg("P1", "G1"), _approved_leg("P2", "G2")]
        C = build_correlation_matrix(legs)
        assert C.shape == (2, 2)
        np.testing.assert_array_almost_equal(C, np.eye(2))

    def test_same_player_uses_rho(self):
        legs = [
            {**_approved_leg("P1", "G1"), "dependence_method_id": "LATENT_EVENT_RHO_V1",
             "same_player_rho": 0.5},
            {**_approved_leg("P1", "G1"), "dependence_method_id": "LATENT_EVENT_RHO_V1",
             "same_player_rho": 0.5},
        ]
        C = build_correlation_matrix(legs)
        assert C[0, 1] == pytest.approx(0.5)
        assert C[1, 0] == pytest.approx(0.5)

    def test_valid_psd_matrix(self):
        C = np.eye(3)
        is_psd, eigenvalues = validate_psd(C)
        assert is_psd is True
        assert all(e >= 0 for e in eigenvalues)

    def test_non_psd_matrix(self):
        C = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        is_psd, _ = validate_psd(C)
        assert is_psd is False

    def test_higham_repair_converges(self):
        C = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
        C_rep, iterations, converged, norm = higham_nearest_corr(C)
        assert converged is True
        # Repaired matrix should be approximately PSD (eigenvalues >= -tolerance)
        eigenvalues = np.linalg.eigvalsh(C_rep)
        assert all(e >= -1e-6 for e in eigenvalues)

    def test_validate_and_repair_identity(self):
        legs = [_approved_leg("P1", "G1"), _approved_leg("P2", "G2")]
        result = validate_and_repair_correlation_matrix(legs)
        assert result["correlation_matrix_integrity_state"] == "VALID"
        assert result["joint_prob_integrity_status"] == "PASS"


# ---------------------------------------------------------------------------
# Joint probability
# ---------------------------------------------------------------------------

class TestJointProbIndependent:
    def test_two_independent_legs(self):
        legs = [
            {"player_id": "P1", "game_id": "G1", "model_prob": 0.60},
            {"player_id": "P2", "game_id": "G2", "model_prob": 0.70},
        ]
        result = compute_joint_prob_independent(legs)
        assert result["joint_prob"] == pytest.approx(0.42)
        assert result["joint_prob_integrity_status"] == "PASS"

    def test_missing_model_prob_fails(self):
        legs = [
            {"player_id": "P1", "game_id": "G1", "model_prob": None},
        ]
        result = compute_joint_prob_independent(legs)
        assert result["joint_prob"] is None
        assert result["joint_prob_integrity_status"] == "FAIL"

    def test_single_leg(self):
        legs = [{"player_id": "P1", "game_id": "G1", "model_prob": 0.55}]
        result = compute_joint_prob_independent(legs)
        assert result["joint_prob"] == pytest.approx(0.55)

    def test_three_independent_legs(self):
        legs = [
            {"player_id": "P1", "game_id": "G1", "model_prob": 0.60},
            {"player_id": "P2", "game_id": "G2", "model_prob": 0.65},
            {"player_id": "P3", "game_id": "G3", "model_prob": 0.70},
        ]
        result = compute_joint_prob_independent(legs)
        assert result["joint_prob"] == pytest.approx(0.60 * 0.65 * 0.70, rel=1e-6)


class TestComputeJointProbability:
    def test_independent_legs_use_product(self):
        legs = [
            {"player_id": "P1", "game_id": "G1", "model_prob": 0.60,
             "dependence_method_id": "NONE_INDEPENDENCE_ALLOWED"},
            {"player_id": "P2", "game_id": "G2", "model_prob": 0.70,
             "dependence_method_id": "NONE_INDEPENDENCE_ALLOWED"},
        ]
        result = compute_joint_probability(legs)
        assert result["joint_prob"] == pytest.approx(0.42)
        assert result["dependence_method_used"] == "NONE_INDEPENDENCE_ALLOWED"


# ---------------------------------------------------------------------------
# Ticket construction rules
# ---------------------------------------------------------------------------

def _make_approved_leg(player_id, game_id, model_prob=0.60, fd_decimal=2.5):
    leg = {
        "player_id": player_id,
        "game_id": game_id,
        "model_prob": model_prob,
        "fd_execution_odds_decimal": fd_decimal,
        "dependence_method_id": "NONE_INDEPENDENCE_ALLOWED",
        "leg_approval_status": "APPROVED",
        "bucket": "SOLID",
        "scope_violation": False,
    }
    return freeze_object_with_hash(leg, "phase2_frozen_hash")


class TestTicketRules:
    def test_unique_player_rule_passes_different_players(self):
        legs = [_make_approved_leg("P1", "G1"), _make_approved_leg("P2", "G2")]
        violation = check_unique_player_rule(legs, "STANDARD_SHORT")
        assert violation is None

    def test_unique_player_rule_fails_same_player(self):
        legs = [_make_approved_leg("P1", "G1"), _make_approved_leg("P1", "G2")]
        violation = check_unique_player_rule(legs, "STANDARD_SHORT")
        assert violation is not None
        assert "DUPLICATE_PLAYER" in violation

    def test_unique_player_rule_sgp_allows_same_player(self):
        legs = [_make_approved_leg("P1", "G1"), _make_approved_leg("P1", "G1")]
        violation = check_unique_player_rule(legs, "SGP_SHORT")
        assert violation is None

    def test_same_game_rule_passes_different_games(self):
        legs = [_make_approved_leg("P1", "G1"), _make_approved_leg("P2", "G2")]
        violation = check_same_game_rule(legs, "STANDARD_SHORT")
        assert violation is None

    def test_ticket_ev_computation(self):
        ev = compute_ticket_ev(0.42, 4.0)
        # 0.42 * 4.0 - 1 = 0.68
        assert ev == pytest.approx(0.68)

    def test_ticket_ev_none_if_missing(self):
        assert compute_ticket_ev(None, 4.0) is None
        assert compute_ticket_ev(0.42, None) is None

    def test_ticket_price_product_of_decimals(self):
        legs = [
            _make_approved_leg("P1", "G1", fd_decimal=2.0),
            _make_approved_leg("P2", "G2", fd_decimal=2.5),
        ]
        result = compute_ticket_price(legs, "STANDARD_SHORT")
        assert result["fd_ticket_price_decimal"] == pytest.approx(5.0)
        assert result["ticket_price_integrity_state"] == "DERIVED_BY_APPROVED_RULE"


class TestBuildTicket:
    def test_build_valid_two_leg_ticket(self):
        legs = [
            _make_approved_leg("P1", "G1"),
            _make_approved_leg("P2", "G2"),
        ]
        ticket = build_ticket(legs, "STANDARD_SHORT")
        assert ticket["ticket_integrity_status"] == "PASS"
        assert "phase3_frozen_hash" in ticket
        assert ticket["joint_prob"] is not None
        assert ticket["fd_ticket_price_decimal"] is not None

    def test_build_fails_duplicate_player(self):
        legs = [
            _make_approved_leg("P1", "G1"),
            _make_approved_leg("P1", "G2"),
        ]
        ticket = build_ticket(legs, "STANDARD_SHORT")
        assert ticket["ticket_integrity_status"] == "FAIL"
        assert "DUPLICATE_PLAYER" in ticket["failure_reason"]
