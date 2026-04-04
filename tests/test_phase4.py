"""
Tests for Phase 4 — Kelly Sizing.
Code-verified examples per Section 0.24 / Section 4.5.
"""

import pytest

from nba_prop_engine.phase4.kelly import (
    compute_kelly_stake,
    validate_kelly_inputs,
)
from nba_prop_engine.phase0.hash_utils import freeze_object_with_hash


def _base_ticket(
    fd_price_decimal=4.25,
    joint_prob=0.572,
    bankroll=10000.0,
    kelly_cap=0.02,
    ticket_ev_pct=0.064,
    min_op_frac=0.001,
    min_op_dollar=1.0,
):
    t = {
        "ticket_id": "TKT_TEST",
        "fd_ticket_price_decimal": fd_price_decimal,
        "joint_prob": joint_prob,
        "joint_prob_integrity_status": "PASS",
        "bankroll_input": bankroll,
        "bankroll_integrity_state": "VERIFIED",
        "ticket_ev_pct": ticket_ev_pct,
        "kelly_cap_fraction_of_bankroll": kelly_cap,
        "min_operational_fraction": min_op_frac,
        "minimum_operational_dollar_threshold": min_op_dollar,
    }
    return freeze_object_with_hash(t, "phase3_frozen_hash")


class TestKellyValidation:
    def test_valid_inputs_pass(self):
        ticket = _base_ticket()
        is_valid, detail = validate_kelly_inputs(ticket)
        assert is_valid is True
        assert detail["kelly_eligibility_status"] == "ELIGIBLE"
        assert detail["b"] == pytest.approx(3.25)

    def test_missing_joint_prob_fails(self):
        ticket = _base_ticket()
        ticket["joint_prob"] = None
        is_valid, detail = validate_kelly_inputs(ticket)
        assert is_valid is False

    def test_joint_prob_below_epsilon_fails(self):
        ticket = _base_ticket(joint_prob=0.005)
        is_valid, _ = validate_kelly_inputs(ticket)
        assert is_valid is False

    def test_joint_prob_above_epsilon_fails(self):
        ticket = _base_ticket(joint_prob=0.995)
        is_valid, _ = validate_kelly_inputs(ticket)
        assert is_valid is False

    def test_invalid_joint_prob_status_fails(self):
        ticket = _base_ticket()
        ticket["joint_prob_integrity_status"] = "FAIL"
        is_valid, _ = validate_kelly_inputs(ticket)
        assert is_valid is False

    def test_payout_below_epsilon_fails(self):
        # fd_price_decimal = 1.005 → b = 0.005 < EPSILON_B=0.01
        ticket = _base_ticket(fd_price_decimal=1.005)
        is_valid, _ = validate_kelly_inputs(ticket)
        assert is_valid is False

    def test_zero_bankroll_fails(self):
        ticket = _base_ticket(bankroll=0.0)
        is_valid, _ = validate_kelly_inputs(ticket)
        assert is_valid is False


class TestKellyStake:
    def test_code_verified_example_TEST_KELLY_003(self):
        """
        Section 4.5 Code-Verified Example TEST_KELLY_003:
        Input: fd_ticket_price_decimal=4.25, joint_prob=0.572, bankroll=10000
        Expected output: f_star≈0.4403, kelly_stake_dollars=200.0, stake_cap_applied=True
        """
        ticket = _base_ticket(fd_price_decimal=4.25, joint_prob=0.572, bankroll=10000)
        stake_dollars, detail = compute_kelly_stake(ticket, kelly_fraction=1/8)

        assert detail["f_star"] == pytest.approx(0.4403076923, rel=1e-4)
        assert detail["kelly_fractional"] == pytest.approx(0.4403076923 / 8, rel=1e-4)
        assert detail["kelly_fractional_capped"] == pytest.approx(0.02, rel=1e-6)
        assert stake_dollars == pytest.approx(200.0, rel=1e-4)
        assert detail["stake_cap_applied"] is True

    def test_stake_cap_applied_when_kelly_exceeds_cap(self):
        # High prob ticket — Kelly fraction will exceed 2% cap
        ticket = _base_ticket(joint_prob=0.80, fd_price_decimal=3.0, bankroll=10000, kelly_cap=0.02)
        stake, detail = compute_kelly_stake(ticket, kelly_fraction=1/8)
        assert detail["stake_cap_applied"] is True
        assert stake == pytest.approx(200.0, rel=1e-4)

    def test_zero_stake_when_edge_negative(self):
        # joint_prob low relative to break_even: negative Kelly
        ticket = _base_ticket(joint_prob=0.20, fd_price_decimal=2.0, bankroll=10000)
        # b = 1.0, p = 0.20, q = 0.80 → f* = (1*0.2 - 0.8)/1 = -0.6 → clamped to 0
        stake, detail = compute_kelly_stake(ticket)
        assert stake == 0.0
        assert detail["f_star"] == pytest.approx(0.0)

    def test_kelly_exceeds_unity_guard_exists(self):
        # f* > 1.0 is mathematically impossible with valid inputs (p ∈ (0,1), b > 0)
        # because f* = p - (1-p)/b ≤ p < 1. The guard exists as a model-error safety net.
        # Verify that valid high-prob inputs produce f* < 1.0.
        ticket = _base_ticket(joint_prob=0.90, fd_price_decimal=5.0, bankroll=10000)
        stake, detail = compute_kelly_stake(ticket)
        # b = 4.0, p = 0.90 → f* = (4*0.90 - 0.10)/4 = 3.5/4 = 0.875 < 1
        assert detail["f_star"] == pytest.approx(0.875, rel=1e-5)
        assert detail["kelly_eligibility_status"] == "ELIGIBLE"

    def test_minimum_dollar_threshold_zeroes_small_stake(self):
        # Very small edge → stake < min threshold
        ticket = _base_ticket(
            joint_prob=0.51, fd_price_decimal=2.0, bankroll=100,
            kelly_cap=0.001, min_op_dollar=50.0, min_op_frac=1.0
        )
        stake, detail = compute_kelly_stake(ticket)
        assert stake == 0.0

    def test_positive_edge_produces_positive_stake(self):
        ticket = _base_ticket(joint_prob=0.60, fd_price_decimal=2.5)
        stake, detail = compute_kelly_stake(ticket)
        assert stake > 0

    def test_fractional_kelly_quarter(self):
        """Quarter-Kelly should produce half the stake of half-Kelly."""
        ticket_half = _base_ticket()
        _, detail_half = compute_kelly_stake(ticket_half, kelly_fraction=1/2)
        _, detail_quarter = compute_kelly_stake(ticket_half, kelly_fraction=1/4)
        # Uncapped fractions should be half
        assert detail_quarter["kelly_fractional"] == pytest.approx(
            detail_half["kelly_fractional"] / 2, rel=1e-6
        )
