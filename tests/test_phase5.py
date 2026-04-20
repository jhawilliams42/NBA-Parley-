"""
Tests for Phase 5 — Tiers, Corridors, Portfolio Display.
"""

import pytest

from nba_prop_engine.phase5.display import (
    assign_tier,
    build_corridor_report,
    build_presentation_object,
    check_portfolio_sufficiency,
)
from nba_prop_engine.phase0.constants import PORTFOLIO_TARGET_TICKETS, SCHEMA_VERSION
from nba_prop_engine.phase0.hash_utils import freeze_object_with_hash


def _make_sized_ticket(
    ticket_score=0.12,
    ticket_ev_pct=0.12,
    kelly_stake=150.0,
    joint_prob=0.40,
    fd_price_decimal=4.5,
    family_id="STANDARD_SHORT",
    bankroll=10000.0,
):
    t = {
        "ticket_id": f"TKT_TEST",
        "family_id": family_id,
        "ticket_score": ticket_score,
        "ticket_ev_pct": ticket_ev_pct,
        "kelly_stake": kelly_stake,
        "joint_prob": joint_prob,
        "joint_prob_integrity_status": "PASS",
        "fd_ticket_price_decimal": fd_price_decimal,
        "bankroll_input": bankroll,
        "corridor_fit_class": "STANDARD",
        "leg_ids": [],
    }
    # Add phase3 hash first, then phase4
    t = freeze_object_with_hash(t, "phase3_frozen_hash")
    t = freeze_object_with_hash(t, "phase4_frozen_hash")
    return t


class TestTierAssignment:
    def test_tier_1_high_score(self):
        t = _make_sized_ticket(ticket_score=0.20, ticket_ev_pct=0.20, kelly_stake=100)
        assert assign_tier(t) == "TIER_1"

    def test_tier_2_medium_score(self):
        t = _make_sized_ticket(ticket_score=0.12, ticket_ev_pct=0.12, kelly_stake=100)
        assert assign_tier(t) == "TIER_2"

    def test_tier_3(self):
        t = _make_sized_ticket(ticket_score=0.07, ticket_ev_pct=0.06, kelly_stake=50)
        assert assign_tier(t) == "TIER_3"

    def test_tier_5_zero_stake(self):
        t = _make_sized_ticket(ticket_score=0.01, ticket_ev_pct=0.01, kelly_stake=0)
        assert assign_tier(t) == "TIER_5"


class TestPortfolioSufficiency:
    def test_21_tickets_sufficient(self):
        tickets = [_make_sized_ticket() for _ in range(21)]
        result = check_portfolio_sufficiency(tickets)
        assert result["sufficient"] is True
        assert result["shortfall"] == 0

    def test_fewer_than_21_insufficient(self):
        tickets = [_make_sized_ticket() for _ in range(10)]
        result = check_portfolio_sufficiency(tickets)
        assert result["sufficient"] is False
        assert result["shortfall"] == 11
        assert "INSUFFICIENT" in result["status"]

    def test_empty_portfolio(self):
        result = check_portfolio_sufficiency([])
        assert result["ticket_count"] == 0
        assert result["sufficient"] is False


class TestCorridorReport:
    def test_empty_tickets(self):
        report = build_corridor_report([])
        assert report["payout_corridor"]["min"] is None
        assert report["concentration_corridor"]["met_target"] is False

    def test_payout_corridor_computed(self):
        tickets = [
            _make_sized_ticket(fd_price_decimal=3.0),
            _make_sized_ticket(fd_price_decimal=5.0),
            _make_sized_ticket(fd_price_decimal=4.0),
        ]
        report = build_corridor_report(tickets)
        assert report["payout_corridor"]["min"] == pytest.approx(3.0)
        assert report["payout_corridor"]["max"] == pytest.approx(5.0)
        assert report["payout_corridor"]["mean"] == pytest.approx(4.0)

    def test_risk_corridor_total_stake(self):
        tickets = [_make_sized_ticket(kelly_stake=100) for _ in range(3)]
        report = build_corridor_report(tickets)
        assert report["risk_corridor"]["total_stake"] == pytest.approx(300.0)

    def test_family_distribution(self):
        tickets = [
            _make_sized_ticket(family_id="STANDARD_SHORT"),
            _make_sized_ticket(family_id="STANDARD_SHORT"),
            _make_sized_ticket(family_id="STANDARD_MID"),
        ]
        report = build_corridor_report(tickets)
        dist = report["family_corridor"]["family_distribution"]
        assert dist["STANDARD_SHORT"] == 2
        assert dist["STANDARD_MID"] == 1


class TestBuildPresentationObject:
    def _run_context(self):
        return {
            "run_id": "RUN_001",
            "schema_version": SCHEMA_VERSION,
            "target_date_utc": "2026-03-10",
            "execution_book": "FANDUEL",
            "bankroll_input": 10000.0,
        }

    def test_complete_portfolio(self):
        tickets = [_make_sized_ticket() for _ in range(21)]
        presentation = build_presentation_object(tickets, self._run_context())
        assert presentation["portfolio_status"] == "COMPLETE"
        assert presentation["total_tickets"] == 21
        assert "corridors" in presentation
        assert "tier_summary" in presentation

    def test_incomplete_portfolio_reported_honestly(self):
        tickets = [_make_sized_ticket() for _ in range(5)]
        presentation = build_presentation_object(tickets, self._run_context())
        assert presentation["portfolio_status"] == "INCOMPLETE"
        assert presentation["portfolio_sufficiency"]["shortfall"] == 16

    def test_validity_conditions_include_execution_book(self):
        tickets = [_make_sized_ticket()]
        presentation = build_presentation_object(tickets, self._run_context())
        conditions = presentation["validity_conditions"]
        assert conditions["execution_book"] is True
