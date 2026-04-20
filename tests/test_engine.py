"""
End-to-end tests for the modular engine orchestrator.
"""

from __future__ import annotations

import pytest

from nba_prop_engine.engine import build_nba_prop_portfolio
from nba_prop_engine.phase0.constants import EXECUTION_BOOK, SCHEMA_VERSION


def _run_context() -> dict:
    return {
        "run_id": "RUN_E2E_001",
        "schema_version": SCHEMA_VERSION,
        "objective_id": "OBJ_E2E",
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


def _snapshot_bundle() -> dict:
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


def _sample_values(seed: int) -> list[float]:
    base = 24 + (seed % 3)
    return [
        base,
        base + 1,
        base + 2,
        base + 3,
        base + 4,
        base + 1,
        base + 2,
        base + 3,
        base + 4,
        base + 2,
        base + 3,
        base + 5,
    ]


def _player_game_object(index: int) -> dict:
    return {
        "player_id": f"P{index:03d}",
        "game_id": f"G{index:03d}",
        "official_injury_designation": "ACTIVE",
        "fd_current_line": 20.5,
        "fd_current_odds_american_over": -110,
        "fd_current_odds_american_under": -110,
        "fd_prop_market_status": "ACTIVE",
        "sample_values": _sample_values(index),
        "stat_family": "points",
        "usage_rate": 0.31,
        "field_goal_attempts": 18,
        "touches": 66,
        "drives": 12,
        "starter_rate_last_n": 1.0,
        "substitution_pattern_stability": 0.86,
        "lineup_continuity_score": 0.83,
        "projected_minutes": 34.0,
        "tip_time_utc": "2026-03-10T20:00:00Z",
        "restriction_severity_class": "LOW",
        "fd_sgp_supported": False,
    }


def _valuation_books_map(player_ids: list[str]) -> dict[str, list[dict]]:
    return {
        player_id: [
            {"book_name": "DraftKings", "odds_over_american": -125, "odds_under_american": 105},
            {"book_name": "BetMGM", "odds_over_american": -125, "odds_under_american": 105},
            {"book_name": "Caesars", "odds_over_american": -125, "odds_under_american": 105},
        ]
        for player_id in player_ids
    }


class TestEngineOrchestration:
    def test_build_nba_prop_portfolio_complete_run(self):
        raw_objects = [_player_game_object(i) for i in range(1, 22)]
        result = build_nba_prop_portfolio(
            run_context=_run_context(),
            snapshot_bundle=_snapshot_bundle(),
            player_game_objects=raw_objects,
            valuation_books_map=_valuation_books_map([obj["player_id"] for obj in raw_objects]),
        )

        assert result.api_keys_consumed == []
        assert len(result.phase1_objects) == 21
        assert len(result.phase2_approved_legs) == 21
        assert result.portfolio["portfolio_status"] == "COMPLETE"
        assert result.portfolio["total_tickets"] == 21

        first_leg = result.phase2_approved_legs[0]
        assert first_leg["valuation_books"][0]["devig_method"] == "PROPORTIONAL_V1"
        assert first_leg["leg_approval_status"] == "APPROVED"

    def test_stale_snapshot_bundle_raises(self):
        stale_bundle = _snapshot_bundle()
        stale_bundle["valuation_market_snapshot_ts_utc"] = "2026-03-10T14:45:00Z"

        with pytest.raises(ValueError, match="snapshot_bundle"):
            build_nba_prop_portfolio(
                run_context=_run_context(),
                snapshot_bundle=stale_bundle,
                player_game_objects=[_player_game_object(1)],
                valuation_books_map=_valuation_books_map(["P001"]),
            )
