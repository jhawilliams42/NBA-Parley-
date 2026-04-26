"""
End-to-end orchestration for the modular NBA prop engine.

This module stitches the validated phase helpers into a single runnable
portfolio builder. It does not fetch remote data and does not consume API keys;
all runtime inputs are expected to be supplied in the snapshot payload.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from .phase0.constants import PORTFOLIO_TARGET_TICKETS
from .phase0.models import RunContext, SnapshotBundle
from .phase1.pipeline import run_phase1_pipeline
from .phase2.pipeline import run_phase2_pipeline
from .phase2.valuation import process_valuation_book
from .phase3.pipeline import build_portfolio
from .phase4.kelly import run_phase4_pipeline
from .phase5.display import build_presentation_object


@dataclass
class EngineRunArtifacts:
    """Structured output for a full engine run."""

    run_context: dict
    snapshot_bundle: dict
    phase1_objects: list[dict]
    phase2_approved_legs: list[dict]
    phase2_rejected_legs: list[dict]
    phase3_tickets: list[dict]
    phase4_sized_tickets: list[dict]
    portfolio: dict
    api_keys_consumed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_context": copy.deepcopy(self.run_context),
            "snapshot_bundle": copy.deepcopy(self.snapshot_bundle),
            "phase1_objects": copy.deepcopy(self.phase1_objects),
            "phase2_approved_legs": copy.deepcopy(self.phase2_approved_legs),
            "phase2_rejected_legs": copy.deepcopy(self.phase2_rejected_legs),
            "phase3_tickets": copy.deepcopy(self.phase3_tickets),
            "phase4_sized_tickets": copy.deepcopy(self.phase4_sized_tickets),
            "portfolio": copy.deepcopy(self.portfolio),
            "api_keys_consumed": list(self.api_keys_consumed),
        }


def _coerce_run_context(run_context: RunContext | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(run_context, RunContext):
        return run_context.to_dict()
    return RunContext(**dict(run_context)).to_dict()


def _coerce_snapshot_bundle(
    snapshot_bundle: SnapshotBundle | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(snapshot_bundle, SnapshotBundle):
        bundle = snapshot_bundle
    else:
        bundle = SnapshotBundle(**dict(snapshot_bundle))

    if bundle.bundle_status != "VALID" or not bundle.bundle_integrity_hash:
        bundle.freeze()
    if bundle.bundle_status != "VALID":
        raise ValueError(
            f"snapshot_bundle must satisfy atomicity before execution, got {bundle.bundle_status}"
        )
    return bundle.to_dict()


def _normalize_sample_values(values: Sequence[Any] | None) -> list[float]:
    if not values:
        return []
    normalized: list[float] = []
    for value in values:
        if value is None:
            continue
        normalized.append(float(value))
    return normalized


def _default_tip_time(snapshot_bundle: Mapping[str, Any]) -> str:
    frozen_ts = str(snapshot_bundle["bundle_frozen_ts_utc"]).replace("Z", "+00:00")
    base = datetime.fromisoformat(frozen_ts)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    return (base + timedelta(hours=6)).astimezone(timezone.utc).isoformat()


def _prepare_player_game_object(
    raw_object: Mapping[str, Any],
    snapshot_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    obj = copy.deepcopy(dict(raw_object))

    sample_values = _normalize_sample_values(obj.get("sample_values"))
    if sample_values:
        obj["sample_values"] = sample_values
        obj.setdefault("sample_n", len(sample_values))
        obj.setdefault("mean_raw", fmean(sample_values))
        obj.setdefault("std_raw", pstdev(sample_values) if len(sample_values) > 1 else 0.0)

    obj.setdefault("stat_family", obj.get("stat_type"))
    obj.setdefault("fd_current_line", obj.get("line"))
    obj.setdefault("fd_execution_side", "OVER")
    obj.setdefault("fd_prop_market_status", "ACTIVE" if obj.get("fd_current_line") is not None else None)
    obj.setdefault("dependence_method_id", "NONE_INDEPENDENCE_ALLOWED")
    obj.setdefault("same_game_dependence_method_id", "NONE_INDEPENDENCE_ALLOWED")
    obj.setdefault("fd_sgp_supported", False)
    obj.setdefault("snapshot_bundle_status", snapshot_bundle.get("bundle_status"))

    if obj.get("normalized_status") and not obj.get("official_injury_designation"):
        obj["official_injury_designation"] = obj["normalized_status"]

    if obj.get("normalized_status") == "GTD":
        obj.setdefault("tip_time_utc", _default_tip_time(snapshot_bundle))

    lineup_loaded = bool(
        obj.get("lineup_context_loaded")
        or obj.get("starter_rate_last_n") is not None
        or obj.get("substitution_pattern_stability") is not None
        or obj.get("lineup_continuity_score") is not None
    )
    injury_loaded = bool(
        obj.get("injury_status_loaded")
        or obj.get("official_injury_designation") is not None
        or obj.get("active_inactive_designation") is not None
    )
    player_stats_loaded = bool(
        obj.get("player_stats_loaded")
        or obj.get("sample_values")
        or obj.get("mean_raw") is not None
        or obj.get("sample_n") is not None
    )
    fanduel_market_loaded = bool(
        obj.get("fanduel_market_loaded")
        or (
            obj.get("fd_current_line") is not None
            and (
                obj.get("fd_current_odds_american_over") is not None
                or obj.get("fd_current_odds_american_under") is not None
            )
        )
    )

    obj["lineup_context_loaded"] = lineup_loaded
    obj["injury_status_loaded"] = injury_loaded
    obj["player_stats_loaded"] = player_stats_loaded
    obj["fanduel_market_loaded"] = fanduel_market_loaded

    return obj


def _normalize_valuation_books_map(
    valuation_books_map: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    if not valuation_books_map:
        return {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for key, books in valuation_books_map.items():
        processed_books: list[dict[str, Any]] = []
        for book in books:
            book_dict = dict(book)
            if book_dict.get("book_status") in {"SUCCESS", "FALLBACK_SUCCESS", "EXCLUDED"}:
                processed_books.append(copy.deepcopy(book_dict))
                continue
            processed_books.append(
                process_valuation_book(
                    book_name=str(book_dict["book_name"]),
                    odds_over_american=float(book_dict["odds_over_american"]),
                    odds_under_american=float(book_dict["odds_under_american"]),
                    preferred_method=str(book_dict.get("preferred_method", "PROPORTIONAL_V1")),
                )
            )
        normalized[str(key)] = processed_books
    return normalized


def build_nba_prop_portfolio(
    *,
    run_context: RunContext | Mapping[str, Any],
    snapshot_bundle: SnapshotBundle | Mapping[str, Any],
    player_game_objects: Sequence[Mapping[str, Any]],
    valuation_books_map: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    kelly_fraction: float = 1 / 8,
    target_tickets: int = PORTFOLIO_TARGET_TICKETS,
) -> EngineRunArtifacts:
    """
    Execute the full v15.1 modular engine from snapshot inputs.

    All data must already be present in the supplied payload. This function does
    not fetch remote data and does not consume API keys.
    """
    run_context_dict = _coerce_run_context(run_context)
    snapshot_bundle_dict = _coerce_snapshot_bundle(snapshot_bundle)
    normalized_valuation_books = _normalize_valuation_books_map(valuation_books_map)

    prepared_objects = [
        _prepare_player_game_object(raw_object, snapshot_bundle_dict)
        for raw_object in player_game_objects
    ]

    phase1_objects = run_phase1_pipeline(prepared_objects, snapshot_bundle_dict)
    phase2_approved_legs, phase2_rejected_legs = run_phase2_pipeline(
        phase1_objects,
        valuation_books_map=normalized_valuation_books,
    )
    phase3_tickets = build_portfolio(
        phase2_approved_legs,
        target_tickets=target_tickets,
    )
    phase4_sized_tickets = run_phase4_pipeline(
        phase3_tickets,
        run_context_dict,
        kelly_fraction=kelly_fraction,
    )
    portfolio = build_presentation_object(phase4_sized_tickets, run_context_dict)

    return EngineRunArtifacts(
        run_context=run_context_dict,
        snapshot_bundle=snapshot_bundle_dict,
        phase1_objects=phase1_objects,
        phase2_approved_legs=phase2_approved_legs,
        phase2_rejected_legs=phase2_rejected_legs,
        phase3_tickets=phase3_tickets,
        phase4_sized_tickets=phase4_sized_tickets,
        portfolio=portfolio,
        api_keys_consumed=[],
    )
