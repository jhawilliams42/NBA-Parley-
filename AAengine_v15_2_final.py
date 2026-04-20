from __future__ import annotations

import copy
import math
import uuid
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# SHARED CONSTITUTIONAL CONSTANTS
#
# These values govern execution integrity across Phase 2 and Phase 5.
# Both phases must use these identically — they describe the same
# physical boundary (how far a FanDuel line can move before the
# engine's edge is invalidated). Any tuning must be done here;
# never in per-phase ADCC blocks, never as literals in method bodies.
# ============================================================

class EngineConstitution:
    """
    Single source of truth for constitutional thresholds that span
    more than one phase. Phases reference these via EngineConstitution.*
    rather than duplicating values in their own ADCC or hardcoding literals.
    """
    # Maximum seconds between a valuation snapshot and the bundle freeze
    # before the valuation is treated as stale.
    MAX_EXECUTION_STALENESS_SEC: int = 300

    # Maximum absolute difference between the Phase 2 valuation consensus
    # and the live FanDuel execution line before a leg is flagged as
    # contaminated. Applied identically in Phase 2 (firewall) and Phase 5
    # (terminal drift check).
    EXECUTION_CONTAMINATION_THRESHOLD: float = 0.5


# ============================================================
# CORE DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class NodeKey:
    phase: str
    field: str
    entity_id: str
    context_id: str


@dataclass
class StagedOutput:
    value: Any
    eligible: bool = True
    code: str = "OK"
    integrity_state: str = "DERIVED_BY_APPROVED_RULE"
    source_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": copy.deepcopy(self.value),
            "eligible": self.eligible,
            "code": self.code,
            "integrity_state": self.integrity_state,
            "source_ids": list(self.source_ids),
        }


@dataclass
class RecomputeRequest:
    requester: str
    target_node: NodeKey
    reason: str
    scope_args: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RepairDecision:
    repairable: bool
    reason: str
    repair_path: List[NodeKey] = field(default_factory=list)


class RepairBudget:
    def __init__(self, max_per_node: int = 3, max_per_run: int = 50) -> None:
        self.max_per_node = max_per_node
        self.max_per_run = max_per_run
        self.run_total = 0
        self.node_counts: Dict[NodeKey, int] = defaultdict(int)

    def check_and_consume(self, target_node: NodeKey) -> bool:
        if self.run_total >= self.max_per_run:
            return False
        if self.node_counts[target_node] >= self.max_per_node:
            return False
        self.run_total += 1
        self.node_counts[target_node] += 1
        return True

    def reset(self) -> None:
        self.run_total = 0
        self.node_counts.clear()


class RecomputeGuard:
    def __init__(self, max_depth: int = 6, max_iterations: int = 50) -> None:
        self.computing_locks: set[NodeKey] = set()
        self.active_chain: List[NodeKey] = []
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.is_active = False

    def check_and_lock(self, node: NodeKey) -> None:
        if self.iteration_count >= self.max_iterations:
            raise RuntimeError("RECOMPUTE_HALT: max session iterations breached")
        if len(self.active_chain) >= self.max_depth:
            raise RuntimeError("RECOMPUTE_HALT: max dependency depth breached")
        if node in self.computing_locks:
            raise RuntimeError(f"CYCLE_VIOLATION: {node}")
        self.computing_locks.add(node)
        self.active_chain.append(node)
        self.iteration_count += 1

    def release(self, node: NodeKey) -> None:
        self.computing_locks.discard(node)
        if self.active_chain and self.active_chain[-1] == node:
            self.active_chain.pop()


class RecomputeSession:
    def __init__(self, guard: RecomputeGuard) -> None:
        self.guard = guard
        self.is_top_level = not guard.is_active

    def __enter__(self) -> RecomputeGuard:
        if self.is_top_level:
            self.guard.is_active = True
            self.guard.active_chain = []
            self.guard.iteration_count = 0
            self.guard.computing_locks.clear()
        return self.guard

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.is_top_level:
            self.guard.is_active = False
            self.guard.active_chain = []
            self.guard.computing_locks.clear()


# ============================================================
# PROVIDER INTERFACE
# ============================================================

class PregameSourceProvider(ABC):
    """
    Autonomous pregame-only provider.

    This interface is the only lawful runtime input surface.
    No phase may accept human-fed market data, logs, rosters, or lines directly.
    """

    @abstractmethod
    def get_run_context(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_games(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_projected_rosters(self, game_id: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_player_stat_logs(self, player_id: str, stat_type: str, game_id: str) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def get_player_context(self, player_id: str, game_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_valuation_books(self, leg_id: str) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_execution_line(self, leg_id: str) -> Dict[str, Any]:
        raise NotImplementedError


class DictPregameSnapshotProvider(PregameSourceProvider):
    """
    Concrete provider for machine-produced pregame snapshots.

    Expected snapshot structure:
    {
      "run_context": {...},
      "games": [{"game_id":..., "slate_id":..., "series_id":..., "playoff_context": bool, ...}],
      "rosters": {game_id: [player_dict, ...]},
      "player_context": {game_id: {player_id: {...}}},
      "player_stat_logs": {game_id: {player_id: {stat_type: [..]}}},
      "valuation_books": {leg_id: {"DK": {...}, "MGM": {...}, "COVERS": {...}}},
      "execution_lines": {leg_id: {...}}
    }
    """

    def __init__(self, snapshot: Dict[str, Any]) -> None:
        self.snapshot = snapshot

    def get_run_context(self) -> Dict[str, Any]:
        return copy.deepcopy(self.snapshot.get("run_context", {}))

    def list_games(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.snapshot.get("games", []))

    def get_projected_rosters(self, game_id: str) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.snapshot.get("rosters", {}).get(game_id, []))

    def get_player_stat_logs(self, player_id: str, stat_type: str, game_id: str) -> List[float]:
        return copy.deepcopy(
            self.snapshot.get("player_stat_logs", {})
            .get(game_id, {})
            .get(player_id, {})
            .get(stat_type, [])
        )

    def get_player_context(self, player_id: str, game_id: str) -> Dict[str, Any]:
        return copy.deepcopy(
            self.snapshot.get("player_context", {})
            .get(game_id, {})
            .get(player_id, {})
        )

    def get_valuation_books(self, leg_id: str) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self.snapshot.get("valuation_books", {}).get(leg_id, {}))

    def get_execution_line(self, leg_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self.snapshot.get("execution_lines", {}).get(leg_id, {}))


# ============================================================
# FACADE & ENGINE STATE
# ============================================================

class PhaseAccessFacade:
    def __init__(self, engine: "NBAPropEngine") -> None:
        self._engine = engine

    def read_state(self, node_key: NodeKey) -> StagedOutput:
        return self._engine.read_state(node_key)

    def request_recompute(self, requester: str, target_node: NodeKey, reason: str, scope_args: Optional[Dict[str, Any]] = None) -> bool:
        return self._engine.request_recompute(requester, target_node, reason, scope_args or {})

    def repair_or_halt(self, target_node: NodeKey, requester: str, scope_args: Optional[Dict[str, Any]] = None) -> bool:
        return self._engine.repair_or_halt(target_node, requester, scope_args or {})

    def write_sealed_output(self, node_key: NodeKey, value: StagedOutput, integrity_state: Optional[str] = None) -> None:
        if integrity_state is not None:
            value.integrity_state = integrity_state
        self._engine.write_initial_state(node_key, value, value.integrity_state)

    @property
    def provider(self) -> PregameSourceProvider:
        return self._engine.provider


class EngineState:
    def __init__(self) -> None:
        self.sealed_outputs: Dict[NodeKey, StagedOutput] = {}
        self.integrity_ledger: Dict[NodeKey, str] = {}
        self.runtime_status_ledger: Dict[NodeKey, str] = {}
        self.freeze_flags: Dict[Any, bool] = {}
        self.audit_log: List[Dict[str, Any]] = []

    def log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        self.audit_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": action_type,
                "details": copy.deepcopy(details),
            }
        )


# ============================================================
# PHASE 0
# ============================================================

class NBAPropEnginePhase0:
    VERSION = "v15.2-PREGAME-AUTONOMOUS"

    DEFAULT_PAYOUT_BANDS = {
        "BAND_1": {"min": 60.0, "max": 120.0, "ticket_count": 6},
        "BAND_2": {"min": 121.0, "max": 250.0, "ticket_count": 5},
        "BAND_3": {"min": 251.0, "max": 400.0, "ticket_count": 4},
        "BAND_4": {"min": 401.0, "max": 700.0, "ticket_count": 4},
        "BAND_5": {"min": 701.0, "max": 1000.0, "ticket_count": 2},
    }

    APPROVED_STAT_FAMILIES = {
        "points",
        "rebounds",
        "assists",
        "threes_made",
        "steals",
        "blocks",
        "PR",
        "PA",
        "RA",
        "PRA",
        "stocks",
        "pts_threes",
    }

    SOURCE_REGISTRY: Dict[int, Dict[str, Any]] = {
        1: {"name": "FD Public", "class": "EXECUTION_BINDING", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_2_VALUATION_EXTRACTION", "PHASE_3_CONSUMPTION_ONLY", "PHASE_4_CONSUMPTION_ONLY", "PHASE_5_PRESENTATION_ONLY"]},
        2: {"name": "FD SGP", "class": "EXECUTION_BINDING", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_2_VALUATION_EXTRACTION", "PHASE_3_CONSUMPTION_ONLY", "PHASE_4_CONSUMPTION_ONLY", "PHASE_5_PRESENTATION_ONLY"]},
        3: {"name": "FD Snapshot", "class": "EXECUTION_BINDING", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_2_VALUATION_EXTRACTION", "PHASE_3_CONSUMPTION_ONLY", "PHASE_4_CONSUMPTION_ONLY", "PHASE_5_PRESENTATION_ONLY"]},
        4: {"name": "Official Injury", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_1_PRIMARY_EXTRACTION"]},
        5: {"name": "Official Schedule", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_1_PRIMARY_EXTRACTION"]},
        6: {"name": "Official Stats", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_1_PRIMARY_EXTRACTION"]},
        7: {"name": "Official Box Score", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        8: {"name": "Official Advanced", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        9: {"name": "Official Lineups", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        10: {"name": "Lineup Advanced", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        11: {"name": "Four Factors", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        12: {"name": "Lineup Opponent", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        13: {"name": "Hustle Stats", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        14: {"name": "Box-Out Stats", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        15: {"name": "Opponent Shooting", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        16: {"name": "Referee Assign", "class": "OFFICIAL_PRIMARY", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_1_PRIMARY_EXTRACTION"]},
        17: {"name": "ESPN Verification", "class": "APPROVED_SECONDARY_VERIFICATION", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_1_SECONDARY_VERIFICATION"]},
        18: {"name": "CBS Verification", "class": "APPROVED_SECONDARY_VERIFICATION", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_1_SECONDARY_VERIFICATION"]},
        19: {"name": "DK Markets", "class": "APPROVED_EXTERNAL_VALUATION", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_2_VALUATION_EXTRACTION"]},
        20: {"name": "MGM Markets", "class": "APPROVED_EXTERNAL_VALUATION", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_2_VALUATION_EXTRACTION"]},
        21: {"name": "Caesars Markets", "class": "APPROVED_EXTERNAL_VALUATION", "connection_state": "DIRECT_VALUATION_CONNECTED", "usability_policy": "LEGACY_DECLARED_NOT_VERIFIED_USABLE_IN_THIS_BUILD", "phase_use": ["PHASE_2_VALUATION_EXTRACTION"]},
        22: {"name": "Future Reserved", "class": "FUTURE_RESERVED_SOURCE", "phase_use": ["PHASE_0_GOVERNANCE_ONLY"]},
        23: {"name": "B-Ref Advanced", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        24: {"name": "B-Ref PBP", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        25: {"name": "NBA.com Passing", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        26: {"name": "NBA.com Touches", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        27: {"name": "NBA.com Drives", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        28: {"name": "NBA.com Defender", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        29: {"name": "NBA.com Navigation", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        30: {"name": "Covers Odds", "class": "APPROVED_EXTERNAL_VALUATION", "phase_use": ["PHASE_0_GOVERNANCE_ONLY", "PHASE_2_VALUATION_EXTRACTION"]},
        31: {"name": "Elevation API", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
        32: {"name": "USGS Mapping", "class": "APPROVED_DERIVATION_SUPPORT", "phase_use": ["PHASE_1_PRIMARY_EXTRACTION"]},
    }

    def __init__(self, facade: PhaseAccessFacade) -> None:
        self.facade = facade

    def initialize_run(self) -> StagedOutput:
        run_context = self.facade.provider.get_run_context()
        frozen_ts = run_context.get("bundle_frozen_ts_utc") or datetime.now(timezone.utc).isoformat()
        family_p = {
            "snapshot_bundle_id": run_context.get("snapshot_bundle_id", str(uuid.uuid4())),
            "bundle_frozen_ts_utc": frozen_ts,
            "nba_status_snapshot_ts_utc": run_context.get("nba_status_snapshot_ts_utc", frozen_ts),
            "nba_schedule_snapshot_ts_utc": run_context.get("nba_schedule_snapshot_ts_utc", frozen_ts),
            "nba_stats_snapshot_ts_utc": run_context.get("nba_stats_snapshot_ts_utc", frozen_ts),
            "team_context_snapshot_ts_utc": run_context.get("team_context_snapshot_ts_utc", frozen_ts),
            "referee_snapshot_ts_utc": run_context.get("referee_snapshot_ts_utc", frozen_ts),
            "valuation_market_snapshot_ts_utc": run_context.get("valuation_market_snapshot_ts_utc", frozen_ts),
            "max_component_time_delta_seconds": run_context.get("max_component_time_delta_seconds", 0),
            "snapshot_sequence_log": list(run_context.get("snapshot_sequence_log", ["INIT", "PREGAME_ONLY", "FAMILY_P_SEEDED"])),
            "version": self.VERSION,
            "playoff_context_mode": bool(run_context.get("playoff_context_mode", False)),
            "overs_only": True,
            "approved_stat_families": sorted(self.APPROVED_STAT_FAMILIES),
            "ticket_count": 21,
            "fixed_stake": 1.0,
            "payout_bands": copy.deepcopy(run_context.get("payout_bands", self.DEFAULT_PAYOUT_BANDS)),
            "payout_band_basis": run_context.get("payout_band_basis", "gross_payout"),
            "source_registry": copy.deepcopy(self.SOURCE_REGISTRY),
            "games": self.facade.provider.list_games(),
            "pregame_only": True,
        }
        node = NodeKey("Phase0", "Family_P", "global", "run")
        out = StagedOutput(family_p, eligible=True, code="FAMILY_P_SEEDED", integrity_state="VERIFIED", source_ids=[4, 5, 6, 16, 19, 20, 30])
        self.facade.write_sealed_output(node, out)
        return out


# ============================================================
# PHASE 1
# ============================================================

class NBAPropEnginePhase1:
    ADCC = {
        "V_P_REGULAR": 1.00,
        "V_P_PLAYOFF": 0.94,
        "MEAN_MINUTES_FLOOR": 8.0,
        "SOURCE_VARIANCE_THRESHOLD": 0.05,
        "FRAGILITY_BASE": 0.10,
        "PLAYOFF_PREV_GAME_WEIGHT": 2.0,
        "RECENT_GAME_DECAY": 0.85,
    }

    def __init__(self, facade: PhaseAccessFacade) -> None:
        self.facade = facade

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> Optional[float]:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @staticmethod
    def _safe_variance(values: Sequence[float], mean_val: float) -> float:
        vals = [float(v) for v in values if v is not None]
        if not vals:
            return 0.0
        return sum((v - mean_val) ** 2 for v in vals) / len(vals)

    def normalize_status(self, raw_status: str) -> str:
        raw = str(raw_status or "").strip().lower()
        if raw in {"available", "active", "probable"}:
            return "ACTIVE"
        if raw in {"gtd", "game-time decision"}:
            return "GTD"
        if raw == "questionable":
            return "QUESTIONABLE"
        if raw == "doubtful":
            return "DOUBTFUL"
        if raw == "out":
            return "OUT"
        if raw in {"inactive", "inactive_other"}:
            return "INACTIVE_OTHER"
        return "UNRESOLVABLE"

    def _weighted_mean(self, logs: Sequence[float], playoff_context: bool) -> Optional[float]:
        vals = [float(v) for v in logs if v is not None]
        if not vals:
            return None
        weights: List[float] = []
        for idx, _ in enumerate(vals):
            if idx == 0 and playoff_context:
                weights.append(self.ADCC["PLAYOFF_PREV_GAME_WEIGHT"])
            else:
                weights.append(self.ADCC["RECENT_GAME_DECAY"] ** idx)
        denom = sum(weights)
        return sum(v * w for v, w in zip(vals, weights)) / denom if denom else None

    def derive_leg_foundation(self, leg_meta: Dict[str, Any], game_meta: Dict[str, Any]) -> None:
        player_id = leg_meta["player_id"]
        leg_id = leg_meta["leg_id"]
        game_id = leg_meta["game_id"]
        stat_type = leg_meta["stat_type"]
        context_id = game_meta["slate_id"]

        player_ctx = self.facade.provider.get_player_context(player_id, game_id)
        status = self.normalize_status(player_ctx.get("status"))
        projected_minutes = float(player_ctx.get("projected_minutes", 0.0) or 0.0)
        playoff_context = bool(game_meta.get("playoff_context"))
        pace_seed = float(player_ctx.get("pace_seed", 96.0) or 96.0)
        team_orb_rate = float(player_ctx.get("team_orb_rate", 0.25) or 0.25)
        opp_orb_rate = float(player_ctx.get("opp_orb_rate", 0.25) or 0.25)

        status_node = NodeKey("Phase1", "normalized_status", leg_id, context_id)
        self.facade.write_sealed_output(
            status_node,
            StagedOutput(status, eligible=status not in {"OUT", "INACTIVE_OTHER", "UNRESOLVABLE"}, code=status, integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[4, 17, 18]),
        )

        poss_node = NodeKey("Phase1", "Poss_max", leg_id, context_id)
        poss_max = pace_seed * (self.ADCC["V_P_PLAYOFF"] if playoff_context else self.ADCC["V_P_REGULAR"]) * (1 - opp_orb_rate + team_orb_rate)
        self.facade.write_sealed_output(
            poss_node,
            StagedOutput(poss_max, eligible=projected_minutes >= self.ADCC["MEAN_MINUTES_FLOOR"], code="POSS_READY", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[5, 6, 23, 29]),
        )

        logs = self.facade.provider.get_player_stat_logs(player_id, stat_type, game_id)
        weighted_mean = self._weighted_mean(logs, playoff_context)
        sample_n = len(logs)

        family_m_node = NodeKey("Phase1", "Family_M", leg_id, context_id)
        if status in {"OUT", "INACTIVE_OTHER", "UNRESOLVABLE"} or sample_n == 0 or weighted_mean is None:
            family_m = {
                "minutes_fragility_class": "UNRESOLVABLE",
                "fragility_score": None,
                "fragility_integrity_status": "DERIVED_FROM_MISSING_INPUT",
                "epistemic_disqualifier": True,
                "unknown_component_count": 1,
                "repeatability_class": "UNKNOWN",
                "sample_n": sample_n,
                "mean_raw": None,
                "std_raw": None,
                "coefficient_of_variation": None,
                "raw_event_prob": None,
                "raw_event_prob_over_current_line": None,
                "distribution_candidate": None,
            }
            self.facade.write_sealed_output(
                family_m_node,
                StagedOutput(family_m, eligible=False, code="EPISTEMIC_DISQUALIFICATION", integrity_state="DERIVED_FROM_MISSING_INPUT", source_ids=[6, 7, 8, 23]),
            )
            lambda_node = NodeKey("Phase1", "lambda_fga", leg_id, context_id)
            self.facade.write_sealed_output(
                lambda_node,
                StagedOutput(None, eligible=False, code="NO_FOUNDATION", integrity_state="DEPENDENCY_FAILURE", source_ids=[6, 7, 8, 23]),
            )
            return

        variance = self._safe_variance(logs, weighted_mean)
        std_raw = math.sqrt(variance)
        cv = std_raw / weighted_mean if weighted_mean > 0 else 0.0
        if sample_n < 5:
            repeatability = "LOW"
        elif cv < 0.20:
            repeatability = "HIGH"
        else:
            repeatability = "MODERATE"

        if projected_minutes >= 32 and cv < 0.20:
            minutes_fragility = "LOW"
        elif projected_minutes >= 20:
            minutes_fragility = "MODERATE"
        elif projected_minutes >= self.ADCC["MEAN_MINUTES_FLOOR"]:
            minutes_fragility = "HIGH_UNCERTAINTY"
        else:
            minutes_fragility = "UNRESOLVABLE"

        current_line = float(leg_meta.get("line", 0.0) or 0.0)
        k = math.ceil(current_line)
        cdf = sum((weighted_mean ** i) * math.exp(-weighted_mean) / math.factorial(i) for i in range(k)) if k > 0 else 0.0
        raw_prob_line = max(0.0, min(1.0, 1 - cdf))
        raw_prob_generic = max(0.0, min(1.0, 1 - math.exp(-weighted_mean)))
        fragility = self.ADCC["FRAGILITY_BASE"] + (cv * 0.5)

        family_m = {
            "minutes_fragility_class": minutes_fragility,
            "fragility_score": fragility,
            "fragility_integrity_status": "VERIFIED" if sample_n >= 5 else "DERIVED_FROM_MISSING_INPUT",
            "epistemic_disqualifier": False,
            "unknown_component_count": 0,
            "repeatability_class": repeatability,
            "sample_n": sample_n,
            "mean_raw": weighted_mean,
            "std_raw": std_raw,
            "coefficient_of_variation": cv,
            "raw_event_prob": raw_prob_generic,
            "raw_event_prob_over_current_line": raw_prob_line,
            "distribution_candidate": "POISSON",
        }
        self.facade.write_sealed_output(
            family_m_node,
            StagedOutput(family_m, eligible=True, code="FAMILY_M_READY", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[6, 7, 8, 23]),
        )

        # Legacy node name retained for compatibility. For non-points markets this is the expected stat mean, not literal FGA.
        lambda_node = NodeKey("Phase1", "lambda_fga", leg_id, context_id)
        self.facade.write_sealed_output(
            lambda_node,
            StagedOutput(weighted_mean, eligible=True, code="LAMBDA_READY", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[6, 7, 8, 23]),
        )

    def recompute_lambda_fga(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        family_m_node = NodeKey("Phase1", "Family_M", node_key.entity_id, node_key.context_id)
        family_m_entry = self.facade.read_state(family_m_node)
        family_m = family_m_entry.value
        mean_raw = family_m.get("mean_raw")
        if mean_raw is None:
            return StagedOutput(None, eligible=False, code="NO_MEAN_RAW", integrity_state="DEPENDENCY_FAILURE"), "DEPENDENCY_FAILURE"
        usage_shift_modifier = float(kwargs.get("usage_shift_modifier", 1.0) or 1.0)
        if usage_shift_modifier is True:
            usage_shift_modifier = 1.08
        value = mean_raw * usage_shift_modifier
        return StagedOutput(value, eligible=True, code="RECOMPUTED_LAMBDA", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=family_m_entry.source_ids), "DERIVED_BY_APPROVED_RULE"

    def recompute_family_m(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        leg_meta = kwargs["leg_meta"]
        game_meta = kwargs["game_meta"]
        self.derive_leg_foundation(leg_meta, game_meta)
        refreshed = self.facade.read_state(node_key)
        return refreshed, refreshed.integrity_state


# ============================================================
# PHASE 2
# ============================================================

class NBAPropEnginePhase2:
    ADCC = {
        "CONTAMINATION_THRESHOLD": EngineConstitution.EXECUTION_CONTAMINATION_THRESHOLD,
        "PPS_EFFICIENCY_CEILING": 1.45,
        "FRAGILITY_PUNISHMENT_MULT": 1.15,
        "MARKET_WEIGHT": 0.60,
        "MAX_EXECUTION_STALENESS_SEC": EngineConstitution.MAX_EXECUTION_STALENESS_SEC,
        "MAX_DISAGREEMENT_TOLERANCE": 1.0,
    }

    APPROVED_STAT_FAMILIES = set(NBAPropEnginePhase0.APPROVED_STAT_FAMILIES)

    def __init__(self, facade: PhaseAccessFacade) -> None:
        self.facade = facade

    @staticmethod
    def _decimal_to_implied_prob(decimal_odds: float) -> Optional[float]:
        if decimal_odds and decimal_odds > 1.0:
            return 1.0 / decimal_odds
        return None

    @staticmethod
    def _parse_ts(ts: Optional[str]) -> Optional[datetime]:
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return None

    def _compute_stale_flag(
        self,
        valuation_ts_utc: Optional[str],
        bundle_ts_utc: Optional[str],
        current_line: Optional[float],
        open_line: Optional[float],
        max_age_seconds: int = 300,
        max_static_seconds: int = 300,
    ) -> bool:
        """
        A valuation snapshot is stale if:
        (a) either timestamp is missing or unparseable, or
        (b) the absolute delta between valuation and bundle timestamps exceeds max_age_seconds, or
        (c) the line has not moved at all AND the age exceeds max_static_seconds.
        Condition (c) catches markets that froze before they were discovered by the engine.
        """
        v_ts = self._parse_ts(valuation_ts_utc)
        b_ts = self._parse_ts(bundle_ts_utc)
        if v_ts is None or b_ts is None:
            return True
        age = abs((v_ts - b_ts).total_seconds())
        no_line_change = (
            current_line is not None
            and open_line is not None
            and current_line == open_line
        )
        if age > max_age_seconds:
            return True
        if no_line_change and age > max_static_seconds:
            return True
        return False

    def derive_family_n(self, leg_meta: Dict[str, Any], context_id: str) -> StagedOutput:
        leg_id = leg_meta["leg_id"]
        node = NodeKey("Phase2", "Family_N", leg_id, context_id)
        books = self.facade.provider.get_valuation_books(leg_id)
        ordered_sources = [("DK", 19), ("MGM", 20), ("COVERS", 30)]

        valid_lines: List[float] = []
        valid_prices: List[float] = []
        open_lines: List[float] = []
        open_prices: List[float] = []
        fair_probs: List[float] = []
        source_tracking: List[int] = []
        book_audits: Dict[str, Dict[str, Any]] = {}

        for book_name, source_id in ordered_sources:
            payload = books.get(book_name, {})
            line = payload.get("line")
            over_dec = payload.get("over_dec")
            under_dec = payload.get("under_dec")
            if line is None or over_dec is None or under_dec is None:
                book_audits[book_name] = {"book_status": "EXCLUDED", "exclusion_reason": "MISSING_DATA", "de_vig_method": "NONE"}
                continue
            over_prob = self._decimal_to_implied_prob(float(over_dec))
            under_prob = self._decimal_to_implied_prob(float(under_dec))
            if over_prob is None or under_prob is None:
                book_audits[book_name] = {"book_status": "EXCLUDED", "exclusion_reason": "INVALID_ODDS", "de_vig_method": "NONE"}
                continue
            hold = over_prob + under_prob
            fair_prob = over_prob / hold if hold > 0 else None
            if fair_prob is None:
                book_audits[book_name] = {"book_status": "EXCLUDED", "exclusion_reason": "INVALID_HOLD", "de_vig_method": "NONE"}
                continue
            valid_lines.append(float(line))
            valid_prices.append(float(over_dec))
            fair_probs.append(fair_prob)
            if payload.get("open_line") is not None:
                open_lines.append(float(payload["open_line"]))
            if payload.get("open_over_dec") is not None:
                open_prices.append(float(payload["open_over_dec"]))
            source_tracking.append(source_id)
            book_audits[book_name] = {
                "book_status": "SUCCESS",
                "exclusion_reason": "NONE",
                "de_vig_method": "MULTIPLICATIVE",
                "market_width": max(0.0, hold - 1.0),
            }

        if not valid_lines:
            out = StagedOutput({}, eligible=False, code="VALUATION_MISSING", integrity_state="DEPENDENCY_FAILURE", source_ids=[])
            self.facade.write_sealed_output(node, out)
            return out

        counts = Counter(valid_lines)
        mode_line, mode_count = counts.most_common(1)[0]
        if mode_count > 1:
            consensus_line = mode_line
        elif len(valid_lines) == 3:
            consensus_line = sorted(valid_lines)[1]
        else:
            consensus_line = valid_lines[0]

        current_price_consensus = sum(valid_prices) / len(valid_prices)
        open_line_consensus = sum(open_lines) / len(open_lines) if open_lines else consensus_line
        open_price_consensus = sum(open_prices) / len(open_prices) if open_prices else current_price_consensus
        fair_p = sum(fair_probs) / len(fair_probs)
        disagreement = max(valid_lines) - min(valid_lines) if len(valid_lines) > 1 else 0.0
        avg_width = sum(v.get("market_width", 0.0) for v in book_audits.values() if v.get("book_status") == "SUCCESS") / len(valid_lines)
        valuation_ts_utc = datetime.now(timezone.utc).isoformat()
        try:
            family_p_val = self.facade.read_state(NodeKey("Phase0", "Family_P", "global", "run")).value
            bundle_ts_utc: Optional[str] = family_p_val.get("bundle_frozen_ts_utc")
        except (KeyError, ValueError):
            bundle_ts_utc = None
        stale_flag = self._compute_stale_flag(
            valuation_ts_utc=valuation_ts_utc,
            bundle_ts_utc=bundle_ts_utc,
            current_line=consensus_line,
            open_line=open_line_consensus,
            max_age_seconds=self.ADCC["MAX_EXECUTION_STALENESS_SEC"],
            max_static_seconds=self.ADCC["MAX_EXECUTION_STALENESS_SEC"],
        )
        inflation_flag = consensus_line > open_line_consensus
        market_drift_flag = abs(current_price_consensus - open_price_consensus) > 0.05
        off_market_flag = disagreement > self.ADCC["MAX_DISAGREEMENT_TOLERANCE"]
        if disagreement == 0:
            quality = "HIGH"
        elif disagreement <= 0.5:
            quality = "MODERATE"
        else:
            quality = "LOW"

        family_n = {
            "val_open_line_consensus": open_line_consensus,
            "val_current_line_consensus": consensus_line,
            "val_open_price_consensus": open_price_consensus,
            "val_current_price_consensus": current_price_consensus,
            "val_fair_p": fair_p,
            "val_disagreement_score": disagreement,
            "val_stale_flag": stale_flag,
            "val_inflation_flag": inflation_flag,
            "val_market_width": avg_width,
            "val_market_drift_flag": market_drift_flag,
            "val_off_market_flag": off_market_flag,
            "val_consensus_quality_class": quality,
            "val_book_count": len(valid_lines),
            "val_source_set_id": source_tracking,
            "val_snapshot_ts_utc": valuation_ts_utc,
            "book_audits": book_audits,
        }
        out = StagedOutput(family_n, eligible=True, code="FAMILY_N_READY", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=source_tracking)
        self.facade.write_sealed_output(node, out)
        return out

    def execute_efficiency_audit(self, leg_id: str, context_id: str) -> StagedOutput:
        node = NodeKey("Phase2", "pps_audit", leg_id, context_id)
        lambda_entry = self.facade.read_state(NodeKey("Phase1", "lambda_fga", leg_id, context_id))
        family_n_entry = self.facade.read_state(NodeKey("Phase2", "Family_N", leg_id, context_id))
        line = family_n_entry.value.get("val_current_line_consensus")
        expected_mean = lambda_entry.value
        if line is None or expected_mean in (None, 0):
            out = StagedOutput(None, eligible=False, code="MISSING_CONSENSUS_OR_EXPECTATION", integrity_state="DEPENDENCY_FAILURE")
            self.facade.write_sealed_output(node, out)
            return out
        ratio = float(line) / float(expected_mean)
        eligible = ratio <= self.ADCC["PPS_EFFICIENCY_CEILING"]
        code = "OK" if eligible else "MATHEMATICALLY_IMPROBABLE_EFFICIENCY"
        out = StagedOutput(ratio, eligible=eligible, code=code, integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[19, 20, 30])
        self.facade.write_sealed_output(node, out)
        return out

    def enforce_execution_firewall(self, leg_id: str, context_id: str) -> StagedOutput:
        node = NodeKey("Phase2", "fd_line_status", leg_id, context_id)
        family_p = self.facade.read_state(NodeKey("Phase0", "Family_P", "global", "run")).value
        family_n = self.facade.read_state(NodeKey("Phase2", "Family_N", leg_id, context_id)).value
        execution = self.facade.provider.get_execution_line(leg_id)
        current_line = execution.get("current_line")
        ts = execution.get("timestamp_utc")
        if current_line is None or ts is None:
            out = StagedOutput(None, eligible=False, code="MISSING_EXECUTION_LINE", integrity_state="DEPENDENCY_FAILURE", source_ids=[1, 2, 3])
            self.facade.write_sealed_output(node, out)
            return out
        frozen_ts = datetime.fromisoformat(family_p["bundle_frozen_ts_utc"])
        ex_ts = datetime.fromisoformat(ts)
        delta = abs((ex_ts - frozen_ts).total_seconds())
        if delta > self.ADCC["MAX_EXECUTION_STALENESS_SEC"]:
            out = StagedOutput(current_line, eligible=False, code="STALE_EXECUTION_LINE", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[1, 2, 3])
            self.facade.write_sealed_output(node, out)
            return out
        consensus_line = family_n.get("val_current_line_consensus")
        if consensus_line is None:
            out = StagedOutput(current_line, eligible=False, code="MISSING_VALUATION_CONSENSUS", integrity_state="DEPENDENCY_FAILURE", source_ids=[1, 2, 3])
            self.facade.write_sealed_output(node, out)
            return out
        if abs(float(consensus_line) - float(current_line)) > self.ADCC["CONTAMINATION_THRESHOLD"]:
            out = StagedOutput(current_line, eligible=False, code="CONTAMINATED", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[1, 2, 3])
            self.facade.write_sealed_output(node, out)
            return out
        payload = {"current_line": float(current_line), "timestamp_utc": ts}
        out = StagedOutput(payload, eligible=True, code="CLEAN", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[1, 2, 3])
        self.facade.write_sealed_output(node, out)
        return out

    def emit_final_leg_governance(self, leg_meta: Dict[str, Any], context_id: str) -> StagedOutput:
        leg_id = leg_meta["leg_id"]
        node = NodeKey("Phase2", "final_leg_governance", leg_id, context_id)
        family_m = self.facade.read_state(NodeKey("Phase1", "Family_M", leg_id, context_id)).value
        family_n = self.facade.read_state(NodeKey("Phase2", "Family_N", leg_id, context_id)).value
        pps = self.facade.read_state(NodeKey("Phase2", "pps_audit", leg_id, context_id))
        fd = self.facade.read_state(NodeKey("Phase2", "fd_line_status", leg_id, context_id))

        def fail(reason: str, integrity: str) -> StagedOutput:
            out = StagedOutput(
                {
                    "final_governed_p": None,
                    "eligible": False,
                    "eligibility_reason": reason,
                    "governance_integrity_status": integrity,
                    "leg_id": leg_id,
                },
                eligible=False,
                code=reason,
                integrity_state=integrity,
            )
            self.facade.write_sealed_output(node, out)
            return out

        if family_m.get("epistemic_disqualifier"):
            return fail("EPISTEMIC_DISQUALIFICATION_PH1", "PROHIBITED_INFERENCE")
        if not pps.eligible:
            return fail("FAILED_PPS_EFFICIENCY_AUDIT", "DERIVED_BY_APPROVED_RULE")
        if not fd.eligible:
            return fail(f"EXECUTION_FIREWALL_FAILURE_{fd.code}", "DERIVED_BY_APPROVED_RULE")
        if family_n.get("val_stale_flag"):
            return fail("VALUATION_STALE_REJECTION", "DERIVED_BY_APPROVED_RULE")
        if family_n.get("val_off_market_flag"):
            return fail("OFF_MARKET_REJECTION", "DERIVED_BY_APPROVED_RULE")
        if family_n.get("val_consensus_quality_class") == "LOW":
            return fail("LOW_QUALITY_CONSENSUS_REJECTION", "DERIVED_BY_APPROVED_RULE")

        raw_p = family_m.get("raw_event_prob_over_current_line")
        fragility = family_m.get("fragility_score")
        val_p = family_n.get("val_fair_p")
        if raw_p is None or fragility is None or val_p is None:
            return fail("MISSING_PROBABILITY_OR_FRAGILITY_METRICS", "DEPENDENCY_FAILURE")

        blended = (raw_p * (1.0 - self.ADCC["MARKET_WEIGHT"])) + (val_p * self.ADCC["MARKET_WEIGHT"])
        governed = blended * (1 - (fragility * self.ADCC["FRAGILITY_PUNISHMENT_MULT"]))
        risk_haircut = 1.0
        if family_n.get("val_market_drift_flag"):
            risk_haircut *= 0.90
        if family_n.get("val_consensus_quality_class") == "MODERATE":
            risk_haircut *= 0.95
        governed = max(0.0, min(1.0, governed * risk_haircut))
        eligible = governed > 0.50
        payload = {
            "leg_id": leg_id,
            "player_id": leg_meta["player_id"],
            "prop_id": leg_meta["prop_id"],
            "game_id": leg_meta["game_id"],
            "stat_type": leg_meta["stat_type"],
            "market_side": leg_meta["market_side"],
            "line": leg_meta["line"],
            "final_governed_p": governed,
            "eligible": eligible,
            "eligibility_reason": "LOCKED" if eligible else "FRAGILITY_REJECTION",
            "governance_integrity_status": "VERIFIED" if eligible else "DERIVED_BY_APPROVED_RULE",
            "risk_haircut_applied": risk_haircut,
            "valuation_risk_flags": {
                "val_stale_flag": family_n.get("val_stale_flag"),
                "val_off_market_flag": family_n.get("val_off_market_flag"),
                "val_market_drift_flag": family_n.get("val_market_drift_flag"),
                "val_consensus_quality_class": family_n.get("val_consensus_quality_class"),
            },
            "val_current_price_consensus": family_n.get("val_current_price_consensus"),
        }
        out = StagedOutput(payload, eligible=eligible, code=payload["eligibility_reason"], integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[19, 20, 30, 1, 2, 3])
        self.facade.write_sealed_output(node, out)
        return out

    def generate_eligible_over_universe(self, game_meta: Dict[str, Any], roster: List[Dict[str, Any]], leg_metas: List[Dict[str, Any]]) -> StagedOutput:
        game_id = game_meta["game_id"]
        context_id = game_meta["slate_id"]
        node = NodeKey("Phase2", "eligible_over_leg_universe", game_id, context_id)
        player_lookup = {p["player_id"]: p for p in roster}
        qualified: List[Dict[str, Any]] = []

        for leg in leg_metas:
            if leg.get("market_side", "").upper() != "OVER":
                continue
            if leg.get("stat_type") not in self.APPROVED_STAT_FAMILIES:
                continue
            pctx = player_lookup.get(leg["player_id"], {})
            projected_minutes = float(pctx.get("projected_minutes", 0.0) or 0.0)
            role_lock_class = pctx.get("role_lock_class", "STANDARD")
            min_threshold = 8.0 if game_meta.get("playoff_context") and role_lock_class != "HIGH_UNCERTAINTY" else 10.0
            if projected_minutes < min_threshold:
                continue
            gov_node = NodeKey("Phase2", "final_leg_governance", leg["leg_id"], context_id)
            self.facade.repair_or_halt(gov_node, "Phase2", {"leg_meta": leg, "game_meta": game_meta})
            gov = self.facade.read_state(gov_node)
            if not gov.eligible:
                continue
            qualified.append(gov.value)

        payload = {
            "game_id": game_id,
            "slate_id": context_id,
            "universe_size": len(qualified),
            "qualified_legs": qualified,
        }
        out = StagedOutput(payload, eligible=True, code="UNIVERSE_BUILT", integrity_state="DERIVED_BY_APPROVED_RULE", source_ids=[19, 20, 30, 1, 2, 3, 4, 5, 6])
        self.facade.write_sealed_output(node, out)
        return out

    def recompute_family_n(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        leg_meta = kwargs["leg_meta"]
        out = self.derive_family_n(leg_meta, node_key.context_id)
        return out, out.integrity_state

    def recompute_pps_audit(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        out = self.execute_efficiency_audit(node_key.entity_id, node_key.context_id)
        return out, out.integrity_state

    def recompute_fd_line_status(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        out = self.enforce_execution_firewall(node_key.entity_id, node_key.context_id)
        return out, out.integrity_state

    def recompute_final_leg_governance(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        leg_meta = kwargs["leg_meta"]
        out = self.emit_final_leg_governance(leg_meta, node_key.context_id)
        return out, out.integrity_state

    def recompute_universe(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        out = self.generate_eligible_over_universe(kwargs["game_meta"], kwargs["roster"], kwargs["leg_metas"])
        return out, out.integrity_state


# ============================================================
# PHASE 3
# ============================================================

class NBAPropEnginePhase3:
    ADCC = {
        "COPULA_THETA_BASE": 1.25,
        "MAX_SEARCH_LEGS": 10,
        "MIN_SEARCH_LEGS": 2,
        "MAX_CANDIDATE_POOL": 36,
        "MAX_CANDIDATE_POOL_EXPANDED": 80,
        # Per-band search budgets applied per leg_count iteration within each band.
        # Higher bands target narrow payout windows requiring longer parlays;
        # they need exponentially more combination checks to find valid tickets.
        "BAND_SEARCH_LIMITS": {
            "BAND_1": 300,
            "BAND_2": 500,
            "BAND_3": 1000,
            "BAND_4": 2500,
            "BAND_5": 7500,
        },
        # Expanded budgets used in second-pass retry for unfilled bands.
        "BAND_SEARCH_LIMITS_EXPANDED": {
            "BAND_1": 600,
            "BAND_2": 1000,
            "BAND_3": 2000,
            "BAND_4": 5000,
            "BAND_5": 15000,
        },
    }

    def __init__(self, facade: PhaseAccessFacade) -> None:
        self.facade = facade

    def _candidate_score(self, legs: Sequence[Dict[str, Any]]) -> float:
        score = 0.0
        for leg in legs:
            p = float(leg["final_governed_p"])
            price = float(leg.get("val_current_price_consensus", 1.8) or 1.8)
            implied = 1.0 / price if price > 1 else 1.0
            edge = p - implied
            score += math.log(max(1e-9, p)) + edge
        return score

    def _gross_payout(self, legs: Sequence[Dict[str, Any]]) -> float:
        gross = 1.0
        for leg in legs:
            gross *= float(leg.get("val_current_price_consensus", 1.80) or 1.80)
        return gross

    def _valid_ticket_structure(self, legs: Sequence[Dict[str, Any]]) -> bool:
        """
        Validate that a combination satisfies the player-diversity contract.
        The first guard rejects any duplicate player. After that guard fires,
        unique_players == leg_count is guaranteed, so the per-leg-count
        minimums below are redundant and have been removed. The minimum-player
        checks that are live in Phase 4's passes_diversity_governor still apply
        at the portfolio level after ticket acceptance.
        """
        player_ids = [leg["player_id"] for leg in legs]
        if len(set(player_ids)) != len(player_ids):
            return False
        return True

    def _rank_legs(self, all_raw_legs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort legs by governed probability descending, then by implied price ascending (edge proxy)."""
        return sorted(
            all_raw_legs,
            key=lambda x: (
                float(x["final_governed_p"]),
                -(1.0 / max(1.01, float(x.get("val_current_price_consensus", 1.8)))),
            ),
            reverse=True,
        )

    def _search_band(
        self,
        band_name: str,
        band: Dict[str, Any],
        leg_pool: List[Dict[str, Any]],
        needed: int,
        seen_signatures: set,
        ticket_counter_start: int,
        slate_id: str,
        max_accepted_per_leg_count: int,
    ) -> List[Dict[str, Any]]:
        """
        Search a single payout band for up to `needed` valid candidate tickets.

        `max_accepted_per_leg_count` caps the number of valid combinations
        accepted per leg_count iteration — it is an acceptance ceiling, not an
        effort ceiling. Combinations that fail the payout or structure checks
        are examined without incrementing the counter, so the actual number of
        combinations visited is always >= the number accepted. This is
        intentional: higher bands must examine more combinations to find the
        narrow payout windows they target.
        """
        target_min = float(band["min"])
        target_max = float(band["max"])
        found_tickets: List[Dict[str, Any]] = []
        counter = ticket_counter_start

        for leg_count in range(self.ADCC["MIN_SEARCH_LEGS"], self.ADCC["MAX_SEARCH_LEGS"] + 1):
            if len(found_tickets) >= needed:
                break
            accepted_count = 0
            for combo in combinations(leg_pool, leg_count):
                if accepted_count >= max_accepted_per_leg_count:
                    break
                if not self._valid_ticket_structure(combo):
                    continue
                gross = self._gross_payout(combo)
                if gross < target_min or gross > target_max:
                    continue
                sig = tuple(sorted(leg["leg_id"] for leg in combo))
                if sig in seen_signatures:
                    continue
                seen_signatures.add(sig)
                accepted_count += 1
                combo_sorted = sorted(combo, key=lambda x: (x["player_id"], x["prop_id"]))
                found_tickets.append(
                    {
                        "ticket_id": f"T{counter:02d}",
                        "leg_ids": [leg["leg_id"] for leg in combo_sorted],
                        "player_ids": [leg["player_id"] for leg in combo_sorted],
                        "prop_ids": [leg["prop_id"] for leg in combo_sorted],
                        "game_ids": sorted({leg["game_id"] for leg in combo_sorted}),
                        "slate_id": slate_id,
                        "context_id": slate_id,
                        "payout_band_target": band_name,
                        "projected_gross_payout_on_1": gross,
                        "leg_count": len(combo_sorted),
                        "score": self._candidate_score(combo_sorted),
                    }
                )
                counter += 1
                if len(found_tickets) >= needed:
                    break

        return found_tickets

    def build_candidate_tickets(self, slate_id: str, universes: List[Dict[str, Any]], payout_bands: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_raw_legs: List[Dict[str, Any]] = []
        for uni in universes:
            all_raw_legs.extend(uni.get("qualified_legs", []))

        ranked_legs = self._rank_legs(all_raw_legs)
        primary_pool = ranked_legs[: self.ADCC["MAX_CANDIDATE_POOL"]]
        expanded_pool = ranked_legs[: self.ADCC["MAX_CANDIDATE_POOL_EXPANDED"]]

        primary_limits = self.ADCC["BAND_SEARCH_LIMITS"]
        expanded_limits = self.ADCC["BAND_SEARCH_LIMITS_EXPANDED"]

        all_candidates: List[Dict[str, Any]] = []
        ticket_counter = 1
        seen_signatures: set[Tuple[str, ...]] = set()

        # --- Primary pass: use standard pool and per-band search budget ---
        band_shortfalls: Dict[str, int] = {}
        for band_name, band in payout_bands.items():
            needed = int(band["ticket_count"])
            limit = primary_limits.get(band_name, 300)
            found = self._search_band(
                band_name, band, primary_pool, needed, seen_signatures, ticket_counter, slate_id,
                max_accepted_per_leg_count=limit,
            )
            all_candidates.extend(found)
            ticket_counter += len(found)
            shortfall = needed - len(found)
            if shortfall > 0:
                band_shortfalls[band_name] = shortfall

        # --- Second pass: expanded pool and budget for any unfilled bands ---
        if band_shortfalls:
            for band_name, shortfall in band_shortfalls.items():
                band = payout_bands[band_name]
                limit = expanded_limits.get(band_name, 1000)
                found = self._search_band(
                    band_name, band, expanded_pool, shortfall, seen_signatures, ticket_counter, slate_id,
                    max_accepted_per_leg_count=limit,
                )
                for ticket in found:
                    # Mark retry tickets so audit log can distinguish them.
                    ticket["search_pass"] = "EXPANDED_RETRY"
                all_candidates.extend(found)
                ticket_counter += len(found)
                remaining = shortfall - len(found)
                if remaining > 0:
                    # Emit a per-band failure marker that survives into the portfolio result.
                    all_candidates.append(
                        {
                            "ticket_id": f"UNFILLABLE_{band_name}",
                            "payout_band_target": band_name,
                            "eligible": False,
                            "code": f"{band_name}_UNFILLABLE",
                            "shortfall": remaining,
                        }
                    )

        all_candidates.sort(key=lambda x: (x.get("payout_band_target", ""), -x.get("score", 0.0)))
        return all_candidates

    def derive_copula_joint_prob(self, ticket: Dict[str, Any], context_id: str) -> StagedOutput:
        node = NodeKey("Phase3", "joint_prob", ticket["ticket_id"], context_id)
        leg_probs: List[float] = []
        for leg_id in ticket["leg_ids"]:
            gov = self.facade.read_state(NodeKey("Phase2", "final_leg_governance", leg_id, context_id))
            if not gov.eligible:
                out = StagedOutput({}, eligible=False, code=f"INELIGIBLE_LEG_{leg_id}", integrity_state="DERIVED_BY_APPROVED_RULE")
                self.facade.write_sealed_output(node, out)
                return out
            leg_probs.append(float(gov.value["final_governed_p"]))

        if not leg_probs:
            out = StagedOutput({}, eligible=False, code="EMPTY_LEG_ARRAY", integrity_state="DEPENDENCY_FAILURE")
            self.facade.write_sealed_output(node, out)
            return out

        if len(leg_probs) == 1:
            joint_p = leg_probs[0]
        else:
            theta = self.ADCC["COPULA_THETA_BASE"]
            sum_inverted = sum((max(1e-9, p) ** -theta) for p in leg_probs) - (len(leg_probs) - 1)
            joint_p = sum_inverted ** (-1 / theta) if sum_inverted > 0 else 0.0
        joint_p = max(0.0, min(1.0, joint_p))

        payload = {
            "ticket_id": ticket["ticket_id"],
            "joint_governed_p": joint_p,
            "ticket_gross_payout_on_1": ticket["projected_gross_payout_on_1"],
            "leg_count": ticket["leg_count"],
            "candidate_ticket_def": ticket,
            "correlation_applied": len(leg_probs) > 1,
        }
        out = StagedOutput(payload, eligible=True, code="ROUTED", integrity_state="DERIVED_BY_APPROVED_RULE")
        self.facade.write_sealed_output(node, out)
        return out

    def recompute_joint_prob(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        ticket = kwargs["candidate_ticket"]
        out = self.derive_copula_joint_prob(ticket, node_key.context_id)
        return out, out.integrity_state


# ============================================================
# PHASE 4
# ============================================================

class PortfolioExposureMap:
    def __init__(self) -> None:
        self.player_counts: Dict[str, int] = defaultdict(int)
        self.band_counts: Dict[str, int] = defaultdict(int)

    @property
    def unique_players_used(self) -> int:
        return len(self.player_counts)

    def register_ticket(self, ticket: Dict[str, Any]) -> None:
        for pid in ticket.get("player_ids", []):
            self.player_counts[pid] += 1
        self.band_counts[ticket.get("payout_band_target", "UNKNOWN")] += 1


class NBAPropEnginePhase4:
    ADCC = {
        "FIXED_STAKE": 1.0,
        "MAX_GROSS_PAYOUT": 1000.0,
        "MIN_GROSS_PAYOUT": 60.0,
    }

    def __init__(self, facade: PhaseAccessFacade) -> None:
        self.facade = facade
        self.portfolios: Dict[str, PortfolioExposureMap] = defaultdict(PortfolioExposureMap)

    def passes_diversity_governor(self, ticket: Dict[str, Any], exposure_map: PortfolioExposureMap, eligible_universe_size: int) -> bool:
        player_ids = ticket.get("player_ids", [])
        if not player_ids:
            return False
        leg_count = len(player_ids)
        unique_players = len(set(player_ids))
        if unique_players != len(player_ids):
            return False
        if leg_count == 4 and unique_players < 4:
            return False
        if leg_count in [5, 6] and unique_players < 5:
            return False
        if leg_count >= 7 and unique_players < 6:
            return False
        if eligible_universe_size >= 14 and exposure_map.unique_players_used < 8 and unique_players < 6:
            return False
        if eligible_universe_size >= 10 and exposure_map.unique_players_used < 6 and unique_players < 5:
            return False
        return True

    def _compute_acceptance(
        self,
        slate_id: str,
        ticket: Dict[str, Any],
        payout_bands: Dict[str, Dict[str, Any]],
        eligible_universe_size: int,
    ) -> StagedOutput:
        """
        Pure computation layer. Evaluates a ticket against all acceptance
        criteria and returns a StagedOutput representing the decision.

        This method does NOT write to any node and does NOT mutate the
        exposure map. It is the only path that recompute_kelly_size may
        call, ensuring the legacy compat handler can never produce a ghost
        write to the ticket_acceptance node.
        """
        context_id = ticket["context_id"]
        joint = self.facade.read_state(NodeKey("Phase3", "joint_prob", ticket["ticket_id"], context_id))
        if not joint.eligible:
            return StagedOutput({}, eligible=False, code="UPSTREAM_TICKET_INELIGIBLE", integrity_state="DEPENDENCY_FAILURE")

        exposure_map = self.portfolios[slate_id]
        gross = float(joint.value["ticket_gross_payout_on_1"])
        band_name = ticket["payout_band_target"]
        band = payout_bands[band_name]
        if gross < band["min"] or gross > band["max"]:
            return StagedOutput({}, eligible=False, code="PAYOUT_BAND_MISMATCH", integrity_state="DERIVED_BY_APPROVED_RULE")
        if gross < self.ADCC["MIN_GROSS_PAYOUT"] or gross > self.ADCC["MAX_GROSS_PAYOUT"]:
            return StagedOutput({}, eligible=False, code="PAYOUT_OUTSIDE_CONSTITUTION", integrity_state="DERIVED_BY_APPROVED_RULE")
        if not self.passes_diversity_governor(ticket, exposure_map, eligible_universe_size):
            return StagedOutput({}, eligible=False, code="DIVERSITY_GOVERNOR_REJECTION", integrity_state="DERIVED_BY_APPROVED_RULE")

        joint_p = float(joint.value["joint_governed_p"])
        ev_margin = (joint_p * gross) - 1.0
        if ev_margin <= 0:
            return StagedOutput({}, eligible=False, code="NEGATIVE_EXPECTED_VALUE", integrity_state="DERIVED_BY_APPROVED_RULE")

        payload = {
            "ticket_id": ticket["ticket_id"],
            "approved_stake": self.ADCC["FIXED_STAKE"],
            "projected_gross_payout": gross,
            "projected_net_profit": gross - self.ADCC["FIXED_STAKE"],
            "payout_band": band_name,
            "ev_margin": ev_margin,
            "diversity_status": "PASSED",
            "portfolio_spread_status": "PENDING_FINAL_PORTFOLIO_CHECK",
            "candidate_ticket_def": ticket,
        }
        return StagedOutput(payload, eligible=True, code="ACCEPTED", integrity_state="DERIVED_BY_APPROVED_RULE")

    def evaluate_ticket(self, slate_id: str, ticket: Dict[str, Any], payout_bands: Dict[str, Dict[str, Any]], eligible_universe_size: int) -> StagedOutput:
        """
        Canonical entry point for the portfolio build loop.
        Calls _compute_acceptance, writes the result to ticket_acceptance,
        and registers the ticket in the exposure map on acceptance.
        """
        context_id = ticket["context_id"]
        node = NodeKey("Phase4", "ticket_acceptance", ticket["ticket_id"], context_id)
        out = self._compute_acceptance(slate_id, ticket, payout_bands, eligible_universe_size)
        self.facade.write_sealed_output(node, out)
        if out.eligible:
            self.portfolios[slate_id].register_ticket(ticket)
        return out

    def recompute_ticket_acceptance(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        """
        Dedicated recompute handler for ticket_acceptance.
        Routes through evaluate_ticket, which is the sole authorised writer
        for this node. The kelly_size compat write is a controlled side effect
        that does not flow back through the coordinator's return path.
        """
        out = self.evaluate_ticket(
            kwargs["slate_id"],
            kwargs["candidate_ticket"],
            kwargs["payout_bands"],
            kwargs["eligible_universe_size"],
        )
        if out.eligible:
            self.write_kelly_compat(
                out.value["ticket_id"],
                ticket_context_id=out.value["candidate_ticket_def"]["context_id"],
                acceptance_payload=out.value,
            )
        return out, out.integrity_state

    def write_kelly_compat(self, ticket_id: str, ticket_context_id: str, acceptance_payload: Dict[str, Any]) -> None:
        """
        Write a kelly_size compat node as a backward-compatibility alias.
        Uses ticket_context_id (from the ticket dict) rather than the recompute
        NodeKey's context_id so the two nodes are guaranteed to share the same
        context regardless of how the recompute was triggered.
        This node must only be written here — never returned through the
        coordinator as the acceptance value.
        """
        compat_node = NodeKey("Phase4", "kelly_size", ticket_id, ticket_context_id)
        compat_payload = copy.deepcopy(acceptance_payload)
        compat_payload["accepted_under_fixed_1_architecture"] = acceptance_payload.get("approved_stake") == 1.0
        compat = StagedOutput(
            compat_payload,
            eligible=True,
            code="COMPAT_ALIAS",
            integrity_state="DERIVED_BY_APPROVED_RULE",
        )
        self.facade.write_sealed_output(compat_node, compat)

    def recompute_kelly_size(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        """
        Legacy recompute path for the kelly_size compat alias node.
        Calls _compute_acceptance — the pure, no-write computation layer —
        so this handler can never produce a ghost write to ticket_acceptance.
        The compat node is then written explicitly via write_kelly_compat.
        """
        result = self._compute_acceptance(
            kwargs["slate_id"],
            kwargs["candidate_ticket"],
            kwargs["payout_bands"],
            kwargs["eligible_universe_size"],
        )
        ticket_context_id = kwargs["candidate_ticket"]["context_id"]
        if result.eligible:
            self.write_kelly_compat(result.value["ticket_id"], ticket_context_id, result.value)
        compat_node = NodeKey("Phase4", "kelly_size", node_key.entity_id, node_key.context_id)
        try:
            compat = self.facade.read_state(compat_node)
        except KeyError:
            compat = result
        return compat, compat.integrity_state


# ============================================================
# PHASE 5
# ============================================================

class NBAPropEnginePhase5:
    def __init__(self, facade: PhaseAccessFacade) -> None:
        self.facade = facade

    def seal_terminal_ticket(self, ticket_id: str, context_id: str, leg_ids: List[str]) -> StagedOutput:
        node = NodeKey("Phase5", "terminal_ticket", ticket_id, context_id)
        family_p = self.facade.read_state(NodeKey("Phase0", "Family_P", "global", "run")).value
        acceptance = self.facade.read_state(NodeKey("Phase4", "ticket_acceptance", ticket_id, context_id))
        if not acceptance.eligible:
            out = StagedOutput({"terminal_status": "ABORTED"}, eligible=False, code="TICKET_FINANCIALLY_INELIGIBLE", integrity_state="DEPENDENCY_FAILURE")
            self.facade.write_sealed_output(node, out)
            return out

        bundle_ts_utc = family_p.get("bundle_frozen_ts_utc")
        frozen_ts: Optional[datetime] = None
        if bundle_ts_utc:
            try:
                frozen_ts = datetime.fromisoformat(bundle_ts_utc.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                frozen_ts = None

        contamination_threshold = EngineConstitution.EXECUTION_CONTAMINATION_THRESHOLD
        max_staleness_sec = EngineConstitution.MAX_EXECUTION_STALENESS_SEC

        for leg_id in leg_ids:
            # Phase 5 re-fetches execution state directly from the provider.
            # This is an independent terminal check — not a replay of Phase 2's cached verdict.
            execution = self.facade.provider.get_execution_line(leg_id)
            current_line = execution.get("current_line")
            ts = execution.get("timestamp_utc")

            if current_line is None or ts is None:
                out = StagedOutput(
                    {"terminal_status": "ABORTED"},
                    eligible=False,
                    code=f"TERMINAL_MISSING_EXECUTION_LINE_{leg_id}",
                    integrity_state="DEPENDENCY_FAILURE",
                )
                self.facade.write_sealed_output(node, out)
                return out

            if frozen_ts is not None:
                try:
                    ex_ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    delta = abs((ex_ts - frozen_ts).total_seconds())
                    if delta > max_staleness_sec:
                        out = StagedOutput(
                            {"terminal_status": "ABORTED"},
                            eligible=False,
                            code=f"TERMINAL_STALE_EXECUTION_LINE_{leg_id}",
                            integrity_state="DERIVED_BY_APPROVED_RULE",
                        )
                        self.facade.write_sealed_output(node, out)
                        return out
                except (ValueError, AttributeError):
                    pass  # Unparseable timestamp — allow; provider is responsible for format.

            # Compare the terminal execution line against the Phase 2 consensus to detect
            # post-Phase-2 line drift that would invalidate the edge.
            phase2_family_n_node = NodeKey("Phase2", "Family_N", leg_id, context_id)
            try:
                family_n = self.facade.read_state(phase2_family_n_node).value
                consensus_line = family_n.get("val_current_line_consensus")
                if consensus_line is not None and abs(float(consensus_line) - float(current_line)) > contamination_threshold:
                    out = StagedOutput(
                        {"terminal_status": "ABORTED"},
                        eligible=False,
                        code=f"TERMINAL_LINE_DRIFT_DETECTED_{leg_id}",
                        integrity_state="DERIVED_BY_APPROVED_RULE",
                    )
                    self.facade.write_sealed_output(node, out)
                    return out
            except (KeyError, ValueError):
                pass  # Family_N missing; contamination check skipped, Phase 2 already governed this leg.

        payload = {
            "ticket_id": ticket_id,
            "snapshot_bundle_id": family_p["snapshot_bundle_id"],
            "approved_stake": acceptance.value["approved_stake"],
            "projected_gross_payout": acceptance.value["projected_gross_payout"],
            "projected_net_profit": acceptance.value["projected_net_profit"],
            "payout_band": acceptance.value["payout_band"],
            "legs_included": leg_ids,
            "terminal_status": "SEALED_AND_READY_FOR_EXECUTION",
            "terminal_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pregame_only": True,
        }
        out = StagedOutput(payload, eligible=True, code="TERMINAL_SEAL", integrity_state="VERIFIED")
        self.facade.write_sealed_output(node, out)
        return out

    def recompute_terminal_ticket(self, node_key: NodeKey, **kwargs) -> Tuple[StagedOutput, str]:
        out = self.seal_terminal_ticket(node_key.entity_id, node_key.context_id, kwargs["leg_ids"])
        return out, out.integrity_state


# ============================================================
# MASTER COORDINATOR
# ============================================================

class NBAPropEngine:
    def __init__(self, provider: PregameSourceProvider) -> None:
        self.provider = provider
        self.state = EngineState()
        self.guard = RecomputeGuard()
        self.repair_budget = RepairBudget()
        self.facade = PhaseAccessFacade(self)

        self.phase0 = NBAPropEnginePhase0(self.facade)
        self.phase1 = NBAPropEnginePhase1(self.facade)
        self.phase2 = NBAPropEnginePhase2(self.facade)
        self.phase3 = NBAPropEnginePhase3(self.facade)
        self.phase4 = NBAPropEnginePhase4(self.facade)
        self.phase5 = NBAPropEnginePhase5(self.facade)

        self.phases = {
            "Phase0": self.phase0,
            "Phase1": self.phase1,
            "Phase2": self.phase2,
            "Phase3": self.phase3,
            "Phase4": self.phase4,
            "Phase5": self.phase5,
        }

        self._initialize_wiring()

    def _initialize_wiring(self) -> None:
        self.dependency_rules = {
            ("Phase1", "lambda_fga"): [("Phase2", "pps_audit")],
            ("Phase1", "Family_M"): [("Phase2", "final_leg_governance")],
            ("Phase2", "Family_N"): [("Phase2", "pps_audit"), ("Phase2", "final_leg_governance")],
            ("Phase2", "pps_audit"): [("Phase2", "final_leg_governance")],
            ("Phase2", "fd_line_status"): [("Phase2", "final_leg_governance")],
            ("Phase2", "final_leg_governance"): [("Phase2", "eligible_over_leg_universe"), ("Phase3", "joint_prob")],
            ("Phase2", "eligible_over_leg_universe"): [("Phase4", "ticket_acceptance"), ("Phase4", "kelly_size")],
            ("Phase3", "joint_prob"): [("Phase4", "ticket_acceptance"), ("Phase4", "kelly_size")],
            ("Phase4", "ticket_acceptance"): [("Phase5", "terminal_ticket")],
            ("Phase4", "kelly_size"): [("Phase5", "terminal_ticket")],
        }

        self.prerequisite_rules = {
            ("Phase2", "pps_audit"): [("Phase1", "lambda_fga"), ("Phase2", "Family_N")],
            ("Phase2", "final_leg_governance"): [("Phase1", "Family_M"), ("Phase2", "Family_N"), ("Phase2", "pps_audit"), ("Phase2", "fd_line_status")],
            ("Phase2", "eligible_over_leg_universe"): [("Phase2", "final_leg_governance")],
            ("Phase3", "joint_prob"): [("Phase2", "final_leg_governance")],
            ("Phase4", "ticket_acceptance"): [("Phase2", "eligible_over_leg_universe"), ("Phase3", "joint_prob")],
            ("Phase4", "kelly_size"): [("Phase2", "eligible_over_leg_universe"), ("Phase3", "joint_prob")],
            ("Phase5", "terminal_ticket"): [("Phase4", "ticket_acceptance")],
        }

        self.recompute_policy = {
            ("COORDINATOR", "Phase1", "lambda_fga"): "ALLOW",
            ("COORDINATOR", "Phase1", "Family_M"): "ALLOW",
            ("COORDINATOR", "Phase2", "Family_N"): "ALLOW",
            ("COORDINATOR", "Phase2", "pps_audit"): "ALLOW",
            ("COORDINATOR", "Phase2", "fd_line_status"): "ALLOW",
            ("COORDINATOR", "Phase2", "final_leg_governance"): "ALLOW",
            ("COORDINATOR", "Phase2", "eligible_over_leg_universe"): "ALLOW",
            ("COORDINATOR", "Phase3", "joint_prob"): "ALLOW",
            ("COORDINATOR", "Phase4", "ticket_acceptance"): "ALLOW",
            ("COORDINATOR", "Phase4", "kelly_size"): "ALLOW",
            ("COORDINATOR", "Phase5", "terminal_ticket"): "ALLOW",
            ("Phase2", "Phase2", "final_leg_governance"): "ALLOW",
            ("Phase3", "Phase1", "lambda_fga"): "ALLOW",
            ("Phase3", "Phase2", "final_leg_governance"): "ALLOW",
            ("Phase4", "Phase3", "joint_prob"): "ALLOW",
            ("Phase4", "Phase2", "eligible_over_leg_universe"): "ALLOW",
            ("Phase5", "Phase4", "ticket_acceptance"): "ALLOW",
            ("Phase5", "Phase2", "fd_line_status"): "ALLOW",
        }

        self.repair_policy = {
            "DIRTY_AWAITING_RECOMPUTE": "AUTO_REPAIR",
            "STALE_EXECUTION_LINE": "AUTO_REFRESH",
            "VALUATION_MISSING": "AUTO_REBUILD",
            "DEPENDENCY_FAILURE": "AUTO_HEAL_IF_RAW_INPUTS_EXIST",
            "PHASE_CONTRACT_VIOLATION": "HALT",
            "PROHIBITED_INFERENCE": "HALT",
            "CYCLE_VIOLATION": "HALT",
        }

        self.recompute_handlers = {
            ("Phase1", "lambda_fga"): self.phase1.recompute_lambda_fga,
            ("Phase1", "Family_M"): self.phase1.recompute_family_m,
            ("Phase2", "Family_N"): self.phase2.recompute_family_n,
            ("Phase2", "pps_audit"): self.phase2.recompute_pps_audit,
            ("Phase2", "fd_line_status"): self.phase2.recompute_fd_line_status,
            ("Phase2", "final_leg_governance"): self.phase2.recompute_final_leg_governance,
            ("Phase2", "eligible_over_leg_universe"): self.phase2.recompute_universe,
            ("Phase3", "joint_prob"): self.phase3.recompute_joint_prob,
            # ticket_acceptance now has its own dedicated handler that returns the clean
            # acceptance payload. kelly_size retains a separate handler as a compat alias.
            # These must never share a handler — see Defect C-2 resolution.
            ("Phase4", "ticket_acceptance"): self.phase4.recompute_ticket_acceptance,
            ("Phase4", "kelly_size"): self.phase4.recompute_kelly_size,
            ("Phase5", "terminal_ticket"): self.phase5.recompute_terminal_ticket,
        }

    # ---------- STATE ACCESS ----------

    def write_initial_state(self, node_key: NodeKey, value: StagedOutput, integrity_state: str) -> None:
        self.state.sealed_outputs[node_key] = copy.deepcopy(value)
        self.state.integrity_ledger[node_key] = integrity_state
        self.state.runtime_status_ledger[node_key] = "CLEAN"
        self.state.log_action("NODE_INITIALIZED", {"node": node_key.__dict__, "code": value.code, "integrity": integrity_state})

    def _read_state_internal(self, node_key: NodeKey, allow_dirty: bool = False) -> StagedOutput:
        if node_key not in self.state.sealed_outputs:
            raise KeyError(f"STATE_READ_ERROR: missing {node_key}")
        if not allow_dirty and self.state.runtime_status_ledger.get(node_key) == "DIRTY_AWAITING_RECOMPUTE":
            raise ValueError(f"STATE_READ_ERROR: dirty {node_key}")
        return copy.deepcopy(self.state.sealed_outputs[node_key])

    def read_state(self, node_key: NodeKey) -> StagedOutput:
        return self._read_state_internal(node_key, allow_dirty=False)

    def _is_frozen(self, node_key: NodeKey) -> bool:
        if self.state.freeze_flags.get(node_key.phase, False):
            return True
        if self.state.freeze_flags.get((node_key.phase, node_key.field), False):
            return True
        if self.state.freeze_flags.get(node_key, False):
            return True
        return False

    # ---------- REPAIR / RECOMPUTE ----------

    def classify_repair(self, target_node: NodeKey) -> RepairDecision:
        status = self.state.runtime_status_ledger.get(target_node, "CLEAN")
        integrity = self.state.integrity_ledger.get(target_node, "MISSING")
        if integrity in {"PROHIBITED_INFERENCE", "PHASE_CONTRACT_VIOLATION"}:
            return RepairDecision(False, f"CONSTITUTIONAL_BREACH: {integrity}")
        if self._is_frozen(target_node):
            return RepairDecision(False, "FREEZE_VIOLATION")
        if (target_node.phase, target_node.field) not in self.recompute_handlers:
            return RepairDecision(False, "NO_REGISTERED_HANDLER")
        policy_action = self.repair_policy.get(status, self.repair_policy.get(integrity, "UNKNOWN"))
        if policy_action in {"AUTO_REPAIR", "AUTO_REFRESH", "AUTO_REBUILD", "AUTO_HEAL_IF_RAW_INPUTS_EXIST"}:
            repair_path: List[NodeKey] = []
            for phase, field in self.prerequisite_rules.get((target_node.phase, target_node.field), []):
                prereq = NodeKey(phase, field, target_node.entity_id, target_node.context_id)
                if self.state.runtime_status_ledger.get(prereq, "CLEAN") != "CLEAN":
                    repair_path.append(prereq)
            repair_path.append(target_node)
            return RepairDecision(True, f"POLICY:{policy_action}", repair_path)
        return RepairDecision(False, f"UNMAPPED_FAILURE_STATE:{status}|{integrity}")

    def repair_or_halt(self, target_node: NodeKey, requester: str, scope_args: Optional[Dict[str, Any]] = None) -> bool:
        scope_args = scope_args or {}
        status = self.state.runtime_status_ledger.get(target_node, "CLEAN")
        integrity = self.state.integrity_ledger.get(target_node)
        if status == "CLEAN" and integrity not in {"DEPENDENCY_FAILURE", "MISSING", "INVALID"}:
            return True
        decision = self.classify_repair(target_node)
        if not decision.repairable:
            raise RuntimeError(f"NON_REPAIRABLE_FAILURE: {target_node} | {decision.reason}")
        if not self.repair_budget.check_and_consume(target_node):
            raise RuntimeError(f"REPAIR_BUDGET_EXHAUSTED: {target_node}")
        for repair_node in decision.repair_path:
            node_scope = self._resolve_prereq_scope(repair_node, scope_args)
            ok = self.request_recompute("COORDINATOR", repair_node, f"AUTO_REPAIR_FOR_{target_node.field}", node_scope)
            if not ok or self.state.runtime_status_ledger.get(repair_node, "CLEAN") != "CLEAN":
                raise RuntimeError(f"REPAIR_FAILED: {repair_node}")
        return True

    def _resolve_prereq_scope(self, prereq_node: NodeKey, parent_scope: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map parent scope to the specific kwargs required by each prerequisite node's handler.
        Every prereq key is explicitly listed. Unknown prereqs receive an empty scope
        so that handler defaults fire rather than receiving incorrect parent kwargs.
        """
        key = (prereq_node.phase, prereq_node.field)
        if key == ("Phase1", "Family_M"):
            return {
                k: parent_scope[k]
                for k in ("leg_meta", "game_meta")
                if k in parent_scope
            }
        if key == ("Phase1", "lambda_fga"):
            return {
                k: parent_scope[k]
                for k in ("usage_shift_modifier",)
                if k in parent_scope
            }
        if key == ("Phase2", "Family_N"):
            return {
                k: parent_scope[k]
                for k in ("leg_meta",)
                if k in parent_scope
            }
        if key == ("Phase2", "pps_audit"):
            # pps_audit reads lambda_fga and Family_N from sealed state; no extra kwargs needed.
            return {}
        if key == ("Phase2", "fd_line_status"):
            # fd_line_status reads provider directly; no extra kwargs needed.
            return {}
        if key == ("Phase2", "final_leg_governance"):
            return {
                k: parent_scope[k]
                for k in ("leg_meta", "game_meta")
                if k in parent_scope
            }
        if key == ("Phase2", "eligible_over_leg_universe"):
            return {
                k: parent_scope[k]
                for k in ("game_meta", "roster", "leg_metas")
                if k in parent_scope
            }
        if key == ("Phase3", "joint_prob"):
            return {
                k: parent_scope[k]
                for k in ("candidate_ticket",)
                if k in parent_scope
            }
        if key == ("Phase4", "ticket_acceptance"):
            return {
                k: parent_scope[k]
                for k in ("slate_id", "candidate_ticket", "payout_bands", "eligible_universe_size")
                if k in parent_scope
            }
        if key == ("Phase4", "kelly_size"):
            return {
                k: parent_scope[k]
                for k in ("slate_id", "candidate_ticket", "payout_bands", "eligible_universe_size")
                if k in parent_scope
            }
        if key == ("Phase5", "terminal_ticket"):
            return {
                k: parent_scope[k]
                for k in ("leg_ids",)
                if k in parent_scope
            }
        # Unknown prerequisite: return empty scope so handler defaults apply safely.
        return {}

    def _heal_prerequisites(self, target_node: NodeKey, scope_args: Dict[str, Any]) -> None:
        for phase, field in self.prerequisite_rules.get((target_node.phase, target_node.field), []):
            prereq = NodeKey(phase, field, target_node.entity_id, target_node.context_id)
            if self.state.runtime_status_ledger.get(prereq, "CLEAN") == "DIRTY_AWAITING_RECOMPUTE":
                ok = self.request_recompute("COORDINATOR", prereq, f"PREREQ_HEAL_{target_node.field}", self._resolve_prereq_scope(prereq, scope_args))
                if not ok or self.state.runtime_status_ledger.get(prereq, "CLEAN") != "CLEAN":
                    raise RuntimeError(f"PREREQ_HEAL_FAILED: {prereq}")

    def request_recompute(self, requester: str, target_node: NodeKey, reason: str, scope_args: Optional[Dict[str, Any]] = None) -> bool:
        scope_args = scope_args or {}
        if self.recompute_policy.get((requester, target_node.phase, target_node.field), "DENY") != "ALLOW":
            self.state.log_action("RECOMPUTE_DENIED_BY_POLICY", {"requester": requester, "target": target_node.__dict__})
            return False
        if self._is_frozen(target_node):
            self.state.log_action("RECOMPUTE_DENIED_FROZEN", {"requester": requester, "target": target_node.__dict__})
            return False
        with RecomputeSession(self.guard) as session_guard:
            try:
                self._heal_prerequisites(target_node, scope_args)
                session_guard.check_and_lock(target_node)
                self.state.runtime_status_ledger[target_node] = "COMPUTING"
                handler = self.recompute_handlers[(target_node.phase, target_node.field)]
                new_val, new_integrity = handler(target_node, **scope_args)
                self.state.sealed_outputs[target_node] = copy.deepcopy(new_val)
                self.state.integrity_ledger[target_node] = new_integrity
                self.state.runtime_status_ledger[target_node] = "CLEAN"
                self.state.log_action("RECOMPUTE_SUCCESS", {"requester": requester, "target": target_node.__dict__, "reason": reason, "code": new_val.code})
                self._cascade_dirty_flags(target_node)
                return True
            except Exception as exc:
                self.state.runtime_status_ledger[target_node] = "DIRTY_AWAITING_RECOMPUTE"
                self.state.log_action("RECOMPUTE_FAILED", {"requester": requester, "target": target_node.__dict__, "reason": reason, "error": str(exc)})
                raise
            finally:
                session_guard.release(target_node)

    def _cascade_dirty_flags(self, modified_node: NodeKey) -> None:
        visited: set[NodeKey] = set()

        def dfs(node: NodeKey) -> None:
            for next_phase, next_field in self.dependency_rules.get((node.phase, node.field), []):
                child = NodeKey(next_phase, next_field, node.entity_id, node.context_id)
                if child in visited or child not in self.state.sealed_outputs:
                    continue
                visited.add(child)
                self.state.runtime_status_ledger[child] = "DIRTY_AWAITING_RECOMPUTE"
                self.state.log_action("NODE_MARKED_DIRTY", {"source": node.__dict__, "target": child.__dict__})
                dfs(child)

        dfs(modified_node)

    # ---------- ENGINE RUN ----------

    @staticmethod
    def build_leg_id(game_id: str, player_id: str, prop_id: str) -> str:
        return f"{game_id}::{player_id}::{prop_id}"

    def bootstrap(self) -> None:
        self.phase0.initialize_run()

    def _collect_leg_metas_for_game(self, game_meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        roster = self.provider.get_projected_rosters(game_meta["game_id"])
        leg_metas: List[Dict[str, Any]] = []
        for player in roster:
            player_id = player["player_id"]
            for prop in player.get("available_props", []):
                leg_id = self.build_leg_id(game_meta["game_id"], player_id, prop["prop_id"])
                leg_metas.append(
                    {
                        "leg_id": leg_id,
                        "player_id": player_id,
                        "prop_id": prop["prop_id"],
                        "game_id": game_meta["game_id"],
                        "stat_type": prop["stat_type"],
                        "market_side": prop["market_side"],
                        "line": prop["line"],
                    }
                )
        return roster, leg_metas

    def build_phase1_and_phase2_for_game(self, game_meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        roster, leg_metas = self._collect_leg_metas_for_game(game_meta)
        context_id = game_meta["slate_id"]
        for leg_meta in leg_metas:
            self.phase1.derive_leg_foundation(leg_meta, game_meta)
            self.phase2.derive_family_n(leg_meta, context_id)
            self.phase2.execute_efficiency_audit(leg_meta["leg_id"], context_id)
            self.phase2.enforce_execution_firewall(leg_meta["leg_id"], context_id)
            self.phase2.emit_final_leg_governance(leg_meta, context_id)
        self.phase2.generate_eligible_over_universe(game_meta, roster, leg_metas)
        return roster, leg_metas

    def build_portfolio(self) -> Dict[str, Any]:
        self.bootstrap()
        family_p = self.read_state(NodeKey("Phase0", "Family_P", "global", "run")).value
        games = list(family_p["games"])
        universes: List[Dict[str, Any]] = []
        all_leg_meta_by_id: Dict[str, Dict[str, Any]] = {}
        for game_meta in games:
            _, leg_metas = self.build_phase1_and_phase2_for_game(game_meta)
            all_leg_meta_by_id.update({lm["leg_id"]: lm for lm in leg_metas})
            universe_node = NodeKey("Phase2", "eligible_over_leg_universe", game_meta["game_id"], game_meta["slate_id"])
            universes.append(self.read_state(universe_node).value)

        payout_bands = family_p["payout_bands"]
        slate_id = games[0]["slate_id"] if games else "SLATE"
        candidates = self.phase3.build_candidate_tickets(slate_id, universes, payout_bands)

        terminal_tickets: List[Dict[str, Any]] = []
        rejected_tickets: List[Dict[str, Any]] = []
        unfillable_bands: List[Dict[str, Any]] = []
        band_accepts: Dict[str, int] = defaultdict(int)
        eligible_universe_size = sum(u["universe_size"] for u in universes)

        for ticket in candidates:
            # Unfillable markers are sentinel dicts emitted by Phase 3 when a band
            # cannot be filled even after the expanded retry pass. Route them directly
            # to the unfillable_bands list without attempting pipeline processing.
            if ticket.get("code", "").endswith("_UNFILLABLE"):
                unfillable_bands.append({
                    "band": ticket["payout_band_target"],
                    "code": ticket["code"],
                    "shortfall": ticket.get("shortfall", -1),
                })
                continue

            context_id = ticket.get("context_id")
            if not context_id:
                rejected_tickets.append({"ticket_id": ticket.get("ticket_id", "UNKNOWN"), "reason": "MISSING_CONTEXT_ID"})
                continue

            joint = self.phase3.derive_copula_joint_prob(ticket, context_id)
            if not joint.eligible:
                rejected_tickets.append({"ticket_id": ticket["ticket_id"], "reason": joint.code})
                continue
            acceptance = self.phase4.evaluate_ticket(slate_id, ticket, payout_bands, eligible_universe_size)
            if not acceptance.eligible:
                rejected_tickets.append({"ticket_id": ticket["ticket_id"], "reason": acceptance.code})
                continue
            term = self.phase5.seal_terminal_ticket(ticket["ticket_id"], context_id, ticket["leg_ids"])
            if not term.eligible:
                rejected_tickets.append({"ticket_id": ticket["ticket_id"], "reason": term.code})
                continue
            terminal_tickets.append(term.value)
            band_accepts[acceptance.value["payout_band"]] += 1
            if len(terminal_tickets) >= 21:
                break

        portfolio_ok = (
            len(terminal_tickets) == 21
            and not unfillable_bands
            and all(band_accepts.get(b, 0) == payout_bands[b]["ticket_count"] for b in payout_bands)
        )

        if unfillable_bands:
            run_status = "BAND_SHORTFALL_AFTER_RETRY"
        elif not portfolio_ok:
            run_status = "INSUFFICIENT_OR_REJECTED"
        else:
            run_status = "SEALED"

        return {
            "snapshot_bundle_id": family_p["snapshot_bundle_id"],
            "ticket_count_built": len(terminal_tickets),
            "tickets": terminal_tickets,
            "tickets_per_band": dict(band_accepts),
            "qualified_player_universe_size": eligible_universe_size,
            "rejected_tickets": rejected_tickets,
            "unfillable_bands": unfillable_bands,
            "portfolio_spread_satisfied": portfolio_ok,
            "run_status": run_status,
            "audit_log_length": len(self.state.audit_log),
        }


__all__ = [
    "NodeKey",
    "StagedOutput",
    "PregameSourceProvider",
    "DictPregameSnapshotProvider",
    "NBAPropEngine",
    "NBAPropEnginePhase0",
    "NBAPropEnginePhase1",
    "NBAPropEnginePhase2",
    "NBAPropEnginePhase3",
    "NBAPropEnginePhase4",
    "NBAPropEnginePhase5",
]
