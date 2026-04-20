"""
Phase 0 — Constants, Thresholds, and Registries
Section 0.7, 0.14, 0.18, 0.19, 3.9, 4.4
"""

SCHEMA_VERSION = "v15.1"

# ---------------------------------------------------------------------------
# 0.7 Binding Field Integrity States
# ---------------------------------------------------------------------------
FIELD_INTEGRITY_STATES = frozenset({
    "VERIFIED",
    "DERIVED_BY_APPROVED_RULE",
    "MISSING",
    "INVALID",
    "UNRESOLVABLE",
    "PROHIBITED_INFERENCE",
    "DERIVED_FROM_MISSING_INPUT",
    "DEPENDENCY_FAILURE",
    "RECOMPUTED_POST_FREEZE",
    "PROVISIONAL_PENDING_FREEZE",
    "RECOMPUTE_FAILED_NON_BINDING",
})

# ---------------------------------------------------------------------------
# 0.14 Source Classes
# ---------------------------------------------------------------------------
SOURCE_CLASSES = frozenset({
    "EXECUTION_BINDING",
    "OFFICIAL_PRIMARY",
    "APPROVED_SECONDARY_VERIFICATION",
    "APPROVED_EXTERNAL_VALUATION",
    "APPROVED_DERIVATION_SUPPORT",
    "FUTURE_RESERVED_SOURCE",
})

# ---------------------------------------------------------------------------
# 0.20 Kill Switch States
# ---------------------------------------------------------------------------
KILL_SWITCH_STATES = frozenset({
    "SNAPSHOT_INTEGRITY_FAIL",
    "EXECUTION_MARKET_MISSING",
    "OFFICIAL_STATUS_MISSING",
    "HASH_MISMATCH",
    "PHASE_CONTRACT_VIOLATION",
    "PROHIBITED_INFERENCE_DETECTED",
    "CORRELATION_MODEL_INVALID",
    "EXECUTION_VALUATION_CONTAMINATION",
    "FANDUEL_BINDING_FIELD_CONTAMINATION",
})

# ---------------------------------------------------------------------------
# 0.22 run_context execution book
# ---------------------------------------------------------------------------
EXECUTION_BOOK = "FANDUEL"

# ---------------------------------------------------------------------------
# 1.5 Allowed normalized_status values
# ---------------------------------------------------------------------------
NORMALIZED_STATUS_VALUES = frozenset({
    "ACTIVE",
    "GTD",
    "QUESTIONABLE",
    "DOUBTFUL",
    "OUT",
    "INACTIVE_OTHER",
    "UNRESOLVABLE",
    "ACTIVE_PENDING_VERIFICATION",
})

# Statuses that block Phase 1 processing
BLOCKING_STATUSES = frozenset({"OUT", "INACTIVE_OTHER", "UNRESOLVABLE"})

# Statuses allowed into Phase 2
PHASE2_ALLOWED_STATUSES = frozenset({"ACTIVE", "GTD", "QUESTIONABLE"})

# ---------------------------------------------------------------------------
# 1.14 Allowed functional_status_class values
# ---------------------------------------------------------------------------
FUNCTIONAL_STATUS_CLASS_VALUES = frozenset({
    "CLEAN",
    "LIMITED",
    "FRAGILE",
    "HIGH_UNCERTAINTY",
    "UNRESOLVABLE",
})

# ---------------------------------------------------------------------------
# 1.16 Fragility class values
# ---------------------------------------------------------------------------
FRAGILITY_CLASS_VALUES = frozenset({
    "LOW",
    "MODERATE",
    "HIGH",
    "HIGH_UNCERTAINTY",
    "UNRESOLVABLE",
})

FRAGILITY_WEIGHTS = {
    "minutes": 0.20,
    "injury": 0.20,
    "foul": 0.10,
    "role": 0.15,
    "rotation": 0.15,
    "blowout": 0.10,
    "dependency": 0.05,
    "uncertainty": 0.05,
}

FRAGILITY_SCORE_MAP = {
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "UNKNOWN": 2.5,
}

FRAGILITY_HIGH_UNCERTAINTY_THRESHOLD = 3  # unknown_count >= 3 triggers disqualifier

# ---------------------------------------------------------------------------
# 1.17 Repeatability class values
# ---------------------------------------------------------------------------
REPEATABILITY_CLASS_VALUES = frozenset({
    "HIGH",
    "MODERATE",
    "LOW",
    "UNKNOWN",
})

# ---------------------------------------------------------------------------
# 1.18 Distribution candidates
# ---------------------------------------------------------------------------
DISTRIBUTION_CANDIDATES = frozenset({
    "EMPIRICAL",
    "NORMAL",
    "LOGNORMAL",
    "GAMMA",
    "POISSON",
})

DISTRIBUTION_MIN_SAMPLE = 10  # parametric fit prohibited below this

# ---------------------------------------------------------------------------
# 1.15A Provisional fields requiring tracking
# ---------------------------------------------------------------------------
PROVISIONAL_DERIVED_FIELDS = [
    "opportunity_context_class",
    "blowout_fragility",
    "role_fragility",
    "minutes_projection_first_pass",
    "usage_rate_adjusted",
    "expected_field_goal_attempts",
    "expected_touches",
    "rotation_volatility_derived",
]

# ---------------------------------------------------------------------------
# 1.22A Binding vs non-binding field classification
# ---------------------------------------------------------------------------
BINDING_FIELDS = [
    "opportunity_context_class",
    "minutes_fragility_class",
    "role_lock_class",
    "repeatability_class",
    "functional_status_class",
]

NON_BINDING_FIELDS = [
    "lineup_continuity_score",
    "coaching_volatility_class",
]

# ---------------------------------------------------------------------------
# 1.10A Dependency lifecycle states
# ---------------------------------------------------------------------------
DEPENDENCY_LIFECYCLE_STATES = frozenset({
    "NOT_COMPUTED",
    "COMPUTING",
    "VALID",
    "FAILED",
})

# ---------------------------------------------------------------------------
# 1.21 Staleness threshold (seconds)
# ---------------------------------------------------------------------------
STALENESS_THRESHOLD_SECONDS = 60

# ---------------------------------------------------------------------------
# 0.21 Snapshot atomicity max delta
# ---------------------------------------------------------------------------
MAX_SNAPSHOT_DELTA_SECONDS = 180

# ---------------------------------------------------------------------------
# 2.5 De-vig method registry (Section 2.5)
# ---------------------------------------------------------------------------
DEVIG_METHOD_REGISTRY = {
    "SHIN_V1": {
        "formal_name": "Shin Implicit Vig Model",
        "validation_status": "SPECIFIED_NOT_PRODUCTION_AUTHORIZED",
        "production_authorized": False,
        "required_before_production": [
            "reference implementation agreement",
            "convergence test coverage",
            "edge case validation",
        ],
    },
    "PROPORTIONAL_V1": {
        "formal_name": "Proportional Normalization",
        "validation_status": "PRODUCTION_AUTHORIZED",
        "production_authorized": True,
        "required_before_production": [],
    },
}

# ---------------------------------------------------------------------------
# 2.8 Default shrinkage table
# ---------------------------------------------------------------------------
SHRINKAGE_TABLE = {
    ("HIGH", "strong"): 0.00,
    ("MODERATE", "strong"): 0.05,
    ("MODERATE", "weak"): 0.15,
    ("LOW", None): 0.25,
    ("UNKNOWN", None): 0.20,
}

SAMPLE_N_STRONG_THRESHOLD = 20

# ---------------------------------------------------------------------------
# 2.12 Gate C branch thresholds
# ---------------------------------------------------------------------------
GATE_C_BRANCH1_EDGE_MIN = 0.03
GATE_C_BRANCH2_BOOK_COUNT_MIN = 3
GATE_C_BRANCH2_DISAGREEMENT_MAX = 0.05
GATE_C_BRANCH2_EDGE_MIN = 0.05
GATE_C_BRANCH2_MODEL_PROB_MIN = 0.55
GATE_C_BRANCH3_EDGE_MIN = 0.08
GATE_C_BRANCH3_MODEL_PROB_MIN = 0.60
GATE_C_BRANCH3_SAMPLE_N_MIN = 20
GATE_C_BRANCH3_MAX_PER_PORTFOLIO = 2

GATE_B_MODEL_PROB_FLOOR = 0.50

# ---------------------------------------------------------------------------
# 2.12 Gate D GTD thresholds
# ---------------------------------------------------------------------------
GATE_D_GTD_PLAY_RATE_MIN = 0.35
GATE_D_GTD_TIP_SECONDS_THRESHOLD = 3600

# ---------------------------------------------------------------------------
# 3.9 Correlation matrix thresholds
# ---------------------------------------------------------------------------
REPAIR_NORM_THRESHOLD = 0.15
MIN_EIGENVALUE_THRESHOLD = 1e-6
CONDITION_NUMBER_THRESHOLD = 1e4
DETERMINANT_THRESHOLD = 1e-8

# ---------------------------------------------------------------------------
# 4.4 Kelly epsilon constants
# ---------------------------------------------------------------------------
EPSILON_B = 0.01
EPSILON_P_LOW = 0.01
EPSILON_P_HIGH = 0.99

# ---------------------------------------------------------------------------
# 2.13 Bucket classification
# ---------------------------------------------------------------------------
BUCKET_VALUES = frozenset({
    "STAR",
    "ELITE",
    "CATALYST",
    "SOLID",
    "GOVERNANCE_LIMITED",
    "INELIGIBLE",
})

# ---------------------------------------------------------------------------
# 3.8 Dependence method IDs
# ---------------------------------------------------------------------------
DEPENDENCE_METHOD_IDS = frozenset({
    "LATENT_EVENT_RHO_V1",
    "EMPIRICAL_EVENT_SIM_V1",
    "NONE_INDEPENDENCE_ALLOWED",
})

# ---------------------------------------------------------------------------
# 5.3 Tier model labels
# ---------------------------------------------------------------------------
TIER_LABELS = frozenset({
    "TIER_1",
    "TIER_2",
    "TIER_3",
    "TIER_4",
    "TIER_5",
})

# ---------------------------------------------------------------------------
# 5.2 Portfolio target
# ---------------------------------------------------------------------------
PORTFOLIO_TARGET_TICKETS = 21

# ---------------------------------------------------------------------------
# 0.16 FanDuel execution field families
# ---------------------------------------------------------------------------
FANDUEL_EXECUTION_FIELDS = frozenset({
    "fd_open_line",
    "fd_current_line",
    "fd_open_odds_american_over",
    "fd_open_odds_american_under",
    "fd_current_odds_american_over",
    "fd_current_odds_american_under",
    "fd_prop_market_status",
    "fd_sgp_supported",
    "fd_sgp_price_american",
    "fd_sgp_price_decimal",
    "fd_ticket_price_american",
    "fd_ticket_price_decimal",
    "fd_break_even",
    "fd_execution_side",
    "fd_execution_market_source_url",
    "fd_execution_market_source_ts_utc",
    "fd_ticket_price_source_name",
    "fd_ticket_price_source_url",
    "fd_ticket_price_source_ts_utc",
    "fd_ticket_price_integrity_state",
    "fd_execution_book",
})

# ---------------------------------------------------------------------------
# 3.3 Ticket family model
# ---------------------------------------------------------------------------
TICKET_FAMILY_TYPES = frozenset({
    "STANDARD_SHORT",
    "STANDARD_MID",
    "STANDARD_LONG",
    "SGP_SHORT",
    "SGP_MID",
    "SGP_LONG",
})
