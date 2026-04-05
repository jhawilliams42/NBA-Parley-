"""
Phase 1 — Distribution Selection and Raw Event Probability
Sections 1.18, 1.19
"""

from __future__ import annotations

import math
from typing import Optional

from ..phase0.constants import DISTRIBUTION_MIN_SAMPLE


# ---------------------------------------------------------------------------
# 1.18 Numeric guards
# ---------------------------------------------------------------------------

def _check_distribution_guards(
    sample_n: int,
    fd_current_line: Optional[float],
    mean_raw: Optional[float],
    std_raw: Optional[float],
    stat_family: str,
) -> Optional[str]:
    """
    Run all Section 1.18 numeric guards.
    Returns error code string if any guard fires, else None.
    """
    if fd_current_line is None:
        return "MISSING_LINE"
    if sample_n < DISTRIBUTION_MIN_SAMPLE:
        return "INSUFFICIENT_SAMPLE"
    if mean_raw is None:
        return "MISSING_MEAN_RAW"
    if std_raw is None:
        return "MISSING_STD_RAW"
    if std_raw <= 0:
        return "INVALID_STD_RAW_NON_POSITIVE"

    positive_families = {"points", "rebounds", "assists", "pra"}
    epsilon = 1e-9
    if stat_family.lower() in positive_families:
        if mean_raw <= epsilon:
            return "INVALID_MEAN_RAW_NON_POSITIVE"

    return None


# ---------------------------------------------------------------------------
# 1.18 Distribution selection
# ---------------------------------------------------------------------------

def select_distribution(
    sample_n: int,
    mean_raw: Optional[float],
    std_raw: Optional[float],
    stat_family: str,
) -> str:
    """
    Select the best distribution from the allowed candidate set.
    Section 1.18. Returns distribution name or 'DISTRIBUTION_INVALID_NUMERICS'.
    """
    epsilon = 1e-9

    if sample_n < DISTRIBUTION_MIN_SAMPLE or mean_raw is None or std_raw is None:
        return "DISTRIBUTION_INVALID_NUMERICS"
    if std_raw <= 0:
        return "DISTRIBUTION_INVALID_NUMERICS"

    # Positive-domain stat families
    positive_families = {"points", "rebounds", "assists", "pra"}
    is_positive_family = stat_family.lower() in positive_families

    if is_positive_family and mean_raw <= epsilon:
        return "DISTRIBUTION_INVALID_NUMERICS"

    # Prefer GAMMA for positive-domain data
    if is_positive_family and mean_raw > 0 and std_raw > 0:
        return "GAMMA"

    # Low-event discrete families → POISSON
    cv = std_raw / mean_raw if mean_raw > epsilon else None
    low_event_families = {"steals", "blocks", "turnovers"}
    if stat_family.lower() in low_event_families and cv is not None:
        # Dispersion tolerance: CV close to 1 (Poisson)
        if abs(cv - 1.0) <= 0.3:
            return "POISSON"

    # Default to NORMAL for continuous data
    return "NORMAL"


# ---------------------------------------------------------------------------
# 1.19 Raw event probability
# ---------------------------------------------------------------------------

def compute_raw_event_prob_over(
    distribution: str,
    fd_current_line: float,
    mean_raw: float,
    std_raw: float,
    sample_values: Optional[list] = None,
) -> dict:
    """
    Compute raw_event_prob_over_current_line and supporting stats.
    Section 1.19.

    Returns dict with:
        raw_event_prob_over_current_line,
        hit_rate_at_line (empirical),
        median, std_dev, interquartile_range, coefficient_of_variation,
        distribution_used, integrity_state
    """
    from scipy import stats as scipy_stats
    import numpy as np

    result: dict = {
        "distribution_used": distribution,
        "fd_current_line": fd_current_line,
    }

    if distribution == "DISTRIBUTION_INVALID_NUMERICS":
        result["raw_event_prob_over_current_line"] = None
        result["integrity_state"] = "INVALID"
        return result

    # Empirical fallback
    if distribution == "EMPIRICAL":
        if not sample_values or len(sample_values) < DISTRIBUTION_MIN_SAMPLE:
            result["raw_event_prob_over_current_line"] = None
            result["integrity_state"] = "INVALID"
            return result
        arr = np.array(sample_values, dtype=float)
        prob_over = float(np.mean(arr > fd_current_line))
        result["raw_event_prob_over_current_line"] = prob_over
        result["hit_rate_at_line"] = float(np.mean(arr >= fd_current_line))
        result["median"] = float(np.median(arr))
        result["std_dev"] = float(np.std(arr))
        result["interquartile_range"] = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        result["coefficient_of_variation"] = (
            float(result["std_dev"] / mean_raw) if mean_raw > 1e-9 else None
        )
        result["integrity_state"] = "DERIVED_BY_APPROVED_RULE"
        return result

    # Parametric distributions
    try:
        if distribution == "NORMAL":
            dist = scipy_stats.norm(loc=mean_raw, scale=std_raw)
        elif distribution == "LOGNORMAL":
            # Parameterize from mean and std via method of moments
            variance = std_raw ** 2
            mu_ln = math.log(mean_raw ** 2 / math.sqrt(variance + mean_raw ** 2))
            sigma_ln = math.sqrt(math.log(1 + variance / (mean_raw ** 2)))
            dist = scipy_stats.lognorm(s=sigma_ln, scale=math.exp(mu_ln))
        elif distribution == "GAMMA":
            # Method of moments: shape=k, scale=theta
            # mean = k*theta, var = k*theta^2
            variance = std_raw ** 2
            theta = variance / mean_raw
            k = mean_raw / theta
            dist = scipy_stats.gamma(a=k, scale=theta)
        elif distribution == "POISSON":
            dist = scipy_stats.poisson(mu=mean_raw)
            # For discrete: P(X > line) = 1 - P(X <= floor(line))
            prob_over = float(1 - dist.cdf(math.floor(fd_current_line)))
            result["raw_event_prob_over_current_line"] = prob_over
            result["median"] = float(dist.median())
            result["std_dev"] = float(dist.std())
            result["coefficient_of_variation"] = (
                float(dist.std() / mean_raw) if mean_raw > 1e-9 else None
            )
            result["integrity_state"] = "DERIVED_BY_APPROVED_RULE"
            return result
        else:
            result["raw_event_prob_over_current_line"] = None
            result["integrity_state"] = "INVALID"
            return result

        prob_over = float(1 - dist.cdf(fd_current_line))
        result["raw_event_prob_over_current_line"] = prob_over
        result["median"] = float(dist.median())
        result["std_dev"] = float(dist.std())
        iqr = float(dist.ppf(0.75) - dist.ppf(0.25))
        result["interquartile_range"] = iqr
        result["coefficient_of_variation"] = (
            float(dist.std() / mean_raw) if mean_raw > 1e-9 else None
        )
        result["integrity_state"] = "DERIVED_BY_APPROVED_RULE"

    except Exception as exc:  # noqa: BLE001
        result["raw_event_prob_over_current_line"] = None
        result["integrity_state"] = "INVALID"
        result["error"] = str(exc)

    return result


def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds."""
    if american_odds >= 100:
        return (american_odds / 100) + 1.0
    elif american_odds <= -100:
        return (100 / abs(american_odds)) + 1.0
    raise ValueError(f"Invalid American odds: {american_odds}")


def american_to_implied_prob(american_odds: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    decimal = american_to_decimal(american_odds)
    return 1.0 / decimal
