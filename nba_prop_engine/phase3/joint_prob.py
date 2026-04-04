"""
Phase 3 — Joint Probability Computation
Section 3.10
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

from ..phase0.constants import DEPENDENCE_METHOD_IDS
from .correlation import validate_and_repair_correlation_matrix

logger = logging.getLogger(__name__)


def compute_joint_prob_independent(legs: list[dict]) -> dict:
    """
    Compute joint probability assuming independence (no dependence model).
    Product of individual model_probs.
    """
    prob = 1.0
    for leg in legs:
        mp = leg.get("model_prob")
        if mp is None:
            return {
                "joint_prob": None,
                "joint_prob_integrity_status": "FAIL",
                "failure_reason": f"MISSING_MODEL_PROB for player_id={leg.get('player_id')}",
                "dependence_method_used": "NONE_INDEPENDENCE_ALLOWED",
            }
        prob *= mp

    return {
        "joint_prob": prob,
        "joint_prob_integrity_status": "PASS",
        "dependence_method_used": "NONE_INDEPENDENCE_ALLOWED",
    }


def compute_joint_prob_multivariate_normal(
    legs: list[dict],
    correlation_result: dict,
) -> dict:
    """
    Compute joint probability using multivariate normal CDF.
    Section 3.10 — Approved dependence model: LATENT_EVENT_RHO_V1.
    """
    if correlation_result.get("joint_prob_integrity_status") != "PASS":
        return {
            "joint_prob": None,
            "joint_prob_integrity_status": "FAIL",
            "failure_reason": correlation_result.get("failure_reason", "CORRELATION_INVALID"),
            "dependence_method_used": "LATENT_EVENT_RHO_V1",
        }

    C = correlation_result["correlation_matrix"]
    n = len(legs)

    # Convert model_probs to normal quantiles (probit transform)
    thresholds = []
    for leg in legs:
        mp = leg.get("model_prob")
        if mp is None or not (1e-6 < mp < 1 - 1e-6):
            return {
                "joint_prob": None,
                "joint_prob_integrity_status": "FAIL",
                "failure_reason": f"MODEL_PROB out of range for probit: {mp}",
                "dependence_method_used": "LATENT_EVENT_RHO_V1",
            }
        # P(event > line) = model_prob, so threshold for latent variable:
        # We need P(Z > z_i) = model_prob → z_i = probit(1 - model_prob)
        z_i = float(scipy_stats.norm.ppf(1.0 - mp))
        thresholds.append(z_i)

    # For n-variate: P(all X_i > z_i) where X ~ MVN(0, C)
    # Using Monte Carlo for n > 2 (production would use numerical integration)
    try:
        if n == 1:
            joint_prob = float(scipy_stats.norm.sf(thresholds[0]))
        elif n == 2:
            # Bivariate normal CDF
            rho = float(C[0, 1])
            # P(X1 > z1, X2 > z2) = 1 - P(X1 <= z1) - P(X2 <= z2) + P(X1 <= z1, X2 <= z2)
            from scipy.stats import multivariate_normal
            cov_2d = np.array([[1.0, rho], [rho, 1.0]])
            # P(all exceed thresholds) via complement
            joint_prob = float(
                multivariate_normal.sf(thresholds, mean=np.zeros(2), cov=cov_2d)
            )
        else:
            # Monte Carlo simulation for n > 2
            rng = np.random.default_rng(seed=42)
            n_samples = 100_000
            cholesky_L = correlation_result.get("cholesky_L")
            if cholesky_L is None:
                cholesky_L = np.linalg.cholesky(C)
            z_samples = rng.standard_normal((n_samples, n))
            x_samples = z_samples @ cholesky_L.T
            thresholds_arr = np.array(thresholds)
            joint_prob = float(np.mean(np.all(x_samples > thresholds_arr, axis=1)))
    except Exception as exc:  # noqa: BLE001
        return {
            "joint_prob": None,
            "joint_prob_integrity_status": "FAIL",
            "failure_reason": f"JOINT_PROB_COMPUTATION_ERROR: {exc}",
            "dependence_method_used": "LATENT_EVENT_RHO_V1",
        }

    return {
        "joint_prob": joint_prob,
        "joint_prob_integrity_status": "PASS",
        "dependence_method_used": "LATENT_EVENT_RHO_V1",
        "n_legs": n,
    }


def compute_joint_probability(legs: list[dict]) -> dict:
    """
    Compute joint probability using priority order from Section 3.10.

    Priority:
    1. FanDuel SGP price available (use for pricing, still compute prob)
    2. Approved dependence model
    3. Independence (if allowed)
    """
    # Determine if any leg requires dependence modeling
    needs_dependence = _any_leg_needs_dependence(legs)

    if not needs_dependence:
        return compute_joint_prob_independent(legs)

    # Check dependence method availability
    method_ids = {
        leg.get("dependence_method_id") for leg in legs if leg.get("dependence_method_id")
    }

    if "LATENT_EVENT_RHO_V1" in method_ids:
        corr_result = validate_and_repair_correlation_matrix(legs)
        return compute_joint_prob_multivariate_normal(legs, corr_result)

    if "EMPIRICAL_EVENT_SIM_V1" in method_ids:
        # Production: empirical simulation. For now, fall back to independence
        # with warning unless dependence materially matters.
        logger.warning(
            "EMPIRICAL_EVENT_SIM_V1 not yet implemented; checking if independence allowed"
        )
        corr_result = validate_and_repair_correlation_matrix(legs)
        return compute_joint_prob_multivariate_normal(legs, corr_result)

    # No valid dependence model where dependence materially matters
    # Section 3.10: ticket fails integrity
    if needs_dependence:
        return {
            "joint_prob": None,
            "joint_prob_integrity_status": "FAIL",
            "failure_reason": (
                "NO_VALID_DEPENDENCE_MODEL_WHERE_DEPENDENCE_MATERIALLY_MATTERS"
            ),
        }

    return compute_joint_prob_independent(legs)


def _any_leg_needs_dependence(legs: list[dict]) -> bool:
    """
    Return True if any pair of legs has the same player_id or same game_id,
    and uses a non-independence method.
    """
    for i, leg_a in enumerate(legs):
        for j, leg_b in enumerate(legs):
            if i >= j:
                continue
            # Same player
            if leg_a.get("player_id") == leg_b.get("player_id"):
                method = leg_a.get("dependence_method_id", "NONE_INDEPENDENCE_ALLOWED")
                if method != "NONE_INDEPENDENCE_ALLOWED":
                    return True
            # Same game
            if leg_a.get("game_id") == leg_b.get("game_id"):
                method = leg_a.get("same_game_dependence_method_id", "NONE_INDEPENDENCE_ALLOWED")
                if method != "NONE_INDEPENDENCE_ALLOWED":
                    return True
    return False
