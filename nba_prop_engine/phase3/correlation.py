"""
Phase 3 — Correlation Matrix Validation and Approval
Section 3.9
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..phase0.constants import (
    CONDITION_NUMBER_THRESHOLD,
    DETERMINANT_THRESHOLD,
    MIN_EIGENVALUE_THRESHOLD,
    REPAIR_NORM_THRESHOLD,
)

logger = logging.getLogger(__name__)


def build_correlation_matrix(legs: list[dict]) -> np.ndarray:
    """
    Build a correlation matrix from leg dependence data. Section 3.9.
    Off-diagonal entries come from leg pair_correlation if set,
    defaulting to 0.0 (independence).
    """
    n = len(legs)
    C = np.eye(n, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            rho = _get_pair_correlation(legs[i], legs[j])
            C[i, j] = rho
            C[j, i] = rho

    return C


def _get_pair_correlation(leg_a: dict, leg_b: dict) -> float:
    """
    Retrieve the pairwise correlation between two legs.
    Checks for same-player or same-game declared correlation.
    """
    # Same-player legs
    if leg_a.get("player_id") == leg_b.get("player_id"):
        method_id = leg_a.get("dependence_method_id") or leg_b.get("dependence_method_id")
        if method_id and method_id != "NONE_INDEPENDENCE_ALLOWED":
            return leg_a.get("same_player_rho", 0.5)

    # Same-game different-player legs
    if leg_a.get("game_id") == leg_b.get("game_id"):
        method_id = leg_a.get("same_game_dependence_method_id") or leg_b.get(
            "same_game_dependence_method_id"
        )
        if method_id and method_id != "NONE_INDEPENDENCE_ALLOWED":
            return leg_a.get("same_game_rho", 0.2)

    return 0.0


def validate_psd(C: np.ndarray) -> tuple[bool, np.ndarray]:
    """Check if matrix is positive semi-definite. Returns (is_psd, eigenvalues)."""
    eigenvalues = np.linalg.eigvalsh(C)
    return bool(np.all(eigenvalues >= 0)), eigenvalues


def higham_nearest_corr(
    A: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, int, bool, float]:
    """
    Compute the nearest positive semidefinite correlation matrix using
    Higham's (2002) alternating projections algorithm.

    Returns: (C_repaired, iterations, converged, repair_norm)
    """
    n = A.shape[0]
    Y = A.copy()
    S = np.zeros_like(A)

    for iteration in range(max_iter):
        R = Y - S
        # Project onto PSD cone
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        eigenvalues = np.maximum(eigenvalues, 0)
        X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        S = X - R
        # Project onto unit diagonal (correlation matrix)
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)

        # Convergence check
        diff = np.linalg.norm(Y - X, "fro")
        if diff < tol:
            repair_norm = float(np.linalg.norm(Y - A, "fro"))
            return Y, iteration + 1, True, repair_norm

    repair_norm = float(np.linalg.norm(Y - A, "fro"))
    return Y, max_iter, False, repair_norm


def validate_and_repair_correlation_matrix(legs: list[dict]) -> dict:
    """
    Validate and if necessary repair the correlation matrix for a set of legs.
    Section 3.9 — Convergence required, min eigenvalue + condition number primary.

    Returns comprehensive governance result dict.
    """
    C_raw = build_correlation_matrix(legs)
    is_psd, eigenvalues_raw = validate_psd(C_raw)

    if is_psd:
        C_final = C_raw
        converged = True
        repair_norm = 0.0
        iterations = 0
    else:
        C_repaired, iterations, converged, repair_norm = higham_nearest_corr(C_raw)

        if not converged:
            return {
                "correlation_matrix_integrity_state": "HIGHAM_NON_CONVERGENCE",
                "joint_prob_integrity_status": "FAIL",
                "failure_reason": (
                    f"Higham repair did not converge after {iterations} iterations"
                ),
            }

        if repair_norm > REPAIR_NORM_THRESHOLD:
            return {
                "correlation_matrix_integrity_state": "REPAIR_EXCESSIVE",
                "joint_prob_integrity_status": "FAIL",
                "failure_reason": (
                    f"Repair norm {repair_norm:.4f} exceeds threshold {REPAIR_NORM_THRESHOLD}"
                ),
            }

        C_final = C_repaired

    eigenvalues = np.linalg.eigvalsh(C_final)
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    condition_number = float(max_eigenvalue / min_eigenvalue) if min_eigenvalue > 0 else float("inf")

    if min_eigenvalue < MIN_EIGENVALUE_THRESHOLD:
        return {
            "correlation_matrix_integrity_state": "NEAR_SINGULAR_MIN_EIGENVALUE",
            "joint_prob_integrity_status": "FAIL",
            "failure_reason": (
                f"Min eigenvalue {min_eigenvalue:.2e} below threshold {MIN_EIGENVALUE_THRESHOLD}"
            ),
        }

    if condition_number > CONDITION_NUMBER_THRESHOLD:
        return {
            "correlation_matrix_integrity_state": "ILL_CONDITIONED",
            "joint_prob_integrity_status": "FAIL",
            "failure_reason": (
                f"Condition number {condition_number:.2e} exceeds threshold {CONDITION_NUMBER_THRESHOLD}"
            ),
        }

    try:
        cholesky_L = np.linalg.cholesky(C_final)
        cholesky_status = "SUCCESS"
    except np.linalg.LinAlgError:
        return {
            "correlation_matrix_integrity_state": "CHOLESKY_FAILURE_POST_REPAIR",
            "joint_prob_integrity_status": "FAIL",
            "failure_reason": "Cholesky decomposition failed",
        }

    determinant = float(np.linalg.det(C_final))

    return {
        "correlation_matrix_integrity_state": "VALID",
        "joint_prob_integrity_status": "PASS",
        "correlation_matrix": C_final,
        "higham_converged": converged,
        "higham_iterations": iterations,
        "repair_norm": repair_norm,
        "min_eigenvalue": min_eigenvalue,
        "condition_number": condition_number,
        "determinant": determinant,
        "determinant_status": (
            "LOW_DIAGNOSTIC" if determinant < DETERMINANT_THRESHOLD else "DIAGNOSTIC_OK"
        ),
        "cholesky_status": cholesky_status,
        "cholesky_L": cholesky_L,
    }
