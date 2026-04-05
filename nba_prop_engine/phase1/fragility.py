"""
Phase 1 — Fragility Model: Epistemic Uncertainty as Disqualifier
Section 1.16
"""

from __future__ import annotations

from typing import Optional

from ..phase0.constants import (
    FRAGILITY_CLASS_VALUES,
    FRAGILITY_HIGH_UNCERTAINTY_THRESHOLD,
    FRAGILITY_SCORE_MAP,
    FRAGILITY_WEIGHTS,
)


def compute_fragility_score(obj: dict) -> dict:
    """
    Compute composite fragility score and class from 8 component inputs.
    Section 1.16.

    Expects obj to have keys:
        minutes_fragility_component, injury_fragility_component,
        foul_fragility_component, role_fragility_component,
        rotation_fragility_component, blowout_fragility_component,
        dependency_fragility_component, uncertainty_fragility_component

    Each component should be one of: 'LOW', 'MODERATE', 'HIGH', 'UNKNOWN'.

    Returns dict with:
        minutes_fragility_class, fragility_score, fragility_integrity_status,
        epistemic_disqualifier, unknown_component_count, unknown_components
    """
    components = {
        "minutes": obj.get("minutes_fragility_component"),
        "injury": obj.get("injury_fragility_component"),
        "foul": obj.get("foul_fragility_component"),
        "role": obj.get("role_fragility_component"),
        "rotation": obj.get("rotation_fragility_component"),
        "blowout": obj.get("blowout_fragility_component"),
        "dependency": obj.get("dependency_fragility_component"),
        "uncertainty": obj.get("uncertainty_fragility_component"),
    }

    unknown_count = sum(1 for v in components.values() if v == "UNKNOWN")
    unknown_components = [k for k, v in components.items() if v == "UNKNOWN"]

    if unknown_count >= FRAGILITY_HIGH_UNCERTAINTY_THRESHOLD:
        return {
            "minutes_fragility_class": "HIGH_UNCERTAINTY",
            "fragility_score": None,
            "fragility_integrity_status": "EPISTEMIC_RISK_FLAG",
            "epistemic_disqualifier": True,
            "unknown_component_count": unknown_count,
            "unknown_components": unknown_components,
        }

    weighted_score = sum(
        FRAGILITY_WEIGHTS[comp] * FRAGILITY_SCORE_MAP.get(components[comp], 2.5)
        for comp in components
    )

    if weighted_score < 1.5:
        fragility_class = "LOW"
    elif weighted_score < 2.3:
        fragility_class = "MODERATE"
    else:
        fragility_class = "HIGH"

    return {
        "minutes_fragility_class": fragility_class,
        "fragility_score": weighted_score,
        "fragility_integrity_status": "VERIFIED",
        "epistemic_disqualifier": False,
        "unknown_component_count": unknown_count,
        "unknown_components": unknown_components,
    }


def fragility_class_is_high(minutes_fragility_class: str) -> bool:
    """Return True if fragility class is HIGH or HIGH_UNCERTAINTY."""
    return minutes_fragility_class in ("HIGH", "HIGH_UNCERTAINTY")


def validate_fragility_class(value: str) -> bool:
    """Return True if value is a recognized fragility class."""
    return value in FRAGILITY_CLASS_VALUES
