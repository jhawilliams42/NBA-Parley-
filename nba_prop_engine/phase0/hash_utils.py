"""
Phase 0 — Hash Contract: RFC 8785 JSON Canonicalization + SHA-256
Section 0.10
"""

import hashlib
import math
from typing import Any


def _validate_no_nan_inf(obj: Any, path: str = "") -> None:
    """
    Section 0.10 Rule 8: NaN, Infinity, and -Infinity are forbidden in hash
    inputs. Raise ValueError on encounter.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Forbidden float value '{obj}' at path '{path}' — "
                "NaN and Infinity are prohibited in hash inputs (Section 0.10 Rule 8)"
            )
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _validate_no_nan_inf(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _validate_no_nan_inf(v, f"{path}[{i}]")


def compute_hash(obj: dict) -> str:
    """
    Compute SHA-256 hash of RFC 8785 canonical JSON serialization.
    Section 0.10 — Canonical Hash Computation.

    Args:
        obj: Dictionary to hash. Must not contain NaN or Infinity values.

    Returns:
        Lowercase hex SHA-256 digest string.

    Raises:
        ValueError: If obj contains NaN, Infinity, or -Infinity.
        TypeError: If obj cannot be canonicalized.
    """
    import jcs

    _validate_no_nan_inf(obj)
    canonical_bytes = jcs.canonicalize(obj)
    return hashlib.sha256(canonical_bytes).hexdigest()


def verify_hash(input_object: dict, hash_field: str, phase: int) -> dict:
    """
    Verify a phase handoff hash per Section 0.10 Hash Verification Protocol.

    Args:
        input_object: The full object including the hash field.
        hash_field: The field name containing the expected hash
                    (e.g., 'phase1_frozen_hash').
        phase: Phase number for error emission context.

    Returns:
        Dict with keys:
          - 'valid': bool
          - 'received_hash': str
          - 'recomputed_hash': str or None
          - 'error': str or None
    """
    received_hash = input_object.get(hash_field)
    if received_hash is None:
        return {
            "valid": False,
            "received_hash": None,
            "recomputed_hash": None,
            "error": f"Hash field '{hash_field}' is missing from object",
        }

    payload = {k: v for k, v in input_object.items() if k != hash_field}

    try:
        recomputed_hash = compute_hash(payload)
    except (ValueError, TypeError) as exc:
        return {
            "valid": False,
            "received_hash": received_hash,
            "recomputed_hash": None,
            "error": str(exc),
        }

    if recomputed_hash != received_hash:
        return {
            "valid": False,
            "received_hash": received_hash,
            "recomputed_hash": recomputed_hash,
            "error": (
                f"HASH_MISMATCH_DETECTED at phase {phase}: "
                f"received={received_hash}, recomputed={recomputed_hash}"
            ),
        }

    return {
        "valid": True,
        "received_hash": received_hash,
        "recomputed_hash": recomputed_hash,
        "error": None,
    }


def freeze_object_with_hash(obj: dict, hash_field: str) -> dict:
    """
    Add a phase-handoff hash to obj and return the augmented dict.
    The hash is computed over all fields except hash_field itself.

    Args:
        obj: Object to freeze.
        hash_field: Name for the hash field to add
                    (e.g., 'phase1_frozen_hash').

    Returns:
        Copy of obj with hash_field populated.
    """
    payload = {k: v for k, v in obj.items() if k != hash_field}
    digest = compute_hash(payload)
    result = dict(obj)
    result[hash_field] = digest
    return result
