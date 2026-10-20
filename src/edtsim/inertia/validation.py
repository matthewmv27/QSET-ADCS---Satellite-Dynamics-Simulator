"""Sanity checks for inertia tensors."""
from __future__ import annotations

import numpy as np

__all__ = ["principal_moments", "is_physical_inertia"]


def principal_moments(I: np.ndarray):
    """Return ``(moments, axes)``: principal moments (ascending) and matching unit axes as columns."""
    I = np.asarray(I, dtype=float)
    return np.linalg.eigh(0.5 * (I + I.T))


def is_physical_inertia(I: np.ndarray, rtol: float = 1e-9) -> bool:
    """True if ``I`` could be the inertia tensor of a real rigid body.

    Requires: symmetric; positive principal moments; and the triangle inequality
    ``I1 + I2 >= I3`` for the principal moments (up to ``rtol``).
    """
    I = np.asarray(I, dtype=float)
    if I.shape != (3, 3):
        return False
    scale = max(float(np.max(np.abs(I))), 1e-300)
    if not np.allclose(I, I.T, rtol=rtol, atol=rtol * scale):
        return False
    w, _ = principal_moments(I)
    tol = rtol * scale
    if w[0] < -tol:
        return False
    return bool(w[0] + w[1] >= w[2] - tol)
