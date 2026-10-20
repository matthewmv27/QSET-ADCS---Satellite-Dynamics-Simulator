"""Inertia of a collection of point masses (used for the bent/settled tether)."""
from __future__ import annotations

import numpy as np

__all__ = ["inertia_from_particles"]


def inertia_from_particles(r_particles: np.ndarray, m_particles: np.ndarray, r_ref) -> np.ndarray:
    """Inertia tensor of point masses about the reference point ``r_ref``.

    Parameters
    ----------
    r_particles : (N, 3) positions, in any frame.
    m_particles : (N,) masses.
    r_ref       : (3,) reference point, in the SAME frame as ``r_particles``.

    Returns
    -------
    (3, 3) tensor in the same axes as the inputs, about ``r_ref``.

    Legacy name: ``compute_inertia_from_particles_about_bus_com_in_inertial``.
    """
    r = np.asarray(r_particles, dtype=float)
    m = np.asarray(m_particles, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"r_particles must have shape (N, 3), got {r.shape}.")
    if m.shape != (r.shape[0],):
        raise ValueError(f"m_particles must have shape ({r.shape[0]},), got {m.shape}.")
    rel = r - np.asarray(r_ref, dtype=float).reshape(1, 3)
    d2 = np.einsum("ij,ij->i", rel, rel)
    return float(np.dot(m, d2)) * np.eye(3) - np.einsum("i,ij,ik->jk", m, rel, rel)
