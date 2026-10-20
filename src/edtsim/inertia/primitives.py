"""Rigid-body inertia building blocks (SI units, kg*m^2).

Reference-point convention
--------------------------
Functions named ``*_about_com`` return the tensor about the body's OWN centre of mass,
expressed in whatever frame the inputs (axes) are given in.
``point_mass_inertia_about_origin`` and ``parallel_axis_shift`` give contributions about
the ORIGIN, where ``r`` is the position of the mass relative to that origin.
In this project the origin is the bus COM and the frame is the body frame unless a name
ends in ``_I`` (inertial axes).
"""
from __future__ import annotations

import numpy as np

from edtsim.core.rotations import unit

__all__ = [
    "parallel_axis_shift",
    "point_mass_inertia_about_origin",
    "box_inertia_about_com",
    "annular_cylinder_inertia_about_com",
    "slender_rod_inertia_about_com",
    "inertia_body_to_inertial",
    "shift_inertia_to_point",
    "shift_inertia_to_com",
]


def parallel_axis_shift(m: float, r) -> np.ndarray:
    """Parallel-axis term ``m (|r|^2 I - r r^T)`` for a mass ``m`` whose COM is at ``r``."""
    r = np.asarray(r, dtype=float).reshape(3)
    return m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def point_mass_inertia_about_origin(m: float, r) -> np.ndarray:
    """Inertia of a point mass ``m`` at position ``r`` about the origin."""
    return parallel_axis_shift(m, r)


def box_inertia_about_com(m: float, a: float, b: float, c: float) -> np.ndarray:
    """Uniform box with side lengths a (x), b (y), c (z); principal axes aligned with the sides."""
    Ixx = (m / 12.0) * (b**2 + c**2)
    Iyy = (m / 12.0) * (a**2 + c**2)
    Izz = (m / 12.0) * (a**2 + b**2)
    return np.diag([Ixx, Iyy, Izz])


def annular_cylinder_inertia_about_com(m: float, r_i: float, r_o: float, h: float, axis_s) -> np.ndarray:
    """Uniform annular cylinder (inner radius r_i, outer r_o, height h) with symmetry axis ``axis_s``.

    I_parallel = m (r_i^2 + r_o^2) / 2
    I_perp     = m (3 (r_i^2 + r_o^2) + h^2) / 12
    """
    s = unit(axis_s)
    I_parallel = 0.5 * m * (r_i**2 + r_o**2)
    I_perp = (m / 12.0) * (3.0 * (r_i**2 + r_o**2) + h**2)
    P = np.outer(s, s)
    return I_perp * (np.eye(3) - P) + I_parallel * P


def slender_rod_inertia_about_com(m: float, L: float, axis_e) -> np.ndarray:
    """Thin uniform rod of length L along ``axis_e``: zero about its axis, m L^2 / 12 about any perpendicular axis."""
    e = unit(axis_e)
    return (m * L**2 / 12.0) * (np.eye(3) - np.outer(e, e))


def inertia_body_to_inertial(I_B: np.ndarray, R_IB: np.ndarray) -> np.ndarray:
    """Express a body-frame inertia tensor in inertial axes: ``R_IB @ I_B @ R_IB^T``."""
    return R_IB @ I_B @ R_IB.T


def shift_inertia_to_point(I_com: np.ndarray, m: float, d) -> np.ndarray:
    """Inertia about point P given the inertia about the body's COM.

    ``d`` = (COM position) - (P position), in the same axes as ``I_com``.
    """
    return I_com + parallel_axis_shift(m, d)


def shift_inertia_to_com(I_P: np.ndarray, m: float, d) -> np.ndarray:
    """Inverse of :func:`shift_inertia_to_point`.

    ``d`` = (COM position) - (P position). Use this to move an inertia that is
    about the bus COM onto the (moving) system COM for the coupled model.
    """
    return I_P - parallel_axis_shift(m, d)
