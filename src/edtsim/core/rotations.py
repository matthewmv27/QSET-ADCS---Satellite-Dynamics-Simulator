"""Rotation and small linear-algebra helpers.

Conventions (see docs/interfaces.md):
  * ``R_IB`` maps body-frame vectors to inertial-frame vectors:  v_I = R_IB @ v_B
  * Euler angles are intrinsic-ZYX (yaw-pitch-roll):  R_IB = Rz(yaw) @ Ry(pitch) @ Rx(roll)
  * All angles in this module are in RADIANS.
"""
from __future__ import annotations

import numpy as np

__all__ = ["hat", "unit", "Rx", "Ry", "Rz", "euler_zyx_to_R", "rotation_matrix_from_a_to_b"]


def hat(v) -> np.ndarray:
    """Skew-symmetric (cross-product) matrix: ``hat(a) @ b == np.cross(a, b)``."""
    vx, vy, vz = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -vz, vy],
                     [vz, 0.0, -vx],
                     [-vy, vx, 0.0]], dtype=float)


def unit(v, tol: float = 1e-12) -> np.ndarray:
    """Return ``v / |v|``; raise ``ValueError`` if ``|v| < tol``."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < tol:
        raise ValueError("Cannot normalize a near-zero vector.")
    return v / n


def Rx(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]], dtype=float)


def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=float)


def Rz(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]], dtype=float)


def euler_zyx_to_R(roll_x: float, pitch_y: float, yaw_z: float) -> np.ndarray:
    """R_IB = Rz(yaw) @ Ry(pitch) @ Rx(roll). Angles in radians."""
    return Rz(yaw_z) @ Ry(pitch_y) @ Rx(roll_x)


def rotation_matrix_from_a_to_b(a, b, tol: float = 1e-12) -> np.ndarray:
    """Smallest rotation taking direction ``a`` onto direction ``b`` (Rodrigues).

    Antiparallel inputs return a 180-degree rotation about an arbitrary axis
    perpendicular to ``a``.
    """
    a_hat = unit(a)
    b_hat = unit(b)
    v = np.cross(a_hat, b_hat)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a_hat, b_hat))

    if s < tol:
        if c > 0:
            return np.eye(3)
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, a_hat)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        axis = unit(np.cross(a_hat, tmp))
        K = hat(axis)
        return np.eye(3) + 2.0 * (K @ K)

    K = hat(v / s)
    return np.eye(3) + K * s + (K @ K) * (1.0 - c)
