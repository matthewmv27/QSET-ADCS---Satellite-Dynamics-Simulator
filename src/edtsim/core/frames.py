"""Quaternion/coordinate utilities for the shared EDTSim frame convention.

Hamilton, scalar-first quaternion q_IB = (w, x, y, z) maps B coordinates to I.
Right-handed active rotation: v_I = R_IB(q_IB) @ v_B.
These helpers define kinematics/interfaces, NOT an attitude dynamics solver.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_quaternion",
    "quaternion_multiply",
    "quaternion_from_euler_zyx",
    "rotation_from_quaternion",
    "quaternion_rate_from_body_omega",
    "attachment_kinematics",
    "attachment_reaction_on_bus",
]


def _vector(x, n: int, name: str) -> np.ndarray:
    value = np.asarray(x, dtype=float)
    if value.shape != (n,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite vector with shape ({n},).")
    return value


def normalize_quaternion(q) -> np.ndarray:
    """Explicitly normalize a scalar-first quaternion; reject zero/nonfinite input."""
    value = _vector(q, 4, "q_IB")
    scale = np.linalg.norm(value)
    if not np.isfinite(scale) or scale < 1e-14:
        raise ValueError("q_IB must be nonzero and finite.")
    return value / scale


def quaternion_multiply(q_left, q_right) -> np.ndarray:
    """Hamilton product q_left ⊗ q_right (raw four-vectors; no normalization)."""
    w1, x1, y1, z1 = _vector(q_left, 4, "q_left")
    w2, x2, y2, z2 = _vector(q_right, 4, "q_right")
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_from_euler_zyx(roll_x: float, pitch_y: float, yaw_z: float) -> np.ndarray:
    """q_IB consistent with Rz(yaw) @ Ry(pitch) @ Rx(roll); radians."""
    angles = np.asarray([roll_x, pitch_y, yaw_z], dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Euler angles must be finite.")
    r, p, y = 0.5 * angles
    qx = [np.cos(r), np.sin(r), 0.0, 0.0]
    qy = [np.cos(p), 0.0, np.sin(p), 0.0]
    qz = [np.cos(y), 0.0, 0.0, np.sin(y)]
    return normalize_quaternion(quaternion_multiply(quaternion_multiply(qz, qy), qx))


def rotation_from_quaternion(q_IB) -> np.ndarray:
    """Return R_IB (3x3); input must already be unit length to tolerance."""
    q = _vector(q_IB, 4, "q_IB")
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("q_IB must be unit length; use normalize_quaternion explicitly.")
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),     1 - 2*(x*x+y*y)],
    ])


def quaternion_rate_from_body_omega(q_IB, omega_B) -> np.ndarray:
    """qdot_IB = 0.5 q_IB ⊗ [0, omega_B] for angular velocity in B [rad/s]."""
    rotation_from_quaternion(q_IB)  # validate normalized input
    omega = _vector(omega_B, 3, "omega_B")
    return 0.5 * quaternion_multiply(q_IB, np.r_[0.0, omega])


def attachment_kinematics(r_bus_I, v_bus_I, q_IB, omega_B, attach_B):
    """Position and velocity of a point fixed to the rigid bus, in inertial axes.

    r_attach_I = r_bus_I + R_IB attach_B
    v_attach_I = v_bus_I + R_IB (omega_B x attach_B)

    The bus position is its COM; attach_B is relative to that COM.  A moving
    reel/attachment requires an additional relative-motion term (not modelled).
    """
    r = _vector(r_bus_I, 3, "r_bus_I")
    v = _vector(v_bus_I, 3, "v_bus_I")
    omega = _vector(omega_B, 3, "omega_B")
    attach = _vector(attach_B, 3, "attach_B")
    R_IB = rotation_from_quaternion(q_IB)
    return r + R_IB @ attach, v + R_IB @ np.cross(omega, attach)


def attachment_reaction_on_bus(q_IB, attach_B, force_on_bus_I, couple_on_bus_I=None):
    """Convert a tether-on-bus reaction to force and torque about the bus COM.

    Inputs: force/couple applied *to the bus* in inertial coordinates.  The
    output (force_B, torque_B) is in body coordinates; if couple is omitted,
    it is zero.  This does NOT add distributed Lorentz loading a second time.
    """
    R_IB = rotation_from_quaternion(q_IB)
    arm_B = _vector(attach_B, 3, "attach_B")
    force_I = _vector(force_on_bus_I, 3, "force_on_bus_I")
    if couple_on_bus_I is None:
        couple_I = np.zeros(3)
    else:
        couple_I = _vector(couple_on_bus_I, 3, "couple_on_bus_I")
    force_B = R_IB.T @ force_I
    torque_B = np.cross(arm_B, force_B) + R_IB.T @ couple_I
    return force_B, torque_B
