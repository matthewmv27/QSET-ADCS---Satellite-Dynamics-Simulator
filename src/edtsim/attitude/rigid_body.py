"""Constant-inertia rigid-bus attitude dynamics in M1 frame conventions.

q_IB is a Hamilton scalar-first quaternion mapping body B to inertial I;
omega_B and torque_B are expressed in body axes. Inertia is about the RIGID
BUS center of mass, not the combined flexible tether/system center of mass.

There is deliberately no assumption about the orbit, environment, or control.
These forces/torques will be provided by future module interfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from edtsim.core.frames import quaternion_multiply, rotation_from_quaternion

TorqueFunction = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


def _vec(value, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite vector of shape ({size},)")
    return arr


def _bus_inertia(inertia_B) -> np.ndarray:
    matrix = np.asarray(inertia_B, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("inertia_B must be a finite 3x3 matrix")
    scale = max(1.0, float(np.linalg.norm(matrix)))
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12*scale):
        raise ValueError("inertia_B must be symmetric")
    moments = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if np.min(moments) <= 0:
        raise ValueError("inertia_B must be positive definite")
    if moments[2] > moments[0] + moments[1] + 1e-10*scale:
        raise ValueError("inertia_B must satisfy the rigid-body triangle inequality")
    return matrix.copy()


def _unit_quaternion(q) -> np.ndarray:
    raw = _vec(q, 4, "q_IB")
    norm = np.linalg.norm(raw)
    if norm < 1e-14:
        raise ValueError("q_IB cannot be zero")
    return raw / norm


def _applied_torque(t: float, q: np.ndarray, omega: np.ndarray,
                    torque_B: Optional[TorqueFunction]) -> np.ndarray:
    return np.zeros(3) if torque_B is None else _vec(torque_B(t, q.copy(), omega.copy()), 3, "torque_B(t,q,omega)")


def rigid_body_rhs(t: float, y, inertia_B, torque_B: Optional[TorqueFunction] = None) -> np.ndarray:
    """Return d[q_IB, omega_B]/dt for constant rigid-bus inertia.

    The input q can drift slightly from unit norm at intermediate numerical
    integration stages. It is normalized ONLY for the torque callback; the raw
    q is propagated with qdot = 0.5*q⊗[0,omega] to preserve its ODE structure.
    """
    state = _vec(y, 7, "state")
    inertia = _bus_inertia(inertia_B)
    q, omega = state[:4], state[4:]
    torque = _applied_torque(float(t), _unit_quaternion(q), omega, torque_B)
    qdot = 0.5 * quaternion_multiply(q, np.r_[0.0, omega])
    wdot = np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))
    return np.r_[qdot, wdot]


@dataclass(frozen=True)
class AttitudeHistory:
    """Time history of rigid-bus attitude and body angular rate.

    q_IB shape (N,4), omega_B shape (N,3), torque_B shape (N,3).
    Energy and inertial angular momentum are diagnostic only: they are NOT
    conserved when nonzero external torque is applied.
    """
    t_s: np.ndarray
    q_IB: np.ndarray
    omega_B: np.ndarray
    torque_B: np.ndarray
    kinetic_energy_J: np.ndarray
    angular_momentum_I: np.ndarray


def propagate_attitude(inertia_B, q0_IB, omega0_B, t_eval_s,
                       torque_B: Optional[TorqueFunction] = None,
                       *, rtol: float = 1e-9, atol: float = 1e-11,
                       method: str = "DOP853") -> AttitudeHistory:
    """Integrate bus-only rigid-body attitude, with optional body torque.

    This M3 solver does NOT propagate bus translation, J2 orbital motion,
    gravity-gradient torque, atmospheric drag, or tether coupling. All initial
    conditions and torques are user inputs. scipy is an optional dependency.
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise RuntimeError("Install dynamics extras: python -m pip install -e '.[dynamics]'") from exc
    inertia = _bus_inertia(inertia_B)
    rotation_from_quaternion(q0_IB)  # M1 contract: callers supply unit quaternions
    q0 = _vec(q0_IB, 4, "q0_IB").copy()
    omega0 = _vec(omega0_B, 3, "omega0_B")
    ts = np.asarray(t_eval_s, dtype=float)
    if (ts.ndim != 1 or ts.size < 2 or not np.all(np.isfinite(ts)) or
            np.any(np.diff(ts) <= 0) or ts[0] < 0):
        raise ValueError("t_eval_s must be a finite, strictly increasing time vector (at least two values)")
    for name, value in (("rtol", rtol), ("atol", atol)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    def rhs(t, y):
        q, omega = y[:4], y[4:]
        tau = _applied_torque(t, _unit_quaternion(q), omega, torque_B)
        return np.r_[0.5 * quaternion_multiply(q, np.r_[0.0, omega]),
                     np.linalg.solve(inertia, tau - np.cross(omega, inertia @ omega))]

    solution = solve_ivp(rhs, (float(ts[0]), float(ts[-1])),
                         np.r_[q0, omega0], method=method, t_eval=ts,
                         rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"attitude integration failed: {solution.message}")
    raw_q = solution.y[:4, :].T
    q_history = np.array([_unit_quaternion(row) for row in raw_q])
    omega_history = solution.y[4:, :].T.copy()
    torque_history = np.array([
        _applied_torque(float(t), q, w, torque_B)
        for t, q, w in zip(ts, q_history, omega_history)
    ])
    angular_momentum_body = omega_history @ inertia.T
    h_I = np.array([rotation_from_quaternion(q) @ h
                    for q, h in zip(q_history, angular_momentum_body)])
    energies = 0.5 * np.einsum('ni,ni->n', omega_history, angular_momentum_body)
    return AttitudeHistory(ts.copy(), q_history, omega_history, torque_history,
                           energies, h_I)
