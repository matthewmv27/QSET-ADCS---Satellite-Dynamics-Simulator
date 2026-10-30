"""Validated, immutable-by-convention interface snapshots for future modules.

All quantities use SI units and the frames in docs/interfaces.md.  Arrays are
copied and marked read-only on construction; each object is a single snapshot,
not an integrator.  Full trajectory/result storage is reserved for later work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edtsim.core.frames import rotation_from_quaternion

__all__ = ["BusState", "AttachmentState", "AttachmentReaction", "MassProperties", "TetherState", "EnvironmentSample"]


def _array(x, name, shape=None, ndim=None):
    a = np.array(x, dtype=float, copy=True)
    if (shape is not None and a.shape != shape) or (ndim is not None and a.ndim != ndim):
        raise ValueError(f"{name} has invalid shape {a.shape}; expected {shape or str(ndim)+' dimensions'}.")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain only finite values.")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class BusState:
    """Rigid-bus COM state: r_I [m], v_I [m/s], q_IB [w,x,y,z], omega_B [rad/s]."""
    r_I: np.ndarray
    v_I: np.ndarray
    q_IB: np.ndarray
    omega_B: np.ndarray

    def __post_init__(self):
        for name, shape in (("r_I", (3,)), ("v_I", (3,)), ("q_IB", (4,)), ("omega_B", (3,))):
            object.__setattr__(self, name, _array(getattr(self, name), name, shape=shape))
        rotation_from_quaternion(self.q_IB)  # require normalized, do not silently alter orientation


@dataclass(frozen=True)
class AttachmentState:
    """A point fixed on the bus: position and velocity in inertial axes."""
    r_I: np.ndarray
    v_I: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "r_I", _array(self.r_I, "r_I", shape=(3,)))
        object.__setattr__(self, "v_I", _array(self.v_I, "v_I", shape=(3,)))


@dataclass(frozen=True)
class AttachmentReaction:
    """Reaction exerted BY TETHER ON BUS at the attachment (in inertial axes).

    An optional free couple is separate from the attachment-arm cross product.
    Do not also apply the tether's distributed external forces directly to the
    bus when those forces are already propagated through the tether.
    """
    force_on_bus_I: np.ndarray
    couple_on_bus_I: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        object.__setattr__(self, "force_on_bus_I", _array(self.force_on_bus_I, "force_on_bus_I", shape=(3,)))
        object.__setattr__(self, "couple_on_bus_I", _array(self.couple_on_bus_I, "couple_on_bus_I", shape=(3,)))


@dataclass(frozen=True)
class MassProperties:
    """Assembly mass properties in B axes, COM measured from bus COM.

    inertia_com_B is about the *assembly* COM and is for diagnostics or rigid
    assemblies, not a direct replacement for rigid-bus inertia in flexible-body
    Euler equations.
    """
    mass_kg: float
    com_B: np.ndarray
    inertia_com_B: np.ndarray

    def __post_init__(self):
        mass = float(self.mass_kg)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("mass_kg must be finite and positive.")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "com_B", _array(self.com_B, "com_B", shape=(3,)))
        inertia = _array(self.inertia_com_B, "inertia_com_B", shape=(3, 3))
        scale = max(1.0, float(np.linalg.norm(inertia)))
        if not np.allclose(inertia, inertia.T, atol=1e-10*scale, rtol=0.0):
            raise ValueError("inertia_com_B must be symmetric.")
        if np.min(np.linalg.eigvalsh((inertia + inertia.T)/2)) < -1e-10*scale:
            raise ValueError("inertia_com_B must be positive semidefinite.")
        object.__setattr__(self, "inertia_com_B", inertia)


@dataclass(frozen=True)
class TetherState:
    """N tether nodes in I; node 0 is attached to bus, node -1 is the tip.

    Node positions/velocities and masses must eventually use one consistent
    discretization (defined in a later tether implementation milestone).
    """
    r_nodes_I: np.ndarray
    v_nodes_I: np.ndarray

    def __post_init__(self):
        r = _array(self.r_nodes_I, "r_nodes_I", ndim=2)
        v = _array(self.v_nodes_I, "v_nodes_I", ndim=2)
        if r.shape[1:] != (3,) or r.shape[0] < 2 or v.shape != r.shape:
            raise ValueError("r_nodes_I and v_nodes_I must have matching (N,3), N>=2.")
        object.__setattr__(self, "r_nodes_I", r)
        object.__setattr__(self, "v_nodes_I", v)


@dataclass(frozen=True)
class EnvironmentSample:
    """Field and plasma flow at a specified position and elapsed time.

    t_s is seconds since simulation start (epoch is separate metadata).
    position_I [m], B_I [T], plasma_velocity_I [m/s] all use I axes.
    """
    t_s: float
    position_I: np.ndarray
    B_I: np.ndarray
    plasma_velocity_I: np.ndarray

    def __post_init__(self):
        t = float(self.t_s)
        if not np.isfinite(t) or t < 0:
            raise ValueError("t_s must be finite and nonnegative.")
        object.__setattr__(self, "t_s", t)
        object.__setattr__(self, "position_I", _array(self.position_I, "position_I", shape=(3,)))
        object.__setattr__(self, "B_I", _array(self.B_I, "B_I", shape=(3,)))
        object.__setattr__(self, "plasma_velocity_I", _array(self.plasma_velocity_I, "plasma_velocity_I", shape=(3,)))
