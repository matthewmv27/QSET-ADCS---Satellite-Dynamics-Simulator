"""Parameter containers shared across subpackages.

All lengths in metres, masses in kg. Vectors are stored as float arrays of shape (3,).
Frame: BODY frame (origin at the bus centre of mass) unless the field name says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BusParams", "SpoolParams", "TetherParams", "PointMassParams"]


def _vec3(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have exactly 3 components, got shape {arr.shape}.")
    return arr


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not value > 0.0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return value


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not value >= 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return value


@dataclass
class BusParams:
    """Uniform rectangular box. ``a``, ``b``, ``c`` are the side lengths along x_B, y_B, z_B."""
    mass: float
    a: float
    b: float
    c: float

    def __post_init__(self):
        self.mass = _positive(self.mass, "bus mass")
        self.a = _positive(self.a, "bus size a")
        self.b = _positive(self.b, "bus size b")
        self.c = _positive(self.c, "bus size c")


@dataclass
class SpoolParams:
    """Stowed tether modelled as a uniform annular cylinder.

    ``axis_s_B``: symmetry axis in the body frame (need not be unit length).
    ``com_B``: spool centre of mass relative to the bus COM, in the body frame.
    """
    mass: float
    r_i: float
    r_o: float
    h: float
    axis_s_B: np.ndarray
    com_B: np.ndarray

    def __post_init__(self):
        self.mass = _positive(self.mass, "spool mass")
        self.r_i = _nonnegative(self.r_i, "spool inner radius")
        self.r_o = _positive(self.r_o, "spool outer radius")
        if not self.r_o > self.r_i:
            raise ValueError(f"spool r_o ({self.r_o}) must exceed r_i ({self.r_i}).")
        self.h = _positive(self.h, "spool height")
        self.axis_s_B = _vec3(self.axis_s_B, "spool axis")
        if np.linalg.norm(self.axis_s_B) < 1e-12:
            raise ValueError("spool axis must be non-zero.")
        self.com_B = _vec3(self.com_B, "spool COM")


@dataclass
class TetherParams:
    """Deployed tether. ``attach_B`` is the attachment point relative to the bus COM, in the body frame."""
    length: float
    mass: float
    attach_B: np.ndarray

    def __post_init__(self):
        self.length = _positive(self.length, "tether length")
        self.mass = _nonnegative(self.mass, "tether mass")
        self.attach_B = _vec3(self.attach_B, "tether attachment point")


@dataclass
class PointMassParams:
    """A point mass (e.g. the end mass at the tether tip)."""
    mass: float

    def __post_init__(self):
        self.mass = _nonnegative(self.mass, "point mass")
