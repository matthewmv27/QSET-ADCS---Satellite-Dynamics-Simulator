"""Interchangeable *assumed* longitudinal tether current-profile sources.

Currents are conventional signed amperes, positive from CubeSat (s=0) toward
far tether tip (s=L). These inputs do not solve plasma collection/closure.
CNT field emission capacity is a separate cathode constraint, not a profile.
"""
from dataclasses import dataclass
import numpy as np


def _midpoints(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError('midpoints_s_m must be a nonempty nonnegative finite vector')
    return x


@dataclass(frozen=True)
class ZeroCurrent:
    def segment_currents_A(self, t_s, midpoints_s_m):
        return np.zeros_like(_midpoints(midpoints_s_m))


@dataclass(frozen=True)
class PrescribedConstantCurrent:
    """A signed current imposed externally; not validated to be available."""
    current_A: float

    def __post_init__(self):
        value = float(self.current_A)
        if not np.isfinite(value):
            raise ValueError('current_A must be finite')
        object.__setattr__(self, 'current_A', value)

    def segment_currents_A(self, t_s, midpoints_s_m):
        if not np.isfinite(float(t_s)):
            raise ValueError('t_s must be finite')
        return np.full_like(_midpoints(midpoints_s_m), self.current_A)
