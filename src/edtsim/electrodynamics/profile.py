"""Prescribed spatial current profile sampled by the mechanical tether model.

Positive conventional current flows bus (s=0) -> far tip (s=L).
This adapter does NOT close the OML/CNT/plasma circuit; it only carries a
previously calculated or assumed longitudinal current into mechanical loading.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpatialCurrentProfile:
    s_m: np.ndarray
    conventional_current_A: np.ndarray
    provenance: str

    def __post_init__(self):
        s = np.array(self.s_m, dtype=float, copy=True)
        current = np.array(self.conventional_current_A, dtype=float, copy=True)
        if (s.ndim != 1 or len(s) < 2 or current.shape != s.shape
                or not np.all(np.isfinite(s)) or not np.all(np.isfinite(current))
                or s[0] != 0. or s[-1] <= 0 or np.any(np.diff(s) <= 0)):
            raise ValueError('Current profile requires finite, matching strictly increasing arc lengths from s=0')
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError('Nonempty provenance is required for a spatial current profile')
        s.setflags(write=False)
        current.setflags(write=False)
        object.__setattr__(self, 's_m', s)
        object.__setattr__(self, 'conventional_current_A', current)

    @classmethod
    def from_oml_solution(cls, solution):
        """Transfer numerical OML profile; preserve preliminary assumptions."""
        return cls(solution.s_m, solution.conventional_current_A,
                   'OML prescribed-bus-potential (NOT self-consistent CNT/plasma circuit)')

    @property
    def length_m(self):
        return float(self.s_m[-1])

    def segment_currents_A(self, t_s, midpoints_s_m):
        if not np.isfinite(float(t_s)):
            raise ValueError('time must be finite')
        mid = np.asarray(midpoints_s_m, dtype=float)
        if (mid.ndim != 1 or mid.size == 0 or not np.all(np.isfinite(mid))
                or np.any(mid < -1e-10) or np.any(mid > self.length_m + 1e-10)):
            raise ValueError('midpoint coordinates outside prescribed profile domain')
        return np.interp(np.clip(mid, 0., self.length_m), self.s_m, self.conventional_current_A)
