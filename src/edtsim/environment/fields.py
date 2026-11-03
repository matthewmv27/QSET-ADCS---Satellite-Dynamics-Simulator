"""Deterministic field providers and strict time-domain sampled fields."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _vec3(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return a


@dataclass(frozen=True)
class UniformMagneticField:
    """Synthetic constant B_I [T] for tests. NOT an Earth field model."""
    B_I_T: tuple

    def __post_init__(self):
        object.__setattr__(self, "B_I_T", tuple(_vec3(self.B_I_T, "B_I_T")))

    def sample(self, t_s, position_I_m):
        if not np.isfinite(float(t_s)):
            raise ValueError("t_s must be finite")
        _vec3(position_I_m, "position_I_m")
        return np.asarray(self.B_I_T, dtype=float)


class SampledMagneticField:
    """Linearly interpolate an inertial B-field table on a CLOSED time interval.

    Out-of-domain requests always raise ValueError; no clamping/extrapolation.
    No automatic periodicity is assumed. Samples are exact at tabulated times.
    """

    def __init__(self, times_s, B_I_T):
        times = np.array(times_s, dtype=float, copy=True)
        B = np.array(B_I_T, dtype=float, copy=True)
        if times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times)):
            raise ValueError("times_s must contain at least two finite times")
        if times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError("times_s must be nonnegative and strictly increasing")
        if B.shape != (len(times), 3) or not np.all(np.isfinite(B)):
            raise ValueError("B_I_T must be finite (N,3) Tesla values")
        times.setflags(write=False)
        B.setflags(write=False)
        self.times_s = times
        self.B_I_T = B

    @property
    def time_bounds_s(self):
        return float(self.times_s[0]), float(self.times_s[-1])

    def sample(self, t_s, position_I_m=None):
        t = float(t_s)
        if not np.isfinite(t) or t < self.times_s[0] or t > self.times_s[-1]:
            raise ValueError(f"field time {t_s} outside supported interval {self.time_bounds_s}")
        if position_I_m is not None:
            raise ValueError("sampled time-only field is defined on its reference trajectory, not at an arbitrary position")
        return np.array([np.interp(t, self.times_s, self.B_I_T[:, k]) for k in range(3)])


def ned_to_ecef_matrix(lat_rad, lon_rad):
    """NED-to-ECEF rotation at geodetic latitude/longitude; no translation."""
    lat, lon = float(lat_rad), float(lon_rad)
    if not np.isfinite(lat) or not np.isfinite(lon) or abs(lat) > np.pi/2:
        raise ValueError("invalid latitude or longitude")
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    north = [-sl*co, -sl*so, cl]
    east = [-so, co, 0.0]
    down = [-cl*co, -cl*so, -sl]
    return np.column_stack([north, east, down])
