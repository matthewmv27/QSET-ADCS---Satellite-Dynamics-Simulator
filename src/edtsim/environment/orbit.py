"""Prescribed near-circular reference orbit with optional secular J2 RAAN drift.

Not a numerical solution of the full J2 equations of motion; intended for
repeatable early-stage environmental studies, not precise orbit prediction.

Coordinates: an idealized equatorial inertial Cartesian frame.  To obtain
Earth-fixed geographic locations and realistic field vectors, use a separately
validated time/epoch-aware frame provider (not part of this starter module).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from edtsim.core.constants import MU_EARTH, R_EARTH_EQ
from edtsim.core.rotations import Rx, Rz

# Earth J2, dimensionless, for first-order orbit-plane precession.
J2_EARTH = 1.08262668e-3
SOLAR_YEAR_DAYS = 365.2422


def _positive_scalar(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _finite_scalar(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def mean_sun_rate_rad_s():
    """Approximate apparent Sun progression, radians per SI second."""
    return 2 * np.pi / (SOLAR_YEAR_DAYS * 86400.0)


def j2_nodal_rate_rad_s(semimajor_axis_m, inclination_rad, eccentricity=0.0):
    """First-order secular d(RAAN)/dt for mean Keplerian elements.

    This is a mean-element approximation, not osculating-element propagation.
    """
    a = _positive_scalar(semimajor_axis_m, "semimajor_axis_m")
    inc = _finite_scalar(inclination_rad, "inclination_rad")
    e = _finite_scalar(eccentricity, "eccentricity")
    if not 0 <= e < 1:
        raise ValueError("eccentricity must be between zero (inclusive) and one")
    n = np.sqrt(MU_EARTH / a**3)
    return -1.5 * J2_EARTH * n * (R_EARTH_EQ / a)**2 * np.cos(inc) / (1-e**2)**2


def sso_inclination_deg(altitude_m, *, eccentricity=0.0):
    """Inclination yielding nominal mean-Sun nodal rate (first-order J2).

    For circular design studies altitude_m is above a spherical Earth with
    radius R_EARTH_EQ. This is not a geodetic altitude/epoch/launch solution.
    """
    a = R_EARTH_EQ + _positive_scalar(altitude_m, "altitude_m")
    e = _finite_scalar(eccentricity, "eccentricity")
    if not 0 <= e < 1:
        raise ValueError("eccentricity must be between zero (inclusive) and one")
    n = np.sqrt(MU_EARTH / a**3)
    cos_i = -mean_sun_rate_rad_s() * (1-e**2)**2 / (1.5 * J2_EARTH * n * (R_EARTH_EQ/a)**2)
    if abs(cos_i) > 1:
        raise ValueError("No first-order J2 sun-synchronous inclination exists for this orbit")
    return float(np.rad2deg(np.arccos(cos_i)))


@dataclass(frozen=True)
class CircularReferenceOrbit:
    """Circular prescribed reference with optional secular RAAN drift.

    The velocity is the analytical derivative of the prescribed position,
    including the J2 RAAN rotation. It is not a full perturbed orbit solution.
    """
    altitude_m: float
    inclination_deg: float
    raan_deg: float = 0.0
    argument_latitude_deg: float = 0.0
    precess_raan: bool = True

    def __post_init__(self):
        _positive_scalar(self.altitude_m, "altitude_m")
        for field in ("inclination_deg", "raan_deg", "argument_latitude_deg"):
            _finite_scalar(getattr(self, field), field)
        if not 0 <= self.inclination_deg <= 180:
            raise ValueError("inclination_deg must be between 0 and 180")
        if not isinstance(self.precess_raan, bool):
            raise ValueError("precess_raan must be a boolean")

    @property
    def radius_m(self):
        return R_EARTH_EQ + self.altitude_m

    @property
    def mean_motion_rad_s(self):
        return float(np.sqrt(MU_EARTH / self.radius_m**3))

    @property
    def period_s(self):
        return float(2 * np.pi / self.mean_motion_rad_s)

    @property
    def nodal_rate_rad_s(self):
        return (j2_nodal_rate_rad_s(self.radius_m, np.deg2rad(self.inclination_deg))
                if self.precess_raan else 0.0)

    def state(self, t_s):
        """Return (r_I [m], v_I [m/s]) at elapsed time in ideal inertial axes."""
        t = _finite_scalar(t_s, "t_s")
        a = self.radius_m
        omega = np.deg2rad(self.raan_deg) + self.nodal_rate_rad_s * t
        u = np.deg2rad(self.argument_latitude_deg) + self.mean_motion_rad_s * t
        rotation = Rz(omega) @ Rx(np.deg2rad(self.inclination_deg))
        r = rotation @ np.array([a*np.cos(u), a*np.sin(u), 0.0])
        # In-plane derivative plus change of ascending-node direction.
        v_plane = rotation @ (a*self.mean_motion_rad_s*np.array([-np.sin(u), np.cos(u), 0.0]))
        v = v_plane + self.nodal_rate_rad_s * np.cross(np.array([0.0, 0.0, 1.0]), r)
        return r, v
