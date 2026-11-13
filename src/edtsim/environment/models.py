"""Explicitly approximate, spatially evaluable field and density providers.

These are synthetic/low-order design-study models, NOT WMM2025 or NRLMSIS.
Use an epoch-aware geomagnetic/thermosphere provider before mission prediction.
"""
from dataclasses import dataclass
import numpy as np
from edtsim.core.constants import R_EARTH_EQ
from edtsim.environment.flows import EARTH_ROTATION_RAD_S


def _vec(v, name):
    x = np.asarray(v, dtype=float)
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        raise ValueError('%s must be finite length-3 vector' % name)
    return x


def _pos(v, name, zero=False):
    f = float(v)
    if not np.isfinite(f) or (f < 0 if zero else f <= 0):
        raise ValueError('%s must be finite %s' % (name, 'nonnegative' if zero else 'positive'))
    return f


@dataclass(frozen=True)
class CenteredDipoleMagneticField:
    """Idealized centered dipole; positive axis points along dipole moment.

    B = B_equator*(R_e/r)^3 * (3(m·rhat)rhat-m).
    The moment may rotate about inertial +Z at the supplied angular rate.
    No crustal/ionospheric/magnetospheric field, epoch, or WMM coefficients.
    """
    equatorial_field_T: float = 3.12e-5
    dipole_axis_I_at_t0: tuple = (0.0, 0.0, -1.0)
    spin_rad_s: float = EARTH_ROTATION_RAD_S

    def __post_init__(self):
        _pos(self.equatorial_field_T, 'equatorial_field_T')
        axis = _vec(self.dipole_axis_I_at_t0, 'dipole_axis_I_at_t0')
        if np.linalg.norm(axis) < 1e-12:
            raise ValueError('dipole moment axis cannot be zero')
        object.__setattr__(self, 'dipole_axis_I_at_t0', tuple(axis/np.linalg.norm(axis)))
        if not np.isfinite(self.spin_rad_s):
            raise ValueError('spin_rad_s must be finite')

    def sample(self, t_s, position_I_m):
        t = float(t_s)
        if not np.isfinite(t):
            raise ValueError('time must be finite')
        r = _vec(position_I_m, 'position_I_m')
        norm = np.linalg.norm(r)
        if norm < R_EARTH_EQ*0.5:
            raise ValueError('invalid position for centered Earth dipole')
        phi = self.spin_rad_s*t
        c, s = np.cos(phi), np.sin(phi)
        x,y,z = self.dipole_axis_I_at_t0
        m = np.array([c*x-s*y, s*x+c*y, z])
        unit = r/norm
        return self.equatorial_field_T*(R_EARTH_EQ/norm)**3*(3*np.dot(m,unit)*unit-m)


@dataclass(frozen=True)
class ConstantAtmosphericDensity:
    """Explicitly synthetic density in kg/m^3, not an atmospheric model."""
    density_kg_m3: float

    def __post_init__(self):
        _pos(self.density_kg_m3, 'density_kg_m3', zero=True)

    def sample(self, t_s, position_I_m):
        if not np.isfinite(float(t_s)):
            raise ValueError('time must be finite')
        _vec(position_I_m, 'position_I_m')
        return self.density_kg_m3


@dataclass(frozen=True)
class ExponentialAtmosphericDensity:
    """Local height-range approximation for design sensitivity, NOT NRLMSIS.

    Uses spherical radius minus WGS84 equatorial radius, not geodetic height.
    Raises outside its explicitly declared validity range.
    """
    reference_altitude_m: float
    reference_density_kg_m3: float
    scale_height_m: float
    min_altitude_m: float = 250e3
    max_altitude_m: float = 800e3

    def __post_init__(self):
        for name in ('reference_altitude_m','reference_density_kg_m3','scale_height_m',
                     'min_altitude_m','max_altitude_m'):
            _pos(getattr(self, name), name)
        if not self.min_altitude_m < self.reference_altitude_m < self.max_altitude_m:
            raise ValueError('reference altitude must be inside valid altitude bounds')

    def sample(self, t_s, position_I_m):
        if not np.isfinite(float(t_s)):
            raise ValueError('time must be finite')
        h = np.linalg.norm(_vec(position_I_m,'position_I_m'))-R_EARTH_EQ
        if h < self.min_altitude_m or h > self.max_altitude_m:
            raise ValueError('altitude is outside exponential-density model validity interval')
        return float(self.reference_density_kg_m3*np.exp(-(h-self.reference_altitude_m)/self.scale_height_m))
