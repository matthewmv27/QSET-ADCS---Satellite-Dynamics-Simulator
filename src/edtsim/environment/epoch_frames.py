"""UTC-aware GCRS/ITRS Earth frames using Astropy (optional dependency).

The simulator's legacy 'I' equatorial inertial frame is treated as GCRS in
this *new* provider. Earlier synthetic orbits used idealized axes; claiming
those states are exact GCRS is an approximation and must be documented.
Units in this module are m, tesla, degrees, and WGS84 ellipsoid km.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import numpy as np


def utc_datetime(value):
    """Strict UTC ISO8601 timestamp (timezone offset or Z required)."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    else:
        raise ValueError('expected timezone-aware datetime or ISO8601 timestamp')
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('epoch must explicitly include UTC/offset')
    return dt.astimezone(timezone.utc)


def epoch_at(start_utc, t_s):
    t = float(t_s)
    if not np.isfinite(t):
        raise ValueError('t_s must be finite')
    return utc_datetime(start_utc) + timedelta(seconds=t)


def _vec3(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be finite 3-vector')
    return a


def _coords():
    try:
        import astropy.units as u
        from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
        from astropy.time import Time
        from astropy.utils import iers
        iers.conf.auto_download = False  # reproducible/offline reference runs; EOP accuracy may be reduced
    except ImportError as exc:
        raise RuntimeError("Epoch-aware Earth frames need Astropy: pip install -e '.[earth-models]'") from exc
    return u, GCRS, ITRS, CartesianRepresentation, EarthLocation, Time


class EarthFrame:
    """Spatial coordinates and rotation-only vectors at an epoch.

    GCRS -> ITRS positions and ITRS -> GCRS vectors are performed at the
    identical observation time. Rotation of a magnetic field uses the
    *difference* between a transformed field endpoint and the transformed
    origin, to avoid assuming that a position transformation is pure rotation.
    """
    def __init__(self, epoch_utc):
        self.epoch_utc = utc_datetime(epoch_utc)

    def _time(self, t_s):
        return epoch_at(self.epoch_utc, t_s)

    def position_ecef_m(self, t_s, position_I_m):
        r = _vec3(position_I_m, 'position_I_m')
        u, GCRS, ITRS, Rep, _, Time = _coords()
        time = Time(self._time(t_s))
        frame = GCRS(Rep(*(r*u.m)), obstime=time)
        return np.asarray(frame.transform_to(ITRS(obstime=time)).cartesian.xyz.to_value(u.m), dtype=float)

    def geodetic(self, t_s, position_I_m):
        """Return geodetic latitude [deg], longitude [deg], altitude [km]."""
        r = self.position_ecef_m(t_s, position_I_m)
        u, _, _, _, EarthLocation, _ = _coords()
        point = EarthLocation.from_geocentric(*(r*u.m))
        lon, lat, alt = point.to_geodetic(ellipsoid='WGS84')
        return float(lat.to_value(u.deg)), float(lon.to_value(u.deg)), float(alt.to_value(u.km))

    def vector_ecef_to_inertial(self, t_s, vector_E):
        v = _vec3(vector_E, 'vector_E')
        u, GCRS, ITRS, Rep, _, Time = _coords()
        time = Time(self._time(t_s))
        # Difference removes any origin translation; component units are
        # arbitrary lengths for the rotation, and the result inherits them.
        endpoint = ITRS(Rep(*(v*u.m)), obstime=time).transform_to(GCRS(obstime=time))
        origin = ITRS(Rep(*(np.zeros(3)*u.m)), obstime=time).transform_to(GCRS(obstime=time))
        return np.asarray((endpoint.cartesian.xyz-origin.cartesian.xyz).to_value(u.m), dtype=float)
