"""Orbit-plane parameter sweeps with explicit, NON-INDEPENDENT RAAN/LTAN.

Right ascension of the ascending node (RAAN) and local time of ascending
node (LTAN) are related at a given epoch. We accept one plane specification
per case; an epoch changes the Sun's RA and thus the RAAN corresponding to
fixed LTAN. The computed LTAN is approximate (Sun's RA at reference epoch,
not the precise epoch of the next nodal crossing).
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from .epoch_frames import utc_datetime
from .orbit import sso_inclination_deg, j2_nodal_rate_rad_s, mean_sun_rate_rad_s
from edtsim.core.constants import R_EARTH_EQ
import numpy as np


def sun_ra_deg(epoch_utc):
    """GCRS right ascension of the Sun from Astropy at requested epoch."""
    try:
        from astropy.coordinates import get_sun
        from astropy.time import Time
        import astropy.units as u
    except ImportError as exc:
        raise RuntimeError("Solar RA for epoch sweeps needs Astropy; install '.[earth-models]'") from exc
    return float(get_sun(Time(utc_datetime(epoch_utc))).ra.to_value(u.deg))


def raan_from_solar_time(sun_ra_deg_value, *, ltan_hours=None, ltdn_hours=None):
    if (ltan_hours is None) == (ltdn_hours is None):
        raise ValueError('supply exactly one of LTAN and LTDN')
    if ltdn_hours is not None:
        ltan_hours = (float(ltdn_hours) - 12) % 24
    h = float(ltan_hours)
    if not np.isfinite(h) or not 0 <= h < 24:
        raise ValueError('solar time must be in [0,24) hours')
    return float((float(sun_ra_deg_value) + 15.0*(h-12.0)) % 360.0)


def solar_times_from_raan(sun_ra_deg_value, raan_deg):
    ltan = (12.0 + (float(raan_deg)-float(sun_ra_deg_value))/15.0) % 24.0
    return float(ltan), float((ltan+12) % 24.0)


@dataclass(frozen=True)
class OrbitSweepCase:
    epoch_utc: str
    altitude_m: float
    inclination_deg: float
    raan_deg: float
    ltan_hours: float
    ltdn_hours: float
    plane_input_type: str
    plane_input_value: float
    nodal_rate_rad_s: float
    target_sun_rate_rad_s: float
    is_sso_design: bool


def make_orbit_cases(config, *, solar_ra_provider=sun_ra_deg):
    """Strict sweep config; plane_sweeps are independent families, not a product.

    Required keys: epochs_utc, altitudes_m, inclinations, plane_sweeps.
    inclinations contains 'sso' or explicit degrees, and plane_sweeps entries
    each have 'type' in {raan_deg,ltan_hours,ltdn_hours} and 'values'.
    """
    required = {'epochs_utc','altitudes_m','inclinations','plane_sweeps'}
    if set(config)!=required:
        raise ValueError('orbit sweep keys missing/unknown: '+str(sorted(set(config)^required)))
    for key in required:
        if not isinstance(config[key],list) or not config[key]:
            raise ValueError(key+' must be a nonempty list')
    cases = []
    for epoch_raw,altitude,inc_value in product(config['epochs_utc'], config['altitudes_m'], config['inclinations']):
        epoch = utc_datetime(epoch_raw)
        alt = float(altitude)
        if not np.isfinite(alt) or not 350e3 <= alt <= 600e3:
            raise ValueError('sweep altitude limited to 350–600 km in this reference study')
        sso = inc_value == 'sso' if isinstance(inc_value,str) else False
        inc = sso_inclination_deg(alt) if sso else float(inc_value)
        if not np.isfinite(inc) or not 0 <= inc <= 180:
            raise ValueError('invalid inclination')
        sun_ra = float(solar_ra_provider(epoch))
        if not np.isfinite(sun_ra):
            raise ValueError('invalid Sun RA')
        rate = j2_nodal_rate_rad_s(R_EARTH_EQ+alt,np.deg2rad(inc))
        for group in config['plane_sweeps']:
            if set(group) != {'type','values'} or group['type'] not in ('raan_deg','ltan_hours','ltdn_hours'):
                raise ValueError('invalid plane sweep definition')
            if not isinstance(group['values'],list) or not group['values']:
                raise ValueError('plane sweep values required')
            for val in group['values']:
                x = float(val)
                if not np.isfinite(x):
                    raise ValueError('invalid plane sweep value')
                if group['type']=='raan_deg':
                    raan = x % 360
                elif group['type']=='ltan_hours':
                    raan = raan_from_solar_time(sun_ra,ltan_hours=x)
                else:
                    raan = raan_from_solar_time(sun_ra,ltdn_hours=x)
                ltan,ltdn = solar_times_from_raan(sun_ra,raan)
                cases.append(OrbitSweepCase(epoch.isoformat(),alt,inc,raan,ltan,ltdn,
                    group['type'],x,rate,mean_sun_rate_rad_s(),sso))
    if len(cases)>5000:
        raise ValueError('too many sweep cases (limit 5000)')
    return cases
