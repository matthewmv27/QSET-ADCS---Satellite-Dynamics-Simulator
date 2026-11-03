"""Relative-flow, drag and Lorentz-force primitives. All inertial and SI.

These are mechanical/electromagnetic force primitives, NOT atmospheric density
or current-collection models. Use explicit neutral/plasma velocity providers.
"""
from __future__ import annotations
import numpy as np

EARTH_ROTATION_RAD_S = 7.2921150e-5


def _vec3(x, name):
    a = np.asarray(x, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite length-3 vector")
    return a


def corotating_flow_I(position_I_m, earth_axis_I=(0, 0, 1), omega_rad_s=EARTH_ROTATION_RAD_S):
    """Illustrative rigid Earth-corotation velocity Ω×r, not measured flow.

    earth_axis_I is the unit Earth spin axis in the inertial axes; only valid
    when those axes and the epoch/orientation are consistently defined.
    Ionospheric plasma and neutral thermosphere may not actually corotate.
    """
    r = _vec3(position_I_m, "position_I_m")
    axis = _vec3(earth_axis_I, "earth_axis_I")
    n = np.linalg.norm(axis)
    omega = float(omega_rad_s)
    if n < 1e-14 or not np.isfinite(omega):
        raise ValueError("invalid Earth rotation parameters")
    return np.cross(omega * axis / n, r)


def motional_electric_field_I(tether_velocity_I_m_s, plasma_velocity_I_m_s, B_I_T):
    """v_rel × B [V/m], for conventional current/EMF sign conventions."""
    v_rel = _vec3(tether_velocity_I_m_s, "tether_velocity_I_m_s") - _vec3(plasma_velocity_I_m_s, "plasma_velocity_I_m_s")
    B = _vec3(B_I_T, "B_I_T")
    return np.cross(v_rel, B)


def segment_lorentz_force_I(current_A, dl_I_m, B_I_T):
    """I dl × B [N]; dl direction must match defined conventional current."""
    I = float(current_A)
    if not np.isfinite(I):
        raise ValueError("current_A must be finite")
    return I * np.cross(_vec3(dl_I_m, "dl_I_m"), _vec3(B_I_T, "B_I_T"))


def aerodynamic_drag_force_I(density_kg_m3, Cd, projected_area_m2, body_velocity_I_m_s, neutral_velocity_I_m_s):
    """First-order scalar-Cd drag for a body, not a validated free-molecular panel model."""
    rho, cd, area = float(density_kg_m3), float(Cd), float(projected_area_m2)
    if not all(np.isfinite(v) and v >= 0 for v in (rho, cd, area)):
        raise ValueError("density, Cd and area must be finite and nonnegative")
    v_rel = _vec3(body_velocity_I_m_s, "body_velocity_I_m_s") - _vec3(neutral_velocity_I_m_s, "neutral_velocity_I_m_s")
    return -0.5 * rho * cd * area * np.linalg.norm(v_rel) * v_rel
