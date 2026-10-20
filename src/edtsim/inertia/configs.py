"""Stowed and fully-extended satellite configurations. All tensors are about the BUS COM, in the BODY frame."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from edtsim.core.params import BusParams, PointMassParams, SpoolParams, TetherParams
from edtsim.inertia.primitives import (
    annular_cylinder_inertia_about_com,
    box_inertia_about_com,
    parallel_axis_shift,
    point_mass_inertia_about_origin,
    slender_rod_inertia_about_com,
)

__all__ = ["moi_config_1_coiled", "moi_config_2_extended_on_minus_z", "Config2Result"]


def moi_config_1_coiled(bus: BusParams, spool: SpoolParams, m_end: float, r_end_B) -> np.ndarray:
    """Config 1: bus box + stowed tether (annular-cylinder spool) + stowed end point mass at ``r_end_B``.

    ``spool.mass`` is taken to be the full stowed tether mass; the end mass is separate.
    """
    I_bus = box_inertia_about_com(bus.mass, bus.a, bus.b, bus.c)
    I_spool_com = annular_cylinder_inertia_about_com(
        spool.mass, spool.r_i, spool.r_o, spool.h, spool.axis_s_B
    )
    I_spool = I_spool_com + parallel_axis_shift(spool.mass, spool.com_B)
    I_end = point_mass_inertia_about_origin(m_end, r_end_B)
    return I_bus + I_spool + I_end


class Config2Result(NamedTuple):
    I_B: np.ndarray        # (3, 3) inertia about bus COM, body frame
    r_p_B0: np.ndarray     # (3,) end-mass position relative to bus COM, body frame
    r_a_B: np.ndarray      # (3,) tether attachment point relative to bus COM, body frame


def moi_config_2_extended_on_minus_z(bus: BusParams, tether: TetherParams,
                                     end_mass: PointMassParams) -> Config2Result:
    """Config 2: bus box + straight uniform rod of length L along -z_B from the attachment point + end point mass.

    Unpacks like the legacy tuple: ``I, r_p, r_a = moi_config_2_extended_on_minus_z(...)``.
    """
    I_bus = box_inertia_about_com(bus.mass, bus.a, bus.b, bus.c)

    L = tether.length
    z_hat = np.array([0.0, 0.0, 1.0])
    r_a_B = tether.attach_B.reshape(3)

    r_p_B0 = r_a_B - L * z_hat                      # end mass at the tip of the rod
    I_pm = point_mass_inertia_about_origin(end_mass.mass, r_p_B0)

    e_hat = -z_hat
    r_c_B = r_a_B - 0.5 * L * z_hat                 # rod midpoint
    I_rod_com = slender_rod_inertia_about_com(tether.mass, L, e_hat)
    I_rod = I_rod_com + parallel_axis_shift(tether.mass, r_c_B)

    return Config2Result(I_bus + I_pm + I_rod, r_p_B0, r_a_B)
