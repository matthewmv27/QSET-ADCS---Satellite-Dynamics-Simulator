import numpy as np
import pytest

from conftest import annulus_particles, assert_close_to_scale
from edtsim.core import BusParams, PointMassParams, SpoolParams, TetherParams
from edtsim.inertia import (
    box_inertia_about_com, inertia_from_particles, is_physical_inertia,
    moi_config_1_coiled, moi_config_2_extended_on_minus_z,
    annular_cylinder_inertia_about_com,
)


def test_config2_closed_form_attach_at_origin():
    bus = BusParams(1.0, 0.1, 0.1, 0.1)
    L, mt, mp = 100.0, 0.5, 0.15
    I, r_p, r_a = moi_config_2_extended_on_minus_z(bus, TetherParams(L, mt, [0, 0, 0]), PointMassParams(mp))
    Ib = box_inertia_about_com(1.0, 0.1, 0.1, 0.1)[0, 0]
    expected_perp = Ib + mt * L**2 / 3 + mp * L**2        # rod about its end + tip mass
    np.testing.assert_allclose(np.diag(I), [expected_perp, expected_perp, Ib], rtol=1e-13)
    np.testing.assert_allclose(r_p, [0, 0, -L])
    np.testing.assert_allclose(r_a, [0, 0, 0])
    np.testing.assert_allclose(I, np.diag(np.diag(I)), atol=1e-12)   # no off-diagonals on-axis


def test_config2_matches_particle_model_with_offset_attachment():
    bus = BusParams(2.5, 0.1, 0.1, 0.2)
    L, mt, mp = 20.0, 0.2, 0.05
    att = np.array([0.03, -0.02, -0.15])
    I, r_p, r_a = moi_config_2_extended_on_minus_z(bus, TetherParams(L, mt, att), PointMassParams(mp))

    N = 4000
    s = (np.arange(N) + 0.5) * L / N                       # distance from attachment, midpoints
    rod = att[None, :] + np.stack([np.zeros(N), np.zeros(N), -s], axis=1)
    pts = np.vstack([rod, r_p[None, :]])
    ms = np.concatenate([np.full(N, mt / N), [mp]])
    I_num = box_inertia_about_com(bus.mass, bus.a, bus.b, bus.c) + inertia_from_particles(pts, ms, [0, 0, 0])
    np.testing.assert_allclose(I, I_num, rtol=1e-6, atol=1e-9)


def test_config2_namedtuple_unpacks_like_legacy_tuple():
    res = moi_config_2_extended_on_minus_z(BusParams(1, .1, .1, .1), TetherParams(5, .1, [0, 0, -.05]), PointMassParams(.1))
    I, r_p, r_a = res
    assert res.I_B is I and res.r_p_B0 is r_p and res.r_a_B is r_a


def test_config1_closed_form_spool_at_origin_end_mass_on_x():
    bus = BusParams(4.0, 0.1, 0.1, 0.3)
    spool = SpoolParams(0.6, 0.015, 0.04, 0.06, [0, 0, 1], [0, 0, 0])
    m_end, d = 0.15, 0.03
    I = moi_config_1_coiled(bus, spool, m_end, [d, 0, 0])
    expected = (box_inertia_about_com(4.0, 0.1, 0.1, 0.3)
                + annular_cylinder_inertia_about_com(0.6, 0.015, 0.04, 0.06, [0, 0, 1])
                + np.diag([0, m_end * d**2, m_end * d**2]))
    np.testing.assert_allclose(I, expected, rtol=1e-13)


def test_config1_matches_particle_model_with_tilted_offset_spool():
    bus = BusParams(3.2, 0.1, 0.1, 0.227)
    spool = SpoolParams(0.4, 0.01, 0.035, 0.05, [1, 1, 0], [0.01, -0.02, 0.05])
    m_end, r_end = 0.1, np.array([0.015, 0.0, 0.06])
    I = moi_config_1_coiled(bus, spool, m_end, r_end)

    pts, ms = annulus_particles(spool.mass, spool.r_i, spool.r_o, spool.h, spool.axis_s_B, spool.com_B)
    pts = np.vstack([pts, r_end[None, :]])
    ms = np.concatenate([ms, [m_end]])
    I_num = box_inertia_about_com(bus.mass, bus.a, bus.b, bus.c) + inertia_from_particles(pts, ms, [0, 0, 0])
    assert_close_to_scale(I, I_num, rel=1e-4)


@pytest.mark.parametrize("cfg", [1, 2])
def test_configs_are_physical(cfg):
    bus = BusParams(4.0, 0.1, 0.1, 0.3)
    if cfg == 1:
        I = moi_config_1_coiled(bus, SpoolParams(0.6, 0.015, 0.04, 0.06, [0, 0, 1], [0, 0, 0.12]), 0.15, [0.02, 0.01, 0.13])
    else:
        I = moi_config_2_extended_on_minus_z(bus, TetherParams(100, 0.5, [0, 0, -0.15]), PointMassParams(0.15)).I_B
    assert is_physical_inertia(I)
