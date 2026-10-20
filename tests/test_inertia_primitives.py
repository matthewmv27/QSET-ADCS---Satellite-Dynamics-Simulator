import numpy as np
import pytest

from conftest import annulus_particles, assert_close_to_scale
from edtsim.core import rotation_matrix_from_a_to_b
from edtsim.inertia import (
    annular_cylinder_inertia_about_com, box_inertia_about_com, inertia_body_to_inertial,
    inertia_from_particles, is_physical_inertia, parallel_axis_shift,
    point_mass_inertia_about_origin, shift_inertia_to_com, shift_inertia_to_point,
    slender_rod_inertia_about_com,
)
from edtsim.core import Rx


def test_parallel_axis_point_on_x():
    m, d = 0.7, 0.3
    np.testing.assert_allclose(parallel_axis_shift(m, [d, 0, 0]), np.diag([0, m * d**2, m * d**2]))
    np.testing.assert_allclose(point_mass_inertia_about_origin(m, [0, 0, 0]), np.zeros((3, 3)))


def test_unit_cube():
    np.testing.assert_allclose(box_inertia_about_com(1.0, 1, 1, 1), np.eye(3) / 6.0)


def test_box_formula():
    m, a, b, c = 4.0, 0.1, 0.1, 0.3
    I = box_inertia_about_com(m, a, b, c)
    np.testing.assert_allclose(np.diag(I), [m / 12 * (b**2 + c**2), m / 12 * (a**2 + c**2), m / 12 * (a**2 + b**2)])
    assert is_physical_inertia(I)


def test_solid_cylinder_limit():
    m, r, h = 2.0, 0.05, 0.2
    I = annular_cylinder_inertia_about_com(m, 0.0, r, h, [0, 0, 1])
    np.testing.assert_allclose(np.diag(I), [m * (3 * r**2 + h**2) / 12] * 2 + [0.5 * m * r**2])


def test_thin_ring_limit():
    m, r = 1.0, 0.1
    I = annular_cylinder_inertia_about_com(m, r, r, 1e-9, [0, 0, 1])
    np.testing.assert_allclose(np.diag(I), [0.5 * m * r**2] * 2 + [m * r**2], rtol=1e-6)


def test_annulus_is_covariant_under_axis_rotation(rng):
    base = annular_cylinder_inertia_about_com(0.6, 0.015, 0.04, 0.06, [0, 0, 1])
    for _ in range(5):
        ax = rng.normal(size=3)
        R = rotation_matrix_from_a_to_b([0, 0, 1], ax)
        np.testing.assert_allclose(
            annular_cylinder_inertia_about_com(0.6, 0.015, 0.04, 0.06, ax), R @ base @ R.T, atol=1e-15)


@pytest.mark.parametrize("axis", [[0, 0, 1], [1, 2, 3], [-1, 0.5, 0.2]])
def test_annulus_matches_numerical_particle_model(axis):
    m, ri, ro, h = 0.6, 0.015, 0.04, 0.06
    pts, ms = annulus_particles(m, ri, ro, h, axis, com=[0, 0, 0])
    I_num = inertia_from_particles(pts, ms, [0, 0, 0])
    # midpoint-rule grid error is O(1/n^2): ~3e-5 of the largest element at this resolution
    assert_close_to_scale(annular_cylinder_inertia_about_com(m, ri, ro, h, axis), I_num, rel=1e-4)


def test_rod_formula_and_axis_has_zero_moment():
    m, L = 0.5, 100.0
    I = slender_rod_inertia_about_com(m, L, [0, 0, -1])
    np.testing.assert_allclose(I, np.diag([m * L**2 / 12, m * L**2 / 12, 0.0]))


@pytest.mark.parametrize("N", [10, 100, 1000])
def test_rod_matches_discrete_midpoint_masses(N):
    # N equal masses at segment midpoints: exact discrete result is  m L^2/12 * (1 - 1/N^2)
    m, L = 0.5, 20.0
    x = (np.arange(N) + 0.5) * L / N - L / 2
    pts = np.stack([np.zeros(N), np.zeros(N), x], axis=1)
    I = inertia_from_particles(pts, np.full(N, m / N), [0, 0, 0])
    np.testing.assert_allclose(I[0, 0], m * L**2 / 12 * (1 - 1 / N**2), rtol=1e-12)
    np.testing.assert_allclose(I[2, 2], 0.0, atol=1e-12)


def test_body_to_inertial_identity_and_roll_swap():
    I_B = box_inertia_about_com(4.0, 0.1, 0.2, 0.3)
    np.testing.assert_allclose(inertia_body_to_inertial(I_B, np.eye(3)), I_B)
    I_I = inertia_body_to_inertial(I_B, Rx(np.pi / 2))      # y_B -> z_I, z_B -> -y_I
    np.testing.assert_allclose(I_I, np.diag([I_B[0, 0], I_B[2, 2], I_B[1, 1]]), atol=1e-15)


def test_rotation_preserves_invariants(rng):
    I_B = annular_cylinder_inertia_about_com(0.6, 0.015, 0.04, 0.06, [1, 2, 3])
    R = rotation_matrix_from_a_to_b(rng.normal(size=3), rng.normal(size=3))
    I_I = inertia_body_to_inertial(I_B, R)
    assert np.isclose(np.trace(I_I), np.trace(I_B))
    np.testing.assert_allclose(np.linalg.eigvalsh(I_I), np.linalg.eigvalsh(I_B), atol=1e-15)
    np.testing.assert_allclose(I_I, I_I.T, atol=1e-15)


def test_shift_to_point_and_back_agrees_with_direct_particle_sum(rng):
    pts = rng.normal(size=(40, 3))
    ms = rng.uniform(0.1, 1.0, size=40)
    M = ms.sum()
    com = (ms[:, None] * pts).sum(axis=0) / M
    P = rng.normal(size=3)
    I_com = inertia_from_particles(pts, ms, com)
    I_P = inertia_from_particles(pts, ms, P)
    d = com - P
    np.testing.assert_allclose(shift_inertia_to_point(I_com, M, d), I_P, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shift_inertia_to_com(I_P, M, d), I_com, rtol=1e-12, atol=1e-12)
