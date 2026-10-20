import numpy as np

from edtsim.core import rotation_matrix_from_a_to_b, unit
from edtsim.inertia import is_physical_inertia, principal_moments, sample_point_in_annular_cylinder


def test_is_physical_inertia_accepts_and_rejects():
    assert is_physical_inertia(np.diag([1.0, 2.0, 2.5]))
    assert not is_physical_inertia(np.array([[1, 2, 0], [0, 1, 0], [0, 0, 1.0]]))     # not symmetric
    assert not is_physical_inertia(np.diag([1.0, -0.5, 1.0]))                         # negative moment
    assert not is_physical_inertia(np.diag([1.0, 1.0, 3.0]))                          # violates triangle inequality
    assert not is_physical_inertia(np.eye(2))


def test_principal_moments_sorted_and_orthonormal():
    w, V = principal_moments(np.diag([3.0, 1.0, 2.0]))
    np.testing.assert_allclose(w, [1, 2, 3])
    np.testing.assert_allclose(V.T @ V, np.eye(3), atol=1e-14)


def test_samples_lie_inside_the_spool():
    ri, ro, h = 0.01, 0.04, 0.05
    axis, com = np.array([1.0, 2.0, 3.0]), np.array([0.01, -0.02, 0.05])
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = sample_point_in_annular_cylinder(ri, ro, h, axis, com, rng) - com
        z = float(p @ unit(axis))
        radial = np.linalg.norm(p - z * unit(axis))
        assert abs(z) <= h / 2 + 1e-12
        assert ri - 1e-12 <= radial <= ro + 1e-12


def test_sampling_is_reproducible_per_seed():
    args = (0.01, 0.04, 0.05, [0, 0, 1], [0, 0, 0])
    a = sample_point_in_annular_cylinder(*args, np.random.default_rng(42))
    b = sample_point_in_annular_cylinder(*args, np.random.default_rng(42))
    c = sample_point_in_annular_cylinder(*args, np.random.default_rng(43))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_sampling_is_uniform_in_volume():
    # For a volume-uniform annulus, E[r^2] = (r_i^2 + r_o^2) / 2 and E[z] = 0.
    ri, ro, h = 0.01, 0.04, 0.05
    rng = np.random.default_rng(1)
    pts = np.array([sample_point_in_annular_cylinder(ri, ro, h, [0, 0, 1], [0, 0, 0], rng) for _ in range(20000)])
    r2 = (pts[:, :2] ** 2).sum(axis=1).mean()
    assert abs(r2 - (ri**2 + ro**2) / 2) / ((ri**2 + ro**2) / 2) < 0.02
    assert abs(pts[:, 2].mean()) < 0.02 * h
