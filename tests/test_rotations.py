import numpy as np
import pytest

from edtsim.core import Rx, Ry, Rz, euler_zyx_to_R, hat, rotation_matrix_from_a_to_b, unit

ANGLES = [0.0, 0.3, -1.2, np.pi / 2, 2.9]


@pytest.mark.parametrize("f", [Rx, Ry, Rz])
@pytest.mark.parametrize("ang", ANGLES)
def test_elementary_rotations_are_proper(f, ang):
    R = f(ang)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_elementary_rotation_directions():
    np.testing.assert_allclose(Rx(np.pi / 2) @ [0, 1, 0], [0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(Ry(np.pi / 2) @ [0, 0, 1], [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(Rz(np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-15)


def test_euler_zyx_order_and_special_cases():
    r, p, y = 0.3, -0.5, 1.1
    np.testing.assert_allclose(euler_zyx_to_R(r, p, y), Rz(y) @ Ry(p) @ Rx(r), atol=1e-15)
    np.testing.assert_allclose(euler_zyx_to_R(0, 0, y), Rz(y), atol=1e-15)
    np.testing.assert_allclose(euler_zyx_to_R(0, 0, 0), np.eye(3), atol=1e-15)


def test_hat_is_cross_product(rng):
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(hat(a) @ b, np.cross(a, b), atol=1e-14)
    np.testing.assert_allclose(hat(a), -hat(a).T)


def test_unit():
    np.testing.assert_allclose(np.linalg.norm(unit([3, 4, 12])), 1.0)
    with pytest.raises(ValueError):
        unit([0, 0, 0])


def test_rotation_a_to_b_generic(rng):
    for _ in range(25):
        a, b = rng.normal(size=3), rng.normal(size=3)
        R = rotation_matrix_from_a_to_b(a, b)
        np.testing.assert_allclose(R @ unit(a), unit(b), atol=1e-12)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_a_to_b_parallel_and_antiparallel():
    np.testing.assert_allclose(rotation_matrix_from_a_to_b([0, 0, 2], [0, 0, 5]), np.eye(3))
    for a in ([0, 0, 1], [1, 0, 0], [1, 1, 1]):
        R = rotation_matrix_from_a_to_b(a, -np.asarray(a, float))
        np.testing.assert_allclose(R @ unit(a), -unit(a), atol=1e-12)
        assert np.isclose(np.linalg.det(R), 1.0)
