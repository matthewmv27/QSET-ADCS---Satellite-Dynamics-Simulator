import numpy as np
import pytest

from edtsim.core.rotations import rotation_matrix_from_a_to_b


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def annulus_particles(m, r_i, r_o, h, axis, com, n_r=100, n_phi=48, n_z=120):
    """Dense deterministic particle model of a uniform annular cylinder (midpoint rule in r, phi, z).

    Independent of the closed-form formulas: used to cross-check them numerically.
    Returns (positions (N,3), masses (N,)) in the body frame.
    """
    dr = (r_o - r_i) / n_r
    r = r_i + (np.arange(n_r) + 0.5) * dr
    phi = (np.arange(n_phi) + 0.5) * (2 * np.pi / n_phi)
    z = (np.arange(n_z) + 0.5) * (h / n_z) - 0.5 * h
    R_, P_, Z_ = np.meshgrid(r, phi, z, indexing="ij")
    w = R_ * dr                      # volume element ~ r dr dphi dz (uniform density)
    pts = np.stack([R_ * np.cos(P_), R_ * np.sin(P_), Z_], axis=-1).reshape(-1, 3)
    masses = (w / w.sum() * m).reshape(-1)
    Rot = rotation_matrix_from_a_to_b(np.array([0.0, 0.0, 1.0]), axis)
    return pts @ Rot.T + np.asarray(com, float), masses


def assert_close_to_scale(actual, desired, rel):
    """Compare tensors with tolerance relative to the LARGEST element of ``desired``.

    (Tiny off-diagonal terms would otherwise dominate an elementwise relative error.)
    """
    desired = np.asarray(desired, float)
    np.testing.assert_allclose(actual, desired, rtol=0.0, atol=rel * np.max(np.abs(desired)))
