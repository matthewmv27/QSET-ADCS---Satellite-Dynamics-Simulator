import numpy as np
import pytest

from edtsim.inertia import inertia_from_particles, is_physical_inertia, parallel_axis_shift


def naive_loop(r, m, ref):
    I = np.zeros((3, 3))
    for ri, mi in zip(r, m):
        rel = ri - ref
        I += mi * (np.dot(rel, rel) * np.eye(3) - np.outer(rel, rel))
    return I


def test_single_particle_equals_parallel_axis():
    r = np.array([[0.3, -0.2, 0.5]])
    np.testing.assert_allclose(inertia_from_particles(r, [0.7], [0, 0, 0]), parallel_axis_shift(0.7, r[0]))


def test_matches_naive_loop(rng):
    r, m, ref = rng.normal(size=(200, 3)) * 5, rng.uniform(0.01, 1, 200), rng.normal(size=3)
    np.testing.assert_allclose(inertia_from_particles(r, m, ref), naive_loop(r, m, ref), rtol=1e-12, atol=1e-12)


def test_translation_invariance(rng):
    r, m, ref, t = rng.normal(size=(30, 3)), rng.uniform(0.1, 1, 30), rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(inertia_from_particles(r + t, m, ref + t), inertia_from_particles(r, m, ref), atol=1e-12)


def test_result_is_physical(rng):
    r, m = rng.normal(size=(60, 3)), rng.uniform(0.1, 1, 60)
    com = (m[:, None] * r).sum(axis=0) / m.sum()
    assert is_physical_inertia(inertia_from_particles(r, m, com))
    assert is_physical_inertia(inertia_from_particles(r, m, [3, -2, 1]))


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        inertia_from_particles(np.zeros((5, 2)), np.ones(5), [0, 0, 0])
    with pytest.raises(ValueError):
        inertia_from_particles(np.zeros((5, 3)), np.ones(4), [0, 0, 0])
