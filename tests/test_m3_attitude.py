"""M3 analytic and conservation tests; independent of orbit and tether."""
import numpy as np
import pytest

from edtsim.attitude import propagate_attitude, rigid_body_rhs


def test_zero_rate_zero_torque_constant_attitude():
    ts = np.linspace(0, 200, 21)
    r = propagate_attitude(np.diag([2., 3., 4.]), [1, 0, 0, 0], [0, 0, 0], ts)
    np.testing.assert_allclose(r.q_IB, np.tile([1, 0, 0, 0], (ts.size, 1)), atol=1e-13)
    np.testing.assert_allclose(r.omega_B, 0, atol=1e-13)
    np.testing.assert_allclose(r.torque_B, 0, atol=1e-13)


def test_single_principal_axis_rotation_exact():
    ts = np.linspace(0, 20, 51)
    w = 0.1
    r = propagate_attitude(np.diag([2., 3., 4.]), [1, 0, 0, 0], [0, 0, w], ts)
    q_exact = np.array([np.cos(w*ts/2), np.zeros_like(ts), np.zeros_like(ts), np.sin(w*ts/2)]).T
    np.testing.assert_allclose(r.q_IB, q_exact, atol=5e-9)
    np.testing.assert_allclose(r.omega_B[:, 2], w, atol=1e-11)
    assert np.max(abs(np.linalg.norm(r.q_IB, axis=1) - 1)) < 1e-13


def test_constant_torque_on_principal_axis():
    ts = np.linspace(0, 10, 21)
    tau = 0.01
    result = propagate_attitude(np.diag([2, 3, 4]), [1, 0, 0, 0], [0, 0, 0], ts,
                                torque_B=lambda t, q, w: [0, 0, tau])
    expected_rate = tau/4.0 * ts
    np.testing.assert_allclose(result.omega_B[:, 2], expected_rate, atol=1e-9)
    theta = 0.5 * tau/4.0 * ts**2
    np.testing.assert_allclose(result.q_IB[:, 0], np.cos(theta/2), atol=5e-9)


def test_torque_free_energy_and_inertial_momentum():
    ts = np.linspace(0, 100, 101)
    r = propagate_attitude(np.diag([1.2, 1.4, 1.8]), [1, 0, 0, 0], [.08, .06, -.02], ts)
    np.testing.assert_allclose(r.kinetic_energy_J, r.kinetic_energy_J[0], rtol=1e-7)
    np.testing.assert_allclose(r.angular_momentum_I,
                               np.broadcast_to(r.angular_momentum_I[0], (len(ts), 3)), atol=2e-8)


def test_rhs_expected_gyroscopic_term():
    y = [1,0,0,0, 1,2,3]
    inertia=np.diag([2,3,4]); rhs=rigid_body_rhs(0,y,inertia)
    expected = -np.linalg.solve(inertia, np.cross([1,2,3],inertia@[1,2,3]))
    np.testing.assert_allclose(rhs[4:],expected)


@pytest.mark.parametrize("inertia", [np.diag([0.,1,1]), np.diag([-1.,1,1]),
                                     np.diag([1,1,3]), [[1,2,0],[0,1,0],[0,0,1]]])
def test_invalid_inertia_rejected(inertia):
    with pytest.raises(ValueError):
        propagate_attitude(inertia, [1,0,0,0], [0,0,0], [0,1])


def test_invalid_inputs_and_torque_rejected():
    with pytest.raises(ValueError):
        propagate_attitude(np.eye(3), [0,0,0,0], [0,0,0], [0,1])
    with pytest.raises(ValueError):
        propagate_attitude(np.eye(3), [2,0,0,0], [0,0,0], [0,1])
    with pytest.raises(ValueError):
        propagate_attitude(np.eye(3), [1,0,0,0], [0,0,0], [0,0,1])
    with pytest.raises(ValueError):
        propagate_attitude(np.eye(3), [1,0,0,0], [0,0,0], [0,1],
                           torque_B=lambda t,q,w: [1,2])
