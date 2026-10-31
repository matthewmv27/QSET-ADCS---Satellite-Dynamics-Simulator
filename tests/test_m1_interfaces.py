"""M1 interface and frame invariants; no legacy physics models invoked."""
import numpy as np
import pytest

from edtsim.core import (
    AttachmentReaction, AttachmentState, BusState, EnvironmentSample,
    MassProperties, TetherState, attachment_kinematics,
    attachment_reaction_on_bus, euler_zyx_to_R,
    normalize_quaternion, quaternion_from_euler_zyx,
    quaternion_multiply, quaternion_rate_from_body_omega,
    rotation_from_quaternion,
)


def test_quaternion_identity_and_double_cover():
    for q in ([1, 0, 0, 0], [-1, 0, 0, 0]):
        np.testing.assert_allclose(rotation_from_quaternion(q), np.eye(3), atol=1e-14)


@pytest.mark.parametrize("angles", [
    (0, 0, 0), (.3, -.5, 1.1), (-1.5, .2, -.8), (np.pi/2, 0, 0),
    (0, np.pi/2, 0), (0, 0, np.pi/2)
])
def test_quaternion_matches_existing_euler_zyx(angles):
    q = quaternion_from_euler_zyx(*angles)
    np.testing.assert_allclose(rotation_from_quaternion(q), euler_zyx_to_R(*angles), atol=2e-15)
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_quaternion_composition_convention():
    qa = quaternion_from_euler_zyx(.3, 0, 0)
    qb = quaternion_from_euler_zyx(0, -.4, 0)
    np.testing.assert_allclose(
        rotation_from_quaternion(quaternion_multiply(qb, qa)),
        rotation_from_quaternion(qb) @ rotation_from_quaternion(qa), atol=1e-15,
    )


def test_quaternion_derivative_body_rate_sign_and_orthogonality():
    q = np.array([1., 0., 0., 0.])
    qdot = quaternion_rate_from_body_omega(q, [0, 0, 2])
    np.testing.assert_allclose(qdot, [0, 0, 0, 1])
    q = quaternion_from_euler_zyx(.4, -.3, .2)
    qdot = quaternion_rate_from_body_omega(q, [.1, .2, .3])
    assert abs(np.dot(q, qdot)) < 1e-14


def test_quaternion_bad_inputs_and_explicit_normalization():
    for invalid in ([0, 0, 0, 0], [np.nan, 0, 0, 1], [1, 0, 0]):
        with pytest.raises(ValueError):
            normalize_quaternion(invalid)
    with pytest.raises(ValueError):
        rotation_from_quaternion([2, 0, 0, 0])
    np.testing.assert_allclose(normalize_quaternion([2, 0, 0, 0]), [1, 0, 0, 0])


def test_attachment_translation_rotation_and_velocity():
    q = quaternion_from_euler_zyx(0, 0, np.pi/2)
    r, v = attachment_kinematics([100, 0, 0], [0, 2, 0], q, [0, 0, 3], [1, 0, 0])
    np.testing.assert_allclose(r, [100, 1, 0], atol=1e-14)
    # omega x arm_B = [0,3,0], rotated +90 degrees around z -> [-3,0,0]
    np.testing.assert_allclose(v, [-3, 2, 0], atol=1e-14)


def test_attachment_reaction_sign_and_moment_arm():
    q = quaternion_from_euler_zyx(0, 0, np.pi/2)
    # force_on_bus_I = [0, 4, 0] -> force_B = [4, 0, 0]
    # arm_B = [0, 2, 0] yields torque_B = [0, 0, -8]
    f_B, tau_B = attachment_reaction_on_bus(q, [0, 2, 0], [0, 4, 0])
    np.testing.assert_allclose(f_B, [4, 0, 0], atol=1e-14)
    np.testing.assert_allclose(tau_B, [0, 0, -8], atol=1e-14)
    _, tau = attachment_reaction_on_bus([1,0,0,0], [0,0,0], [1,0,0], [0,0,2])
    np.testing.assert_allclose(tau, [0, 0, 2])


def test_bus_state_immutable_copy_and_frame_validation():
    r = np.array([1., 2., 3.])
    bus = BusState(r, [0,0,0], [1,0,0,0], [0,0,0])
    r[0] = 999
    assert bus.r_I[0] == 1.0
    with pytest.raises(ValueError):
        bus.r_I[0] = 0
    with pytest.raises(ValueError):
        BusState([0,0,0], [0,0,0], [2,0,0,0], [0,0,0])
    with pytest.raises(ValueError):
        BusState([0,0], [0,0,0], [1,0,0,0], [0,0,0])
    with pytest.raises(ValueError):
        BusState([0,0,0], [np.inf,0,0], [1,0,0,0], [0,0,0])


def test_other_interface_snapshots_validate_dimensions():
    AttachmentState([1,2,3], [4,5,6])
    AttachmentReaction([1,0,0])
    MassProperties(4., [0,0,0], np.diag([2., 3., 4.]))
    EnvironmentSample(2.0, [6900000,0,0], [1e-5,0,0], [0,0,0])
    TetherState([[0,0,0], [0,0,1]], [[0,0,0], [0,0,0]])
    with pytest.raises(ValueError):
        AttachmentReaction([1, 2])
    with pytest.raises(ValueError):
        MassProperties(-1, [0,0,0], np.eye(3))
    with pytest.raises(ValueError):
        MassProperties(1, [0,0,0], [[1,2,0],[0,1,0],[0,0,1]])
    with pytest.raises(ValueError):
        MassProperties(1, [0,0,0], np.diag([-1,1,1]))
    with pytest.raises(ValueError):
        EnvironmentSample(-1.0, [6900000,0,0], [0,0,0], [0,0,0])
    with pytest.raises(ValueError):
        TetherState([[0,0,0]], [[0,0,0]])
    with pytest.raises(ValueError):
        TetherState([[0,0,0], [0,0,1]], [[0,0,0]])
