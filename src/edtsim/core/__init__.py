from edtsim.core.constants import MU_EARTH, R_EARTH_EQ
from edtsim.core.params import BusParams, PointMassParams, SpoolParams, TetherParams
from edtsim.core.rotations import (
    Rx,
    Ry,
    Rz,
    euler_zyx_to_R,
    hat,
    rotation_matrix_from_a_to_b,
    unit,
)

__all__ = [
    "MU_EARTH", "R_EARTH_EQ",
    "BusParams", "SpoolParams", "TetherParams", "PointMassParams",
    "Rx", "Ry", "Rz", "euler_zyx_to_R", "hat", "unit", "rotation_matrix_from_a_to_b",
]

# M1: frame/attachment conventions and shared data contracts.
from edtsim.core.frames import (
    normalize_quaternion, quaternion_multiply, quaternion_from_euler_zyx,
    rotation_from_quaternion, quaternion_rate_from_body_omega,
    attachment_kinematics, attachment_reaction_on_bus,
)
from edtsim.core.states import (
    BusState, AttachmentState, AttachmentReaction, MassProperties,
    TetherState, EnvironmentSample,
)

__all__ += [
    "normalize_quaternion", "quaternion_multiply", "quaternion_from_euler_zyx",
    "rotation_from_quaternion", "quaternion_rate_from_body_omega",
    "attachment_kinematics", "attachment_reaction_on_bus",
    "BusState", "AttachmentState", "AttachmentReaction", "MassProperties",
    "TetherState", "EnvironmentSample",
]
