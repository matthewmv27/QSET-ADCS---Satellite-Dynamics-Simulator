"""M1 example: demonstrate shared frame/attachment conventions (no solver)."""
import numpy as np
from edtsim.core import (
    BusState, AttachmentReaction, attachment_kinematics,
    attachment_reaction_on_bus, quaternion_from_euler_zyx,
)

bus = BusState(
    r_I=np.array([6_900_000.0, 0, 0]),
    v_I=np.array([0, 7_600.0, 0]),
    q_IB=quaternion_from_euler_zyx(0, 0, np.pi/2),
    omega_B=np.array([0, 0, 0.01]),
)
attach_B = np.array([0.10, 0, 0])
attachment_r_I, attachment_v_I = attachment_kinematics(
    bus.r_I, bus.v_I, bus.q_IB, bus.omega_B, attach_B,
)
reaction = AttachmentReaction(force_on_bus_I=np.array([0.1, 0.0, 0.0]))
force_B, torque_B = attachment_reaction_on_bus(
    bus.q_IB, attach_B, reaction.force_on_bus_I, reaction.couple_on_bus_I,
)
print("Attachment r_I [m]:", attachment_r_I)
print("Attachment v_I [m/s]:", attachment_v_I)
print("Tether-on-bus force_B [N]:", force_B)
print("Tether-on-bus torque_B [N m]:", torque_B)
