"""Non-interactive replacement for the notebook's input() prompts for configs 1 and 2.

Run:  python examples/inertia_quickstart.py
"""
import numpy as np

from edtsim.core import BusParams, PointMassParams, SpoolParams, TetherParams, euler_zyx_to_R
from edtsim.inertia import (
    inertia_body_to_inertial, is_physical_inertia, moi_config_1_coiled,
    moi_config_2_extended_on_minus_z, principal_moments,
)

np.set_printoptions(precision=6, suppress=True)

# --- Inputs (SI units; body frame, origin at bus COM). Replace with your mission values. ---
bus = BusParams(mass=4.0, a=0.1, b=0.1, c=0.3)
spool = SpoolParams(mass=0.6, r_i=0.015, r_o=0.04, h=0.06, axis_s_B=[0, 0, 1], com_B=[0, 0, 0.12])
tether = TetherParams(length=100.0, mass=0.5, attach_B=[0, 0, -0.15])
end_mass = PointMassParams(0.15)
r_end_stowed_B = [0.02, 0.01, 0.13]
roll_deg, pitch_deg, yaw_deg = 0.0, 0.0, 0.0

# --- Config 1: stowed ---
I1_B = moi_config_1_coiled(bus, spool, end_mass.mass, r_end_stowed_B)
# --- Config 2: fully extended along -z_B ---
I2_B, r_p_B, r_a_B = moi_config_2_extended_on_minus_z(bus, tether, end_mass)

R_IB = euler_zyx_to_R(*np.deg2rad([roll_deg, pitch_deg, yaw_deg]))

for name, I_B in (("Config 1 (stowed)", I1_B), ("Config 2 (extended)", I2_B)):
    w, _ = principal_moments(I_B)
    print(f"\n{name}\n  I about bus COM, body frame [kg m^2]:\n{I_B}")
    print(f"  I in inertial axes:\n{inertia_body_to_inertial(I_B, R_IB)}")
    print(f"  principal moments: {w}   physical: {is_physical_inertia(I_B)}")
print(f"\nEnd mass (deployed) at {r_p_B} m from bus COM; attachment at {r_a_B} m.")
