"""M2 demonstration with one consistent *illustrative* mass budget (not QSET flight data)."""
import numpy as np

from edtsim.core import BusParams, PointMassParams, TetherParams
from edtsim.inertia import MassComponent, MissionMassModel, WoundTetherGeometry

np.set_printoptions(precision=6, suppress=True)

# All values are deliberately illustrative, not flight-qualified measurements.
mission = MissionMassModel(
    bus=BusParams(4.0, .10, .10, .30),
    spool_hardware=MassComponent("spool_hardware", .1, [0,0,.12],
                                 np.diag([.0001,.0001,.00005])),
    tether=TetherParams(10., .5, [0,0,-.15]),
    end_mass=PointMassParams(.15),
    wound_geometry=WoundTetherGeometry(.015, .04, .06, [0,0,1], [0,0,.12]),
    end_stowed_B=[.02,.01,.13],
)

for label, result in (
    ("Stowed", mission.straight(0.)),
    ("50% straight deployment", mission.straight(.5)),
    ("Fully straight deployment", mission.straight(1.)),
    ("Fully bent geometry", mission.bent(1., [[0,0,-.15], [3,0,-4.15], [0,0,-8.15]])),
):
    print(f"\n{label}: total mass {result.mass_kg:.4f} kg")
    print(f"Assembly COM in B relative to bus COM [m]: {result.com_B}")
    print(f"Assembly inertia about assembly COM in B [kg m^2]:\n{result.inertia_com_B}")
    print(f"Assembly inertia about bus COM in B [kg m^2]:\n{result.inertia_bus_B}")
    print("Ledger [kg]:", result.masses_by_name())

print("\nWARNING: Fixed annulus wound-tether geometry is approximate.")
print("These are static mass-property snapshots, not deployment dynamics.")
