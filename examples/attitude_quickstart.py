"""M3: attitude of a standalone rigid bus under zero external torque.

Illustrative mass properties and initial rates; NOT QSET flight conditions.
"""
import numpy as np

from edtsim.attitude import propagate_attitude


def main():
    # Explicitly a *rigid bus* inertia tensor, not full flexible-assembly inertia.
    inertia_B = np.diag([0.045, 0.046, 0.008])  # kg m^2; illustrative only
    times_s = np.linspace(0, 120, 121)
    result = propagate_attitude(inertia_B, [1, 0, 0, 0],
                                [.01, .02, .005], times_s)
    print("Attitude simulation complete; samples:", len(result.t_s))
    print("Final quaternion q_IB:", result.q_IB[-1])
    print("Final body rates [rad/s]:", result.omega_B[-1])
    print("Maximum torque-free energy drift [J]:",
          np.max(np.abs(result.kinetic_energy_J - result.kinetic_energy_J[0])))
    print("Maximum inertial angular momentum drift [N m s]:",
          np.max(np.linalg.norm(result.angular_momentum_I-result.angular_momentum_I[0],axis=1)))


if __name__ == '__main__':
    main()
