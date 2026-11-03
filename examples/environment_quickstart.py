"""M4A deterministic orbit+synthetic magnetic field demonstration (not WMM)."""
import numpy as np
from edtsim.environment import (
    CircularReferenceOrbit, sso_inclination_deg, SampledMagneticField,
    motional_electric_field_I, corotating_flow_I,
)

alt = 450e3
inclination_deg = sso_inclination_deg(alt)
orbit = CircularReferenceOrbit(altitude_m=alt, inclination_deg=inclination_deg,
                               raan_deg=0, argument_latitude_deg=0, precess_raan=True)
print(f"SSO approximate inclination: {inclination_deg:.5f} deg")
print(f"Nominal circular orbital period: {orbit.period_s/60:.2f} min")
print(f"Mean first-order nodal rate: {orbit.nodal_rate_rad_s*180/np.pi*86400:.5f} deg/day")

# Deliberately synthetic B-field values: no claim of real Earth field accuracy.
field = SampledMagneticField([0, 60, 120], [[0,0,25e-6], [0,0,30e-6], [0,0,20e-6]])
r_I, v_I = orbit.state(60.0)
plasma_v_I = corotating_flow_I(r_I)  # simple corotation approximation only
emf = motional_electric_field_I(v_I, plasma_v_I, field.sample(60.0))
print(f"Synthetic B(60 s) [T]: {field.sample(60.0)}")
print(f"Illustrative v_rel cross B [V/m]: {emf}")
print("WARNING: Synthetic field and assumed corotation; not a validated EDT mission prediction.")
