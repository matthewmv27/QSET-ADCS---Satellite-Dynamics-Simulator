"""M4A deterministic environmental primitives. Full WMM/atmosphere adapters pending."""
from .orbit import CircularReferenceOrbit, j2_nodal_rate_rad_s, sso_inclination_deg
from .fields import SampledMagneticField, UniformMagneticField, ned_to_ecef_matrix
from .models import CenteredDipoleMagneticField, ConstantAtmosphericDensity, ExponentialAtmosphericDensity
from .flows import (corotating_flow_I, motional_electric_field_I,
                    segment_lorentz_force_I, aerodynamic_drag_force_I)

__all__ = [
    "CenteredDipoleMagneticField", "ConstantAtmosphericDensity", "ExponentialAtmosphericDensity",
    "CircularReferenceOrbit", "j2_nodal_rate_rad_s", "sso_inclination_deg",
    "SampledMagneticField", "UniformMagneticField", "ned_to_ecef_matrix",
    "corotating_flow_I", "motional_electric_field_I",
    "segment_lorentz_force_I", "aerodynamic_drag_force_I",
]
