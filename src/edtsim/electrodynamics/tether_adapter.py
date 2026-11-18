"""Transfer M4e distributed loads to an M5 *prescribed-attachment* cable.

Converts segment net EXTERNAL forces into equivalent nodal loads by allocating
half of each segment force to each endpoint. This conserves total force and,
for a midpoint-concentrated force, its moment about any reference point.
The anchored node's external share is not discarded: M5 includes it in the
constraint reaction. This is NOT a bus attitude or orbit coupling model.
"""
from __future__ import annotations

import numpy as np

from edtsim.electrodynamics.loading import evaluate_distributed_loading
from edtsim.environment.fields import UniformMagneticField
from edtsim.electrodynamics.current import PrescribedConstantCurrent


def segment_forces_to_nodes(segment_forces_I_N):
    """Return N+1 nodal forces from N 3D midpoint forces, preserving wrench."""
    forces = np.asarray(segment_forces_I_N, dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3 or len(forces) < 1 or not np.all(np.isfinite(forces)):
        raise ValueError("segment_forces_I_N must be finite (N,3) with N>=1")
    nodal = np.zeros((len(forces) + 1, 3), dtype=float)
    nodal[:-1] += 0.5 * forces
    nodal[1:] += 0.5 * forces
    return nodal


def prescribed_loading_callback(field, current_source, *, reference_point_I_m=None):
    """Create M5 external_nodal_forces_I(t, state) from M4e Lorentz loads.

    The field and current provider are explicit inputs (no circuit closure).
    The default reference point for M4e diagnostics is the attachment node.
    The callback returns forces only, as required by M5 mechanics.
    """
    def external_nodal_forces_I(t_s, tether_state):
        if isinstance(field, UniformMagneticField) and isinstance(current_source, PrescribedConstantCurrent):
            # Identical I dl x B for the controlled synthetic constant-field
            # demo; vectorization avoids 100 Python callback calls per RHS.
            segment_edges = np.diff(tether_state.r_nodes_I, axis=0)
            seg = current_source.current_A * np.cross(
                segment_edges, np.asarray(field.B_I_T, dtype=float))
            return segment_forces_to_nodes(seg)
        ref = (tether_state.r_nodes_I[0] if reference_point_I_m is None
               else reference_point_I_m)
        load = evaluate_distributed_loading(
            t_s, tether_state, field, current_source, reference_point_I_m=ref)
        return segment_forces_to_nodes(load.total_segment_forces_I_N)
    return external_nodal_forces_I
