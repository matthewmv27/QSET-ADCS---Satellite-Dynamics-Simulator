"""M4e distributed environmental loads on a prescribed tether geometry.

This computes EXTERNAL loads ON THE TETHER, not a force on the spacecraft bus.
Attachment force/couple on the bus must be supplied by a separate tether
mechanical solver (M5/M6) through core.frames.attachment_reaction_on_bus.

All vectors use an explicitly shared inertial frame I and SI units.  Positive
current follows ordered nodes bus -> far tether tip. Local forces at segment
midpoints approximate the continuous distributed load; refine the mesh before
using torque estimates. Neutral drag is optional and illustrative (scalar Cd).
"""
from dataclasses import dataclass
import numpy as np
from edtsim.core.states import TetherState
from edtsim.environment.flows import (segment_lorentz_force_I,
    motional_electric_field_I, aerodynamic_drag_force_I)
from edtsim.tether.geometry import HomogeneousTether


def _vec3(value, name):
    x = np.asarray(value, dtype=float)
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        raise ValueError(name+' must be a finite vector with shape (3,)')
    return x


def _readonly(array):
    a = np.array(array, dtype=float, copy=True)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class DistributedTetherLoad:
    segment_midpoints_I_m: np.ndarray
    segment_currents_A: np.ndarray
    lorentz_forces_I_N: np.ndarray
    drag_forces_I_N: np.ndarray
    total_segment_forces_I_N: np.ndarray
    net_tether_external_force_I_N: np.ndarray
    net_tether_external_torque_about_ref_I_Nm: np.ndarray
    reference_point_I_m: np.ndarray
    lorentz_power_local_W: float
    motional_emf_power_W: float
    plasma_motion_power_W: float
    power_identity_residual_W: float
    drag_power_local_W: float
    model_warning: str = ('EXTERNAL tether load; NOT attachment reaction on bus. '
                          'Circuit current assumed or prescribed-bias, NOT closed.')


def evaluate_distributed_loading(time_s, tether_state, field, current_source, *,
                                  reference_point_I_m, plasma_velocity_provider=None,
                                  tether=None, neutral_velocity_provider=None,
                                  neutral_density_provider=None, drag_coefficient=None,
                                  tape_width_direction_I=None):
    """Integrate segment-level Lorentz force and optional first-order neutral drag.

    field.sample(time_s, midpoint_I_m) -> B_I_T; current source implements
    segment_currents_A(time_s, midpoints_s_m).

    Plasma and neutral flow callbacks take (time_s, position_I_m) and return
    a vector in m/s. Neutral density takes (time_s, position_I_m) and returns
    nonnegative kg/m3. Drag requires all of tether, neutral flow, density, Cd.
    For zero relative velocity, aerodynamic force is exactly zero.
    """
    t = float(time_s)
    if not np.isfinite(t) or t < 0:
        raise ValueError('time_s must be finite and nonnegative')
    if not isinstance(tether_state, TetherState):
        raise TypeError('tether_state must be a TetherState')
    ref = _vec3(reference_point_I_m, 'reference_point_I_m')
    if not callable(getattr(field, 'sample', None)):
        raise TypeError('field must implement sample(t_s, position_I_m)')
    if not callable(getattr(current_source, 'segment_currents_A', None)):
        raise TypeError('current_source must implement segment_currents_A(t_s, midpoints_s_m)')
    if plasma_velocity_provider is not None and not callable(plasma_velocity_provider):
        raise TypeError('plasma_velocity_provider must be callable or None')
    drag_opts = (tether, neutral_velocity_provider, neutral_density_provider, drag_coefficient)
    use_drag = any(opt is not None for opt in drag_opts)
    if use_drag and (not isinstance(tether, HomogeneousTether)
                    or not callable(neutral_velocity_provider)
                    or not callable(neutral_density_provider)
                    or drag_coefficient is None):
        raise ValueError('Drag requires tether, neutral_velocity_provider, neutral_density_provider, drag_coefficient')
    if use_drag and (not np.isfinite(float(drag_coefficient)) or float(drag_coefficient) < 0):
        raise ValueError('drag_coefficient must be finite and nonnegative')

    edges = np.diff(tether_state.r_nodes_I, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 0):
        raise ValueError('tether nodes must define positive-length segments')
    midpoints = (tether_state.r_nodes_I[1:] + tether_state.r_nodes_I[:-1])/2
    velocities = (tether_state.v_nodes_I[1:] + tether_state.v_nodes_I[:-1])/2
    s_mid = np.cumsum(lengths)-lengths/2
    current = np.asarray(current_source.segment_currents_A(t,s_mid), dtype=float)
    if current.shape != (len(lengths),) or not np.all(np.isfinite(current)):
        raise ValueError('current source must return one finite current per segment')
    if use_drag and not np.isclose(np.sum(lengths), tether.length_m, rtol=1e-6, atol=1e-6):
        raise ValueError('tether material length and node arc length disagree')

    magnetic = np.zeros((len(lengths),3))
    drag = np.zeros_like(magnetic)
    local_power = emf_power = plasma_power = drag_power = 0.
    for j,(mid,v,dl,ds,I) in enumerate(zip(midpoints,velocities,edges,lengths,current)):
        B = _vec3(field.sample(t, mid), 'B_I_T')
        vp = (_vec3(plasma_velocity_provider(t, mid), 'plasma velocity')
              if plasma_velocity_provider is not None else np.zeros(3))
        magnetic[j] = segment_lorentz_force_I(I, dl, B)
        local_power += float(magnetic[j]@v)
        emf_power += float(I * (motional_electric_field_I(v,vp,B)@dl))
        plasma_power += float(magnetic[j]@vp)
        if use_drag:
            vn = _vec3(neutral_velocity_provider(t,mid), 'neutral velocity')
            rho = float(neutral_density_provider(t,mid))
            if not np.isfinite(rho) or rho < 0:
                raise ValueError('neutral density must be nonnegative and finite')
            relative = v-vn
            area = tether.cross_section.segment_projected_area_m2(
                float(ds),dl,relative,tape_width_direction_I)
            drag[j] = aerodynamic_drag_force_I(rho,drag_coefficient,area,v,vn)
            drag_power += float(drag[j]@v)
    combined = magnetic+drag
    net = np.sum(combined,axis=0)
    torque = np.sum(np.cross(midpoints-ref,combined),axis=0)
    residual = local_power+emf_power-plasma_power
    return DistributedTetherLoad(
        _readonly(midpoints),_readonly(current),_readonly(magnetic),
        _readonly(drag),_readonly(combined),_readonly(net),_readonly(torque),
        _readonly(ref),float(local_power),float(emf_power),float(plasma_power),
        float(residual),float(drag_power))
