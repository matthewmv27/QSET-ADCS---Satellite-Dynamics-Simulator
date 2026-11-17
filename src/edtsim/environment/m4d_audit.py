"""M4d independent sign/power audit for a prescribed radial conducting tether.

This module does NOT solve an EDT electrical circuit. All segment currents
are prescribed signed conventional currents: positive = bus -> zenith tip.
Vectors are expressed in the simulator's inertial frame and SI units.
The prescribed circular orbit is only an idealized GCRS approximation when
using epoch-aware Earth main-field models.
"""
from dataclasses import dataclass
import numpy as np
from .flows import motional_electric_field_I, segment_lorentz_force_I


def _vector(value, name):
    x = np.asarray(value, dtype=float)
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        raise ValueError(f'{name} must be a finite 3-vector')
    return x


def _positive(value, name):
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return x


@dataclass(frozen=True)
class TetherForceAudit:
    """SI quantities for a *single instantaneous prescribed-tether case*.

    power_com_W: total Lorentz force dot bus/COM orbital velocity.
    power_distributed_W: sum of each force dot its local moving segment velocity.
    motional_emf_power_W: sum I*(v_segment - v_plasma)xB dot dl.
    plasma_motion_power_W: sum force dot v_plasma.

    Identity: power_distributed_W + motional_emf_power_W
            = plasma_motion_power_W (within rounding).
    This is an algebraic consistency test, not a circuit energy budget.
    """
    force_I_N: np.ndarray
    along_track_force_N: float
    orbit_normal_B_weighted_T: float
    power_com_W: float
    power_distributed_W: float
    motional_emf_power_W: float
    plasma_motion_power_W: float
    identity_residual_W: float


def audit_radial_tether(time_s, position_I_m, velocity_I_m_s,
                        field, currents_A, tether_length_m,
                        plasma_velocity_provider=None):
    """Evaluate distributed I dl x B and its mechanical/electrical power signs.

    field.sample(t_s, r_I_m) must support spatial queries.
    currents_A is an array, one signed conventional current per equal segment.
    plasma_velocity_provider(t_s, r_I_m) returns a 3-vector in m/s, or None
    indicates stationary inertial plasma *for a test only*.
    """
    t = float(time_s)
    if not np.isfinite(t):
        raise ValueError('time_s must be finite')
    r = _vector(position_I_m, 'position_I_m')
    v = _vector(velocity_I_m_s, 'velocity_I_m_s')
    radius = _positive(np.linalg.norm(r), 'position radius')
    speed = _positive(np.linalg.norm(v), 'orbital speed')
    length = _positive(tether_length_m, 'tether length')
    current = np.asarray(currents_A, dtype=float)
    if current.ndim != 1 or len(current) < 1 or len(current) > 10000 or not np.all(np.isfinite(current)):
        raise ValueError('currents_A must be a nonempty, finite one-dimensional array')
    if not callable(getattr(field, 'sample', None)):
        raise TypeError('field must implement sample(time_s, position_I_m)')
    if plasma_velocity_provider is not None and not callable(plasma_velocity_provider):
        raise TypeError('plasma_velocity_provider must be callable or None')

    radial = r/radius
    vhat = v/speed
    orbit_normal = np.cross(radial, vhat)
    dradial_dt = (v - radial*np.dot(radial,v))/radius
    ds = length/len(current)
    net_force = np.zeros(3)
    power_distributed = 0.
    emf_power = 0.
    plasma_power = 0.
    predicted_along = 0.
    B_h_integral = 0.
    for j, I_j in enumerate(current):
        s = (j+0.5)*ds
        position = r+s*radial
        velocity = v+s*dradial_dt
        B = _vector(field.sample(t, position), 'B_I_T')
        vp = (np.zeros(3) if plasma_velocity_provider is None
              else _vector(plasma_velocity_provider(t, position), 'plasma velocity'))
        dl = ds*radial
        force = segment_lorentz_force_I(I_j, dl, B)
        net_force += force
        power_distributed += np.dot(force, velocity)
        emf_power += I_j*np.dot(motional_electric_field_I(velocity, vp, B), dl)
        plasma_power += np.dot(force,vp)
        # From (radial x B) dot vhat = -B dot (radial x vhat).
        predicted_along -= I_j*ds*np.dot(B,orbit_normal)
        B_h_integral += ds*np.dot(B,orbit_normal)
    along = float(np.dot(net_force,vhat))
    if not np.isclose(along,predicted_along,rtol=1e-10,atol=1e-14):
        raise ArithmeticError('cross-product sign identity failed')
    residual = power_distributed+emf_power-plasma_power
    return TetherForceAudit(net_force,along,float(B_h_integral/length),
                float(np.dot(net_force,v)),float(power_distributed),
                float(emf_power),float(plasma_power),float(residual))


def sample_orbit_audit(orbit, field, times_s, current_A, tether_length_m,
                       segments=20, plasma_velocity_provider=None):
    """Sample a reference orbit without propagating orbital decay.

    Returns plain dictionaries for JSON/CSV; does not claim an EDT current
    is electrically achievable or that current signs correspond to a passive
    operating mode.  This deliberately does NOT integrate a physical orbit.
    """
    times=np.asarray(times_s,dtype=float)
    if times.ndim!=1 or len(times)<2 or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0):
        raise ValueError('times_s must contain at least two strictly increasing finite samples')
    if not isinstance(segments,int) or not 1 <= segments <= 10000:
        raise ValueError('segments must be a positive integer <=10000')
    I=float(current_A)
    if not np.isfinite(I):
        raise ValueError('current_A must be finite')
    rows=[]
    for t in times:
        r,v=orbit.state(float(t))
        a=audit_radial_tether(t,r,v,field,np.full(segments,I),tether_length_m,
                              plasma_velocity_provider=plasma_velocity_provider)
        rows.append(dict(time_s=float(t), along_track_force_N=a.along_track_force_N,
                         B_orbit_normal_T=a.orbit_normal_B_weighted_T,
                         power_com_W=a.power_com_W,
                         power_distributed_W=a.power_distributed_W,
                         motional_emf_power_W=a.motional_emf_power_W,
                         plasma_motion_power_W=a.plasma_motion_power_W,
                         power_identity_residual_W=a.identity_residual_W,
                         magnetic_force_x_N=float(a.force_I_N[0]),
                         magnetic_force_y_N=float(a.force_I_N[1]),
                         magnetic_force_z_N=float(a.force_I_N[2])))
    return rows
