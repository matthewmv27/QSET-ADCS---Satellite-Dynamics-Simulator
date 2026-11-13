"""M4 reduced-order orbit-decay demonstration, not coupled EDT flight dynamics.

Propagates a *single constant-mass point COM* in a spherical central gravity +
Earth J2 field, with optional scalar-Cd drag and ASSUMED tether currents.
Radial tether is prescribed to be straight and zenith-pointing. Its changing
COM, internal dynamics, deployed attitude and mechanical coupling are omitted.
Field must accept sample(t_s, position_I_m) with spatial queries.

Positive tether conventional current runs outward from the CubeSat (s=0)
toward the far/zenith collecting tip (s=L); electron flow in conductor is inward.
The CNT emitter alone does NOT solve collection/circuit current closure.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from edtsim.core.constants import MU_EARTH, R_EARTH_EQ
from edtsim.environment.orbit import J2_EARTH
from edtsim.environment.flows import (corotating_flow_I, segment_lorentz_force_I,
                                       aerodynamic_drag_force_I)
from edtsim.tether.geometry import HomogeneousTether, RectangularTape


def _vec(v, name):
    a = np.asarray(v, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError('%s must be a finite length-3 vector' % name)
    return a


def _positive(x, name, zero=False):
    f = float(x)
    if not np.isfinite(f) or (f < 0 if zero else f <= 0):
        raise ValueError('%s must be finite and %s' % (name, 'nonnegative' if zero else 'positive'))
    return f


def earth_gravity_J2_I(position_I_m, *, include_J2=True):
    """Point-mass gravity and optional standard axisymmetric J2 acceleration."""
    r = _vec(position_I_m, 'position_I_m')
    radius = np.linalg.norm(r)
    if radius < R_EARTH_EQ*0.8:
        raise ValueError('position lies inside invalid Earth gravity model domain')
    a = -MU_EARTH*r/radius**3
    if include_J2:
        q = (r[2]/radius)**2
        factor = 1.5*J2_EARTH*MU_EARTH*R_EARTH_EQ**2/radius**5
        a = a + factor*np.array([r[0]*(5*q-1),r[1]*(5*q-1),r[2]*(5*q-3)])
    return a


def earth_specific_potential_J2(position_I_m, *, include_J2=True):
    """Conservative potential in J/kg; include J2 for work-energy audit."""
    r = _vec(position_I_m, 'position_I_m')
    norm = np.linalg.norm(r)
    if norm < R_EARTH_EQ*0.8:
        raise ValueError('invalid position')
    U = -MU_EARTH/norm
    if include_J2:
        U += MU_EARTH*J2_EARTH*R_EARTH_EQ**2/norm**3 * (3*(r[2]/norm)**2-1)/2
    return float(U)


@dataclass(frozen=True)
class RadialTetherForces:
    """Optional bus drag, segment drag and distributed Lorentz force.

    `current_profile` implements segment_currents_A(t, midpoint_distances).
    With `braking_only=True`, a hypothetical controller switches current OFF
    whenever the instantaneous total prescribed-current Lorentz force does
    positive work on the bus orbit. This is not a validated circuit controller.
    Neutral atmosphere is approximated as Earth-corotating; plasma velocity
    is NOT calculated because this module does not calculate circuit current.
    """
    field: object
    density: object
    current_profile: object
    tether: HomogeneousTether
    dry_mass_kg: float
    bus_Cd: float = 2.2
    bus_projected_area_m2: float = 0.03
    tether_Cd: float = 2.2
    segments: int = 10
    braking_only: bool = False

    def __post_init__(self):
        if not isinstance(self.tether, HomogeneousTether):
            raise TypeError('tether must be HomogeneousTether')
        for name in ('dry_mass_kg','bus_Cd','bus_projected_area_m2','tether_Cd'):
            _positive(getattr(self,name),name,zero=(name!='dry_mass_kg'))
        if type(self.segments) is not int or not 1 <= self.segments <= 200:
            raise ValueError('segments must be integer in [1,200]')
        if not isinstance(self.braking_only,bool):
            raise TypeError('braking_only must be bool')
        if not callable(getattr(self.field,'sample',None)):
            raise TypeError('field must implement sample(t_s, position_I_m)')
        if not callable(getattr(self.density,'sample',None)):
            raise TypeError('density must implement sample(t_s, position_I_m)')
        if not callable(getattr(self.current_profile,'segment_currents_A',None)):
            raise TypeError('current_profile must implement segment_currents_A')

    @property
    def system_mass_kg(self):
        return self.dry_mass_kg+self.tether.mass_kg

    def forces_I(self, t_s, r_I_m, v_I_m_s):
        r = _vec(r_I_m,'r_I_m')
        v = _vec(v_I_m_s,'v_I_m_s')
        radius = np.linalg.norm(r)
        if radius < R_EARTH_EQ:
            raise ValueError('spacecraft below reference Earth radius')
        u_r = r/radius
        dlength = self.tether.length_m/self.segments
        midpoints = (np.arange(self.segments)+0.5)*dlength
        currents = np.asarray(self.current_profile.segment_currents_A(t_s,midpoints),dtype=float)
        if currents.shape != (self.segments,) or not np.all(np.isfinite(currents)):
            raise ValueError('current source returned invalid segment profile')
        # A radially pointing line rotates with rhat, even for noncircular orbits.
        radial_direction_rate = (v-u_r*np.dot(v,u_r))/radius
        forces = np.zeros((self.segments,3))
        drag = np.zeros(3)
        for index,s in enumerate(midpoints):
            position = r+s*u_r
            velocity = v+s*radial_direction_rate
            B = self.field.sample(t_s,position)
            forces[index] = segment_lorentz_force_I(currents[index],dlength*u_r,B)
            neutral = corotating_flow_I(position)
            flow = velocity-neutral
            # Tape width is assumed aligned with along-track tangent, NOT flight attitude.
            if np.linalg.norm(radial_direction_rate) > 1e-12:
                width_dir = radial_direction_rate
            else:
                # Project least-aligned Cartesian basis onto plane normal to tether.
                basis = np.eye(3)[int(np.argmin(np.abs(u_r)))]
                width_dir = basis - np.dot(basis, u_r)*u_r
            area = self.tether.cross_section.segment_projected_area_m2(
                dlength,u_r,flow,width_dir_I=width_dir if isinstance(self.tether.cross_section,RectangularTape) else None)
            drag += aerodynamic_drag_force_I(self.density.sample(t_s,position),
                    self.tether_Cd,area,velocity,neutral)
        lorentz = forces.sum(axis=0)
        if self.braking_only and np.dot(lorentz,v)>0:
            lorentz = np.zeros(3)
            currents = np.zeros_like(currents)
        neutral_bus = corotating_flow_I(r)
        drag += aerodynamic_drag_force_I(self.density.sample(t_s,r),
                self.bus_Cd,self.bus_projected_area_m2,v,neutral_bus)
        return lorentz,drag,currents


@dataclass(frozen=True)
class OrbitDecayHistory:
    t_s: np.ndarray
    r_I_m: np.ndarray
    v_I_m_s: np.ndarray
    lorentz_force_I_N: np.ndarray
    drag_force_I_N: np.ndarray
    along_track_lorentz_N: np.ndarray
    work_lorentz_J: np.ndarray
    work_drag_J: np.ndarray
    mechanical_energy_J: np.ndarray
    osculating_semimajor_axis_m: np.ndarray
    perigee_altitude_m: np.ndarray
    segment_currents_A: np.ndarray


def _osculating_kepler_elements(r,v):
    """Semimajor axis and perigee of instantaneous TWO-BODY osculating orbit."""
    rn = np.linalg.norm(r)
    energy = .5*np.dot(v,v)-MU_EARTH/rn
    if energy >= 0:
        return float('inf'),float('nan')
    a = -MU_EARTH/(2*energy)
    h = np.cross(r,v)
    evec = np.cross(v,h)/MU_EARTH-r/rn
    e = np.linalg.norm(evec)
    return float(a),float(a*(1-e)-R_EARTH_EQ)


def propagate_orbit_decay(r0_I_m, v0_I_m_s, t_eval_s, forces,
                          *, include_J2=True, rtol=1e-9, atol=1e-8,
                          max_step_s=60.0):
    """Integrate translation, J2 gravity, drag and assumed prescribed current.

    Works w/ independent spatial field provider. No WMM, CNT circuit closure,
    solar activity or coupled tether attitude is implicit here.
    Full J2 potential is used for energy accounting; osculating a/perigee
    use standard TWO-BODY definitions and fluctuate within an orbit under J2.
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise RuntimeError("Install dynamics extra: pip install -e '.[dynamics]'") from exc
    if not isinstance(forces,RadialTetherForces):
        raise TypeError('forces must be RadialTetherForces')
    r0,v0 = _vec(r0_I_m,'r0_I_m'),_vec(v0_I_m_s,'v0_I_m_s')
    times = np.asarray(t_eval_s,dtype=float)
    if times.ndim!=1 or len(times)<2 or not np.all(np.isfinite(times)) or np.any(np.diff(times)<=0) or times[0]<0:
        raise ValueError('times must be finite, nonnegative and increasing')
    _positive(rtol,'rtol');_positive(atol,'atol');_positive(max_step_s,'max_step_s')
    m = forces.system_mass_kg
    def rhs(t,y):
        r,v = y[:3],y[3:6]
        FL,FD,_ = forces.forces_I(t,r,v)
        a = earth_gravity_J2_I(r,include_J2=include_J2)+(FL+FD)/m
        return np.r_[v,a,np.dot(FL,v),np.dot(FD,v)]
    y0 = np.r_[r0,v0,0.,0.]
    solution = solve_ivp(rhs,(float(times[0]),float(times[-1])),y0,
                         t_eval=times,rtol=rtol,atol=atol,max_step=max_step_s,method='DOP853')
    if not solution.success:
        raise RuntimeError('orbit integration failed: '+solution.message)
    r = solution.y[:3,:].T
    v = solution.y[3:6,:].T
    FLs,FDs,Is = [],[],[]
    for time,position,velocity in zip(times,r,v):
        L,D,I=forces.forces_I(float(time),position,velocity)
        FLs.append(L);FDs.append(D);Is.append(I)
    FLs=np.asarray(FLs);FDs=np.asarray(FDs);Is=np.asarray(Is)
    E = np.array([m*(.5*np.dot(vv,vv)+earth_specific_potential_J2(rr,include_J2=include_J2)) for rr,vv in zip(r,v)])
    elem=np.array([_osculating_kepler_elements(rr,vv) for rr,vv in zip(r,v)])
    return OrbitDecayHistory(times.copy(),r,v,FLs,FDs,
        np.einsum('ij,ij->i',FLs,v)/np.linalg.norm(v,axis=1),
        solution.y[6,:].copy(),solution.y[7,:].copy(),E,elem[:,0],elem[:,1],Is)
