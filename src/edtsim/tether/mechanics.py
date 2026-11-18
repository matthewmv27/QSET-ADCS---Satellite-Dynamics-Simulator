"""M5: physical, tension-only lumped-cable mechanics with prescribed attachment.

Inertial (I) coordinates and SI units. Node 0 is prescribed at the bus
attachment; nodes 1..N are integrated. N material segments of rest length
L/N carry exactly the *physical* tether mass with trapezoidal node weights.
End mass is a separate point mass at node N. No mass-inflation or current.

Axial constitutive model: for strain extension e = length - rest_length,
T = max(0, EA/rest_length * e + etaA/rest_length * extension_rate) if e>0,
else T = 0. Thus nominal rest and all slack states have exactly zero force.
This deliberately uses a nonsmooth unilateral contact/tension boundary.

Mechanical attachment reaction is derived from constraint balance at the
anchored half-node. It is NOT the distributed Lorentz load (M4e).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from edtsim.core.states import AttachmentReaction, TetherState
from .geometry import HomogeneousTether


def _vec(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return a


def _nodes(value, count, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (count, 3) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape ({count},3), finite")
    return a


def _nonnegative(value, name):
    v = float(value)
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return v


def _frozen(value):
    arr = np.array(value, dtype=float, copy=True)
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class CableModel:
    tether: HomogeneousTether
    segments: int
    end_mass_kg: float = 0.0
    viscous_axial_rigidity_Ns: float = 0.0

    def __post_init__(self):
        if not isinstance(self.tether, HomogeneousTether):
            raise TypeError("tether must be HomogeneousTether")
        if isinstance(self.segments, bool) or not isinstance(self.segments, (int, np.integer)) or self.segments < 1:
            raise ValueError("segments must be an integer >= 1")
        object.__setattr__(self, "segments", int(self.segments))
        object.__setattr__(self, "end_mass_kg", _nonnegative(self.end_mass_kg, "end_mass_kg"))
        object.__setattr__(self, "viscous_axial_rigidity_Ns", _nonnegative(
            self.viscous_axial_rigidity_Ns, "viscous_axial_rigidity_Ns"))

    @property
    def rest_length_m(self):
        return self.tether.length_m / self.segments

    @property
    def spring_constant_N_m(self):
        return self.tether.axial_rigidity_N / self.rest_length_m

    @property
    def damping_Ns_m(self):
        return self.viscous_axial_rigidity_Ns / self.rest_length_m

    @property
    def node_masses_kg(self):
        dm = self.tether.mass_kg / self.segments
        weights = np.ones(self.segments + 1)
        weights[0] = weights[-1] = 0.5
        weights *= dm
        weights[-1] += self.end_mass_kg
        return _frozen(weights)

    @property
    def total_mass_kg(self):
        return self.tether.mass_kg + self.end_mass_kg


def segment_tension(length_m, rest_length_m, extension_rate_m_s, EA_N, viscous_axial_rigidity_Ns=0.):
    """Exactly zero for length <= rest_length; damping never acts in slack.

    For stretched segments a compression-rate dashpot cannot generate a
    negative tension. No force is applied for a zero-length slack segment.
    """
    ell = _nonnegative(length_m, "length_m")
    l0 = float(rest_length_m)
    rate = float(extension_rate_m_s)
    EA = _nonnegative(EA_N, "EA_N")
    eta = _nonnegative(viscous_axial_rigidity_Ns, "viscous_axial_rigidity_Ns")
    if not np.isfinite(l0) or l0 <= 0 or not np.isfinite(rate):
        raise ValueError("rest length must be positive and extension rate finite")
    # Suppress representational roundoff at nominal rest; this is not
    # physical smoothing and adds no tensile preload.
    if ell - l0 <= 8*np.finfo(float).eps * max(ell,l0):
        return 0.0
    return max(0.0, (EA * (ell - l0) + eta * rate) / l0)


def spring_pair_forces(r0, r1, v0, v1, rest_length_m, EA_N, viscous_axial_rigidity_Ns=0.):
    """Equal-and-opposite 2-mass spring forces, (on node 0, on node 1, tension)."""
    delta = _vec(r1, "r1") - _vec(r0, "r0")
    dv = _vec(v1, "v1") - _vec(v0, "v0")
    ell = float(np.linalg.norm(delta))
    if ell <= 0:
        if ell < rest_length_m:
            return np.zeros(3), np.zeros(3), 0.0
        raise ValueError("Coincident stretched nodes cannot define an axial direction")
    axis = delta / ell
    force = segment_tension(ell, rest_length_m, float(dv @ axis), EA_N,
                            viscous_axial_rigidity_Ns) * axis
    return force, -force, float(np.linalg.norm(force))


def mass_properties_I(model: CableModel, positions_I):
    """(total mass, COM_I, 3x3 I_COM_I) from consistent lumped nodal masses.

    Nodal inertia approaches the uniform slender rod as N increases. Point
    masses have zero intrinsic cross-sectional inertia in this M5 model.
    """
    r = _nodes(positions_I, model.segments + 1, "positions_I")
    m = model.node_masses_kg
    total = model.total_mass_kg
    com = np.einsum("i,ij->j", m, r) / total
    dr = r - com
    inertia = np.eye(3) * np.sum(m * np.einsum("ij,ij->i", dr, dr)) - (
        dr.T * m) @ dr
    return total, _frozen(com), _frozen((inertia + inertia.T) * 0.5)


def constant_attachment(position_I, velocity_I=None, acceleration_I=None):
    """Return (time)->(r,v,a), fixed-velocity or prescribed constant acceleration.

    For a stationary attachment, leave velocity/acceleration unspecified.
    """
    r = _vec(position_I, "position_I").copy()
    v = np.zeros(3) if velocity_I is None else _vec(velocity_I, "velocity_I").copy()
    a = np.zeros(3) if acceleration_I is None else _vec(acceleration_I, "acceleration_I").copy()
    def provider(t):
        return r + v*t + 0.5*a*t*t, v + a*t, a.copy()
    return provider


def _attachment(provider, t):
    if not callable(provider):
        raise TypeError("attachment must be callable(t)->(position, velocity, acceleration)")
    r, v, a = provider(float(t))
    return _vec(r, "attachment position"), _vec(v, "attachment velocity"), _vec(a, "attachment acceleration")


def _gravity(gravity_I, t, r):
    # Optional vectorized adapter for many-node orbital gravity: preserves
    # the original scalar callback contract and checks every output.
    if callable(getattr(gravity_I, "sample_nodes", None)):
        return _nodes(gravity_I.sample_nodes(float(t), r.copy()), len(r),
                      "gravity_I.sample_nodes(t,r_nodes)")
    if callable(gravity_I):
        return np.stack([_vec(gravity_I(float(t), pos.copy()), "gravity_I(t,pos)") for pos in r])
    g = np.zeros(3) if gravity_I is None else _vec(gravity_I, "gravity_I")
    return np.broadcast_to(g, r.shape).copy()


def _forces(model, r, v):
    """Return nodal forces, segment tensions and lengths (vectorized).

    Uses exactly the same unilateral tension law as ``spring_pair_forces``.
    No physical smoothing, fake preload, or cable mass scaling is introduced.
    """
    delta = np.diff(r, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    extension = lengths - model.rest_length_m
    tol = 8 * np.finfo(float).eps * np.maximum(lengths, model.rest_length_m)
    active = extension > tol
    # Slack and coincident nodes have no tension and require no unit tangent.
    tangent = np.zeros_like(delta)
    nonzero = lengths > 0
    tangent[nonzero] = delta[nonzero] / lengths[nonzero, None]
    rates = np.einsum('ij,ij->i', np.diff(v, axis=0), tangent)
    tensions = np.zeros(model.segments, dtype=float)
    tensions[active] = np.maximum(0., (
        model.tether.axial_rigidity_N * extension[active]
        + model.viscous_axial_rigidity_Ns * rates[active]
    ) / model.rest_length_m)
    segment_forces = tensions[:, None] * tangent
    f = np.zeros_like(r)
    f[:-1] += segment_forces
    f[1:] -= segment_forces
    return f, tensions, lengths


def _external(external_nodal_forces_I, t, state, count):
    if external_nodal_forces_I is None:
        return np.zeros((count, 3))
    if not callable(external_nodal_forces_I):
        raise TypeError("external_nodal_forces_I must be callable or None")
    return _nodes(external_nodal_forces_I(float(t), state), count,
                  "external_nodal_forces_I(t,state)")


@dataclass(frozen=True)
class CableHistory:
    t_s: np.ndarray
    r_nodes_I_m: np.ndarray  # (T,N+1,3)
    v_nodes_I_m_s: np.ndarray
    node_masses_kg: np.ndarray
    actual_segment_lengths_m: np.ndarray  # (T,N)
    segment_tensions_N: np.ndarray
    kinetic_energy_J: np.ndarray
    elastic_energy_J: np.ndarray
    gravity_potential_J: np.ndarray  # NaN for arbitrary gravity function
    total_mechanical_energy_J: np.ndarray  # meaningful for constant gravity/no external forcing
    com_I_m: np.ndarray
    inertia_about_com_I_kgm2: np.ndarray
    attachment_force_on_bus_I_N: np.ndarray
    solver_diagnostics: dict

    def tether_state(self, index):
        return TetherState(self.r_nodes_I_m[index], self.v_nodes_I_m_s[index])

    def attachment_reaction(self, index):
        return AttachmentReaction(self.attachment_force_on_bus_I_N[index])


def simulate_cable(model: CableModel, t_eval_s, r_nodes0_I_m, v_nodes0_I_m_s,
                   *, attachment, gravity_I=None, external_nodal_forces_I=None,
                   method="Radau", rtol=1e-7, atol=1e-9, max_step_s=np.inf):
    """Physically massed transient with prescribed attachment and full histories.

    Evolves *relative* node coordinates to avoid catastrophic cancellation of
    metre-scale tether motion at LEO orbital radii. Gravity is a constant
    inertial acceleration, or gravity_I(t,position) callable. External nodal
    forces (e.g. future M4e distribution) are optional; default exactly zero.
    The nonzero anchor half-mass is included in the on-bus reaction balance.
    No Jacobian sparsity is supplied (the legacy pattern was unvalidated).
    """
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise RuntimeError("Install tether extras: python -m pip install -e '.[tether]'") from exc
    if not isinstance(model, CableModel):
        raise TypeError("model must be CableModel")
    ts = np.asarray(t_eval_s, dtype=float)
    if (ts.ndim != 1 or len(ts) < 2 or not np.all(np.isfinite(ts)) or
            ts[0] < 0 or np.any(np.diff(ts) <= 0)):
        raise ValueError("t_eval_s must be finite, strictly increasing, length >= 2")
    for name,val in (("rtol", rtol), ("atol",atol),("max_step_s",max_step_s)):
        v=float(val)
        if (not np.isfinite(v) and not (name=="max_step_s" and v==np.inf)) or v<=0:
            raise ValueError(name+" must be positive (max_step may be inf)")
    nr = model.segments + 1
    r0 = _nodes(r_nodes0_I_m, nr, "r_nodes0_I_m")
    v0 = _nodes(v_nodes0_I_m_s, nr, "v_nodes0_I_m_s")
    anchor_r0, anchor_v0, _ = _attachment(attachment, ts[0])
    if not np.allclose(r0[0], anchor_r0, atol=1e-9, rtol=0) or not np.allclose(v0[0], anchor_v0, atol=1e-9, rtol=0):
        raise ValueError("node 0 must match prescribed attachment position/velocity")
    relative_r0 = r0[1:] - anchor_r0
    relative_v0 = v0[1:] - anchor_v0
    masses = np.asarray(model.node_masses_kg)
    n = model.segments
    # Dynamic state contains nodes 1..N only.
    y0 = np.r_[relative_r0.ravel(), relative_v0.ravel()]

    def unpack(t, y):
        ar, av, aa = _attachment(attachment, t)
        rr = y[:3*n].reshape(n, 3)
        vv = y[3*n:].reshape(n, 3)
        return (np.vstack([ar, ar+rr]), np.vstack([av, av+vv]), aa)

    def rhs(t, y):
        r, v, a_anchor = unpack(t, y)
        internal, _, _ = _forces(model, r, v)
        grav = _gravity(gravity_I, t, r)
        external = _external(external_nodal_forces_I, t, TetherState(r,v), nr)
        accel_relative = (internal[1:] + external[1:]) / masses[1:, None] + grav[1:] - a_anchor
        return np.r_[y[3*n:], accel_relative.ravel()]

    sol = solve_ivp(rhs, (float(ts[0]), float(ts[-1])), y0,
                    method=method, t_eval=ts, rtol=rtol, atol=atol,
                    max_step=float(max_step_s))
    if not sol.success:
        raise RuntimeError("cable dynamics integration failed: "+sol.message)
    histories = [unpack(float(t), y) for t,y in zip(ts, sol.y.T)]
    r_hist = np.stack([h[0] for h in histories])
    v_hist = np.stack([h[1] for h in histories])
    force_hist = []; tension_hist=[]; length_hist=[]; com_hist=[]; inertia_hist=[]
    kinetic=[]; elastic=[]; grav_potential=[]; on_bus=[]
    constant_g = not callable(gravity_I)
    for t, r, v, (_,_,a_anchor) in zip(ts,r_hist,v_hist,histories):
        f, tension, lengths = _forces(model, r, v)
        g = _gravity(gravity_I, float(t),r)
        ext = _external(external_nodal_forces_I, float(t),TetherState(r,v),nr)
        on_bus.append(f[0] + ext[0] + masses[0]*(g[0] - a_anchor))
        tension_hist.append(tension); length_hist.append(lengths)
        kinetic.append(0.5*np.sum(masses[:,None]*v**2))
        elastic.append(0.5*model.spring_constant_N_m *
                       np.sum(np.maximum(lengths - model.rest_length_m, 0)**2))
        grav_potential.append(-np.sum(masses[:,None]*g*r) if constant_g else np.nan)
        _, com, iner = mass_properties_I(model, r)
        com_hist.append(com); inertia_hist.append(iner)
    kinetic = np.array(kinetic); elastic=np.array(elastic)
    potential = np.array(grav_potential)
    diag = dict(method=method, success=bool(sol.success), message=str(sol.message),
                nfev=int(sol.nfev), njev=int(sol.njev), nlu=int(sol.nlu),
                n_segments=n, mass_scale_used=1.0,
                current_A=0.0 if external_nodal_forces_I is None else None,
                jac_sparsity="disabled: no validated pattern",
                energy_conservation_applicable=bool(constant_g and external_nodal_forces_I is None and
                    np.allclose(r_hist[:,0],r_hist[0,0]) and np.allclose(v_hist[:,0],0) and
                    model.viscous_axial_rigidity_Ns==0))
    return CableHistory(*map(_frozen,(ts,r_hist,v_hist,masses,np.array(length_hist),
            np.array(tension_hist),kinetic,elastic,potential,kinetic+elastic+potential,
            np.array(com_hist),np.array(inertia_hist),np.array(on_bus))), diag)


@dataclass(frozen=True)
class StaticEquilibrium:
    r_nodes_I_m: np.ndarray
    actual_segment_lengths_m: np.ndarray
    segment_tensions_N: np.ndarray
    attachment_force_on_bus_I_N: np.ndarray
    total_potential_J: float
    solver_diagnostics: dict


def solve_static_equilibrium(model, initial_nodes_I_m, anchor_position_I_m, gravity_I,
                             *, maxiter=2000, gtol=1e-10):
    """Independent quasi-static energy minimization (constant gravity only).

    No dynamic integration, damping or changed masses. Slack regions can have
    nonunique equilibria; only a converged stationary solution is reported.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("Install tether extras: python -m pip install -e '.[tether]'") from exc
    if not isinstance(model, CableModel):
        raise TypeError("model must be CableModel")
    a = _vec(anchor_position_I_m, "anchor_position_I_m")
    g = _vec(gravity_I, "gravity_I")
    r0 = _nodes(initial_nodes_I_m, model.segments+1, "initial_nodes_I_m")
    if not np.allclose(r0[0], a, atol=1e-9, rtol=0):
        raise ValueError("initial node 0 does not match fixed attachment")
    m = np.asarray(model.node_masses_kg)
    n = model.segments
    def objective(x):
        r = np.vstack([a, a+x.reshape(n,3)])
        internal, _, lengths = _forces(model,r,np.zeros_like(r))
        energy = (0.5*model.spring_constant_N_m *
                  np.sum(np.maximum(lengths-model.rest_length_m,0.)**2) -
                  np.sum(m[:,None]*g*r))
        grad = -internal[1:] -m[1:,None]*g
        return float(energy),grad.ravel()
    sol = minimize(objective,(r0[1:]-a).ravel(),method="L-BFGS-B",jac=True,
                   options=dict(maxiter=int(maxiter),gtol=float(gtol),ftol=1e-14))
    if not sol.success and np.linalg.norm(sol.jac,ord=np.inf)>max(1e-8,10*gtol):
        raise RuntimeError("static equilibrium failed: "+sol.message)
    r = np.vstack([a,a+sol.x.reshape(n,3)])
    f,tension,lengths = _forces(model,r,np.zeros_like(r))
    reaction = f[0]+m[0]*g
    return StaticEquilibrium(_frozen(r),_frozen(lengths),_frozen(tension),
        _frozen(reaction),float(sol.fun),dict(success=bool(sol.success),
        message=str(sol.message), iterations=int(sol.nit), mass_scale_used=1.0,
        residual_N=float(np.linalg.norm(sol.jac,ord=np.inf))))
