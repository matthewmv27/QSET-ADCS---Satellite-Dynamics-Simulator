"""M5 independent mechanics/physics checks (no EDT current, no orbit model)."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from edtsim.tether import (RoundWire, HomogeneousTether, CableModel,
    segment_tension, spring_pair_forces, constant_attachment, mass_properties_I,
    simulate_cable, solve_static_equilibrium)


def model(n=4, *, length=2., EA_scale=1., damping=0., tip=0.05):
    cross = RoundWire(0.01)
    return CableModel(HomogeneousTether(cross, length, 1000.,
                      1e5*EA_scale, 1e7), n, tip, damping)


def straight(cfg, *, extension=0., direction=(1,0,0)):
    u=np.array(direction,dtype=float); u=u/np.linalg.norm(u)
    return np.linspace(0.,cfg.tether.length_m+extension,cfg.segments+1)[:,None]*u


def run(cfg, r0, t_end=.3, damping=None, gravity=None, method='DOP853', **kwargs):
    options=dict(method=method, rtol=2e-9, atol=1e-11)
    options.update(kwargs)
    return simulate_cable(cfg, np.linspace(0,t_end,81),r0,np.zeros_like(r0),
        attachment=constant_attachment(r0[0]),gravity_I=gravity,**options)


def test_legacy_phantom_tension_reproduced_but_removed():
    # Archived N_seg=5, length=10m, EA=5000 N, eps_frac=1e-4.
    # Legacy smooth_max0 applied twice, creating tension at exact rest.
    def legacy_smooth(x, eps):
        return (x+np.sqrt(x*x+eps*eps))/2
    L=10.; n=5; l0=L/n; EA=5000.; eps_frac=1e-4
    k=EA/l0
    force_raw=k*legacy_smooth(0.,eps_frac*l0)
    legacy_force=legacy_smooth(force_raw,eps_frac*EA)
    assert legacy_force == pytest.approx(0.4045084971874737,abs=1e-10)
    assert segment_tension(l0,l0,0,EA,0) == 0.0
    assert segment_tension(l0,l0,100,EA,200) == 0.0


def test_single_segment_rest_slack_compression_and_engagement():
    cfg=model(1)
    l0=cfg.rest_length_m
    assert segment_tension(l0,l0,0,10,12)==0
    assert segment_tension(l0*0.5,l0,100,10,12)==0
    assert segment_tension(0,l0,0,10,12)==0
    assert segment_tension(l0*1.1,l0,-100,10,12)==0
    assert segment_tension(l0*1.1,l0,0,10,12)==pytest.approx(1.)
    f0,f1,T=spring_pair_forces([0,0,0],[l0*1.1,0,0],[0,0,0],[0,0,0],l0,10)
    np.testing.assert_allclose(f0,-f1,atol=0)
    np.testing.assert_allclose(f0,[1,0,0],atol=1e-12)
    assert T==pytest.approx(1.)


def test_no_spurious_damping_for_slack_moving_nodes():
    f0,f1,T=spring_pair_forces([0,0,0],[0.3,0,0],[0,0,0],[100,0,0],1.,1000,5000)
    assert T==0
    np.testing.assert_array_equal(f0,np.zeros(3))
    np.testing.assert_array_equal(f1,np.zeros(3))


def test_two_free_masses_conserve_linear_momentum_and_energy_without_damping():
    m0,m1=1.2,0.7
    def rhs(_, y):
        r0,r1,v0,v1=y[:3],y[3:6],y[6:9],y[9:12]
        f0,f1,_=spring_pair_forces(r0,r1,v0,v1,1.,50.,0.)
        return np.r_[v0,v1,f0/m0,f1/m1]
    y0=np.r_[np.zeros(3),[1.2,0,0],np.zeros(3),np.zeros(3)]
    sol=solve_ivp(rhs,(0,2.),y0,method='DOP853',rtol=2e-10,atol=1e-12,
                  max_step=0.005,t_eval=np.linspace(0,2,201))
    assert sol.success
    mom=m0*sol.y[6:9].T + m1*sol.y[9:12].T
    np.testing.assert_allclose(mom,np.zeros_like(mom),atol=2e-8)
    delta=sol.y[3:6].T-sol.y[:3].T
    U=25*np.maximum(np.linalg.norm(delta,axis=1)-1.,0)**2
    K=0.5*m0*np.sum(sol.y[6:9]**2,axis=0)+0.5*m1*np.sum(sol.y[9:12]**2,axis=0)
    assert np.max(abs((U+K)-(U+K)[0])) < 3e-6


def test_two_free_masses_with_damping_reduce_mechanical_energy():
    m0=m1=1.
    def rhs(_,y):
        f0,f1,_=spring_pair_forces(y[:3],y[3:6],y[6:9],y[9:],1.,25.,1.)
        return np.r_[y[6:],f0/m0,f1/m1]
    y0=np.r_[np.zeros(3),[1.2,0,0],np.zeros(6)]
    sol=solve_ivp(rhs,(0,0.3),y0,method='DOP853',rtol=1e-10,atol=1e-12,max_step=0.001)
    y=sol.y[:,-1]; separation=np.linalg.norm(y[3:6]-y[:3])
    energy=0.5*(y[6:]@y[6:])+12.5*max(separation-1.,0)**2
    assert energy < 12.5*0.2**2
    assert energy >= 0


def test_mass_accounting_com_and_rod_inertia_convergence():
    length=2.7; linear_density=0.12
    errors=[]
    for n in [2,4,8,16,32]:
        c=RoundWire(.01); rho=linear_density/c.area_m2
        cfg=CableModel(HomogeneousTether(c,length,rho,1e6,1e5),n)
        masses=cfg.node_masses_kg
        assert sum(masses)==pytest.approx(linear_density*length)
        assert masses[0]==pytest.approx(linear_density*length/(2*n))
        assert masses[-1]==pytest.approx(masses[0])
        total,com,I=mass_properties_I(cfg,straight(cfg))
        assert total==pytest.approx(linear_density*length)
        np.testing.assert_allclose(com,[length/2,0,0],atol=1e-12)
        analytic=total*length**2/12
        errors.append(abs(I[1,1]-analytic))
        assert I[0,0]==pytest.approx(0.,abs=1e-12)
        assert I[2,2]==pytest.approx(I[1,1])
    assert all(a>b for a,b in zip(errors,errors[1:]))
    assert errors[-1]<errors[0]/200


def test_end_mass_kept_separate_and_correct_point_inertia():
    cfg=model(4,tip=1.2)
    total,com,I=mass_properties_I(cfg,straight(cfg))
    assert total==pytest.approx(cfg.tether.mass_kg+1.2)
    expected_com=(cfg.tether.mass_kg*cfg.tether.length_m/2 + 1.2*cfg.tether.length_m)/total
    assert com[0]==pytest.approx(expected_com)
    assert np.linalg.eigvalsh(I).min()>=-1e-12


def test_nominal_rest_cable_does_not_collapse_no_gravity():
    cfg=model(5,tip=.2,damping=.5)
    r0=straight(cfg)
    h=run(cfg,r0,t_end=.6,method='Radau')
    np.testing.assert_allclose(h.r_nodes_I_m, np.broadcast_to(r0,h.r_nodes_I_m.shape),atol=1e-12)
    assert np.max(abs(h.segment_tensions_N))==0
    assert np.max(abs(h.attachment_force_on_bus_I_N))==0
    np.testing.assert_allclose(h.actual_segment_lengths_m,cfg.rest_length_m,atol=1e-12)
    assert h.solver_diagnostics['mass_scale_used']==1.0
    assert h.solver_diagnostics['current_A']==0.0
    assert h.solver_diagnostics['jac_sparsity'].startswith('disabled')
    assert not h.solver_diagnostics['energy_conservation_applicable']  # damping enabled


def test_transient_extended_cable_dynamics_and_energy_conservation():
    cfg=model(1,tip=.2)
    r0=straight(cfg,extension=.06)
    h=run(cfg,r0,.5,method='DOP853',max_step_s=0.001)
    assert h.segment_tensions_N[0,0]>0
    assert h.segment_tensions_N.max()>0
    assert np.max(abs(h.total_mechanical_energy_J-h.total_mechanical_energy_J[0])) < 1e-5
    assert np.min(h.actual_segment_lengths_m)>0
    assert h.inertia_about_com_I_kgm2.shape==(len(h.t_s),3,3)
    assert h.tether_state(-1).r_nodes_I.shape==(2,3)
    np.testing.assert_allclose(h.attachment_reaction(0).force_on_bus_I,
                               h.attachment_force_on_bus_I_N[0])


def test_damped_cable_energy_decreases_and_no_mass_inflation():
    cfg=model(1,tip=.2,damping=.1)
    h=run(cfg,straight(cfg,extension=.04),.1,max_step_s=0.001)
    assert h.total_mechanical_energy_J[-1] < h.total_mechanical_energy_J[0]
    assert sum(h.node_masses_kg)==pytest.approx(cfg.total_mass_kg)
    assert h.solver_diagnostics['mass_scale_used']==1.0


def test_static_one_segment_gravity_analytical_extension_reaction_and_transient_settle():
    cfg=model(1,tip=.3,damping=.1)
    g=np.array([0,0,-9.81]); L=cfg.rest_length_m
    m1=cfg.node_masses_kg[1]
    expected_elongation=m1*np.linalg.norm(g)/cfg.spring_constant_N_m
    r=np.array([[0,0,0],[0,0,-(L+expected_elongation*.8)]])
    equ=solve_static_equilibrium(cfg,r,[0,0,0],g,gtol=1e-11)
    assert equ.actual_segment_lengths_m[0] == pytest.approx(L+expected_elongation,rel=1e-8)
    assert equ.segment_tensions_N[0] == pytest.approx(m1*9.81,rel=1e-7)
    np.testing.assert_allclose(equ.attachment_force_on_bus_I_N,[0,0,-cfg.total_mass_kg*9.81],atol=1e-6)
    assert equ.solver_diagnostics['mass_scale_used']==1.0
    # Static equilibrium is a separate optimization, not a time-history shortcut.
    assert not hasattr(equ,'t_s')


def test_stationary_gravity_no_initial_tension_does_not_invent_0p4N():
    cfg=model(5)
    r0=straight(cfg,direction=(0,0,-1))
    h=run(cfg,r0,.001,gravity=[0,0,-9.81],method='Radau')
    assert np.max(abs(h.segment_tensions_N[0]))==0.0
    assert np.linalg.norm(h.attachment_force_on_bus_I_N[0])==pytest.approx(cfg.node_masses_kg[0]*9.81)


def test_prescribed_attachment_acceleration_and_external_anchor_load_reaction():
    cfg=model(1)
    r0=straight(cfg)
    ar=constant_attachment([0,0,0],acceleration_I=[2,0,0])
    def ext(t,state):
        f=np.zeros((2,3)); f[0]=[3,0,0];return f
    h=simulate_cable(cfg,[0,.001],r0,np.zeros_like(r0),attachment=ar,
                     external_nodal_forces_I=ext,method='RK45')
    expected=3-2*cfg.node_masses_kg[0]
    assert h.attachment_force_on_bus_I_N[0,0]==pytest.approx(expected)
    assert h.solver_diagnostics['current_A'] is None
    assert not h.solver_diagnostics['energy_conservation_applicable']


def test_energy_stiffness_and_timestep_convergence():
    cfg=model(1,tip=.2)
    r0=straight(cfg,extension=.03)
    values=[]
    for step in [.02,.01,.005]:
        h=run(cfg,r0,.1,method='RK45',max_step_s=step,rtol=1e-5,atol=1e-8)
        values.append(h.r_nodes_I_m[-1,-1,0])
    reference=run(cfg,r0,.1,method='DOP853',max_step_s=.0005).r_nodes_I_m[-1,-1,0]
    assert abs(values[-1]-reference)<2e-4
    assert abs(values[-1]-reference) <= abs(values[0]-reference)+1e-8


def test_finite_difference_elastic_gradient_matches_pair_forces():
    r0=np.array([0.,0.,0.]);r1=np.array([1.25,.2,0.1]);l0=1.;EA=50.
    f0,f1,_=spring_pair_forces(r0,r1,[0,0,0],[0,0,0],l0,EA)
    def U(x):
        return .5*EA/l0*max(np.linalg.norm(x-r0)-l0,0)**2
    h=1e-6
    finite_grad=np.array([(U(r1+np.eye(3)[j]*h)-U(r1-np.eye(3)[j]*h))/(2*h) for j in range(3)])
    np.testing.assert_allclose(-finite_grad,f1,rtol=1e-8,atol=1e-8)


@pytest.mark.parametrize('bad',[-1,np.inf,np.nan])
def test_invalid_damping_rejected(bad):
    with pytest.raises(ValueError):
        model(damping=bad)


def test_invalid_mesh_initial_conditions_and_times_rejected():
    with pytest.raises(ValueError): model(n=0)
    cfg=model(2); r0=straight(cfg)
    with pytest.raises(ValueError):
        simulate_cable(cfg,[0,0],r0,np.zeros_like(r0),attachment=constant_attachment([0,0,0]))
    with pytest.raises(ValueError):
        simulate_cable(cfg,[0,1],r0,np.zeros_like(r0),attachment=constant_attachment([1,0,0]))
    with pytest.raises(ValueError):
        simulate_cable(cfg,[0,1],r0,np.zeros_like(r0),attachment=constant_attachment([0,0,0]),
                       gravity_I=[np.nan,0,0])


def test_segment_count_convergence_of_axial_transient():
    # Same physical cable & tip mass, increasingly fine mesh. This measures
    # convergence of a *physical transient*, not just static rod inertia.
    values=[]
    for n in [2,4,8,16,32]:
        cfg=model(n,length=2.,EA_scale=.005,tip=.2)
        r0=straight(cfg,extension=.04)
        h=run(cfg,r0,.4,method='DOP853',max_step_s=.005)
        values.append(float(h.r_nodes_I_m[-1,-1,0]))
        assert h.solver_diagnostics['mass_scale_used']==1.
    diffs=np.abs(np.diff(values))
    assert diffs[-1] < diffs[0]/10
    assert abs(values[-1]-values[-2]) < 3e-6


def test_tolerance_refinement_against_reference():
    cfg=model(2,EA_scale=.2,tip=.1)
    r0=straight(cfg,extension=.02)
    ref=run(cfg,r0,.15,method='DOP853',rtol=5e-11,atol=1e-13,max_step_s=.001)
    loose=run(cfg,r0,.15,method='DOP853',rtol=1e-3,atol=1e-6,max_step_s=.05)
    tight=run(cfg,r0,.15,method='DOP853',rtol=1e-9,atol=1e-11,max_step_s=.005)
    e_loose=np.linalg.norm(loose.r_nodes_I_m[-1]-ref.r_nodes_I_m[-1])
    e_tight=np.linalg.norm(tight.r_nodes_I_m[-1]-ref.r_nodes_I_m[-1])
    assert e_tight < max(e_loose,1e-10)
