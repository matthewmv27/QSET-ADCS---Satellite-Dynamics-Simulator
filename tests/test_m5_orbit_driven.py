"""Prescribed orbit M5: consistent kinematics, gravity and electrical load transfer."""
import json
from pathlib import Path

import numpy as np
import pytest

from edtsim.core.constants import MU_EARTH
from edtsim.core.states import TetherState
from edtsim.electrodynamics.current import PrescribedConstantCurrent
from edtsim.electrodynamics.loading import evaluate_distributed_loading
from edtsim.electrodynamics.tether_adapter import (
    segment_forces_to_nodes, prescribed_loading_callback)
from edtsim.environment.fields import UniformMagneticField
from scripts.run_m5_mechanical_reference import _model
from scripts.run_m5_orbit_driven_reference import (
    circular_orbit_from_config, circular_attachment, central_gravity_I,
    TranslatedCentralGravity, straight_initial_state, orbital_frame,
    run, SCENARIOS)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/scenarios/m5_orbit_driven_500km.json'


def config():
    return json.loads(CONFIG.read_text())


def test_prescribed_orbit_has_two_body_acceleration_and_fixed_raan():
    orbit = circular_orbit_from_config(config())
    assert orbit.altitude_m == pytest.approx(500000)
    assert orbit.precess_raan is False
    assert 97 < orbit.inclination_deg < 98
    r, v = orbit.state(20.)
    dr = (orbit.state(20.01)[0] - orbit.state(19.99)[0]) / .02
    dv = (orbit.state(20.01)[1] - orbit.state(19.99)[1]) / .02
    np.testing.assert_allclose(dr, v, rtol=1e-9, atol=1e-4)
    np.testing.assert_allclose(dv, central_gravity_I(20.,r), rtol=2e-9, atol=2e-6)
    assert abs(r @ v) < 1e-5


def test_orbit_attachment_and_translated_gravity_consistent():
    orbit = circular_orbit_from_config(config())
    origin = orbit.state(0.)[0]
    attachment = circular_attachment(orbit, origin)
    g = TranslatedCentralGravity(origin)
    for t in (0., 5., 25.):
        position, v, acceleration = attachment(t)
        np.testing.assert_allclose(acceleration, g(t, position), atol=2e-15)
        np.testing.assert_allclose(acceleration,
                                   g.sample_nodes(t,np.array([position]))[0],atol=2e-15)
    with pytest.raises(ValueError):
        circular_orbit_from_config({**config(), 'orbit':{**config()['orbit'], 'precess_raan':True}})


def test_straight_initialization_has_zero_bow_and_corotating_velocity():
    data = config()
    model = _model(data)
    orbit = circular_orbit_from_config(data)
    origin = orbit.state(0.)[0]
    r0, v0 = straight_initial_state(model, orbit, origin_I_m=origin)
    _, v_attach, er, et, eh = orbital_frame(orbit, 0.)
    np.testing.assert_allclose(r0[0], [0.,0.,0.],atol=0)
    np.testing.assert_allclose(v0[0], v_attach,atol=0)
    np.testing.assert_allclose(r0[-1],50*er,atol=1e-11)
    assert abs((r0[-1]-r0[0])@et) < 1e-11
    np.testing.assert_allclose(v0[-1]-v_attach,
        np.cross(orbit.mean_motion_rad_s*eh, 50*er), atol=1e-12)
    assert np.isclose(sum(model.node_masses_kg), model.total_mass_kg)


def test_small_initial_angle_is_still_a_straight_line():
    data=config(); model=_model(data); orbit=circular_orbit_from_config(data)
    origin=orbit.state(0.)[0]
    r,_=straight_initial_state(model,orbit,2.,origin_I_m=origin)
    u=np.linspace(0.,1.,model.segments+1)
    np.testing.assert_allclose(r,u[:,None]*r[-1],atol=1e-11)


def test_nodal_force_partition_preserves_net_force_and_midpoint_torque():
    fseg=np.array([[1.,2.,3.],[-2.,0.5,1.],[0.,1.,-4.]])
    nodes=np.array([[0.,0.,0.],[2.,0.,0.],[4.,0.,0.],[6.,0.,0.]])
    fn=segment_forces_to_nodes(fseg)
    np.testing.assert_allclose(fn.sum(axis=0),fseg.sum(axis=0),atol=0)
    tau_nodes=np.cross(nodes,fn).sum(axis=0)
    tau_mid=np.cross((nodes[:-1]+nodes[1:])/2,fseg).sum(axis=0)
    np.testing.assert_allclose(tau_nodes,tau_mid,atol=1e-14)
    with pytest.raises(ValueError):
        segment_forces_to_nodes([[float('nan'),0,0]])


def test_m4e_loading_callback_matches_full_segment_loading():
    pos=np.array([[0.,0,0],[1.,0,0],[2.,0.1,0]])
    vel=np.array([[0.,1.,0],[0.,1.,0],[0.,1.,0]])
    state=TetherState(pos,vel)
    field=UniformMagneticField((0.,0.,3e-5))
    current=PrescribedConstantCurrent(.0025)
    expected=evaluate_distributed_loading(0.,state,field,current,reference_point_I_m=np.zeros(3))
    f=prescribed_loading_callback(field,current)(0.,state)
    np.testing.assert_allclose(f,segment_forces_to_nodes(expected.total_segment_forces_I_N),atol=1e-18)
    assert f.sum(axis=0)[1] < 0 # positive +X current cross +Z B => -Y


def test_orbit_driven_straight_control_and_lorentz_comparison(tmp_path):
    data=config()
    data['mechanics']['segments']=8
    data['integration'].update(duration_s=3.0,samples=7,max_step_s=.15)
    input_file=tmp_path/'small.json';input_file.write_text(json.dumps(data))
    summary=run(input_file,tmp_path/'output')
    assert list(summary['results'])==list(SCENARIOS)
    assert summary['orbit']['precess_raan'] is False
    assert summary['inertial_positions_use_fixed_translation'] is True
    assert summary['electrical']['actual_CNT_OML_circuit_closed'] is False
    assert summary['tether']['mass_scale_used'] == 1.0
    for case in SCENARIOS:
        case_dir=tmp_path/'output'/case
        assert (case_dir/'history.npz').exists()
        assert (case_dir/'history.csv').exists()
        assert (case_dir/'shape_samples.csv').exists()
        assert summary['results'][case]['solver']['success']
        with np.load(case_dir/'history.npz') as history:
            assert history['positions_LVLH_m'].shape == (7,9,3)
            assert history['segment_tensions_N'].shape == (7,8)
            assert (history['segment_tensions_N']>=0).all()
    base=summary['results']['straight_gravity']
    loaded=summary['results']['straight_lorentz']
    tilted=summary['results']['tilted_gravity']
    assert abs(base['initial_tip_LVLH_m'][1])<1e-9
    assert abs(loaded['initial_tip_LVLH_m'][1])<1e-9
    assert tilted['initial_tip_LVLH_m'][1]>0.5
    assert loaded['final_tip_LVLH_m'][1]<base['final_tip_LVLH_m'][1]
    assert base['max_tension_N']>0


def test_reject_precessing_circular_reference_and_zero_current_claim(tmp_path):
    data=config(); data['orbit']['precess_raan']=True
    path=tmp_path/'bad.json';path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='precess_raan'):
        run(path,tmp_path/'out')
    data['orbit']['precess_raan']=False
    data['electrodynamics']['assumed_signed_uniform_current_A']=0.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='requires nonzero'):
        run(path,tmp_path/'out')


def test_orbit_mesh_comparison_preserves_mass_and_reference_zero(tmp_path):
    from scripts.run_m5_orbit_mesh_convergence import run_study
    data=config()
    data['integration'].update(duration_s=.4,samples=3,max_step_s=.2)
    path=tmp_path/'mesh.json';path.write_text(json.dumps(data))
    result=run_study(path,tmp_path/'study',segment_counts=(4,8),cases=('straight_gravity',))
    assert len(result['rows'])==2
    assert result['rows'][1]['final_shape_rms_difference_m_vs_finest']==pytest.approx(0,abs=1e-12)
    assert result['rows'][0]['mass_kg']==pytest.approx(result['rows'][1]['mass_kg'])
    assert (tmp_path/'study/mesh_comparison.csv').exists()
