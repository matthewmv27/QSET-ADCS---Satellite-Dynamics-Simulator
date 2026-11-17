"""M4e: external distributed loads and diagnostic OML/CNT capacity (no orbit dynamics)."""
import json
from pathlib import Path
import numpy as np
import pytest

from edtsim.core.frames import attachment_reaction_on_bus
from edtsim.core.states import TetherState
from edtsim.electrodynamics import (ZeroCurrent, PrescribedConstantCurrent,
    SpatialCurrentProfile, EmpiricalCNTCathode, check_cnt_demand,
    solve_prescribed_bias_snapshot, evaluate_distributed_loading)
from edtsim.tether import (HomogeneousTether, RoundWire, RectangularTape)
from scripts.run_m4e_loading_reference import run


class ConstField:
    def __init__(self, B=(0, 1, 0)):
        self.B = B

    def sample(self, t, pos):
        return self.B


def straight(n=5, length=2, transverse=0, velocity=(1,0,0)):
    zs = np.linspace(0, length, n)
    return TetherState(np.stack([np.full(n,transverse), np.zeros(n), zs],axis=1),
                       np.tile(velocity,(n,1)))


def wire(length=2, bare=1):
    return HomogeneousTether(RoundWire(.0002),length,2700,70e9,35e6,bare)


def test_analytic_uniform_lorentz_force_and_torque_on_tether():
    state=straight(n=5,length=2,transverse=.5)
    a=evaluate_distributed_loading(0,state,ConstField(),PrescribedConstantCurrent(.01),
                                   reference_point_I_m=[0,0,0])
    # dl=z; B=y -> dl x B = -x. Net = -0.02 N.
    np.testing.assert_allclose(a.net_tether_external_force_I_N,[-.02,0,0],atol=1e-14)
    np.testing.assert_allclose(a.net_tether_external_torque_about_ref_I_Nm,[0,-.02,0],atol=1e-14)
    assert a.power_identity_residual_W == pytest.approx(0,abs=1e-13)
    assert a.lorentz_power_local_W == pytest.approx(-.02)
    assert a.motional_emf_power_W == pytest.approx(.02)
    assert a.model_warning.startswith('EXTERNAL tether load')


def test_mesh_refinement_invariance_uniform_field():
    for n in (2,3,5,51,501):
        a=evaluate_distributed_loading(0,straight(n,2),ConstField(),
             PrescribedConstantCurrent(.01),reference_point_I_m=[0,0,0])
        np.testing.assert_allclose(a.net_tether_external_force_I_N,[-.02,0,0],atol=1e-12)
        np.testing.assert_allclose(a.net_tether_external_torque_about_ref_I_Nm,[0,-.02,0],atol=1e-12)


def test_force_sign_reverses_with_conventional_current():
    state=straight()
    a=evaluate_distributed_loading(0,state,ConstField(),PrescribedConstantCurrent(.01),reference_point_I_m=[0,0,0])
    b=evaluate_distributed_loading(0,state,ConstField(),PrescribedConstantCurrent(-.01),reference_point_I_m=[0,0,0])
    np.testing.assert_allclose(a.net_tether_external_force_I_N,-b.net_tether_external_force_I_N)
    np.testing.assert_allclose(a.net_tether_external_torque_about_ref_I_Nm,-b.net_tether_external_torque_about_ref_I_Nm)


def test_power_identity_includes_plasma_velocity():
    # Moving plasma: local mechanical power != minus motional EMF power.
    a=evaluate_distributed_loading(2,straight(),ConstField(),PrescribedConstantCurrent(.01),
            reference_point_I_m=[0,0,0],plasma_velocity_provider=lambda t,r:[.2,0,0])
    assert a.lorentz_power_local_W == pytest.approx(-.02)
    assert a.plasma_motion_power_W == pytest.approx(-.004)
    assert a.motional_emf_power_W == pytest.approx(.016)
    assert a.power_identity_residual_W == pytest.approx(0,abs=1e-13)


def test_zero_current_is_zero_force_even_with_nonzero_B():
    a=evaluate_distributed_loading(0,straight(),ConstField(),ZeroCurrent(),reference_point_I_m=[0,0,0])
    np.testing.assert_array_equal(a.net_tether_external_force_I_N,np.zeros(3))
    np.testing.assert_array_equal(a.net_tether_external_torque_about_ref_I_Nm,np.zeros(3))
    assert a.lorentz_power_local_W == 0


def test_nonuniform_field_midpoint_result_and_torque():
    class LinearField:
        def sample(self,t,p): return [0,1+p[2],0]
    a=evaluate_distributed_loading(0,straight(n=51),LinearField(),PrescribedConstantCurrent(.01),
                                   reference_point_I_m=[0,0,0])
    # analytic integral_0^2 (1+z) dz =4
    np.testing.assert_allclose(a.net_tether_external_force_I_N,[-.04,0,0],atol=1e-12)
    # torque_y = integral z F_x dz = -I*(L^2/2 + L^3/3) = -.0466667
    assert a.net_tether_external_torque_about_ref_I_Nm[1] == pytest.approx(-.01*(2+8/3),rel=0.005)



def test_bent_tether_segment_forces_follow_actual_element_directions():
    state=TetherState([[0,0,0],[1,0,1],[1,0,2]],np.zeros((3,3)))
    a=evaluate_distributed_loading(0,state,ConstField((0,0,2)),
            PrescribedConstantCurrent(.01),reference_point_I_m=[0,0,0])
    # First segment [1,0,1] x [0,0,2] = [0,-2,0]; second zero.
    np.testing.assert_allclose(a.net_tether_external_force_I_N,[0,-.02,0],atol=1e-14)
    assert a.net_tether_external_torque_about_ref_I_Nm[0] > 0
    assert a.power_identity_residual_W == pytest.approx(0,abs=1e-13)


def test_stationary_relative_neutral_flow_has_zero_drag():
    state=straight()
    a=evaluate_distributed_loading(0,state,ConstField(),ZeroCurrent(),
            reference_point_I_m=[0,0,0],tether=wire(),
            neutral_velocity_provider=lambda t,p:[1,0,0],
            neutral_density_provider=lambda t,p:1e-10,
            drag_coefficient=2.2)
    np.testing.assert_array_equal(a.drag_forces_I_N,np.zeros_like(a.drag_forces_I_N))


def test_drag_requires_explicit_model_inputs_and_dissipates_relative_energy():
    state=straight(5,2)
    rho=lambda t,r:1e-12
    vn=lambda t,r:[0,0,0]
    with pytest.raises(ValueError,match='requires'):
        evaluate_distributed_loading(0,state,ConstField(),ZeroCurrent(),
                                     reference_point_I_m=[0,0,0],drag_coefficient=2.2)
    a=evaluate_distributed_loading(0,state,ConstField(),ZeroCurrent(),
            reference_point_I_m=[0,0,0], tether=wire(),neutral_velocity_provider=vn,
            neutral_density_provider=rho,drag_coefficient=2.2)
    assert a.net_tether_external_force_I_N[0]<0
    assert a.drag_power_local_W<0
    assert np.all(a.lorentz_forces_I_N==0)
    with pytest.raises(ValueError,match='density'):
        evaluate_distributed_loading(0,state,ConstField(),ZeroCurrent(),
            reference_point_I_m=[0,0,0],tether=wire(),neutral_velocity_provider=vn,
            neutral_density_provider=lambda t,r:-1,drag_coefficient=2.2)


def test_tether_length_mismatch_and_degenerate_geometry_rejected():
    with pytest.raises(ValueError,match='length'):
        evaluate_distributed_loading(0,straight(),ConstField(),ZeroCurrent(),
            reference_point_I_m=[0,0,0],tether=wire(3),
            neutral_velocity_provider=lambda t,r:[0,0,0],
            neutral_density_provider=lambda t,r:1e-12,drag_coefficient=2)
    nodes=straight().r_nodes_I.copy()
    nodes[1]=nodes[0]
    bad=TetherState(nodes,np.zeros_like(nodes))
    with pytest.raises(ValueError,match='positive-length'):
        evaluate_distributed_loading(0,bad,ConstField(),ZeroCurrent(),reference_point_I_m=[0,0,0])


def test_oml_profile_interpolation_and_immutable_copy():
    s=[0,1,2]
    I=[.1,.05,0]
    profile=SpatialCurrentProfile(s,I,'synthetic assumed current')
    s[1]=200
    I[1]=999
    np.testing.assert_allclose(profile.segment_currents_A(0,[.5,1.5]),[.075,.025])
    assert not profile.s_m.flags.writeable
    with pytest.raises(ValueError,match='domain'):
        profile.segment_currents_A(0,[2.1])
    with pytest.raises(ValueError):
        SpatialCurrentProfile([0,2,1],[1,1,1],'wrong')
    with pytest.raises(ValueError):
        SpatialCurrentProfile([0,2],[1,1],'')


def test_oml_profile_and_cnt_demand_are_kept_separate():
    snap=solve_prescribed_bias_snapshot(wire(100,.5),
          electron_density_m3=1e11,electron_temperature_K=1500,
          bus_plasma_bias_V=2,motional_E_along_V_m=.2,
          available_gate_voltage_V=500,nodes=51)
    assert snap.profile.length_m==100
    assert snap.oml_solution.tip_current_residual_A<1e-9
    np.testing.assert_allclose(snap.profile.segment_currents_A(0,[0,100]),
                               [snap.oml_solution.conventional_current_A[0],
                                snap.oml_solution.conventional_current_A[-1]])
    assert snap.cathode_check.status.startswith('diagnostic')
    assert 'NOT' in snap.profile.provenance
    assert snap.cathode_check.current_required_A>0
    assert snap.cathode_check.supports_assumed_demand


def test_infeasible_cathode_capacity_does_not_silently_clip_current():
    constraint=check_cnt_demand(.0025,EmpiricalCNTCathode(),
        available_gate_voltage_V=100,max_cathode_power_W=.01,max_emission_current_A=.001)
    assert not constraint.supports_assumed_demand
    assert not constraint.within_gate_voltage_limit
    assert not constraint.within_power_limit
    assert not constraint.within_assumed_current_limit
    assert constraint.current_required_A==pytest.approx(.0025)
    zero=check_cnt_demand(0,available_gate_voltage_V=0,max_cathode_power_W=0)
    assert zero.gate_voltage_required_V == 0
    assert zero.estimated_cathode_power_W == 0
    with pytest.raises(ValueError):
        check_cnt_demand(.1,available_gate_voltage_V=-1)


def test_tape_drag_needs_normal_width_axis():
    tape=HomogeneousTether(RectangularTape(.01,.0001),2,2700,70e9,35e6)
    args=dict(reference_point_I_m=[0,0,0],tether=tape,
        neutral_velocity_provider=lambda t,r:[0,0,0],
        neutral_density_provider=lambda t,r:1e-12,drag_coefficient=2)
    with pytest.raises(ValueError,match='width_dir'):
        evaluate_distributed_loading(0,straight(),ConstField(),ZeroCurrent(),**args)
    result=evaluate_distributed_loading(0,straight(),ConstField(),ZeroCurrent(),
                                        tape_width_direction_I=[0,1,0],**args)
    assert result.drag_power_local_W<0


def test_invalid_current_source_field_and_plasma_flow_rejected():
    with pytest.raises(TypeError,match='field'):
        evaluate_distributed_loading(0,straight(),object(),ZeroCurrent(),reference_point_I_m=[0,0,0])
    with pytest.raises(ValueError,match='current source'):
        class Bad:
            def segment_currents_A(self,t,s): return [1]
        evaluate_distributed_loading(0,straight(),ConstField(),Bad(),reference_point_I_m=[0,0,0])
    with pytest.raises(ValueError,match='plasma velocity'):
        evaluate_distributed_loading(0,straight(),ConstField(),ZeroCurrent(),
           reference_point_I_m=[0,0,0],plasma_velocity_provider=lambda t,p:[float('nan'),0,0])


def test_external_moment_is_not_the_bus_attachment_torque():
    # Distinct interface: bus force is supplied by a mechanical solver.
    external=evaluate_distributed_loading(0,straight(5,2),ConstField(),
         PrescribedConstantCurrent(.01),reference_point_I_m=[0,0,0])
    force_B,torque_B=attachment_reaction_on_bus([1,0,0,0],[.5,0,0],[0,0,0])
    np.testing.assert_array_equal(force_B,np.zeros(3))
    np.testing.assert_array_equal(torque_B,np.zeros(3))
    assert np.linalg.norm(external.net_tether_external_torque_about_ref_I_Nm)>0


def test_m4e_runner_exports_and_flags_preliminary(tmp_path):
    src=Path(__file__).resolve().parents[1]/'configs/scenarios/m4e_loading_reference.json'
    summary=run(src,tmp_path/'out')
    assert summary['milestone']=='M4e'
    assert summary['operating_point_closed'] is False
    assert summary['within_assumed_cathode_limits']
    assert summary['net_external_tether_force_I_N'][1]<0
    assert abs(summary['power_identity_residual_W'])<1e-12
    assert (tmp_path/'out'/'segment_loading.csv').exists()
    written=json.loads((tmp_path/'out'/'summary.json').read_text())
    assert written['current_at_bus_A']==summary['current_at_bus_A']
    bad=json.loads(src.read_text())
    bad['environment']['mode']='igrf14'
    x=tmp_path/'bad.json'
    x.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='synthetic'):
        run(x,tmp_path/'bad_out')
