"""Independent M4 analytical and numerical sign/conservation checks."""
import json
import numpy as np
import pytest

from edtsim.core.constants import MU_EARTH, R_EARTH_EQ
from edtsim.environment import (UniformMagneticField, CenteredDipoleMagneticField,
                                 ConstantAtmosphericDensity,ExponentialAtmosphericDensity)
from edtsim.tether import RoundWire, RectangularTape, HomogeneousTether
from edtsim.electrodynamics import EmpiricalCNTCathode, PrescribedConstantCurrent, ZeroCurrent
from edtsim.orbit_dynamics import (earth_gravity_J2_I, earth_specific_potential_J2,
                                   RadialTetherForces, propagate_orbit_decay)


def example_tether(cross=None):
    return HomogeneousTether(cross or RoundWire(.001), 100.,2700,70e9,35e6)


def example_forces(B=(0,0,3e-5),current=.0025,braking=False,rho=0):
    return RadialTetherForces(UniformMagneticField(B),ConstantAtmosphericDensity(rho),
         PrescribedConstantCurrent(current),example_tether(),4.,bus_projected_area_m2=0,
         tether_Cd=0,segments=10,braking_only=braking)


def test_wire_geometry_and_material_derived_values():
    d=.001; L=100.; rho=2700.; sigma=35e6; E=70e9
    section=RoundWire(d)
    np.testing.assert_allclose(section.area_m2,np.pi*d*d/4)
    np.testing.assert_allclose(section.perimeter_m,np.pi*d)
    tether=HomogeneousTether(section,L,rho,E,sigma,bare_fraction=.5)
    assert tether.mass_kg==pytest.approx(L*rho*np.pi*d*d/4)
    assert tether.axial_rigidity_N==pytest.approx(E*np.pi*d*d/4)
    assert tether.resistance_ohm==pytest.approx(L/(sigma*section.area_m2))
    assert tether.bare_length_m==50
    assert section.segment_projected_area_m2(2,[0,0,1],[1,0,0])==pytest.approx(2*d)


def test_tape_geometry_projection_and_validation():
    tape=RectangularTape(.01,.0001)
    assert tape.area_m2==pytest.approx(1e-6)
    assert tape.perimeter_m==pytest.approx(.0202)
    # tape broad-face normal is +y if axis +z and width +x
    assert tape.segment_projected_area_m2(5,[0,0,1],[0,1,0],[1,0,0])==pytest.approx(.05)
    assert tape.segment_projected_area_m2(5,[0,0,1],[1,0,0],[1,0,0])==pytest.approx(.0005)
    with pytest.raises(ValueError,match='parallel'):
        tape.segment_projected_area_m2(5,[0,0,1],[1,0,0],[0,0,2])


@pytest.mark.parametrize('diameter',[-1,0,float('nan')])
def test_invalid_wire_diameters(diameter):
    with pytest.raises(ValueError):RoundWire(diameter)


def test_dipole_inverse_cube_law_and_pole_field():
    f=CenteredDipoleMagneticField(3e-5,(0,0,1),spin_rad_s=0)
    r=R_EARTH_EQ
    np.testing.assert_allclose(f.sample(0,[r,0,0]),[0,0,-3e-5])
    np.testing.assert_allclose(f.sample(0,[0,0,r]),[0,0,6e-5])
    np.testing.assert_allclose(f.sample(0,[2*r,0,0]),f.sample(0,[r,0,0])/8)


def test_exponential_density_has_strict_height_limits():
    m=ExponentialAtmosphericDensity(500e3,1e-12,70e3,400e3,550e3)
    assert m.sample(0,[R_EARTH_EQ+500e3,0,0])==pytest.approx(1e-12)
    assert m.sample(0,[R_EARTH_EQ+430e3,0,0])==pytest.approx(1e-12*np.e)
    with pytest.raises(ValueError,match='outside'):
        m.sample(0,[R_EARTH_EQ+600e3,0,0])


def test_j2_gravity_is_negative_gradient_of_potential():
    r=np.array([R_EARTH_EQ+500e3,110e3,220e3])
    h=10.
    grad=np.array([(earth_specific_potential_J2(r+np.eye(3)[i]*h)-earth_specific_potential_J2(r-np.eye(3)[i]*h))/(2*h)
                   for i in range(3)])
    np.testing.assert_allclose(earth_gravity_J2_I(r),-grad,rtol=2e-7,atol=1e-7)


def test_lorentz_current_polarity_along_track():
    r=np.array([R_EARTH_EQ+500e3,0,0.]);v=np.array([0.,7600.,0.])
    f=example_forces(B=(0,0,3e-5),current=.0025)
    L,drag,cur=f.forces_I(0,r,v)
    np.testing.assert_allclose(L,[0,-.0025*100*3e-5,0],rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(drag,[0,0,0])
    assert np.all(cur>0)
    rev=example_forces(B=(0,0,3e-5),current=-.0025)
    np.testing.assert_allclose(rev.forces_I(0,r,v)[0],-L)
    opposite=example_forces(B=(0,0,-3e-5),current=.0025,braking=True)
    np.testing.assert_array_equal(opposite.forces_I(0,r,v)[0],[0,0,0])
    np.testing.assert_array_equal(opposite.forces_I(0,r,v)[2],np.zeros(10))


def test_orbit_zero_forces_j2_conserves_total_mechanical_energy():
    r=R_EARTH_EQ+500e3
    # equatorial circular two-body speed is only an initial condition under J2
    v=np.sqrt(MU_EARTH/r)
    f=RadialTetherForces(UniformMagneticField((0,0,3e-5)),ConstantAtmosphericDensity(0),
        ZeroCurrent(),example_tether(),4.,bus_projected_area_m2=0,bus_Cd=0,tether_Cd=0,segments=2)
    out=propagate_orbit_decay([r,0,0],[0,v,0],np.linspace(0,5500,18),f,max_step_s=90)
    assert max(abs(out.mechanical_energy_J-out.mechanical_energy_J[0]))<1e-4
    assert np.all(out.lorentz_force_I_N==0)
    assert np.all(out.drag_force_I_N==0)
    assert np.all(out.work_lorentz_J==0)
    assert np.all(out.work_drag_J==0)


def test_orbit_energy_work_balance_and_current_sign():
    r=R_EARTH_EQ+500e3;v=np.sqrt(MU_EARTH/r)
    ts=np.linspace(0,200,10)
    runs=[]
    for polarity in [.0025,-.0025]:
        forces=RadialTetherForces(UniformMagneticField((0,0,3e-5)),ConstantAtmosphericDensity(1e-12),
            PrescribedConstantCurrent(polarity),example_tether(),4.,bus_projected_area_m2=.03,
            tether_Cd=2.2,segments=4)
        out=propagate_orbit_decay([r,0,0],[0,v,0],ts,forces,max_step_s=30)
        residual=(out.mechanical_energy_J-out.mechanical_energy_J[0]
                  -out.work_lorentz_J-out.work_drag_J)
        assert max(abs(residual))<5e-4
        runs.append(out)
    assert runs[0].work_lorentz_J[-1]<0
    assert runs[1].work_lorentz_J[-1]>0
    assert runs[0].work_drag_J[-1]<0


def test_cnt_model_anchors_power_and_unit_conversion():
    cath=EmpiricalCNTCathode()
    assert cath.nominal_voltage_V(0)==pytest.approx(248)
    assert cath.nominal_voltage_V(500)==pytest.approx(411)
    assert cath.nominal_voltage_V(750)==pytest.approx(503.5)
    for age in [0,100,500,750]:
        V=cath.nominal_voltage_V(age)
        assert cath.emitted_current_A(V,age)==pytest.approx(.0025,rel=1e-10)
        assert cath.required_voltage_V(.0025,age)==pytest.approx(V,rel=1e-9)
    assert cath.estimated_cathode_power_W(.0025,248)==pytest.approx(.0025*248/.70)
    assert not cath.support_warnings(248,0)
    assert cath.support_warnings(248,750)
    assert cath.equivalent_age_increment_h(3600,.0025)==pytest.approx(1)
    assert cath.equivalent_age_increment_h(1800,.005)==pytest.approx(1)


def test_cnt_current_is_not_tether_profile():
    c=EmpiricalCNTCathode()
    assert not hasattr(c,'segment_currents_A')
    with pytest.raises(TypeError,match='current_profile'):
        RadialTetherForces(UniformMagneticField((0,0,3e-5)),ConstantAtmosphericDensity(0),
                           c,example_tether(),4.)
    with pytest.raises(ValueError):c.emitted_current_A(-4)


def test_script_short_scenario_runs_and_writes_evidence(tmp_path):
    from scripts.run_m4_decay_reference import run_scenario
    cfg=json.loads((__import__('pathlib').Path(__file__).resolve().parents[1]/
        'configs/scenarios/m4_500km_assumed_current.json').read_text())
    cfg['integration'].update(orbits=.02,samples=4,segments=2)
    cfg['runs']=['off','outward_always','inward_always']
    path=tmp_path/'scenario.json'; path.write_text(json.dumps(cfg))
    result=run_scenario(path,tmp_path/'output')
    assert len(result['cases'])==3
    assert result['metadata']['cathode_emission_capacity_A_at_assumed_bias']==pytest.approx(.0025)
    assert (tmp_path/'output'/'off.csv').exists()
    assert (tmp_path/'output'/'summary.json').exists()
    assert result['cases'][0]['lorentz_work_J']==0


def test_geometry_grid_generates_derived_properties(tmp_path):
    from scripts.sweep_m4_tether_geometry import run_geometry_sweep
    root=__import__('pathlib').Path(__file__).resolve().parents[1]
    rows=run_geometry_sweep(root/'configs/scenarios/m4_geometry_sweep.json',tmp_path/'out.csv')
    assert len(rows)==9
    assert (tmp_path/'out.csv').exists()
    assert rows[0]['tether_mass_kg'] < rows[1]['tether_mass_kg']
    assert rows[0]['geometric_perimeter_m']==pytest.approx(np.pi*0.0003)
    assert rows[0]['provenance_status']=='illustrative'
