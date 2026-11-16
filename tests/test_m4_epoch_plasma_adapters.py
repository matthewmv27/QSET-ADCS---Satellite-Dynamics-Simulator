"""Deterministic tests; no network or external model installations required."""
from datetime import datetime, timezone
import sys
import types
from pathlib import Path
import numpy as np
import pytest
from edtsim.environment.epoch_frames import utc_datetime, epoch_at
from edtsim.environment.orbit_sweeps import (make_orbit_cases,raan_from_solar_time,solar_times_from_raan)
from edtsim.environment.earth_models import GeomagneticMainField, MSIS2AtmosphericDensity
from edtsim.electrodynamics.oml_collection import (ThinCollectorOMLEnvelope,E_CHARGE_C,E_MASS_KG)
from edtsim.tether import HomogeneousTether, RoundWire, RectangularTape
from edtsim.electrodynamics import EmpiricalCNTCathode
from scripts.run_m4_epoch_sweeps import run_sweeps


def tether(cross=None, bare=1):
    return HomogeneousTether(cross or RoundWire(.0002),100,2700,70e9,35e6,bare)


def test_utc_parsing_and_time_arithmetic():
    assert utc_datetime('2026-11-15T12:00:00Z').utcoffset().total_seconds()==0
    assert utc_datetime('2026-11-15T07:00:00-05:00').hour==12
    with pytest.raises(ValueError): utc_datetime('2026-11-15T12:00:00')
    assert epoch_at('2026-11-15T12:00:00Z',60).minute==1
    with pytest.raises(ValueError): epoch_at('2026-11-15T12:00:00Z',np.nan)


def test_raan_ltan_relation():
    sun=80.
    assert np.isclose(raan_from_solar_time(sun,ltan_hours=12),80)
    assert np.isclose(raan_from_solar_time(sun,ltdn_hours=0),80)
    assert np.isclose(raan_from_solar_time(sun,ltan_hours=6),350)
    assert np.allclose(solar_times_from_raan(sun,80),[12,0])
    with pytest.raises(ValueError): raan_from_solar_time(sun,ltan_hours=10,ltdn_hours=22)
    with pytest.raises(ValueError): raan_from_solar_time(sun,ltan_hours=24)


def test_epoch_sweep_families_not_cross_product():
    cfg={'epochs_utc':['2026-11-15T12:00:00Z','2027-01-01T12:00:00Z'],
         'altitudes_m':[500e3], 'inclinations':['sso',97.5],
         'plane_sweeps':[{'type':'ltan_hours','values':[10,12]},
                         {'type':'raan_deg','values':[100]},
                         {'type':'ltdn_hours','values':[22]}]}
    cases=make_orbit_cases(cfg,solar_ra_provider=lambda date:80 if date.year==2026 else 100)
    assert len(cases)==2*2*4
    assert cases[0].is_sso_design
    assert cases[0].raan_deg==50
    assert cases[4].is_sso_design is False
    assert any(c.plane_input_type=='raan_deg' and abs(c.ltan_hours-(12+(100-80)/15))<1e-9 for c in cases)
    assert len({c.raan_deg for c in cases})>1


def test_sso_bad_input_rejected():
    cfg={'epochs_utc':['2026-11-15T12:00:00Z'], 'altitudes_m':[500e3],
        'inclinations':['sso'], 'plane_sweeps':[{'type':'ltan_hours','values':[25]}]}
    with pytest.raises(ValueError): make_orbit_cases(cfg,solar_ra_provider=lambda _:80)
    cfg['plane_sweeps']=[{'type':'raan_deg','values':[0], 'ltan_hours':[10]}]
    with pytest.raises(ValueError): make_orbit_cases(cfg,solar_ra_provider=lambda _:80)


def test_oml_high_bias_limit_and_sign():
    obj=ThinCollectorOMLEnvelope(tether(),1e11,1500,2)
    j=(obj.tether.cross_section.perimeter_m/np.pi)*E_CHARGE_C*1e11*np.sqrt(2*E_CHARGE_C*2/E_MASS_KG)
    assert np.isclose(obj.collected_current_density_A_m,j)
    s=np.array([0,25,50,75,100.])
    profile=obj.segment_currents_A(0,s)
    assert np.isclose(profile[0],j*100)
    assert np.isclose(profile[-1],0)
    assert np.all(np.diff(profile)<=0)
    assert obj.geometry_ratio>0


def test_oml_bare_tip_region_and_cathode_requirement():
    obj=ThinCollectorOMLEnvelope(tether(bare=.4),1e10,1000,1)
    assert np.isclose(obj.segment_currents_A(1,np.array([0.,30.,60.]))[0],obj.bus_current_upper_bound_A)
    assert np.isclose(obj.segment_currents_A(0,np.array([100.]))[0],0)
    data=obj.cathode_requirement(EmpiricalCNTCathode(),max_bias_V=500)
    assert data['collection_upper_bound_A']>0
    assert data['cathode_voltage_required_V']>0
    assert data['model'].startswith('decoupled')


def test_oml_rejects_invalid_regime_and_segment_positions():
    with pytest.raises(ValueError): ThinCollectorOMLEnvelope(tether(),1e11,100000,0.01)
    obj=ThinCollectorOMLEnvelope(tether(),1e11,1000,1)
    with pytest.raises(ValueError): obj.segment_currents_A(0,[-1,5])
    with pytest.raises(ValueError): obj.segment_currents_A(0,[101])
    with pytest.raises(ValueError): ThinCollectorOMLEnvelope(tether(),-1,1000,1)


class FakeFrame:
    def __init__(self,*args): pass
    def geodetic(self,*args): return (0.,0.,500.)
    def vector_ecef_to_inertial(self,t,B): return np.asarray(B)


def test_igrf_adapter_coordinate_conversion(monkeypatch):
    import edtsim.environment.earth_models as em
    monkeypatch.setattr(em,'EarthFrame',FakeFrame)
    monkeypatch.setitem(sys.modules,'ppigrf',types.SimpleNamespace(igrf=lambda lon,lat,alt,date:(3,2,-4)))
    provider=GeomagneticMainField('2026-11-15T12:00:00Z','igrf14')
    field=provider.sample(0,[7000000,0,0])
    # At equator lon=0: north points +z, east +y, down points -x.
    assert np.allclose(field,np.array([-4,3,2])*1e-9)


def test_wmm_adapter_units_and_validity(monkeypatch):
    import edtsim.environment.earth_models as em
    monkeypatch.setattr(em,'EarthFrame',FakeFrame)
    class MockWMM:
        def setup_time(self,*args): pass
        def setup_env(self,*args): pass
        def get_all(self): return {'x':np.array([20000.]),'y':np.array([3000.]),'z':np.array([40000.])}
    monkeypatch.setitem(sys.modules,'wmm',types.SimpleNamespace(wmm_calc=MockWMM))
    field=GeomagneticMainField('2026-01-01T00:00:00Z','wmm2025').sample(0,[7000000,0,0])
    assert np.allclose(field,np.array([-40000,3000,20000])*1e-9)
    with pytest.raises(ValueError): GeomagneticMainField('2032-01-01T00:00:00Z','wmm2025').sample(0,[7000000,0,0])


def test_msis_adapter_explicit_activity(monkeypatch):
    import edtsim.environment.earth_models as em
    monkeypatch.setattr(em,'EarthFrame',FakeFrame)
    def fake_calc(*args,**kw):
        assert kw['version']==2.0
        assert kw['f107s'].tolist()==[120]
        assert np.asarray(kw['aps']).shape==(1,7)
        return np.array([[2e-12,0,0]])
    monkeypatch.setitem(sys.modules,'pymsis',types.SimpleNamespace(calculate=fake_calc,
                                             Variable=types.SimpleNamespace(MASS_DENSITY=0)))
    density=MSIS2AtmosphericDensity('2026-11-15T00:00:00Z',120,120,7)
    assert density.sample(0,[7000000,0,0])==2e-12
    with pytest.raises(ValueError): MSIS2AtmosphericDensity('2026-11-15T00:00:00Z',0,120,7)


def test_epoch_runner_with_injected_synthetic_providers(tmp_path):
    class Field:
        def sample(self,t,r): return np.array([0.,0.,-3e-5])
    class Density:
        def sample(self,t,r): return 2e-12
    orig=Path(__file__).parents[1]/'configs/scenarios/m4_epoch_sweep_example.json'
    cfg=__import__('json').loads(orig.read_text())
    cfg['orbit_sweep']['epochs_utc']=['2026-11-15T12:00:00Z']
    cfg['orbit_sweep']['inclinations']=['sso']
    cfg['orbit_sweep']['plane_sweeps']=[{'type':'raan_deg','values':[0,90]}]
    cfg['sampling']={'steps_per_orbit':4,'segment_count':1,'orbits':.25}
    cfgpath=tmp_path/'input.json';cfgpath.write_text(__import__('json').dumps(cfg))
    results=run_sweeps(cfgpath,tmp_path/'out',solar_ra_provider=lambda _:80,
                      provider_factory=lambda c,e:(Field(),Density()))
    assert len(results)==4
    assert {r['activity_case'] for r in results}=={'low_activity','higher_activity'}
    assert (tmp_path/'out/summary.csv').exists()
    assert (tmp_path/'out/summary.json').exists()
    assert all(result['delta_osculating_a_m'] is None for result in results)
    assert all(np.isfinite(result['lorentz_work_J']) for result in results)


def test_oml_shooting_sign_residual_and_conservation():
    from edtsim.electrodynamics.oml_collection import PrescribedBusPotentialOML
    config=PrescribedBusPotentialOML(tether(bare=.5),1e10,1000,2,.05)
    sol=config.solve(nodes=101)
    assert sol.bus_emission_current_required_A>0
    assert abs(sol.tip_current_residual_A)<max(1e-9,1e-5*sol.bus_emission_current_required_A)
    assert np.all(np.diff(sol.conventional_current_A)<1e-8)
    assert np.all(sol.conventional_current_A>=0)
    assert sol.ohmic_loss_W>=0
    assert np.isclose(sol.conventional_current_A[0],sol.bus_emission_current_required_A)
    # Conservation: collected electrons integrated along the bare part
    # equal current returning to cathode, approximately (mesh quadrature).
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert np.isclose(
    integrate(
        sol.collected_electron_current_per_m_A_m,
        sol.s_m
    ),
    sol.bus_emission_current_required_A,
    rtol=.03
)


def test_oml_smooth_through_zero_bias_and_high_bias_limit():
    from edtsim.electrodynamics.oml_collection import PrescribedBusPotentialOML
    obj=PrescribedBusPotentialOML(tether(),1e11,1000,1,.01)
    j0=obj.collected_density_A_m(50,0)
    assert np.isclose(j0,obj.electron_thermal_flux_A_m)
    assert obj.collected_density_A_m(50,-5)<j0
    assert obj.collected_density_A_m(50,10)>j0
    ratio=obj.collected_density_A_m(50,100)/obj.collected_density_A_m(50,25)
    assert 1.8 < ratio < 2.2
