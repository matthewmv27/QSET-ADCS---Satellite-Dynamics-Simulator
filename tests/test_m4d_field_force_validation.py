"""M4d: independent vector identities and official WMM2025 references."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from importlib.util import find_spec

import numpy as np
import pytest

from edtsim.environment.m4d_audit import audit_radial_tether, sample_orbit_audit
from edtsim.environment.fields import UniformMagneticField, ned_to_ecef_matrix
from edtsim.environment.earth_models import GeomagneticMainField
from edtsim.environment.orbit import CircularReferenceOrbit, sso_inclination_deg
from edtsim.environment.flows import corotating_flow_I
from edtsim.core.constants import R_EARTH_EQ

FIXTURE = Path(__file__).parent/'fixtures'/'wmm2025_noaa_reference.json'


def test_reference_source_components_have_correct_orientation():
    rows=json.loads(FIXTURE.read_text())['cases']
    assert len(rows)==6
    R=ned_to_ecef_matrix(0,np.deg2rad(120))
    x=np.array([rows[1][k] for k in ('north_nT','east_nT','down_nT')])
    assert x[0]>0 and x[1]<0 and x[2]<0
    np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(R@x),np.linalg.norm(x),rtol=1e-14)


def test_igrf_adapter_unpacks_east_north_up(monkeypatch):
    monkeypatch.setitem(sys.modules,'ppigrf',SimpleNamespace(igrf=lambda lon,lat,h,date:
                       (np.array([123.]),np.array([456.]),np.array([-789.]))))
    model=GeomagneticMainField('2025-01-01T00:00:00Z','igrf14')
    B=model._field_ned_nT(datetime(2025,1,1,tzinfo=timezone.utc),12,34,500)
    np.testing.assert_array_equal(B,[456,123,789])


def test_wmm_adapter_unpacks_north_east_down(monkeypatch):
    class Dummy:
        def setup_time(self,*args):pass
        def setup_env(self,*args):pass
        def get_all(self):return {'x':[456.], 'y':[123.], 'z':[789.]}
    monkeypatch.setitem(sys.modules,'wmm',SimpleNamespace(wmm_calc=Dummy))
    model=GeomagneticMainField('2025-01-01T00:00:00Z','wmm2025')
    B=model._field_ned_nT(datetime(2025,1,1,tzinfo=timezone.utc),12,34,500)
    np.testing.assert_array_equal(B,[456,123,789])


def test_provider_rotation_at_nontrivial_geodetic_location():
    provider=GeomagneticMainField('2025-01-01T00:00:00Z','igrf14')
    lat,lon=47.,-76.
    provider.frame=SimpleNamespace(geodetic=lambda t,r:(lat,lon,500.),
                   vector_ecef_to_inertial=lambda t,B:np.asarray(B))
    provider._field_ned_nT=lambda date,lat,lon,alt:np.array([12000.,-7000.,30000.])
    out=provider.sample(0,np.array([6900000.,0.,0.]))
    # Independent analytic basis vectors (not the ned_to_ecef_matrix function).
    la,lo=np.deg2rad([lat,lon])
    north=np.array([-np.sin(la)*np.cos(lo),-np.sin(la)*np.sin(lo),np.cos(la)])
    east=np.array([-np.sin(lo),np.cos(lo),0.])
    down=np.array([-np.cos(la)*np.cos(lo),-np.cos(la)*np.sin(lo),-np.sin(la)])
    np.testing.assert_allclose(out,(12000*north-7000*east+30000*down)*1e-9,atol=2e-20)


@pytest.mark.skipif(not (find_spec('astropy') and find_spec('wmm')),
                    reason='Install optional earth-models extras for NOAA reference benchmarking')
@pytest.mark.parametrize('case',json.loads(FIXTURE.read_text())['cases'])
def test_wmm2025_against_published_noaa_ned_components(case):
    """Compare all signed components to NOAA WMM2025 2025.0 reference.

    5 nT numerical envelope deliberately wider than NOAA's 0.1-nT
    publication rounding; this is an adapter regression tolerance, NOT an
    uncertainty model for geomagnetic disturbances.
    """
    model=GeomagneticMainField('2025-01-01T00:00:00Z','wmm2025')
    observed=model._field_ned_nT(datetime(2025,1,1,tzinfo=timezone.utc),
             case['latitude_deg'],case['longitude_deg'],case['altitude_km'])
    expected=np.array([case[k] for k in ('north_nT','east_nT','down_nT')])
    np.testing.assert_allclose(observed,expected,rtol=0,atol=5.0)


def test_outward_current_brakes_for_positive_orbit_normal_field():
    r=np.array([R_EARTH_EQ+500000.,0.,0.]);v=np.array([0.,7600.,0.])
    h=np.array([0.,0.,1.])
    B=UniformMagneticField((0,0,30e-6))
    a=audit_radial_tether(0,r,v,B,np.full(10,0.0025),100)
    assert a.along_track_force_N==pytest.approx(-0.0025*100*30e-6)
    assert a.power_com_W <0
    assert a.power_distributed_W == pytest.approx(-a.motional_emf_power_W,abs=1e-13)
    assert abs(a.identity_residual_W)<1e-12
    b=audit_radial_tether(0,r,v,B,np.full(10,-0.0025),100)
    assert b.along_track_force_N==pytest.approx(-a.along_track_force_N)
    assert b.power_com_W==pytest.approx(-a.power_com_W)
    assert np.dot(np.cross(r,v),h)>0


def test_plasma_moving_changes_emf_vs_mechanical_work():
    r=np.array([R_EARTH_EQ+500000.,2000.,3000.]);v=np.array([1.,7600.,20.])
    B=UniformMagneticField((4e-6,21e-6,35e-6))
    a=audit_radial_tether(10,r,v,B,[0.003,0.002,0.001],150,
                          plasma_velocity_provider=lambda t,r:corotating_flow_I(r))
    assert abs(a.identity_residual_W)<1e-12
    assert abs(a.plasma_motion_power_W)>1e-9
    assert a.power_distributed_W+a.motional_emf_power_W==pytest.approx(a.plasma_motion_power_W,abs=1e-12)
    assert a.power_distributed_W != pytest.approx(-a.motional_emf_power_W,abs=1e-9)


def test_variable_current_profile_signed_force_identity():
    r=np.array([R_EARTH_EQ+500000.,4000.,0.]);v=np.array([2.,7300.,300.])
    B=UniformMagneticField((11e-6,-5e-6,25e-6))
    a=audit_radial_tether(0,r,v,B,[0.003,0.001,0.0],90)
    h=np.cross(r/np.linalg.norm(r),v/np.linalg.norm(v))
    expected=-np.dot(B.B_I_T,h)*30*(.003+.001)
    assert a.along_track_force_N==pytest.approx(expected,abs=1e-15)


def test_one_orbit_report_preserves_sign_and_time_order():
    orbit=CircularReferenceOrbit(500000,sso_inclination_deg(500000),0,0)
    r,v=orbit.state(0)
    field=UniformMagneticField(tuple(30e-6*np.cross(r,v)/np.linalg.norm(np.cross(r,v))))
    times=np.linspace(0,orbit.period_s,9)
    rows=sample_orbit_audit(orbit,field,times,0.0025,100,segments=4)
    assert len(rows)==9
    assert all(row['along_track_force_N'] <0 for row in rows)
    assert all(abs(row['power_identity_residual_W'])<1e-10 for row in rows)


@pytest.mark.parametrize('bad', [[],[np.nan],[0,np.inf]])
def test_invalid_current_arrays_rejected(bad):
    with pytest.raises(ValueError):
        audit_radial_tether(0,[7000e3,0,0],[0,7500,0],UniformMagneticField((0,0,3e-5)),bad,100)
