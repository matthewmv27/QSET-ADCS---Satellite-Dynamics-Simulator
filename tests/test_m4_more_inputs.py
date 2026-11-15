import json
from pathlib import Path
import numpy as np
import pytest
from edtsim.environment.orbit_sweeps import make_orbit_cases
from scripts.run_m4_epoch_sweeps import _load_config


def test_epoch_input_has_both_solar_activity_cases():
    cfg=_load_config(Path(__file__).parents[1]/'configs/scenarios/m4_epoch_sweep_example.json')
    assert len(cfg['environment']['solar_activity_cases'])==2
    assert all(set(x)=={'label','f107','f107a','ap'} for x in cfg['environment']['solar_activity_cases'])


def test_epoch_sweep_inclination_ss0_j2_mismatch_known():
    cfg={'epochs_utc':['2026-11-15T12:00:00Z'], 'altitudes_m':[500000],
         'inclinations':['sso',97.5],
         'plane_sweeps':[{'type':'ltan_hours','values':[10.5]}]}
    a,b=make_orbit_cases(cfg,solar_ra_provider=lambda epoch:90)
    assert abs(a.nodal_rate_rad_s-a.target_sun_rate_rad_s) < 1e-15
    assert b.is_sso_design is False
    assert abs(b.nodal_rate_rad_s-b.target_sun_rate_rad_s)>1e-12
    assert np.isclose(a.ltan_hours,10.5)
