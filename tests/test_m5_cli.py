"""M5 runner stores actual histories; static mode never pretends to be transient."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_m5_mechanical_reference import run

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/scenarios/m5_mechanical_small_regression.json'


def test_cli_transient_exports_all_state_and_diagnostics(tmp_path):
    summary=run(CONFIG,tmp_path/'output')
    output=tmp_path/'output'
    assert summary['mass_scale_used']==1.0
    assert summary['electrical_current_A']==0
    assert summary['mode']=='transient'
    assert summary['solver']['success']
    assert summary['solver']['current_A']==0
    assert (output/'summary.json').exists()
    assert (output/'history.csv').exists()
    assert (output/'segments.csv').exists()
    assert (output/'nodes.csv').exists()
    with np.load(output/'history.npz') as arr:
        assert arr['r_nodes_I_m'].shape==(51,6,3)
        assert arr['segment_tensions_N'].shape==(51,5)
        assert arr['actual_segment_lengths_m'].shape==(51,5)
        assert arr['inertia_about_com_I_kgm2'].shape==(51,3,3)
        assert arr['attachment_force_on_bus_I_N'].shape==(51,3)
        np.testing.assert_allclose(arr['node_masses_kg'].sum(),summary['mass_kg'])
        assert (arr['segment_tensions_N']>=0).all()
    assert json.loads((output/'summary.json').read_text())['milestone']=='M5'


def test_cli_quasi_static_is_separate_solver(tmp_path):
    config=json.loads(CONFIG.read_text())
    config['mode']='quasi_static'
    config['initial']['midpoint_bow_I_m']=[0,0,0]
    config['initial']['extension_fraction']=0
    config['environment']['gravity_I_m_s2']=[0,0,0]
    p=tmp_path/'inputs.json';p.write_text(json.dumps(config))
    summary=run(p,tmp_path/'output')
    assert summary['mode']=='quasi_static'
    assert summary['solver']['mass_scale_used']==1.0
    assert not (tmp_path/'output/history.npz').exists()
    assert (tmp_path/'output/summary.json').exists()
    assert np.max(np.abs(summary['tensions_N']))==0


def test_cli_nonzero_current_rejected_in_mechanics_only_reference(tmp_path):
    config=json.loads(CONFIG.read_text())
    config['environment']['current_A']=0.001
    p=tmp_path/'inputs.json';p.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='current_A exactly 0'):
        run(p,tmp_path/'output')
