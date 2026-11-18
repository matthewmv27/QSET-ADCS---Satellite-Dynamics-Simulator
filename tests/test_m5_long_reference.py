"""Large 50 m reference and vectorized segment forces, independent of QSET material data."""
import json
from pathlib import Path
import numpy as np
import pytest
from edtsim.tether import CableModel, RoundWire, HomogeneousTether, mass_properties_I, spring_pair_forces
from edtsim.tether.mechanics import _forces
from scripts.run_m5_mechanical_reference import _model, _initial, run
from scripts.run_m5_mesh_convergence import run_mesh_study

ROOT = Path(__file__).resolve().parents[1]
LARGE = ROOT/'configs/scenarios/m5_mechanical_example.json'


def test_50m_example_is_100_segments_with_physical_mass_and_rod_inertia():
    data=json.loads(LARGE.read_text())
    model=_model(data)
    assert model.segments == 100
    assert model.tether.length_m == 50.
    assert model.rest_length_m == pytest.approx(.5)
    assert len(model.node_masses_kg) == 101
    assert sum(model.node_masses_kg) == pytest.approx(model.total_mass_kg)
    assert model.total_mass_kg == pytest.approx(
        1000*np.pi*(.004/2)**2*50+.05)
    assert data['environment']['current_A'] == 0
    # Exact straight equilibrium mass properties including the point tip mass.
    node_s = np.linspace(0, model.tether.length_m, model.segments+1)
    positions = np.c_[node_s,np.zeros((101,2))]
    total,com,inertia=mass_properties_I(model,positions)
    assert total == pytest.approx(model.total_mass_kg)
    assert 0 < com[0] < 50
    assert inertia[1,1] == pytest.approx(inertia[2,2])
    assert inertia[0,0] == pytest.approx(0,abs=1e-9)
    assert np.linalg.eigvalsh(inertia).min() >= -1e-9


def test_vectorized_forces_match_individual_pair_springs():
    c = RoundWire(.004)
    tether = HomogeneousTether(c, 5., 1000., 1e5, 1e7)
    model = CableModel(tether, 10, viscous_axial_rigidity_Ns=.1)
    steps = [0.3,0.6,0.5,0.51,0.49,0.7,0.5,0.4,0.8,0.51]
    r = np.cumsum(np.array([[0.,0.,0.]]+[[step,0.,0.] for step in steps]),axis=0)
    v = np.random.default_rng(42).normal(0,.1,r.shape)
    forces,tensions,lengths = _forces(model,r,v)
    reference=np.zeros_like(r)
    expected_t=[]
    for j in range(model.segments):
        a,b,t=spring_pair_forces(r[j],r[j+1],v[j],v[j+1],
            model.rest_length_m,model.tether.axial_rigidity_N,
            model.viscous_axial_rigidity_Ns)
        reference[j]+=a;reference[j+1]+=b;expected_t.append(t)
    np.testing.assert_allclose(forces,reference,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(tensions,expected_t,atol=1e-12,rtol=1e-12)
    assert np.all(tensions>=0)
    np.testing.assert_allclose(forces.sum(axis=0),0,atol=1e-12)
    assert len(lengths)==model.segments


def test_50m_rest_configuration_has_exactly_zero_tension():
    model = _model(json.loads(LARGE.read_text()))
    positions = np.c_[np.linspace(0.,50.,101),np.zeros((101,2))]
    forces, tension, lengths = _forces(model, positions, np.zeros_like(positions))
    np.testing.assert_array_equal(tension, np.zeros(100))
    np.testing.assert_array_equal(forces, np.zeros((101,3)))
    np.testing.assert_allclose(lengths,0.5,atol=1e-14)


def test_50m_high_resolution_transient_exports_history(tmp_path):
    data=json.loads(LARGE.read_text())
    data['integration'].update(duration_s=.1,samples=6,max_step_s=.02)
    path=tmp_path/'50.json';path.write_text(json.dumps(data))
    summary=run(path,tmp_path/'output')
    assert summary['segments']==100
    assert summary['solver']['success']
    assert summary['mass_scale_used']==1.
    assert summary['electrical_current_A']==0.
    with np.load(tmp_path/'output/history.npz') as h:
        assert h['r_nodes_I_m'].shape==(6,101,3)
        assert h['segment_tensions_N'].shape==(6,100)
        assert h['node_masses_kg'].sum()==pytest.approx(summary['mass_kg'])
        assert np.isfinite(h['attachment_force_on_bus_I_N']).all()


def test_mesh_study_reports_fixed_physical_mass_and_errors(tmp_path):
    data=json.loads(LARGE.read_text())
    data['integration'].update(duration_s=.10,samples=5,max_step_s=.025)
    path=tmp_path/'mesh.json';path.write_text(json.dumps(data))
    summary=run_mesh_study(path,tmp_path/'out',(10,20,40))
    assert summary['reference_segments']==40
    assert (tmp_path/'out/mesh_comparison.csv').exists()
    assert (tmp_path/'out/mesh_summary.json').exists()
    rows=summary['rows']
    assert len(rows)==3
    assert all(r['mass_scale_used']==1 for r in rows)
    assert all(r['physical_mass_kg']==pytest.approx(rows[0]['physical_mass_kg']) for r in rows)
    assert rows[-1]['final_shape_rms_error_m_vs_finest']==pytest.approx(0.,abs=1e-12)
    assert rows[-1]['tip_time_rms_error_m_vs_finest']==pytest.approx(0.,abs=1e-12)


def test_mesh_study_rejects_duplicate_segment_counts(tmp_path):
    with pytest.raises(ValueError,match='unique'):
        run_mesh_study(LARGE,tmp_path,(10,10))
