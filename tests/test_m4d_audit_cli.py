import json
from pathlib import Path
import pytest
from scripts.run_m4d_field_audit import run


def test_m4d_cli_summary(tmp_path):
    config=Path(__file__).resolve().parents[1]/'configs/scenarios/m4d_field_audit.json'
    summary=run(config,tmp_path/'out')
    assert summary['model']=='centered_dipole'
    assert summary['samples']==81
    assert summary['max_abs_identity_residual_W'] < 1e-10
    assert (tmp_path/'out'/'samples.csv').exists()
    loaded=json.loads((tmp_path/'out'/'summary.json').read_text())
    assert loaded['lorentz_work_at_bus_orbit_velocity_J']==pytest.approx(summary['lorentz_work_at_bus_orbit_velocity_J'])


def test_m4d_rejects_bad_model(tmp_path):
    config=Path(__file__).resolve().parents[1]/'configs/scenarios/m4d_field_audit.json'
    with pytest.raises(ValueError,match='field_model'):
        run(config,tmp_path/'bad',model_override='nonsense')
