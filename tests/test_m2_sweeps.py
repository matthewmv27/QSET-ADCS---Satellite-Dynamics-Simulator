"""Independent checks for M2 configuration parsing, sweeps and outputs."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from edtsim.inertia.scenario import load_scenario
from edtsim.inertia.sweeps import run_sweeps, save_results

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "scenarios" / "m2_illustrative.json"


def _scenario(tmp_path, modify=None):
    raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    if modify:
        modify(raw)
    p = tmp_path / "scenario.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    return p


def test_example_deployment_and_design_grid_case_counts():
    s = load_scenario(EXAMPLE)
    records = run_sweeps(s)
    assert len(records) == 5 + 2 * 2 * 3 + 1
    assert len(set(r["case_id"] for r in records)) == len(records)
    assert [r["kind"] for r in records[:5]] == ["deployment"] * 5
    assert records[-1]["kind"] == "bent_cases"


def test_deployment_sweep_conserves_mass_and_tether_partition():
    scenario = load_scenario(EXAMPLE)
    rows = [r for r in run_sweeps(scenario) if r["sweep_id"] == "deployment_fraction"]
    assert all(r["mass_kg"] == pytest.approx(4.75) for r in rows)
    for r in rows:
        f = r["deployment_fraction"]
        masses = r["component_masses_kg"]
        assert masses.get("wound_tether", 0) == pytest.approx(.5 * (1 - f))
        assert masses.get("deployed_tether", 0) == pytest.approx(.5 * f)
        Ibus = np.array(r["inertia_about_bus_com_B_kgm2"])
        Icom = np.array(r["inertia_about_system_com_B_kgm2"])
        x = np.array(r["com_B_m"])
        np.testing.assert_allclose(Ibus, Icom + r["mass_kg"] *
                                   ((x @ x) * np.eye(3) - np.outer(x, x)), atol=1e-10)
        assert np.linalg.eigvalsh(Icom).min() >= -1e-10


def test_design_length_changes_total_material_mass_not_density():
    s = load_scenario(EXAMPLE)
    rows = [r for r in run_sweeps(s) if r["sweep_id"] == "design_comparison"]
    for r in rows:
        length = r["input_model"]["tether"]["length_m"]
        tip = r["input_model"]["end_mass"]["mass_kg"]
        assert r["mass_kg"] == pytest.approx(4 + .1 + .05 * length + tip)
        assert r["component_masses_kg"].get("wound_tether", 0) == pytest.approx(.05 * length * (1-r["deployment_fraction"]))
        assert r["component_masses_kg"].get("deployed_tether", 0) == pytest.approx(.05 * length * r["deployment_fraction"])
    # The chosen hardware design must conserve its own mass over all fractions.
    unique = set((r["input_model"]["tether"]["length_m"],r["input_model"]["end_mass"]["mass_kg"]) for r in rows)
    for length, tip in unique:
        values = [r["mass_kg"] for r in rows if r["input_model"]["tether"]["length_m"] == length and r["input_model"]["end_mass"]["mass_kg"] == tip]
        assert len(values) == 3
        assert max(values) - min(values) < 1e-12


def test_bent_node_history_available_in_json():
    s = load_scenario(EXAMPLE)
    r = run_sweeps(s)[-1]
    assert r["geometry_id"] == "bent_A"
    assert len(r["geometry"]["nodes_B_m"]) == 3
    assert r["component_masses_kg"]["deployed_tether_segment_0000"] == pytest.approx(.25)
    assert r["component_masses_kg"]["deployed_tether_segment_0001"] == pytest.approx(.25)
    assert r["mass_kg"] == pytest.approx(4.75)


def test_deterministic_csv_json_and_complete_columns(tmp_path):
    scenario = load_scenario(EXAMPLE)
    rows = run_sweeps(scenario)
    files_a = save_results(scenario, rows, tmp_path / "a")
    files_b = save_results(scenario, rows, tmp_path / "b")
    for a, b in zip(files_a, files_b):
        assert a.read_bytes() == b.read_bytes()
    data = json.loads(files_a[0].read_text())
    assert data["declared_provenance"]["status"] == "illustrative"
    assert data["reference_frame"].startswith("body axes B")
    assert len(data["source_configuration_sha256"]) == 64
    assert len(data["cases"]) == 18
    with files_a[1].open(newline="") as f:
        records = list(csv.DictReader(f))
    assert len(records) == 18
    for key in ("I_bus_xy_kgm2", "I_system_com_yx_kgm2", "com_z_B_m", "principal_3_kgm2", "model_sha256", "component_masses_kg_json"):
        assert key in records[0]
    assert records[0]["declared_input_status"] == "illustrative"
    assert records[-1]["declared_sweep_status"] == "illustrative"


def test_command_line_smoke(tmp_path):
    out = tmp_path / "results"
    cmd = [sys.executable, str(ROOT / "scripts" / "sweep_mass_properties.py"),
           str(EXAMPLE), "-o", str(out)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    assert p.returncode == 0, p.stderr
    assert "18 cases" in p.stdout
    assert (out / "mass_properties.json").exists()
    assert (out / "mass_properties.csv").exists()


def test_invalid_fractions_and_unknown_design_axes_rejected(tmp_path):
    def bad_fraction(raw):
        raw["sweeps"][0]["fractions"] = [0, 1.1]
    with pytest.raises(ValueError, match="fractions"):
        run_sweeps(load_scenario(_scenario(tmp_path, bad_fraction)))

    def bad_axis(raw):
        raw["sweeps"][1]["axes"] = {"tether.__dict__": [1]}
    with pytest.raises(ValueError, match="Unknown design axis"):
        run_sweeps(load_scenario(_scenario(tmp_path, bad_axis)))


def test_bad_scenario_metadata_and_malformed_inputs_rejected(tmp_path):
    def bad_status(raw):
        raw["provenance"]["status"] = "mission_verified_by_chatgpt"
    with pytest.raises(ValueError, match="status"):
        load_scenario(_scenario(tmp_path, bad_status))
    def bad_mass(raw):
        raw["model"]["bus"]["mass_kg"] = -1
    with pytest.raises(ValueError, match="bus.mass_kg"):
        load_scenario(_scenario(tmp_path, bad_mass))
    def invalid_dim(raw):
        raw["model"]["bus"]["dimensions_m"] = [0.1, 0.1, "0.3"]
    with pytest.raises(ValueError, match="dimensions_m"):
        load_scenario(_scenario(tmp_path, invalid_dim))
    def invalid_winding(raw):
        raw["model"]["wound_geometry"]["r_outer_m"] = 0.001
    with pytest.raises(ValueError):
        load_scenario(_scenario(tmp_path, invalid_winding))


def test_duplicate_json_keys_and_nonfinite_json_rejected(tmp_path):
    p = tmp_path / "duplicate.json"
    p.write_text('{"id": "one", "id": "two"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_scenario(p)
    p.write_text('{"value": NaN}')
    with pytest.raises(ValueError, match="Nonstandard JSON"):
        load_scenario(p)


def test_case_limit_enforced_before_grid_evaluation():
    s = load_scenario(EXAMPLE)
    with pytest.raises(ValueError, match="limit"):
        run_sweeps(s, max_cases=9)


def test_bent_attach_and_reference_length_validation(tmp_path):
    def invalid_geom(raw):
        raw["sweeps"][2]["cases"][0]["nodes_B_m"][0] = [1, 0, 0]
    with pytest.raises(ValueError, match="first node"):
        run_sweeps(load_scenario(_scenario(tmp_path, invalid_geom)))
    def invalid_length(raw):
        raw["sweeps"][2]["cases"][0]["reference_lengths_m"] = [4,4]
    with pytest.raises(ValueError, match="Rest lengths"):
        run_sweeps(load_scenario(_scenario(tmp_path, invalid_length)))


def test_hypothetical_design_sweep_requires_separate_provenance(tmp_path):
    def without_source(raw):
        raw["sweeps"][1].pop("provenance")
    with pytest.raises(ValueError, match="must declare provenance"):
        run_sweeps(load_scenario(_scenario(tmp_path, without_source)))
