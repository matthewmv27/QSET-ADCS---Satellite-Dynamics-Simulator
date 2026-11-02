"""Deterministic, bounded static M2 configuration and design sweeps.

Only edtsim.inertia.mission computes the physics. A sweep provides inputs,
checks conservation, then serializes the results in explicit body axes.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from edtsim.inertia.scenario import (
    DESIGN_AXES, Scenario, build_mission_model, finite_number, validate_model_config, validate_provenance,
    _check_keys, _vector,
)

MAX_CASES_DEFAULT = 2000


def _fractions(raw: Any) -> List[float]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("fractions must be a nonempty JSON array")
    values = [finite_number(f, "fraction", nonnegative=True) for f in raw]
    if any(f > 1.0 for f in values):
        raise ValueError("fractions must be between 0 and 1")
    if len(set(values)) != len(values):
        raise ValueError("fractions must not contain duplicates")
    return values


def _direction(raw: Any) -> List[float]:
    v = _vector(raw, "direction_B")
    if np.linalg.norm(v) < 1e-12:
        raise ValueError("direction_B must be nonzero")
    return v.tolist()


def _make_overridden_model(model: dict, overrides: dict) -> dict:
    out = copy.deepcopy(model)
    for path, value in overrides.items():
        if path not in DESIGN_AXES:
            raise ValueError("Unsupported design axis: {}".format(path))
        section, field = path.split(".", 1)
        out[section][field] = finite_number(value, path)
    return validate_model_config(out)


def _case(sweep_id: str, kind: str, serial: int, input_model: dict, f: float,
          result, overrides: dict, geometry_id: str, geometry: dict, sweep_provenance: dict) -> dict:
    # Inertia about bus COM and system COM must not be confused.
    eigenvalues = np.linalg.eigvalsh(result.inertia_com_B)
    return {
        "case_id": "{}__{:04d}".format(sweep_id, serial),
        "declared_sweep_provenance": copy.deepcopy(sweep_provenance),
        "sweep_id": sweep_id, "kind": kind, "geometry_id": geometry_id,
        "deployment_fraction": f,
        "design_overrides": copy.deepcopy(overrides),
        "input_model": copy.deepcopy(input_model),
        "geometry": copy.deepcopy(geometry),
        "mass_kg": float(result.mass_kg),
        "component_masses_kg": result.masses_by_name(),
        "com_B_m": result.com_B.tolist(),
        "inertia_about_bus_com_B_kgm2": result.inertia_bus_B.tolist(),
        "inertia_about_system_com_B_kgm2": result.inertia_com_B.tolist(),
        "principal_moments_system_com_kgm2": eigenvalues.tolist(),
    }


def run_sweeps(scenario: Scenario, max_cases: int = MAX_CASES_DEFAULT) -> List[dict]:
    """Evaluate all requested sweeps in order. Raise on invalid input or failed physics.

    A complete sweep is prepared in memory before any file is written, so an
    invalid case does not result in misleading partial result files.
    """
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1:
        raise ValueError("max_cases must be a positive integer")
    records = []
    for sweep in scenario.sweeps:
        kind, sid = sweep.get("kind"), sweep["id"]
        if kind in ("design_grid", "bent_cases") and "provenance" not in sweep:
            raise ValueError("{} must declare provenance for proposed design/geometry variations".format(sid))
        source = validate_provenance(sweep.get("provenance", scenario.provenance), sid + ".provenance")
        serial = 0
        if kind == "deployment":
            _check_keys(sweep, {"id", "kind", "fractions"}, {"direction_B", "provenance"}, sid)
            fractions = _fractions(sweep["fractions"])
            direction = _direction(sweep.get("direction_B", [0, 0, -1]))
            mission = build_mission_model(scenario.model)
            for fraction in fractions:
                r = mission.straight(fraction, direction)
                records.append(_case(sid, kind, serial, scenario.model, fraction, r, {},
                                     "straight", {"direction_B": direction}, source))
                serial += 1
        elif kind == "design_grid":
            _check_keys(sweep, {"id", "kind", "axes", "fractions"}, {"direction_B", "provenance"}, sid)
            axes = sweep["axes"]
            if not isinstance(axes, dict) or not axes:
                raise ValueError("{} axes must be a nonempty object".format(sid))
            if any(path not in DESIGN_AXES for path in axes):
                raise ValueError("Unknown design axis: {}".format(sorted(set(axes)-DESIGN_AXES)))
            axis_names = sorted(axes)
            axis_values = []
            for path in axis_names:
                values = axes[path]
                if not isinstance(values, list) or not values:
                    raise ValueError("{}: axis {} needs nonempty values".format(sid, path))
                casted = [finite_number(v, path) for v in values]
                if len(set(casted)) != len(casted):
                    raise ValueError("{}: duplicate values in {}".format(sid, path))
                axis_values.append(casted)
            fractions = _fractions(sweep["fractions"])
            direction = _direction(sweep.get("direction_B", [0, 0, -1]))
            planned = math.prod(len(v) for v in axis_values) * len(fractions)
            if planned + len(records) > max_cases:
                raise ValueError("{} requests {} cases; limit is {}".format(sid, planned, max_cases))
            for combo in itertools.product(*axis_values):
                overrides = dict(zip(axis_names, combo))
                config = _make_overridden_model(scenario.model, overrides)
                mission = build_mission_model(config)
                for fraction in fractions:
                    r = mission.straight(fraction, direction)
                    records.append(_case(sid, kind, serial, config, fraction, r, overrides,
                                         "straight", {"direction_B": direction}, source))
                    serial += 1
        elif kind == "bent_cases":
            _check_keys(sweep, {"id", "kind", "cases"}, {"provenance"}, sid)
            cases = sweep["cases"]
            if not isinstance(cases, list) or not cases:
                raise ValueError("{} cases must be a nonempty array".format(sid))
            mission = build_mission_model(scenario.model)
            seen = set()
            for c in cases:
                _check_keys(c, {"id", "fraction", "nodes_B_m"}, {"reference_lengths_m"}, sid+".case")
                gid = c["id"]
                if not isinstance(gid, str) or not gid.strip() or gid in seen:
                    raise ValueError("Bent geometry ids must be unique, nonempty strings")
                seen.add(gid)
                fraction = finite_number(c["fraction"], "bent fraction", positive=True)
                if fraction > 1.0:
                    raise ValueError("Bent fraction must not exceed one")
                nodes = c["nodes_B_m"]
                if not isinstance(nodes, list) or len(nodes) < 2:
                    raise ValueError("nodes_B_m must be a list of at least two positions")
                pts = [_vector(p, "nodes_B_m").tolist() for p in nodes]
                lengths = c.get("reference_lengths_m")
                if lengths is not None:
                    if not isinstance(lengths, list):
                        raise ValueError("reference_lengths_m must be a list")
                    lengths = [finite_number(v, "reference_lengths_m", positive=True) for v in lengths]
                r = mission.bent(fraction, pts, lengths)
                geometry = {"nodes_B_m": pts, "reference_lengths_m": lengths}
                records.append(_case(sid, kind, serial, scenario.model, fraction, r, {}, gid, geometry, source))
                serial += 1
        else:
            raise ValueError("Unknown sweep kind for {}: {}".format(sid, kind))
        if len(records) > max_cases:
            raise ValueError("Too many cases (limit {})".format(max_cases))
    return records


def _sha_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _flatten_case(scenario: Scenario, case: dict) -> dict:
    b, t, e = (case["input_model"][x] for x in ("bus", "tether", "end_mass"))
    row = {
        "scenario_id": scenario.identifier, "scenario_revision": scenario.revision,
        "declared_input_status": scenario.provenance["status"],
        "declared_sweep_status": case["declared_sweep_provenance"]["status"],
        "case_id": case["case_id"], "sweep_id": case["sweep_id"], "kind": case["kind"],
        "geometry_id": case["geometry_id"], "deployment_fraction": case["deployment_fraction"],
        "bus_mass_kg": b["mass_kg"], "tether_length_m": t["length_m"],
        "tether_linear_density_kg_per_m": t["linear_density_kg_per_m"],
        "tether_mass_kg": t["length_m"] * t["linear_density_kg_per_m"],
        "end_mass_kg": e["mass_kg"], "total_mass_kg": case["mass_kg"],
        "model_sha256": _sha_json(case["input_model"]),
        "design_overrides_json": json.dumps(case["design_overrides"], sort_keys=True),
        "component_masses_kg_json": json.dumps(case["component_masses_kg"], sort_keys=True),
    }
    for i, axis in enumerate("xyz"):
        row["com_{}_B_m".format(axis)] = case["com_B_m"][i]
    for label, matrix in (("I_bus", case["inertia_about_bus_com_B_kgm2"]),
                          ("I_system_com", case["inertia_about_system_com_B_kgm2"])):
        for i in range(3):
            for j in range(3):
                row["{}_{}{}_kgm2".format(label, "xyz"[i], "xyz"[j])] = matrix[i][j]
    for i, moment in enumerate(case["principal_moments_system_com_kgm2"], 1):
        row["principal_{}_kgm2".format(i)] = moment
    return row


def save_results(scenario: Scenario, cases: List[dict], output_dir: Any) -> tuple:
    """Write full JSON (all geometry and source data) and flat CSV for analysis."""
    if not cases:
        raise ValueError("Cannot save an empty sweep")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    jsonpath, csvpath = path / "mass_properties.json", path / "mass_properties.csv"
    manifest = {
        "schema_version": 1, "scenario_id": scenario.identifier,
        "scenario_description": scenario.description, "scenario_revision": scenario.revision,
        "declared_provenance": scenario.provenance,
        "parameter_provenance": scenario.parameter_provenance,
        "source_file": str(Path(scenario.source).name),
        "source_configuration_sha256": _sha_json({
            "schema_version": 1, "id": scenario.identifier, "description": scenario.description,
            "revision": scenario.revision, "provenance": scenario.provenance,
            "parameter_provenance": scenario.parameter_provenance,
            "model": scenario.model, "sweeps": list(scenario.sweeps),
        }),
        "reference_frame": "body axes B, origin at rigid bus COM",
        "physical_scope": "static mass-properties snapshots; no deployment dynamics or drag",
        "limitations": [
            "Fixed annular shape with decreasing mass approximates wound tether",
            "At zero deployment the stowed tip location may differ discontinuously from paid-out position",
            "Cable segments are slender, zero-radius rods; end mass is a point mass",
            "Provenance status is declared by input author and not independently verified",
            "Results are not flight-qualified without supporting hardware data and validation",
        ],
        "cases": cases,
    }
    # Deterministic outputs: no timestamps, stable column order and JSON fields.
    jsonpath.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = [_flatten_case(scenario, case) for case in cases]
    if not rows:
        raise ValueError("Cannot save an empty sweep")
    with csvpath.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return jsonpath, csvpath


def plot_results(cases: List[dict], output_dir: Any) -> List[Path]:
    """Optional plots of diagonal assembly inertia vs deployment fraction.

    Only meaningful for deployment and design-grid sweeps; bent cases are
    written to CSV/JSON instead of assigning an artificial x-coordinate.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plots require matplotlib: python -m pip install -e '.[viz]'") from exc
    output_dir = Path(output_dir)
    plots = []
    grouped = defaultdict(list)
    for c in cases:
        if c["kind"] in ("deployment", "design_grid"):
            grouped[(c["sweep_id"], json.dumps(c["design_overrides"], sort_keys=True))].append(c)
    for sweep_id in sorted(set(k[0] for k in grouped)):
        series = [(json.loads(overrides), sorted(grouped[(sid, overrides)], key=lambda x: x["deployment_fraction"]))
                  for sid, overrides in sorted(grouped) if sid == sweep_id]
        if len(series) > 16:
            raise ValueError("Plot has {} design combinations; limit plot series to 16 (CSV/JSON still available)".format(len(series)))
        for i, axis in enumerate("xyz"):
            fig, ax = plt.subplots(figsize=(9, 5))
            for overrides, values in series:
                label = ", ".join("{}={}".format(k, v) for k, v in sorted(overrides.items())) or "baseline"
                xx = [c["deployment_fraction"] for c in values]
                yy = [c["inertia_about_system_com_B_kgm2"][i][i] for c in values]
                ax.plot(xx, yy, marker="o", label=label)
            ax.set_xlabel("Deployed fraction")
            ax.set_ylabel("I{}{} about system COM (kg m²)".format(axis, axis))
            ax.set_title("{}: static assembly inertia, body axes".format(sweep_id))
            ax.grid(True, alpha=0.3)
            if len(series) <= 8:
                ax.legend(fontsize="small")
            fig.tight_layout()
            target = output_dir / "{}_I{}{}.png".format(sweep_id, axis, axis)
            fig.savefig(target, dpi=150)
            plt.close(fig)
            plots.append(target)
    return plots
