"""Versioned JSON scenario inputs for M2 static mass-property studies.

This module intentionally uses only the Python standard library and the
already existing M2 parameter/model classes. JSON is **input data**, never
executable Python. The provenance status is user-declared, not verified by code.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from edtsim.core.params import BusParams, PointMassParams, TetherParams
from edtsim.inertia.mission import MassComponent, MissionMassModel, WoundTetherGeometry

SCHEMA_VERSION = 1
PROVENANCE_STATUSES = frozenset(("illustrative", "preliminary", "verified"))
# Only these scalar design variables can be changed by a design-grid sweep.
# Changing tether length with fixed linear density changes the total tether mass.
DESIGN_AXES = frozenset((
    "bus.mass_kg", "tether.length_m", "tether.linear_density_kg_per_m",
    "end_mass.mass_kg", "spool_hardware.mass_kg",
    "wound_geometry.r_inner_m", "wound_geometry.r_outer_m",
    "wound_geometry.width_m",
))


def _check_keys(obj: Mapping[str, Any], required: set, optional: set, label: str) -> None:
    if not isinstance(obj, dict):
        raise ValueError("{} must be a JSON object".format(label))
    missing = required - set(obj)
    extra = set(obj) - required - optional
    if missing or extra:
        raise ValueError("{} fields: missing={}, unexpected={}".format(label, sorted(missing), sorted(extra)))


def finite_number(value: Any, label: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a finite JSON number".format(label))
    v = float(value)
    if not math.isfinite(v) or (positive and v <= 0) or (nonnegative and v < 0):
        raise ValueError("{} is not a valid {}number".format(label, "positive " if positive else "nonnegative " if nonnegative else "finite "))
    return v


def _vector(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("{} must be an array of three finite numbers".format(label))
    return np.asarray([finite_number(x, label) for x in value], dtype=float)


def _tensor(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("{} must be a 3x3 array".format(label))
    return np.asarray([_vector(row, label) for row in value], dtype=float)


def validate_provenance(raw: Any, label: str) -> Dict[str, str]:
    _check_keys(raw, {"status", "source", "revision"}, set(), label)
    if not isinstance(raw["status"], str) or raw["status"] not in PROVENANCE_STATUSES:
        raise ValueError("{} status must be illustrative, preliminary or verified".format(label))
    for key in ("source", "revision"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError("{}.{} must be a nonempty string".format(label, key))
    return dict(raw)


def _component(raw: Any, label: str, *, fixed_name: str = "") -> MassComponent:
    required = {"mass_kg", "com_B_m", "inertia_com_B_kgm2"}
    if not fixed_name:
        required.add("name")
    _check_keys(raw, required, set(), label)
    name = fixed_name or raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("{}.name cannot be empty".format(label))
    return MassComponent(name, finite_number(raw["mass_kg"], label + ".mass_kg", positive=True),
                         _vector(raw["com_B_m"], label + ".com_B_m"),
                         _tensor(raw["inertia_com_B_kgm2"], label + ".inertia_com_B_kgm2"))


def validate_model_config(raw: Any) -> dict:
    """Validate a model dictionary, preserving its JSON-native representation."""
    _check_keys(raw, {"bus", "spool_hardware", "tether", "end_mass", "wound_geometry"},
                {"other_fixed"}, "model")
    bus = raw["bus"]
    _check_keys(bus, {"mass_kg", "dimensions_m"}, set(), "model.bus")
    finite_number(bus["mass_kg"], "bus.mass_kg", positive=True)
    if not isinstance(bus["dimensions_m"], list) or len(bus["dimensions_m"]) != 3:
        raise ValueError("bus.dimensions_m must be a three-number list")
    for x in bus["dimensions_m"]:
        finite_number(x, "bus.dimensions_m", positive=True)
    _component(raw["spool_hardware"], "model.spool_hardware", fixed_name="spool_hardware")
    tether = raw["tether"]
    _check_keys(tether, {"length_m", "linear_density_kg_per_m", "attach_B_m"}, set(), "model.tether")
    finite_number(tether["length_m"], "tether.length_m", positive=True)
    finite_number(tether["linear_density_kg_per_m"], "tether.linear_density_kg_per_m", positive=True)
    finite_number(tether["length_m"] * tether["linear_density_kg_per_m"], "tether derived mass", positive=True)
    _vector(tether["attach_B_m"], "tether.attach_B_m")
    end = raw["end_mass"]
    _check_keys(end, {"mass_kg", "stowed_B_m"}, set(), "model.end_mass")
    finite_number(end["mass_kg"], "end_mass.mass_kg", positive=True)
    _vector(end["stowed_B_m"], "end_mass.stowed_B_m")
    wound = raw["wound_geometry"]
    _check_keys(wound, {"r_inner_m", "r_outer_m", "width_m", "axis_B", "com_B_m"}, set(), "model.wound_geometry")
    finite_number(wound["r_inner_m"], "wound_geometry.r_inner_m", nonnegative=True)
    finite_number(wound["r_outer_m"], "wound_geometry.r_outer_m", positive=True)
    finite_number(wound["width_m"], "wound_geometry.width_m", positive=True)
    _vector(wound["axis_B"], "wound_geometry.axis_B")
    _vector(wound["com_B_m"], "wound_geometry.com_B_m")
    others = raw.get("other_fixed", [])
    if not isinstance(others, list):
        raise ValueError("model.other_fixed must be a list")
    for i, item in enumerate(others):
        _component(item, "model.other_fixed[{}]".format(i))
    # Validate physical tensor and geometry consistency as the model does.
    build_mission_model(raw)
    return copy.deepcopy(raw)


def build_mission_model(raw: Mapping[str, Any]) -> MissionMassModel:
    """Instantiate the existing M2 engine; it remains the only physics source."""
    b, s, t, e, g = (raw[x] for x in ("bus", "spool_hardware", "tether", "end_mass", "wound_geometry"))
    bus = BusParams(b["mass_kg"], *b["dimensions_m"])
    spool = _component(s, "spool_hardware", fixed_name="spool_hardware")
    tether = TetherParams(t["length_m"], t["length_m"] * t["linear_density_kg_per_m"], t["attach_B_m"])
    end = PointMassParams(e["mass_kg"])
    geom = WoundTetherGeometry(g["r_inner_m"], g["r_outer_m"], g["width_m"], g["axis_B"], g["com_B_m"])
    others = tuple(_component(item, "other_fixed") for item in raw.get("other_fixed", []))
    return MissionMassModel(bus, spool, tether, end, geom, e["stowed_B_m"], others)


def _no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("Duplicate JSON key: {}".format(key))
        out[key] = value
    return out


def _reject_nonfinite(value):
    raise ValueError("Nonstandard JSON numeric constant: {}".format(value))


@dataclass(frozen=True)
class Scenario:
    identifier: str
    description: str
    revision: str
    provenance: dict
    parameter_provenance: dict
    model: dict
    sweeps: Tuple[dict, ...]
    source: str = ""


def load_scenario(path: Any) -> Scenario:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys,
                         parse_constant=_reject_nonfinite)
    except (OSError, json.JSONDecodeError) as ex:
        raise ValueError("Cannot load scenario JSON {}: {}".format(path, ex)) from ex
    _check_keys(raw, {"schema_version", "id", "description", "revision", "provenance", "model", "sweeps"},
                {"parameter_provenance"}, "scenario")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported scenario schema version (expected 1)")
    for key in ("id", "description", "revision"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError("Scenario {} must be a nonempty string".format(key))
    if not all(c.isalnum() or c in "-_" for c in raw["id"]):
        raise ValueError("Scenario id must use only letters, digits, hyphen or underscore")
    provenance = validate_provenance(raw["provenance"], "provenance")
    pp = raw.get("parameter_provenance", {})
    if not isinstance(pp, dict):
        raise ValueError("parameter_provenance must be a JSON object")
    for key, value in pp.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("parameter_provenance keys cannot be empty")
        validate_provenance(value, "parameter_provenance." + key)
    model = validate_model_config(raw["model"])
    sweeps = raw["sweeps"]
    if not isinstance(sweeps, list) or not sweeps:
        raise ValueError("sweeps must be a nonempty array")
    seen = set()
    for sweep in sweeps:
        if not isinstance(sweep, dict) or not isinstance(sweep.get("id"), str) or not sweep["id"]:
            raise ValueError("Each sweep requires a nonempty id")
        if not all(c.isalnum() or c in "-_" for c in sweep["id"]):
            raise ValueError("Sweep ids may contain only letters, numbers, - and _")
        if sweep["id"] in seen:
            raise ValueError("Sweep ids must be unique")
        seen.add(sweep["id"])
    return Scenario(raw["id"], raw["description"], raw["revision"], provenance,
                    copy.deepcopy(pp), model, tuple(copy.deepcopy(sweeps)), str(path))
