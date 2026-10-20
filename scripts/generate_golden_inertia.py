#!/usr/bin/env python
"""Generate tests/fixtures/golden_inertia.json from the LEGACY notebook.

The inertia / rotation functions are pulled out of the notebook source with ``ast`` and
executed in isolation (no astropy / scipy / WMM / input() needed), so the fixture records
exactly what the original code produced. Re-run only if you deliberately change the
reference inputs; the numbers in the fixture are the regression target for ``edtsim``.

Usage:
    python scripts/generate_golden_inertia.py \
        --notebook legacy/MOI_model_current_profile_only__1_.ipynb \
        --out tests/fixtures/golden_inertia.json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

WANTED = {
    "hat", "unit", "parallel_axis_shift", "point_mass_inertia_about_origin",
    "box_inertia_about_com", "annular_cylinder_inertia_about_com",
    "slender_rod_inertia_about_com", "inertia_body_to_inertial",
    "rotation_matrix_from_a_to_b", "sample_point_in_annular_cylinder",
    "BusParams", "SpoolParams", "TetherParams", "PointMassParams",
    "moi_config_1_coiled", "moi_config_2_extended_on_minus_z",
    "Rx", "Ry", "Rz", "euler_zyx_to_R",
    "compute_inertia_from_particles_about_bus_com_in_inertial",
}

HEADER = (
    "from __future__ import annotations\n"
    "from dataclasses import dataclass\n"
    "from typing import Tuple\n"
    "import numpy as np\n\n"
)


def load_legacy(nb_path: Path) -> types.ModuleType:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    chunks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:      # e.g. the '%pip install ...' cell
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANTED:
                chunks.append(ast.unparse(node))
    mod = types.ModuleType("legacy_moi")
    sys.modules["legacy_moi"] = mod
    exec(compile(HEADER + "\n\n".join(chunks), "legacy_moi", "exec"), mod.__dict__)
    missing = [n for n in WANTED if not hasattr(mod, n)]
    if missing:
        raise RuntimeError(f"Legacy notebook is missing expected definitions: {missing}")
    return mod


def arr(x):
    return np.asarray(x, dtype=float).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    L = load_legacy(args.notebook)

    # ---- Reference SATELLITE cases (synthetic but CubeSat-plausible; ADD YOUR MISSION VALUES HERE) ----
    cases = {
        "3U_known_end_mass": dict(
            bus=dict(mass=4.0, a=0.1, b=0.1, c=0.3),
            spool=dict(mass=0.6, r_i=0.015, r_o=0.04, h=0.06, axis_s_B=[0, 0, 1], com_B=[0, 0, 0.12]),
            m_end_stowed=0.15, end_mode="known", r_end_B=[0.02, 0.01, 0.13],
            tether=dict(length=100.0, mass=0.5, attach_B=[0, 0, -0.15]), m_end_deployed=0.15),
        "tilted_spool_random_seed42": dict(
            bus=dict(mass=3.2, a=0.1, b=0.1, c=0.227),
            spool=dict(mass=0.4, r_i=0.01, r_o=0.035, h=0.05, axis_s_B=[1, 1, 0], com_B=[0.01, -0.02, 0.05]),
            m_end_stowed=0.1, end_mode="random", seed=42,
            tether=dict(length=50.0, mass=0.3, attach_B=[0.02, 0.0, -0.1135]), m_end_deployed=0.1),
        "offset_attach_short_tether": dict(
            bus=dict(mass=2.5, a=0.1, b=0.1, c=0.2),
            spool=dict(mass=0.25, r_i=0.0, r_o=0.03, h=0.04, axis_s_B=[0, 1, 0], com_B=[0.0, 0.0, -0.05]),
            m_end_stowed=0.05, end_mode="known", r_end_B=[-0.01, 0.0, -0.06],
            tether=dict(length=20.0, mass=0.2, attach_B=[0.03, -0.02, -0.15]), m_end_deployed=0.05),
    }

    out_cases = {}
    for name, c in cases.items():
        bus = L.BusParams(**c["bus"])
        spool = L.SpoolParams(**{k: (np.array(v, float) if k in ("axis_s_B", "com_B") else v)
                                 for k, v in c["spool"].items()})
        if c["end_mode"] == "known":
            r_end = np.array(c["r_end_B"], float)
        else:
            rng = np.random.default_rng(c["seed"])
            r_end = L.sample_point_in_annular_cylinder(
                r_i=spool.r_i, r_o=spool.r_o, h=spool.h, axis_s_B=spool.axis_s_B,
                com_B=spool.com_B, rng=rng)
        I1 = L.moi_config_1_coiled(bus, spool, c["m_end_stowed"], r_end)
        tether = L.TetherParams(length=c["tether"]["length"], mass=c["tether"]["mass"],
                                attach_B=np.array(c["tether"]["attach_B"], float))
        I2, r_p, r_a = L.moi_config_2_extended_on_minus_z(
            bus, tether, L.PointMassParams(mass=c["m_end_deployed"]))
        out_cases[name] = dict(inputs=c, r_end_B=arr(r_end), I1=arr(I1), I2=arr(I2),
                               r_p_B0=arr(r_p), r_a_B=arr(r_a))

    # ---- Primitive spot checks ----
    prim = dict(
        parallel_axis=[dict(m=0.7, r=r, I=arr(L.parallel_axis_shift(0.7, np.array(r, float))))
                       for r in ([0.1, 0.0, 0.0], [0.03, -0.2, 0.5], [0, 0, 0])],
        box=[dict(m=m, a=a, b=b, c=c, I=arr(L.box_inertia_about_com(m, a, b, c)))
             for (m, a, b, c) in [(1.0, 1, 1, 1), (4.0, 0.1, 0.1, 0.3), (2.0, 0.3, 0.2, 0.1)]],
        annulus=[dict(m=m, r_i=ri, r_o=ro, h=h, axis=ax,
                      I=arr(L.annular_cylinder_inertia_about_com(m, ri, ro, h, np.array(ax, float))))
                 for (m, ri, ro, h, ax) in [(0.6, 0.015, 0.04, 0.06, [0, 0, 1]),
                                            (0.3, 0.0, 0.05, 0.1, [1, 2, 3]),
                                            (1.0, 0.02, 0.03, 0.01, [-1, 0.5, 0.2])]],
        rod=[dict(m=m, L=Ln, axis=ax, I=arr(L.slender_rod_inertia_about_com(m, Ln, np.array(ax, float))))
             for (m, Ln, ax) in [(0.5, 100.0, [0, 0, -1]), (0.2, 3.0, [1, 1, 0])]],
    )
    euler = []
    for (r_, p_, y_) in [(0, 0, 0), (0.3, -0.5, 1.1), (-1.2, 0.7, 2.9), (np.pi / 2, 0, 0)]:
        R = L.euler_zyx_to_R(r_, p_, y_)
        I_B = L.box_inertia_about_com(4.0, 0.1, 0.1, 0.3)
        euler.append(dict(roll=r_, pitch=p_, yaw=y_, R=arr(R),
                          I_I=arr(L.inertia_body_to_inertial(I_B, R))))
    a2b = []
    for (a, b) in [([0, 0, 1], [1, 0, 0]), ([0, 0, 1], [0, 0, 1]), ([0, 0, 1], [0, 0, -1]),
                   ([1, 0, 0], [-1, 0, 0]), ([1, 2, 3], [-2, 0.5, 1])]:
        a2b.append(dict(a=a, b=b, R=arr(L.rotation_matrix_from_a_to_b(np.array(a, float), np.array(b, float)))))

    rng = np.random.default_rng(7)
    rp = rng.normal(size=(50, 3)) * 3.0
    mp = rng.uniform(0.01, 0.5, size=50)
    ref = rng.normal(size=3)
    particles = dict(r=arr(rp), m=arr(mp), ref=arr(ref),
                     I=arr(L.compute_inertia_from_particles_about_bus_com_in_inertial(rp, mp, ref)))

    payload = dict(
        provenance=dict(
            generated_from=args.notebook.name,
            notebook_sha256=hashlib.sha256(args.notebook.read_bytes()).hexdigest(),
            note="Reference values computed by the ORIGINAL notebook functions. Case inputs are synthetic.",
        ),
        satellite_cases=out_cases, primitives=prim, euler_zyx=euler,
        rotation_a_to_b=a2b, particles=particles,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.out}  ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
