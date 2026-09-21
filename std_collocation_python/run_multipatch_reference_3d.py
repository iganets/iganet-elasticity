#!/usr/bin/env python3
"""
Classical isogeometric collocation reference solution for the 3D multipatch
cube-chain example (sim_config_3D_multi_patch_parametrized.json).

Replaces the old collocation_reference_3D_multipatch_parametrized.cxx, which
was hardcoded to exactly two cubes and solved a penalty-style (MSE) IgANet
objective via matrix-free CG instead of assembling and solving a classical
collocation system directly. This script reads multipatch.num_patches from
the config so the number of chained unit cubes is flexible, assembles the
system with solve_elasticity_collocation_multipatch_chain_3d(), and writes
the result to the SAME output path and the SAME
multipatch_elasticity.patches[] JSON schema as the old executable, so
show_3D_multi_patch_parametrized.py keeps working unchanged.

Usage:
  python3 -m std_collocation_python.run_multipatch_reference_3d [config_path] [output_json_path]

Defaults to src/examples3D/multiPatch/sim_config_3D_multi_patch_parametrized.json
and results/result_collocation_reference_3D_multipatch_parametrized.json.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from std_collocation_python.apply_bc_3d import BC, BCConfig3D
from std_collocation_python.iga_collocation_multipatch_3d import (
    solve_elasticity_collocation_multipatch_chain_3d,
)
from std_collocation_python.run_std_coll import (
    REPO_ROOT,
    build_bc_config_3d,
    get_optional,
    get_required,
    load_json_with_line_comments,
    require_vector_length,
    resolve_out_path,
)

DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "src" / "examples3D" / "multiPatch" /
    "sim_config_3D_multi_patch_parametrized.json"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "result_collocation_reference_3D_multipatch_parametrized.json"
)


def _load_bc(cfg: dict) -> tuple[BCConfig3D, tuple[float, float, float]]:
    bc_cfg = get_required(cfg, "multipatch.boundary_conditions")

    force_sides = []
    for i, side in enumerate(get_optional(bc_cfg, "force_sides", [])):
        s = require_vector_length(side, 4, f"multipatch.boundary_conditions.force_sides[{i}]")
        force_sides.append((int(s[0]), float(s[1]), float(s[2]), float(s[3])))

    diri_sides = []
    for i, side in enumerate(get_optional(bc_cfg, "diri_sides", [])):
        s = require_vector_length(side, 4, f"multipatch.boundary_conditions.diri_sides[{i}]")
        diri_sides.append((int(s[0]), float(s[1]), float(s[2]), float(s[3])))

    tfbc_sides = [int(s) for s in get_optional(bc_cfg, "tfbc_sides", [])]

    # A side listed in both tfbc_sides (traction-free) and force_sides (a
    # prescribed nonzero traction) is an ambiguous config: this Python
    # reference silently lets force_sides win (see build_bc_config_3d), but
    # the C++ IgANet trainer's loss adds BOTH a "traction == 0" term and a
    # "traction == force" term for that same side instead of picking one, so
    # it ends up optimizing two conflicting objectives on the same boundary.
    # Fail loudly here so a config like that gets fixed rather than silently
    # producing two solvers that disagree.
    force_side_ids = {entry[0] for entry in force_sides}
    diri_side_ids = {entry[0] for entry in diri_sides}
    overlap = force_side_ids & set(tfbc_sides)
    if overlap:
        raise ValueError(
            "multipatch.boundary_conditions: side(s) "
            f"{sorted(overlap)} appear in both 'tfbc_sides' and 'force_sides'. "
            "Remove them from 'tfbc_sides' (force_sides already implies a "
            "non-free traction there) -- this ambiguity makes the IgANet "
            "trainer optimize a conflicting objective on that side.")
    diri_overlap = (force_side_ids | set(tfbc_sides)) & diri_side_ids
    if diri_overlap:
        raise ValueError(
            "multipatch.boundary_conditions: side(s) "
            f"{sorted(diri_overlap)} appear in both 'diri_sides' and "
            "'tfbc_sides'/'force_sides'. A side can only have one boundary "
            "condition type.")

    bc = build_bc_config_3d(force_sides, diri_sides, tfbc_sides)

    bf_raw = get_optional(cfg, "multipatch.body_force", [0.0, 0.0, 0.0])
    bf_raw = require_vector_length(bf_raw, 3, "multipatch.body_force")
    body_force = (float(bf_raw[0]), float(bf_raw[1]), float(bf_raw[2]))

    return bc, body_force


def _patch_to_json(pidx: int, patch: dict, degree: int) -> dict:
    degrees = [degree, degree, degree]

    X0, Y0, Z0 = patch["X0"], patch["Y0"], patch["Z0"]
    u, v, w = patch["u"], patch["v"], patch["w"]

    control_points = np.column_stack([
        X0.reshape(-1, order="F"), Y0.reshape(-1, order="F"), Z0.reshape(-1, order="F")
    ])
    displacements = np.column_stack([
        u.reshape(-1, order="F"), v.reshape(-1, order="F"), w.reshape(-1, order="F")
    ])
    deformed = control_points + displacements

    return {
        "index": pidx,
        "xml_id": pidx,
        "degrees": degrees,
        "knot_vectors": [patch["csi"].tolist(), patch["eta"].tolist(), patch["zeta"].tolist()],
        "control_points": control_points.tolist(),
        "displacements": displacements.tolist(),
        "deformed_control_points": deformed.tolist(),
    }


def run(config_path: Path, out_path: Path, quiet: bool = False) -> None:
    cfg = load_json_with_line_comments(str(config_path))

    E = float(get_required(cfg, "material.young_modulus"))
    nu = float(get_required(cfg, "material.poisson_ratio"))

    geometry_ncp = int(get_required(cfg, "geometry_spline.nr_ctrl_pts"))
    geometry_degree = int(get_required(cfg, "geometry_spline.degree"))
    solution_ncp = int(get_required(cfg, "solution_spline.nr_ctrl_pts"))
    solution_degree = int(get_required(cfg, "solution_spline.degree"))
    if geometry_ncp != solution_ncp or geometry_degree != solution_degree:
        raise ValueError(
            "This reference solver assumes isoparametric spaces: "
            "geometry_spline and solution_spline must match "
            f"(got geometry={geometry_ncp}/{geometry_degree}, "
            f"solution={solution_ncp}/{solution_degree})")
    ncp = solution_ncp
    degree = solution_degree

    num_patches = int(get_required(cfg, "multipatch.num_patches"))
    cube_size = float(get_optional(cfg, "multipatch.cube_size", 1.0))

    bc, body_force = _load_bc(cfg)

    patches, meta = solve_elasticity_collocation_multipatch_chain_3d(
        num_patches=num_patches,
        p=degree, q=degree, r=degree,
        mcp=ncp, ncp=ncp, lcp=ncp,
        E=E, nu=nu, bc=bc, body_force=body_force, cube_size=cube_size,
    )

    if not quiet:
        print("\n=== COLLOCATION REFERENCE (parametric N-cube chain) ===")
        print(f"config: {config_path}")
        print(f"patches: {num_patches}")
        print(f"cube_size: {cube_size}")
        print(f"global scalar dofs: {meta['ndof']}")
        for pidx, patch in enumerate(patches):
            u = patch["u"]
            print(f"  patch {pidx}: u range [{u.min():+.6e}, {u.max():+.6e}]  "
                  f"sigma_vm range [{patch['sigma_vm'].min():.4f}, {patch['sigma_vm'].max():.4f}]")

    summary = {
        "example": "collocation_reference_multipatch_parametric_3d",
        "method": (
            "Classical isogeometric collocation: direct sparse linear solve of "
            "the assembled system (Navier-Lame collocation + boundary conditions "
            "+ interface traction-continuity equations), no least-squares/penalty "
            "fit and no neural network."),
        "device": "cpu",
        "npatches": num_patches,
        "cube_size": cube_size,
        "geometry_scalar_dofs": meta["ndof"],
        "displacement_scalar_dofs": meta["ndof"],
        "patches": [_patch_to_json(pidx, patch, degree) for pidx, patch in enumerate(patches)],
        "note": (
            "Classical isogeometric collocation reference solution (direct "
            "linear solve, not trained), generalized to a config-driven number "
            "of chained unit cubes. See std_collocation_python/"
            "iga_collocation_multipatch_3d.py."),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            with open(out_path, "r") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {}
        except json.JSONDecodeError:
            existing = {}
    existing["multipatch_elasticity"] = summary
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=1)

    if not quiet:
        print(f"result: {out_path}")
        print("======================================\n")


def main() -> None:
    quiet = "--quiet" in sys.argv[1:]
    args = [arg for arg in sys.argv[1:] if arg != "--quiet"]
    if len(args) > 2:
        raise SystemExit(
            "Usage: python3 -m std_collocation_python.run_multipatch_reference_3d "
            "[config_path] [output_json_path] [--quiet]")

    config_path = Path(args[0]) if args else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    out_path = resolve_out_path(args[1]) if len(args) == 2 else DEFAULT_OUTPUT_PATH
    out_path = Path(out_path)

    run(config_path, out_path, quiet=quiet)


if __name__ == "__main__":
    main()
