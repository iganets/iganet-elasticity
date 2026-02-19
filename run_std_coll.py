#!/usr/bin/env python3
import os
import re
import json
import numpy as np

from std_collocation_python.iga_collocation import solve_elasticity_collocation_2d
from std_collocation_python.apply_bc import BC, BCConfig

# Side convention (match C++):
# 1 = left, 2 = right, 3 = bottom, 4 = top
_SIDE_TO_NAME = {1: "left", 2: "right", 3: "bottom", 4: "top"}


def load_json_with_line_comments(path: str) -> dict:
    """
    Loads JSON that may contain // line comments.
    NOTE: This does NOT fix invalid JSON (e.g. missing commas, trailing commas).
    """
    with open(path, "r") as f:
        txt = f.read()
    txt = re.sub(r"//.*?$", "", txt, flags=re.MULTILINE)  # remove // comments
    return json.loads(txt)


def get_required(d: dict, path: str):
    """Get nested key 'a.b.c' or raise KeyError."""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required config key: '{path}'")
        cur = cur[key]
    return cur


def get_optional(d: dict, path: str, default):
    """Get nested key 'a.b.c' or return default."""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _build_bc_config(force_sides, diri_sides, tfbc_sides) -> BCConfig:
    """
    Build BCConfig from lists:
      force_sides: [(side, tx, ty), ...]
      diri_sides:  [(side, ux, uy), ...]
      tfbc_sides:  [side, ...]
    Dirichlet always wins.
    """
    bc_map = {
        "left": BC("free", np.array([0.0, 0.0])),
        "right": BC("free", np.array([0.0, 0.0])),
        "bottom": BC("free", np.array([0.0, 0.0])),
        "top": BC("free", np.array([0.0, 0.0])),
    }

    # traction-free: explicit free (no change, but keeps intent)
    for s in tfbc_sides:
        name = _SIDE_TO_NAME[int(s)]
        bc_map[name] = BC("free", np.array([0.0, 0.0]))

    # Neumann
    for (s, tx, ty) in force_sides:
        name = _SIDE_TO_NAME[int(s)]
        bc_map[name] = BC("neumann", np.array([float(tx), float(ty)], dtype=float))

    # Dirichlet overrides
    for (s, ux, uy) in diri_sides:
        name = _SIDE_TO_NAME[int(s)]
        bc_map[name] = BC("dirichlet", np.array([float(ux), float(uy)], dtype=float))

    return BCConfig(
        left=bc_map["left"],
        right=bc_map["right"],
        bottom=bc_map["bottom"],
        top=bc_map["top"],
    )


def resolve_out_path(out_path_from_cfg: str) -> str:
    """
    If config gives an absolute path -> use it.
    If it gives a relative path -> interpret relative to run_std_coll.py directory.
    """
    if os.path.isabs(out_path_from_cfg):
        return out_path_from_cfg
    return os.path.join(os.path.dirname(__file__), out_path_from_cfg)


def main():
    # Config is always next to this script
    config_path = os.path.join(os.path.dirname(__file__), "sim_config.json")
    cfg = load_json_with_line_comments(config_path)

    # --- required: material ---
    E = float(get_required(cfg, "material.young_modulus"))
    nu = float(get_required(cfg, "material.poisson_ratio"))

    # --- required: spline ---
    ncp = int(get_required(cfg, "spline.nr_ctrl_pts"))
    degree = int(get_required(cfg, "spline.degree"))

    # --- required: body force ---
    body_force = get_required(cfg, "body_force")
    body_force = (float(body_force[0]), float(body_force[1]))

    # --- boundary conditions ---
    bc_cfg = get_required(cfg, "boundary_conditions")
    force_sides = [(int(s[0]), float(s[1]), float(s[2])) for s in get_optional(bc_cfg, "force_sides", [])]
    diri_sides = [(int(s[0]), float(s[1]), float(s[2])) for s in get_optional(bc_cfg, "diri_sides", [])]
    tfbc_sides = [int(s) for s in get_optional(bc_cfg, "tfbc_sides", [3, 4])]

    bc = _build_bc_config(force_sides, diri_sides, tfbc_sides)

    # --- output path ---
    out_path_cfg = get_optional(cfg, "simulation.json_path", "result.json")
    out_path = resolve_out_path(out_path_cfg)

    # --- run solver ---
    mcp = ncp
    p = degree
    q = degree

    u, v, sigma_vm, meta = solve_elasticity_collocation_2d(
        p=p, q=q, mcp=mcp, ncp=ncp,
        E=E, nu=nu,
        bc=bc,
        body_force=body_force
    )

    # quick range check
    print("\n=== RANGE CHECK ===")
    print(f"u: min = {float(u.min()):+.6e}, max = {float(u.max()):+.6e}")
    print(f"v: min = {float(v.min()):+.6e}, max = {float(v.max()):+.6e}")
    print(f"sigma_vm: min = {float(sigma_vm.min()):+.6e}, max = {float(sigma_vm.max()):+.6e}")
    print("===================\n")

    # --- export payload (MATLAB compatible order="F") ---
    X0 = meta["X0"]
    Y0 = meta["Y0"]

    U = u.reshape(mcp, ncp, order="F")
    V = v.reshape(mcp, ncp, order="F")

    X_def = X0 + U
    Y_def = Y0 + V

    xd = X_def.reshape(-1, order="F")
    yd = Y_def.reshape(-1, order="F")
    uf = U.reshape(-1, order="F")
    vf = V.reshape(-1, order="F")
    sf = sigma_vm.reshape(-1, order="F")

    new_payload = {
        "stdCollCtrlPts": np.column_stack([xd, yd]).tolist(),          # [[x,y], ...]
        "stdCollDeformations": np.column_stack([uf, vf]).tolist(),     # [[u,v], ...]
        "stdCollStresses": sf.tolist()                                 # [sigma, ...]
    }

    # --- update existing json without deleting other keys ---
    existing = {}
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            with open(out_path, "r") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {"_previous_content": existing}
        except json.JSONDecodeError:
            existing = {"_previous_raw": "INVALID_JSON"}

    existing.update(new_payload)

    # Ensure target directory exists (if user specified one)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Updated JSON keys in: {out_path}")


if __name__ == "__main__":
    main()
