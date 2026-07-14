#!/usr/bin/env python3
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from std_collocation_python.apply_bc_2d import BC as BC2D
from std_collocation_python.apply_bc_2d import BCConfig as BCConfig2D
from std_collocation_python.apply_bc_3d import BC as BC3D
from std_collocation_python.apply_bc_3d import BCConfig3D
from std_collocation_python.iga_collocation_2d import solve_elasticity_collocation_2d
from std_collocation_python.iga_collocation_3d import solve_elasticity_collocation_3d

_SIDE_TO_NAME_2D = {
    1: "left",
    2: "right",
    3: "bottom",
    4: "top",
}

_SIDE_TO_NAME_3D = {
    1: "left",
    2: "right",
    3: "bottom",
    4: "top",
    5: "front",
    6: "back",
}

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_json_with_line_comments(path: str) -> dict:
    with open(path, "r") as f:
        txt = f.read()
    txt = re.sub(r"//.*?$", "", txt, flags=re.MULTILINE)
    return json.loads(txt)


def get_required(d: dict, path: str):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required config key: '{path}'")
        cur = cur[key]
    return cur


def get_optional(d: dict, path: str, default):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def require_vector_length(values, expected_length: int, name: str):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"Config key '{name}' must be a list with {expected_length} entries.")
    if len(values) != expected_length:
        raise ValueError(
            f"Config key '{name}' must have exactly {expected_length} entries, "
            f"but has {len(values)}: {values}"
        )
    return values


def resolve_out_path(out_path_from_cfg: str) -> str:
    out_path = Path(out_path_from_cfg)
    if out_path.is_absolute():
        return str(out_path)
    return str(REPO_ROOT / out_path)


def detect_dimension(cfg: dict) -> int:
    if "single_patch_2D" in cfg:
        return 2
    if "multipatch_2D" in cfg:
        return 2
    if "single_patch_3D" in cfg:
        return 3
    if "multipatch_3D" in cfg:
        return 3
    if "patches_2d" in cfg:
        return 2

    body_force = get_required(cfg, "body_force")
    if not isinstance(body_force, (list, tuple)):
        raise ValueError("Config key 'body_force' must be a list.")
    if len(body_force) == 2:
        return 2
    if len(body_force) == 3:
        return 3
    raise ValueError(
        f"Unsupported problem dimension: body_force has length {len(body_force)}. "
        "Expected 2 for 2D or 3 for 3D."
    )


def _zero(dim: int) -> np.ndarray:
    return np.zeros(dim, dtype=float)


def build_bc_config_2d(force_sides, diri_sides, tfbc_sides) -> BCConfig2D:
    bc_map = {name: BC2D("free", _zero(2)) for name in _SIDE_TO_NAME_2D.values()}

    for side in tfbc_sides:
        bc_map[_SIDE_TO_NAME_2D[int(side)]] = BC2D("free", _zero(2))

    for side, tx, ty in force_sides:
        bc_map[_SIDE_TO_NAME_2D[int(side)]] = BC2D(
            "neumann", np.array([float(tx), float(ty)], dtype=float)
        )

    for side, ux, uy in diri_sides:
        bc_map[_SIDE_TO_NAME_2D[int(side)]] = BC2D(
            "dirichlet", np.array([float(ux), float(uy)], dtype=float)
        )

    return BCConfig2D(
        left=bc_map["left"],
        right=bc_map["right"],
        bottom=bc_map["bottom"],
        top=bc_map["top"],
    )


def build_bc_config_3d(force_sides, diri_sides, tfbc_sides) -> BCConfig3D:
    bc_map = {name: BC3D("free", _zero(3)) for name in _SIDE_TO_NAME_3D.values()}

    for side in tfbc_sides:
        bc_map[_SIDE_TO_NAME_3D[int(side)]] = BC3D("free", _zero(3))

    for side, tx, ty, tz in force_sides:
        bc_map[_SIDE_TO_NAME_3D[int(side)]] = BC3D(
            "neumann", np.array([float(tx), float(ty), float(tz)], dtype=float)
        )

    for side, ux, uy, uz in diri_sides:
        bc_map[_SIDE_TO_NAME_3D[int(side)]] = BC3D(
            "dirichlet", np.array([float(ux), float(uy), float(uz)], dtype=float)
        )

    return BCConfig3D(
        left=bc_map["left"],
        right=bc_map["right"],
        bottom=bc_map["bottom"],
        top=bc_map["top"],
        front=bc_map["front"],
        back=bc_map["back"],
    )


def update_json_payload(out_path: str, payload: dict) -> None:
    existing = {}
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        try:
            with open(out_path, "r") as f:
                existing = json.load(f)
            if not isinstance(existing, dict):
                existing = {"_previous_content": existing}
        except json.JSONDecodeError:
            existing = {"_previous_raw": "INVALID_JSON"}

    existing.update(payload)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)


def load_patch_config_2d(cfg: dict) -> dict:
    if "single_patch_2D" in cfg:
        return get_required(cfg, "single_patch_2D")

    patches = get_required(cfg, "patches_2d")
    if not isinstance(patches, list) or not patches:
        raise ValueError("Config key 'patches_2d' must contain at least one patch entry.")
    return patches[0]


def run_2d(cfg: dict, out_path: str) -> None:
    E = float(get_required(cfg, "material.young_modulus"))
    nu = float(get_required(cfg, "material.poisson_ratio"))
    ncp = int(get_required(cfg, "solution_spline.nr_ctrl_pts") if "solution_spline" in cfg else get_required(cfg, "spline.nr_ctrl_pts"))
    degree = int(get_required(cfg, "solution_spline.degree") if "solution_spline" in cfg else get_required(cfg, "spline.degree"))

    patch_cfg = load_patch_config_2d(cfg)
    bf_raw = require_vector_length(
        patch_cfg["body_force"],
        2,
        "single_patch_2D.body_force" if "single_patch_2D" in cfg else "patches_2d[0].body_force",
    )
    body_force = (float(bf_raw[0]), float(bf_raw[1]))

    bc_cfg = get_required(patch_cfg, "boundary_conditions")
    force_sides = []
    for i, side in enumerate(get_optional(bc_cfg, "force_sides", [])):
        path = (
            f"single_patch_2D.boundary_conditions.force_sides[{i}]"
            if "single_patch_2D" in cfg
            else f"patches_2d[0].boundary_conditions.force_sides[{i}]"
        )
        s = require_vector_length(side, 3, path)
        force_sides.append((int(s[0]), float(s[1]), float(s[2])))

    diri_sides = []
    for i, side in enumerate(get_optional(bc_cfg, "diri_sides", [])):
        path = (
            f"single_patch_2D.boundary_conditions.diri_sides[{i}]"
            if "single_patch_2D" in cfg
            else f"patches_2d[0].boundary_conditions.diri_sides[{i}]"
        )
        s = require_vector_length(side, 3, path)
        diri_sides.append((int(s[0]), float(s[1]), float(s[2])))

    tfbc_sides = [int(s) for s in get_optional(bc_cfg, "tfbc_sides", [])]
    bc = build_bc_config_2d(force_sides, diri_sides, tfbc_sides)

    u, v, sigma_vm, meta = solve_elasticity_collocation_2d(
        p=degree,
        q=degree,
        mcp=ncp,
        ncp=ncp,
        E=E,
        nu=nu,
        bc=bc,
        body_force=body_force,
    )

    print("\n=== RANGE CHECK (2D) ===")
    print(f"u: min = {float(u.min()):+.6e}, max = {float(u.max()):+.6e}")
    print(f"v: min = {float(v.min()):+.6e}, max = {float(v.max()):+.6e}")
    print(f"sigma_vm: min = {float(sigma_vm.min()):+.6e}, max = {float(sigma_vm.max()):+.6e}")
    print("========================\n")

    X0 = meta["X0"]
    Y0 = meta["Y0"]

    U = u.reshape(ncp, ncp, order="F")
    V = v.reshape(ncp, ncp, order="F")

    X_def = X0 + U
    Y_def = Y0 + V

    xd = X_def.reshape(-1, order="F")
    yd = Y_def.reshape(-1, order="F")
    uf = U.reshape(-1, order="F")
    vf = V.reshape(-1, order="F")
    sf = sigma_vm.reshape(-1, order="F")

    payload = {
        "stdCollCtrlPts": np.column_stack([xd, yd]).tolist(),
        "stdCollDeformations": np.column_stack([uf, vf]).tolist(),
        "stdCollStresses": sf.tolist(),
        "stdCollDisplacement": np.column_stack([uf, vf]).tolist(),
    }
    update_json_payload(out_path, payload)


def run_3d(cfg: dict, out_path: str) -> None:
    E = float(get_required(cfg, "material.young_modulus"))
    nu = float(get_required(cfg, "material.poisson_ratio"))
    ncp = int(get_required(cfg, "solution_spline.nr_ctrl_pts") if "solution_spline" in cfg else get_required(cfg, "spline.nr_ctrl_pts"))
    degree = int(get_required(cfg, "solution_spline.degree") if "solution_spline" in cfg else get_required(cfg, "spline.degree"))

    sp_cfg = get_optional(cfg, "single_patch_3D", None)
    cfg_3d = sp_cfg if sp_cfg is not None else cfg

    bf_raw = require_vector_length(
        get_required(cfg_3d, "body_force"),
        3,
        "single_patch_3D.body_force" if sp_cfg is not None else "body_force",
    )
    body_force = (float(bf_raw[0]), float(bf_raw[1]), float(bf_raw[2]))

    bc_cfg = get_required(cfg_3d, "boundary_conditions")
    force_sides = []
    for i, side in enumerate(get_optional(bc_cfg, "force_sides", [])):
        path = (
            f"single_patch_3D.boundary_conditions.force_sides[{i}]"
            if sp_cfg is not None
            else f"boundary_conditions.force_sides[{i}]"
        )
        s = require_vector_length(side, 4, path)
        force_sides.append((int(s[0]), float(s[1]), float(s[2]), float(s[3])))

    diri_sides = []
    for i, side in enumerate(get_optional(bc_cfg, "diri_sides", [])):
        path = (
            f"single_patch_3D.boundary_conditions.diri_sides[{i}]"
            if sp_cfg is not None
            else f"boundary_conditions.diri_sides[{i}]"
        )
        s = require_vector_length(side, 4, path)
        diri_sides.append((int(s[0]), float(s[1]), float(s[2]), float(s[3])))

    tfbc_sides = [int(s) for s in get_optional(bc_cfg, "tfbc_sides", [])]
    bc = build_bc_config_3d(force_sides, diri_sides, tfbc_sides)

    u, v, w, sigma_vm, meta = solve_elasticity_collocation_3d(
        p=degree,
        q=degree,
        r=degree,
        mcp=ncp,
        ncp=ncp,
        lcp=ncp,
        E=E,
        nu=nu,
        bc=bc,
        body_force=body_force,
    )

    print("\n=== RANGE CHECK (3D) ===")
    print(f"u: min = {float(u.min()):+.6e},  max = {float(u.max()):+.6e}")
    print(f"v: min = {float(v.min()):+.6e},  max = {float(v.max()):+.6e}")
    print(f"w: min = {float(w.min()):+.6e},  max = {float(w.max()):+.6e}")
    print(f"sigma_vm: min = {float(sigma_vm.min()):+.6e},  max = {float(sigma_vm.max()):+.6e}")
    print("========================\n")

    X0 = meta["X0"]
    Y0 = meta["Y0"]
    Z0 = meta["Z0"]

    U = u.reshape(ncp, ncp, ncp, order="F")
    V = v.reshape(ncp, ncp, ncp, order="F")
    W = w.reshape(ncp, ncp, ncp, order="F")

    X_def = X0 + U
    Y_def = Y0 + V
    Z_def = Z0 + W

    xd = X_def.reshape(-1, order="F")
    yd = Y_def.reshape(-1, order="F")
    zd = Z_def.reshape(-1, order="F")
    uf = U.reshape(-1, order="F")
    vf = V.reshape(-1, order="F")
    wf = W.reshape(-1, order="F")
    sf = sigma_vm.reshape(-1, order="F")

    payload = {
        "stdCollCtrlPts": np.column_stack([xd, yd, zd]).tolist(),
        "stdCollDeformations": np.column_stack([uf, vf, wf]).tolist(),
        "stdCollStresses": sf.tolist(),
        "stdCollDisplacement": np.column_stack([uf, vf, wf]).tolist(),
    }
    update_json_payload(out_path, payload)


def main():
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "Usage: python3 std_collocation_python/run_std_coll.py <config_path> [output_json_path]"
        )

    config_arg = sys.argv[1]
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    cfg = load_json_with_line_comments(str(config_path))

    if len(sys.argv) == 3:
        out_path = resolve_out_path(sys.argv[2])
    else:
        out_path_cfg = get_optional(cfg, "simulation.json_path", "results/result.json")
        out_path = resolve_out_path(out_path_cfg)

    dim = detect_dimension(cfg)
    if dim == 2:
        run_2d(cfg, out_path)
    elif dim == 3:
        run_3d(cfg, out_path)
    else:
        raise ValueError(f"Unsupported dimension {dim}")

    print(f"Updated JSON keys in: {out_path}")


if __name__ == "__main__":
    main()
