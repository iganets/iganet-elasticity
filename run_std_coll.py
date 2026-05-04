#!/usr/bin/env python3
import os
import re
import json
import numpy as np

from std_collocation_python.iga_collocation_3d import solve_elasticity_collocation_3d
from std_collocation_python.apply_bc_3d import BC, BCConfig3D

# Side convention (matches C++ code exactly):
#   1 = x=0 (left),   2 = x=1 (right)
#   3 = y=0 (bottom), 4 = y=1 (top)
#   5 = z=0 (front),  6 = z=1 (back)
_SIDE_TO_NAME = {
    1: "left",
    2: "right",
    3: "bottom",
    4: "top",
    5: "front",
    6: "back",
}


def load_json_with_line_comments(path: str) -> dict:
    """Loads JSON that may contain // line comments."""
    with open(path, "r") as f:
        txt = f.read()
    txt = re.sub(r"//.*?$", "", txt, flags=re.MULTILINE)
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


def _zero3():
    return np.zeros(3, dtype=float)


def _build_bc_config(force_sides, diri_sides, tfbc_sides) -> BCConfig3D:
    """
    Build BCConfig3D from lists.

    force_sides : [(side, tx, ty, tz), ...]
    diri_sides  : [(side, ux, uy, uz), ...]
    tfbc_sides  : [side, ...]

    Priority: Dirichlet > Neumann > free  (same as C++ bc_priority logic).
    """
    bc_map = {name: BC("free", _zero3()) for name in _SIDE_TO_NAME.values()}

    # Traction-free (explicit, no change from default but kept for clarity)
    for s in tfbc_sides:
        name = _SIDE_TO_NAME[int(s)]
        bc_map[name] = BC("free", _zero3())

    # Neumann
    for entry in force_sides:
        s, tx, ty, tz = int(entry[0]), float(entry[1]), float(entry[2]), float(entry[3])
        name = _SIDE_TO_NAME[s]
        bc_map[name] = BC("neumann", np.array([tx, ty, tz], dtype=float))

    # Dirichlet overrides everything
    for entry in diri_sides:
        s, ux, uy, uz = int(entry[0]), float(entry[1]), float(entry[2]), float(entry[3])
        name = _SIDE_TO_NAME[s]
        bc_map[name] = BC("dirichlet", np.array([ux, uy, uz], dtype=float))

    return BCConfig3D(
        left=bc_map["left"],
        right=bc_map["right"],
        bottom=bc_map["bottom"],
        top=bc_map["top"],
        front=bc_map["front"],
        back=bc_map["back"],
    )


def resolve_out_path(out_path_from_cfg: str) -> str:
    if os.path.isabs(out_path_from_cfg):
        return out_path_from_cfg
    return os.path.join(os.path.dirname(__file__), out_path_from_cfg)


def main():
    config_path = os.path.join(os.path.dirname(__file__), "sim_config.json")
    cfg = load_json_with_line_comments(config_path)

    # --- Material ---
    E  = float(get_required(cfg, "material.young_modulus"))
    nu = float(get_required(cfg, "material.poisson_ratio"))

    # --- Spline ---
    ncp    = int(get_required(cfg, "spline.nr_ctrl_pts"))
    degree = int(get_required(cfg, "spline.degree"))

    # --- Body force (3 components) ---
    bf_raw     = get_required(cfg, "body_force")
    body_force = (float(bf_raw[0]), float(bf_raw[1]), float(bf_raw[2]))

    # --- Boundary conditions ---
    bc_cfg = get_required(cfg, "boundary_conditions")

    # Each force/diri entry now has 4 values: [side, x, y, z]
    force_sides = [
        (int(s[0]), float(s[1]), float(s[2]), float(s[3]))
        for s in get_optional(bc_cfg, "force_sides", [])
    ]
    diri_sides = [
        (int(s[0]), float(s[1]), float(s[2]), float(s[3]))
        for s in get_optional(bc_cfg, "diri_sides", [])
    ]
    tfbc_sides = [int(s) for s in get_optional(bc_cfg, "tfbc_sides", [])]

    bc = _build_bc_config(force_sides, diri_sides, tfbc_sides)

    # --- Output path ---
    out_path_cfg = get_optional(cfg, "simulation.json_path", "result.json")
    out_path     = resolve_out_path(out_path_cfg)

    # --- Run solver ---
    # Cubic grid: mcp = ncp = lcp, p = q = r = degree
    mcp = lcp = ncp
    p = q = r = degree

    u, v, w, sigma_vm, meta = solve_elasticity_collocation_3d(
        p=p, q=q, r=r,
        mcp=mcp, ncp=ncp, lcp=lcp,
        E=E, nu=nu,
        bc=bc,
        body_force=body_force,
    )

    # --- Range check ---
    print("\n=== RANGE CHECK ===")
    print(f"u: min = {float(u.min()):+.6e},  max = {float(u.max()):+.6e}")
    print(f"v: min = {float(v.min()):+.6e},  max = {float(v.max()):+.6e}")
    print(f"w: min = {float(w.min()):+.6e},  max = {float(w.max()):+.6e}")
    print(f"sigma_vm: min = {float(sigma_vm.min()):+.6e},  max = {float(sigma_vm.max()):+.6e}")
    print("===================\n")

    # --- Export ---
    X0 = meta["X0"]   # (mcp, ncp, lcp)
    Y0 = meta["Y0"]
    Z0 = meta["Z0"]

    # Reshape displacements to (mcp, ncp, lcp) in column-major order
    U = u.reshape(mcp, ncp, lcp, order='F')
    V = v.reshape(mcp, ncp, lcp, order='F')
    W = w.reshape(mcp, ncp, lcp, order='F')

    # Deformed positions
    X_def = X0 + U
    Y_def = Y0 + V
    Z_def = Z0 + W

    # Flatten back for JSON export (column-major)
    xd = X_def.reshape(-1, order='F')
    yd = Y_def.reshape(-1, order='F')
    zd = Z_def.reshape(-1, order='F')
    uf = U.reshape(-1, order='F')
    vf = V.reshape(-1, order='F')
    wf = W.reshape(-1, order='F')
    sf = sigma_vm.reshape(-1, order='F')

    # stdCollDisplacement is what C++ loadDisplacements() reads: shape (N, 3)
    new_payload = {
        "stdCollCtrlPts":      np.column_stack([xd, yd, zd]).tolist(),  # [[x,y,z], ...]
        "stdCollDeformations": np.column_stack([uf, vf, wf]).tolist(),  # [[u,v,w], ...]
        "stdCollStresses":     sf.tolist(),                              # [sigma_vm, ...]
        "stdCollDisplacement": np.column_stack([uf, vf, wf]).tolist(),  # [[u,v,w], ...] -- read by C++ loadDisplacements()
    }

    # --- Update existing JSON without deleting other keys ---
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

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Updated JSON keys in: {out_path}")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# import os
# import re
# import json
# import numpy as np

# from std_collocation_python.iga_collocation import solve_elasticity_collocation_2d
# from std_collocation_python.apply_bc import BC, BCConfig

# # Side convention (match C++):
# # 1 = left, 2 = right, 3 = bottom, 4 = top
# _SIDE_TO_NAME = {1: "left", 2: "right", 3: "bottom", 4: "top"}


# def load_json_with_line_comments(path: str) -> dict:
#     """
#     Loads JSON that may contain // line comments.
#     NOTE: This does NOT fix invalid JSON (e.g. missing commas, trailing commas).
#     """
#     with open(path, "r") as f:
#         txt = f.read()
#     txt = re.sub(r"//.*?$", "", txt, flags=re.MULTILINE)  # remove // comments
#     return json.loads(txt)


# def get_required(d: dict, path: str):
#     """Get nested key 'a.b.c' or raise KeyError."""
#     cur = d
#     for key in path.split("."):
#         if not isinstance(cur, dict) or key not in cur:
#             raise KeyError(f"Missing required config key: '{path}'")
#         cur = cur[key]
#     return cur


# def get_optional(d: dict, path: str, default):
#     """Get nested key 'a.b.c' or return default."""
#     cur = d
#     for key in path.split("."):
#         if not isinstance(cur, dict) or key not in cur:
#             return default
#         cur = cur[key]
#     return cur


# def _build_bc_config(force_sides, diri_sides, tfbc_sides) -> BCConfig:
#     """
#     Build BCConfig from lists:
#       force_sides: [(side, tx, ty), ...]
#       diri_sides:  [(side, ux, uy), ...]
#       tfbc_sides:  [side, ...]
#     Dirichlet always wins.
#     """
#     bc_map = {
#         "left": BC("free", np.array([0.0, 0.0])),
#         "right": BC("free", np.array([0.0, 0.0])),
#         "bottom": BC("free", np.array([0.0, 0.0])),
#         "top": BC("free", np.array([0.0, 0.0])),
#     }

#     # traction-free: explicit free (no change, but keeps intent)
#     for s in tfbc_sides:
#         name = _SIDE_TO_NAME[int(s)]
#         bc_map[name] = BC("free", np.array([0.0, 0.0]))

#     # Neumann
#     for (s, tx, ty) in force_sides:
#         name = _SIDE_TO_NAME[int(s)]
#         bc_map[name] = BC("neumann", np.array([float(tx), float(ty)], dtype=float))

#     # Dirichlet overrides
#     for (s, ux, uy) in diri_sides:
#         name = _SIDE_TO_NAME[int(s)]
#         bc_map[name] = BC("dirichlet", np.array([float(ux), float(uy)], dtype=float))

#     return BCConfig(
#         left=bc_map["left"],
#         right=bc_map["right"],
#         bottom=bc_map["bottom"],
#         top=bc_map["top"],
#     )


# def resolve_out_path(out_path_from_cfg: str) -> str:
#     """
#     If config gives an absolute path -> use it.
#     If it gives a relative path -> interpret relative to run_std_coll.py directory.
#     """
#     if os.path.isabs(out_path_from_cfg):
#         return out_path_from_cfg
#     return os.path.join(os.path.dirname(__file__), out_path_from_cfg)


# def main():
#     # Config is always next to this script
#     config_path = os.path.join(os.path.dirname(__file__), "sim_config.json")
#     cfg = load_json_with_line_comments(config_path)

#     # --- required: material ---
#     E = float(get_required(cfg, "material.young_modulus"))
#     nu = float(get_required(cfg, "material.poisson_ratio"))

#     # --- required: spline ---
#     ncp = int(get_required(cfg, "spline.nr_ctrl_pts"))
#     degree = int(get_required(cfg, "spline.degree"))

#     # --- required: body force ---
#     body_force = get_required(cfg, "body_force")
#     body_force = (float(body_force[0]), float(body_force[1]))

#     # --- boundary conditions ---
#     bc_cfg = get_required(cfg, "boundary_conditions")
#     force_sides = [(int(s[0]), float(s[1]), float(s[2])) for s in get_optional(bc_cfg, "force_sides", [])]
#     diri_sides = [(int(s[0]), float(s[1]), float(s[2])) for s in get_optional(bc_cfg, "diri_sides", [])]
#     tfbc_sides = [int(s) for s in get_optional(bc_cfg, "tfbc_sides", [3, 4])]

#     bc = _build_bc_config(force_sides, diri_sides, tfbc_sides)

#     # --- output path ---
#     out_path_cfg = get_optional(cfg, "simulation.json_path", "result.json")
#     out_path = resolve_out_path(out_path_cfg)

#     # --- run solver ---
#     mcp = ncp
#     p = degree
#     q = degree

#     u, v, sigma_vm, meta = solve_elasticity_collocation_2d(
#         p=p, q=q, mcp=mcp, ncp=ncp,
#         E=E, nu=nu,
#         bc=bc,
#         body_force=body_force
#     )

#     # quick range check
#     print("\n=== RANGE CHECK ===")
#     print(f"u: min = {float(u.min()):+.6e}, max = {float(u.max()):+.6e}")
#     print(f"v: min = {float(v.min()):+.6e}, max = {float(v.max()):+.6e}")
#     print(f"sigma_vm: min = {float(sigma_vm.min()):+.6e}, max = {float(sigma_vm.max()):+.6e}")
#     print("===================\n")

#     # --- export payload (MATLAB compatible order="F") ---
#     X0 = meta["X0"]
#     Y0 = meta["Y0"]

#     U = u.reshape(mcp, ncp, order="F")
#     V = v.reshape(mcp, ncp, order="F")

#     X_def = X0 + U
#     Y_def = Y0 + V

#     xd = X_def.reshape(-1, order="F")
#     yd = Y_def.reshape(-1, order="F")
#     uf = U.reshape(-1, order="F")
#     vf = V.reshape(-1, order="F")
#     sf = sigma_vm.reshape(-1, order="F")

#     new_payload = {
#         "stdCollCtrlPts": np.column_stack([xd, yd]).tolist(),          # [[x,y], ...]
#         "stdCollDeformations": np.column_stack([uf, vf]).tolist(),     # [[u,v], ...]
#         "stdCollStresses": sf.tolist()                                 # [sigma, ...]
#     }

#     # --- update existing json without deleting other keys ---
#     existing = {}
#     if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
#         try:
#             with open(out_path, "r") as f:
#                 existing = json.load(f)
#             if not isinstance(existing, dict):
#                 existing = {"_previous_content": existing}
#         except json.JSONDecodeError:
#             existing = {"_previous_raw": "INVALID_JSON"}

#     existing.update(new_payload)

#     # Ensure target directory exists (if user specified one)
#     os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

#     with open(out_path, "w") as f:
#         json.dump(existing, f, indent=2)

#     print(f"Updated JSON keys in: {out_path}")


# if __name__ == "__main__":
#     main()
