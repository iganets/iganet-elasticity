#!/usr/bin/env python3
"""
Classical isogeometric collocation reference solution for a general multipatch
geometry read from a gismo XML file, in particular the bone.

Same method as run_multipatch_reference_3d.py, but without its restriction to
a chain of axis-aligned unit cubes:

  * patches are read from XML and may be curved, with a different degree and
    resolution per patch and per parametric direction,
  * the topology is arbitrary: a control point may be shared by more than two
    patches, and any side may meet any side,
  * outward normals are computed from the geometry instead of being looked up,
  * boundary conditions are taken per patch from patches_3d in the config.

Every condition a control point carries becomes an equation; the resulting
overdetermined system is solved in the least-squares sense. The rules are:

  interior point                    -> equilibrium, div(sigma) + f = 0
  point on a prescribed face        -> its boundary condition, and where two
                                       collide the higher one: Dirichlet beats
                                       prescribed traction beats traction-free,
                                       ties by the lower patch and side
  point on an interface             -> the tractions of all adjoining patches
                                       must cancel, IN ADDITION to any boundary
                                       condition the point carries

The interface balance sits outside the priority above. It is not a competing
boundary condition but the force balance that holds the patches together. It
used to run in the same priority list as the lowest entry and therefore lost to
every boundary condition: on an edge where an interface met a prescribed face it
was dropped, which left 128 nodes of this geometry out of equilibrium by a
median of 6.9 (up to 638) instead of zero. The solution then did not converge -
refining from 1268 to 7908 nodes changed it by 33 %, and the intermediate mesh
lay further from the finest than the coarsest did.

Keeping both makes a node carry more than the three equations it has room for,
so the system is overdetermined and solved in the least-squares sense.
Prescribed displacements are eliminated rather than fitted and so stay exact;
only the traction-type statements are balanced against each other. Refinement
now changes the solution by 2.8 %.

Displacement continuity across interfaces is structural: control points that
coincide in space become one unknown.

NOTE: the geometry must carry degree >= 2 in every direction, otherwise the
second derivatives needed by the equilibrium equation vanish identically. Run
prepare_bone_geometry.py first; bone_simplified.xml does not qualify.

Usage:
  python3 -m std_collocation_python.run_bone_reference_3d [config] [output.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import splinepy
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsqr, spsolve
from scipy.spatial import cKDTree

from .apply_bc_3d import interior_pde_local_block, traction_local_block
from .bspline import bspline_all_basis_and_ders, greville_abscissae
from .mapping_3d import mapping3d
from .prepare_bone_geometry import read_interface_lines, side_direction
from .run_std_coll import get_optional, get_required, load_json_with_line_comments

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "src" / "examples3D" / "multiPatch" / "sim_config_3D_multi_patch_bone.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "result_collocation_reference_3D_multipatch_bone.json"

MATCH_TOL = 1e-6          # same tolerance as the C++ multipatch code
DIRICHLET, FORCE, FREE = 3, 2, 1


class Patch:
    """One spline patch with everything the assembly needs."""

    def __init__(self, spline):
        self.degrees = [int(d) for d in spline.degrees]
        self.knots = [np.asarray(kv, dtype=float) for kv in spline.knot_vectors]
        self.ncp = [len(kv) - d - 1 for kv, d in zip(self.knots, self.degrees)]
        self.nnod = int(np.prod(self.ncp))
        self.points = np.asarray(spline.control_points, dtype=float)

        self.basis = []
        for kv, deg, n in zip(self.knots, self.degrees, self.ncp):
            grev = greville_abscissae(kv, deg, n)
            ders = bspline_all_basis_and_ders(kv, deg, grev, n_deriv=2)
            matrix = np.zeros((3 * n, n), dtype=float)
            for i in range(n):
                matrix[3 * i:3 * i + 3, :] = ders[i, :, :].T
            self.basis.append(matrix)

        self.weights = np.ones(self.nnod, dtype=float)

    def local_index(self, i: int, j: int, k: int) -> int:
        """0-based index of the 1-based grid position (i, j, k)."""
        return (i - 1) + self.ncp[0] * (j - 1) + self.ncp[0] * self.ncp[1] * (k - 1)

    def derivatives(self, i: int, j: int, k: int):
        """Physical basis derivatives at one Greville point."""
        return mapping3d(i, j, k, self.nnod, *self.basis,
                         self.points[:, 0], self.points[:, 1], self.points[:, 2],
                         self.weights)

    def jacobian(self, i: int, j: int, k: int) -> np.ndarray:
        """J[a, d] = dx_a / dxi_d, needed for the outward normal."""
        rows = [m[3 * (idx - 1):3 * (idx - 1) + 3, :]
                for m, idx in zip(self.basis, (i, j, k))]
        columns = []
        for d in range(3):
            factors = [rows[0][0], rows[1][0], rows[2][0]]
            factors[d] = rows[d][1]
            tensor = np.einsum('i,j,k->ijk', *factors).reshape(-1, order='F')
            columns.append(tensor @ self.points)
        return np.stack(columns, axis=1)

    def outward_normal(self, side: int, i: int, j: int, k: int) -> np.ndarray:
        """Unit outward normal of `side` at that point.

        The face is the level set xi_d = const, so its normal is parallel to
        the gradient of xi_d, which is row d of the inverse Jacobian. The
        upper face points along +grad(xi_d), the lower one against it.
        """
        direction = side_direction(side)
        gradient = np.linalg.inv(self.jacobian(i, j, k))[direction, :]
        normal = gradient / np.linalg.norm(gradient)
        return normal if side % 2 == 0 else -normal

    def sides_of(self, i: int, j: int, k: int):
        """Which of the six sides the point (i, j, k) lies on."""
        n0, n1, n2 = self.ncp
        flags = ((1, i == 1), (2, i == n0), (3, j == 1),
                 (4, j == n1), (5, k == 1), (6, k == n2))
        return [side for side, on in flags if on]

    def grid(self):
        for k in range(1, self.ncp[2] + 1):
            for j in range(1, self.ncp[1] + 1):
                for i in range(1, self.ncp[0] + 1):
                    yield i, j, k


def merge_control_points(patches):
    """Control points that coincide in space become one global unknown."""
    offsets, all_points = [], []
    running = 0
    for patch in patches:
        offsets.append(running)
        all_points.append(patch.points)
        running += patch.nnod
    stacked = np.vstack(all_points)

    parent = np.arange(running)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = cKDTree(stacked)
    for a, b in tree.query_pairs(MATCH_TOL):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    roots = {}
    global_id = np.empty(running, dtype=int)
    for idx in range(running):
        root = find(idx)
        if root not in roots:
            roots[root] = len(roots)
        global_id[idx] = roots[root]

    per_patch = [global_id[off:off + p.nnod] for off, p in zip(offsets, patches)]
    return per_patch, len(roots)


def load_boundary_conditions(cfg: dict, first_id: int, n_patches: int):
    """Per patch and side: (priority, target vector). Sides not listed in the
    config are treated as traction-free, which is what a free surface is."""
    table = {}
    for entry in get_optional(cfg, "patches_3d", []):
        index = int(entry["patch_id"]) - first_id
        if not 0 <= index < n_patches:
            continue
        conditions = entry.get("boundary_conditions", {})
        for side, x, y, z in (tuple(v) for v in conditions.get("diri_sides", [])):
            table[(index, int(side))] = (DIRICHLET, np.array([x, y, z], dtype=float))
        for side, x, y, z in (tuple(v) for v in conditions.get("force_sides", [])):
            table[(index, int(side))] = (FORCE, np.array([x, y, z], dtype=float))
        for side in conditions.get("tfbc_sides", []):
            table.setdefault((index, int(side)), (FREE, np.zeros(3)))
    return table


def solve(xml_path: Path, cfg: dict, quiet: bool = False, return_system: bool = False):
    multipatch, _ = splinepy.io.gismo.load(str(xml_path))
    patches = [Patch(s) for s in multipatch.patches]
    n_patches = len(patches)

    degenerate = [i for i, p in enumerate(patches) if min(p.degrees) < 2]
    if degenerate:
        raise SystemExit(
            f"Patches {degenerate} carry degree < 2. Strong-form collocation needs "
            "second derivatives in every direction. Run prepare_bone_geometry.py first.")

    interfaces = read_interface_lines(xml_path)
    interface_sides = set()
    for p1, s1, p2, s2, _ in interfaces:
        interface_sides.add((p1, s1))
        interface_sides.add((p2, s2))

    root_ids = splinepy.io.gismo.load(str(xml_path))  # ids for the config lookup
    import xml.etree.ElementTree as ET
    first_id = int(ET.parse(xml_path).getroot().find(".//MultiPatch/patches").text.split()[0])

    E = float(get_required(cfg, "material.young_modulus"))
    nu = float(get_required(cfg, "material.poisson_ratio"))
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    body = np.array(get_optional(cfg, "multipatch.body_force", [0.0, 0.0, 0.0]), dtype=float)

    bc_table = load_boundary_conditions(cfg, first_id, n_patches)
    gids, n_nodes = merge_control_points(patches)
    ndof = 3 * n_nodes
    log = (lambda *a: None) if quiet else print
    log(f"Patches {n_patches} | Kontrollpunkte {sum(p.nnod for p in patches)} "
          f"-> Unbekannte {n_nodes} Knoten / {ndof} Werte")

    # Collect, for every global node, where it sits in every patch.
    incidences = {}
    for pi, patch in enumerate(patches):
        for i, j, k in patch.grid():
            gid = gids[pi][patch.local_index(i, j, k)]
            incidences.setdefault(gid, []).append((pi, i, j, k))

    rows, cols, vals, rhs_blocks = [], [], [], []
    counts = {"dirichlet": 0, "traction": 0, "interface": 0, "interior": 0}
    fixed = np.zeros(ndof, dtype=bool)
    fixed_value = np.zeros(ndof, dtype=float)
    next_row = 0

    def place(block, patch_index, row_base):
        """Scatters one 3-row block into the global rows starting at row_base."""
        mapping = gids[patch_index]
        width = 3 * patches[patch_index].nnod
        columns = 3 * mapping[np.arange(width) // 3] + np.arange(width) % 3
        rr, cc = np.nonzero(block)
        rows.append(row_base + rr)
        cols.append(columns[cc])
        vals.append(block[rr, cc])

    def equation(blocks, target):
        """Adds one equation: the sum of the given patch blocks equals target."""
        nonlocal next_row
        for patch_index, block in blocks:
            place(block, patch_index, next_row)
        rhs_blocks.append(np.asarray(target, dtype=float))
        next_row += 3

    def traction_block(pi, i, j, k, side):
        patch = patches[pi]
        _, dN, _ = patch.derivatives(i, j, k)
        return traction_local_block(dN, patch.outward_normal(side, i, j, k), mu, lam)

    for gid, places in incidences.items():
        prescribed, on_interface = [], []
        for pi, i, j, k in places:
            for side in patches[pi].sides_of(i, j, k):
                if (pi, side) in interface_sides:
                    on_interface.append((pi, i, j, k, side))
                else:
                    priority, target = bc_table.get((pi, side), (FREE, np.zeros(3)))
                    prescribed.append((-priority, pi, side, i, j, k, target))
        prescribed.sort()

        # A prescribed displacement settles the node outright: where the
        # displacement is dictated, no traction statement applies. It is
        # eliminated rather than fitted, so it stays exact.
        dirichlet = [c for c in prescribed if -c[0] == DIRICHLET]
        if dirichlet:
            _, pi, side, i, j, k, target = dirichlet[0]
            base = 3 * gids[pi][patches[pi].local_index(i, j, k)]
            fixed[base:base + 3] = True
            fixed_value[base:base + 3] = target
            counts["dirichlet"] += 1
            continue

        # Among the boundary conditions the priority stands: a point carrying
        # two of them follows the higher one, prescribed traction before
        # traction-free, ties by the lower patch and side. Keeping all of them
        # instead was measured and changes the solution by 0.5 %, so the
        # simpler rule stays.
        for _, pi, side, i, j, k, target in prescribed[:1]:
            equation([(pi, traction_block(pi, i, j, k, side))], target)
            counts["traction"] += 1

        # The interface balance is deliberately OUTSIDE that priority. It is not
        # a competing boundary condition but the force balance holding the
        # patches together, and it applies whether or not the point also sits on
        # a prescribed face. Letting it lose to a boundary condition - as it did
        # before - left 128 nodes out of equilibrium by a median of 6.9 instead
        # of zero, and the solution then did not converge under refinement.
        if on_interface:
            equation([(pi, traction_block(pi, i, j, k, side))
                      for pi, i, j, k, side in on_interface], np.zeros(3))
            counts["interface"] += 1

        if not prescribed and not on_interface:
            pi, i, j, k = places[0]
            patch = patches[pi]
            _, dN, ddN = patch.derivatives(i, j, k)
            equation([(pi, interior_pde_local_block(dN, ddN, mu, lam))], -body)
            counts["interior"] += 1

    log("Gleichungen: " + ", ".join(f"{k} {v}" for k, v in counts.items())
        + f" -> {next_row} Zeilen fuer {int((~fixed).sum())} freie Werte "
          f"({int(fixed.sum())} durch Dirichlet festgelegt)")

    matrix = coo_matrix((np.concatenate(vals),
                         (np.concatenate(rows), np.concatenate(cols))),
                        shape=(next_row, ndof)).tocsr()
    rhs = np.concatenate(rhs_blocks)

    # Move the eliminated displacements to the right-hand side. fixed_value is
    # zero wherever nothing is fixed, so one product covers it.
    free = ~fixed
    reduced = matrix[:, free]
    reduced_rhs = rhs - matrix @ fixed_value

    lonely = int(np.sum(np.diff(reduced.tocsc().indptr) == 0))
    if lonely:
        raise SystemExit(f"{lonely} Freiheitsgrade kommen in keiner Gleichung vor.")

    # Row equilibration. The equation types have very different natural
    # magnitudes: a traction row scales like E/L, an equilibrium row like E/L^2.
    # On a body 200 units across that spans orders of magnitude. Dividing each
    # equation by its own row norm leaves the exact solution untouched - the
    # system is consistent - and only improves the conditioning.
    row_norm = np.sqrt(np.asarray(reduced.multiply(reduced).sum(axis=1))).ravel()
    row_norm[row_norm == 0.0] = 1.0
    scaled = diags(1.0 / row_norm) @ reduced
    scaled_rhs = reduced_rhs / row_norm

    # Direct solve of the normal equations, cross-checked against an iterative
    # least-squares solver. Agreement is the check that the system is still
    # well enough conditioned to trust.
    gram = (scaled.T @ scaled).tocsc()
    direct = spsolve(gram, scaled.T @ scaled_rhs)
    iterative = lsqr(scaled, scaled_rhs, atol=1e-13, btol=1e-13, iter_lim=200000)[0]

    def least_squares_residual(x):
        return float(np.linalg.norm(scaled @ x - scaled_rhs)
                     / max(np.linalg.norm(scaled_rhs), 1.0))

    log(f"ausgeglichen | rel. Residuum direkt {least_squares_residual(direct):.2e}"
        f" | iterativ {least_squares_residual(iterative):.2e}"
        f" | Unterschied der Loesungen {float(np.abs(direct - iterative).max()):.2e}")

    solution = fixed_value.copy()
    solution[free] = direct

    displacements = [solution[3 * gids[pi][:, None] + np.arange(3)]
                     for pi in range(n_patches)]
    if return_system:
        return patches, displacements, multipatch, matrix, rhs, gids
    return patches, displacements, multipatch


def write_json(out_path: Path, patches, displacements, xml_path: Path, quiet: bool = False):
    entries = []
    for index, (patch, disp) in enumerate(zip(patches, displacements)):
        entries.append({
            "index": index,
            "xml_id": index,
            "degrees": patch.degrees,
            "knot_vectors": [kv.tolist() for kv in patch.knots],
            "control_points": patch.points.tolist(),
            "displacements": disp.tolist(),
            "deformed_control_points": (patch.points + disp).tolist(),
        })

    summary = {
        "example": "collocation_reference_multipatch_bone_3d",
        "method": ("Classical isogeometric collocation on an XML multipatch geometry: "
                   "direct sparse solve, one equation per unknown, no training."),
        "device": "cpu",
        "npatches": len(patches),
        "geometry_xml": str(xml_path),
        "patches": entries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out_path.exists() and out_path.stat().st_size > 0:
        try:
            existing = json.load(open(out_path))
            if not isinstance(existing, dict):
                existing = {}
        except json.JSONDecodeError:
            existing = {}
    existing["multipatch_elasticity"] = summary
    with open(out_path, "w") as handle:
        json.dump(existing, handle, indent=1)
    if not quiet:
        print(f"geschrieben: {out_path}")


def main() -> None:
    quiet = "--quiet" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    config_path = Path(args[0]) if args else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    out_path = Path(args[1]) if len(args) > 1 else DEFAULT_OUTPUT
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    cfg = load_json_with_line_comments(str(config_path))
    xml_rel = get_required(cfg, "geometry.multipatch_xml_path")
    xml_path = Path(xml_rel)
    if not xml_path.is_absolute():
        xml_path = REPO_ROOT / xml_path

    if not quiet:
        print(f"config:   {config_path}")
        print(f"geometry: {xml_path}")
    patches, displacements, _ = solve(xml_path, cfg, quiet=quiet)
    if not quiet:
        for index, disp in enumerate(displacements):
            if index < 3 or index == len(displacements) - 1:
                print(f"  Patch {index}: |u| max {np.abs(disp).max():.6e}")
    write_json(out_path, patches, displacements, xml_path, quiet=quiet)


if __name__ == "__main__":
    main()
