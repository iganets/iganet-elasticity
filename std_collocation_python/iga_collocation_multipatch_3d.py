# std_collocation_python/iga_collocation_multipatch_3d.py
"""
Classical isogeometric collocation reference solution for a chain of N
axis-aligned unit cubes glued face-to-face along x.

This generalizes solve_elasticity_collocation_3d() (single unit cube) to an
arbitrary, config-driven number of patches, replacing the old
collocation_reference_3D_multipatch_parametrized.cxx, which was hardcoded to
exactly two cubes and solved a penalty-style (MSE) objective via matrix-free
CG instead of assembling and solving a classical collocation system
directly.

Geometry
--------
Patch p (0-based) occupies x in [p*cube_size, (p+1)*cube_size], y,z in
[0,1]. All patches share the same degree/nr_ctrl_pts (isoparametric,
conforming discretization), so neighboring patches' shared face control
points coincide exactly and can be merged into single global DOFs.

Side convention (matches the single-patch solver and the C++ examples):
  1 = x=0 (left of the whole chain, i.e. patch 0's left face)
  2 = x=N*cube_size (right of the whole chain, i.e. the last patch's right face)
  3 = y=0 (bottom), 4 = y=1 (top)      -- applies to every patch
  5 = z=0 (front),  6 = z=1 (back)     -- applies to every patch

Coupling at interior interfaces
--------------------------------
- Displacement continuity (C0) is built in structurally: the shared
  control points between two neighboring patches are the same global DOF,
  not two independent DOFs tied together by a penalty term.
- Traction/stress equilibrium across the interface is enforced as a hard
  equation, replacing the interior PDE row that a genuinely interior point
  would otherwise get at that location:
      sigma(u_left)  * n_left  +  sigma(u_right) * n_right  = 0
  where n_left = +x (outward normal of the left patch's right face) and
  n_right = -x (outward normal of the right patch's left face). Both
  patches contribute a traction_local_block() to the *same* 3 global
  equations; scipy's coo_matrix sums duplicate (row, col) entries, which is
  exactly the superposition this equilibrium condition needs.

This keeps the assembled system square (one set of 3 equations per
distinct global control point) and solves it directly with spsolve -- a
real system-of-equations reference solution, not a trained network and not
a least-squares/penalty fit.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .bspline import greville_abscissae, bspline_all_basis_and_ders
from .mapping_3d import mapping3d
from .apply_bc_3d import (
    BC, BCConfig3D,
    boundary_flags, face_normal, pick_active_face,
    traction_local_block, interior_pde_local_block,
)
from .iga_collocation_3d import open_uniform_knots


def _local_node_index(i: int, j: int, k: int, mcp: int, ncp: int) -> int:
    """0-based local node index within one patch (i fastest, then j, then
    k), matching the F-order flattening used throughout this package."""
    return (i - 1) + mcp * (j - 1) + mcp * ncp * (k - 1)


def solve_elasticity_collocation_multipatch_chain_3d(
        num_patches: int,
        p: int, q: int, r: int,
        mcp: int, ncp: int, lcp: int,
        E: float = 210.0,
        nu: float = 0.25,
        bc: BCConfig3D = None,
        body_force: tuple[float, float, float] = (0.0, 0.0, 0.0),
        cube_size: float = 1.0):
    """
    Solve linear elasticity by direct isogeometric collocation on a chain
    of `num_patches` unit cubes (cube_size each) glued along x.

    Returns
    -------
    patches : list of dict, one per cube in chain order (index 0 = leftmost),
        each with keys:
          "csi", "eta", "zeta"  : 1D knot vectors (identical for every patch)
          "X0", "Y0", "Z0"      : undeformed control-point grids, shape (mcp,ncp,lcp)
          "u", "v", "w"         : displacement fields, shape (mcp,ncp,lcp)
          "sigma_vm"            : von Mises stress, shape (mcp,ncp,lcp)
    meta : dict with "mu", "lam", "num_patches", "cube_size",
        "n_global_nodes", "ndof".
    """
    if bc is None:
        raise ValueError("bc must be provided (BCConfig3D). Got None.")
    if num_patches < 1:
        raise ValueError("num_patches must be >= 1")

    nnod = mcp * ncp * lcp

    # --- Knot vectors / basis functions: identical for every patch, only
    # the physical control-point positions differ by an x-offset ---
    csi = open_uniform_knots(mcp, p)
    eta = open_uniform_knots(ncp, q)
    zeta = open_uniform_knots(lcp, r)

    grev_x = greville_abscissae(csi, p, mcp)
    grev_y = greville_abscissae(eta, q, ncp)
    grev_z = greville_abscissae(zeta, r, lcp)

    Ax = bspline_all_basis_and_ders(csi, p, grev_x, n_deriv=2)
    Ay = bspline_all_basis_and_ders(eta, q, grev_y, n_deriv=2)
    Az = bspline_all_basis_and_ders(zeta, r, grev_z, n_deriv=2)

    NN = np.zeros((3 * mcp, mcp), dtype=float)
    MM = np.zeros((3 * ncp, ncp), dtype=float)
    LL = np.zeros((3 * lcp, lcp), dtype=float)
    for ii in range(mcp):
        NN[3 * ii:3 * ii + 3, :] = Ax[ii, :, :].T
    for jj in range(ncp):
        MM[3 * jj:3 * jj + 3, :] = Ay[jj, :, :].T
    for kk in range(lcp):
        LL[3 * kk:3 * kk + 3, :] = Az[kk, :, :].T

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # --- Per-patch geometry (control points at Greville abscissae) ---
    patch_X0, patch_Y0, patch_Z0 = [], [], []
    patch_xflat, patch_yflat, patch_zflat, patch_wflat = [], [], [], []
    for pidx in range(num_patches):
        x_offset = pidx * cube_size
        # All three directions are scaled by cube_size (not just x): each
        # patch must remain an isotropic cube. Scaling x only (leaving y,z
        # at [0,1]) turns patches into aspect-ratio-distorted boxes for
        # cube_size != 1, which severely destabilizes the interface
        # traction-continuity equations (verified empirically).
        X0, Y0, Z0 = np.meshgrid(
            x_offset + cube_size * grev_x, cube_size * grev_y, cube_size * grev_z,
            indexing='ij')
        patch_X0.append(X0)
        patch_Y0.append(Y0)
        patch_Z0.append(Z0)
        patch_xflat.append(X0.reshape(-1, order='F'))
        patch_yflat.append(Y0.reshape(-1, order='F'))
        patch_zflat.append(Z0.reshape(-1, order='F'))
        patch_wflat.append(np.ones(nnod, dtype=float))

    # --- Global DOF numbering: merge shared interface control points so
    # that displacement continuity is exact/structural, not enforced by a
    # constraint equation ---
    patch_global_ids = [np.full(nnod, -1, dtype=int) for _ in range(num_patches)]
    next_gid = 0
    for pidx in range(num_patches):
        for kk in range(1, lcp + 1):
            for jj in range(1, ncp + 1):
                for ii in range(1, mcp + 1):
                    local = _local_node_index(ii, jj, kk, mcp, ncp)
                    if ii == 1 and pidx > 0:
                        prev_local = _local_node_index(mcp, jj, kk, mcp, ncp)
                        patch_global_ids[pidx][local] = patch_global_ids[pidx - 1][prev_local]
                    else:
                        patch_global_ids[pidx][local] = next_gid
                        next_gid += 1
    n_global_nodes = next_gid
    ndof = 3 * n_global_nodes

    rows_list, cols_list, vals_list = [], [], []
    f_gl = np.zeros(ndof, dtype=float)
    visited = np.zeros(n_global_nodes, dtype=bool)

    local_cols = np.arange(3 * nnod)
    local_col_node = local_cols // 3
    local_col_component = local_cols % 3

    def scatter(k_local: np.ndarray, row_ids: np.ndarray, gids: np.ndarray) -> None:
        global_cols = 3 * gids[local_col_node] + local_col_component
        rr, cc = np.nonzero(k_local)
        rows_list.append(row_ids[rr])
        cols_list.append(global_cols[cc])
        vals_list.append(k_local[rr, cc])

    east = np.array([1.0, 0.0, 0.0])
    west = np.array([-1.0, 0.0, 0.0])

    for pidx in range(num_patches):
        x_flat, y_flat, z_flat, w_flat = (
            patch_xflat[pidx], patch_yflat[pidx], patch_zflat[pidx], patch_wflat[pidx])
        gids = patch_global_ids[pidx]

        for kk in range(1, lcp + 1):
            for jj in range(1, ncp + 1):
                for ii in range(1, mcp + 1):
                    local = _local_node_index(ii, jj, kk, mcp, ncp)
                    gid = gids[local]

                    on_left, on_right, on_bottom, on_top, on_front, on_back, _ = \
                        boundary_flags(ii, jj, kk, mcp, ncp, lcp)
                    # x-faces that adjoin another patch are interior to the
                    # assembly, even though they are boundary faces of this
                    # one patch's own parametrization.
                    if on_left and pidx > 0:
                        on_left = False
                    if on_right and pidx < num_patches - 1:
                        on_right = False
                    is_exterior = on_left or on_right or on_bottom or on_top or on_front or on_back
                    is_interface = (not is_exterior) and (
                        (ii == mcp and pidx < num_patches - 1) or (ii == 1 and pidx > 0))

                    row_ids = 3 * gid + np.array([0, 1, 2])

                    if is_exterior:
                        if visited[gid]:
                            continue
                        visited[gid] = True

                        face = pick_active_face(on_left, on_right, on_bottom, on_top,
                                                on_front, on_back, bc)
                        bc_obj = getattr(bc, face)
                        _, dN, _ = mapping3d(ii, jj, kk, nnod, NN, MM, LL,
                                             x_flat, y_flat, z_flat, w_flat)

                        if bc_obj.type == "dirichlet":
                            k_local = np.zeros((3, 3 * nnod), dtype=float)
                            k_local[0, 3 * local + 0] = 1.0
                            k_local[1, 3 * local + 1] = 1.0
                            k_local[2, 3 * local + 2] = 1.0
                            f_gl[row_ids] = bc_obj.value
                        else:
                            n = face_normal(face)
                            t = bc_obj.value if bc_obj.type == "neumann" else np.zeros(3)
                            k_local = traction_local_block(dN, n, mu, lam)
                            f_gl[row_ids] = t
                        scatter(k_local, row_ids, gids)

                    elif is_interface:
                        # Contributed once per adjoining patch; both land on
                        # the same 3 global rows and are summed by coo_matrix.
                        normal = east if ii == mcp else west
                        _, dN, _ = mapping3d(ii, jj, kk, nnod, NN, MM, LL,
                                             x_flat, y_flat, z_flat, w_flat)
                        k_local = traction_local_block(dN, normal, mu, lam)
                        # f_gl stays 0: pure equilibrium, no external interface load.
                        scatter(k_local, row_ids, gids)

                    else:
                        if visited[gid]:
                            continue
                        visited[gid] = True

                        _, dN, ddN = mapping3d(ii, jj, kk, nnod, NN, MM, LL,
                                               x_flat, y_flat, z_flat, w_flat)
                        # div(sigma) + f = 0, so the right-hand side of the
                        # collocation row carries minus the body force.
                        f_gl[row_ids] = -np.asarray(body_force, dtype=float)
                        k_local = interior_pde_local_block(dN, ddN, mu, lam)
                        scatter(k_local, row_ids, gids)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    sol = spsolve(K, f_gl)

    u_global = sol[0::3]
    v_global = sol[1::3]
    w_global = sol[2::3]

    patches = []
    for pidx in range(num_patches):
        gids = patch_global_ids[pidx]
        u = u_global[gids]
        v = v_global[gids]
        w = w_global[gids]

        x_flat, y_flat, z_flat, w_flat = (
            patch_xflat[pidx], patch_yflat[pidx], patch_zflat[pidx], patch_wflat[pidx])

        sigma_vm = np.zeros((mcp, ncp, lcp), dtype=float)
        for kk in range(1, lcp + 1):
            for jj in range(1, ncp + 1):
                for ii in range(1, mcp + 1):
                    _, dN, _ = mapping3d(ii, jj, kk, nnod, NN, MM, LL,
                                         x_flat, y_flat, z_flat, w_flat)
                    ux = dN[0, :] @ u
                    uy = dN[1, :] @ u
                    uz = dN[2, :] @ u
                    vx = dN[0, :] @ v
                    vy = dN[1, :] @ v
                    vz = dN[2, :] @ v
                    wx = dN[0, :] @ w
                    wy = dN[1, :] @ w
                    wz = dN[2, :] @ w

                    exx, eyy, ezz = ux, vy, wz
                    exy = 0.5 * (uy + vx)
                    exz = 0.5 * (uz + wx)
                    eyz = 0.5 * (vz + wy)

                    tr_e = exx + eyy + ezz
                    sxx = lam * tr_e + 2 * mu * exx
                    syy = lam * tr_e + 2 * mu * eyy
                    szz = lam * tr_e + 2 * mu * ezz
                    sxy = 2 * mu * exy
                    sxz = 2 * mu * exz
                    syz = 2 * mu * eyz

                    sigma_vm[ii - 1, jj - 1, kk - 1] = np.sqrt(0.5 * (
                        (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
                        + 6.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)))

        patches.append(dict(
            csi=csi, eta=eta, zeta=zeta,
            X0=patch_X0[pidx], Y0=patch_Y0[pidx], Z0=patch_Z0[pidx],
            u=u.reshape(mcp, ncp, lcp, order='F'),
            v=v.reshape(mcp, ncp, lcp, order='F'),
            w=w.reshape(mcp, ncp, lcp, order='F'),
            sigma_vm=sigma_vm,
        ))

    meta = dict(
        mu=mu, lam=lam, num_patches=num_patches, cube_size=cube_size,
        n_global_nodes=n_global_nodes, ndof=ndof, degree=(p, q, r),
    )
    return patches, meta
