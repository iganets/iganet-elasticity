# std_collocation_python/iga_collocation_3d.py
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .bspline import greville_abscissae, bspline_all_basis_and_ders
from .mapping3d import mapping3d
from .apply_bc_3d import apply_bc_3d, BC, BCConfig3D


def open_uniform_knots(ncp: int, degree: int) -> np.ndarray:
    """Open uniform knot vector: p zeros, linspace(0,1,ncp-p+1), p ones."""
    inner = np.linspace(0.0, 1.0, ncp - degree + 1)
    return np.concatenate([np.zeros(degree), inner, np.ones(degree)])


def solve_elasticity_collocation_3d(
        p: int, q: int, r: int,
        mcp: int, ncp: int, lcp: int,
        E: float = 210.0,
        nu: float = 0.25,
        bc: BCConfig3D = None,
        body_force: tuple[float, float, float] = (0.0, 0.0, 0.0),
        weights: np.ndarray | None = None):
    """
    3D isogeometric collocation solver for linear elasticity on a unit cube.

    Grid layout (column-major / Fortran order, matching C++ convention):
      index = i + mcp*j + mcp*ncp*k,  i in [0,mcp), j in [0,ncp), k in [0,lcp)
      i -> xi   (x-direction, side 1/2)
      j -> eta  (y-direction, side 3/4)
      k -> zeta (z-direction, side 5/6)

    Side convention (matches C++ code):
      1 = x=0 (left),   2 = x=1 (right)
      3 = y=0 (bottom), 4 = y=1 (top)
      5 = z=0 (front),  6 = z=1 (back)

    Returns
    -------
    u, v, w   : displacement arrays at control points, shape (nnod,)
    sigma_vm  : von Mises stress, shape (mcp, ncp, lcp)
    meta      : dict with knots, control-point positions, Lamé parameters
    """

    if bc is None:
        raise ValueError("bc must be provided (BCConfig3D). Got None.")

    if weights is None:
        w = np.ones((mcp, ncp, lcp), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(mcp, ncp, lcp)

    # --- Knot vectors (open uniform) ---
    csi  = open_uniform_knots(mcp, p)
    eta  = open_uniform_knots(ncp, q)
    zeta = open_uniform_knots(lcp, r)

    # --- Control points at Greville abscissae (unit-cube geometry) ---
    grev_x = greville_abscissae(csi,  p, mcp)
    grev_y = greville_abscissae(eta,  q, ncp)
    grev_z = greville_abscissae(zeta, r, lcp)

    # meshgrid with indexing='ij' so X0[i,j,k] = grev_x[i], etc.
    X0, Y0, Z0 = np.meshgrid(grev_x, grev_y, grev_z, indexing='ij')  # (mcp,ncp,lcp)

    nnod = mcp * ncp * lcp
    ndof = 3 * nnod

    # Flatten column-major (F-order): i varies fastest, then j, then k
    x_flat = X0.reshape(-1, order='F')
    y_flat = Y0.reshape(-1, order='F')
    z_flat = Z0.reshape(-1, order='F')
    w_flat = w.reshape(-1, order='F')

    # --- Precompute basis functions and derivatives at Greville points ---
    # bspline_all_basis_and_ders returns shape (n_points, n_basis, n_deriv+1)
    # A[ip, ib, k] = k-th derivative of basis ib at collocation point ip
    Ax = bspline_all_basis_and_ders(csi,  p, grev_x, n_deriv=2)  # (mcp, mcp, 3)
    Ay = bspline_all_basis_and_ders(eta,  q, grev_y, n_deriv=2)  # (ncp, ncp, 3)
    Az = bspline_all_basis_and_ders(zeta, r, grev_z, n_deriv=2)  # (lcp, lcp, 3)

    # Pack into dense matrices NN, MM, LL:
    # NN[3*i + d, :] = d-th derivative of all xi-basis functions at collocation point i
    NN = np.zeros((3*mcp, mcp), dtype=float)
    MM = np.zeros((3*ncp, ncp), dtype=float)
    LL = np.zeros((3*lcp, lcp), dtype=float)

    for ii in range(mcp):
        NN[3*ii + 0, :] = Ax[ii, :, 0]
        NN[3*ii + 1, :] = Ax[ii, :, 1]
        NN[3*ii + 2, :] = Ax[ii, :, 2]
    for jj in range(ncp):
        MM[3*jj + 0, :] = Ay[jj, :, 0]
        MM[3*jj + 1, :] = Ay[jj, :, 1]
        MM[3*jj + 2, :] = Ay[jj, :, 2]
    for kk in range(lcp):
        LL[3*kk + 0, :] = Az[kk, :, 0]
        LL[3*kk + 1, :] = Az[kk, :, 1]
        LL[3*kk + 2, :] = Az[kk, :, 2]

    # --- Lamé parameters (3D, no plane-strain assumption) ---
    mu  = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0*nu))

    # --- Assembly ---
    f_gl = np.zeros(ndof, dtype=float)
    k_rk = np.zeros((3, ndof), dtype=float)

    rows_list = []
    cols_list = []
    vals_list = []

    # Loop order: k (zeta/z), j (eta/y), i (xi/x)  -> column-major flattening
    icount = 1
    for k in range(1, lcp + 1):
        for j in range(1, ncp + 1):
            for i in range(1, mcp + 1):

                N, dN, ddN = mapping3d(
                    i, j, k, nnod,
                    NN, MM, LL,
                    x_flat, y_flat, z_flat, w_flat
                )

                k_rk, f_gl = apply_bc_3d(
                    i, j, k, mcp, ncp, lcp,
                    N, dN, ddN, icount,
                    k_rk, f_gl, mu, lam, bc, body_force
                )

                rr, cc = np.nonzero(k_rk)
                vv = k_rk[rr, cc]

                # Global row: collocation point icount contributes rows
                # [3*(icount-1), 3*(icount-1)+1, 3*(icount-1)+2]
                global_r = 3*(icount - 1) + rr   # rr in {0, 1, 2}

                rows_list.append(global_r)
                cols_list.append(cc)
                vals_list.append(vv)

                icount += 1

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.concatenate(vals_list)

    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    sol = spsolve(K, f_gl)

    u = sol[0::3]
    v = sol[1::3]
    w_sol = sol[2::3]

    # --- Von Mises stresses at collocation points ---
    sigma_vm = np.zeros((mcp, ncp, lcp), dtype=float)

    icount = 1
    for k in range(1, lcp + 1):
        for j in range(1, ncp + 1):
            for i in range(1, mcp + 1):

                _, dN, _ = mapping3d(
                    i, j, k, nnod,
                    NN, MM, LL,
                    x_flat, y_flat, z_flat, w_flat
                )

                # Displacement gradients
                ux = dN[0, :] @ u
                uy = dN[1, :] @ u
                uz = dN[2, :] @ u

                vx = dN[0, :] @ v
                vy = dN[1, :] @ v
                vz = dN[2, :] @ v

                wx = dN[0, :] @ w_sol
                wy = dN[1, :] @ w_sol
                wz = dN[2, :] @ w_sol

                # Strain tensor (symmetric)
                exx = ux
                eyy = vy
                ezz = wz
                exy = 0.5 * (uy + vx)
                exz = 0.5 * (uz + wx)
                eyz = 0.5 * (vz + wy)

                # Stress tensor
                tr_e = exx + eyy + ezz
                sxx = lam * tr_e + 2*mu * exx
                syy = lam * tr_e + 2*mu * eyy
                szz = lam * tr_e + 2*mu * ezz
                sxy = 2*mu * exy
                sxz = 2*mu * exz
                syz = 2*mu * eyz

                # Von Mises
                sigma_vm[i-1, j-1, k-1] = np.sqrt(0.5 * (
                    (sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2
                    + 6.0 * (sxy**2 + syz**2 + sxz**2)
                ))

                icount += 1

    meta = dict(
        csi=csi, eta=eta, zeta=zeta,
        X0=X0, Y0=Y0, Z0=Z0,
        weights=w,
        mu=mu, lam=lam
    )
    return u, v, w_sol, sigma_vm, meta
    
# # std_collocation_python/iga_collocation.py
# from __future__ import annotations
# import numpy as np
# from scipy.sparse import coo_matrix
# from scipy.sparse.linalg import spsolve

# from .bspline import greville_abscissae, bspline_all_basis_and_ders
# from .mapping2d import mapping2d
# from .apply_bc import apply_bc, BC, BCConfig


# def open_uniform_knots(ncp: int, degree: int) -> np.ndarray:
#     # MATLAB: [zeros(1,p), linspace(0,1,ncp-p+1), ones(1,p)]
#     inner = np.linspace(0.0, 1.0, ncp - degree + 1)
#     return np.concatenate([np.zeros(degree), inner, np.ones(degree)])


# def solve_elasticity_collocation_2d(p: int, q: int, mcp: int, ncp: int,
#                                    E: float = 210.0, nu: float = 0.25,
#                                    bc: BCConfig = None,
#                                    body_force: tuple[float, float] = (0.0, 0.0),
#                                    weights: np.ndarray | None = None):

#     """
#     Returns:
#       u, v: displacement vectors at control points (shape nnod,)
#       sigma_vm: von Mises on collocation/control points (shape (mcp,ncp))
#       meta: dict with knots, control points etc.
#     """
 
#     if bc is None:
#         raise ValueError("bc must be provided (BCConfig). Got None.")

#     if weights is None:
#         w = np.ones((mcp, ncp), dtype=float)
#     else:
#         w = np.asarray(weights, dtype=float).reshape(mcp, ncp)

#     # knots (open uniform)
#     csi = open_uniform_knots(mcp, p)
#     eta = open_uniform_knots(ncp, q)

#     # control points placed at Greville (like MATLAB)
#     grev_x = greville_abscissae(csi, p, mcp)
#     grev_y = greville_abscissae(eta, q, ncp)
#     X0, Y0 = np.meshgrid(grev_x, grev_y, indexing="ij")  # (mcp,ncp)

#     nnod = mcp * ncp
#     ndof = 2 * nnod

#     x = X0.reshape(-1, order='F')  # flatten in column-major order to match MATLAB
#     y = Y0.reshape(-1, order='F')
#     w_flat = w.reshape(-1, order='F')

#     # collocation points = Greville (MATLAB: aveknt(knots, p+1))
#     coll_csi = grev_x
#     coll_eta = grev_y

#     # precompute basis/derivatives at collocation points
#     # for mapping2d we mimic NN/MM layout: 3 rows per point i/j
#     Ax = bspline_all_basis_and_ders(csi, p, coll_csi, n_deriv=2)  # (mcp, mcp, 3)
#     Ay = bspline_all_basis_and_ders(eta, q, coll_eta, n_deriv=2)  # (ncp, ncp, 3)

#     NN = np.zeros((3*mcp, mcp), dtype=float)
#     MM = np.zeros((3*ncp, ncp), dtype=float)
#     for i in range(mcp):
#         NN[3*i + 0, :] = Ax[i, :, 0]
#         NN[3*i + 1, :] = Ax[i, :, 1]
#         NN[3*i + 2, :] = Ax[i, :, 2]
#     for j in range(ncp):
#         MM[3*j + 0, :] = Ay[j, :, 0]
#         MM[3*j + 1, :] = Ay[j, :, 1]
#         MM[3*j + 2, :] = Ay[j, :, 2]

#     # Lame parameters (plane strain)
#     mu = E / (2.0 * (1.0 + nu))
#     lam = E * nu / ((1.0 + nu) * (1.0 - 2.0*nu))

#     f_gl = np.zeros(ndof, dtype=float)
#     k_rk = np.zeros((2, ndof), dtype=float)

#     rows = []
#     cols = []
#     vals = []

#     icount = 1
#     for j in range(1, ncp+1):
#         for i in range(1, mcp+1):
#             N, dN, ddN = mapping2d(i, j, nnod, NN, MM, x, y, w_flat)
#             k_rk, f_gl = apply_bc(i, j, mcp, ncp, N, dN, ddN, icount, k_rk, f_gl, mu, lam, bc, body_force)

#             rr, cc = np.nonzero(k_rk)
#             vv = k_rk[rr, cc]
#             # shift rows to global equation indices like MATLAB:
#             # row(kk+1:kk+l) = 2*icount-2 + rowk; with rowk in {1,2}
#             global_r = (2*icount - 2) + rr  # rr is 0/1
#             rows.append(global_r)
#             cols.append(cc)
#             vals.append(vv)

#             icount += 1

#     rows = np.concatenate(rows)
#     cols = np.concatenate(cols)
#     vals = np.concatenate(vals)

#     K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
#     sol = spsolve(K, f_gl)

#     u = sol[0::2]
#     v = sol[1::2]

#     # von Mises at collocation points (as in MATLAB)
#     sigma_vm = np.zeros((mcp, ncp), dtype=float)
#     for j in range(1, ncp+1):
#         for i in range(1, mcp+1):
#             _, dN, _ = mapping2d(i, j, nnod, NN, MM, x, y, w_flat)
#             u_x = dN[0, :] @ u
#             v_y = dN[1, :] @ v
#             u_y = dN[1, :] @ u
#             v_x = dN[0, :] @ v

#             strain_xx = u_x
#             strain_yy = v_y
#             strain_xy = u_y + v_x

#             stress_xx = lam*(strain_xx + strain_yy) + 2*mu*strain_xx
#             stress_yy = lam*(strain_xx + strain_yy) + 2*mu*strain_yy
#             stress_xy = mu*strain_xy
#             stress_zz = lam*(strain_xx + strain_yy)  # plane strain

#             sigma_vm[i-1, j-1] = np.sqrt(
#                 0.5*((stress_xx - stress_yy)**2 + (stress_yy - stress_zz)**2 + (stress_zz - stress_xx)**2)
#                 + 3.0*(stress_xy**2)
#             )

#     meta = dict(csi=csi, eta=eta, X0=X0, Y0=Y0, weights=w, mu=mu, lam=lam)
#     return u, v, sigma_vm, meta
