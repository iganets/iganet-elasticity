# ref_collocation/iga_collocation.py
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .bspline import greville_abscissae, bspline_all_basis_and_ders
from .mapping2d import mapping2d
from .apply_bc import apply_bc, BC, BCConfig


def open_uniform_knots(ncp: int, degree: int) -> np.ndarray:
    # MATLAB: [zeros(1,p), linspace(0,1,ncp-p+1), ones(1,p)]
    inner = np.linspace(0.0, 1.0, ncp - degree + 1)
    return np.concatenate([np.zeros(degree), inner, np.ones(degree)])


def solve_elasticity_collocation_2d(p: int, q: int, mcp: int, ncp: int,
                                   E: float = 210.0, nu: float = 0.25,
                                   bc: BCConfig = None,
                                   body_force: tuple[float, float] = (0.0, 0.0),
                                   weights: np.ndarray | None = None):

    """
    Returns:
      u, v: displacement vectors at control points (shape nnod,)
      sigma_vm: von Mises on collocation/control points (shape (mcp,ncp))
      meta: dict with knots, control points etc.
    """
 
    if bc is None:
        raise ValueError("bc must be provided (BCConfig). Got None.")

    if weights is None:
        w = np.ones((mcp, ncp), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(mcp, ncp)

    # knots (open uniform)
    csi = open_uniform_knots(mcp, p)
    eta = open_uniform_knots(ncp, q)

    # control points placed at Greville (like MATLAB)
    grev_x = greville_abscissae(csi, p, mcp)
    grev_y = greville_abscissae(eta, q, ncp)
    X0, Y0 = np.meshgrid(grev_x, grev_y, indexing="ij")  # (mcp,ncp)

    nnod = mcp * ncp
    ndof = 2 * nnod

    x = X0.reshape(-1, order='F')  # flatten in column-major order to match MATLAB
    y = Y0.reshape(-1, order='F')
    w_flat = w.reshape(-1, order='F')

    # collocation points = Greville (MATLAB: aveknt(knots, p+1))
    coll_csi = grev_x
    coll_eta = grev_y

    # precompute basis/derivatives at collocation points
    # for mapping2d we mimic NN/MM layout: 3 rows per point i/j
    Ax = bspline_all_basis_and_ders(csi, p, coll_csi, n_deriv=2)  # (mcp, mcp, 3)
    Ay = bspline_all_basis_and_ders(eta, q, coll_eta, n_deriv=2)  # (ncp, ncp, 3)

    NN = np.zeros((3*mcp, mcp), dtype=float)
    MM = np.zeros((3*ncp, ncp), dtype=float)
    for i in range(mcp):
        NN[3*i + 0, :] = Ax[i, :, 0]
        NN[3*i + 1, :] = Ax[i, :, 1]
        NN[3*i + 2, :] = Ax[i, :, 2]
    for j in range(ncp):
        MM[3*j + 0, :] = Ay[j, :, 0]
        MM[3*j + 1, :] = Ay[j, :, 1]
        MM[3*j + 2, :] = Ay[j, :, 2]

    # Lame parameters (plane strain)
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0*nu))

    f_gl = np.zeros(ndof, dtype=float)
    k_rk = np.zeros((2, ndof), dtype=float)

    rows = []
    cols = []
    vals = []

    icount = 1
    for j in range(1, ncp+1):
        for i in range(1, mcp+1):
            N, dN, ddN = mapping2d(i, j, nnod, NN, MM, x, y, w_flat)
            k_rk, f_gl = apply_bc(i, j, mcp, ncp, N, dN, ddN, icount, k_rk, f_gl, mu, lam, bc, body_force)

            rr, cc = np.nonzero(k_rk)
            vv = k_rk[rr, cc]
            # shift rows to global equation indices like MATLAB:
            # row(kk+1:kk+l) = 2*icount-2 + rowk; with rowk in {1,2}
            global_r = (2*icount - 2) + rr  # rr is 0/1
            rows.append(global_r)
            cols.append(cc)
            vals.append(vv)

            icount += 1

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)

    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    sol = spsolve(K, f_gl)

    u = sol[0::2]
    v = sol[1::2]

    # von Mises at collocation points (as in MATLAB)
    sigma_vm = np.zeros((mcp, ncp), dtype=float)
    for j in range(1, ncp+1):
        for i in range(1, mcp+1):
            _, dN, _ = mapping2d(i, j, nnod, NN, MM, x, y, w_flat)
            u_x = dN[0, :] @ u
            v_y = dN[1, :] @ v
            u_y = dN[1, :] @ u
            v_x = dN[0, :] @ v

            strain_xx = u_x
            strain_yy = v_y
            strain_xy = u_y + v_x

            stress_xx = lam*(strain_xx + strain_yy) + 2*mu*strain_xx
            stress_yy = lam*(strain_xx + strain_yy) + 2*mu*strain_yy
            stress_xy = mu*strain_xy
            stress_zz = lam*(strain_xx + strain_yy)  # plane strain

            sigma_vm[i-1, j-1] = np.sqrt(
                0.5*((stress_xx - stress_yy)**2 + (stress_yy - stress_zz)**2 + (stress_zz - stress_xx)**2)
                + 3.0*(stress_xy**2)
            )

    meta = dict(csi=csi, eta=eta, X0=X0, Y0=Y0, weights=w, mu=mu, lam=lam)
    return u, v, sigma_vm, meta
