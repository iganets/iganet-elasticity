# std_collocation_python/mapping3d.py
from __future__ import annotations
import numpy as np

def mapping3d(i: int, j: int, k: int, nnod: int,
              NN: np.ndarray, MM: np.ndarray, LL: np.ndarray,
              x: np.ndarray, y: np.ndarray, z: np.ndarray,
              w: np.ndarray):
    """
    3D extension of mapping2D.m

    Indices i, j, k are 1-based (like MATLAB convention kept from 2D version).

    Basis matrices NN, MM, LL are structured so that for each index three rows exist:
      (basis value, 1st derivative, 2nd derivative) in parameter coordinates.
      NN: shape (3*mcp, mcp)  -- xi direction
      MM: shape (3*ncp, ncp)  -- eta direction
      LL: shape (3*lcp, lcp)  -- zeta direction

    Returns:
      N     : shape (nnod,)       -- basis functions
      dN    : shape (3, nnod)     -- physical first derivatives [dN/dx, dN/dy, dN/dz]
      ddN   : shape (6, nnod)     -- physical second derivatives
                                     [d2N/dxx, d2N/dxy, d2N/dxz, d2N/dyy, d2N/dyz, d2N/dzz]
    """
    ii = i - 1
    jj = j - 1
    kk = k - 1

    # Extract basis values and derivatives for each direction
    # Shape: (3, mcp/ncp/lcp)
    Nx = NN[3*ii:3*ii+3, :]   # [N, dN/dxi,   d2N/dxi2  ]
    Ny = MM[3*jj:3*jj+3, :]   # [N, dN/deta,  d2N/deta2 ]
    Nz = LL[3*kk:3*kk+3, :]   # [N, dN/dzeta, d2N/dzeta2]

    mcp = NN.shape[1]
    ncp = MM.shape[1]
    lcp = LL.shape[1]

    w = np.asarray(w).reshape(-1)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    z = np.asarray(z).reshape(-1)

    # --- Tensor products (parameter space) ---
    # N = N_xi(i) * N_eta(j) * N_zeta(k), flattened column-major (F-order: i varies fastest)
    # Shape convention: (mcp, ncp, lcp) -> flatten with order='F'

    def _tp(ax, ay, az):
        """Outer product of three 1D arrays, flattened column-major."""
        return np.einsum('i,j,k->ijk', ax, ay, az).reshape(-1, order='F')

    N_param   = _tp(Nx[0], Ny[0], Nz[0])

    # First derivatives in parameter space
    dN_dxi    = _tp(Nx[1], Ny[0], Nz[0])
    dN_deta   = _tp(Nx[0], Ny[1], Nz[0])
    dN_dzeta  = _tp(Nx[0], Ny[0], Nz[1])

    # Second derivatives in parameter space (6 independent components)
    d2N_dxi2    = _tp(Nx[2], Ny[0], Nz[0])
    d2N_dxieta  = _tp(Nx[1], Ny[1], Nz[0])
    d2N_dxizeta = _tp(Nx[1], Ny[0], Nz[1])
    d2N_deta2   = _tp(Nx[0], Ny[2], Nz[0])
    d2N_detazeta= _tp(Nx[0], Ny[1], Nz[1])
    d2N_dzeta2  = _tp(Nx[0], Ny[0], Nz[2])

    # --- NURBS rational normalization ---
    W = N_param * w

    # Weighted sums (denominators)
    W_sum       = np.sum(W)
    dWdxi_sum   = np.sum(dN_dxi   * w)
    dWdeta_sum  = np.sum(dN_deta  * w)
    dWdzeta_sum = np.sum(dN_dzeta * w)

    d2Wdxi2_sum    = np.sum(d2N_dxi2    * w)
    d2Wdxieta_sum  = np.sum(d2N_dxieta  * w)
    d2Wdxizeta_sum = np.sum(d2N_dxizeta * w)
    d2Wdeta2_sum   = np.sum(d2N_deta2   * w)
    d2Wdetazeta_sum= np.sum(d2N_detazeta* w)
    d2Wdzeta2_sum  = np.sum(d2N_dzeta2  * w)

    # Rational basis (R = W / W_sum)
    R = W / W_sum

    # Rational first derivatives
    dRdxi   = (dN_dxi   * w - R * dWdxi_sum)   / W_sum
    dRdeta  = (dN_deta  * w - R * dWdeta_sum)  / W_sum
    dRdzeta = (dN_dzeta * w - R * dWdzeta_sum) / W_sum

    # Rational second derivatives (quotient rule applied twice)
    def _d2R(d2N_w_sum, dNa_w, dWa_sum, dNb_w, dWb_sum):
        """
        Second cross-derivative of rational basis.
        d2R/da db = (d2N*w - dR/da * dWb - dR/db * dWa - R * d2W) / W_sum
        """
        return (d2N_w_sum * w
                - dNa_w * dWb_sum
                - dNb_w * dWa_sum
                - R * d2Wdxi2_sum) / W_sum  # placeholder -- see below

    # Direct formulation (more explicit, avoids confusion):
    d2Rdxi2    = (d2N_dxi2    * w - 2*dRdxi   * dWdxi_sum   - R * d2Wdxi2_sum)    / W_sum
    d2Rdxieta  = (d2N_dxieta  * w -   dRdxi   * dWdeta_sum
                                  -   dRdeta  * dWdxi_sum   - R * d2Wdxieta_sum)  / W_sum
    d2Rdxizeta = (d2N_dxizeta * w -   dRdxi   * dWdzeta_sum
                                  -   dRdzeta * dWdxi_sum   - R * d2Wdxizeta_sum) / W_sum
    d2Rdeta2   = (d2N_deta2   * w - 2*dRdeta  * dWdeta_sum  - R * d2Wdeta2_sum)   / W_sum
    d2Rdetazeta= (d2N_detazeta* w -   dRdeta  * dWdzeta_sum
                                  -   dRdzeta * dWdeta_sum  - R * d2Wdetazeta_sum) / W_sum
    d2Rdzeta2  = (d2N_dzeta2  * w - 2*dRdzeta * dWdzeta_sum - R * d2Wdzeta2_sum)  / W_sum

    # Stack parameter-space quantities
    # dR_param: (3, nnod)  -- [dR/dxi, dR/deta, dR/dzeta]
    dR_param = np.vstack([dRdxi, dRdeta, dRdzeta])

    # d2R_param: (6, nnod) -- [dxi2, dxi_eta, dxi_zeta, deta2, deta_zeta, dzeta2]
    d2R_param = np.vstack([d2Rdxi2, d2Rdxieta, d2Rdxizeta,
                           d2Rdeta2, d2Rdetazeta, d2Rdzeta2])

    # --- Jacobian of physical mapping x(xi,eta,zeta) ---
    # J[a, b] = dx_a / dxi_b   (3x3)
    J = np.array([
        [dRdxi @ x,   dRdeta @ x,   dRdzeta @ x],
        [dRdxi @ y,   dRdeta @ y,   dRdzeta @ y],
        [dRdxi @ z,   dRdeta @ z,   dRdzeta @ z],
    ], dtype=float)  # shape (3, 3)

    J_inv = np.linalg.inv(J)   # dxi/dx, shape (3, 3)

    dR_phys = J_inv.T @ dR_param   # (3, nnod)

    d2x_param = np.array([
        [d2Rdxi2 @ x, d2Rdxieta @ x, d2Rdxizeta @ x, d2Rdeta2 @ x, d2Rdetazeta @ x, d2Rdzeta2 @ x],
        [d2Rdxi2 @ y, d2Rdxieta @ y, d2Rdxizeta @ y, d2Rdeta2 @ y, d2Rdetazeta @ y, d2Rdzeta2 @ y],
        [d2Rdxi2 @ z, d2Rdxieta @ z, d2Rdxizeta @ z, d2Rdeta2 @ z, d2Rdetazeta @ z, d2Rdzeta2 @ z],
    ], dtype=float)  # shape (3, 6)

    # Build the 6x6 second-order metric tensor H such that:
    #   d2R/dphys[a,b] = H^{-T} * (d2R_param - correction)
    # where the correction accounts for the curvature of the mapping.
    #
    # Index mapping for 6-vector: [0]=xx, [1]=xy, [2]=xz, [3]=yy, [4]=yz, [5]=zz
    # Parameter index mapping:    [0]=xi2,[1]=xi_eta,[2]=xi_zeta,[3]=eta2,[4]=eta_zeta,[5]=zeta2

    Ji = J_inv  # (3,3), Ji[p,a] = dxi_p/dx_a

    # H[m, n] maps physical index pair (a,b) to parameter pair (p,q):
    # H[m, n] = Ji[p, a] * Ji[q, b]  where m=(a,b), n=(p,q) with symmetry
    # Physical pairs: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    # Parameter pairs: same ordering
    phys_pairs  = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
    param_pairs = [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]

    H = np.zeros((6, 6), dtype=float)
    for m, (a, b) in enumerate(phys_pairs):
        for n, (p, q) in enumerate(param_pairs):
            if p == q:
                H[m, n] = Ji[p, a] * Ji[q, b]
            else:
                # off-diagonal parameter pairs appear once in our list but twice in the sum
                H[m, n] = Ji[p, a] * Ji[q, b] + Ji[q, a] * Ji[p, b]

    # Correction for mapping curvature:
    # corr[n, pt] = sum over physical coords c: d2x_c/dxi_p dxi_q * dR_phys[c, pt]
    # n indexes parameter pairs
    corr = np.zeros((6, nnod), dtype=float)
    for n, (p, q) in enumerate(param_pairs):
        # d2x_param[:, n] has shape (3,) -- second deriv of x,y,z wrt (p,q)
        corr[n, :] = d2x_param[:, n] @ dR_phys   # (3,) dot (3, nnod) -> (nnod,)

    # Solve: H @ d2R_phys = d2R_param - corr
    d2R_phys = np.linalg.solve(H, d2R_param - corr)   # (6, nnod)

    return R, dR_phys, d2R_phys
    
    # # tensor product (outer products), then flatten
    # N  = (np.outer(Nx[0, :], Ny[0, :]).reshape(1, nnod, order='F') * w)
    # dN = np.zeros((2, nnod), dtype=float)
    # ddN = np.zeros((3, nnod), dtype=float)

    # dN[0, :]  = (np.outer(Nx[1, :], Ny[0, :]).reshape(1, nnod, order='F') * w)
    # dN[1, :]  = (np.outer(Nx[0, :], Ny[1, :]).reshape(1, nnod, order='F') * w)

    # ddN[0, :] = (np.outer(Nx[2, :], Ny[0, :]).reshape(1, nnod, order='F') * w)
    # ddN[1, :] = (np.outer(Nx[1, :], Ny[1, :]).reshape(1, nnod, order='F') * w)
    # ddN[2, :] = (np.outer(Nx[0, :], Ny[2, :]).reshape(1, nnod, order='F') * w)

    # # rational normalization
    # den_sum    = np.sum(N)
    # der_sumx   = np.sum(dN[0, :])
    # der_sumy   = np.sum(dN[1, :])
    # der2_sumx  = np.sum(ddN[0, :])
    # der2_sumxy = np.sum(ddN[1, :])
    # der2_sumy  = np.sum(ddN[2, :])

    # ddN[0, :] = ddN[0, :]/den_sum - (2*dN[0, :]*der_sumx + N*der2_sumx)/den_sum**2 + 2*N*(der_sumx**2)/den_sum**3
    # ddN[1, :] = ddN[1, :]/den_sum - (dN[0, :]*der_sumy + dN[1, :]*der_sumx + N*der2_sumxy)/den_sum**2 + 2*N*der_sumx*der_sumy/den_sum**3
    # ddN[2, :] = ddN[2, :]/den_sum - (2*dN[1, :]*der_sumy + N*der2_sumy)/den_sum**2 + 2*N*(der_sumy**2)/den_sum**3

    # dN[0, :]  = dN[0, :]/den_sum - N*der_sumx/den_sum**2
    # dN[1, :]  = dN[1, :]/den_sum - N*der_sumy/den_sum**2
    # N         = N/den_sum

    # # mapping derivatives to physical coordinates
    # dxdxi = np.array([[dN[0, :] @ x, dN[1, :] @ x],
    #                   [dN[0, :] @ y, dN[1, :] @ y]], dtype=float)
    # dxidx = np.linalg.inv(dxdxi)

    # d2xdxi2 = np.array([[ddN[0, :] @ x, ddN[1, :] @ x, ddN[2, :] @ x],
    #                     [ddN[0, :] @ y, ddN[1, :] @ y, ddN[2, :] @ y]], dtype=float)

    # dxdxi2 = np.array([
    #     [dxdxi[0,0]**2,              dxdxi[0,0]*dxdxi[0,1],                   dxdxi[0,1]**2],
    #     [2*dxdxi[0,0]*dxdxi[1,0],    dxdxi[0,0]*dxdxi[1,1] + dxdxi[0,1]*dxdxi[1,0],  2*dxdxi[0,1]*dxdxi[1,1]],
    #     [dxdxi[1,0]**2,              dxdxi[1,0]*dxdxi[1,1],                   dxdxi[1,1]**2]
    # ], dtype=float)
    # dxidx2 = np.linalg.inv(dxdxi2)

    # dN_phys  = dxidx.T @ dN
    # ddN_phys = dxidx2.T @ (ddN - d2xdxi2.T @ dN_phys)

    # return N.reshape(-1), dN_phys, ddN_phys 
