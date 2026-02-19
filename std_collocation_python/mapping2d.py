# std_collocation_python/mapping2d.py
from __future__ import annotations
import numpy as np

def mapping2d(i: int, j: int, nnod: int,
              NN: np.ndarray, MM: np.ndarray,
              x: np.ndarray, y: np.ndarray, w: np.ndarray):
    """
    Python port of mapping2D.m

    Indizes i,j sind 1-basiert wie in MATLAB.
    NN/MM sind so aufgebaut, dass für jeden i bzw. j drei Zeilen existieren:
      (Basis, 1. Ableitung, 2. Ableitung) in Parameterkoordinaten.
    """
    # MATLAB -> Python index shift
    ii = i - 1
    jj = j - 1

    Nx = np.vstack([NN[3*ii + 0, :], NN[3*ii + 1, :], NN[3*ii + 2, :]])  # (3, mcp)
    Ny = np.vstack([MM[3*jj + 0, :], MM[3*jj + 1, :], MM[3*jj + 2, :]])  # (3, ncp)

    w = np.asarray(w).reshape(-1)  # (nnod,)
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    # tensor product (outer products), then flatten
    N  = (np.outer(Nx[0, :], Ny[0, :]).reshape(1, nnod, order='F') * w)
    dN = np.zeros((2, nnod), dtype=float)
    ddN = np.zeros((3, nnod), dtype=float)

    dN[0, :]  = (np.outer(Nx[1, :], Ny[0, :]).reshape(1, nnod, order='F') * w)
    dN[1, :]  = (np.outer(Nx[0, :], Ny[1, :]).reshape(1, nnod, order='F') * w)

    ddN[0, :] = (np.outer(Nx[2, :], Ny[0, :]).reshape(1, nnod, order='F') * w)
    ddN[1, :] = (np.outer(Nx[1, :], Ny[1, :]).reshape(1, nnod, order='F') * w)
    ddN[2, :] = (np.outer(Nx[0, :], Ny[2, :]).reshape(1, nnod, order='F') * w)

    # rational normalization
    den_sum    = np.sum(N)
    der_sumx   = np.sum(dN[0, :])
    der_sumy   = np.sum(dN[1, :])
    der2_sumx  = np.sum(ddN[0, :])
    der2_sumxy = np.sum(ddN[1, :])
    der2_sumy  = np.sum(ddN[2, :])

    ddN[0, :] = ddN[0, :]/den_sum - (2*dN[0, :]*der_sumx + N*der2_sumx)/den_sum**2 + 2*N*(der_sumx**2)/den_sum**3
    ddN[1, :] = ddN[1, :]/den_sum - (dN[0, :]*der_sumy + dN[1, :]*der_sumx + N*der2_sumxy)/den_sum**2 + 2*N*der_sumx*der_sumy/den_sum**3
    ddN[2, :] = ddN[2, :]/den_sum - (2*dN[1, :]*der_sumy + N*der2_sumy)/den_sum**2 + 2*N*(der_sumy**2)/den_sum**3

    dN[0, :]  = dN[0, :]/den_sum - N*der_sumx/den_sum**2
    dN[1, :]  = dN[1, :]/den_sum - N*der_sumy/den_sum**2
    N         = N/den_sum

    # mapping derivatives to physical coordinates
    dxdxi = np.array([[dN[0, :] @ x, dN[1, :] @ x],
                      [dN[0, :] @ y, dN[1, :] @ y]], dtype=float)
    dxidx = np.linalg.inv(dxdxi)

    d2xdxi2 = np.array([[ddN[0, :] @ x, ddN[1, :] @ x, ddN[2, :] @ x],
                        [ddN[0, :] @ y, ddN[1, :] @ y, ddN[2, :] @ y]], dtype=float)

    dxdxi2 = np.array([
        [dxdxi[0,0]**2,              dxdxi[0,0]*dxdxi[0,1],                   dxdxi[0,1]**2],
        [2*dxdxi[0,0]*dxdxi[1,0],    dxdxi[0,0]*dxdxi[1,1] + dxdxi[0,1]*dxdxi[1,0],  2*dxdxi[0,1]*dxdxi[1,1]],
        [dxdxi[1,0]**2,              dxdxi[1,0]*dxdxi[1,1],                   dxdxi[1,1]**2]
    ], dtype=float)
    dxidx2 = np.linalg.inv(dxdxi2)

    dN_phys  = dxidx.T @ dN
    ddN_phys = dxidx2.T @ (ddN - d2xdxi2.T @ dN_phys)

    return N.reshape(-1), dN_phys, ddN_phys
