# std_collocation_python/apply_bc_3d.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BC:
    type: str           # "dirichlet" | "neumann" | "free"
    value: np.ndarray   # shape (3,)


@dataclass(frozen=True)
class BCConfig3D:
    """
    Side convention (matches C++ code):
      side 1 = x=0  (left)
      side 2 = x=1  (right)
      side 3 = y=0  (bottom)
      side 4 = y=1  (top)
      side 5 = z=0  (front)
      side 6 = z=1  (back)
    """
    left:   BC   # side 1, x=0
    right:  BC   # side 2, x=1
    bottom: BC   # side 3, y=0
    top:    BC   # side 4, y=1
    front:  BC   # side 5, z=0
    back:   BC   # side 6, z=1


def boundary_flags(i: int, j: int, k: int, mcp: int, ncp: int, lcp: int):
    """Returns per-face flags and a combined is_on_boundary flag."""
    on_left   = (i == 1)
    on_right  = (i == mcp)
    on_bottom = (j == 1)
    on_top    = (j == ncp)
    on_front  = (k == 1)
    on_back   = (k == lcp)
    is_on_bnd = on_left or on_right or on_bottom or on_top or on_front or on_back
    return on_left, on_right, on_bottom, on_top, on_front, on_back, is_on_bnd


def face_normal(face: str) -> np.ndarray:
    """Outward unit normal for each face of the unit cube."""
    normals = {
        "left":   np.array([-1.0,  0.0,  0.0]),
        "right":  np.array([ 1.0,  0.0,  0.0]),
        "bottom": np.array([ 0.0, -1.0,  0.0]),
        "top":    np.array([ 0.0,  1.0,  0.0]),
        "front":  np.array([ 0.0,  0.0, -1.0]),
        "back":   np.array([ 0.0,  0.0,  1.0]),
    }
    if face not in normals:
        raise ValueError(f"Invalid face '{face}'")
    return normals[face]


# Priority for BC selection when a node sits on multiple faces.
# Dirichlet wins over Neumann wins over free.
_BC_PRIORITY = {"dirichlet": 3, "neumann": 2, "free": 1}

_FACE_ORDER = ["left", "right", "bottom", "top", "front", "back"]   # lower index wins ties


def pick_active_face(on_left, on_right, on_bottom, on_top, on_front, on_back,
                      bc: BCConfig3D) -> str:
    """
    Among all faces this node touches, return the face whose BC has the highest
    priority.  Ties are broken by face order (left > right > bottom > top > front > back).
    """
    flags = {
        "left":   on_left,
        "right":  on_right,
        "bottom": on_bottom,
        "top":    on_top,
        "front":  on_front,
        "back":   on_back,
    }
    best_face = None
    best_prio = -1
    for face in _FACE_ORDER:
        if not flags[face]:
            continue
        bc_obj = getattr(bc, face)
        prio = _BC_PRIORITY[bc_obj.type]
        if prio > best_prio:
            best_prio = prio
            best_face = face
    return best_face   # never None for boundary nodes


def traction_local_block(dN: np.ndarray, n: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """
    Cauchy traction sigma(u)*n expressed as rows over the local (u,v,w) DOFs
    of a single patch, i.e. sigma_ij = lam*div(u)*delta_ij + mu*(u_i,j + u_j,i).

    Parameters
    ----------
    dN : shape (3, nnod) physical first derivatives [dx, dy, dz] of the patch's
         own basis functions at the point of interest.
    n  : shape (3,) unit normal (in physical space) at that point.

    Returns
    -------
    k_rows : shape (3, 3*nnod) -- rows for t_x, t_y, t_z; columns are
             u-DOFs (0::3), v-DOFs (1::3), w-DOFs (2::3).

    Extracted from apply_bc_3d() so the same formula can be reused for
    exterior Neumann/free boundaries (single patch) and for interior
    patch-to-patch interface traction-continuity equations (multipatch).
    """
    nnod = dN.shape[1]
    k_rows = np.zeros((3, 3*nnod), dtype=float)

    nx, ny, nz = n
    dNx = dN[0, :]
    dNy = dN[1, :]
    dNz = dN[2, :]

    k_rows[0, 0::3] = (lam + 2*mu)*dNx*nx + mu*dNy*ny + mu*dNz*nz
    k_rows[0, 1::3] = lam*dNy*nx + mu*dNx*ny
    k_rows[0, 2::3] = lam*dNz*nx + mu*dNx*nz

    k_rows[1, 0::3] = mu*dNy*nx + lam*dNx*ny
    k_rows[1, 1::3] = mu*dNx*nx + (lam + 2*mu)*dNy*ny + mu*dNz*nz
    k_rows[1, 2::3] = lam*dNz*ny + mu*dNy*nz

    k_rows[2, 0::3] = mu*dNz*nx + lam*dNx*nz
    k_rows[2, 1::3] = mu*dNz*ny + lam*dNy*nz
    k_rows[2, 2::3] = mu*dNx*nx + mu*dNy*ny + (lam + 2*mu)*dNz*nz

    return k_rows


def interior_pde_local_block(dN: np.ndarray, ddN: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """
    Strong-form Navier-Lame residual rows over the local (u,v,w) DOFs of a
    single patch:  mu*Laplacian(u) + (lam+mu)*grad(div(u)) + f = 0.

    Extracted from apply_bc_3d() so the same formula can be reused for
    interior collocation points (single patch) and for interior-patch
    collocation points of any one patch in a multipatch assembly.
    """
    dNxx = ddN[0, :]
    dNxy = ddN[1, :]
    dNxz = ddN[2, :]
    dNyy = ddN[3, :]
    dNyz = ddN[4, :]
    dNzz = ddN[5, :]
    dNlap = dNxx + dNyy + dNzz

    nnod = dN.shape[1]
    k_rows = np.zeros((3, 3*nnod), dtype=float)

    k_rows[0, 0::3] = mu * dNlap + (lam + mu) * dNxx
    k_rows[0, 1::3] = (lam + mu) * dNxy
    k_rows[0, 2::3] = (lam + mu) * dNxz

    k_rows[1, 0::3] = (lam + mu) * dNxy
    k_rows[1, 1::3] = mu * dNlap + (lam + mu) * dNyy
    k_rows[1, 2::3] = (lam + mu) * dNyz

    k_rows[2, 0::3] = (lam + mu) * dNxz
    k_rows[2, 1::3] = (lam + mu) * dNyz
    k_rows[2, 2::3] = mu * dNlap + (lam + mu) * dNzz

    return k_rows


def apply_bc_3d(i: int, j: int, k: int,
                mcp: int, ncp: int, lcp: int,
                N: np.ndarray, dN: np.ndarray, ddN: np.ndarray,
                icount: int,
                k_rk: np.ndarray, f_gl: np.ndarray,
                mu: float, lam: float,
                bc: BCConfig3D,
                body_force: tuple[float, float, float],
                ) -> tuple[np.ndarray, np.ndarray]:
    """
    Assembles one collocation row into k_rk and f_gl.

    DOF ordering: [u1 v1 w1  u2 v2 w2  ...  uN vN wN]
    i.e. every node contributes 3 consecutive DOFs.

    Parameters
    ----------
    i, j, k   : 1-based grid indices
    mcp,ncp,lcp: number of control points in xi, eta, zeta
    N          : shape (nnod,)   rational basis values
    dN         : shape (3, nnod) physical first derivatives  [dx, dy, dz]
    ddN        : shape (6, nnod) physical second derivatives
                 [dxx, dxy, dxz, dyy, dyz, dzz]
    icount     : 1-based node counter (same traversal order as outer loop)
    k_rk       : shape (3, 3*nnod)  -- rows for this collocation point (overwritten)
    f_gl       : shape (3*nnod,)    -- global RHS (modified in-place)
    """

    k_rk[:] = 0.0

    # Global DOF rows for this collocation point (0-based)
    row_u = 3 * (icount - 1)
    row_v = row_u + 1
    row_w = row_u + 2

    on_left, on_right, on_bottom, on_top, on_front, on_back, is_on_bnd = \
        boundary_flags(i, j, k, mcp, ncp, lcp)

    # ------------------------------------------------------------------ #
    #  BOUNDARY NODE                                                       #
    # ------------------------------------------------------------------ #
    if is_on_bnd:
        face = pick_active_face(on_left, on_right, on_bottom, on_top,
                                 on_front, on_back, bc)
        bc_obj = getattr(bc, face)

        # --- Dirichlet: identity rows ---
        if bc_obj.type == "dirichlet":
            uv = bc_obj.value   # shape (3,)

            k_rk[0, :] = 0.0
            k_rk[0, row_u] = 1.0
            f_gl[row_u] = float(uv[0])

            k_rk[1, :] = 0.0
            k_rk[1, row_v] = 1.0
            f_gl[row_v] = float(uv[1])

            k_rk[2, :] = 0.0
            k_rk[2, row_w] = 1.0
            f_gl[row_w] = float(uv[2])

            return k_rk, f_gl

        # --- Neumann or free: traction condition  sigma * n = t ---
        n = face_normal(face)
        if bc_obj.type == "neumann":
            t = bc_obj.value
        else:   # "free"
            t = np.zeros(3)

        k_rk[:] = traction_local_block(dN, n, mu, lam)
        f_gl[row_u] = float(t[0])
        f_gl[row_v] = float(t[1])
        f_gl[row_w] = float(t[2])

        return k_rk, f_gl

    # ------------------------------------------------------------------ #
    #  INTERIOR NODE: strong-form Navier-Lamé                             #
    # ------------------------------------------------------------------ #
    # mu * Laplacian(u) + (lam + mu) * grad(div(u)) + f = 0

    # interior_pde_local_block() assembles div(sigma) on the left-hand side,
    # and the equilibrium equation is div(sigma) + f = 0, so the right-hand
    # side carries MINUS the body force. With +f the body moved against the
    # applied load, e.g. a block under gravity rose instead of sagging.
    fx, fy, fz = body_force
    f_gl[row_u] = -fx
    f_gl[row_v] = -fy
    f_gl[row_w] = -fz

    k_rk[:] = interior_pde_local_block(dN, ddN, mu, lam)

    return k_rk, f_gl



# # std_collocation_python/apply_bc_2d.py
# from __future__ import annotations
# from dataclasses import dataclass
# import numpy as np

# @dataclass(frozen=True)
# class BC:
#     type: str          # "dirichlet" | "neumann" | "free"
#     value: np.ndarray  # shape (2,)

# @dataclass(frozen=True)
# class BCConfig:
#     left: BC
#     right: BC
#     bottom: BC
#     top: BC


# def _boundary_flags(i: int, j: int, mcp: int, ncp: int):
#     on_left = (i == 1)
#     on_right = (i == mcp)
#     on_bottom = (j == 1)
#     on_top = (j == ncp)
#     return on_left, on_right, on_bottom, on_top, (on_left or on_right or on_bottom or on_top)


# def _pick_dirichlet_value(on_left, on_right, on_bottom, on_top, bc: BCConfig) -> np.ndarray:
#     # priority: left -> right -> bottom -> top
#     if on_left and bc.left.type == "dirichlet":
#         return bc.left.value
#     if on_right and bc.right.type == "dirichlet":
#         return bc.right.value
#     if on_bottom and bc.bottom.type == "dirichlet":
#         return bc.bottom.value
#     return bc.top.value


# def _pick_non_dirichlet_edge(on_left, on_right, on_bottom, on_top) -> str:
#     # priority: left -> right -> bottom -> top
#     if on_left:
#         return "left"
#     if on_right:
#         return "right"
#     if on_bottom:
#         return "bottom"
#     return "top"


# def _edge_normal(edge: str) -> np.ndarray:
#     if edge == "left":
#         return np.array([-1.0, 0.0])
#     if edge == "right":
#         return np.array([1.0, 0.0])
#     if edge == "bottom":
#         return np.array([0.0, -1.0])
#     if edge == "top":
#         return np.array([0.0, 1.0])
#     raise ValueError(f"Invalid edge '{edge}'")


# def apply_bc(i: int, j: int, mcp: int, ncp: int,
#              N: np.ndarray, dN: np.ndarray, ddN: np.ndarray,
#              icount: int,
#              k_rk: np.ndarray, f_gl: np.ndarray,
#              mu: float, lam: float,
#              bc: BCConfig,
#              body_force: tuple[float, float],
#              ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Returns (k_rk, f_gl) like MATLAB.
#     DOF ordering: [u1 v1 u2 v2 ...]
#     """
#     dNx = dN[0, :]
#     dNy = dN[1, :]
#     ddNxx = ddN[0, :]
#     ddNxy = ddN[1, :]
#     ddNyy = ddN[2, :]

#     nnod = len(dNx)
#     k_rk[:] = 0.0  # overwrite

#     a = 2*icount - 1  # 1-based DOF index in MATLAB sense
#     b = 2*icount

#     # convert to python col indices
#     a_col = a - 1
#     b_col = b - 1
#     a_row = a - 1
#     b_row = b - 1

#     on_left, on_right, on_bottom, on_top, is_on_bnd = _boundary_flags(i, j, mcp, ncp)

#     if is_on_bnd:
#         dirichlet_applies = (
#             (on_left and bc.left.type == "dirichlet") or
#             (on_right and bc.right.type == "dirichlet") or
#             (on_bottom and bc.bottom.type == "dirichlet") or
#             (on_top and bc.top.type == "dirichlet")
#         )
#         if dirichlet_applies:
#             uv = _pick_dirichlet_value(on_left, on_right, on_bottom, on_top, bc)
#             # identity rows
#             k_rk[0, :] = 0.0
#             k_rk[0, a_col] = 1.0
#             f_gl[a_row] = float(uv[0])

#             k_rk[1, :] = 0.0
#             k_rk[1, b_col] = 1.0
#             f_gl[b_row] = float(uv[1])
#             return k_rk, f_gl

#         edge = _pick_non_dirichlet_edge(on_left, on_right, on_bottom, on_top)
#         n = _edge_normal(edge)

#         edge_bc = getattr(bc, edge)
#         if edge_bc.type == "neumann":
#             t = edge_bc.value
#         elif edge_bc.type == "free":
#             t = np.array([0.0, 0.0])
#         else:
#             raise ValueError(f"Unsupported BC type '{edge_bc.type}' on edge '{edge}'")

#         # traction x-component
#         k_rk[0, 0::2] = mu*(2*dNx*n[0] + dNy*n[1]) + lam*dNx*n[0]
#         k_rk[0, 1::2] = mu*(dNx*n[1])              + lam*dNy*n[0]
#         f_gl[a_row] = float(t[0])

#         # traction y-component
#         k_rk[1, 0::2] = mu*(dNy*n[0])              + lam*dNx*n[1]
#         k_rk[1, 1::2] = mu*(2*dNy*n[1] + dNx*n[0]) + lam*dNy*n[1]
#         f_gl[b_row] = float(t[1])
#         return k_rk, f_gl

#     # interior strong form (Navier–Lamé)
#     fx, fy = body_force
#     f_gl[a_row] = fx
#     f_gl[b_row] = fy

#     k_rk[0, 0::2] = mu*(ddNxx + ddNyy) + (lam+mu)*ddNxx
#     k_rk[0, 1::2] = (lam+mu)*ddNxy

#     k_rk[1, 0::2] = (lam+mu)*ddNxy
#     k_rk[1, 1::2] = mu*(ddNxx + ddNyy) + (lam+mu)*ddNyy

#     return k_rk, f_gl
