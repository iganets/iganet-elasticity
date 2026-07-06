from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BC:
    type: str
    value: np.ndarray


@dataclass(frozen=True)
class BCConfig:
    left: BC
    right: BC
    bottom: BC
    top: BC


def _boundary_flags(i: int, j: int, mcp: int, ncp: int):
    on_left = i == 1
    on_right = i == mcp
    on_bottom = j == 1
    on_top = j == ncp
    return on_left, on_right, on_bottom, on_top, (
        on_left or on_right or on_bottom or on_top
    )


def _pick_dirichlet_value(on_left, on_right, on_bottom, on_top, bc: BCConfig) -> np.ndarray:
    if on_left and bc.left.type == "dirichlet":
        return bc.left.value
    if on_right and bc.right.type == "dirichlet":
        return bc.right.value
    if on_bottom and bc.bottom.type == "dirichlet":
        return bc.bottom.value
    return bc.top.value


def _pick_non_dirichlet_edge(on_left, on_right, on_bottom, on_top) -> str:
    if on_left:
        return "left"
    if on_right:
        return "right"
    if on_bottom:
        return "bottom"
    return "top"


def _edge_normal(edge: str) -> np.ndarray:
    if edge == "left":
        return np.array([-1.0, 0.0])
    if edge == "right":
        return np.array([1.0, 0.0])
    if edge == "bottom":
        return np.array([0.0, -1.0])
    if edge == "top":
        return np.array([0.0, 1.0])
    raise ValueError(f"Invalid edge '{edge}'")


def apply_bc(
    i: int,
    j: int,
    mcp: int,
    ncp: int,
    N: np.ndarray,
    dN: np.ndarray,
    ddN: np.ndarray,
    icount: int,
    k_rk: np.ndarray,
    f_gl: np.ndarray,
    mu: float,
    lam: float,
    bc: BCConfig,
    body_force: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply 2D collocation boundary or interior equations at one collocation point.

    Returns (k_rk, f_gl) with the MATLAB-compatible DOF ordering
    [u1, v1, u2, v2, ...].
    """
    dNx = dN[0, :]
    dNy = dN[1, :]
    ddNxx = ddN[0, :]
    ddNxy = ddN[1, :]
    ddNyy = ddN[2, :]

    k_rk[:] = 0.0

    a = 2 * icount - 1
    b = 2 * icount
    a_col = a - 1
    b_col = b - 1
    a_row = a - 1
    b_row = b - 1

    on_left, on_right, on_bottom, on_top, is_on_bnd = _boundary_flags(i, j, mcp, ncp)

    if is_on_bnd:
        dirichlet_applies = (
            (on_left and bc.left.type == "dirichlet")
            or (on_right and bc.right.type == "dirichlet")
            or (on_bottom and bc.bottom.type == "dirichlet")
            or (on_top and bc.top.type == "dirichlet")
        )
        if dirichlet_applies:
            uv = _pick_dirichlet_value(on_left, on_right, on_bottom, on_top, bc)
            k_rk[0, :] = 0.0
            k_rk[0, a_col] = 1.0
            f_gl[a_row] = float(uv[0])

            k_rk[1, :] = 0.0
            k_rk[1, b_col] = 1.0
            f_gl[b_row] = float(uv[1])
            return k_rk, f_gl

        edge = _pick_non_dirichlet_edge(on_left, on_right, on_bottom, on_top)
        n = _edge_normal(edge)

        edge_bc = getattr(bc, edge)
        if edge_bc.type == "neumann":
            t = edge_bc.value
        elif edge_bc.type == "free":
            t = np.array([0.0, 0.0])
        else:
            raise ValueError(f"Unsupported BC type '{edge_bc.type}' on edge '{edge}'")

        k_rk[0, 0::2] = mu * (2 * dNx * n[0] + dNy * n[1]) + lam * dNx * n[0]
        k_rk[0, 1::2] = mu * (dNx * n[1]) + lam * dNy * n[0]
        f_gl[a_row] = float(t[0])

        k_rk[1, 0::2] = mu * (dNy * n[0]) + lam * dNx * n[1]
        k_rk[1, 1::2] = mu * (2 * dNy * n[1] + dNx * n[0]) + lam * dNy * n[1]
        f_gl[b_row] = float(t[1])
        return k_rk, f_gl

    fx, fy = body_force
    f_gl[a_row] = fx
    f_gl[b_row] = fy

    k_rk[0, 0::2] = mu * (ddNxx + ddNyy) + (lam + mu) * ddNxx
    k_rk[0, 1::2] = (lam + mu) * ddNxy

    k_rk[1, 0::2] = (lam + mu) * ddNxy
    k_rk[1, 1::2] = mu * (ddNxx + ddNyy) + (lam + mu) * ddNyy

    return k_rk, f_gl
