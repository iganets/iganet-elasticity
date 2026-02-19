# std_collocation_python/bspline.py
from __future__ import annotations
import numpy as np

def greville_abscissae(knots: np.ndarray, degree: int, n_basis: int) -> np.ndarray:
    """
    Greville points for open knot vector.
    MATLAB: aveknt(knots, degree+1) behaves like Greville for standard use here.
    """
    # For basis i (0-based): avg of knots[i+1 : i+degree+1]
    g = np.zeros(n_basis, dtype=float)
    for i in range(n_basis):
        g[i] = np.sum(knots[i+1:i+degree+1]) / degree
    return g


def find_span(n_basis: int, degree: int, u: float, U: np.ndarray) -> int:
    """
    Piegl & Tiller FindSpan (The NURBS Book).
    n_basis = number of basis functions
    """
    n = n_basis - 1
    if u >= U[n+1]:
        return n
    if u <= U[degree]:
        return degree
    low = degree
    high = n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid+1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def ders_basis_funs(span: int, u: float, degree: int, n_deriv: int, U: np.ndarray) -> np.ndarray:
    """
    Piegl & Tiller DersBasisFuns.
    Returns ders[k, j] = k-th derivative of N_{span-degree+j}(u), j=0..degree
    Shape: (n_deriv+1, degree+1)
    """
    ndu = np.zeros((degree+1, degree+1), dtype=float)
    left = np.zeros(degree+1, dtype=float)
    right = np.zeros(degree+1, dtype=float)

    ndu[0, 0] = 1.0
    for j in range(1, degree+1):
        left[j] = u - U[span+1-j]
        right[j] = U[span+j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r+1] + left[j-r]
            temp = ndu[r, j-1] / ndu[j, r]
            ndu[r, j] = saved + right[r+1] * temp
            saved = left[j-r] * temp
        ndu[j, j] = saved

    ders = np.zeros((n_deriv+1, degree+1), dtype=float)
    ders[0, :] = ndu[:, degree]

    a = np.zeros((2, degree+1), dtype=float)
    for r in range(degree+1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0

        for k in range(1, n_deriv+1):
            d = 0.0
            rk = r - k
            pk = degree - k

            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk+1, rk]
                d = a[s2, 0] * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k-1 if r-1 <= pk else degree - r

            for j in range(j1, j2+1):
                a[s2, j] = (a[s1, j] - a[s1, j-1]) / ndu[pk+1, rk+j]
                d += a[s2, j] * ndu[rk+j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k-1] / ndu[pk+1, r]
                d += a[s2, k] * ndu[r, pk]

            ders[k, r] = d
            s1, s2 = s2, s1

    from math import factorial
    p = degree
    for k in range(1, n_deriv+1):
        ders[k, :] *= factorial(p) / factorial(p - k)   # p!/(p-k)!

    return ders


def bspline_all_basis_and_ders(U: np.ndarray, degree: int, points: np.ndarray, n_deriv: int = 2) -> np.ndarray:
    """
    Returns array A with shape (n_points, n_basis, n_deriv+1)
    A[ip, i, k] = k-th derivative of basis i at points[ip]
    """
    U = np.asarray(U, dtype=float)
    points = np.asarray(points, dtype=float)

    n_basis = len(U) - degree - 1
    A = np.zeros((len(points), n_basis, n_deriv+1), dtype=float)

    for ip, u in enumerate(points):
        span = find_span(n_basis, degree, float(u), U)
        ders = ders_basis_funs(span, float(u), degree, n_deriv, U)  # (n_deriv+1, degree+1)
        first = span - degree
        for j in range(degree+1):
            A[ip, first + j, :] = ders[:, j]
    return A
