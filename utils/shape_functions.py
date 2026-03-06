"""
Q4 shape functions, Gauss quadrature, and related utilities.
"""

from functools import lru_cache

import numpy as np


# =====================================================================
# Gauss quadrature
# =====================================================================
@lru_cache(maxsize=None)
def _gauss_rule(ngp):
    """
    Return 1D Gauss-Legendre nodes/weights in float64.

    Cached because the same ngp values are requested repeatedly.
    """
    try:
        n = int(ngp)
    except (TypeError, ValueError):
        raise ValueError(f"Unsupported ngp={ngp}")
    if n < 1:
        raise ValueError(f"Unsupported ngp={ngp}")

    # leggauss gives full float64-precision nodes and weights.
    xi, w = np.polynomial.legendre.leggauss(n)
    return np.asarray(xi, dtype=np.float64), np.asarray(w, dtype=np.float64)


def GP(ngp):
    """Return Gauss point locations for *ngp* points on [-1, 1]."""
    xi, _ = _gauss_rule(ngp)
    return xi.copy()


def GW(ngp):
    """Return Gauss weights for *ngp* points on [-1, 1]."""
    _, w = _gauss_rule(ngp)
    return w.copy()


# =====================================================================
# Q4 (bilinear quad) shape functions
# =====================================================================
def N1(xi, eta):
    return 0.25 * (1.0 - xi) * (1.0 - eta)

def N2(xi, eta):
    return 0.25 * (1.0 + xi) * (1.0 - eta)

def N3(xi, eta):
    return 0.25 * (1.0 + xi) * (1.0 + eta)

def N4(xi, eta):
    return 0.25 * (1.0 - xi) * (1.0 + eta)


def intpsub(xi, eta):
    """2×8 interpolation matrix."""
    return np.array([
        [N1(xi, eta), 0., N2(xi, eta), 0., N3(xi, eta), 0., N4(xi, eta), 0.],
        [0., N1(xi, eta), 0., N2(xi, eta), 0., N3(xi, eta), 0., N4(xi, eta)],
    ], dtype=float)


def shp(pt):
    """Shape function values at parent coordinate *pt* = (ξ, η).

    Returns shape (1, 4).
    """
    xi, eta = float(pt[0]), float(pt[1])
    return np.array([[
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ]], dtype=float) * 0.25


def Dshp(pt):
    """Shape function derivatives at parent coordinate *pt* = (ξ, η).

    Returns shape (2, 4).
    """
    xi, eta = float(pt[0]), float(pt[1])
    return np.array([
        [-(1.0 - eta),  (1.0 - eta),  (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),   (1.0 + xi),   (1.0 - xi)],
    ], dtype=float) * 0.25


# =====================================================================
# B-matrix helpers
# =====================================================================
def enlarge(mat):
    """Convert a (2×4) derivative matrix to a (3×8) B-matrix layout."""
    x1, x2, x3, x4 = mat[0, :]
    y1, y2, y3, y4 = mat[1, :]
    return np.array([
        [x1, 0., x2, 0., x3, 0., x4, 0.],
        [0., y1, 0., y2, 0., y3, 0., y4],
        [y1, x1, y2, x2, y3, x3, y4, x4],
    ], dtype=float)


def enlarge2(row):
    """Convert a length-4 shape function row to a (2×8) interpolation matrix."""
    r = np.asarray(row, dtype=float).ravel()
    if r.size != 4:
        raise ValueError(f"enlarge2: expected 4 entries, got {r.size}")
    x1, x2, x3, x4 = r
    return np.array([
        [x1, 0., x2, 0., x3, 0., x4, 0.],
        [0., x1, 0., x2, 0., x3, 0., x4],
    ], dtype=float)
