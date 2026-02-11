"""
Q4 shape functions, Gauss quadrature, and related utilities.
"""

import numpy as np


# =====================================================================
# Gauss quadrature
# =====================================================================
def GP(ngp):
    """Return Gauss point locations for *ngp* points on [-1, 1]."""
    if ngp == 1: return np.array([0.0])
    if ngp == 2: return np.array([-0.5773502, 0.5773502])
    if ngp == 3: return np.array([0.0, -0.7745966, 0.7745966])
    if ngp == 4: return np.array([-0.3399810, 0.3399810, -0.8611363, 0.8611363])
    if ngp == 5: return np.array([0.0, -0.5384693, 0.5384693, -0.9061798, 0.9061798])
    if ngp == 6: return np.array([-0.2386191, 0.2386191, -0.6612093, 0.6612093, -0.9324695, 0.9324695])
    raise ValueError(f"Unsupported ngp={ngp}")


def GW(ngp):
    """Return Gauss weights for *ngp* points on [-1, 1]."""
    if ngp == 1: return np.array([2.0])
    if ngp == 2: return np.array([1.0, 1.0])
    if ngp == 3: return np.array([0.8888889, 0.5555555, 0.5555555])
    if ngp == 4: return np.array([0.6521452, 0.6521452, 0.3478548, 0.3478548])
    if ngp == 5: return np.array([0.5688888, 0.4786288, 0.4786288, 0.2369268, 0.2369268])
    if ngp == 6: return np.array([0.4679140, 0.4679140, 0.3607616, 0.3607616, 0.1713244, 0.1713244])
    raise ValueError(f"Unsupported ngp={ngp}")


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
