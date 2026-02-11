"""
Parametric-space mapping helpers and misc. utilities.
"""

import numpy as np


def nurb2proj(nob, control_points, weights):
    """Convert control points and weights to projective (homogeneous) coordinates."""
    assert isinstance(nob, int), "nob must be int"
    assert len(control_points) == len(weights) == nob
    control_points = np.array(control_points, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    projcoord = control_points * weights[:, None]
    projcoord = np.hstack([projcoord, weights[:, None]])
    return projcoord


def parent2ParametricSpace(xi_range, xi_bar):
    """Map parent coordinate *xi_bar* ∈ [-1,1] to parametric interval [xi_min, xi_max]."""
    xmin, xmax = xi_range
    return ((xmax - xmin) * xi_bar + xmax + xmin) / 2.0


def jacobianPaPaMapping2d(uRange, vRange):
    """Jacobian of the parent→parametric mapping in 2-D."""
    uMin, uMax = uRange
    vMin, vMax = vRange
    J2xi = 0.5 * (uMax - uMin)
    J2eta = 0.5 * (vMax - vMin)
    return J2xi * J2eta


def delete_duplicates_preserve_order(arr):
    """Remove duplicates while preserving first-occurrence order (like Mathematica's DeleteDuplicates)."""
    out = []
    seen = set()
    for x in arr:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
