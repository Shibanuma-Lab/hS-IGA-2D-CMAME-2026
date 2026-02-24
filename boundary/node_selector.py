"""
Robust node-selection helpers for geometric boundaries.
"""

from __future__ import annotations

import numpy as np


def _characteristic_spacing(values: np.ndarray) -> float:
    uniq = np.unique(np.asarray(values, dtype=float))
    if uniq.size < 2:
        return float("nan")
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0.0]
    if diffs.size == 0:
        return float("nan")
    # Use a robust spacing estimate (median) so tiny jitter between points on
    # the same intended boundary level does not collapse tolerance.
    return float(np.median(diffs))


def _auto_tolerance(values: np.ndarray, user_atol=None) -> float:
    if user_atol is not None:
        return float(abs(user_atol))

    arr = np.asarray(values, dtype=float)
    spacing = _characteristic_spacing(arr)
    scale = max(1.0, float(np.max(np.abs(arr))))
    tol_eps = 64.0 * np.finfo(float).eps * scale

    if np.isfinite(spacing):
        tol = max(tol_eps, 1.0e-6 * spacing, 1.0e-9 * scale)
        return float(min(tol, 0.49 * spacing))
    return float(tol_eps)


def find_nodes_near_axis_value(points: np.ndarray, axis: int, target: float, atol=None) -> np.ndarray:
    """
    Find node indices on a coordinate line robustly.

    The algorithm snaps ``target`` to the nearest existing coordinate level,
    then selects all nodes on that level within a tolerance.
    """
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] <= axis:
        raise ValueError(f"points must have shape (n, {axis + 1}+), got {coords.shape}")

    axis_vals = coords[:, axis]
    if axis_vals.size == 0:
        return np.empty(0, dtype=int)

    nearest_idx = int(np.argmin(np.abs(axis_vals - float(target))))
    snapped = float(axis_vals[nearest_idx])
    tol = _auto_tolerance(axis_vals, user_atol=atol)

    node_ids = np.where(np.abs(axis_vals - snapped) <= tol)[0]
    if node_ids.size == 0:
        node_ids = np.array([nearest_idx], dtype=int)
    return np.sort(node_ids.astype(int))


def find_nodes_on_extreme(points: np.ndarray, axis: int, side: str, atol=None) -> np.ndarray:
    """Find node indices on axis minimum/maximum coordinate."""
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[1] <= axis:
        raise ValueError(f"points must have shape (n, {axis + 1}+), got {coords.shape}")

    if side == "min":
        target = float(np.min(coords[:, axis]))
    elif side == "max":
        target = float(np.max(coords[:, axis]))
    else:
        raise ValueError(f"side must be 'min' or 'max', got {side}")
    return find_nodes_near_axis_value(coords, axis=axis, target=target, atol=atol)
