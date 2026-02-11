"""
Bilinear quadrilateral interpolator.

Matches Mathematica's ``Interpolation`` behaviour for structured /
unstructured quad meshes.
"""

import numpy as np

try:
    from scipy.optimize import fsolve
except ImportError:
    fsolve = None


class BilinearQuadInterpolator:
    """Bilinear interpolator on quadrilateral mesh elements."""

    def __init__(self, nodes, elements, values, name="mesh"):
        self.nodes = np.asarray(nodes, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.name = name

        self.elements = np.array(elements, dtype=int)
        if np.min(self.elements) == 1:
            self.elements = self.elements - 1

        self._build_element_bbox()

    # ------------------------------------------------------------------
    def _build_element_bbox(self):
        nelem = len(self.elements)
        self.bbox_min = np.zeros((nelem, 2))
        self.bbox_max = np.zeros((nelem, 2))
        for e in range(nelem):
            elem_nodes = self.nodes[self.elements[e]]
            self.bbox_min[e] = np.min(elem_nodes, axis=0)
            self.bbox_max[e] = np.max(elem_nodes, axis=0)

    # ------------------------------------------------------------------
    def _find_containing_element(self, x, y, tol=1e-6):
        bbox_tol = tol * 10
        candidates = np.where(
            (x >= self.bbox_min[:, 0] - bbox_tol)
            & (x <= self.bbox_max[:, 0] + bbox_tol)
            & (y >= self.bbox_min[:, 1] - bbox_tol)
            & (y <= self.bbox_max[:, 1] + bbox_tol)
        )[0]

        best_elem = None
        best_xi = None
        best_eta = None
        best_dist = float("inf")

        for e in candidates:
            elem_coords = self.nodes[self.elements[e]]
            xi, eta, success = self._point_to_parent(x, y, elem_coords, tol)
            if success:
                dist_outside = max(0, abs(xi) - 1.0) + max(0, abs(eta) - 1.0)
                if dist_outside < best_dist:
                    best_dist = dist_outside
                    best_elem = e
                    best_xi = xi
                    best_eta = eta
                    if dist_outside == 0:
                        return e, xi, eta

        if best_elem is not None and best_dist < 0.1:
            return best_elem, best_xi, best_eta

        return None, None, None

    # ------------------------------------------------------------------
    def _point_to_parent(self, x, y, elem_coords, tol=1e-6):
        x_coords = elem_coords[:, 0]
        y_coords = elem_coords[:, 1]

        rect_tol = tol * 100
        if (
            np.abs(x_coords[0] - x_coords[3]) < rect_tol
            and np.abs(x_coords[1] - x_coords[2]) < rect_tol
            and np.abs(y_coords[0] - y_coords[1]) < rect_tol
            and np.abs(y_coords[2] - y_coords[3]) < rect_tol
        ):
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_center = 0.5 * (x_min + x_max)
            y_center = 0.5 * (y_min + y_max)
            x_half = 0.5 * (x_max - x_min)
            y_half = 0.5 * (y_max - y_min)
            if x_half > 1e-12 and y_half > 1e-12:
                xi = (x - x_center) / x_half
                eta = (y - y_center) / y_half
                if abs(xi) <= 1.0 + tol * 10 and abs(eta) <= 1.0 + tol * 10:
                    return xi, eta, True
            return None, None, False

        def residual(p):
            xi, eta = p
            N = np.array([
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
            ])
            pos = N @ elem_coords
            return [pos[0] - x, pos[1] - y]

        try:
            initial_guesses = [
                [0.0, 0.0], [-0.5, -0.5], [0.5, 0.5],
                [-0.5, 0.5], [0.5, -0.5],
            ]
            for guess in initial_guesses:
                result = fsolve(residual, guess, full_output=True)
                xi, eta = result[0]
                info = result[1]
                residual_norm = info["fvec"][0] ** 2 + info["fvec"][1] ** 2
                if residual_norm < tol ** 2 * 100:
                    if abs(xi) <= 1.2 and abs(eta) <= 1.2:
                        return xi, eta, True
        except Exception:
            pass

        return None, None, False

    # ------------------------------------------------------------------
    def __call__(self, points):
        points = np.atleast_2d(points)
        result = np.full(len(points), np.nan, dtype=float)

        for i, (x, y) in enumerate(points):
            e, xi, eta = self._find_containing_element(x, y)
            if e is not None:
                N = np.array([
                    0.25 * (1 - xi) * (1 - eta),
                    0.25 * (1 + xi) * (1 - eta),
                    0.25 * (1 + xi) * (1 + eta),
                    0.25 * (1 - xi) * (1 + eta),
                ])
                elem_values = self.values[self.elements[e]]
                result[i] = N @ elem_values

        return result if len(result) > 1 else result[0]
