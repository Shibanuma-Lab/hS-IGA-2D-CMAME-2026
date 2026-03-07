"""
Interpolators for scattered 2D data.

Provides two interpolation methods:
1. LinearDelaunayInterpolator: Delaunay triangulation + linear interpolation
   (matches Mathematica's Interpolation[..., InterpolationOrder -> 1])
2. BilinearQuadInterpolator: Bilinear interpolation on quad meshes
   (for structured meshes with element connectivity)
"""

import numpy as np

try:
    from scipy.optimize import fsolve
    from scipy.spatial import Delaunay
except ImportError:
    fsolve = None
    Delaunay = None


class LinearDelaunayInterpolator:
    """
    Linear interpolator using Delaunay triangulation.
    
    Matches Mathematica's Interpolation[data, InterpolationOrder -> 1] behavior:
    - Performs Delaunay triangulation on scattered 2D points
    - Uses barycentric coordinates for linear interpolation within each triangle
    - Does not require element connectivity information
    
    Args:
        nodes: Array of shape (n_nodes, 2) containing (x, y) coordinates
        values: Array of shape (n_nodes,) containing values at each node
        name: Optional name for debugging output
    """
    
    def __init__(self, nodes, values, name="delaunay"):
        if Delaunay is None:
            raise ImportError("scipy.spatial.Delaunay is required for LinearDelaunayInterpolator")
        
        self.nodes = np.asarray(nodes, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.name = name
        
        if len(self.nodes.shape) != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (n_nodes, 2)")
        if len(self.values.shape) != 1:
            raise ValueError("values must have shape (n_nodes,)")
        if self.nodes.shape[0] != self.values.shape[0]:
            raise ValueError("nodes and values must have same length")
        
        # Perform Delaunay triangulation
        self.tri = Delaunay(self.nodes)
    
    def __call__(self, points):
        """
        Interpolate values at given points.
        
        Args:
            points: Array of shape (n_points, 2) or (2,) containing query points
        
        Returns:
            Array of interpolated values at query points
        """
        points = np.asarray(points, dtype=float)
        single_point = False
        
        if points.ndim == 1:
            if len(points) != 2:
                raise ValueError("Single point must have shape (2,)")
            points = points.reshape(1, 2)
            single_point = True
        elif points.ndim == 2:
            if points.shape[1] != 2:
                raise ValueError("points must have shape (n_points, 2)")
        else:
            raise ValueError("points must be 1D or 2D array")
        
        n_points = points.shape[0]
        result = np.zeros(n_points)
        
        # Find which simplex (triangle) each point belongs to
        simplex_indices = self.tri.find_simplex(points)
        
        for i in range(n_points):
            simplex_idx = simplex_indices[i]
            
            if simplex_idx == -1:
                # Point is outside the convex hull
                # Use nearest neighbor extrapolation
                distances = np.linalg.norm(self.nodes - points[i], axis=1)
                nearest_idx = np.argmin(distances)
                result[i] = self.values[nearest_idx]
            else:
                # Point is inside a triangle
                # Get the vertices of the containing triangle
                vertex_indices = self.tri.simplices[simplex_idx]
                
                # Get vertex coordinates
                vertices = self.nodes[vertex_indices]  # shape (3, 2)
                
                # Compute barycentric coordinates
                # For a triangle with vertices v0, v1, v2 and point p:
                # p = b0*v0 + b1*v1 + b2*v2, where b0 + b1 + b2 = 1
                v0 = vertices[0]
                v1 = vertices[1]
                v2 = vertices[2]
                p = points[i]
                
                # Solve the linear system:
                # [v1-v0, v2-v0] * [b1, b2]^T = p - v0
                A = np.column_stack([v1 - v0, v2 - v0])
                b = p - v0
                
                try:
                    bary = np.linalg.solve(A, b)
                    b1, b2 = bary[0], bary[1]
                    b0 = 1.0 - b1 - b2
                except np.linalg.LinAlgError:
                    # Degenerate triangle, use nearest neighbor
                    distances = np.linalg.norm(vertices - p, axis=1)
                    nearest_local = np.argmin(distances)
                    result[i] = self.values[vertex_indices[nearest_local]]
                    continue
                
                # Linear interpolation using barycentric coordinates
                values_at_vertices = self.values[vertex_indices]
                result[i] = b0 * values_at_vertices[0] + b1 * values_at_vertices[1] + b2 * values_at_vertices[2]
        
        return result[0] if single_point else result


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
    def precompute_point_map(self, points):
        """
        Precompute element IDs + bilinear shape weights for fixed query points.

        This is useful for time stepping where geometry is fixed and only nodal
        field values change between steps.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        npt = int(pts.shape[0])

        elem_ids = np.full(npt, -1, dtype=int)
        weights = np.zeros((npt, 4), dtype=float)
        nearest_ids = np.full(npt, -1, dtype=int)

        for i, (x, y) in enumerate(pts):
            e, xi, eta = self._find_containing_element(float(x), float(y))
            if e is not None:
                elem_ids[i] = int(e)
                weights[i, :] = np.array(
                    [
                        0.25 * (1 - xi) * (1 - eta),
                        0.25 * (1 + xi) * (1 - eta),
                        0.25 * (1 + xi) * (1 + eta),
                        0.25 * (1 - xi) * (1 + eta),
                    ],
                    dtype=float,
                )
            else:
                # Fallback for out-of-mesh points: nearest-node extrapolation
                d2 = np.sum((self.nodes - pts[i]) ** 2, axis=1)
                nearest_ids[i] = int(np.argmin(d2))

        return {
            "points": pts,
            "elem_ids": elem_ids,
            "weights": weights,
            "nearest_ids": nearest_ids,
        }

    # ------------------------------------------------------------------
    def evaluate_from_point_map(self, point_map, values):
        """Evaluate nodal field ``values`` using a map from ``precompute_point_map``."""
        vals = np.asarray(values, dtype=float)
        elem_ids = np.asarray(point_map["elem_ids"], dtype=int)
        weights = np.asarray(point_map["weights"], dtype=float)
        nearest_ids = np.asarray(point_map["nearest_ids"], dtype=int)

        if vals.ndim != 1:
            raise ValueError(f"values must be 1D, got shape={vals.shape}")

        out = np.full(len(elem_ids), np.nan, dtype=float)
        valid = elem_ids >= 0
        if np.any(valid):
            conn = self.elements[elem_ids[valid]]
            out[valid] = np.einsum("ij,ij->i", weights[valid], vals[conn])

        invalid = ~valid
        if np.any(invalid):
            nid = nearest_ids[invalid]
            ok = nid >= 0
            if np.any(ok):
                tmp = out[invalid]
                tmp[ok] = vals[nid[ok]]
                out[invalid] = tmp

        return out

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
