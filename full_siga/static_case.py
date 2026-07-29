"""
Static full s-IGA comparison utilities.

This module is intentionally isolated from the production hS-IGA path.  It
uses the existing global IGA mesh and solver, but replaces the local Q4
Lagrange correction mesh by a tensor-product quadratic B-spline patch.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

import core.state as st
from boundary.node_selector import find_nodes_on_extreme
from utils.mapping import jacobianPaPaMapping2d, parent2ParametricSpace
from utils.nurbs import BasisFuns, FindSpanMinus, NURBS2DBasisDers
from utils.shape_functions import GP, GW
from utils.static_crack import (
    exact_mode_i_displacement,
    exact_mode_i_stress_yy,
)


def _open_uniform_knot_vector(
    nelem: int,
    degree: int,
    *,
    c0_internal_indices: Iterable[int] = (),
) -> np.ndarray:
    """
    Build an open uniform knot vector.

    An internal knot has multiplicity ``degree`` when its element-boundary
    index is listed in ``c0_internal_indices``.  This gives C0 continuity
    there while preserving uniformly spaced nonzero knot spans.
    """
    nelem = int(nelem)
    degree = int(degree)
    if nelem < 1:
        raise ValueError(f"nelem must be positive, got {nelem}")
    if degree < 1:
        raise ValueError(f"degree must be positive, got {degree}")

    c0_indices = {int(value) for value in c0_internal_indices}
    if any(value <= 0 or value >= nelem for value in c0_indices):
        raise ValueError(
            f"C0 knot indices must lie in 1..{nelem - 1}, got {c0_indices}"
        )

    knots = [0.0] * (degree + 1)
    for index in range(1, nelem):
        multiplicity = degree if index in c0_indices else 1
        knots.extend([float(index) / float(nelem)] * multiplicity)
    knots.extend([1.0] * (degree + 1))
    return np.asarray(knots, dtype=float)


def _greville_abscissae(knots: Sequence[float], degree: int) -> np.ndarray:
    """Return Greville abscissae for a B-spline basis."""
    knots = np.asarray(knots, dtype=float)
    degree = int(degree)
    ncp = len(knots) - degree - 1
    return np.asarray(
        [np.mean(knots[index + 1:index + degree + 1]) for index in range(ncp)],
        dtype=float,
    )


def _unique_nondecreasing(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(dict.fromkeys(float(value) for value in values)), dtype=float)


def _surface_connectivity(
    span_u: int,
    span_v: int,
    degree_u: int,
    degree_v: int,
    ncp_u: int,
) -> np.ndarray:
    return np.asarray(
        [
            (span_u - degree_u + iu) + ncp_u * (span_v - degree_v + iv)
            for iv in range(degree_v + 1)
            for iu in range(degree_u + 1)
        ],
        dtype=int,
    )


def build_local_iga_mesh() -> None:
    """
    Build a uniform physical local IGA patch for the current static case.

    Two crack-tip variants are supported for the quadratic local basis:

    - ``C1``: a plain open-uniform knot vector.  No additional/repeated
      crack-tip knot is inserted, and no control-point column lies exactly at
      the tip.
    - ``C0``: one additional crack-tip knot is inserted, producing a tip
      control-point column and C0 continuity at that section.
    """
    p_local = int(getattr(st, "full_siga_local_p", st.p))
    q_local = int(getattr(st, "full_siga_local_q", st.q))
    requested_continuity = str(
        getattr(st, "full_siga_local_tip_continuity", "C1")
    ).upper()
    if requested_continuity not in {"C1", "C0"}:
        raise ValueError(
            "full_siga_local_tip_continuity must be 'C1' or 'C0'"
        )
    if p_local != 2:
        raise ValueError(
            "The C1/C0 reviewer comparison currently assumes a quadratic "
            f"local basis, got p={p_local}"
        )

    nelem_u = int(st.nLr)
    nelem_v = int(st.HL)
    if nelem_u % 2 != 0:
        raise ValueError(
            "The full s-IGA comparison requires an even local x-element count "
            "so the crack tip is a knot line."
        )

    tip_knot_index = int(st.aL)
    if tip_knot_index <= 0 or tip_knot_index >= nelem_u:
        raise ValueError(
            f"Invalid crack-tip knot index aL={tip_knot_index} for {nelem_u} elements"
        )

    c0_indices = (
        (tip_knot_index,) if requested_continuity == "C0" else ()
    )
    knot_u = _open_uniform_knot_vector(
        nelem_u,
        p_local,
        c0_internal_indices=c0_indices,
    )
    knot_v = _open_uniform_knot_vector(nelem_v, q_local)
    unique_u = _unique_nondecreasing(knot_u)
    unique_v = _unique_nondecreasing(knot_v)

    ncp_u = len(knot_u) - p_local - 1
    ncp_v = len(knot_v) - q_local - 1
    greville_u = _greville_abscissae(knot_u, p_local)
    greville_v = _greville_abscissae(knot_v, q_local)

    half_span = float(st.static_local_half_span)
    x_min = float(st.static_crack_tip_x) - half_span
    x_max = float(st.static_crack_tip_x) + half_span
    y_min = 0.0
    y_max = half_span

    cp_x = x_min + (x_max - x_min) * greville_u
    cp_y = y_min + (y_max - y_min) * greville_v
    control_points = np.asarray(
        [[x, y] for y in cp_y for x in cp_x],
        dtype=float,
    )
    weights = np.ones(ncp_u * ncp_v, dtype=float)

    ranges_u = [
        (float(unique_u[index]), float(unique_u[index + 1]))
        for index in range(len(unique_u) - 1)
    ]
    ranges_v = [
        (float(unique_v[index]), float(unique_v[index + 1]))
        for index in range(len(unique_v) - 1)
    ]
    spans_u = [
        FindSpanMinus(ncp_u - 1, p_local, 0.5 * (left + right), knot_u)
        for left, right in ranges_u
    ]
    spans_v = [
        FindSpanMinus(ncp_v - 1, q_local, 0.5 * (bottom + top), knot_v)
        for bottom, top in ranges_v
    ]

    element = []
    element_ranges = []
    element_spans = []
    for iv, span_v in enumerate(spans_v):
        for iu, span_u in enumerate(spans_u):
            element.append(
                _surface_connectivity(
                    span_u,
                    span_v,
                    p_local,
                    q_local,
                    ncp_u,
                )
            )
            element_ranges.append((ranges_u[iu], ranges_v[iv]))
            element_spans.append((span_u, span_v))

    vis_x = x_min + (x_max - x_min) * unique_u
    vis_y = y_min + (y_max - y_min) * unique_v
    vis_nodes = np.asarray([[x, y] for y in vis_y for x in vis_x], dtype=float)
    vis_nx = len(vis_x) - 1
    vis_ny = len(vis_y) - 1
    vis_elements = []
    for iy in range(vis_ny):
        for ix in range(vis_nx):
            lower_left = ix + (vis_nx + 1) * iy
            vis_elements.append(
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + (vis_nx + 1) + 1,
                    lower_left + (vis_nx + 1),
                ]
            )

    st.local_iga_p = p_local
    st.local_iga_q = q_local
    st.local_iga_knot_u = knot_u
    st.local_iga_knot_v = knot_v
    st.local_iga_unique_u = unique_u
    st.local_iga_unique_v = unique_v
    st.local_iga_ncp_u = ncp_u
    st.local_iga_ncp_v = ncp_v
    st.local_iga_weights = weights
    st.local_iga_control_points = control_points
    st.local_iga_element = np.asarray(element, dtype=int)
    st.local_iga_element_ranges = element_ranges
    st.local_iga_element_spans = element_spans
    st.local_iga_vis_nodes = vis_nodes
    st.local_iga_vis_elements = np.asarray(vis_elements, dtype=int)
    st.local_iga_bounds = (x_min, x_max, y_min, y_max)
    tip_param = float(tip_knot_index) / float(nelem_u)
    tip_knot_multiplicity = int(
        np.count_nonzero(
            np.isclose(knot_u, tip_param, rtol=0.0, atol=1.0e-14)
        )
    )
    actual_continuity = f"C{p_local - tip_knot_multiplicity}"
    if actual_continuity != requested_continuity:
        raise RuntimeError(
            "Unexpected crack-tip continuity: requested "
            f"{requested_continuity}, constructed {actual_continuity}"
        )
    st.local_iga_tip_param = tip_param
    st.local_iga_tip_knot_multiplicity = tip_knot_multiplicity
    st.local_iga_tip_continuity = actual_continuity
    st.local_iga_tip_extra_knot_count = max(
        tip_knot_multiplicity - 1,
        0,
    )

    # Compatibility aliases used by getresult() and the common linear solver.
    st.nodeL = control_points
    st.elemL = st.local_iga_element
    st.nodeLx = cp_x
    st.nodeLy = cp_y

    physical_dx = np.diff(x_min + (x_max - x_min) * unique_u)
    physical_dy = np.diff(y_min + (y_max - y_min) * unique_v)
    if not np.allclose(physical_dx, physical_dx[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Local IGA x knot spans are not uniform in physical space")
    if not np.allclose(physical_dy, physical_dy[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Local IGA y knot spans are not uniform in physical space")

    tip_columns = np.where(
        np.isclose(cp_x, float(st.static_crack_tip_x), rtol=0.0, atol=1.0e-12)
    )[0]
    expected_tip_columns = 1 if requested_continuity == "C0" else 0
    if len(tip_columns) != expected_tip_columns:
        raise RuntimeError(
            f"Expected {expected_tip_columns} local IGA control-point columns "
            f"at the crack tip for {requested_continuity}, got {tip_columns}"
        )
    st.local_iga_tip_columns = [int(value) for value in tip_columns]
    st.local_iga_tip_column = (
        int(tip_columns[0]) if len(tip_columns) == 1 else None
    )


def setup_combined_mesh_state() -> None:
    """Populate the mesh sizes required by the common coupled static solver."""
    st.ndof = 2
    st.nnmG = len(st.nodeG)
    st.nemG = len(st.elemG)
    st.neqG = st.ndof * st.nnmG
    st.enodeG = np.asarray(
        [[st.nodeG[index - 1] for index in elem] for elem in st.elemG],
        dtype=float,
    )

    st.nnmL = len(st.local_iga_control_points)
    st.nemL = len(st.local_iga_element)
    st.neqL = st.ndof * st.nnmL
    st.enodeL = np.asarray(
        [st.local_iga_control_points[conn] for conn in st.local_iga_element],
        dtype=float,
    )

    st.nnm = st.nnmG + st.nnmL
    st.nem = st.nemG + st.nemL
    st.neq = st.neqG + st.neqL
    st.nodeL2DAllMa2D.append(np.asarray(st.nodeL, dtype=float))


def _basis_data(
    *,
    degree_u: int,
    degree_v: int,
    knot_u: Sequence[float],
    knot_v: Sequence[float],
    weights: Sequence[float],
    control_points: np.ndarray,
    param_u: float,
    param_v: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Return connectivity, basis, physical derivatives, detJ, and position."""
    knot_u = np.asarray(knot_u, dtype=float)
    knot_v = np.asarray(knot_v, dtype=float)
    weights = np.asarray(weights, dtype=float)
    control_points = np.asarray(control_points, dtype=float)
    ncp_u = len(knot_u) - int(degree_u) - 1
    ncp_v = len(knot_v) - int(degree_v) - 1

    u_min = float(knot_u[degree_u])
    u_max = float(knot_u[ncp_u])
    v_min = float(knot_v[degree_v])
    v_max = float(knot_v[ncp_v])
    param_u = float(np.clip(param_u, u_min, u_max))
    param_v = float(np.clip(param_v, v_min, v_max))

    span_u = FindSpanMinus(ncp_u - 1, degree_u, param_u, knot_u)
    span_v = FindSpanMinus(ncp_v - 1, degree_v, param_v, knot_v)
    conn = _surface_connectivity(
        span_u,
        span_v,
        degree_u,
        degree_v,
        ncp_u,
    )
    basis, deriv_u, deriv_v = NURBS2DBasisDers(
        span_u,
        span_v,
        degree_u,
        degree_v,
        knot_u,
        knot_v,
        param_u,
        param_v,
        weights,
        ncp_u - 1,
        ncp_v - 1,
    )
    deriv_param = np.vstack([deriv_u, deriv_v])
    element_points = control_points[conn]
    jacobian = deriv_param @ element_points
    det_jacobian = float(np.linalg.det(jacobian))
    if det_jacobian <= 0.0:
        raise RuntimeError(
            f"Non-positive IGA geometry Jacobian det={det_jacobian} "
            f"at ({param_u}, {param_v})"
        )
    deriv_physical = np.linalg.inv(jacobian) @ deriv_param
    position = np.asarray(basis @ element_points, dtype=float)
    return conn, basis, deriv_physical, det_jacobian, position


def _b_matrix(deriv_physical: np.ndarray) -> np.ndarray:
    count = deriv_physical.shape[1]
    matrix = np.zeros((3, 2 * count), dtype=float)
    matrix[0, 0::2] = deriv_physical[0]
    matrix[1, 1::2] = deriv_physical[1]
    matrix[2, 0::2] = deriv_physical[1]
    matrix[2, 1::2] = deriv_physical[0]
    return matrix


def _dof_indices(connectivity: np.ndarray) -> np.ndarray:
    connectivity = np.asarray(connectivity, dtype=int)
    dofs = np.empty(2 * len(connectivity), dtype=int)
    dofs[0::2] = 2 * connectivity
    dofs[1::2] = 2 * connectivity + 1
    return dofs


def _sparse_from_element_blocks(
    row_blocks: list[np.ndarray],
    col_blocks: list[np.ndarray],
    data_blocks: list[np.ndarray],
    shape: tuple[int, int],
) -> sp.csr_matrix:
    if not data_blocks:
        return sp.csr_matrix(shape, dtype=float)
    rows = np.concatenate(row_blocks)
    cols = np.concatenate(col_blocks)
    data = np.concatenate(data_blocks)
    matrix = sp.coo_matrix((data, (rows, cols)), shape=shape, dtype=float).tocsr()
    matrix.eliminate_zeros()
    return matrix


def assemble_local_iga_stiffness() -> None:
    """Assemble the local IGA correction stiffness matrix KL."""
    order = int(getattr(st, "full_siga_local_ngp", st.local_iga_p + 1))
    points = np.asarray(GP(order), dtype=float)
    weights_1d = np.asarray(GW(order), dtype=float)

    row_blocks: list[np.ndarray] = []
    col_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []

    for range_u, range_v in st.local_iga_element_ranges:
        midpoint_u = 0.5 * (range_u[0] + range_u[1])
        midpoint_v = 0.5 * (range_v[0] + range_v[1])
        conn, _, _, _, _ = _basis_data(
            degree_u=st.local_iga_p,
            degree_v=st.local_iga_q,
            knot_u=st.local_iga_knot_u,
            knot_v=st.local_iga_knot_v,
            weights=st.local_iga_weights,
            control_points=st.local_iga_control_points,
            param_u=midpoint_u,
            param_v=midpoint_v,
        )
        element_stiffness = np.zeros((2 * len(conn), 2 * len(conn)), dtype=float)
        det_parent_to_param = jacobianPaPaMapping2d(range_u, range_v)

        for iv, eta_parent in enumerate(points):
            for iu, xi_parent in enumerate(points):
                param_u = parent2ParametricSpace(range_u, float(xi_parent))
                param_v = parent2ParametricSpace(range_v, float(eta_parent))
                gp_conn, _, deriv_physical, det_geometry, _ = _basis_data(
                    degree_u=st.local_iga_p,
                    degree_v=st.local_iga_q,
                    knot_u=st.local_iga_knot_u,
                    knot_v=st.local_iga_knot_v,
                    weights=st.local_iga_weights,
                    control_points=st.local_iga_control_points,
                    param_u=param_u,
                    param_v=param_v,
                )
                if not np.array_equal(gp_conn, conn):
                    raise RuntimeError("Local IGA Gauss point changed element connectivity")
                b_local = _b_matrix(deriv_physical)
                factor = (
                    float(weights_1d[iu])
                    * float(weights_1d[iv])
                    * det_geometry
                    * det_parent_to_param
                    * float(st.thi)
                )
                element_stiffness += (b_local.T @ st.de @ b_local) * factor

        dofs = _dof_indices(conn)
        row_blocks.append(np.repeat(dofs, len(dofs)))
        col_blocks.append(np.tile(dofs, len(dofs)))
        data_blocks.append(element_stiffness.ravel())

    st.KL = _sparse_from_element_blocks(
        row_blocks,
        col_blocks,
        data_blocks,
        (int(st.neqL), int(st.neqL)),
    )
    st.ML = None


def _global_element_rectangles() -> list[tuple[int, tuple[float, float], tuple[float, float]]]:
    rectangles = []
    width = float(st.static_width)
    height = float(st.static_height)
    for element_id, (index_u, index_v) in enumerate(st.index):
        range_u = st.elRangeU[int(index_u) - 1]
        range_v = st.elRangeV[int(index_v) - 1]
        rectangles.append(
            (
                element_id,
                (width * float(range_u[0]), width * float(range_u[1])),
                (height * float(range_v[0]), height * float(range_v[1])),
            )
        )
    return rectangles


def assemble_global_local_iga_coupling() -> None:
    """
    Assemble KGL by integrating intersections of local and global knot cells.

    Both geometries are affine and their nonzero knot spans are uniform in
    physical space.  Explicit cell intersections prevent a Gauss cell from
    crossing a basis-function knot line in either patch.
    """
    order = int(getattr(st, "static_kgl_ngpGL", 3))
    points = np.asarray(GP(order), dtype=float)
    weights_1d = np.asarray(GW(order), dtype=float)
    x_min, x_max, y_min, y_max = st.local_iga_bounds
    local_width = float(x_max - x_min)
    local_height = float(y_max - y_min)
    global_width = float(st.static_width)
    global_height = float(st.static_height)
    global_rectangles = _global_element_rectangles()
    tolerance = 1.0e-14

    row_blocks: list[np.ndarray] = []
    col_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    intersection_count = 0

    for local_range_u, local_range_v in st.local_iga_element_ranges:
        local_rect_x = (
            x_min + local_width * float(local_range_u[0]),
            x_min + local_width * float(local_range_u[1]),
        )
        local_rect_y = (
            y_min + local_height * float(local_range_v[0]),
            y_min + local_height * float(local_range_v[1]),
        )

        local_mid_u = 0.5 * (local_range_u[0] + local_range_u[1])
        local_mid_v = 0.5 * (local_range_v[0] + local_range_v[1])
        local_conn, _, _, _, _ = _basis_data(
            degree_u=st.local_iga_p,
            degree_v=st.local_iga_q,
            knot_u=st.local_iga_knot_u,
            knot_v=st.local_iga_knot_v,
            weights=st.local_iga_weights,
            control_points=st.local_iga_control_points,
            param_u=local_mid_u,
            param_v=local_mid_v,
        )
        local_dofs = _dof_indices(local_conn)

        for _, global_rect_x, global_rect_y in global_rectangles:
            intersection_x = (
                max(local_rect_x[0], global_rect_x[0]),
                min(local_rect_x[1], global_rect_x[1]),
            )
            intersection_y = (
                max(local_rect_y[0], global_rect_y[0]),
                min(local_rect_y[1], global_rect_y[1]),
            )
            if (
                intersection_x[1] - intersection_x[0] <= tolerance
                or intersection_y[1] - intersection_y[0] <= tolerance
            ):
                continue

            center_x = 0.5 * (intersection_x[0] + intersection_x[1])
            center_y = 0.5 * (intersection_y[0] + intersection_y[1])
            global_conn, _, _, _, _ = _basis_data(
                degree_u=int(st.p),
                degree_v=int(st.q),
                knot_u=st.uKnot,
                knot_v=st.vKnot,
                weights=st.weights,
                control_points=st.controlPts,
                param_u=center_x / global_width,
                param_v=center_y / global_height,
            )
            global_dofs = _dof_indices(global_conn)
            block = np.zeros((len(global_dofs), len(local_dofs)), dtype=float)
            det_parent_to_physical = (
                0.25
                * (intersection_x[1] - intersection_x[0])
                * (intersection_y[1] - intersection_y[0])
            )

            for iv, eta_parent in enumerate(points):
                for iu, xi_parent in enumerate(points):
                    x_coord = parent2ParametricSpace(
                        intersection_x, float(xi_parent)
                    )
                    y_coord = parent2ParametricSpace(
                        intersection_y, float(eta_parent)
                    )
                    gp_global_conn, _, deriv_global, _, _ = _basis_data(
                        degree_u=int(st.p),
                        degree_v=int(st.q),
                        knot_u=st.uKnot,
                        knot_v=st.vKnot,
                        weights=st.weights,
                        control_points=st.controlPts,
                        param_u=x_coord / global_width,
                        param_v=y_coord / global_height,
                    )
                    gp_local_conn, _, deriv_local, _, _ = _basis_data(
                        degree_u=st.local_iga_p,
                        degree_v=st.local_iga_q,
                        knot_u=st.local_iga_knot_u,
                        knot_v=st.local_iga_knot_v,
                        weights=st.local_iga_weights,
                        control_points=st.local_iga_control_points,
                        param_u=(x_coord - x_min) / local_width,
                        param_v=(y_coord - y_min) / local_height,
                    )
                    if not np.array_equal(gp_global_conn, global_conn):
                        raise RuntimeError(
                            "Coupling subcell crossed a global IGA knot line"
                        )
                    if not np.array_equal(gp_local_conn, local_conn):
                        raise RuntimeError(
                            "Coupling subcell crossed a local IGA knot line"
                        )
                    b_global = _b_matrix(deriv_global)
                    b_local = _b_matrix(deriv_local)
                    factor = (
                        float(weights_1d[iu])
                        * float(weights_1d[iv])
                        * det_parent_to_physical
                        * float(st.thi)
                    )
                    block += (b_global.T @ st.de @ b_local) * factor

            row_blocks.append(np.repeat(global_dofs, len(local_dofs)))
            col_blocks.append(np.tile(local_dofs, len(global_dofs)))
            data_blocks.append(block.ravel())
            intersection_count += 1

    st.KGL = _sparse_from_element_blocks(
        row_blocks,
        col_blocks,
        data_blocks,
        (int(st.neqG), int(st.neqL)),
    )
    st.MGL = None
    st.full_siga_coupling_intersections = int(intersection_count)


def apply_full_siga_static_boundary(
    *,
    ligament_fixity: str = "normal",
) -> None:
    """
    Apply analytical global BCs and local IGA correction-field constraints.

    The local correction is fixed in both components on the artificial left,
    right, and top patch boundaries.  Along y=0, crack-face control points
    behind the tip are released.  Ligament control points have their normal
    (y) correction fixed; a C0 mesh also has an exact crack-tip control point
    that is fixed.  A plain C1 mesh has no exact tip control point, so basis
    support necessarily crosses the boundary-condition transition.
    ``ligament_fixity='xy'`` can fix both correction components for a literal
    fully-fixed variant.
    """
    ligament_fixity = str(ligament_fixity).lower()
    if ligament_fixity not in {"normal", "xy"}:
        raise ValueError("ligament_fixity must be 'normal' or 'xy'")

    coord_tol = getattr(st, "boundary_coord_tol", None)
    node_global = np.asarray(st.nodeG, dtype=float)
    left_nodes = find_nodes_on_extreme(
        node_global, axis=0, side="min", atol=coord_tol
    )
    right_nodes = find_nodes_on_extreme(
        node_global, axis=0, side="max", atol=coord_tol
    )
    top_nodes = find_nodes_on_extreme(
        node_global, axis=1, side="max", atol=coord_tol
    )
    down_nodes = find_nodes_on_extreme(
        node_global, axis=1, side="min", atol=coord_tol
    )
    prescribed_global = np.unique(
        np.concatenate([left_nodes, right_nodes, top_nodes])
    ).astype(int)

    counter = 0
    for node_id in down_nodes:
        counter += 1
        if node_global[int(node_id), 0] >= float(st.static_crack_tip_x):
            break
    cut_count = max(counter - int(st.p), 0)
    cut_nodes = set(int(node_id) for node_id in down_nodes[:cut_count])
    yfix_global = sorted(
        set(int(node_id) for node_id in down_nodes)
        - cut_nodes
        - set(int(node_id) for node_id in prescribed_global)
    )

    bc_map: dict[tuple[int, int], float] = {}
    for node_id in prescribed_global:
        displacement = exact_mode_i_displacement(
            node_global[int(node_id)],
            st.static_crack_tip_x,
            st.mu,
            st.kappa,
        )
        bc_map[(int(node_id), 1)] = float(displacement[0])
        bc_map[(int(node_id), 2)] = float(displacement[1])
    for node_id in yfix_global:
        bc_map[(int(node_id), 2)] = 0.0

    ncp_u = int(st.local_iga_ncp_u)
    ncp_v = int(st.local_iga_ncp_v)
    local_cp = np.asarray(st.local_iga_control_points, dtype=float)
    left_local = {row * ncp_u for row in range(ncp_v)}
    right_local = {row * ncp_u + ncp_u - 1 for row in range(ncp_v)}
    top_local = {
        (ncp_v - 1) * ncp_u + column for column in range(ncp_u)
    }
    outer_local = left_local | right_local | top_local

    tip_x = float(st.static_crack_tip_x)
    tolerance = 1.0e-12
    bottom_local = set(range(ncp_u))
    crack_surface_local = {
        index
        for index in bottom_local
        if local_cp[index, 0] < tip_x - tolerance
    }
    tip_local = {
        index
        for index in bottom_local
        if abs(local_cp[index, 0] - tip_x) <= tolerance
    }
    ligament_local = {
        index
        for index in bottom_local
        if local_cp[index, 0] > tip_x + tolerance
    }
    tip_and_ligament = tip_local | ligament_local

    for local_id in outer_local:
        combined_id = int(st.nnmG) + int(local_id)
        bc_map[(combined_id, 1)] = 0.0
        bc_map[(combined_id, 2)] = 0.0
    for local_id in tip_and_ligament:
        combined_id = int(st.nnmG) + int(local_id)
        bc_map[(combined_id, 2)] = 0.0
        if ligament_fixity == "xy":
            bc_map[(combined_id, 1)] = 0.0

    tip_span = FindSpanMinus(
        ncp_u - 1,
        int(st.local_iga_p),
        float(st.local_iga_tip_param),
        st.local_iga_knot_u,
    )
    tip_conn = np.arange(
        tip_span - int(st.local_iga_p),
        tip_span + 1,
        dtype=int,
    )
    tip_basis = np.asarray(
        BasisFuns(
            tip_span,
            float(st.local_iga_tip_param),
            int(st.local_iga_p),
            st.local_iga_knot_u,
        ),
        dtype=float,
    )
    nonzero_tip_support = [
        int(local_id)
        for local_id, value in zip(tip_conn, tip_basis)
        if abs(float(value)) > 1.0e-12
    ]

    st.ebc = np.asarray(
        [
            [node_id, direction, value]
            for (node_id, direction), value in sorted(bc_map.items())
        ],
        dtype=float,
    )
    st.nbc = np.empty((0, 3), dtype=float)

    st.full_siga_ligament_fixity = ligament_fixity
    st.full_siga_local_outer_cp = sorted(outer_local)
    st.full_siga_local_crack_surface_cp = sorted(crack_surface_local)
    st.full_siga_local_tip_cp = sorted(tip_local)
    st.full_siga_local_ligament_cp = sorted(ligament_local)
    st.full_siga_tip_nonzero_basis_cp = nonzero_tip_support
    st.full_siga_tip_nonzero_basis_values = [
        float(value)
        for value in tip_basis
        if abs(float(value)) > 1.0e-12
    ]
    st.full_siga_tip_nonzero_basis_fix_y = [
        int((int(st.nnmG) + local_id, 2) in bc_map)
        for local_id in nonzero_tip_support
    ]
    st.full_siga_tip_bc_transition_exact = bool(
        st.local_iga_tip_continuity == "C0"
        and len(tip_local) == 1
    )
    st.full_siga_bc_map = bc_map


def _surface_field(
    *,
    degree_u: int,
    degree_v: int,
    knot_u: Sequence[float],
    knot_v: Sequence[float],
    weights: Sequence[float],
    control_points: np.ndarray,
    displacement: np.ndarray,
    param_u: float,
    param_v: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conn, basis, deriv_physical, _, position = _basis_data(
        degree_u=degree_u,
        degree_v=degree_v,
        knot_u=knot_u,
        knot_v=knot_v,
        weights=weights,
        control_points=control_points,
        param_u=param_u,
        param_v=param_v,
    )
    element_displacement = np.asarray(displacement[conn], dtype=float)
    physical_displacement = np.asarray(basis @ element_displacement, dtype=float)
    strain = _b_matrix(deriv_physical) @ element_displacement.reshape(-1)
    return position, physical_displacement, np.asarray(strain, dtype=float)


def _global_field_at(x_coord: float, y_coord: float):
    return _surface_field(
        degree_u=int(st.p),
        degree_v=int(st.q),
        knot_u=st.uKnot,
        knot_v=st.vKnot,
        weights=st.weights,
        control_points=st.controlPts,
        displacement=st.disG2D,
        param_u=float(x_coord) / float(st.static_width),
        param_v=float(y_coord) / float(st.static_height),
    )


def _local_field_at(x_coord: float, y_coord: float):
    x_min, x_max, y_min, y_max = st.local_iga_bounds
    return _surface_field(
        degree_u=int(st.local_iga_p),
        degree_v=int(st.local_iga_q),
        knot_u=st.local_iga_knot_u,
        knot_v=st.local_iga_knot_v,
        weights=st.local_iga_weights,
        control_points=st.local_iga_control_points,
        displacement=st.disL2D,
        param_u=(float(x_coord) - x_min) / (x_max - x_min),
        param_v=(float(y_coord) - y_min) / (y_max - y_min),
    )


def _project_ligament_reaction_stress(
    sample_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Recover ligament traction from generalized IGA control-point reactions.

    For the constrained bottom basis coefficients, ``r_i`` represents a
    generalized boundary force rather than a point traction.  We solve

        integral N_i N_j dx * c_j = -r_i

    on the ligament and then evaluate ``N_j(x)c_j`` at the requested points.
    This is the IGA counterpart of the nodal-reaction recovery used by the
    existing hS-IGA static benchmark.  It is not the same smoothing operator
    as Q4 nodal lumping.  In the C1 variant, basis functions on the ligament
    also involve released crack-face coefficients, so this projection is only
    diagnostic.  Direct total-field stress is the primary C1 result.
    """
    coupled = sp.bmat(
        [[st.KG, st.KGL], [st.KGL.T, st.KL]],
        format="csr",
    )
    residual = np.asarray(coupled @ st.dis - st.force, dtype=float)
    residual_local = residual[int(st.neqG):]

    p_local = int(st.local_iga_p)
    knot_u = np.asarray(st.local_iga_knot_u, dtype=float)
    ncp_u = int(st.local_iga_ncp_u)
    tip_param = float(st.local_iga_tip_param)
    unique_u = np.asarray(st.local_iga_unique_u, dtype=float)
    right_ranges = [
        (float(unique_u[index]), float(unique_u[index + 1]))
        for index in range(len(unique_u) - 1)
        if float(unique_u[index]) >= tip_param - 1.0e-14
    ]

    active_cp: set[int] = set()
    range_connectivity: list[tuple[tuple[float, float], np.ndarray]] = []
    for param_range in right_ranges:
        midpoint = 0.5 * (param_range[0] + param_range[1])
        span = FindSpanMinus(ncp_u - 1, p_local, midpoint, knot_u)
        conn = np.arange(span - p_local, span + 1, dtype=int)
        active_cp.update(int(value) for value in conn)
        range_connectivity.append((param_range, conn))

    active = np.asarray(sorted(active_cp), dtype=int)
    active_position = {int(cp): index for index, cp in enumerate(active)}
    mass = np.zeros((len(active), len(active)), dtype=float)
    points = np.asarray(GP(p_local + 1), dtype=float)
    weights = np.asarray(GW(p_local + 1), dtype=float)
    local_width = float(st.local_iga_bounds[1] - st.local_iga_bounds[0])

    for param_range, conn in range_connectivity:
        span = int(conn[-1])
        local_ids = np.asarray(
            [active_position[int(cp)] for cp in conn],
            dtype=int,
        )
        block = np.zeros((len(conn), len(conn)), dtype=float)
        jacobian = (
            0.5
            * (param_range[1] - param_range[0])
            * local_width
            * float(st.thi)
        )
        for point, weight in zip(points, weights):
            param_u = parent2ParametricSpace(param_range, float(point))
            basis = np.asarray(
                BasisFuns(span, param_u, p_local, knot_u),
                dtype=float,
            )
            block += np.outer(basis, basis) * jacobian * float(weight)
        mass[np.ix_(local_ids, local_ids)] += block

    reaction_y = np.asarray(
        [residual_local[2 * int(cp) + 1] for cp in active],
        dtype=float,
    )
    constrained_active = np.asarray(
        [
            (int(st.nnmG) + int(cp), 2) in st.full_siga_bc_map
            for cp in active
        ],
        dtype=bool,
    )
    consistent_coefficients = np.linalg.solve(mass, -reaction_y)
    lumped_measure = np.sum(mass, axis=1)
    lumped_coefficients = np.divide(
        -reaction_y,
        lumped_measure,
        out=np.zeros_like(reaction_y),
        where=np.abs(lumped_measure) > 1.0e-16,
    )

    x_min, x_max, _, _ = st.local_iga_bounds
    recovered_lumped = np.zeros(len(sample_x), dtype=float)
    recovered_consistent = np.zeros(len(sample_x), dtype=float)
    for index, x_coord in enumerate(sample_x):
        param_u = (float(x_coord) - x_min) / (x_max - x_min)
        span = FindSpanMinus(ncp_u - 1, p_local, param_u, knot_u)
        conn = np.arange(span - p_local, span + 1, dtype=int)
        basis = np.asarray(
            BasisFuns(span, param_u, p_local, knot_u),
            dtype=float,
        )
        consistent_local = np.asarray(
            [consistent_coefficients[active_position[int(cp)]] for cp in conn],
            dtype=float,
        )
        lumped_local = np.asarray(
            [lumped_coefficients[active_position[int(cp)]] for cp in conn],
            dtype=float,
        )
        recovered_consistent[index] = float(basis @ consistent_local)
        recovered_lumped[index] = float(basis @ lumped_local)
    unconstrained_active = active[~constrained_active]
    diagnostics = {
        "active_bottom_control_point_count": int(len(active)),
        "constrained_active_control_point_count": int(
            np.count_nonzero(constrained_active)
        ),
        "unconstrained_active_control_point_ids": [
            int(value) for value in unconstrained_active
        ],
        "basis_support_crosses_tip_bc_transition": bool(
            len(unconstrained_active) > 0
        ),
        "valid_as_pure_constrained_reaction_projection": bool(
            len(unconstrained_active) == 0
        ),
        "unconstrained_active_residual_max_abs": (
            float(
                np.max(
                    np.abs(
                        reaction_y[~constrained_active]
                    )
                )
            )
            if len(unconstrained_active) > 0
            else 0.0
        ),
    }
    return recovered_lumped, recovered_consistent, diagnostics


def compute_full_siga_normalized_stress_yy() -> dict:
    """
    Recover normalized ligament stress with two explicitly named operators.

    ``direct_normalized_stress_yy`` evaluates the same physical quantity
    ``D (B_G u_G + B_L u_L)`` and is the recommended result for a strict
    hS-IGA/full-s-IGA stress-field comparison, provided the hS-IGA curve is
    evaluated with the same direct operator.

    ``lumped_projected_normalized_stress_yy`` is a useful B-spline analogue
    of the legacy hS-IGA nodal-reaction curve for C0.  It is diagnostic only
    for C1 because the trace basis crosses the crack-face/ligament boundary
    transition.  The shorter ``normalized_stress_yy`` key is retained as an
    output-format compatibility alias for this projected result.
    """
    x_min, x_max, _, _ = st.local_iga_bounds
    tip_x = float(st.static_crack_tip_x)
    knot_x = x_min + (x_max - x_min) * np.asarray(
        st.local_iga_unique_u, dtype=float
    )
    sample_x = knot_x[(knot_x > tip_x + 1.0e-12) & (knot_x < x_max - 1.0e-12)]

    direct_stress = []
    exact = []
    for x_coord in sample_x:
        _, _, strain_global = _global_field_at(float(x_coord), 0.0)
        _, _, strain_local = _local_field_at(float(x_coord), 0.0)
        stress = np.asarray(st.de @ (strain_global + strain_local), dtype=float)
        stress_exact = float(
            exact_mode_i_stress_yy(
                (float(x_coord), 0.0),
                float(st.static_crack_tip_x),
            )
        )
        direct_stress.append(float(stress[1]))
        exact.append(stress_exact)

    (
        projected_stress,
        consistent_projected_stress,
        reaction_projection_diagnostics,
    ) = (
        _project_ligament_reaction_stress(sample_x)
    )
    exact_array = np.asarray(exact, dtype=float)
    projected_normalized = np.divide(
        projected_stress,
        exact_array,
        out=np.full_like(projected_stress, np.nan),
        where=np.abs(exact_array) > 1.0e-14,
    )
    direct_array = np.asarray(direct_stress, dtype=float)
    direct_normalized = np.divide(
        direct_array,
        exact_array,
        out=np.full_like(direct_array, np.nan),
        where=np.abs(exact_array) > 1.0e-14,
    )
    consistent_projected_normalized = np.divide(
        consistent_projected_stress,
        exact_array,
        out=np.full_like(consistent_projected_stress, np.nan),
        where=np.abs(exact_array) > 1.0e-14,
    )

    _, tip_global_displacement, _ = _global_field_at(tip_x, 0.0)
    _, tip_local_correction, _ = _local_field_at(tip_x, 0.0)
    tip_total_displacement = tip_global_displacement + tip_local_correction
    tip_support_ids = [
        int(value) for value in st.full_siga_tip_nonzero_basis_cp
    ]
    crack_surface_ids = set(st.full_siga_local_crack_surface_cp)
    exact_tip_ids = set(st.full_siga_local_tip_cp)
    ligament_ids = set(st.full_siga_local_ligament_cp)
    tip_support_classes = []
    for local_id in tip_support_ids:
        if local_id in crack_surface_ids:
            tip_support_classes.append("crack_surface")
        elif local_id in exact_tip_ids:
            tip_support_classes.append("crack_tip")
        elif local_id in ligament_ids:
            tip_support_classes.append("ligament")
        else:
            tip_support_classes.append("unclassified")

    tip_boundary_diagnostics = {
        "continuity": str(st.local_iga_tip_continuity),
        "knot_multiplicity": int(st.local_iga_tip_knot_multiplicity),
        "extra_tip_knots_inserted": int(
            st.local_iga_tip_extra_knot_count
        ),
        "exact_tip_control_point_count": int(
            len(st.full_siga_local_tip_cp)
        ),
        "bc_transition_exactly_representable": bool(
            st.full_siga_tip_bc_transition_exact
        ),
        "nonzero_basis_control_point_ids_at_tip": tip_support_ids,
        "nonzero_basis_values_at_tip": [
            float(value)
            for value in st.full_siga_tip_nonzero_basis_values
        ],
        "nonzero_basis_control_point_x_at_tip": [
            float(st.local_iga_control_points[local_id, 0])
            for local_id in tip_support_ids
        ],
        "nonzero_basis_control_point_classes_at_tip": (
            tip_support_classes
        ),
        "nonzero_basis_control_point_fix_y_at_tip": [
            int(value)
            for value in st.full_siga_tip_nonzero_basis_fix_y
        ],
        "global_displacement_at_tip": (
            np.asarray(tip_global_displacement, dtype=float).tolist()
        ),
        "local_correction_displacement_at_tip": (
            np.asarray(tip_local_correction, dtype=float).tolist()
        ),
        "total_displacement_at_tip": (
            np.asarray(tip_total_displacement, dtype=float).tolist()
        ),
        "exact_displacement_at_tip": [0.0, 0.0],
    }

    return {
        "case_name": str(st.static_case_label),
        "hG": float(st.static_case_hG),
        "hL": float(st.static_case_hL),
        "rGL": float(st.rGL),
        "nGx": int(st.static_case_nGx),
        "nGy": int(st.static_case_nGy),
        "nhL": int(st.static_case_nhL),
        "dof": int(st.neq),
        "local_tip_continuity": str(st.local_iga_tip_continuity),
        "crack_tip_boundary_diagnostics": tip_boundary_diagnostics,
        "reaction_projection_diagnostics": (
            reaction_projection_diagnostics
        ),
        "stress_recovery": "iga_boundary_reaction_lumped_projection",
        "recommended_cross_formulation_stress_recovery": (
            "direct_total_field_D_times_BG_uG_plus_BL_uL"
        ),
        "native_reaction_stress_recovery": (
            "iga_boundary_reaction_lumped_projection"
        ),
        "reaction_projection_role": (
            "diagnostic_only_due_to_C1_tip_BC_support_overlap"
            if st.local_iga_tip_continuity == "C1"
            else "native_C0_boundary_reaction_analogue"
        ),
        "sample_x": np.asarray(sample_x, dtype=float).tolist(),
        "distance_from_tip": (np.asarray(sample_x) - tip_x).tolist(),
        "lumped_projected_stress_yy": projected_stress.tolist(),
        "lumped_projected_normalized_stress_yy": (
            projected_normalized.tolist()
        ),
        # Backward-compatible aliases matching the existing static CSV name.
        "stress_yy": projected_stress.tolist(),
        "exact_stress_yy": exact,
        "normalized_stress_yy": projected_normalized.tolist(),
        "direct_stress_yy": direct_stress,
        "direct_normalized_stress_yy": direct_normalized.tolist(),
        "consistent_projected_stress_yy": consistent_projected_stress.tolist(),
        "consistent_projected_normalized_stress_yy": (
            consistent_projected_normalized.tolist()
        ),
    }


def _point_fields_for_vtu(
    points: np.ndarray,
    *,
    include_local: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    displacement = np.zeros((len(points), 3), dtype=float)
    stress = np.zeros((len(points), 3), dtype=float)
    stress_diff = np.zeros(len(points), dtype=float)
    normalized = np.full(len(points), np.nan, dtype=float)

    for index, point in enumerate(points):
        _, disp_global, strain_global = _global_field_at(point[0], point[1])
        total_disp = disp_global.copy()
        total_strain = strain_global.copy()
        if include_local:
            _, disp_local, strain_local = _local_field_at(point[0], point[1])
            total_disp += disp_local
            total_strain += strain_local

        total_stress = np.asarray(st.de @ total_strain, dtype=float)
        exact_yy = float(
            exact_mode_i_stress_yy(point, float(st.static_crack_tip_x))
        )
        displacement[index, :2] = total_disp
        stress[index, :] = total_stress
        stress_diff[index] = total_stress[1] - exact_yy
        if abs(exact_yy) > 1.0e-14:
            normalized[index] = total_stress[1] / exact_yy

    return displacement, stress, stress_diff, normalized


def write_full_siga_vtu(case_dir: Path, step: int) -> Path:
    """
    Write one combined global/local visualization VTU file.

    The two visual meshes overlap.  Global cells contain the global field
    only, while local cells contain the total G+L field.  Point- and cell-data
    flags are included so ParaView can select the local total-field patch.
    """
    case_dir = Path(case_dir)
    global_points = np.asarray(st.nodeVis, dtype=float)
    global_cells = np.asarray(st.elemVis, dtype=int)
    local_points = np.asarray(st.local_iga_vis_nodes, dtype=float)
    local_cells = np.asarray(st.local_iga_vis_elements, dtype=int)

    global_disp, global_stress, global_diff, global_norm = _point_fields_for_vtu(
        global_points,
        include_local=False,
    )
    local_disp, local_stress, local_diff, local_norm = _point_fields_for_vtu(
        local_points,
        include_local=True,
    )

    points_2d = np.vstack([global_points, local_points])
    points = np.column_stack([points_2d, np.zeros(len(points_2d), dtype=float)])
    cells = np.vstack([global_cells, local_cells + len(global_points)])
    disp = np.vstack([global_disp, local_disp])
    stress = np.vstack([global_stress, local_stress])
    stress_diff = np.concatenate([global_diff, local_diff])
    normalized = np.concatenate([global_norm, local_norm])
    is_local_point = np.concatenate(
        [
            np.zeros(len(global_points), dtype=int),
            np.ones(len(local_points), dtype=int),
        ]
    )
    is_local_cell = np.concatenate(
        [
            np.zeros(len(global_cells), dtype=int),
            np.ones(len(local_cells), dtype=int),
        ]
    )
    offsets = 4 * np.arange(1, len(cells) + 1, dtype=int)
    cell_types = np.full(len(cells), 9, dtype=int)

    output = case_dir / f"step_{int(step):05d}_full_siga.vtu"
    with output.open("w") as stream:
        stream.write('<?xml version="1.0"?>\n')
        stream.write(
            '<VTKFile type="UnstructuredGrid" version="0.1" '
            'byte_order="LittleEndian">\n'
        )
        stream.write("  <UnstructuredGrid>\n")
        stream.write(
            f'    <Piece NumberOfPoints="{len(points)}" '
            f'NumberOfCells="{len(cells)}">\n'
        )
        stream.write("      <Points>\n")
        stream.write(
            '        <DataArray type="Float64" NumberOfComponents="3" '
            'format="ascii">\n'
        )
        for point in points:
            stream.write(f"          {point[0]:.12e} {point[1]:.12e} {point[2]:.12e}\n")
        stream.write("        </DataArray>\n")
        stream.write("      </Points>\n")
        stream.write("      <Cells>\n")
        stream.write(
            '        <DataArray type="Int32" Name="connectivity" format="ascii">\n'
        )
        for cell in cells:
            stream.write("          " + " ".join(str(int(value)) for value in cell) + "\n")
        stream.write("        </DataArray>\n")
        stream.write(
            '        <DataArray type="Int32" Name="offsets" format="ascii">\n'
        )
        stream.write("          " + " ".join(str(int(value)) for value in offsets) + "\n")
        stream.write("        </DataArray>\n")
        stream.write(
            '        <DataArray type="UInt8" Name="types" format="ascii">\n'
        )
        stream.write("          " + " ".join(str(int(value)) for value in cell_types) + "\n")
        stream.write("        </DataArray>\n")
        stream.write("      </Cells>\n")
        stream.write("      <PointData>\n")

        def write_vector(name: str, values: np.ndarray) -> None:
            stream.write(
                f'        <DataArray type="Float64" Name="{name}" '
                'NumberOfComponents="3" format="ascii">\n'
            )
            for value in values:
                stream.write(
                    f"          {value[0]:.12e} {value[1]:.12e} {value[2]:.12e}\n"
                )
            stream.write("        </DataArray>\n")

        def write_scalar(name: str, values: np.ndarray, data_type: str = "Float64") -> None:
            stream.write(
                f'        <DataArray type="{data_type}" Name="{name}" format="ascii">\n'
            )
            for value in values:
                if data_type == "Int32":
                    stream.write(f"          {int(value)}\n")
                else:
                    stream.write(f"          {float(value):.12e}\n")
            stream.write("        </DataArray>\n")

        write_vector("displacement", disp)
        write_vector("direct_stress_xx_yy_xy", stress)
        write_scalar("direct_stress_xx", stress[:, 0])
        write_scalar("direct_stress_yy", stress[:, 1])
        write_scalar("direct_stress_xy", stress[:, 2])
        write_scalar("direct_stress_yy_difference_from_exact", stress_diff)
        write_scalar("direct_normalized_stress_yy", normalized)
        write_scalar("is_local_patch", is_local_point, data_type="Int32")
        write_scalar("is_total_field", is_local_point, data_type="Int32")
        stream.write("      </PointData>\n")
        stream.write("      <CellData>\n")
        write_scalar(
            "is_local_patch_cell",
            is_local_cell,
            data_type="Int32",
        )
        write_scalar(
            "is_total_field_cell",
            is_local_cell,
            data_type="Int32",
        )
        stream.write("      </CellData>\n")
        stream.write("    </Piece>\n")
        stream.write("  </UnstructuredGrid>\n")
        stream.write("</VTKFile>\n")
    return output


def _local_bc_rows() -> list[dict]:
    outer = set(st.full_siga_local_outer_cp)
    crack = set(st.full_siga_local_crack_surface_cp)
    tip = set(st.full_siga_local_tip_cp)
    ligament = set(st.full_siga_local_ligament_cp)
    tip_basis_by_id = dict(
        zip(
            st.full_siga_tip_nonzero_basis_cp,
            st.full_siga_tip_nonzero_basis_values,
        )
    )
    rows = []
    for local_id, point in enumerate(st.local_iga_control_points):
        combined_id = int(st.nnmG) + int(local_id)
        rows.append(
            {
                "local_control_point_id": int(local_id),
                "combined_node_id": int(combined_id),
                "x": float(point[0]),
                "y": float(point[1]),
                "is_outer_boundary": int(local_id in outer),
                "is_crack_surface": int(local_id in crack),
                "is_crack_tip": int(local_id in tip),
                "is_ligament": int(local_id in ligament),
                "is_nonzero_basis_at_tip": int(
                    local_id in tip_basis_by_id
                ),
                "basis_value_at_tip": float(
                    tip_basis_by_id.get(local_id, 0.0)
                ),
                "fix_x": int((combined_id, 1) in st.full_siga_bc_map),
                "fix_y": int((combined_id, 2) in st.full_siga_bc_map),
            }
        )
    return rows


def write_full_siga_case_outputs(case_dir: Path, result: dict) -> None:
    """Write stress results, mesh metadata, and a control-point BC audit."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    physical_local_span_x = (
        float(st.local_iga_bounds[1] - st.local_iga_bounds[0])
        / float(st.nLr)
    )
    legacy_length_scale_ratio = physical_local_span_x / float(st.hL)

    with (case_dir / "full_siga_case_result.json").open("w") as stream:
        json.dump(result, stream, indent=2)
    with (case_dir / "crack_tip_boundary_diagnostics.json").open(
        "w"
    ) as stream:
        json.dump(
            result["crack_tip_boundary_diagnostics"],
            stream,
            indent=2,
        )

    with (case_dir / "normalized_stress_yy.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case_name", "dof"]
            + [f"point_{index + 1}" for index in range(len(result["normalized_stress_yy"]))]
        )
        writer.writerow(
            [result["case_name"], result["dof"], *result["normalized_stress_yy"]]
        )

    with (case_dir / "lumped_projected_normalized_stress_yy.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case_name", "dof"]
            + [
                f"point_{index + 1}"
                for index in range(
                    len(result["lumped_projected_normalized_stress_yy"])
                )
            ]
        )
        writer.writerow(
            [
                result["case_name"],
                result["dof"],
                *result["lumped_projected_normalized_stress_yy"],
            ]
        )

    with (case_dir / "direct_normalized_stress_yy.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case_name", "dof"]
            + [
                f"point_{index + 1}"
                for index in range(len(result["direct_normalized_stress_yy"]))
            ]
        )
        writer.writerow(
            [
                result["case_name"],
                result["dof"],
                *result["direct_normalized_stress_yy"],
            ]
        )

    with (case_dir / "normalized_stress_yy_detailed.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case_name",
                "dof",
                "point_id",
                "x",
                "distance_from_tip",
                "lumped_projected_stress_yy",
                "exact_stress_yy",
                "lumped_projected_normalized_stress_yy",
                "direct_stress_yy",
                "direct_normalized_stress_yy",
                "consistent_projected_stress_yy",
                "consistent_projected_normalized_stress_yy",
            ]
        )
        for index, values in enumerate(
            zip(
                result["sample_x"],
                result["distance_from_tip"],
                result["stress_yy"],
                result["exact_stress_yy"],
                result["normalized_stress_yy"],
                result["direct_stress_yy"],
                result["direct_normalized_stress_yy"],
                result["consistent_projected_stress_yy"],
                result["consistent_projected_normalized_stress_yy"],
            ),
            start=1,
        ):
            writer.writerow(
                [result["case_name"], result["dof"], index, *values]
            )

    bc_rows = _local_bc_rows()
    with (case_dir / "local_control_point_boundary_conditions.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bc_rows)

    metadata = {
        "formulation": "full_s_iga_static_reviewer_comparison",
        "global_degree": [int(st.p), int(st.q)],
        "local_degree": [int(st.local_iga_p), int(st.local_iga_q)],
        "local_elements": [int(st.nLr), int(st.HL)],
        "local_control_points": [
            int(st.local_iga_ncp_u),
            int(st.local_iga_ncp_v),
        ],
        "local_physical_bounds": [float(value) for value in st.local_iga_bounds],
        "uniform_physical_knot_spans": True,
        "local_physical_knot_span_x": physical_local_span_x,
        "nominal_hL": float(st.hL),
        "physical_span_over_nominal_hL": legacy_length_scale_ratio,
        "crack_tip_parameter": float(st.local_iga_tip_param),
        "crack_tip_knot_multiplicity": int(
            st.local_iga_tip_knot_multiplicity
        ),
        "crack_tip_extra_knots_inserted": int(
            st.local_iga_tip_extra_knot_count
        ),
        "crack_tip_continuity": str(st.local_iga_tip_continuity),
        "crack_tip_control_point_count": int(
            len(st.full_siga_local_tip_cp)
        ),
        "crack_tip_bc_transition_policy": (
            "split_bottom_control_coefficients_by_Greville_abscissa"
        ),
        "crack_tip_bc_transition_exactly_representable": bool(
            st.full_siga_tip_bc_transition_exact
        ),
        "plain_open_uniform_local_knot_vector": bool(
            st.local_iga_tip_extra_knot_count == 0
        ),
        "ligament_fixity": str(st.full_siga_ligament_fixity),
        "coupling_gauss_points_per_direction": int(st.static_kgl_ngpGL),
        "coupling_cell_intersections": int(st.full_siga_coupling_intersections),
        "recommended_cross_formulation_stress_recovery": (
            "direct_total_field_D_times_BG_uG_plus_BL_uL"
        ),
        "native_reaction_stress_recovery": (
            "iga_boundary_reaction_lumped_projection"
        ),
        "reaction_projection_role": result["reaction_projection_role"],
        "reaction_projection_diagnostics": result[
            "reaction_projection_diagnostics"
        ],
        "diagnostic_stress_recovery": "iga_boundary_reaction_consistent_l2_projection",
        "normalized_stress_yy_csv_semantics": (
            "alias_of_lumped_projected_normalized_stress_yy"
        ),
        "legacy_hs_reaction_length_note": (
            "The existing hS-IGA metric divides Q4 reactions by nominal hL, "
            "whereas the physical local edge length is "
            "static_local_half_span/lL. Do not interpret the two native "
            "reaction curves as an identical recovery operator."
        ),
        "vtu_field_note": (
            "Global visual cells store the global field only; local visual "
            "cells store the total G+L field. Filter is_total_field_cell=1 "
            "to inspect the local total-field patch."
        ),
        "c1_methodological_note": (
            "For C1, the standard trace basis crosses the crack-face/"
            "ligament boundary-condition transition because no control point "
            "lies exactly at the tip. Under direct coefficient constraints, "
            "that transition is not exactly representable without added "
            "modeling machinery."
            if st.local_iga_tip_continuity == "C1"
            else "Not applicable: the repeated C0 tip knot supplies an exact "
            "tip control-point column."
        ),
        "computed_metrics": [
            "lumped_projected_normalized_stress_yy",
            "direct_normalized_stress_yy",
        ],
        "skipped_metrics": ["l2_norm", "normalized_sif", "j_integral"],
    }
    with (case_dir / "full_siga_mesh_metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)


def read_case_result(path: Path) -> dict:
    with Path(path).open() as stream:
        return json.load(stream)


def write_full_siga_parent_summaries(
    parent_dir: Path,
    results: Sequence[dict],
) -> None:
    """Write wide and detailed normalized-stress summaries for all cases."""
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(results, key=lambda row: (int(row["nGx"]), int(row["nhL"])))
    settings = {
        (
            str(row.get("local_tip_continuity", "C0")),
            str(row.get("ligament_fixity", "")),
            int(row.get("coupling_order", -1)),
            str(row.get("recommended_cross_formulation_stress_recovery", "")),
        )
        for row in rows
    }
    if len(settings) > 1:
        raise ValueError(
            "Refusing to combine full s-IGA cases with different local tip "
            "continuity, ligament fixity, coupling quadrature, or "
            "stress-recovery semantics. Use a different --output-name for "
            "each campaign."
        )
    max_points = max(
        (len(row["normalized_stress_yy"]) for row in rows),
        default=0,
    )

    with (parent_dir / "normalized_stress_yy.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case_name", "hG", "hL", "rGL", "nGx", "nGy", "nhL", "dof"]
            + [f"point_{index + 1}" for index in range(max_points)]
        )
        for row in rows:
            values = list(row["normalized_stress_yy"])
            values += [float("nan")] * (max_points - len(values))
            writer.writerow(
                [
                    row["case_name"],
                    row["hG"],
                    row["hL"],
                    row["rGL"],
                    row["nGx"],
                    row["nGy"],
                    row["nhL"],
                    row["dof"],
                    *values,
                ]
            )

    with (parent_dir / "lumped_projected_normalized_stress_yy.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case_name", "hG", "hL", "rGL", "nGx", "nGy", "nhL", "dof"]
            + [f"point_{index + 1}" for index in range(max_points)]
        )
        for row in rows:
            values = list(
                row.get(
                    "lumped_projected_normalized_stress_yy",
                    row["normalized_stress_yy"],
                )
            )
            values += [float("nan")] * (max_points - len(values))
            writer.writerow(
                [
                    row["case_name"],
                    row["hG"],
                    row["hL"],
                    row["rGL"],
                    row["nGx"],
                    row["nGy"],
                    row["nhL"],
                    row["dof"],
                    *values,
                ]
            )

    with (parent_dir / "direct_normalized_stress_yy.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["case_name", "hG", "hL", "rGL", "nGx", "nGy", "nhL", "dof"]
            + [f"point_{index + 1}" for index in range(max_points)]
        )
        for row in rows:
            values = list(
                row.get(
                    "direct_normalized_stress_yy",
                    [float("nan")] * len(row["normalized_stress_yy"]),
                )
            )
            values += [float("nan")] * (max_points - len(values))
            writer.writerow(
                [
                    row["case_name"],
                    row["hG"],
                    row["hL"],
                    row["rGL"],
                    row["nGx"],
                    row["nGy"],
                    row["nhL"],
                    row["dof"],
                    *values,
                ]
            )

    with (parent_dir / "normalized_stress_yy_detailed.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case_name",
                "hG",
                "hL",
                "rGL",
                "nGx",
                "nGy",
                "nhL",
                "dof",
                "point_id",
                "x",
                "distance_from_tip",
                "lumped_projected_stress_yy",
                "exact_stress_yy",
                "lumped_projected_normalized_stress_yy",
                "direct_stress_yy",
                "direct_normalized_stress_yy",
                "consistent_projected_stress_yy",
                "consistent_projected_normalized_stress_yy",
            ]
        )
        for row in rows:
            direct_stress = row.get(
                "direct_stress_yy",
                [float("nan")] * len(row["normalized_stress_yy"]),
            )
            direct_normalized = row.get(
                "direct_normalized_stress_yy",
                [float("nan")] * len(row["normalized_stress_yy"]),
            )
            consistent_projected_stress = row.get(
                "consistent_projected_stress_yy",
                [float("nan")] * len(row["normalized_stress_yy"]),
            )
            consistent_projected_normalized = row.get(
                "consistent_projected_normalized_stress_yy",
                [float("nan")] * len(row["normalized_stress_yy"]),
            )
            for index, values in enumerate(
                zip(
                    row["sample_x"],
                    row["distance_from_tip"],
                    row["stress_yy"],
                    row["exact_stress_yy"],
                    row["normalized_stress_yy"],
                    direct_stress,
                    direct_normalized,
                    consistent_projected_stress,
                    consistent_projected_normalized,
                ),
                start=1,
            ):
                writer.writerow(
                    [
                        row["case_name"],
                        row["hG"],
                        row["hL"],
                        row["rGL"],
                        row["nGx"],
                        row["nGy"],
                        row["nhL"],
                        row["dof"],
                        index,
                        *values,
                    ]
                )

    campaign_summary = {
        "formulation": "full_s_iga_static_reviewer_comparison",
        "case_count": len(rows),
        "case_names": [str(row["case_name"]) for row in rows],
        "local_tip_continuity": (
            str(rows[0].get("local_tip_continuity", "C0"))
            if rows
            else ""
        ),
        "ligament_fixity": (
            str(rows[0].get("ligament_fixity", "")) if rows else ""
        ),
        "coupling_gauss_points_per_direction": (
            int(rows[0].get("coupling_order", -1)) if rows else -1
        ),
        "recommended_cross_formulation_stress_recovery": (
            "direct_total_field_D_times_BG_uG_plus_BL_uL"
        ),
        "native_reaction_stress_recovery": (
            "iga_boundary_reaction_lumped_projection"
        ),
        "reaction_projection_role": (
            str(
                rows[0].get(
                    "reaction_projection_role",
                    "native_C0_boundary_reaction_analogue",
                )
            )
            if rows
            else ""
        ),
        "comparison_warning": (
            "For a strict hS-IGA/full-s-IGA stress-field comparison, use "
            "direct_normalized_stress_yy for both formulations. The native "
            "Q4 and B-spline reaction projections use different smoothing "
            "operators; the legacy hS-IGA metric also uses nominal hL rather "
            "than the actual physical local edge length."
        ),
        "c1_boundary_transition_warning": (
            "The plain C1 local patch has no exact crack-tip control point. "
            "Its trace basis crosses the crack-face/ligament coefficient-BC "
            "transition, so reaction projection is diagnostic only."
            if rows
            and str(
                rows[0].get("local_tip_continuity", "C0")
            ).upper()
            == "C1"
            else ""
        ),
    }
    with (parent_dir / "full_siga_campaign_summary.json").open("w") as stream:
        json.dump(campaign_summary, stream, indent=2)
