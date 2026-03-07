"""
boundaryFEM – Dirichlet BCs for global-only analysis (islocal == 0).
"""

import numpy as np
import core.state as st
from utils.interpolator import BilinearQuadInterpolator
from pathlib import Path
from boundary.node_selector import find_nodes_on_extreme
from config.fem_data import get_fem_step_displacement


def boundaryFEM(step):
    """
    Construct ``st.ebc`` / ``st.nbc`` for the global-only (no local mesh) case.
    FEM reference solution is interpolated onto IGA boundary nodes.
    """
    coord_tol = getattr(st, "boundary_coord_tol", None)
    rightNodes = find_nodes_on_extreme(st.controlPts, axis=0, side="max", atol=coord_tol)
    upNodes    = find_nodes_on_extreme(st.controlPts, axis=1, side="max", atol=coord_tol)
    leftNodes  = find_nodes_on_extreme(st.controlPts, axis=0, side="min", atol=coord_tol)
    downNodes  = find_nodes_on_extreme(st.controlPts, axis=1, side="min", atol=coord_tol)

    up_part    = upNodes[:-1]
    right_part = np.sort(rightNodes)[::-1]
    FEMbc = np.concatenate([up_part, right_part])

    # Check if we should use precomputed BC from Mathematica (debug mode)
    use_precomputed = getattr(st, 'use_precomputed_bc', False)
    
    if use_precomputed:
        # Load precomputed BC values from CSV
        # CSV format: step, node_id, disp_x, disp_y
        # NOTE: CSV uses Mathematica's 1-based node indices, Python uses 0-based
        precomputed_bc = _load_precomputed_bc(step)
        
        # Directly construct ebc from precomputed values
        # Convert Python's 0-based nid to Mathematica's 1-based for CSV lookup
        ebc1 = [[nid, 1, precomputed_bc.get((step, nid + 1, 'x'), 0.0)] for nid in FEMbc]
        ebc2 = [[nid, 2, precomputed_bc.get((step, nid + 1, 'y'), 0.0)] for nid in FEMbc]
    else:
        dis_step = get_fem_step_displacement(step)
        bc_x, bc_y = _interpolate_bilinear_cached(FEMbc, dis_step)
        ebc1 = [[int(nid), 1, float(val)] for nid, val in zip(FEMbc, bc_x)]
        ebc2 = [[int(nid), 2, float(val)] for nid, val in zip(FEMbc, bc_y)]

    fixbcX = sorted(list(set(leftNodes) - set(upNodes)))
    _tmp   = sorted(list(set(downNodes) - set(rightNodes)))
    fixbcY = _tmp[step:] if step < len(_tmp) else []

    # ebc1 and ebc2 are already constructed above (either from precomputed or interpolation)
    ebc3 = [[int(nid), 1, 0.0] for nid in fixbcX]
    ebc4 = [[int(nid), 2, 0.0] for nid in fixbcY]
    st.ebc = np.array(ebc1 + ebc2 + ebc3 + ebc4, dtype=float)

    st.nbc = np.array([[1.0, 1.0, 0.0]], dtype=float)


def _interpolate_bilinear_cached(FEMbc, dis_step):
    cache = getattr(st, "_fem_bilinear_bc_cache", None)
    fembc_ids = np.asarray(FEMbc, dtype=int).ravel()
    mesh_sig = (
        id(st.nodeFEM),
        id(st.elemFEM),
        int(len(st.nodeFEM)),
        int(len(st.elemFEM)),
    )

    needs_rebuild = (
        cache is None
        or cache.get("mesh_sig") != mesh_sig
        or cache.get("n_query") != int(len(fembc_ids))
        or not np.array_equal(cache.get("fembc_ids"), fembc_ids)
    )

    if needs_rebuild:
        interp = BilinearQuadInterpolator(
            st.nodeFEM,
            st.elemFEM,
            np.zeros(len(st.nodeFEM), dtype=float),
            name="FEM_bc",
        )
        query_points = np.asarray([st.nodeG[int(nid)] for nid in fembc_ids], dtype=float)
        point_map = interp.precompute_point_map(query_points)
        cache = {
            "mesh_sig": mesh_sig,
            "n_query": int(len(fembc_ids)),
            "fembc_ids": fembc_ids.copy(),
            "interp": interp,
            "point_map": point_map,
        }
        st._fem_bilinear_bc_cache = cache

    vals_x = np.asarray(dis_step[:, 0], dtype=float)
    vals_y = np.asarray(dis_step[:, 1], dtype=float)
    out_x = cache["interp"].evaluate_from_point_map(cache["point_map"], vals_x)
    out_y = cache["interp"].evaluate_from_point_map(cache["point_map"], vals_y)
    return out_x, out_y


def _load_precomputed_bc(step):
    """
    Load precomputed boundary conditions from Mathematica CSV file.
    
    CSV format: step, node_id, disp_x, disp_y
    Returns: dict with keys (step, node_id, direction) -> value
    """
    # Use absolute path relative to this file's location
    csv_path = Path(__file__).resolve().parent.parent / "FEM_data" / "interpolated_bc_all.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Precomputed BC file not found: {csv_path}\n"
            f"Set st.use_precomputed_bc = False or generate the CSV from Mathematica.")
    
    bc_dict = {}
    with open(csv_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:
                s, nid, dx, dy = parts
                s = int(float(s))
                nid = int(float(nid))
                dx = float(dx)
                dy = float(dy)
                bc_dict[(s, nid, 'x')] = dx
                bc_dict[(s, nid, 'y')] = dy
    
    print(f"Loaded precomputed BC for step {step}: {sum(1 for k in bc_dict if k[0] == step)} entries")
    return bc_dict
