"""
boundarysFEM – Dirichlet BCs for S-version FEM (islocal == 1).
"""

import numpy as np
import core.state as st
from utils.interpolator import BilinearQuadInterpolator, LinearDelaunayInterpolator
from pathlib import Path
from boundary.node_selector import find_nodes_on_extreme
from config.fem_data import get_fem_step_displacement


def boundarysFEM(step):
    """
    Build ``st.ebc`` / ``st.nbc`` for the coupled global + local case.

    Global boundary: FEM-interpolated forced displacement on top / right edges,
    zero-displacement on left (x) and bottom-ahead-of-crack (y).

    Local boundary: edges of local mesh are fully fixed.
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
        ebcs1 = [[nid, 1, precomputed_bc.get((step, nid + 1, 'x'), 0.0)] for nid in FEMbc]
        ebcs2 = [[nid, 2, precomputed_bc.get((step, nid + 1, 'y'), 0.0)] for nid in FEMbc]
    else:
        # FEM interpolators - select based on st.interpolator_type
        interpolator_type = getattr(st, 'interpolator_type', 'delaunay')  # Default to delaunay
        dis_step = get_fem_step_displacement(step)
        
        if interpolator_type == "delaunay":
            IntpdisGx = LinearDelaunayInterpolator(
                st.nodeFEM, dis_step[:, 0], name="FEM_x")
            IntpdisGy = LinearDelaunayInterpolator(
                st.nodeFEM, dis_step[:, 1], name="FEM_y")
            ebcs1 = [[nid, 1, IntpdisGx(st.nodeG[nid])] for nid in FEMbc]
            ebcs2 = [[nid, 2, IntpdisGy(st.nodeG[nid])] for nid in FEMbc]
        elif interpolator_type == "bilinear":
            bc_x, bc_y = _interpolate_bilinear_cached(FEMbc, dis_step)
            ebcs1 = [[int(nid), 1, float(val)] for nid, val in zip(FEMbc, bc_x)]
            ebcs2 = [[int(nid), 2, float(val)] for nid, val in zip(FEMbc, bc_y)]
        else:
            raise ValueError(f"Unknown interpolator_type: {interpolator_type}. Use 'delaunay' or 'bilinear'.")

    # ---- Global fixity ----
    fixbcX = sorted(list(set(leftNodes) - set(upNodes)))
    xfixG  = fixbcX
    nxfixG = len(xfixG)

    # Identify y-fixed global nodes ahead of crack tip
    eGFixY = []
    for e in range(st.nelemU):
        elem_nodes = st.elemVis[e]
        xs = [st.nodeVis[n][0] for n in elem_nodes]
        if max(xs) >= step * st.hL:
            eGFixY.append(e)

    # Convert st.element (1-based) to 0-based node indices
    nGFixY = sorted(set(j - 1 for e in eGFixY for j in st.element[e]))
    yfixG  = sorted(set(nGFixY).intersection(downNodes))
    yfixG  = sorted(set(yfixG) - set(rightNodes))
    nyfixG = len(yfixG)

    # ---- Local fixity ----
    nnmG = len(st.controlPts)
    nLr  = int(st.nLr)
    HL   = int(st.HL)

    xfixL  = []
    xfixL += [nnmG + (nLr + 1) * (i - 1) + 1 for i in range(1, HL + 1)]
    xfixL += [nnmG + (nLr + 1) * i           for i in range(1, HL + 1)]
    xfixL += [nnmG + (nLr + 1) * HL + i      for i in range(1, nLr + 2)]

    mostleftfixL = (step + 1) if (step <= st.aL) else (int(st.aL) + 1)

    cond = (len(yfixG) > 0 and yfixG[0] == 0) or (step <= st.aL)
    if cond:
        yfixL  = []
        yfixL += [nnmG + i for i in range(int(mostleftfixL), nLr + 2)]
        yfixL += [nnmG + (nLr + 1) * i for i in range(2, HL + 1)]
        yfixL += [nnmG + (nLr + 1) * HL + i for i in range(1, nLr + 2)]
    else:
        yfixL  = []
        yfixL += [nnmG + (nLr + 1) * (i - 1) + 1 for i in range(1, HL + 1)]
        yfixL += [nnmG + i for i in range(int(mostleftfixL), nLr + 2)]
        yfixL += [nnmG + (nLr + 1) * i for i in range(2, HL + 1)]
        yfixL += [nnmG + (nLr + 1) * HL + i for i in range(1, nLr + 2)]

    # ---- Assemble ebc ----
    # ebcs1 and ebcs2 are already constructed above (either from precomputed or interpolation)
    ebcs3 = [[nid, 1, 0.0] for nid in xfixG]
    ebcs4 = [[nid, 2, 0.0] for nid in yfixG]
    ebcs5 = [[nid - 1, 1, 0.0] for nid in xfixL]
    ebcs6 = [[nid - 1, 2, 0.0] for nid in yfixL]
    st.ebc = np.array(ebcs1 + ebcs2 + ebcs3 + ebcs4 + ebcs5 + ebcs6, dtype=float)

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
