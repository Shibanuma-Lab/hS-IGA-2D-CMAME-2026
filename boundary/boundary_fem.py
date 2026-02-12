"""
boundaryFEM – Dirichlet BCs for global-only analysis (islocal == 0).
"""

import numpy as np
import core.state as st
from utils.interpolator import BilinearQuadInterpolator
import os
from pathlib import Path


def boundaryFEM(step):
    """
    Construct ``st.ebc`` / ``st.nbc`` for the global-only (no local mesh) case.
    FEM reference solution is interpolated onto IGA boundary nodes.
    """
    right_val = st.hG * (st.nPtsX - 1)
    up_val    = st.hG * (st.nPtsY - 1)

    rightNodes = np.where(st.controlPts[:, 0] == right_val)[0]
    upNodes    = np.where(st.controlPts[:, 1] == up_val)[0]
    leftNodes  = np.where(st.controlPts[:, 0] == 0.0)[0]
    downNodes  = np.where(st.controlPts[:, 1] == 0.0)[0]

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
        IntpdisGx = BilinearQuadInterpolator(
            st.nodeFEM, st.elemFEM, st.disFEMsolutionAll[step, :, 0], name="FEM_x")
        IntpdisGy = BilinearQuadInterpolator(
            st.nodeFEM, st.elemFEM, st.disFEMsolutionAll[step, :, 1], name="FEM_y")
        
        # Construct ebc using interpolation
        ebc1 = [[nid, 1, IntpdisGx(st.nodeG[nid])] for nid in FEMbc]
        ebc2 = [[nid, 2, IntpdisGy(st.nodeG[nid])] for nid in FEMbc]

    fixbcX = sorted(list(set(leftNodes) - set(upNodes)))
    _tmp   = sorted(list(set(downNodes) - set(rightNodes)))
    fixbcY = _tmp[step:] if step < len(_tmp) else []

    # ebc1 and ebc2 are already constructed above (either from precomputed or interpolation)
    ebc3 = [[int(nid), 1, 0.0] for nid in fixbcX]
    ebc4 = [[int(nid), 2, 0.0] for nid in fixbcY]
    st.ebc = np.array(ebc1 + ebc2 + ebc3 + ebc4, dtype=float)

    st.nbc = np.array([[1.0, 1.0, 0.0]], dtype=float)


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
