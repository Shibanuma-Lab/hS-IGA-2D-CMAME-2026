"""
boundaryFEM – Dirichlet BCs for global-only analysis (islocal == 0).
"""

import numpy as np
import core.state as st
from utils.interpolator import BilinearQuadInterpolator


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

    IntpdisGx = BilinearQuadInterpolator(
        st.nodeFEM, st.elemFEM, st.disFEMsolutionAll[step, :, 0], name="FEM_x")
    IntpdisGy = BilinearQuadInterpolator(
        st.nodeFEM, st.elemFEM, st.disFEMsolutionAll[step, :, 1], name="FEM_y")

    fixbcX = sorted(list(set(leftNodes) - set(upNodes)))
    _tmp   = sorted(list(set(downNodes) - set(rightNodes)))
    fixbcY = _tmp[step:] if step < len(_tmp) else []

    ebc1 = [[nid, 1, IntpdisGx(st.nodeG[nid])] for nid in FEMbc]
    ebc2 = [[nid, 2, IntpdisGy(st.nodeG[nid])] for nid in FEMbc]
    ebc3 = [[int(nid), 1, 0.0] for nid in fixbcX]
    ebc4 = [[int(nid), 2, 0.0] for nid in fixbcY]
    st.ebc = np.array(ebc1 + ebc2 + ebc3 + ebc4, dtype=float)

    st.nbc = np.array([[1.0, 1.0, 0.0]], dtype=float)
