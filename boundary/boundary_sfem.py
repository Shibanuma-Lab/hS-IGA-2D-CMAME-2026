"""
boundarysFEM – Dirichlet BCs for S-version FEM (islocal == 1).
"""

import numpy as np
import core.state as st
from utils.interpolator import BilinearQuadInterpolator


def boundarysFEM(step):
    """
    Build ``st.ebc`` / ``st.nbc`` for the coupled global + local case.

    Global boundary: FEM-interpolated forced displacement on top / right edges,
    zero-displacement on left (x) and bottom-ahead-of-crack (y).

    Local boundary: edges of local mesh are fully fixed.
    """
    right_val = st.hG * (st.nPtsX - 2)
    up_val    = st.hG * (st.nPtsY - 2)

    rightNodes = np.where(st.controlPts[:, 0] == right_val)[0]
    upNodes    = np.where(st.controlPts[:, 1] == up_val)[0]
    leftNodes  = np.where(st.controlPts[:, 0] == 0.0)[0]
    downNodes  = np.where(st.controlPts[:, 1] == 0.0)[0]

    up_part    = upNodes[:-1]
    right_part = np.sort(rightNodes)[::-1]
    FEMbc = np.concatenate([up_part, right_part])

    # FEM interpolators
    IntpdisGx = BilinearQuadInterpolator(
        st.nodeFEM, st.elemFEM, st.disFEMsolutionAll[step, :, 0], name="FEM_x")
    IntpdisGy = BilinearQuadInterpolator(
        st.nodeFEM, st.elemFEM, st.disFEMsolutionAll[step, :, 1], name="FEM_y")

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
    ebcs1 = [[nid, 1, IntpdisGx(st.nodeG[nid])] for nid in FEMbc]
    ebcs2 = [[nid, 2, IntpdisGy(st.nodeG[nid])] for nid in FEMbc]
    ebcs3 = [[nid, 1, 0.0] for nid in xfixG]
    ebcs4 = [[nid, 2, 0.0] for nid in yfixG]
    ebcs5 = [[nid - 1, 1, 0.0] for nid in xfixL]
    ebcs6 = [[nid - 1, 2, 0.0] for nid in yfixL]
    st.ebc = np.array(ebcs1 + ebcs2 + ebcs3 + ebcs4 + ebcs5 + ebcs6, dtype=float)

    st.nbc = np.array([[1.0, 1.0, 0.0]], dtype=float)
