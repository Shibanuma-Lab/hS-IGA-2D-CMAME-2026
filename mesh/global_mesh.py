"""
makeGlobalMesh – generate the global NURBS/IGA mesh, knot vectors, control
points, and the visualisation quad mesh.

Mathematica: makeGlobalMesh[hG_, nnx_, nny_] := Module[{}, ...]
"""

import numpy as np

import core.state as st
from utils.nurbs import BasisFuns, FindSpanMinus
from utils.mapping import nurb2proj, delete_duplicates_preserve_order


# ------------------------------------------------------------------
def SurfacePoint(n, p, U, m, q, V, P, dim, u, v):
    """Evaluate a NURBS surface point in homogeneous coordinates."""
    uspan = FindSpanMinus(n, p, u, U)
    vspan = FindSpanMinus(m, q, v, V)

    Nu = BasisFuns(uspan, u, p, U)
    Nv = BasisFuns(vspan, v, q, V)

    uind = uspan - p
    S = np.zeros(dim, dtype=float)

    for j in range(0, q + 1):
        temp = np.zeros(dim, dtype=float)
        vind = vspan - q + j
        for i in range(0, p + 1):
            row = (uind + i) + (n + 1) * vind
            CP = P[row, :]
            temp = temp + Nu[i] * CP
        S = S + Nv[j] * temp
    return S


# ------------------------------------------------------------------
def _buildVisual2Dmesh(controlPts_, weights_, uKnot_, vKnot_, p_, q_):
    """Build a simple Q4 visualisation mesh from the IGA knot grid."""

    noPtsX = len(uKnot_) - p_ - 1
    noPtsY = len(vKnot_) - q_ - 1

    uKnotVec = delete_duplicates_preserve_order(uKnot_)
    vKnotVec = delete_duplicates_preserve_order(vKnot_)

    noKnotsU = len(uKnotVec)
    noKnotsV = len(vKnotVec)

    projcoord = nurb2proj(noPtsX * noPtsY, controlPts_, weights_)
    dim = len(projcoord[0])

    nodeVis = np.zeros((noKnotsU * noKnotsV, 2), dtype=float)
    count = 0
    for vk in range(1, noKnotsV + 1):
        etaVal = vKnotVec[vk - 1]
        for uk in range(1, noKnotsU + 1):
            xiVal = uKnotVec[uk - 1]
            tem = SurfacePoint(
                noPtsX - 1, p_, uKnot_,
                noPtsY - 1, q_, vKnot_,
                projcoord, dim, xiVal, etaVal,
            )
            nodeVis[count, 0] = tem[0] / tem[2]
            nodeVis[count, 1] = tem[1] / tem[2]
            count += 1

    nx = noKnotsU - 1
    ny = noKnotsV - 1

    ien0xy = []
    for _y in range(1, ny + 1):
        for _x in range(1, nx + 1):
            ien0xy.append(_x + (nx + 1) * (_y - 1))
    ien0xy = np.array(ien0xy, dtype=int)

    ienxy = np.vstack([
        ien0xy - 1,
        ien0xy,
        ien0xy + (nx + 1),
        ien0xy + (nx + 1) - 1,
    ])
    elemVis = ienxy.T.astype(int)
    nelemVis = elemVis.shape[0]

    return nodeVis, elemVis, nelemVis


# ------------------------------------------------------------------
def makeGlobalMesh(hG, nnx, nny):
    """Create the global IGA mesh and its Q4 visualisation mesh."""

    p = st.p
    q = st.q
    nPtsX = st.nPtsX
    nPtsY = st.nPtsY

    # Knot vectors
    knotUTemp = [i / (nPtsX - p) for i in range(0, (nPtsX - p) + 1)]
    knotVTemp = [j / (nPtsY - q) for j in range(0, (nPtsY - q) + 1)]

    st.uKnot = [0, 0] + list(knotUTemp) + [1, 1]
    st.vKnot = [0, 0] + list(knotVTemp) + [1, 1]

    st.uniqU = delete_duplicates_preserve_order(st.uKnot)
    st.nelemU = len(st.uniqU) - 1
    st.uniqV = delete_duplicates_preserve_order(st.vKnot)
    st.nelemV = len(st.uniqV) - 1

    st.weights = np.ones(nPtsX * nPtsY, dtype=float)

    Lx = st.nelemU * hG
    Ly = st.nelemV * hG

    st.noU = nPtsX
    st.noV = nPtsY

    nodex = []
    for i in range(1, st.noU + 1):
        if i == 1:
            nodex.append(0.0)
        elif i == st.noU:
            nodex.append(Lx)
        else:
            nodex.append(Lx * (2 * (i - 1) - 1) / (2.0 * st.nelemU))
    nodey = []
    for j in range(1, st.noV + 1):
        if j == 1:
            nodey.append(0.0)
        elif j == st.noV:
            nodey.append(Ly)
        else:
            nodey.append(Ly * (2 * (j - 1) - 1) / (2.0 * st.nelemV))

    nodex = np.array(nodex, dtype=float)
    nodey = np.array(nodey, dtype=float)
    st.controlPts = np.array([[x, y] for y in nodey for x in nodex], dtype=float)

    st.elRangeU = [[st.uniqU[i], st.uniqU[i + 1]] for i in range(st.nelemU)]
    st.elRangeV = [[st.uniqV[j], st.uniqV[j + 1]] for j in range(st.nelemV)]

    st.elConnU = []
    for i in range(st.nelemU):
        _ximid = 0.5 * (st.elRangeU[i][0] + st.elRangeU[i][1])
        _span = FindSpanMinus(st.noU - 1, p, _ximid, st.uKnot)
        st.elConnU.append(list(range((_span - p) + 1, _span + 1 + 1)))

    st.elConnV = []
    for j in range(st.nelemV):
        _etamid = 0.5 * (st.elRangeV[j][0] + st.elRangeV[j][1])
        _span = FindSpanMinus(st.noV - 1, q, _etamid, st.vKnot)
        st.elConnV.append(list(range((_span - q) + 1, _span + 1 + 1)))

    st.chan = np.arange(1, st.noU * st.noV + 1, dtype=int).reshape(st.noV, st.noU)

    st.element = []
    for vv in range(1, st.nelemV + 1):
        for uu in range(1, st.nelemU + 1):
            block = []
            for vIdx in st.elConnV[vv - 1]:
                for uIdx in st.elConnU[uu - 1]:
                    block.append(int(st.chan[vIdx - 1, uIdx - 1]))
            st.element.append(block)
    st.nelem = st.nelemU * st.nelemV

    st.index = [[u, v] for v in range(1, st.nelemV + 1) for u in range(1, st.nelemU + 1)]

    # Build visualisation mesh
    st.nodeVis, st.elemVis, st.nelemVis = _buildVisual2Dmesh(
        st.controlPts, st.weights, st.uKnot, st.vKnot, p, q,
    )

    return {
        "uKnot": st.uKnot, "vKnot": st.vKnot, "weights": st.weights,
        "controlPts": st.controlPts, "element": st.element, "index": st.index,
        "nodeVis": st.nodeVis, "elemVis": st.elemVis, "nelemVis": st.nelemVis,
        "noU": st.noU, "noV": st.noV,
    }
