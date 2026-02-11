"""
buildVisual2D – compute stresses and displacements on the IGA visualisation mesh.
"""

import numpy as np
import core.state as st
from mesh.global_mesh import SurfacePoint
from utils.nurbs import NURBS2DBasisDers, FindSpanMinus
from utils.mapping import nurb2proj


def buildVisual2D():
    """
    Build visualisation-mesh node coordinates, element connectivity,
    and evaluate stress / displacement at knot-span corners.
    """
    p = int(st.p);  q = int(st.q)
    uKnot = st.uKnot;  vKnot = st.vKnot
    controlPts = st.controlPts;  weights = st.weights

    noPtsX = len(uKnot) - p - 1
    noPtsY = len(vKnot) - q - 1

    def _ddpo(seq):
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    uKnotVec = _ddpo(list(uKnot))
    vKnotVec = _ddpo(list(vKnot))
    noKnotsU = len(uKnotVec)
    noKnotsV = len(vKnotVec)

    projcoord = nurb2proj(noPtsX * noPtsY, controlPts, weights)
    dim = len(projcoord[0])

    # ---- nodeVis ----
    st.nodeVis = np.zeros((noKnotsU * noKnotsV, 2), dtype=float)
    count = 0
    for vk in range(noKnotsV):
        etaVal = vKnotVec[vk]
        for uk in range(noKnotsU):
            xiVal = uKnotVec[uk]
            tem = SurfacePoint(
                noPtsX - 1, p, uKnot, noPtsY - 1, q, vKnot,
                projcoord, dim, xiVal, etaVal,
            )
            st.nodeVis[count, 0] = tem[0] / tem[2]
            st.nodeVis[count, 1] = tem[1] / tem[2]
            count += 1

    # ---- elemVis ----
    nx = noKnotsU - 1;  ny = noKnotsV - 1
    elemVis_list = []
    for j in range(ny):
        for i in range(nx):
            ll = i + (nx + 1) * j
            elemVis_list.append([ll, ll + 1, ll + (nx + 1) + 1, ll + (nx + 1)])
    st.elemVis = np.array(elemVis_list, dtype=int)
    nelemVis = st.elemVis.shape[0]
    noelemV  = nelemVis

    stress = np.zeros((noelemV, 4, 3), dtype=float)
    disp   = np.zeros((noelemV, 4, 2), dtype=float)

    ndof = int(st.ndof)
    lenu = int(st.lenu);  lenv = int(st.lenv)

    for e in range(noelemV):
        idu, idv = st.index[e]
        xiE  = st.elRangeU[idu - 1]
        etaE = st.elRangeV[idv - 1]

        sctr = st.element[e]
        nn   = len(sctr)
        B    = np.zeros((3, ndof * nn), dtype=float)
        pts  = controlPts[np.array(sctr, dtype=int) - 1, :]

        uspan = FindSpanMinus(noPtsX - 1, p, xiE[0], uKnot)
        vspan = FindSpanMinus(noPtsY - 1, q, etaE[0], vKnot)

        elemDisp = np.column_stack([
            st.disG2D[np.array(sctr) - 1, 0],
            st.disG2D[np.array(sctr) - 1, 1],
        ])

        gp = 0
        for iv in range(2):
            Eta = etaE[iv]
            for iu in range(2):
                Xi = xiE[iu]
                NN, dNdxi, dNdeta = NURBS2DBasisDers(
                    uspan, vspan, p, q, uKnot, vKnot, Xi, Eta,
                    weights, lenu, lenv,
                )
                dNbf = np.vstack([dNdxi, dNdeta])
                Jxu  = dNbf @ pts
                dN   = np.linalg.inv(Jxu) @ dNbf

                B.fill(0.0)
                for j in range(1, nn + 1):
                    dNdx = dN[0, j - 1]; dNdy = dN[1, j - 1]
                    j1 = 2 * j - 1; j2 = 2 * j
                    B[0, j1 - 1] = dNdx
                    B[1, j2 - 1] = dNdy
                    B[2, j1 - 1] = dNdy
                    B[2, j2 - 1] = dNdx

                ue = st.disG2D[np.array(st.element[e]) - 1, :].reshape(-1)
                strain = B @ ue
                stress[e, gp, :] = st.de @ strain
                disp[e, gp, :]   = NN @ elemDisp
                gp += 1

        # swap corners 3 and 4
        disp[e, 2, :], disp[e, 3, :] = disp[e, 3, :].copy(), disp[e, 2, :].copy()

    # ---- Scatter to nodes ----
    nnodeVis = st.nodeVis.shape[0]
    st.sigmaXX = np.zeros(nnodeVis, dtype=float)
    st.sigmaYY = np.zeros(nnodeVis, dtype=float)
    st.sigmaYY_with_ids = np.zeros((nnodeVis, 4), dtype=float)
    st.sigmaXY = np.zeros(nnodeVis, dtype=float)
    st.dispX   = np.zeros(nnodeVis, dtype=float)
    st.dispY   = np.zeros(nnodeVis, dtype=float)

    for e in range(len(st.elemVis)):
        connect = st.elemVis[e]
        for loc in range(4):
            nid = connect[loc]
            st.sigmaXX[nid] = stress[e, loc, 0]
            st.sigmaYY[nid] = stress[e, loc, 1]
            if e >= (nx - 1) / 2 and e <= nx and loc == 1:
                st.sigmaYY_with_ids[nid, 0] = stress[e, loc, 1]
                st.sigmaYY_with_ids[nid, 1] = int(e)
                st.sigmaYY_with_ids[nid, 2] = int(loc)
                st.sigmaYY_with_ids[nid, 3] = int(nid)
            st.sigmaXY[nid] = stress[e, loc, 2]
            st.dispX[nid]   = disp[e, loc, 0]
            st.dispY[nid]   = disp[e, loc, 1]
