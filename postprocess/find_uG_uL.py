"""
finduGuL – compute disLG2D = disL + disLofGIGA (IGA contribution at local nodes).
"""

import os
import numpy as np
import core.state as st
from postprocess.build_visual import buildVisual2D
from utils.nurbs import NURBS2DBasisDers, FindSpanMinus
from utils.mapping import parent2ParametricSpace
from utils.shape_functions import enlarge
from matrix.iga_xi_eta import IGAgemoGetXiEta


def finduGuL():
    """
    1. Call ``buildVisual2D`` to get visualisation-mesh stresses / displacements.
    2. For each local node, interpolate the IGA displacement via NURBS basis.
    3. Sum: ``disLG2D = disL + disLofGIGA``.
    4. Compute local stresses at element corner points.
    """
    # ---- 1) IGA visualisation ----
    buildVisual2D()
    st.stressVis = np.vstack([st.sigmaXX, st.sigmaYY, st.sigmaXY]).T

    os.makedirs("result_sigma", exist_ok=True)
    np.savetxt(os.path.join("result_sigma", "sigmaYY_with_ids.csv"),
               st.sigmaYY_with_ids, delimiter=",")

    st.disVis  = np.column_stack((st.dispX, st.dispY))
    st.edisVis = [st.disVis[np.array(conn, dtype=int), :] for conn in st.elemVis]

    # ---- 2) Map local nodes → IGA parent coordinates ----
    p = int(st.p);  q = int(st.q)
    uKnot = st.uKnot;  vKnot = st.vKnot
    noPtsX = len(uKnot) - p - 1
    noPtsY = len(vKnot) - q - 1
    lenu = int(st.lenu);  lenv = int(st.lenv)

    nnmL = len(st.nodeL)
    nodeL2XiEtaInIGA = [
        IGAgemoGetXiEta(
            st.emGe[st.XiEtaGeG[i][1] - 1] - 1,
            st.nodeL[i],
            st.XiEtaGeG[i][0],
        )
        for i in range(nnmL)
    ]

    edisIGA = [st.disG2D[np.array(sctr, dtype=int) - 1, :] for sctr in st.element]

    # ---- 3) Interpolate IGA displacement at local nodes ----
    st.disLofGIGA = np.zeros((nnmL, 2), dtype=float)
    for i in range(nnmL):
        e_idx = st.emGe[st.XiEtaGeG[i][1] - 1] - 1
        idu_1b, idv_1b = st.index[e_idx]
        xiE  = st.elRangeU[int(idu_1b) - 1]
        etaE = st.elRangeV[int(idv_1b) - 1]

        xi_parent, eta_parent = nodeL2XiEtaInIGA[i]
        Xi  = parent2ParametricSpace(tuple(xiE), xi_parent)
        Eta = parent2ParametricSpace(tuple(etaE), eta_parent)

        ni = FindSpanMinus(noPtsX - 1, p, float(Xi), uKnot)
        nj = FindSpanMinus(noPtsY - 1, q, float(Eta), vKnot)

        NN, _, _ = NURBS2DBasisDers(
            ni, nj, p, q, uKnot, vKnot, float(Xi), float(Eta),
            st.weights, lenu, lenv,
        )
        st.disLofGIGA[i, :] = NN @ edisIGA[e_idx]

    # ---- 4) Total local displacement ----
    st.disLG2D = st.disL + st.disLofGIGA

    # ---- 5) Local stress computation at element corners ----
    disLG2Delem = np.array([st.disLG2D[elem] for elem in st.elemL])
    xyL = np.array([st.nodeL[elem] for elem in st.elemL])

    def DNi(xi, eta):
        return 0.25 * np.array([
            [-1 + eta,  1 - eta,  1 + eta, -1 - eta],
            [-1 + xi,  -1 - xi,   1 + xi,   1 - xi],
        ])

    n4XiEta = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    nemL = int(st.nemL)

    JL = np.array([
        [DNi(xi, eta) @ xyL[e] for (xi, eta) in n4XiEta]
        for e in range(nemL)
    ])

    BLb = np.zeros((nemL, 4, 2, 4))
    for e in range(nemL):
        for igp in range(4):
            xi, eta = n4XiEta[igp]
            BLb[e, igp] = np.linalg.inv(JL[e, igp]) @ DNi(xi, eta)

    BL = np.array([
        [enlarge(BLb[e, igp]) for igp in range(4)]
        for e in range(nemL)
    ])

    sigmaTauL = np.zeros((nemL, 4, 3))
    for e in range(nemL):
        ue = disLG2Delem[e].reshape(-1)
        for igp in range(4):
            sigmaTauL[e, igp] = st.de @ (BL[e, igp] @ ue)

    nL4sigma_xx = sigmaTauL[:, :, 0]
    nL4sigma_yy = sigmaTauL[:, :, 1]
    nL4tau_xy   = sigmaTauL[:, :, 2]

    nLeL = [[] for _ in range(nnmL)]
    for e, conn in enumerate(st.elemL):
        for loc_id, node in enumerate(conn):
            nLeL[node].append((e, loc_id))

    def getave(sigma_table, elem_loc_list):
        vals = [sigma_table[e][loc] for (e, loc) in elem_loc_list]
        return np.mean(vals) if vals else 0.0

    sigma_xxL = np.array([getave(nL4sigma_xx, lst) for lst in nLeL])
    sigma_yyL = np.array([getave(nL4sigma_yy, lst) for lst in nLeL])
    tau_xyL   = np.array([getave(nL4tau_xy,   lst) for lst in nLeL])

    st.stressL = np.vstack([sigma_xxL, sigma_yyL, tau_xyL]).T
