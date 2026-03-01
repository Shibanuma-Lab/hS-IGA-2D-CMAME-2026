"""
makeKG – assemble global IGA stiffness and mass matrices.
"""

import numpy as np
import scipy.sparse as sp

import core.state as st
from utils.shape_functions import GP, GW
from utils.mapping import parent2ParametricSpace, jacobianPaPaMapping2d
from utils.nurbs import NURBS2DBasisDers, FindSpanMinus as FindSpan


def makeKG():
    """Assemble global stiffness ``st.KG`` and mass ``st.MG``."""

    st.nGPs = 3
    st.ndof = 2
    st.nCtrPts = st.nPtsX * st.nPtsY
    st.neqG = st.nCtrPts * st.ndof

    is_static_case = (
        getattr(st, "analysis_mode", "dynamic") == "static"
        or int(getattr(st, "isdynamic", 1)) == 0
    )
    use_sparse_static = is_static_case and int(getattr(st, "static_use_sparse", 1)) == 1
    assemble_mass = not (
        is_static_case and int(getattr(st, "static_skip_mass", 1)) == 1
    )

    if use_sparse_static:
        stiff = sp.lil_matrix((st.neqG, st.neqG), dtype=np.float64)
        mass = sp.lil_matrix((st.neqG, st.neqG), dtype=np.float64) if assemble_mass else None
    else:
        stiff = np.zeros((st.neqG, st.neqG), dtype=np.float64)
        mass = np.zeros((st.neqG, st.neqG), dtype=np.float64) if assemble_mass else None

    Q = [(gp2, gp1) for gp1 in GP(st.nGPs) for gp2 in GP(st.nGPs)]
    W = [gw2 * gw1 for gw1 in GW(st.nGPs) for gw2 in GW(st.nGPs)]

    st.lenu = len(st.uKnot) - 1 - st.p - 1
    st.lenv = len(st.vKnot) - 1 - st.q - 1

    for e in range(1, st.nelem + 1):
        idu, idv = st.index[e - 1]
        xiE = st.elRangeU[idu - 1]
        etaE = st.elRangeV[idv - 1]

        sctr = st.element[e - 1]
        ncpelem = len(sctr)

        localStiff = np.zeros((st.ndof * ncpelem, st.ndof * ncpelem), dtype=np.float64)
        localMass = (
            np.zeros((st.ndof * ncpelem, st.ndof * ncpelem), dtype=np.float64)
            if assemble_mass
            else None
        )
        scrtx = np.zeros((st.ndof * ncpelem,), dtype=int)

        B = np.zeros((3, st.ndof * ncpelem), dtype=np.float64)
        func = np.zeros((2, st.ndof * ncpelem), dtype=np.float64) if assemble_mass else None

        for gp, (pt, wt) in enumerate(zip(Q, W)):
            Xi = parent2ParametricSpace(xiE, pt[0])
            Eta = parent2ParametricSpace(etaE, pt[1])
            detJuxi = jacobianPaPaMapping2d(xiE, etaE)

            ni = FindSpan(st.lenu, st.p, Xi, st.uKnot)
            nj = FindSpan(st.lenv, st.q, Eta, st.vKnot)

            NN, dNdxi, dNdeta = NURBS2DBasisDers(
                ni, nj, st.p, st.q, st.uKnot, st.vKnot, Xi, Eta,
                st.weights, st.lenu, st.lenv,
            )
            dNbf = np.vstack((dNdxi, dNdeta))

            pts = st.controlPts[np.array(sctr, dtype=int) - 1, :].astype(np.float64)
            Jxu = dNbf @ pts
            invJxu = np.linalg.inv(Jxu)
            dN = invJxu @ dNbf

            B.fill(0.0)
            if assemble_mass:
                func.fill(0.0)

            for nid in range(1, ncpelem + 1):
                globnum = sctr[nid - 1]
                dNdx = dN[0, nid - 1]
                dNdy = dN[1, nid - 1]

                scrtx[2 * nid - 2] = 2 * globnum - 1
                scrtx[2 * nid - 1] = 2 * globnum

                j1 = 2 * nid - 2
                j2 = 2 * nid - 1

                B[0, j1] = dNdx
                B[1, j2] = dNdy
                B[2, j1] = dNdy
                B[2, j2] = dNdx

                if assemble_mass:
                    func[0, j1] = NN[nid - 1]
                    func[1, j2] = NN[nid - 1]

            detJ = np.linalg.det(Jxu)
            localStiff += (B.T @ st.de @ B) * detJ * detJuxi * wt
            if assemble_mass:
                localMass += st.thi * (func.T @ st.dRho @ func) * detJ * detJuxi * wt

        I = np.array(scrtx, dtype=int) - 1
        if use_sparse_static:
            stiff[np.ix_(I, I)] += localStiff
            if assemble_mass:
                mass[np.ix_(I, I)] += localMass
        else:
            stiff[np.ix_(I, I)] += localStiff
            if assemble_mass:
                mass[np.ix_(I, I)] += localMass

    if use_sparse_static:
        st.KG = stiff.tocsr()
        st.KG.eliminate_zeros()
        if assemble_mass:
            st.MG = mass.tocsr()
            st.MG.eliminate_zeros()
        else:
            st.MG = None
    else:
        st.KG = stiff
        st.MG = mass
