"""
makeKL – assemble local Q4 stiffness and mass matrices.
"""

import numpy as np
import scipy.sparse as sp

import core.state as st
from utils.shape_functions import Dshp, enlarge, enlarge2


def makeKL():
    """Assemble local stiffness ``st.KL`` and mass ``st.ML``."""

    is_static_case = (
        getattr(st, "analysis_mode", "dynamic") == "static"
        or int(getattr(st, "isdynamic", 1)) == 0
    )
    use_sparse_static = is_static_case and int(getattr(st, "static_use_sparse", 1)) == 1
    assemble_mass = not (
        is_static_case and int(getattr(st, "static_skip_mass", 1)) == 1
    )

    if use_sparse_static:
        st.KL = sp.lil_matrix((st.neqL, st.neqL), dtype=float)
        st.ML = sp.lil_matrix((st.neqL, st.neqL), dtype=float) if assemble_mass else None
    else:
        st.KL = np.zeros((st.neqL, st.neqL), dtype=float)
        st.ML = np.zeros((st.neqL, st.neqL), dtype=float) if assemble_mass else None

    def calJL(e, xi_eta):
        return Dshp(xi_eta) @ st.enodeL[e - 1]

    ngp2 = st.ngpL ** 2

    # Jacobians
    JL = np.empty((st.nemL, ngp2), dtype=object)
    for e in range(1, st.nemL + 1):
        for gp in range(1, ngp2 + 1):
            JL[e - 1, gp - 1] = calJL(e, st.xi_etaL[gp - 1])

    # B-matrices (2×4 → 3×8)
    BLb = np.empty((st.nemL, ngp2), dtype=object)
    for e in range(1, st.nemL + 1):
        for gp in range(1, ngp2 + 1):
            BLb[e - 1, gp - 1] = (
                np.linalg.inv(JL[e - 1, gp - 1]) @ Dshp(st.xi_etaL[gp - 1])
            )

    BL = np.empty((st.nemL, ngp2), dtype=object)
    for e in range(st.nemL):
        for gp in range(ngp2):
            BL[e, gp] = enlarge(BLb[e, gp])

    # Element stiffness
    def calKL(BL_e, JL_e):
        total = np.zeros((BL_e[0].shape[1], BL_e[0].shape[1]), dtype=float)
        for w, B, J in zip(st.weightL, BL_e, JL_e):
            total += (B.T @ st.de @ B) * np.linalg.det(J) * w
        return total

    KLe = [calKL(BL[e, :], JL[e, :]) for e in range(st.nemL)]

    # DOF index per local element
    iL = []
    for elem in st.elemL:
        dofs = []
        for n in elem:
            dofs.extend([2 * n, 2 * n + 1])
        iL.append(dofs)

    for eL in range(st.nemL):
        idx = iL[eL]
        st.KL[np.ix_(idx, idx)] += KLe[eL]

    if not assemble_mass:
        if use_sparse_static:
            st.KL = st.KL.tocsr()
            st.KL.eliminate_zeros()
        return

    # Shape-function matrix for mass
    NL = np.empty(ngp2, dtype=object)
    for gp in range(ngp2):
        NL[gp] = enlarge2(st.nnL[gp, 0, :])

    def calML(JL_e):
        ndof_loc = NL[0].shape[1]
        total = np.zeros((ndof_loc, ndof_loc), dtype=float)
        for w, N, J in zip(st.weightL, NL, JL_e):
            total += w * st.rho * st.thi * (N.T @ N) * np.linalg.det(J)
        return total

    MLe = [calML(JL[e]) for e in range(st.nemL)]

    for eL in range(st.nemL):
        idx = iL[eL]
        st.ML[np.ix_(idx, idx)] += MLe[eL]

    if use_sparse_static:
        st.KL = st.KL.tocsr()
        st.KL.eliminate_zeros()
        st.ML = st.ML.tocsr()
        st.ML.eliminate_zeros()
