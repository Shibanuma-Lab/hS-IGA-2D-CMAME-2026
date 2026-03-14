"""
makeKG – assemble global IGA stiffness and mass matrices.
"""

import numpy as np
import scipy.sparse as sp

import core.state as st
from utils.shape_functions import GP, GW
from utils.mapping import parent2ParametricSpace, jacobianPaPaMapping2d
from utils.nurbs import BasisFuns, DerBasisFuns


def _rational_basis_from_tensor(nx, ny, dnx, dny, weight_local):
    """
    Build 2D NURBS basis (R, dR/dxi, dR/deta) from tensor-product 1D bases.
    """
    basis = np.outer(ny, nx).ravel()
    deriv_xi = np.outer(ny, dnx).ravel()
    deriv_eta = np.outer(dny, nx).ravel()

    num = basis * weight_local
    w_tot = float(np.sum(num))
    dwdxi = float(np.dot(deriv_xi, weight_local))
    dwdeta = float(np.dot(deriv_eta, weight_local))

    inv_w = 1.0 / w_tot
    inv_w2 = inv_w * inv_w

    r = num * inv_w
    dr_dxi = (deriv_xi * weight_local * w_tot - num * dwdxi) * inv_w2
    dr_deta = (deriv_eta * weight_local * w_tot - num * dwdeta) * inv_w2
    return r, dr_dxi, dr_deta


def makeKG():
    """Assemble global stiffness ``st.KG`` and mass ``st.MG``."""

    st.nGPs = int(getattr(st, "ngpG", 3))
    if st.nGPs <= 0:
        raise ValueError(f"Invalid global Gauss order ngpG={st.nGPs}")
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

    gp_vals = np.asarray(GP(st.nGPs), dtype=np.float64)
    gw_vals = np.asarray(GW(st.nGPs), dtype=np.float64)

    st.lenu = len(st.uKnot) - 1 - st.p - 1
    st.lenv = len(st.vKnot) - 1 - st.q - 1
    u_knot = np.asarray(st.uKnot, dtype=np.float64)
    v_knot = np.asarray(st.vKnot, dtype=np.float64)
    cp_weights = np.asarray(st.weights, dtype=np.float64)
    control_pts = np.asarray(st.controlPts, dtype=np.float64)
    de = st.de
    d_rho = st.dRho
    thi = st.thi

    # Cache 1D basis/derivative values per knot-span index and Gauss point.
    n_u_elem = len(st.elRangeU)
    n_v_elem = len(st.elRangeV)
    u_basis = [[None for _ in range(st.nGPs)] for _ in range(n_u_elem)]
    v_basis = [[None for _ in range(st.nGPs)] for _ in range(n_v_elem)]

    for idu in range(1, n_u_elem + 1):
        xiE = st.elRangeU[idu - 1]
        span_u = idu + st.p - 1
        for gu, gp_u in enumerate(gp_vals):
            xi = parent2ParametricSpace(xiE, gp_u)
            nx = np.asarray(BasisFuns(span_u, xi, st.p, u_knot), dtype=np.float64)
            dnx = np.asarray(DerBasisFuns(span_u, xi, st.p, 1, u_knot)[1, :], dtype=np.float64)
            u_basis[idu - 1][gu] = (nx, dnx)

    for idv in range(1, n_v_elem + 1):
        etaE = st.elRangeV[idv - 1]
        span_v = idv + st.q - 1
        for gv, gp_v in enumerate(gp_vals):
            eta = parent2ParametricSpace(etaE, gp_v)
            ny = np.asarray(BasisFuns(span_v, eta, st.q, v_knot), dtype=np.float64)
            dny = np.asarray(DerBasisFuns(span_v, eta, st.q, 1, v_knot)[1, :], dtype=np.float64)
            v_basis[idv - 1][gv] = (ny, dny)

    for e in range(1, st.nelem + 1):
        idu, idv = st.index[e - 1]
        xiE = st.elRangeU[idu - 1]
        etaE = st.elRangeV[idv - 1]
        detJuxi = jacobianPaPaMapping2d(xiE, etaE)

        sctr = np.asarray(st.element[e - 1], dtype=int)
        ncpelem = sctr.size

        localStiff = np.zeros((st.ndof * ncpelem, st.ndof * ncpelem), dtype=np.float64)
        localMass = (
            np.zeros((st.ndof * ncpelem, st.ndof * ncpelem), dtype=np.float64)
            if assemble_mass
            else None
        )
        scrtx = np.empty((st.ndof * ncpelem,), dtype=int)
        scrtx[0::2] = 2 * sctr - 1
        scrtx[1::2] = 2 * sctr

        B = np.zeros((3, st.ndof * ncpelem), dtype=np.float64)
        dNbf = np.zeros((2, ncpelem), dtype=np.float64)
        func = np.zeros((2, st.ndof * ncpelem), dtype=np.float64) if assemble_mass else None
        cols_u = np.arange(0, st.ndof * ncpelem, 2, dtype=int)
        cols_v = cols_u + 1

        pts = control_pts[sctr - 1, :]
        weight_local = cp_weights[sctr - 1]

        for gv in range(st.nGPs):
            ny, dny = v_basis[idv - 1][gv]
            for gu in range(st.nGPs):
                nx, dnx = u_basis[idu - 1][gu]
                wt = gw_vals[gv] * gw_vals[gu]

                NN, dNdxi, dNdeta = _rational_basis_from_tensor(
                    nx, ny, dnx, dny, weight_local
                )
                dNbf[0, :] = dNdxi
                dNbf[1, :] = dNdeta

                Jxu = dNbf @ pts
                a = Jxu[0, 0]
                b = Jxu[0, 1]
                c = Jxu[1, 0]
                d = Jxu[1, 1]
                detJ = a * d - b * c
                inv_det = 1.0 / detJ
                invJxu = np.array(
                    [[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]],
                    dtype=np.float64,
                )
                dN = invJxu @ dNbf

                B.fill(0.0)
                if assemble_mass:
                    func.fill(0.0)
                B[0, cols_u] = dN[0, :]
                B[1, cols_v] = dN[1, :]
                B[2, cols_u] = dN[1, :]
                B[2, cols_v] = dN[0, :]

                if assemble_mass:
                    func[0, cols_u] = NN
                    func[1, cols_v] = NN

                localStiff += (B.T @ de @ B) * detJ * detJuxi * wt
                if assemble_mass:
                    localMass += thi * (func.T @ d_rho @ func) * detJ * detJuxi * wt

        I = scrtx - 1
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
