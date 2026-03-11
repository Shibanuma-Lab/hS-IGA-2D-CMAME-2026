"""
makeKGL – assemble the global–local coupling matrices K^GL and M^GL.

Contains:  makeKGL(), makeKGL1(), makeKGL3(), makeKGL6()
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import fsolve

import core.state as st
from utils.shape_functions import (
    N1, N2, N3, N4, shp, Dshp, enlarge, enlarge2, GP, GW,
)
from utils.mapping import parent2ParametricSpace, jacobianPaPaMapping2d
from utils.nurbs import NURBS2DBasisDers, FindSpanMinus as FindSpan
from matrix.iga_xi_eta import IGAgemoGetXiEta


# ======================================================================
# makeKGL dispatcher
# ======================================================================
def makeKGL():
    """Dispatch coupling matrix assembly based on ``st.ismortar``."""
    if st.ismortar == 1:
        makeKGL1()
        makeKGL3()
    else:
        makeKGL1()
        makeKGL3()
        makeKGL6()

    try:
        current_step = st.step
        if current_step is not None and current_step >= 0 and (current_step % 10 == 0 or current_step > 95):
            if sp.issparse(st.KGL):
                kgl_nnz = int(st.KGL.nnz)
                kgl_max = float(np.max(np.abs(st.KGL.data))) if st.KGL.nnz > 0 else 0.0
            else:
                kgl_nnz = int(np.count_nonzero(st.KGL))
                kgl_max = float(np.max(np.abs(st.KGL)))

            if st.MGL is None:
                mgl_nnz = 0
            elif sp.issparse(st.MGL):
                mgl_nnz = int(st.MGL.nnz)
            else:
                mgl_nnz = int(np.count_nonzero(st.MGL))
            print(f"[DIAG] Step {current_step} makeKGL: KGL_nnz={kgl_nnz}, max|KGL|={kgl_max:.2e}, MGL_nnz={mgl_nnz}")
    except Exception:
        pass


# ======================================================================
# makeKGL1  – map s-global nodes → local element parent coordinates
# ======================================================================
def makeKGL1():

    st.xmaxL = st.mmLn[:, 0, 1]
    st.xminL = st.mmLn[:, 0, 0]
    st.ymaxL = st.mmLn[:, 1, 1]
    st.yminL = st.mmLn[:, 1, 0]

    # Candidate local elements per s-global node (bbox filter)
    npGeelLabb = []
    for j in range(1, st.nnmGe + 1):
        cands = []
        for eL in range(1, st.nemL + 1):
            if (st.xmaxL[eL - 1] >= st.nodeGe[j - 1][0]
                    and st.xminL[eL - 1] <= st.nodeGe[j - 1][0]
                    and st.ymaxL[eL - 1] >= st.nodeGe[j - 1][1]
                    and st.yminL[eL - 1] <= st.nodeGe[j - 1][1]):
                cands.append(eL)
        npGeelLabb.append(cands)
    st.npGeelLabb = npGeelLabb
    st.nnpGeelLabb = [len(lst) for lst in npGeelLabb]

    # ------------------------------------------------------------------
    def solveXiEtaLRoot(eL, XY):
        xy = st.enodeL[eL - 1]
        if (xy[0, 0] == xy[3, 0] and xy[1, 0] == xy[2, 0]
                and xy[0, 1] == xy[1, 1] and xy[2, 1] == xy[3, 1]):
            xi = (XY[0] - 0.5 * (xy[2, 0] + xy[0, 0])) / (0.5 * (xy[2, 0] - xy[0, 0]))
            eta = (XY[1] - 0.5 * (xy[2, 1] + xy[0, 1])) / (0.5 * (xy[2, 1] - xy[0, 1]))
            return np.array([[xi, eta]], dtype=float)

        tol = 1.0e-12; maxIter = 200; dampTh = 0.5
        prLo, prHi = -1.0, 1.0
        xi, eta = 0.0, 0.0
        XY = np.asarray(XY, dtype=float)
        for _ in range(maxIter):
            N = np.array([N1(xi, eta), N2(xi, eta), N3(xi, eta), N4(xi, eta)], dtype=float)
            Xh = N @ xy
            r = Xh - XY
            if np.linalg.norm(r) < tol:
                break
            J = Dshp([xi, eta]) @ xy
            detJ = np.linalg.det(J)
            if abs(detJ) < 1.0e-14:
                break
            dx = -np.linalg.solve(J, r)
            step = np.linalg.norm(dx)
            if step > dampTh:
                dx *= (dampTh / step)
            xi += dx[0]; eta += dx[1]
            xi = prLo if xi < prLo else (prHi if xi > prHi else xi)
            eta = prLo if eta < prLo else (prHi if eta > prHi else eta)
            if np.linalg.norm(dx) < tol:
                break
        return np.array([xi, eta], dtype=float)

    # Map each s-global node into local parent space
    npGeXiEtaLabb = []
    for j in range(st.nnmGe):
        cand_eL = npGeelLabb[j]
        if len(cand_eL) == 0:
            npGeXiEtaLabb.append([])
            continue
        xi_list = []
        for i in range(st.nnpGeelLabb[j]):
            eL = cand_eL[i]
            sol = solveXiEtaLRoot(eL, st.nodeGe[j])
            sol = np.asarray(sol, dtype=float)
            if sol.ndim == 2:
                sol = sol[0]
            xi_list.append(sol.tolist())
        npGeXiEtaLabb.append(xi_list)

    def rch(xi_etas, eL, tol=1e-8):
        if (-1 - tol <= xi_etas[0] <= 1 + tol) and (-1 - tol <= xi_etas[1] <= 1 + tol):
            return [xi_etas, eL]
        return 0

    def getin(xi_etas, eL_list):
        if len(eL_list) == 0:
            return []
        if isinstance(xi_etas, (list, np.ndarray)) and len(xi_etas) > 0 and isinstance(xi_etas[0], (list, np.ndarray, np.generic)):
            xi_list = [np.array(v, dtype=float).tolist() for v in xi_etas]
        else:
            xi_list = [np.array(xi_etas, dtype=float).tolist()]
        L = min(len(xi_list), len(eL_list))
        return [rch(xi_list[k], eL_list[k]) for k in range(L)]

    XiEtaLeLbf = []
    for j in range(st.nnmGe):
        got = getin(npGeXiEtaLabb[j], npGeelLabb[j])
        filtered = [item for item in got if item != 0]
        XiEtaLeLbf.append(filtered)

    # Keep only the first valid match per node
    _XiEtaLeL = [([] if len(items) == 0 else items[0]) for items in XiEtaLeLbf]
    # store for makeKGL6 if needed
    st._XiEtaLeL = _XiEtaLeL


# ======================================================================
# makeKGL3  – map local nodes → s-global element parent coordinates
# ======================================================================
def makeKGL3():

    xmaxG = [max(p[0] for p in elem) for elem in st.enodeGe]
    xminG = [min(p[0] for p in elem) for elem in st.enodeGe]
    ymaxG = [max(p[1] for p in elem) for elem in st.enodeGe]
    yminG = [min(p[1] for p in elem) for elem in st.enodeGe]

    npLelGeabb = []
    for pL in range(1, st.nnmL + 1):
        cand = []
        x, y = st.nodeL[pL - 1, 0], st.nodeL[pL - 1, 1]
        for eGe in range(1, st.nemGe + 1):
            if (xmaxG[eGe - 1] >= x and xminG[eGe - 1] <= x
                    and ymaxG[eGe - 1] >= y and yminG[eGe - 1] <= y):
                cand.append(eGe)
        npLelGeabb.append(cand)
    st.npLelGeabb = npLelGeabb
    st.nnpLelGeabb = [len(c) for c in npLelGeabb]

    def solveXiEtaGe(eGe, XY):
        xy = np.asarray(st.enodeGe[eGe - 1], dtype=float)
        X1, Y1 = xy[0][0], xy[0][1]
        X2, Y2 = xy[1][0], xy[1][1]
        X3, Y3 = xy[2][0], xy[2][1]
        X4, Y4 = xy[3][0], xy[3][1]
        if (X1 == X4 and X2 == X3 and Y1 == Y2 and Y3 == Y4):
            xi = (XY[0] - 0.5 * (X3 + X1)) / (0.5 * (X3 - X1))
            eta = (XY[1] - 0.5 * (Y3 + Y1)) / (0.5 * (Y3 - Y1))
            return np.array([xi, eta], dtype=float)

        tol = 1.0e-12; maxIter = 200; dampTh = 0.5
        prLo, prHi = -1.0, 1.0
        xi, eta = 0.0, 0.0
        XY = np.asarray(XY, dtype=float)
        for _ in range(maxIter):
            N = np.array([N1(xi, eta), N2(xi, eta), N3(xi, eta), N4(xi, eta)], dtype=float)
            Xh = N @ xy
            r = Xh - XY
            if np.linalg.norm(r) < tol:
                break
            J = Dshp([xi, eta]) @ xy
            detJ = np.linalg.det(J)
            if abs(detJ) < 1.0e-14:
                break
            dx = -np.linalg.solve(J, r)
            step_sz = np.linalg.norm(dx)
            if step_sz > dampTh:
                dx *= (dampTh / step_sz)
            xi += dx[0]; eta += dx[1]
            xi = prLo if xi < prLo else (prHi if xi > prHi else xi)
            eta = prLo if eta < prLo else (prHi if eta > prHi else eta)
            if np.linalg.norm(dx) < tol:
                break
        return np.array([xi, eta], dtype=float)

    st.npLXiEtaGabb = [
        [solveXiEtaGe(npLelGeabb[pL][c], st.nodeL[pL])
         for c in range(st.nnpLelGeabb[pL])]
        for pL in range(st.nnmL)
    ]

    def rch(XiEta_list, eL_list, tol=1e-12):
        L = min(len(XiEta_list), len(eL_list))
        for i in range(L):
            if len(XiEta_list[i]) == 0:
                continue
            xi, eta = XiEta_list[i][0], XiEta_list[i][1]
            if (-1 - tol <= xi <= 1 + tol) and (-1 - tol <= eta <= 1 + tol):
                return [XiEta_list[i].tolist(), eL_list[i]]
        return [[], 0]

    st.XiEtaGeG = [
        rch(st.npLXiEtaGabb[pL], npLelGeabb[pL]) for pL in range(st.nnmL)
    ]

    failed_mappings = sum(1 for xeg in st.XiEtaGeG if xeg is None or (isinstance(xeg, list) and len(xeg) == 0))
    if failed_mappings > 0:
        print(f"[DIAG] makeKGL3: {failed_mappings}/{st.nnmL} local nodes failed to map")


# ======================================================================
# makeKGL6  – assemble K^GL / M^GL using s-global ↔ local overlap
# ======================================================================
def makeKGL6():

    is_static_case = (
        getattr(st, "analysis_mode", "dynamic") == "static"
        or int(getattr(st, "isdynamic", 1)) == 0
    )
    use_sparse_static = is_static_case and int(getattr(st, "static_use_sparse", 1)) == 1
    assemble_mass = not (
        is_static_case and int(getattr(st, "static_skip_mass", 1)) == 1
    )

    if use_sparse_static:
        st.KGL = sp.lil_matrix((st.neqG, st.neqL), dtype=float)
        st.MGL = sp.lil_matrix((st.neqG, st.neqL), dtype=float) if assemble_mass else None
    else:
        st.KGL = np.zeros((st.neqG, st.neqL), dtype=float)
        st.MGL = np.zeros((st.neqG, st.neqL), dtype=float) if assemble_mass else None

    kgl_ngp = int(st.ngpGL)
    if is_static_case:
        kgl_ngp = int(getattr(st, "static_kgl_ngpGL", st.ngpGL))
    if kgl_ngp <= 0:
        raise ValueError(f"Invalid KGL Gauss order: {kgl_ngp}")

    xi_etaGL = np.asarray([(b, a) for a in GP(kgl_ngp) for b in GP(kgl_ngp)], dtype=float)
    weightGL = np.asarray([w1 * w2 for w1 in GW(kgl_ngp) for w2 in GW(kgl_ngp)], dtype=float)
    ngpGL2 = kgl_ngp ** 2
    tol_map = 1.0e-12

    nnGL = np.asarray([np.asarray(shp(pt), dtype=float).ravel() for pt in xi_etaGL], dtype=float)
    dshpGL = np.asarray([Dshp(pt) for pt in xi_etaGL], dtype=float)
    nlsGL = np.asarray([enlarge2(shp(pt)) for pt in xi_etaGL], dtype=float) if assemble_mass else None

    enodeGe = np.asarray(st.enodeGe, dtype=float)
    ge_xmid = 0.5 * (enodeGe[:, 2, 0] + enodeGe[:, 0, 0])
    ge_ymid = 0.5 * (enodeGe[:, 2, 1] + enodeGe[:, 0, 1])
    ge_half_dx = 0.5 * (enodeGe[:, 2, 0] - enodeGe[:, 0, 0])
    ge_half_dy = 0.5 * (enodeGe[:, 2, 1] - enodeGe[:, 0, 1])
    ge_ok = (np.abs(ge_half_dx) > tol_map) & (np.abs(ge_half_dy) > tol_map)
    ge_inv_half_dx = np.zeros_like(ge_half_dx)
    ge_inv_half_dy = np.zeros_like(ge_half_dy)
    ge_inv_half_dx[ge_ok] = 1.0 / ge_half_dx[ge_ok]
    ge_inv_half_dy[ge_ok] = 1.0 / ge_half_dy[ge_ok]
    ge_is_rect = (
        np.isclose(enodeGe[:, 0, 0], enodeGe[:, 3, 0], atol=tol_map)
    ) & (
        np.isclose(enodeGe[:, 1, 0], enodeGe[:, 2, 0], atol=tol_map)
    ) & (
        np.isclose(enodeGe[:, 0, 1], enodeGe[:, 1, 1], atol=tol_map)
    ) & (
        np.isclose(enodeGe[:, 2, 1], enodeGe[:, 3, 1], atol=tol_map)
    ) & ge_ok

    emGe_arr = np.asarray(st.emGe, dtype=int)
    elemL_arr = np.asarray(st.elemL, dtype=int)
    de = st.de
    dRho = st.dRho
    thi = st.thi

    ncpelem = (st.p + 1) * (st.q + 1)
    ncol = 2 * ncpelem
    cols_u = np.arange(0, ncol, 2, dtype=int)
    cols_v = cols_u + 1

    ge_data_cache = {}

    def get_ge_data(iga_elem_id):
        cached = ge_data_cache.get(iga_elem_id)
        if cached is not None:
            return cached

        idu, idv = st.index[iga_elem_id]
        xiE = st.elRangeU[idu - 1]
        etaE = st.elRangeV[idv - 1]
        conn = np.asarray(st.element[iga_elem_id], dtype=int)
        cp_elem = st.controlPts[conn - 1, :]
        iGem = np.empty(2 * conn.size, dtype=int)
        iGem[0::2] = 2 * conn - 2
        iGem[1::2] = 2 * conn - 1
        out = (xiE, etaE, cp_elem, iGem)
        ge_data_cache[iga_elem_id] = out
        return out

    if isinstance(st.enodeL, np.ndarray):
        st.enodeLs = st.enodeL[np.array(st.emLs, dtype=int)]
    else:
        st.enodeLs = np.array([st.enodeL[i] for i in st.emLs], dtype=float)

    st.phyposs = np.einsum("gi,eik->egk", nnGL, st.enodeLs, optimize=True)

    st.elLemGes = [eid for el in st.emLs for eid in st.elLemGe[el]]
    st.XiEtaGeGs = [st.XiEtaGeG[i] for i in st.emLs]

    def solveXiEtaGe(eGe, XY):
        idx = int(eGe) - 1
        X, Y = float(XY[0]), float(XY[1])
        if ge_is_rect[idx]:
            xi = (X - ge_xmid[idx]) * ge_inv_half_dx[idx]
            eta = (Y - ge_ymid[idx]) * ge_inv_half_dy[idx]
            return np.array([xi, eta], dtype=float)

        xy = enodeGe[idx]
        if ge_ok[idx]:
            xi0 = (X - ge_xmid[idx]) * ge_inv_half_dx[idx]
            eta0 = (Y - ge_ymid[idx]) * ge_inv_half_dy[idx]
        else:
            xi0 = 0.0
            eta0 = 0.0

        def equations(p):
            xi, eta = p
            x_eq = (N1(xi, eta) * xy[0, 0] + N2(xi, eta) * xy[1, 0]
                    + N3(xi, eta) * xy[2, 0] + N4(xi, eta) * xy[3, 0] - X)
            y_eq = (N1(xi, eta) * xy[0, 1] + N2(xi, eta) * xy[1, 1]
                    + N3(xi, eta) * xy[2, 1] + N4(xi, eta) * xy[3, 1] - Y)
            return [x_eq, y_eq]

        xi_eta, _, _, _ = fsolve(equations, [xi0, eta0], xtol=1e-12, maxfev=200, full_output=True)
        xi_eta = np.clip(xi_eta, -1.0 - 1e-12, 1.0 + 1e-12)
        return xi_eta

    # ------------------------------------------------------------------
    #  --- emLs: local elements completely inside a single s-global element ---
    # ------------------------------------------------------------------
    elLemGes_arr = np.asarray(st.elLemGes, dtype=int)
    iga_elem_ls = emGe_arr[elLemGes_arr - 1] - 1 if st.nemLs > 0 else np.zeros((0,), dtype=int)

    XiEtaGs = [
        [solveXiEtaGe(elLemGes_arr[eLs], st.phyposs[eLs, gp]) for gp in range(ngpGL2)]
        for eLs in range(st.nemLs)
    ]

    IGAXiEtas = []
    for eLs in range(st.nemLs):
        iga_id = int(iga_elem_ls[eLs])
        xiE, etaE, cp_elem, _ = get_ge_data(iga_id)
        row = []
        for j in range(ngpGL2):
            row.append(
                IGAgemoGetXiEta(
                    iga_id,
                    st.phyposs[eLs, j],
                    XiEtaGs[eLs][j],
                    debug=False,
                    xiE=xiE,
                    etaE=etaE,
                    coord=cp_elem,
                )
            )
        IGAXiEtas.append(row)

    iLes_ls = np.empty((st.nemLs, 8), dtype=int)
    for eLs, elem_id in enumerate(st.emLs):
        nodes = elemL_arr[int(elem_id)]
        iLes = np.empty(8, dtype=int)
        iLes[0::2] = 2 * nodes
        iLes[1::2] = 2 * nodes + 1
        iLes_ls[eLs] = iLes

    for eLs in range(st.nemLs):
        iga_id = int(iga_elem_ls[eLs])
        xiE, etaE, cp_elem, iGem = get_ge_data(iga_id)

        KGLes = np.zeros((ncol, 8), dtype=float)
        MGLes = np.zeros((ncol, 8), dtype=float) if assemble_mass else None
        B = np.zeros((3, ncol), dtype=float)
        func = np.zeros((2, ncol), dtype=float) if assemble_mass else None
        enodeLs_e = st.enodeLs[eLs]

        for j in range(ngpGL2):
            dshp_j = dshpGL[j]
            JLs = dshp_j @ enodeLs_e
            invJLs = np.linalg.inv(JLs)
            BLSs = enlarge(invJLs @ dshp_j)
            if assemble_mass:
                NLSs = nlsGL[j]

            Xi_p, Eta_p = IGAXiEtas[eLs][j]
            Xi = parent2ParametricSpace(xiE, Xi_p)
            Eta = parent2ParametricSpace(etaE, Eta_p)

            ni = FindSpan(st.lenu, st.p, Xi, st.uKnot)
            nj = FindSpan(st.lenv, st.q, Eta, st.vKnot)
            NN, dNdxi, dNdeta = NURBS2DBasisDers(
                ni, nj, st.p, st.q, st.uKnot, st.vKnot, Xi, Eta,
                st.weights, st.lenu, st.lenv,
            )
            dNbf = np.vstack((dNdxi, dNdeta))
            Jxu = dNbf @ cp_elem
            dN = np.linalg.inv(Jxu) @ dNbf

            B.fill(0.0)
            B[0, cols_u] = dN[0, :]
            B[1, cols_v] = dN[1, :]
            B[2, cols_u] = dN[1, :]
            B[2, cols_v] = dN[0, :]
            if assemble_mass:
                func.fill(0.0)
                func[0, cols_u] = NN
                func[1, cols_v] = NN

            jw = np.linalg.det(JLs) * weightGL[j] * thi
            KGLes += (B.T @ de @ BLSs) * jw
            if assemble_mass:
                MGLes += (func.T @ dRho @ NLSs) * jw

        iLes = iLes_ls[eLs]
        st.KGL[np.ix_(iGem, iLes)] += KGLes
        if assemble_mass:
            st.MGL[np.ix_(iGem, iLes)] += MGLes

    # ------------------------------------------------------------------
    #  --- emLm: local elements spanning multiple s-global elements ---
    # ------------------------------------------------------------------
    if st.nemLm == 0:
        if use_sparse_static:
            st.KGL = st.KGL.tocsr()
            st.KGL.eliminate_zeros()
            if assemble_mass:
                st.MGL = st.MGL.tocsr()
                st.MGL.eliminate_zeros()
        return

    enodeLm = np.asarray([st.enodeL[idx] for idx in st.emLm], dtype=float)
    hrefL = st.hrefL

    def hd(quad):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = quad
        j_arr = np.arange(0, hrefL + 1, dtype=np.float64)
        xl1 = x1 + j_arr * (x4 - x1) / hrefL
        yl1 = y1 + j_arr * (y4 - y1) / hrefL
        xl2 = x2 + j_arr * (x3 - x2) / hrefL
        yl2 = y2 + j_arr * (y3 - y2) / hrefL
        t = np.arange(0, hrefL + 1, dtype=np.float64) / hrefL
        X = (xl1[:, None] + (xl2[:, None] - xl1[:, None]) * t[None, :]).reshape(-1)
        Y = (yl1[:, None] + (yl2[:, None] - yl1[:, None]) * t[None, :]).reshape(-1)
        return np.column_stack([X, Y])

    nodeLmh = np.asarray([hd(enodeLm[e]) for e in range(st.nemLm)], dtype=float)

    h = hrefL
    _rng1 = list(range(1, (h + 1) * h + 1))
    _rm_right = {(h + 1) * k for k in range(1, h + 2)}
    _rm_leftp1 = {(h + 1) * k + 1 for k in range(0, h + 1)}
    _rng2 = list(range(h + 2, (h + 1) * (h + 1) + 1))

    e1gh = [i for i in _rng1 if i not in _rm_right]
    e2gh = [i for i in _rng1 if i not in _rm_leftp1]
    e5gh = [i for i in _rng2 if i not in _rm_right]
    e6gh = [i for i in _rng2 if i not in _rm_leftp1]

    nemLh = h * h
    elemLh = list(zip(e1gh, e2gh, e6gh, e5gh))

    conn_h = np.asarray(elemLh, dtype=int) - 1
    enodeLmh = np.empty((st.nemLm, nemLh, 4, 2), dtype=float)
    for eG in range(st.nemLm):
        enodeLmh[eG, :, :, :] = nodeLmh[eG][conn_h, :]

    phyposm = np.einsum("gi,ehik->ehgk", nnGL, enodeLmh, optimize=True)

    elLelGem = [st.elLemGe[idx] for idx in st.emLm]
    nelLelGem = [st.nelLemGe[idx] for idx in st.emLm]

    XiEtaGmbb = [
        [
            [
                [solveXiEtaGe(elLelGem[eL][eG], phyposm[eL, hh, ii])
                 for eG in range(nelLelGem[eL])]
                for ii in range(ngpGL2)
            ]
            for hh in range(nemLh)
        ]
        for eL in range(st.nemLm)
    ]

    def getXiEtaGeG(a, tol=1.0e-12):
        L = len(a)
        for i in range(L):
            if a[i] is None or len(a[i]) == 0:
                continue
            xi, eta = float(a[i][0]), float(a[i][1])
            if (-1.0 - tol <= xi <= 1.0 + tol) and (-1.0 - tol <= eta <= 1.0 + tol):
                return ([xi, eta], i + 1)
        return ([], 0)

    XiEtaGm = [
        [[getXiEtaGeG(XiEtaGmbb[eL][h][i]) for i in range(ngpGL2)]
         for h in range(nemLh)]
        for eL in range(st.nemLm)
    ]

    enodeL_all = np.asarray(st.enodeL, dtype=float)
    l_xmid = 0.5 * (enodeL_all[:, 2, 0] + enodeL_all[:, 0, 0])
    l_ymid = 0.5 * (enodeL_all[:, 2, 1] + enodeL_all[:, 0, 1])
    l_half_dx = 0.5 * (enodeL_all[:, 2, 0] - enodeL_all[:, 0, 0])
    l_half_dy = 0.5 * (enodeL_all[:, 2, 1] - enodeL_all[:, 0, 1])
    l_ok = (np.abs(l_half_dx) > tol_map) & (np.abs(l_half_dy) > tol_map)
    l_inv_half_dx = np.zeros_like(l_half_dx)
    l_inv_half_dy = np.zeros_like(l_half_dy)
    l_inv_half_dx[l_ok] = 1.0 / l_half_dx[l_ok]
    l_inv_half_dy[l_ok] = 1.0 / l_half_dy[l_ok]
    l_is_rect = (
        np.isclose(enodeL_all[:, 0, 0], enodeL_all[:, 3, 0], atol=tol_map)
    ) & (
        np.isclose(enodeL_all[:, 1, 0], enodeL_all[:, 2, 0], atol=tol_map)
    ) & (
        np.isclose(enodeL_all[:, 0, 1], enodeL_all[:, 1, 1], atol=tol_map)
    ) & (
        np.isclose(enodeL_all[:, 2, 1], enodeL_all[:, 3, 1], atol=tol_map)
    ) & l_ok

    def solveXiEtaL(eL, XY):
        idx = int(eL)
        X, Y = float(XY[0]), float(XY[1])
        if l_is_rect[idx]:
            xi = (X - l_xmid[idx]) * l_inv_half_dx[idx]
            eta = (Y - l_ymid[idx]) * l_inv_half_dy[idx]
            return (xi, eta)
        xy = enodeL_all[idx]
        def equations(p):
            xi, eta = p
            eq1 = N1(xi, eta) * xy[0, 0] + N2(xi, eta) * xy[1, 0] + N3(xi, eta) * xy[2, 0] + N4(xi, eta) * xy[3, 0] - X
            eq2 = N1(xi, eta) * xy[0, 1] + N2(xi, eta) * xy[1, 1] + N3(xi, eta) * xy[2, 1] + N4(xi, eta) * xy[3, 1] - Y
            return [eq1, eq2]
        xi_eta, _, ier, _ = fsolve(equations, [0.0, 0.0], full_output=True)
        return tuple(xi_eta)

    XiEtaLm = [
        [[solveXiEtaL(st.emLm[eL], phyposm[eL, h, i, :]) for i in range(ngpGL2)]
         for h in range(nemLh)]
        for eL in range(st.nemLm)
    ]

    elLhelGe = []
    for eL in range(st.nemLm):
        by_sub = []
        nmap = len(elLelGem[eL])
        for h in range(nemLh):
            by_gp = []
            for i in range(ngpGL2):
                hit_idx = int(XiEtaGm[eL][h][i][1])
                if 1 <= hit_idx <= nmap:
                    by_gp.append(int(elLelGem[eL][hit_idx - 1]))
                else:
                    by_gp.append(0)
            by_sub.append(by_gp)
        elLhelGe.append(by_sub)

    IGAXiEtam = []
    for eLm in range(st.nemLm):
        by_sub = []
        for h in range(nemLh):
            by_gp = []
            for i in range(ngpGL2):
                ge_lid = int(elLhelGe[eLm][h][i])
                init = XiEtaGm[eLm][h][i][0]
                # Skip if this integration point doesn't map to any global element
                if ge_lid <= 0 or len(init) == 0:
                    by_gp.append(None)
                    continue
                eG_id = emGe_arr[ge_lid - 1] - 1
                xiE, etaE, cp_elem, _ = get_ge_data(eG_id)
                pos = phyposm[eLm, h, i, :]
                by_gp.append(
                    IGAgemoGetXiEta(
                        eG_id,
                        pos,
                        init,
                        xiE=xiE,
                        etaE=etaE,
                        coord=cp_elem,
                    )
                )
            by_sub.append(by_gp)
        IGAXiEtam.append(by_sub)

    for eLm in range(1, st.nemLm + 1):
        local_nodes = elemL_arr[int(st.emLm[eLm - 1])]
        iLem = np.empty(8, dtype=int)
        iLem[0::2] = 2 * local_nodes
        iLem[1::2] = 2 * local_nodes + 1

        for eLh in range(1, nemLh + 1):
            B = np.zeros((3, ncol), dtype=float)
            func = np.zeros((2, ncol), dtype=float) if assemble_mass else None
            for j in range(1, ngpGL2 + 1):
                # Skip integration points that don't map to global elements
                if IGAXiEtam[eLm - 1][eLh - 1][j - 1] is None:
                    continue

                ge_lid = int(elLhelGe[eLm - 1][eLh - 1][j - 1])
                if ge_lid <= 0:
                    continue

                xi_eta_l = XiEtaLm[eLm - 1][eLh - 1][j - 1]
                dshp_l = Dshp(xi_eta_l)
                JLm = dshp_l @ enodeLm[eLm - 1]
                JLhm = dshpGL[j - 1] @ enodeLmh[eLm - 1, eLh - 1]

                BLSbm = np.linalg.inv(JLm) @ dshp_l
                BLSm = enlarge(BLSbm)
                if assemble_mass:
                    NLSm = enlarge2(shp(xi_eta_l))

                elem_id = emGe_arr[ge_lid - 1] - 1
                xiE, etaE, cp_elem, iGem = get_ge_data(elem_id)

                xi_p, eta_p = IGAXiEtam[eLm - 1][eLh - 1][j - 1]
                Xi = parent2ParametricSpace(xiE, xi_p)
                Eta = parent2ParametricSpace(etaE, eta_p)

                ni = FindSpan(st.lenu, st.p, Xi, st.uKnot)
                nj = FindSpan(st.lenv, st.q, Eta, st.vKnot)
                NN, dNdxi, dNdeta = NURBS2DBasisDers(
                    ni, nj, st.p, st.q, st.uKnot, st.vKnot, Xi, Eta,
                    st.weights, st.lenu, st.lenv,
                )
                dNbf = np.vstack((dNdxi, dNdeta))
                Jxu = dNbf @ cp_elem
                dN = np.linalg.inv(Jxu) @ dNbf

                B.fill(0.0)
                B[0, cols_u] = dN[0, :]
                B[1, cols_v] = dN[1, :]
                B[2, cols_u] = dN[1, :]
                B[2, cols_v] = dN[0, :]
                if assemble_mass:
                    func.fill(0.0)
                    func[0, cols_u] = NN
                    func[1, cols_v] = NN

                jw = np.linalg.det(JLhm) * weightGL[j - 1] * thi
                KGLhe = (B.T @ de @ BLSm) * jw
                if assemble_mass:
                    MGLhe = (func.T @ dRho @ NLSm) * jw

                st.KGL[np.ix_(iGem, iLem)] += KGLhe
                if assemble_mass:
                    st.MGL[np.ix_(iGem, iLem)] += MGLhe

    if use_sparse_static:
        st.KGL = st.KGL.tocsr()
        st.KGL.eliminate_zeros()
        if assemble_mass:
            st.MGL = st.MGL.tocsr()
            st.MGL.eliminate_zeros()
