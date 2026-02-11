"""
makeKGL – assemble the global–local coupling matrices K^GL and M^GL.

Contains:  makeKGL(), makeKGL1(), makeKGL3(), makeKGL6()
"""

import numpy as np
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
            kgl_nnz = np.count_nonzero(st.KGL)
            kgl_max = np.max(np.abs(st.KGL))
            mgl_nnz = np.count_nonzero(st.MGL)
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

    st.KGL = np.zeros((st.neqG, st.neqL), dtype=float)
    st.MGL = np.zeros((st.neqG, st.neqL), dtype=float)

    if isinstance(st.enodeL, np.ndarray):
        st.enodeLs = st.enodeL[np.array(st.emLs, dtype=int)]
    else:
        st.enodeLs = np.array([st.enodeL[i] for i in st.emLs], dtype=float)

    st.phyposs = np.array([
        [(np.atleast_2d(nnGL_g) @ enodeLs_e).ravel() for nnGL_g in st.nnGL]
        for enodeLs_e in st.enodeLs
    ], dtype=float)

    st.elLemGes = [eid for el in st.emLs for eid in st.elLemGe[el]]
    st.XiEtaGeGs = [st.XiEtaGeG[i] for i in st.emLs]

    # ------------------------------------------------------------------
    def solveXiEtaGe(eGe, XY, tol=1e-12):
        xy = np.asarray(st.enodeGe[eGe - 1], dtype=float)
        X, Y = float(XY[0]), float(XY[1])
        is_rect = (np.isclose(xy[0, 0], xy[3, 0], atol=tol)
                   and np.isclose(xy[1, 0], xy[2, 0], atol=tol)
                   and np.isclose(xy[0, 1], xy[1, 1], atol=tol)
                   and np.isclose(xy[2, 1], xy[3, 1], atol=tol))
        if is_rect:
            dx = xy[2, 0] - xy[0, 0]; dy = xy[2, 1] - xy[0, 1]
            if abs(dx) > tol and abs(dy) > tol:
                xi = (X - 0.5 * (xy[2, 0] + xy[0, 0])) / (0.5 * dx)
                eta = (Y - 0.5 * (xy[2, 1] + xy[0, 1])) / (0.5 * dy)
                return np.array([xi, eta], dtype=float)

        dx = xy[2, 0] - xy[0, 0]; dy = xy[2, 1] - xy[0, 1]
        if abs(dx) > tol and abs(dy) > tol:
            xi0 = (X - 0.5 * (xy[2, 0] + xy[0, 0])) / (0.5 * dx)
            eta0 = (Y - 0.5 * (xy[2, 1] + xy[0, 1])) / (0.5 * dy)
        else:
            xi0, eta0 = 0.0, 0.0

        def equations(p):
            xi, eta = p
            x_eq = (N1(xi, eta) * xy[0, 0] + N2(xi, eta) * xy[1, 0]
                    + N3(xi, eta) * xy[2, 0] + N4(xi, eta) * xy[3, 0] - X)
            y_eq = (N1(xi, eta) * xy[0, 1] + N2(xi, eta) * xy[1, 1]
                    + N3(xi, eta) * xy[2, 1] + N4(xi, eta) * xy[3, 1] - Y)
            return [x_eq, y_eq]

        xi_eta, info, ier, _ = fsolve(equations, [xi0, eta0], xtol=1e-12, maxfev=200, full_output=True)
        xi_eta = np.clip(xi_eta, -1.0 - 1e-12, 1.0 + 1e-12)
        return xi_eta

    # ------------------------------------------------------------------
    #  --- emLs: local elements completely inside a single s-global element ---
    # ------------------------------------------------------------------
    ngpGL2 = st.ngpGL ** 2

    XiEtaGs = [
        [solveXiEtaGe(st.elLemGes[eL - 1], st.phyposs[eL - 1, i - 1])
         for i in range(1, ngpGL2 + 1)]
        for eL in range(1, st.nemLs + 1)
    ]

    IGAXiEtas = [
        [IGAgemoGetXiEta(
            (st.emGe[st.elLemGes[eLs - 1] - 1] - 1),
            st.phyposs[eLs - 1, j],
            XiEtaGs[eLs - 1][j],
            debug=(j == 0 and eLs == 1),
        ) for j in range(ngpGL2)]
        for eLs in range(1, st.nemLs + 1)
    ]

    ncpelem = (st.p + 1) * (st.q + 1)

    for eLs in range(1, st.nemLs + 1):
        uvid = st.index[st.emGe[st.elLemGes[eLs - 1] - 1] - 1]
        idu, idv = uvid
        xiE = st.elRangeU[idu - 1]
        etaE = st.elRangeV[idv - 1]

        KGLes = np.zeros((2 * ncpelem, 8), dtype=float)
        MGLes = np.zeros((2 * ncpelem, 8), dtype=float)

        for j in range(1, ngpGL2 + 1):
            JLs = Dshp(st.xi_etaGL[j - 1]) @ st.enodeLs[eLs - 1]
            BLSbs = np.linalg.inv(JLs) @ Dshp(st.xi_etaGL[j - 1])
            BLSs = enlarge(BLSbs)
            NLSs = enlarge2(shp(st.xi_etaGL[j - 1]))

            B = np.zeros((3, 2 * ncpelem), dtype=float)
            func = np.zeros((2, 2 * ncpelem), dtype=float)

            Xi_p, Eta_p = IGAXiEtas[eLs - 1][j - 1]
            Xi = parent2ParametricSpace(xiE, Xi_p)
            Eta = parent2ParametricSpace(etaE, Eta_p)

            ni = FindSpan(st.lenu, st.p, Xi, st.uKnot)
            nj = FindSpan(st.lenv, st.q, Eta, st.vKnot)
            NN, dNdxi, dNdeta = NURBS2DBasisDers(
                ni, nj, st.p, st.q, st.uKnot, st.vKnot, Xi, Eta,
                st.weights, st.lenu, st.lenv,
            )
            dNbf = np.vstack([dNdxi, dNdeta])
            cp_elem = st.controlPts[
                np.array(st.element[st.emGe[st.elLemGes[eLs - 1] - 1] - 1], dtype=int) - 1, :
            ]
            Jxu = dNbf @ cp_elem
            dN = np.linalg.inv(Jxu) @ dNbf

            for nid in range(1, ncpelem + 1):
                dNdx = dN[0, nid - 1]; dNdy = dN[1, nid - 1]
                j1 = 2 * nid - 1; j2 = 2 * nid
                B[0, j1 - 1] = dNdx; B[1, j2 - 1] = dNdy
                B[2, j1 - 1] = dNdy; B[2, j2 - 1] = dNdx
                func[0, j1 - 1] = NN[nid - 1]; func[1, j2 - 1] = NN[nid - 1]

            KGLes += (B.T @ st.de @ BLSs) * np.linalg.det(JLs) * st.weightL[j - 1] * st.thi
            MGLes += (func.T @ st.dRho @ NLSs) * np.linalg.det(JLs) * st.weightL[j - 1] * st.thi

        iGes = np.array([
            [2 * st.element[st.emGe[st.elLemGes[eLs - 1] - 1] - 1][i] - 2,
             2 * st.element[st.emGe[st.elLemGes[eLs - 1] - 1] - 1][i] - 1]
            for i in range(ncpelem)
        ], dtype=int).ravel()

        iLes = np.array([
            [2 * st.elemL[st.emLs[eLs - 1]][i],
             2 * st.elemL[st.emLs[eLs - 1]][i] + 1]
            for i in range(4)
        ], dtype=int).ravel()

        st.KGL[np.ix_(iGes, iLes)] += KGLes
        st.MGL[np.ix_(iGes, iLes)] += MGLes

    # ------------------------------------------------------------------
    #  --- emLm: local elements spanning multiple s-global elements ---
    # ------------------------------------------------------------------
    enodeLm = [st.enodeL[idx] for idx in st.emLm]
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

    nodeLmh = [hd(enodeLm[e]) for e in range(st.nemLm)]

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

    enodeLmh = np.empty((st.nemLm, nemLh, 4, 2), dtype=float)
    for eG in range(st.nemLm):
        for hh in range(nemLh):
            conn = np.array(elemLh[hh], dtype=int) - 1
            enodeLmh[eG, hh, :, :] = np.asarray(nodeLmh[eG])[conn, :]

    phyposm = np.empty((st.nemLm, nemLh, ngpGL2, 2), dtype=float)
    for eG in range(st.nemLm):
        for hh in range(nemLh):
            for i in range(ngpGL2):
                Ni = np.atleast_2d(st.nnGL[i])
                xy = (Ni @ enodeLmh[eG, hh]).ravel()
                phyposm[eG, hh, i, :] = xy

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

    def solveXiEtaL(eL, XY):
        xy = st.enodeL[eL]
        if (xy[0, 0] == xy[3, 0] and xy[1, 0] == xy[2, 0]
                and xy[0, 1] == xy[1, 1] and xy[2, 1] == xy[3, 1]):
            xi = (XY[0] - 0.5 * (xy[2, 0] + xy[0, 0])) / (0.5 * (xy[2, 0] - xy[0, 0]))
            eta = (XY[1] - 0.5 * (xy[2, 1] + xy[0, 1])) / (0.5 * (xy[2, 1] - xy[0, 1]))
            return (xi, eta)
        def equations(p):
            xi, eta = p
            eq1 = N1(xi, eta) * xy[0, 0] + N2(xi, eta) * xy[1, 0] + N3(xi, eta) * xy[2, 0] + N4(xi, eta) * xy[3, 0] - XY[0]
            eq2 = N1(xi, eta) * xy[0, 1] + N2(xi, eta) * xy[1, 1] + N3(xi, eta) * xy[2, 1] + N4(xi, eta) * xy[3, 1] - XY[1]
            return [eq1, eq2]
        xi_eta, _, ier, _ = fsolve(equations, [0.0, 0.0], full_output=True)
        return tuple(xi_eta)

    XiEtaLm = [
        [[solveXiEtaL(st.emLm[eL], phyposm[eL, h, i, :]) for i in range(ngpGL2)]
         for h in range(nemLh)]
        for eL in range(st.nemLm)
    ]

    elLhelGe = [
        [[elLelGem[eL][XiEtaGm[eL][h][i][1] - 1] for i in range(ngpGL2)]
         for h in range(nemLh)]
        for eL in range(st.nemLm)
    ]

    IGAXiEtam = []
    for eLm in range(st.nemLm):
        row = []
        for i in range(ngpGL2):
            # Skip if this integration point doesn't map to any global element
            if len(XiEtaGm[eLm][0][i][0]) == 0:
                row.append(None)
            else:
                eG_id = st.emGe[elLhelGe[eLm][0][i] - 1] - 1
                pos = phyposm[eLm, 0, i, :]
                init = XiEtaGm[eLm][0][i][0]
                row.append(IGAgemoGetXiEta(eG_id, pos, init))
        IGAXiEtam.append(row)

    ncpelem = (st.p + 1) ** 2

    for eLm in range(1, st.nemLm + 1):
        for eLh in range(1, nemLh + 1):
            for j in range(1, ngpGL2 + 1):
                # Skip integration points that don't map to global elements
                if IGAXiEtam[eLm - 1][j - 1] is None:
                    continue

                KGLhe = np.zeros((2 * ncpelem, 8), dtype=float)
                MGLhe = np.zeros((2 * ncpelem, 8), dtype=float)

                JLm = Dshp(XiEtaLm[eLm - 1][eLh - 1][j - 1]) @ enodeLm[eLm - 1]
                JLhm = Dshp(st.xi_etaGL[j - 1]) @ enodeLmh[eLm - 1, eLh - 1]

                BLSbm = np.linalg.inv(JLm) @ Dshp(XiEtaLm[eLm - 1][eLh - 1][j - 1])
                BLSm = enlarge(BLSbm)
                NLSm = enlarge2(shp(XiEtaLm[eLm - 1][eLh - 1][j - 1]))

                idu, idv = st.index[st.emGe[elLhelGe[eLm - 1][eLh - 1][j - 1] - 1] - 1]
                xiE = st.elRangeU[idu - 1]
                etaE = st.elRangeV[idv - 1]

                B = np.zeros((3, 2 * ncpelem), dtype=float)
                func = np.zeros((2, 2 * ncpelem), dtype=float)

                xi_p, eta_p = IGAXiEtam[eLm - 1][j - 1]
                Xi = parent2ParametricSpace(xiE, xi_p)
                Eta = parent2ParametricSpace(etaE, eta_p)

                ni = FindSpan(st.lenu, st.p, Xi, st.uKnot)
                nj = FindSpan(st.lenv, st.q, Eta, st.vKnot)
                NN, dNdxi, dNdeta = NURBS2DBasisDers(
                    ni, nj, st.p, st.q, st.uKnot, st.vKnot, Xi, Eta,
                    st.weights, st.lenu, st.lenv,
                )
                dNbf = np.vstack([dNdxi, dNdeta])

                elem_id = st.emGe[elLhelGe[eLm - 1][eLh - 1][j - 1] - 1] - 1
                cp_idx = np.array(st.element[elem_id], dtype=int) - 1
                cp_elem = st.controlPts[cp_idx, :]
                Jxu = dNbf @ cp_elem
                dN = np.linalg.inv(Jxu) @ dNbf

                for nid in range(1, ncpelem + 1):
                    dNdx = dN[0, nid - 1]; dNdy = dN[1, nid - 1]
                    j1 = 2 * nid - 1; j2 = 2 * nid
                    B[0, j1 - 1] = dNdx; B[1, j2 - 1] = dNdy
                    B[2, j1 - 1] = dNdy; B[2, j2 - 1] = dNdx
                    func[0, j1 - 1] = NN[nid - 1]; func[1, j2 - 1] = NN[nid - 1]

                KGLhe += (B.T @ st.de @ BLSm) * np.linalg.det(JLhm) * st.weightGL[j - 1] * st.thi
                MGLhe += (func.T @ st.dRho @ NLSm) * np.linalg.det(JLhm) * st.weightGL[j - 1] * st.thi

                iGem = []
                for ii in range(1, (st.p + 1) ** 2 + 1):
                    n = st.element[elem_id][ii - 1]
                    iGem.extend([2 * n - 2, 2 * n - 1])

                iLem = []
                for i in range(1, 5):
                    n = st.elemL[st.emLm[eLm - 1]][i - 1]
                    iLem.extend([2 * n, 2 * n + 1])

                st.KGL[np.ix_(iGem, iLem)] += KGLhe
                st.MGL[np.ix_(iGem, iLem)] += MGLhe
