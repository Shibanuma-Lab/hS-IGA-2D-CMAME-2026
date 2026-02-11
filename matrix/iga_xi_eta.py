"""
IGAgemoGetXiEta – Newton's method mapping from physical (x,y) to IGA
parent space (ξ,η).

Mathematica: IGAgemoGetXiEta[e_, pos_, init_]
"""

import numpy as np

import core.state as st
from utils.mapping import parent2ParametricSpace
from utils.nurbs import NURBS2DBasisDers, FindSpanMinus as FindSpan


def IGAgemoGetXiEta(
    e, pos, init, debug=False,
    *,
    xiE=None, etaE=None, coord=None, uspan=None, vspan=None,
    tolR=1.0e-12, tolS=1.0e-12, max_iter=100000, damp=0.5,
    prLo=-1.0, prHi=1.0, eps=1.0e-12
):
    """
    Map physical coordinate *pos* to IGA parent space for element *e*.

    Parameters
    ----------
    e : int, 0-based IGA element index
    pos : array-like (x, y)
    init : array-like (ξ₀, η₀) initial guess in [-1,1]²
    """

    if coord is None:
        sctr = np.asarray(st.element[e], dtype=int)
        coord = np.asarray(st.controlPts[sctr - 1], dtype=float)

    if (xiE is None) or (etaE is None):
        idu, idv = st.index[e]
        xiE = st.elRangeU[idu - 1]
        etaE = st.elRangeV[idv - 1]

    out = np.array(init, dtype=float).copy()
    pos = np.asarray(pos, dtype=float)

    eps_p = 1e-11
    if out[0] >= 1.0:
        out[0] = 1.0 - eps_p
    if out[1] >= 1.0:
        out[1] = 1.0 - eps_p
    if out[0] <= -1.0:
        out[0] = -1.0 + eps_p
    if out[1] <= -1.0:
        out[1] = -1.0 + eps_p

    hint_u = None if uspan is None else uspan
    hint_v = None if vspan is None else vspan

    nrmFr = 1.0
    it = 0

    for it in range(max_iter):
        Xi = parent2ParametricSpace(xiE, out[0])
        Eta = parent2ParametricSpace(etaE, out[1])

        if uspan is None:
            uspan_local = FindSpan(st.lenu, st.p, Xi, st.uKnot)
            hint_u = uspan_local
        else:
            uspan_local = uspan

        if vspan is None:
            vspan_local = FindSpan(st.lenv, st.q, Eta, st.vKnot)
            hint_v = vspan_local
        else:
            vspan_local = vspan

        NN, dNdxi, dNdeta = NURBS2DBasisDers(
            uspan_local, vspan_local, st.p, st.q,
            st.uKnot, st.vKnot, Xi, Eta, st.weights, st.lenu, st.lenv,
        )

        fr = NN @ coord - pos
        nrmFr = np.linalg.norm(fr)
        if nrmFr < tolR:
            break

        dNbf = np.vstack([dNdxi, dNdeta])
        a = dNbf[0, :] @ coord[:, 0]
        b = dNbf[0, :] @ coord[:, 1]
        c = dNbf[1, :] @ coord[:, 0]
        d_ = dNbf[1, :] @ coord[:, 1]

        det = a * d_ - b * c
        if abs(det) < 1.0e-14:
            break

        dx0 = -(d_ * fr[0] - b * fr[1]) / det
        dx1 = -(-c * fr[0] + a * fr[1]) / det

        step_size = np.hypot(dx0, dx1)
        if step_size > damp:
            sc = damp / step_size
            dx0 *= sc
            dx1 *= sc
            step_size = damp

        out[0] += dx0
        out[1] += dx1

        out[0] = prLo if out[0] < prLo else (prHi if out[0] > prHi else out[0])
        out[1] = prLo if out[1] < prLo else (prHi if out[1] > prHi else out[1])

        if step_size < tolS:
            break

    if it > 50 or nrmFr > 1e-6:
        print(f"[DIAG] IGAgemoGetXiEta: iter={it}, nrmFr={nrmFr:.2e}, e={e}")

    return out
