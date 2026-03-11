"""
NURBS / B-spline utility functions.

Contains: BasisFuns, DerBasisFuns, FindSpanMinus, NURBS2DBasisDers
"""

import numpy as np


def BasisFuns(span, u, p, U):
    """Evaluate nonzero B-spline basis functions N_{span-p} … N_{span} at *u*."""
    if not isinstance(U, np.ndarray):
        U = np.asarray(U, dtype=float)
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    N[0] = 1.0
    eps = 1.0e-14
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(0, j):
            denom = right[r + 1] + left[j - r]
            temp = N[r] / denom if abs(denom) > eps else 0.0
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N


def DerBasisFuns(span, u, p, order, U):
    """
    Derivatives of B-spline basis functions.

    Returns ``ders[k, j]`` with k = 0 … order, j = 0 … p.
    """
    if not isinstance(U, np.ndarray):
        U = np.asarray(U, dtype=float)
    nMat = np.zeros((p + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    nMat[0, 0] = 1.0

    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            nMat[j, r] = right[r + 1] + left[j - r]
            temp = nMat[r, j - 1] / nMat[j, r] if nMat[j, r] != 0 else 0.0
            nMat[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        nMat[j, j] = saved

    ders = np.zeros((order + 1, p + 1))
    aMat = np.zeros((2, p + 1))

    for j in range(p + 1):
        ders[0, j] = nMat[j, p]

    if order == 0:
        return ders

    for r in range(p + 1):
        s1, s2 = 0, 1
        aMat[0, 0] = 1.0
        for k in range(1, order + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                denom = nMat[pk + 1, rk] if nMat[pk + 1, rk] != 0 else 1e-14
                aMat[s2, 0] = aMat[s1, 0] / denom
                d = aMat[s2, 0] * nMat[rk, pk]
            j1 = 1 if rk >= 0 else -rk
            j2 = (k - 1) if (r - 1) <= pk else (p - r)
            for j in range(j1, j2 + 1):
                denom = nMat[pk + 1, rk + j] if nMat[pk + 1, rk + j] != 0 else 1e-14
                aMat[s2, j] = (aMat[s1, j] - aMat[s1, j - 1]) / denom
                d += aMat[s2, j] * nMat[rk + j, pk]
            if r <= pk:
                denom = nMat[pk + 1, r] if nMat[pk + 1, r] != 0 else 1e-14
                aMat[s2, k] = -aMat[s1, k - 1] / denom
                d += aMat[s2, k] * nMat[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1

    r = p
    for k in range(1, order + 1):
        for j in range(p + 1):
            ders[k, j] *= r
        r *= (p - k)

    return ders


def FindSpanMinus(n, p, u, U, debug=False):
    """Find knot span index (binary search, 0-based)."""
    if not isinstance(U, np.ndarray):
        U = np.asarray(U, dtype=float)
    eps = 1.0e-14
    # Right boundary follows the standard NURBS convention: return n (not n-1).
    if u >= U[n + 1] - eps:
        return n
    if u <= U[p] + eps:
        return p
    low = p
    high = n + 1
    mid = (low + high) // 2
    while (u < U[mid]) or (u >= U[mid + 1]):
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


# Default alias
FindSpan = FindSpanMinus


def NURBS2DBasisDers(spanU, spanV, p, q, knotU, knotV,
                     Xi, Eta, weightsGlobal, nU, nV):
    """
    2-D NURBS basis functions and their first derivatives.

    Returns (R, dRdxi, dRdeta), each of shape ((p+1)*(q+1),).
    """
    if not isinstance(knotU, np.ndarray):
        knotU = np.asarray(knotU, dtype=float)
    if not isinstance(knotV, np.ndarray):
        knotV = np.asarray(knotV, dtype=float)
    if not isinstance(weightsGlobal, np.ndarray):
        weightsGlobal = np.asarray(weightsGlobal, dtype=float)

    # Local weight indices
    localIdx = []
    for j in range(q + 1):
        vk = spanV - q + j
        for i in range(p + 1):
            localIdx.append((spanU - p + i) + (nU + 1) * vk)
    weightLocal = weightsGlobal[np.array(localIdx, dtype=int)]

    # 1-D B-spline basis and derivatives
    Nx = np.asarray(BasisFuns(spanU, Xi, p, knotU), dtype=np.float64)
    Ny = np.asarray(BasisFuns(spanV, Eta, q, knotV), dtype=np.float64)
    DNx = np.asarray(DerBasisFuns(spanU, Xi, p, 1, knotU)[1, :], dtype=np.float64)
    DNy = np.asarray(DerBasisFuns(spanV, Eta, q, 1, knotV)[1, :], dtype=np.float64)

    # Tensor product
    basis  = np.array([Nx[i] * Ny[j] for j in range(q + 1) for i in range(p + 1)], dtype=np.float64)
    derivX = np.array([DNx[i] * Ny[j] for j in range(q + 1) for i in range(p + 1)], dtype=np.float64)
    derivY = np.array([Nx[i] * DNy[j] for j in range(q + 1) for i in range(p + 1)], dtype=np.float64)

    # Weight and normalize
    num = (basis * weightLocal).astype(np.float64)
    wTot = np.sum(num, dtype=np.float64)
    dwdxi = np.sum((derivX * weightLocal), dtype=np.float64)
    dwdeta = np.sum((derivY * weightLocal), dtype=np.float64)

    R = (num / wTot).astype(np.float64)
    dRdxi = ((derivX * weightLocal * wTot - num * dwdxi) / (wTot * wTot)).astype(np.float64)
    dRdeta = ((derivY * weightLocal * wTot - num * dwdeta) / (wTot * wTot)).astype(np.float64)

    return R, dRdxi, dRdeta
