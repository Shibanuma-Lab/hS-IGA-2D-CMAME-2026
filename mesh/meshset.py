"""
meshset – compute mesh info, s-global element identification, and
overlap data needed for the S-version coupling.

Mathematica: meshset[] := Module[{}, ...]
"""

import numpy as np

import core.state as st


def meshset():
    """Compute combined mesh information and s-global element data."""

    st.ndof = 2

    # --- Global mesh info ---
    st.nnmG = len(st.nodeG)
    st.nemG = len(st.elemG)
    st.neqG = st.ndof * st.nnmG
    st.enodeG = np.array(
        [[st.nodeG[idx - 1] for idx in elem] for elem in st.elemG], dtype=float
    )

    if int(st.islocal) == 0:
        st.nnm = st.nnmG
        st.nem = st.nemG
        st.neq = st.neqG
        return

    # ===== islocal == 1: S-version FEM =====
    st.nodeL2DAllMa2D.append(np.asarray(st.nodeL, dtype=float))

    st.nnmL = len(st.nodeL)
    st.nemL = len(st.elemL)
    st.neqL = st.ndof * st.nnmL
    st.enodeL = np.array(
        [[st.nodeL[idx] for idx in elem] for elem in st.elemL], dtype=float
    )

    st.nnm = st.nnmG + st.nnmL
    st.nem = st.nemG + st.nemL
    st.neq = st.neqG + st.neqL

    # --- min/max per element for (global-visual mesh, local mesh) ---
    def minmaxX(nodeX, elemX):
        nodeposisions = [nodeX[elemX[i]] for i in range(len(elemX))]
        elementxyposisions = np.array(
            [(nodeposisions[i]).T for i in range(len(nodeposisions))], dtype=np.float64
        )
        xminX = np.array([min(e[0]) for e in elementxyposisions], dtype=np.float64)
        xmaxX = np.array([max(e[0]) for e in elementxyposisions], dtype=np.float64)
        yminX = np.array([min(e[1]) for e in elementxyposisions], dtype=np.float64)
        ymaxX = np.array([max(e[1]) for e in elementxyposisions], dtype=np.float64)
        return xminX, xmaxX, yminX, ymaxX

    xminG, xmaxG, yminG, ymaxG = minmaxX(st.nodeVis, st.elemVis)
    xminL, xmaxL, yminL, ymaxL = minmaxX(st.nodeL, st.elemL)

    st.mmGn = np.stack(
        [np.stack([xminG, xmaxG], axis=1), np.stack([yminG, ymaxG], axis=1)], axis=1
    )
    st.mmLn = np.stack(
        [np.stack([xminL, xmaxL], axis=1), np.stack([yminL, ymaxL], axis=1)], axis=1
    )

    st.minxG = float(np.min(st.mmLn[:, 0, 0]))
    st.maxxG = float(np.max(st.mmLn[:, 0, 1]))
    st.minyG = float(np.min(st.mmLn[:, 1, 0]))
    st.maxyG = float(np.max(st.mmLn[:, 1, 1]))

    scale = max(float(np.max(st.mmLn[:, :, 1]) - np.min(st.mmLn[:, :, 0])), 1.0)
    eps = 1e-12 * scale

    st.LGsupabb = [
        i
        for (i, mm) in ((k, st.mmGn[k - 1]) for k in range(1, len(st.mmGn) + 1))
        if (
            mm[0][0] <= st.maxxG + eps
            and mm[0][1] + eps >= st.minxG
            and mm[1][0] <= st.maxyG + eps
            and mm[1][1] + eps >= st.minyG
        )
    ]

    st.LGsup = [
        [
            eG
            for eG in st.LGsupabb
            if (
                st.mmGn[eG - 1][0][0] <= st.mmLn[eL][0][1] + eps
                and st.mmGn[eG - 1][0][1] + eps >= st.mmLn[eL][0][0]
                and st.mmGn[eG - 1][1][0] <= st.mmLn[eL][1][1] + eps
                and st.mmGn[eG - 1][1][1] + eps >= st.mmLn[eL][1][0]
            )
        ]
        for eL in range(st.nemL)
    ]

    st.emGe = sorted(set(sum(st.LGsup, [])))
    st.nemGe = len(st.emGe)

    st.nmGe = sorted({int(n) for e in st.emGe for n in st.elemVis[e - 1]})
    st.nnmGe = len(st.nmGe)

    st.nodeGe = [st.nodeVis[n] for n in st.nmGe]

    st.emGeR = [0] * st.nemG
    for idx, e in enumerate(st.emGe):
        st.emGeR[e - 1] = idx + 1

    st.nmGeR = [0] * len(st.nodeVis)
    for idx, n in enumerate(st.nmGe):
        st.nmGeR[n] = idx + 1

    st.neqGe = st.ndof * st.nnmGe

    st.elemGe = [[st.nmGeR[n] for n in st.elemVis[e - 1]] for e in st.emGe]
    st.enodeGe = [[st.nodeVis[n] for n in st.elemVis[e - 1]] for e in st.emGe]

    st.elLemGe = [[st.emGeR[e - 1] for e in lst] for lst in st.LGsup]
    st.nelLemGe = [len(lst) for lst in st.elLemGe]

    st.emLs = [idx for idx, val in enumerate(st.nelLemGe) if val == 1]
    st.nemLs = len(st.emLs)
    st.emLm = [i for i in range(st.nemL) if i not in st.emLs]
    st.nemLm = len(st.emLm)
