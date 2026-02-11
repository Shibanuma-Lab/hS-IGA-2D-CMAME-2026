"""
solvestatic – linear static solve (step == 0).
"""

import numpy as np
import core.state as st


def solvestatic():
    """Solve K·u = f for the static case (step 0)."""
    neq  = int(st.neq)
    ndof = int(st.ndof)

    st.dis   = np.zeros(neq, dtype=float)
    st.V     = np.zeros(neq, dtype=float)
    st.A     = np.zeros(neq, dtype=float)
    st.force = np.zeros(neq, dtype=float)
    st.RF    = np.zeros(neq, dtype=float)
    st.RFM   = np.zeros(neq, dtype=float)

    ebc = st.ebc
    nbc = st.nbc
    nebc  = len(ebc)
    nnbc  = len(nbc)
    nebc0 = int(np.count_nonzero(ebc[:, 2] == 0.0))

    # Neumann (force)
    if nnbc > 0:
        forcedof = (ndof * (nbc[:, 0].astype(int) - 1) +
                    (nbc[:, 1].astype(int) - 1))
        st.force[forcedof] = nbc[:, 2]

    # Dirichlet
    fixdof = (ndof * ebc[:, 0].astype(int) +
              (ebc[:, 1].astype(int) - 1)).astype(int)
    st.dis[fixdof] = ebc[:, 2]

    all_dofs = np.arange(neq, dtype=int)
    st.freedof = np.setdiff1d(all_dofs, fixdof, assume_unique=False)

    # Coupled stiffness
    if int(st.islocal) == 0:
        kk = np.asarray(st.KG, dtype=float)
    else:
        upper = np.concatenate((st.KG, st.KGL), axis=1)
        lower = np.concatenate((st.KGL.T, st.KL), axis=1)
        kk = np.concatenate((upper, lower), axis=0)

    kke  = kk[np.ix_(fixdof, fixdof)]
    kkef = kk[np.ix_(fixdof, st.freedof)]
    kkfe = kkef.T
    kkf  = kk[np.ix_(st.freedof, st.freedof)]

    forcef = st.force[st.freedof]
    dise   = st.dis[fixdof]

    if nebc != nebc0:
        forcef = forcef - kkfe @ dise

    st.disf = np.linalg.solve(kkf, forcef)
    st.dis[st.freedof] = st.disf

    RFf = kkef @ st.disf
    st.RF[fixdof] = RFf
