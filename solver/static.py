"""
solvestatic – linear static solve (step == 0).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
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
        kk = st.KG
    else:
        if sp.issparse(st.KG) or sp.issparse(st.KGL) or sp.issparse(st.KL):
            KG = st.KG if sp.issparse(st.KG) else sp.csr_matrix(np.asarray(st.KG, dtype=float))
            KGL = st.KGL if sp.issparse(st.KGL) else sp.csr_matrix(np.asarray(st.KGL, dtype=float))
            KL = st.KL if sp.issparse(st.KL) else sp.csr_matrix(np.asarray(st.KL, dtype=float))
            kk = sp.bmat([[KG, KGL], [KGL.T, KL]], format="csr")
        else:
            upper = np.concatenate((st.KG, st.KGL), axis=1)
            lower = np.concatenate((st.KGL.T, st.KL), axis=1)
            kk = np.concatenate((upper, lower), axis=0)

    if sp.issparse(kk):
        kkef = kk[fixdof, :][:, st.freedof]
        kkfe = kkef.T
        kkf = kk[st.freedof, :][:, st.freedof]
    else:
        kkef = kk[np.ix_(fixdof, st.freedof)]
        kkfe = kkef.T
        kkf = kk[np.ix_(st.freedof, st.freedof)]

    forcef = st.force[st.freedof]
    dise   = st.dis[fixdof]

    if nebc != nebc0:
        forcef = forcef - kkfe @ dise

    if sp.issparse(kkf):
        solver_mode = str(getattr(st, "static_linear_solver", "auto")).lower()
        iter_tol = float(getattr(st, "static_iter_tol", 1.0e-10))
        iter_maxiter = int(getattr(st, "static_iter_maxiter", 50000))
        iter_switch_dof = int(getattr(st, "static_iter_switch_dof", 120000))

        use_iterative = (
            solver_mode == "cg"
            or (solver_mode == "auto" and int(kkf.shape[0]) >= iter_switch_dof)
        )

        if use_iterative:
            diag = np.asarray(kkf.diagonal(), dtype=float)
            safe_diag = np.where(np.abs(diag) > 1.0e-16, diag, 1.0)
            M = spla.LinearOperator(
                kkf.shape,
                matvec=lambda x: x / safe_diag,
                dtype=float,
            )
            disp, info = spla.cg(
                kkf,
                forcef,
                rtol=iter_tol,
                atol=0.0,
                maxiter=iter_maxiter,
                M=M,
            )
            if info == 0:
                st.disf = disp
            else:
                print(f"[WARN] CG did not fully converge (info={info}), fallback to sparse direct solve.")
                st.disf = spla.spsolve(kkf.tocsc(), forcef)
        else:
            st.disf = spla.spsolve(kkf.tocsc(), forcef)
    else:
        st.disf = np.linalg.solve(kkf, forcef)
    st.dis[st.freedof] = st.disf

    RFf = kkef @ st.disf
    st.RF[fixdof] = np.asarray(RFf, dtype=float).reshape(-1)
