"""
solvedynamic – Newmark-β / HHT-α time integration for steps ≥ 1.
"""

import numpy as np
from datetime import datetime
import core.state as st


def solvedynamic():
    """Perform one dynamic time step with Newmark or HHT-α integration."""
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

    # Neumann
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

    # ---- Build coupled kk / mm ----
    if int(st.islocal) == 0:
        kk = np.asarray(st.KG, dtype=float)
        mm = np.asarray(st.MG, dtype=float)
    else:
        kk = np.block([
            [np.asarray(st.KG, dtype=float),  np.asarray(st.KGL, dtype=float)],
            [np.asarray(st.KGL, dtype=float).T, np.asarray(st.KL, dtype=float)],
        ])
        mm = np.block([
            [np.asarray(st.MG, dtype=float),  np.asarray(st.MGL, dtype=float)],
            [np.asarray(st.MGL, dtype=float).T, np.asarray(st.ML, dtype=float)],
        ])

    # Rayleigh damping
    if st.alpha_rayleigh == 0:
        cc = st.beta_rayleigh * kk
    else:
        cc = st.alpha_rayleigh * mm + st.beta_rayleigh * kk

    # ---- Block extraction ----
    kke  = kk[np.ix_(fixdof, fixdof)]
    kkef = kk[np.ix_(fixdof, st.freedof)]
    kkfe = kkef.T
    kkf  = kk[np.ix_(st.freedof, st.freedof)]

    mme  = mm[np.ix_(fixdof, fixdof)]
    mmef = mm[np.ix_(fixdof, st.freedof)]
    mmfe = mmef.T
    mmf  = mm[np.ix_(st.freedof, st.freedof)]

    cce  = cc[np.ix_(fixdof, fixdof)]
    ccef = cc[np.ix_(fixdof, st.freedof)]
    ccfe = ccef.T
    ccf  = cc[np.ix_(st.freedof, st.freedof)]

    forcef = st.force[st.freedof]
    dise   = st.dis[fixdof]

    if nebc != nebc0:
        forcef = forcef - kkfe @ dise

    dt    = float(st.Delta_t)
    gamma = float(st.HHT_gamma)
    beta  = float(st.HHT_beta)

    # ---- Standard Newmark (zentai == 0) vs HHT (zentai != 0) ----
    if int(st.zentai) == 0:
        disinif = st.disini[st.freedof].copy()
        Vinif   = np.asarray(st.Vini, dtype=float).ravel()[st.freedof]
        Ainif   = np.asarray(st.Aini, dtype=float).ravel()[st.freedof]

        a1 = gamma * dt
        a2 = beta * dt * dt
        b1 = (1.0 - gamma) * dt
        b2 = dt
        b3 = (0.5 - beta) * dt * dt
        c1 = 1.0 + st.alpha_rayleigh * a1
        c2 = a2 + st.beta_rayleigh * a1

        mcklf = c1 * mmf + c2 * kkf
        mckrf = (forcef
                 - st.alpha_rayleigh * mmf @ (Vinif + b1 * Ainif)
                 - st.beta_rayleigh  * kkf @ (Vinif + b1 * Ainif)
                 - kkf @ (disinif + b2 * Vinif + b3 * Ainif))

        # Optional diagnostics (disabled by default due high cost on large systems)
        try:
            cond_diag_enable = int(getattr(st, "diag_cond_enable", 0)) == 1
            if cond_diag_enable:
                current_step = int(getattr(st, "step", -1))
                cond_diag_start = int(getattr(st, "diag_cond_start", 95))
                cond_diag_every = max(1, int(getattr(st, "diag_cond_every", 10)))
                if (
                    current_step >= cond_diag_start
                    and (current_step - cond_diag_start) % cond_diag_every == 0
                ):
                    cond_diag_mode = str(getattr(st, "diag_cond_mode", "proxy")).strip().lower()
                    if cond_diag_mode == "exact":
                        cond_num = np.linalg.cond(mcklf)
                        print(f"[DIAG] Step {current_step}: cond_exact(mcklf)={cond_num:.2e}")
                    else:
                        # Cheap proxy (one linear solve) to avoid SVD-level cost.
                        rhs_probe = np.ones(mcklf.shape[0], dtype=float)
                        sol_probe = np.linalg.solve(mcklf, rhs_probe)
                        num = np.linalg.norm(mcklf, ord=np.inf) * np.linalg.norm(sol_probe, ord=np.inf)
                        den = max(np.linalg.norm(rhs_probe, ord=np.inf), 1.0e-30)
                        cond_proxy = num / den
                        print(f"[DIAG] Step {current_step}: cond_proxy(mcklf)={cond_proxy:.2e}")
        except Exception:
            pass

        print("LinearSolve", datetime.now())
        Af = np.linalg.solve(mcklf, mckrf)
        print("LinearSolve Finish", datetime.now())

        disf = disinif + dt * Vinif + dt * dt * ((0.5 - beta) * Ainif + beta * Af)
        Vf   = Vinif + dt * ((1.0 - gamma) * Ainif + gamma * Af)

        RFf  = kkef @ disf
        RFMf = mmef @ Af + ccef @ Vf + kkef @ disf

        st.dis[st.freedof] = disf
        st.V[st.freedof]   = Vf
        st.A[st.freedof]   = Af
        st.RF[fixdof]      = RFf
        st.RFM[fixdof]     = RFMf

    else:
        # ========== HHT-alpha method ==========
        disinif = np.asarray(st.disini, dtype=float).ravel()[st.freedof]
        Vinif   = np.asarray(st.Vini,   dtype=float).ravel()[st.freedof]
        Ainif   = np.asarray(st.Aini,   dtype=float).ravel()[st.freedof]

        alphaH = float(st.HHT_alpha)

        a1 = gamma * dt
        a2 = beta * dt * dt

        # Predictor terms
        uPred = disinif + dt * Vinif + (0.5 - beta) * dt * dt * Ainif
        vPred = Vinif + (1.0 - gamma) * dt * Ainif

        # Forces at n+1 and n (for constant load, fn = fnp1)
        fnp1 = forcef
        fn   = forcef  # Assuming constant load; store previous step if needed

        # Effective matrix
        mcklf = mmf + (1.0 + alphaH) * a1 * ccf + (1.0 + alphaH) * a2 * kkf

        # Effective RHS
        mckrf = ((1.0 + alphaH) * fnp1 - alphaH * fn
                 + alphaH * (ccf @ Vinif + kkf @ disinif)
                 - (1.0 + alphaH) * (ccf @ vPred + kkf @ uPred))

        print("LinearSolve", datetime.now())
        Af = np.linalg.solve(mcklf, mckrf)
        print("LinearSolve Finish", datetime.now())

        # Update displacement and velocity
        disf = uPred + beta * dt * dt * Af
        Vf   = vPred + gamma * dt * Af

        # Reaction forces
        RFf  = kkef @ disf
        RFMf = mmef @ Af + ccef @ Vf + kkef @ disf

        st.dis[st.freedof] = disf
        st.V[st.freedof]   = Vf
        st.A[st.freedof]   = Af
        st.RF[fixdof]      = RFf
        st.RFM[fixdof]     = RFMf
