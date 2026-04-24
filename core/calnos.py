"""
calnos – compute normalised stress / stress intensity factors.

Mathematica: calnos[] (invoked after all steps of a job).
"""

import os

import numpy as np
from scipy import special as sp
from scipy.optimize import brentq

import core.state as st


def wave_speeds(EE=None, nu=None, rho=None):
    """Return longitudinal/shear/Rayleigh wave speeds used by Broberg formulas."""
    EE = float(getattr(st, "EE", 2.06e11) if EE is None else EE)
    nu = float(getattr(st, "nu", 0.3) if nu is None else nu)
    rho = float(getattr(st, "rho", 7800.0) if rho is None else rho)

    C1 = np.sqrt((1 - nu) / ((1 + nu) * (1 - 2 * nu)) * EE / rho)
    C2 = np.sqrt(1 / (2 * (1 + nu)) * EE / rho)
    VR = 0.577 * np.sqrt(EE / rho)
    return C1, C2, VR


def _broberg_k(nu):
    return np.sqrt((1 - 2 * nu) / (2 * (1 - nu)))


def _first_if_sequence(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError("Expected non-empty list/tuple/array.")
        return value[0]
    return value


def _E1C(m):
    return sp.ellipk(m ** 2)


def _E2C(m):
    return sp.ellipe(m ** 2)


def _broberg_R(beta, k):
    return (
        4 * k ** 3 * np.sqrt(1 - beta ** 2) * np.sqrt(k ** 2 - beta ** 2)
        - (2 * k ** 2 - beta ** 2) ** 2
    )


def _broberg_g1(beta, k):
    return (
        ((1 - 4 * k ** 2) * beta ** 2 + 4 * k ** 4) * _E1C(np.sqrt(1 - beta ** 2))
        - ((beta ** 4 - 4 * k ** 2 * (1 + k ** 2) * beta ** 2 + 8 * k ** 4) / beta ** 2)
        * _E2C(np.sqrt(1 - beta ** 2))
        - 4 * k ** 2 * (1 - beta ** 2) * _E1C(np.sqrt(1 - beta ** 2 / k ** 2))
        + (8 * k ** 4 * (1 - beta ** 2) / beta ** 2) * _E2C(np.sqrt(1 - beta ** 2 / k ** 2))
    )


def dynamic_sif_factor(V=None, EE=None, nu=None, rho=None):
    """
    Broberg dynamic correction factor for mode-I SIF.

    This is the BB(V) factor already used by ``calnos`` for the crack-tip
    stress solution:
      sqrt(1 - beta^2) R(beta) / (beta^2 g1(beta))
    """
    nu = float(getattr(st, "nu", 0.3) if nu is None else nu)
    if V is None:
        V = getattr(st, "v", None)
        if V is None:
            V = getattr(st, "vlist", 0.0)
    V = float(_first_if_sequence(V))
    C1, _, _ = wave_speeds(EE=EE, nu=nu, rho=rho)

    k = _broberg_k(nu)
    beta = V / C1
    if beta <= 0.0:
        return 1.0
    if beta >= k:
        return np.nan

    return (np.sqrt(1 - beta ** 2) * _broberg_R(beta, k)) / (beta ** 2 * _broberg_g1(beta, k))


def analytical_sif(step=None, crack_length=None, V=None, sigma_inf=None, hL=None, EE=None, nu=None, rho=None):
    """
    Analytical dynamic mode-I SIF for a straight crack.

    Parameters
    ----------
    step : int, optional
        Crack-growth step. When supplied, ``crack_length = step * hL``.
    crack_length : float, optional
        Crack length in meters. Used directly when ``step`` is not supplied.
    V : float, optional
        Crack velocity.
    sigma_inf : float, optional
        Far-field uniform stress.
    hL : float, optional
        Step length in meters.
    """
    if step is None and crack_length is None:
        raise ValueError("analytical_sif requires either step or crack_length.")

    if crack_length is None:
        hL = float(getattr(st, "hL", 0.05e-3) if hL is None else hL)
        crack_length = int(step) * hL
    crack_length = float(crack_length)
    if crack_length <= 0.0:
        return 0.0

    sigma_inf = float(getattr(st, "SigmaInfinity", getattr(st, "Sigma_app", 1.0)) if sigma_inf is None else sigma_inf)
    return float(sigma_inf * dynamic_sif_factor(V=V, EE=EE, nu=nu, rho=rho) * np.sqrt(np.pi * crack_length))


def analytical_sif_by_steps(steps, V=None, sigma_inf=None, hL=None, EE=None, nu=None, rho=None):
    """Return analytical SIF rows for an iterable of step numbers."""
    hL_eff = float(getattr(st, "hL", 0.05e-3) if hL is None else hL)
    rows = []
    for step in steps:
        step_i = int(step)
        crack_length = step_i * hL_eff
        rows.append(
            {
                "step": step_i,
                "crack_length": float(crack_length),
                "K_I_analytical": analytical_sif(
                    crack_length=crack_length,
                    V=V,
                    sigma_inf=sigma_inf,
                    EE=EE,
                    nu=nu,
                    rho=rho,
                ),
            }
        )
    return rows


def write_analytical_sif_csv(path, steps=None, V=None, sigma_inf=None, hL=None, EE=None, nu=None, rho=None):
    """Write analytical SIF values to CSV."""
    if steps is None:
        steps = range(0, int(getattr(st, "stepall", 0)) + 1)
    rows = analytical_sif_by_steps(steps, V=V, sigma_inf=sigma_inf, hL=hL, EE=EE, nu=nu, rho=rho)
    data = np.asarray(
        [[r["step"], r["crack_length"], r["K_I_analytical"]] for r in rows],
        dtype=float,
    )
    np.savetxt(
        path,
        data,
        delimiter=",",
        header="Step,crack_length_m,K_I_analytical",
        comments="",
    )
    return rows


def calnos():
    """Evaluate crack-tip stress against analytical Broberg solution."""

    # Wave speeds
    C1, C2, VR = wave_speeds()

    def Ks(a_, sigma_):
        return sigma_ * np.sqrt(np.pi * a_)

    def BB(V):
        return dynamic_sif_factor(V)

    # Broberg sigma_yyB
    def sigma_yyB(x, a_, V_):
        k = np.sqrt((1 - 2 * st.nu) / (2 * (1 - st.nu)))
        beta = V_ / C1
        t = a_ / V_
        xi = (C1 * t) / (x + a_)

        kappa1 = np.arcsin(
            np.sqrt(np.maximum(0, 1 - 1 / xi ** 2)) / np.sqrt(np.maximum(0, 1 - beta ** 2))
        )
        kappa2 = np.arcsin(
            np.sqrt(np.maximum(0, k ** 2 - 1 / xi ** 2)) / np.sqrt(np.maximum(0, k ** 2 - beta ** 2))
        )
        q1 = np.sqrt(np.maximum(0, 1 - beta ** 2))
        q2 = np.sqrt(np.maximum(0, 1 - beta ** 2 / k ** 2))

        def E1C(m):
            return sp.ellipk(m ** 2)
        def E2C(m):
            return sp.ellipe(m ** 2)
        def E1(phi, m):
            return sp.ellipkinc(phi, m ** 2)
        def E2(phi, m):
            return sp.ellipeinc(phi, m ** 2)

        def g1(beta_):
            return (
                ((1 - 4 * k ** 2) * beta_ ** 2 + 4 * k ** 4) * E1C(np.sqrt(1 - beta_ ** 2))
                - ((beta_ ** 4 - 4 * k ** 2 * (1 + k ** 2) * beta_ ** 2 + 8 * k ** 4) / beta_ ** 2) * E2C(np.sqrt(1 - beta_ ** 2))
                - 4 * k ** 2 * (1 - beta_ ** 2) * E1C(np.sqrt(1 - beta_ ** 2 / k ** 2))
                + (8 * k ** 4 * (1 - beta_ ** 2) / beta_ ** 2) * E2C(np.sqrt(1 - beta_ ** 2 / k ** 2))
            )

        if xi < 1 / k:
            return -(1 / (beta ** 2 * g1(beta))) * (
                beta ** 2 * (4 * k ** 4 + (1 - 4 * k ** 2) * beta ** 2) * E1(kappa1, q1)
                - (8 * k ** 4 - 4 * k ** 2 * (1 + k ** 2) * beta ** 2 + beta ** 4) * E2(kappa1, q1)
                + (8 * k ** 4 - 4 * k ** 2 * (1 + k ** 2) * beta ** 2 + beta ** 4
                   - 4 * k ** 4 * beta ** 2 * (1 - beta ** 2) * xi ** 2)
                * np.sqrt(np.maximum(0, 1 - 1 / xi ** 2))
                / np.sqrt(np.maximum(0, 1 - beta ** 2 * xi ** 2))
            )
        else:
            return (
                -(1 / (beta ** 2 * g1(beta))) * (
                    beta ** 2 * (4 * k ** 4 + (1 - 4 * k ** 2) * beta ** 2) * E1(kappa1, q1)
                    - (8 * k ** 4 - 4 * k ** 2 * (1 + k ** 2) * beta ** 2 + beta ** 4) * E2(kappa1, q1)
                    + (8 * k ** 4 - 4 * k ** 2 * (1 + k ** 2) * beta ** 2 + beta ** 4
                       - 4 * k ** 4 * beta ** 2 * (1 - beta ** 2) * xi ** 2)
                    * np.sqrt(np.maximum(0, 1 - 1 / xi ** 2))
                    / np.sqrt(np.maximum(0, 1 - beta ** 2 * xi ** 2))
                )
                + (4 * k ** 2 * (1 - beta ** 2) / (beta ** 2 * g1(beta))) * (
                    beta ** 2 * E1(kappa2, q2)
                    - 2 * k ** 2 * E2(kappa2, q2)
                    + k ** 2 * (2 - beta ** 2 * xi ** 2)
                    * np.sqrt(np.maximum(0, 1 - 1 / (k ** 2 * xi ** 2)))
                    / np.sqrt(np.maximum(0, 1 - beta ** 2 * xi ** 2))
                )
            )

    def syy(x_, a_, V_, Sigma_inf_):
        return sigma_yyB(x_, a_, V_) * Sigma_inf_ + Sigma_inf_

    # Broberg table
    V_values = np.arange(10, VR + 1e-9, 10)
    blist = [(float(Vv), float(BB(Vv))) for Vv in V_values]

    # SigmaB
    SigmaB = []
    for stp in range(1, int(st.stepall) + 1):
        row = []
        rstart = (stp + 2) if (stp <= st.aL) else (st.aL + 2)
        # Include r = nLr (1-based upper bound in the original indexing),
        # so the post-tip sampling count becomes lL-1 for step >= aL.
        for r in range(rstart, int(st.nLr) + 1):
            xval = st.hL * ((r - 1) - stp) if (stp <= st.aL) else st.hL * ((r - 1) - st.aL)
            row.append(syy(xval, st.hL * stp, st.v, st.Sigma_app))
        SigmaB.append(row)

    # SigmaL2DAll
    SigmaL2DAll = (-1.0) * np.array(st.rfML2DAllMa2D[1:]) / st.hL

    Sigma_sol = []
    for stp in range(1, int(st.stepall) + 1):
        row = []
        rstart = (stp + 2) if (stp <= st.aL) else (st.aL + 2)
        for r in range(rstart, int(st.nLr) + 1):
            row.append(float(SigmaL2DAll[stp - 1][r - 1][1]))
        Sigma_sol.append(row)

    max_len_sol = max(len(row) for row in Sigma_sol)
    Sigma_sol_padded = [row + [np.nan] * (max_len_sol - len(row)) for row in Sigma_sol]

    Sigma_nos = []
    for srow, brow in zip(Sigma_sol, SigmaB):
        Sigma_nos.append([a / b if b != 0 else np.nan for a, b in zip(srow, brow)])
    print("sigma_nos calculated", Sigma_nos)

    max_len = max(len(row) for row in Sigma_nos)
    Sigma_nos_padded = [row + [np.nan] * (max_len - len(row)) for row in Sigma_nos]

    max_len_B = max(len(row) for row in SigmaB)
    SigmaB_padded = [row + [np.nan] * (max_len_B - len(row)) for row in SigmaB]

    max_len_sol = max(len(row) for row in Sigma_sol)
    Sigma_sol_padded = [row + [np.nan] * (max_len_sol - len(row)) for row in Sigma_sol]

    np.savetxt(f'sigmaB_v{st.vlist}_rGL{st.rGLlist}_py.csv', SigmaB_padded, delimiter=', ')
    np.savetxt(f'sigmasol_v{st.vlist}_rGL{st.rGLlist}_py.csv', Sigma_sol_padded, delimiter=', ')
    np.savetxt(f'sigmanos_v{st.vlist}_rGL{st.rGLlist}_py.csv', Sigma_nos_padded, delimiter=', ')

    KID_rows = write_analytical_sif_csv(
        f'KID_analytical_v{st.vlist}_rGL{st.rGLlist}_py.csv',
        steps=range(0, int(st.stepall) + 1),
        V=st.v,
        sigma_inf=getattr(st, "SigmaInfinity", getattr(st, "Sigma_app", 1.0)),
        hL=st.hL,
    )

    return {'C1': C1, 'C2': C2, 'VR': VR, 'blist': blist, 'KID_analytical': KID_rows}
