"""
calnos – compute normalised stress / stress intensity factors.

Mathematica: calnos[] (invoked after all steps of a job).
"""

import os

import numpy as np
from scipy import special as sp
from scipy.optimize import brentq

import core.state as st


def calnos():
    """Evaluate crack-tip stress against analytical Broberg solution."""

    # Wave speeds
    C1 = np.sqrt((1 - st.nu) / ((1 + st.nu) * (1 - 2 * st.nu)) * st.EE / st.rho)
    C2 = np.sqrt(1 / (2 * (1 + st.nu)) * st.EE / st.rho)
    VR = 0.577 * np.sqrt(st.EE / st.rho)

    def Ks(a_, sigma_):
        return sigma_ * np.sqrt(np.pi * a_)

    def BB(V):
        k = np.sqrt((1 - 2 * st.nu) / (2 * (1 - st.nu)))
        beta = V / C1

        def R(beta_):
            return (
                4 * k ** 3 * np.sqrt(1 - beta_ ** 2) * np.sqrt(k ** 2 - beta_ ** 2)
                - (2 * k ** 2 - beta_ ** 2) ** 2
            )

        def E1C(m):
            return sp.ellipk(m ** 2)

        def E2C(m):
            return sp.ellipe(m ** 2)

        def g1(beta_):
            return (
                ((1 - 4 * k ** 2) * beta_ ** 2 + 4 * k ** 4) * E1C(np.sqrt(1 - beta_ ** 2))
                - ((beta_ ** 4 - 4 * k ** 2 * (1 + k ** 2) * beta_ ** 2 + 8 * k ** 4) / beta_ ** 2) * E2C(np.sqrt(1 - beta_ ** 2))
                - 4 * k ** 2 * (1 - beta_ ** 2) * E1C(np.sqrt(1 - beta_ ** 2 / k ** 2))
                + (8 * k ** 4 * (1 - beta_ ** 2) / beta_ ** 2) * E2C(np.sqrt(1 - beta_ ** 2 / k ** 2))
            )

        return (np.sqrt(1 - beta ** 2) * R(beta)) / (beta ** 2 * g1(beta))

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
        for r in range(rstart, int(st.nLr)):
            xval = st.hL * ((r - 1) - stp) if (stp <= st.aL) else st.hL * ((r - 1) - st.aL)
            row.append(syy(xval, st.hL * stp, st.v, st.Sigma_app))
        SigmaB.append(row)

    # SigmaL2DAll
    SigmaL2DAll = (-1.0) * np.array(st.rfML2DAllMa2D[1:]) / st.hL

    Sigma_sol = []
    for stp in range(1, int(st.stepall) + 1):
        row = []
        rstart = (stp + 2) if (stp <= st.aL) else (st.aL + 2)
        for r in range(rstart, int(st.nLr)):
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

    return {'C1': C1, 'C2': C2, 'VR': VR, 'blist': blist}
