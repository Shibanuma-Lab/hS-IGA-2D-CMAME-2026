"""
initial – set up disini / Vini / Aini from the previous step's solution.
"""

import numpy as np
import core.state as st


def initial(step):
    """
    Prepare initial conditions for the current time step.

    * step == 0          → zero everything
    * 1 ≤ step ≤ aL or islocal == 0  → copy previous solution
    * step ≥ aL + 1      → copy + re-initialise local DOFs with a
                            2-DOF shift pattern
    """
    if getattr(st, "analysis_mode", "dynamic") == "static" or int(getattr(st, "isdynamic", 1)) == 0:
        st.disini = np.zeros(st.neq)
        st.Vini   = np.zeros(st.neq)
        st.Aini   = np.zeros(st.neq)

    elif step == 0:
        st.disini = np.zeros(st.neq)
        st.Vini   = np.zeros(st.neq)
        st.Aini   = np.zeros(st.neq)

    elif (1 <= step <= st.aL) or (st.islocal == 0):
        st.disini = st.dis.copy()
        st.Vini   = st.V.copy()
        st.Aini   = st.A.copy()

    elif step >= st.aL + 1:
        st.disini = st.dis.copy()
        st.Vini   = st.V.copy()
        st.Aini   = st.A.copy()

        for i in range(st.neqG, st.neq):
            modv = (i + 1 - st.neqG) % (2 * (st.nLr + 1))

            cond1 = (modv == 0)
            cond2 = (modv == 2 * (st.nLr + 1) - 1)
            cond3 = (modv == (1 if st.ini1x == 1 else 0))
            cond4 = (modv == (2 if st.ini1x == 1 else 0))

            if cond1 or cond2 or cond3 or cond4:
                st.disini[i] = 0.0
                st.Vini[i]   = 0.0
                st.Aini[i]   = 0.0
            else:
                st.disini[i] = st.dis[i + 2]
                st.Vini[i]   = st.V[i + 2]
                st.Aini[i]   = st.A[i + 2]
