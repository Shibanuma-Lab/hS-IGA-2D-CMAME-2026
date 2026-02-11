"""
makemesh – generate / update global and local meshes per step.

Mathematica: makemesh[step_] := Module[{}, ...]
"""

import numpy as np

import core.state as st
from mesh.global_mesh import makeGlobalMesh


def makemesh(step):
    """Build or refresh global IGA + local Q4 meshes for *step*."""

    np.set_printoptions(precision=15, suppress=True)

    # --- global mesh ---
    if (step == 0) or (st.REstart == 1 and step == st.stepini):
        makeGlobalMesh(st.hG, st.nPtsX, st.nPtsY)
        st.nodeG = st.controlPts
        st.elemG = [conn[:] for conn in st.element]

    # --- local mesh (S-version) ---
    if int(st.islocal) == 1:
        moveL = 0 if (step <= st.aL) else (step - st.aL)

        yy = np.arange(0, int(st.HL) + 1, dtype=float)
        xx = np.arange(0, int(st.nLr) + 1, dtype=float)
        st.nodeL = np.array(
            [[(x + moveL) * st.hL, y * st.hL] for y in yy for x in xx],
            dtype=float,
        )

        nx = int(st.nLr)
        ny = int(st.HL)
        stride = nx + 1
        bases = [j + i * stride for i in range(ny) for j in range(nx)]
        e1 = np.array(bases, dtype=int)
        e2 = e1 + 1
        e3 = e1 + stride + 1
        e4 = e1 + stride
        st.elemL = np.vstack([e1, e2, e3, e4]).T  # (ny*nx, 4), 0-based
