"""
Analytical Dirichlet boundary conditions for the static crack benchmark.
"""

from __future__ import annotations

import numpy as np

import core.state as st
from boundary.node_selector import find_nodes_on_extreme
from utils.static_crack import exact_mode_i_displacement


def boundary_static(step):
    """Build static benchmark boundary conditions from the analytical solution."""
    del step

    coord_tol = getattr(st, "boundary_coord_tol", None)
    nodeG = np.asarray(st.nodeG, dtype=float)
    nodeL = np.asarray(st.nodeL, dtype=float)

    left_nodes = find_nodes_on_extreme(nodeG, axis=0, side="min", atol=coord_tol)
    right_nodes = find_nodes_on_extreme(nodeG, axis=0, side="max", atol=coord_tol)
    top_nodes = find_nodes_on_extreme(nodeG, axis=1, side="max", atol=coord_tol)
    down_nodes = find_nodes_on_extreme(nodeG, axis=1, side="min", atol=coord_tol)

    bcGEX = np.unique(np.concatenate([left_nodes, right_nodes, top_nodes])).astype(int)

    jj = 0
    for nid in down_nodes:
        jj += 1
        if nodeG[int(nid), 0] >= float(st.static_crack_tip_x):
            break
    cut_count = max(jj - int(st.p), 0)
    cut_nodes = set(int(nid) for nid in down_nodes[:cut_count])
    yfixG = sorted(set(int(nid) for nid in down_nodes) - cut_nodes - set(int(nid) for nid in bcGEX))

    nnmG = int(st.nnmG)
    a = float(st.static_crack_tip_x)
    width = float(st.static_width)

    mask_local_bottom = (
        np.isclose(nodeL[:, 1], 0.0)
        & (nodeL[:, 0] >= a)
        & (nodeL[:, 0] < width)
    )
    bcLFixY = (np.where(mask_local_bottom)[0] + nnmG).astype(int)

    nLx = int(st.nLr)
    HL = int(st.HL)
    bcLFixXY = np.unique(
        np.concatenate(
        [
            nnmG + (nLx + 1) * (np.arange(1, HL + 1) - 1),
            nnmG + (nLx + 1) * np.arange(1, HL + 1) - 1,
            nnmG + (nLx + 1) * HL + np.arange(0, nLx + 1),
        ]
        )
    ).astype(int)

    ebc1 = [
        [int(nid), 1, exact_mode_i_displacement(nodeG[int(nid)], st.static_crack_tip_x, st.mu, st.kappa)[0]]
        for nid in bcGEX
    ]
    ebc2 = [
        [int(nid), 2, exact_mode_i_displacement(nodeG[int(nid)], st.static_crack_tip_x, st.mu, st.kappa)[1]]
        for nid in bcGEX
    ]
    ebc3 = [[int(nid), 2, 0.0] for nid in yfixG]
    ebc4 = [[int(nid), 2, 0.0] for nid in bcLFixY]
    ebc5 = [[int(nid), 1, 0.0] for nid in bcLFixXY]
    ebc6 = [[int(nid), 2, 0.0] for nid in bcLFixXY]

    st.ebc = np.array(ebc1 + ebc2 + ebc3 + ebc4 + ebc5 + ebc6, dtype=float)
    st.nbc = np.array([[0.0, 1.0, 0.0]], dtype=float)
