"""
getresult – extract global/local displacement/velocity/acceleration/reaction arrays.
"""

import numpy as np
import core.state as st
from postprocess.build_visual import buildVisual2D


def getresult():
    """
    Partition the monolithic solution vectors into G / L parts
    and call ``buildVisual2D`` to populate the visualisation arrays.
    """
    if st.islocal == 0:
        st.disG2D  = np.reshape(st.dis, (-1, 2))
        st.velG2D  = np.reshape(st.V,   (-1, 2))
        st.acceG2D = np.reshape(st.A,   (-1, 2))
        st.rfG2D   = np.reshape(st.RF,  (-1, 2))

        buildVisual2D()
        st.disVis = np.column_stack((st.dispX, st.dispY))

    elif st.islocal == 1:
        dis_2d = np.reshape(st.dis, (-1, 2))
        V_2d   = np.reshape(st.V,   (-1, 2))
        A_2d   = np.reshape(st.A,   (-1, 2))
        RF_2d  = np.reshape(st.RF,  (-1, 2))
        RFM_2d = np.reshape(st.RFM, (-1, 2))

        # G part
        st.disG   = dis_2d[:st.nnmG];   st.disG2D  = st.disG
        st.velG   = V_2d[:st.nnmG];     st.velG2D  = st.velG
        st.acceG  = A_2d[:st.nnmG];     st.acceG2D = st.acceG
        st.rfG    = RF_2d[:st.nnmG];    st.rfG2D   = st.rfG
        st.rfMG   = RFM_2d[:st.nnmG];   st.rfMG2D  = st.rfMG

        # L part
        st.disL   = dis_2d[st.nnmG:st.nnm];   st.disL2D  = st.disL
        st.velL   = V_2d[st.nnmG:st.nnm];     st.velL2D  = st.velL
        st.acceL  = A_2d[st.nnmG:st.nnm];     st.acceL2D = st.acceL
        st.rfL    = RF_2d[st.nnmG:st.nnm];    st.rfL2D   = st.rfL
        st.rfML   = RFM_2d[st.nnmG:st.nnm];   st.rfML2D  = st.rfML
