"""
Alldata – reset all accumulation buffers at the start of each job.

Mathematica: Alldata[] := Module[{}, nodeL2DAllMa2D = {}; ...];
"""

import core.state as st


def Alldata():
    """Clear every per-job history buffer."""

    st.nodeL2DAllMa2D = []
    st.disG2DAllMa2D = []
    st.disVis2DAllMa2D = []
    st.disL2DAllMa2D = []
    st.disGL2DAllMa2D = []
    st.disLG2DAllMa2D = []
    st.velG2DAllMa2D = []
    st.velL2DAllMa2D = []
    st.velLG2DAllMa2D = []
    st.acceG2DAllMa2D = []
    st.acceL2DAllMa2D = []
    st.acceLG2DAllMa2D = []
    st.rfG2DAllMa2D = []
    st.rfL2DAllMa2D = []
    st.rfMG2DAlllMa2D = []
    st.rfML2DAlllMa2D = []
    st.stressVisAllMa2D = []
    st.stressLAllMa2D = []
    st.rfMG2DAllMa2D = []
    st.stressViaAllMa2D = []
    st.rfML2DAllMa2D = []
