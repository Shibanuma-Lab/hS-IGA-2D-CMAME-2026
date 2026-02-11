"""
Simulation parameters — loaded into ``core.state`` at startup.

Edit the values here to change the simulation configuration.
"""

import numpy as np
import math
import core.state as st


def load_parameters(rGL_value=2):
    """Populate ``core.state`` with default simulation parameters."""

    st.p = 2
    st.q = 2

    st.dmat = 2                        # 1: plane stress, 2: plane strain
    st.SigmaInfinity = 1.0e11          # Far-field uniform stress [Pa]
    st.Sigma_app = st.SigmaInfinity    # Applied load stress [Pa]
    st.c_crack = 10.0e-3               # Crack length [m]
    st.nu = 0.3                        # Poisson's ratio
    st.EE = 2.06e11                    # Young's modulus [Pa]
    st.SigmaY0 = 400.0e6               # Yield stress [Pa] (unused)
    st.rho = 7800.0                    # Density [kg/m^3]
    st.thi = 1.0                       # Plate thickness

    # IGA mesh
    st.nPtsX = 120
    st.nPtsY = 7
    st.hL = 0.05e-3

    # Scale / ratio
    st.rGLlist = rGL_value
    st.HLlist = math.ceil((18/10)*st.rGLlist)      # Ensure HL >= 1.8 * rGL
    st.lLlist = 15
    st.aLlist = math.ceil((25/10)*st.rGLlist)      # Ensure aL >= 2.5 * rGL

    # HHT-alpha method (alpha=-0.02 for numerical damping)
    st.HHT_alpha = -0.02
    st.HHT_beta = (1.0 - st.HHT_alpha) ** 2 / 4.0     # β = (1-α)²/4
    st.HHT_gamma = 0.5 - st.HHT_alpha                 # γ = 0.5 - α

    # Rayleigh damping
    st.alpha_rayleigh = 0.0
    st.beta_rayleigh = 0.0

    # Gauss integration
    st.ngpG = 3
    st.ngpL = 3
    st.ngpGL = 3

    # Analysis control
    st.inc = 1
    st.hrefLlist = 1
    st.vlist = 400.0                   # Crack velocity [m/s]
    st.v = int(st.vlist)               # FEM data version (initial, overridden by jobset)
    st.ismortar = 0
    st.nofix = 1
    st.abo = 0
    st.ini1x = 2                       # Left boundary initial condition: 1=0, 2=interpolated
    st.zentai = 0

    # Method switches
    st.islocallist = 1                 # 0: Standard FEM, 1: S-version FEM
    st.isdynamiclist = 1               # 0: Static, 1: Static-Dynamic

    # Step management
    st.stepini = 0
    st.stepend = 200
    st.stepall = st.stepend
    st.step_label_fem = 200            # FEM reference data step label (for filename)
    st.REstart = 0

    # Post-processing / saving
    st.postprocess = 0
    st.issave = 1
    st.printcheck = 0
    st.meshonly = 0

    # Job management
    st.jobnamelist = f"Default_v_{st.vlist}_rGL{st.rGLlist}_"
    st.jobstart = 1
    st.jobend = 1

    # Derived flags
    st.islocal = int(st.islocallist)
    st.isdynamic = int(st.isdynamiclist)

    # Derived material constants
    st.mu = st.EE / (2.0 * (1.0 + st.nu))
    st.de = (st.EE / ((1.0 + st.nu) * (1.0 - 2.0 * st.nu))) * np.array([
        [1.0 - st.nu, st.nu,        0.0],
        [st.nu,       1.0 - st.nu,  0.0],
        [0.0,         0.0,          0.5 - st.nu],
    ])
    st.kappa = 3.0 - 4.0 * st.nu
