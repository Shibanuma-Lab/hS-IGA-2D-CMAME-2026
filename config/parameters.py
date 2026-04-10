"""
Simulation parameters — loaded into ``core.state`` at startup.

Edit the values here to change the simulation configuration.
"""

import numpy as np
import math
import core.state as st


def load_parameters(rGL_value=2):
    """Populate ``core.state`` with default simulation parameters."""
    st.c_crack = 10.0e-3               # Crack length [m]
    st.hL = 0.05e-3
    st.vlist = 500.0                   # Crack velocity [m/s]

    # Step management
    st.stepini = 0
    # -1 => run all steps, where stepall = round(c_crack / hL)
    st.stepend = -1
    st.stepall = int(round(st.c_crack / st.hL))
    st.REstart = 0

    st.p = 2
    st.q = 2

    st.dmat = 2                        # 1: plane stress, 2: plane strain
    st.SigmaInfinity = 1.0e11          # Far-field uniform stress [Pa]
    st.Sigma_app = st.SigmaInfinity    # Applied load stress [Pa]
    st.nu = 0.3                        # Poisson's ratio
    st.EE = 2.06e11                    # Young's modulus [Pa]
    st.SigmaY0 = 400.0e6               # Yield stress [Pa] (unused)
    st.rho = 7800.0                    # Density [kg/m^3]
    st.thi = 1.0                       # Plate thickness

    # IGA mesh (global nPtsX/nPtsY will be auto-derived in jobset from target size)
    st.nPtsX = 120  # fallback value when auto_global_domain=0
    st.nPtsY = 7    # fallback value when auto_global_domain=0
    st.auto_global_domain = 1
    # Global x-domain follows crack length by default:
    #   Lx = domain_target_x_from_crack_scale * c_crack
    st.domain_target_x_auto_from_crack = 1
    st.domain_target_x_from_crack_scale = 1.8
    st.domain_target_x = st.domain_target_x_from_crack_scale * st.c_crack
    st.domain_target_y = 6.0e-3
    # "closest": choose the nearest mesh to target, even if not fully covering.
    # "cover"  : enforce actual domain > target domain.
    st.global_domain_fit_mode = "closest"

    # Scale / ratio
    st.rGLlist = rGL_value
    st.hG = st.hL * st.rGLlist * math.sqrt(0.97)  # Global mesh element size
    st.HLlist = math.ceil((18/10)*st.rGLlist)      # Ensure HL >= 1.8 * rGL
    st.lLlist = math.ceil((12/10)*st.rGLlist)
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

    if isinstance(st.vlist, (list, tuple, np.ndarray)):
        st.v = int(float(st.vlist[0]))
    else:
        st.v = int(float(st.vlist))    # FEM data version (initial, overridden by jobset)
    st.ismortar = 0
    st.nofix = 1
    st.abo = 0
    st.ini1x = 2                       # Left boundary initial condition: 1=0, 2=interpolated
    st.zentai = 0

    # Interpolation method selection
    # Options:
    #   "delaunay": Linear interpolation with Delaunay triangulation (matches Mathematica)
    #   "bilinear": Bilinear interpolation on quad mesh (uses element connectivity)
    st.interpolator_type = "bilinear"  # Default: Delaunay (Mathematica-compatible)

    # FEM reference input (single source for BC interpolation + FEM J-integral)
    # "auto" mode:
    #   - if c_crack <= fem_mat_max_crack_mm: MAT
    #   - else: H5 directory FEM_data/h5_export_v{int(st.v)}
    st.fem_mat_prefix = "uvaG2DAllFEM2D"
    st.fem_mat_file = "auto"
    st.fem_reference_source = "h5"   # auto | mat | h5
    st.fem_h5_dir_prefix = "h5_export_v"
    st.fem_h5_dir = "FEM_data/h5_export_v500_a_50"
    st.fem_h5_plane_z = 0.0
    st.fem_h5_plane_tol = None
    st.fem_mat_max_crack_mm = 10

    # Optional absolute tolerance override for geometric boundary node selection.
    # None => auto tolerance based on control-point spacing.
    st.boundary_coord_tol = None

    # Debug: Use pre-computed boundary conditions from Mathematica
    # When True, reads interpolated BC values from FEM_data/interpolated_bc_all.csv
    # instead of computing interpolation in Python. Use this to isolate interpolation
    # errors when comparing Python vs Mathematica results.
    # CSV format: step, node_id, disp_x, disp_y
    st.use_precomputed_bc = False  # Set to True to debug interpolation differences

    # Method switches
    st.islocallist = 1                 # 0: Standard FEM, 1: S-version FEM
    st.isdynamiclist = 1               # 0: Static, 1: Static-Dynamic

    # Post-processing / saving
    st.postprocess = 0
    st.issave = 1
    st.printcheck = 0
    st.meshonly = 0
    st.debug_output = 1                # 1: Output detailed mesh/BC/initial info for each step (for debugging)

    # Runtime diagnostics (can be expensive on large systems)
    # Condition number check in solver/dynamic.py:
    #   diag_cond_mode = "proxy" (cheap estimate) or "exact" (np.linalg.cond, very expensive)
    st.diag_cond_enable = 0
    st.diag_cond_start = 95
    st.diag_cond_every = 10
    st.diag_cond_mode = "proxy"
    # KGL nnz/max diagnostics in matrix/make_KGL.py
    st.diag_kgl_enable = 0
    st.diag_kgl_start = 95
    st.diag_kgl_every = 10

    # J-integral / DSIF post-processing
    # 1: calculate automatically at the end of each job in main.py
    st.calc_jintegral = 1
    # Domain parameters (in units of hL)
    st.jintegral_Rj0 = 1.5
    st.jintegral_Rj1 = 1.515
    # Step range control: -1 means auto (start=0, end=stepall)
    st.jintegral_step_start = -1
    st.jintegral_step_end = -1
    # Data source options
    st.jintegral_use_saved_files = 1
    st.jintegral_extend_symmetric = 1
    # Formula/scaling convention:
    #   "mathematica": match current Mathematica debug notebook workflow.
    #   "standard": previous Python implementation.
    st.jintegral_scheme = "mathematica"
    # 1: write mirrored local mesh/fields for each J step (debug)
    st.jintegral_save_extended = 1
    # 1: also calculate FEM reference and hS/FEM normalized comparison in main.py
    st.jintegral_compare_fem = 1
    # FEM reference for J-integral comparison (supports .mat file or H5 directory)
    st.jintegral_fem_mat_file = "auto"

    # Optional suffix appended to result folder name in jobset.
    # Keep empty by default; sweep scripts can override per case.
    st.result_folder_tag = ""

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
