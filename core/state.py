"""
Shared mutable state for the hS-IGA-2D simulation.

Every global variable used by the original monolithic script is stored here
as a module-level attribute.  Other modules do::

    import core.state as st
    st.KG = ...          # write
    value = st.neqG      # read

This replaces hundreds of ``global`` declarations while keeping the exact
same runtime semantics.
"""

import numpy as np

# ===================================================================
# FEM input data (loaded once at startup)
# ===================================================================
nodeFEM = None                # (N, 2)  FEM node coordinates
disFEMsolutionAll = None      # (nt, N, 2)  FEM displacement solution
velFEMsolutionAll = None      # (nt, N, 2)  FEM velocity solution
acceFEMsolutionAll = None     # (nt, N, 2)  FEM acceleration solution
elemFEM = None                # (M, 4)  FEM element connectivity (0-based)
fem_mat_prefix = "uvaG2DAllFEM2D"
fem_mat_file = "auto"
fem_mat_file_resolved = None
fem_reference_source = "auto"  # auto | mat | h5
fem_reference_kind = "mat"     # runtime-resolved source kind
fem_h5_dir_prefix = "h5_export_V_"
fem_h5_dir = "auto"
fem_h5_plane_z = 0.0
fem_h5_plane_tol = None
fem_mat_max_crack_mm = 10
fem_h5_reader = None

# ===================================================================
# Fundamental parameters (set by config/parameters.py)
# ===================================================================
p = 2
q = 2

dmat = 2                      # 1: plane stress, 2: plane strain
SigmaInfinity = 1.0e11
Sigma_app = 1.0e11
c_crack = 10.0e-3
nu = 0.3
EE = 2.06e11
SigmaY0 = 400.0e6
rho = 7800.0
thi = 1.0

nPtsX = 120
nPtsY = 7
auto_global_domain = 1
domain_target_x_auto_from_crack = 1
domain_target_x_from_crack_scale = 1.8
domain_target_x = 18.0e-3
domain_target_y = 6.0e-3
global_domain_fit_mode = "closest"
domain_x_actual = None
domain_y_actual = None
nelemX = None
nelemY = None
hL = 0.05e-3

rGLlist = 8
aLlist = 20
lLlist = 15
HLlist = 15

HHT_alpha = 0.0
HHT_beta = 0.25
HHT_gamma = 0.5

alpha_rayleigh = 0.0
beta_rayleigh = 0.0

ngpG = 3
ngpL = 3
ngpGL = 3

inc = 1
hrefLlist = 1
vlist = 1200.0
ismortar = 0
nofix = 1
abo = 0
ini1x = 2
zentai = 0

islocallist = 1
isdynamiclist = 1

stepini = 0
stepend = -1
stepall = 200
REstart = 0

postprocess = 0
issave = 1
printcheck = 0
meshonly = 0
calc_jintegral = 0
jintegral_Rj0 = 4.0
jintegral_Rj1 = 5.0
jintegral_step_start = -1
jintegral_step_end = -1
jintegral_use_saved_files = 1
jintegral_extend_symmetric = 1
jintegral_scheme = "mathematica"
jintegral_save_extended = 0
jintegral_compare_fem = 0
jintegral_fem_mat_file = "auto"

jobnamelist = ""
jobstart = 1
jobend = 1

islocal = 1
isdynamic = 1

# Derived material constants
mu = None
de = None
kappa = None
dRho = None
gamma = None       # HHT_gamma alias used by solvedynamic

# ===================================================================
# Job / step derived quantities (set by jobset)
# ===================================================================
rGL = None
aL = None
lL = None
HL = None
hrefL = None
v = None
jobname = None
hG = None
nLr = None
Delta_t = None
dt = None
dirname = None
pvd = None
xld = None
dirnamestep = None

# ===================================================================
# Pre-computed Gauss-point arrays (set by jobset / init_gauss)
# ===================================================================
XiEtaG = None;   weightG = None
XiEtaL = None;   weightL = None
XiEtaGL = None;  weightGL = None
nnG = None;   DnnG = None
nnL = None;   DnnL = None
nnGL = None;  DnnGL = None

# lowercase variants (created during initial setup, used by some functions)
xi_etaG = None;  xi_etaL = None;  xi_etaGL = None

# ===================================================================
# Mesh: Global IGA
# ===================================================================
uKnot = None
vKnot = None
weights = None
controlPts = None
noU = None
noV = None
uniqU = None; nelemU = None
uniqV = None; nelemV = None
elRangeU = None; elRangeV = None
elConnU = None; elConnV = None
chan = None
element = None
nelem = None
index = None
nodeVis = None
elemVis = None
nelemVis = None
nodeG = None
elemG = None

# ===================================================================
# Mesh: Local
# ===================================================================
nodeL = None
elemL = None

# ===================================================================
# meshset outputs
# ===================================================================
ndof = 2
nnmG = None; nemG = None; neqG = None; enodeG = None
nnm = None;  nem = None;  neq = None
nnmL = None; nemL = None; neqL = None; enodeL = None
mmGn = None; mmLn = None
minxG = None; maxxG = None; minyG = None; maxyG = None
LGsupabb = None; LGsup = None
emGe = None; nemGe = None; nmGe = None; nnmGe = None
nodeGe = None; emGeR = None; nmGeR = None
neqGe = None; elemGe = None; enodeGe = None
elLemGe = None; nelLemGe = None
emLs = None; nemLs = None; emLm = None; nemLm = None

# ===================================================================
# Matrix assembly
# ===================================================================
nGPs = None; nCtrPts = None
Q = None; W = None
lenu = None; lenv = None
stiff = None; mass = None
KG = None; MG = None
KL = None; ML = None
KGL = None; MGL = None

# makeKGL1 outputs
xmaxL = None; xminL = None; ymaxL = None; yminL = None
npGeelLabb = None; nnpGeelLabb = None

# makeKGL3 outputs
xmaxG_kgl = None; xminG_kgl = None; ymaxG_kgl = None; yminG_kgl = None
npLelGeabb = None; nnpLelGeabb = None; npLXiEtaGabb = None
XiEtaGeG = None

# makeKGL6 outputs
enodeLs = None; phyposs = None; elLemGes = None; XiEtaGeGs = None

# ===================================================================
# Boundary conditions
# ===================================================================
ebc = None
nbc = None
ebc1 = None
ebc2 = None
boundary_coord_tol = None

# ===================================================================
# Solver vectors
# ===================================================================
dis = None
V = None
A = None
force = None
RF = None
RFM = None
freedof = None
disf = None
disini = None
Vini = None
Aini = None
step = None       # current step number

# ===================================================================
# Result / post-processing
# ===================================================================
disG2D = None; velG2D = None; acceG2D = None; rfG2D = None
disVis = None
disL2D = None; velL2D = None; acceL2D = None; rfL2D = None
disG = None; disL = None; velG = None; velL = None
acceG = None; acceL = None; rfG = None; rfL = None
rfMG = None; rfMG2D = None; rfML = None; rfML2D = None
disLG2D = None; velLG2D = None; acceLG2D = None
disLofGIGA = None; velLofGIGA = None; acceLofGIGA = None
stressVis = None; stressL = None

# Visual arrays from buildVisual2D
stress = None; disp = None
sigmaXX = None; sigmaYY = None; sigmaXY = None
sigmaYY_with_ids = None
dispX = None; dispY = None

# ===================================================================
# Alldata accumulation buffers
# ===================================================================
nodeL2DAllMa2D = []
disG2DAllMa2D = []
disVis2DAllMa2D = []
disL2DAllMa2D = []
disGL2DAllMa2D = []
disLG2DAllMa2D = []
velG2DAllMa2D = []
velL2DAllMa2D = []
velLG2DAllMa2D = []
acceG2DAllMa2D = []
acceL2DAllMa2D = []
acceLG2DAllMa2D = []
rfG2DAllMa2D = []
rfL2DAllMa2D = []
rfMG2DAlllMa2D = []
rfML2DAlllMa2D = []
stressVisAllMa2D = []
stressLAllMa2D = []
rfMG2DAllMa2D = []
stressViaAllMa2D = []
rfML2DAllMa2D = []
