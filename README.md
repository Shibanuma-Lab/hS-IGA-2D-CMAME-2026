# hS-IGA-2D-straight-crack

**2D Straight Crack Dynamic Propagation Simulation** using S-version Isogeometric Analysis (IGA) coupled with Finite Element Method (FEM).  
This is a Python + NumPy/SciPy port of the original Mathematica prototype, refactored into a modular structure.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Running the Program](#running-the-program)
5. [Parameter Configuration](#parameter-configuration)
6. [Analysis Workflow](#analysis-workflow)
7. [Module Descriptions](#module-descriptions)
8. [Logging](#logging)
9. [Output](#output)

---

## Project Structure

```
hS-IGA-2D-straight-crack/
├── main.py                     # Entry point (execution script)
│
├── config/                     # Configuration & data loading
│   ├── parameters.py           #   Simulation parameter definitions
│   └── fem_data.py             #   FEM reference data loading
│
├── core/                       # Core logic
│   ├── state.py                #   Shared state (centralized global variables)
│   ├── alldata.py              #   Reset step accumulation buffers
│   ├── jobset.py               #   Job configuration (directories, Hooke's law, etc.)
│   ├── stepset.py              #   Step configuration (working directory switching)
│   └── calnos.py               #   SIF comparison with Broberg analytical solution
│
├── utils/                      # Utility functions
│   ├── logger.py               #   Logging (logzero / stdlib fallback)
│   ├── shape_functions.py      #   Gauss integration points, Q4 shape functions
│   ├── nurbs.py                #   NURBS basis functions & derivatives
│   ├── mapping.py              #   Coordinate transformation, parametric space mapping
│   └── interpolator.py         #   Bilinear quadrilateral interpolator (for FEM solution)
│
├── mesh/                       # Mesh generation
│   ├── global_mesh.py          #   NURBS global mesh & visualization mesh
│   ├── local_mesh.py           #   Local Q4 mesh generation
│   └── meshset.py              #   Overlapping element identification & bounding box analysis
│
├── matrix/                     # Stiffness & mass matrix assembly
│   ├── assemble.py             #   makematrix dispatcher
│   ├── iga_xi_eta.py           #   IGA parent space Newton's method inverse mapping
│   ├── make_KG.py              #   Global IGA stiffness/mass matrices
│   ├── make_KL.py              #   Local Q4 stiffness/mass matrices
│   └── make_KGL.py             #   Global-local coupling matrices (KGL/MGL)
│
├── boundary/                   # Boundary conditions
│   ├── apply.py                #   boundary dispatcher
│   ├── boundary_fem.py         #   Dirichlet BC for global-only
│   └── boundary_sfem.py        #   Dirichlet BC for S-version FEM coupling
│
├── solver/                     # Solvers
│   ├── solve.py                #   solve dispatcher
│   ├── initial.py              #   Initial condition setup
│   ├── static.py               #   Static linear solver (step 0)
│   └── dynamic.py              #   Newmark-β / HHT-α time integration
│
├── postprocess/                # Post-processing
│   ├── build_visual.py         #   Stress & displacement calculation on visualization mesh
│   ├── getresult.py            #   Extract separated G/L results
│   ├── find_uG_uL.py           #   Compute disLG2D & local stress evaluation
│   └── savedata.py             #   Accumulate step data & export VTU files
│
├── FEM_data/                   # FEM reference data (.gitignore)
│   ├── nodeFEMv400.csv
│   ├── nodeFEMv1200.csv
│   ├── elemFEMv400.dat
│   ├── elemFEMv1200.dat
│   ├── disFEMsolutionAllrGLV400step200.mat
│   └── disFEMsolutionAllrGLV1200step200.mat
│
├── results/                    # Output directory (created at runtime, .gitignore)
│   └── v_400_rGL_2_aL_5_lL_15_HL_4/  # Example result folder (named by parameters)
│       ├── paraview/vtu/       # VTU files for each step
│       ├── excel/              # Excel output (if enabled)
│       └── s_*/                # Step directories
│
├── DSIGA_卒論_rGL.py            # Original file before refactoring (for reference)
└── .gitignore
```

---

## Requirements

- **Python** 3.10 or higher (recommended: 3.12)
- **Package manager**: `pipenv` (for virtual environment management)
- **Required packages**: `numpy`, `scipy`
- **Optional package**: `logzero` (enhanced log formatting; fallback to stdlib logging if not available)

---

## Setup

### Option 1: Using pipenv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd hS-IGA-2D-straight-crack

# Install pipenv (if not already installed)
pip install pipenv

# Install dependencies and create virtual environment
pipenv install

# Install development dependencies (including logzero for enhanced logging)
pipenv install --dev

# Activate the virtual environment
pipenv shell
```

### Option 2: Using venv

```bash
# Clone the repository
git clone <repository-url>
cd hS-IGA-2D-straight-crack

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scipy

# (Optional) Enhanced logging
pip install logzero
```

Ensure that FEM reference data (`.csv` / `.dat` / `.mat` files) are placed in the `FEM_data/` directory.

---

## Running the Program

### Using pipenv

```bash
# Run inside pipenv shell
pipenv shell
python main.py

# Or run directly without activating shell
pipenv run python main.py
```

### Using venv or system Python

```bash
# Basic execution
python main.py
```

The `main.py` entry point calls `load_parameters()` and `load_fem_data()`, then executes `execution()`.  
To change parameters, edit `config/parameters.py` or modify `main.py` directly.

### main.py Entry Point

```python
if __name__ == "__main__":
    # 1) Load parameters (rGL value can be changed here)
    load_parameters(rGL_value=1.0)

    # 2) Load FEM reference data
    load_fem_data(version=st.v, step_label=st.step_label_fem)

    # 3) Run simulation
    execution()
```

---

## Parameter Configuration

All simulation parameters are defined in the `load_parameters()` function in **`config/parameters.py`**.

### Material Parameters

| Parameter | Variable | Default | Description |
|---|---|---|---|
| Young's modulus | `st.EE` | `2.06e11` Pa | Steel equivalent |
| Poisson's ratio | `st.nu` | `0.3` | — |
| Density | `st.rho` | `7800.0` kg/m³ | — |
| Plate thickness | `st.thi` | `1.0` | Plane strain thickness |
| Constitutive model | `st.dmat` | `2` | 1: Plane stress, 2: Plane strain |

### Crack & Loading Parameters

| Parameter | Variable | Default | Description |
|---|---|---|---|
| Far-field stress | `st.SigmaInfinity` | `1.0e11` Pa | Uniform tensile stress at infinity |
| Initial crack length | `st.c_crack` | `10.0e-3` m | — |
| Crack velocity | `st.vlist` | `400.0` m/s | — |

### Mesh Parameters

| Parameter | Variable | Default | Description |
|---|---|---|---|
| IGA control points (X) | `st.nPtsX` | `120` | — |
| IGA control points (Y) | `st.nPtsY` | `7` | — |
| Local element size | `st.hL` | `0.05e-3` m | — |
| Global-Local ratio | `st.rGLlist` | `rGL_value` | `load_parameters()` argument |
| Local mesh height | `st.HLlist` | `4` | Local mesh rows |
| Local mesh width | `st.lLlist` | `15` | Local mesh columns |
| Initial crack step | `st.aLlist` | `5` | — |

### Time Integration Parameters

| Parameter | Variable | Default | Description |
|---|---|---|---|
| HHT α | `st.HHT_alpha` | `0.0` | 0 for standard Newmark method |
| Newmark β | `st.HHT_beta` | `0.25` | Average acceleration method |
| Newmark γ | `st.HHT_gamma` | `0.5` | — |
| Rayleigh α | `st.alpha_rayleigh` | `0.0` | Mass-proportional damping |
| Rayleigh β | `st.beta_rayleigh` | `0.0` | Stiffness-proportional damping |

### Analysis Control

| Parameter | Variable | Default | Description |
|---|---|---|---|
| Analysis method | `st.islocallist` | `1` | 0: Standard IGA, 1: S-version FEM |
| Static/Dynamic | `st.isdynamiclist` | `1` | 0: All static, 1: Step 0 static → dynamic |
| Gauss points (G) | `st.ngpG` | `3` | Global mesh |
| Gauss points (L) | `st.ngpL` | `3` | Local mesh |
| Gauss points (GL) | `st.ngpGL` | `3` | Coupling integration |
| Start step | `st.stepini` | `0` | — |
| End step | `st.stepend` | `200` | — |
| Save data | `st.issave` | `1` | 0: Don't save, 1: Save |
| Mesh only | `st.meshonly` | `0` | 1: Mesh generation only (no solve) |

---

## Analysis Workflow

The `execution()` function in `main.py` runs the following nested loops:

```
for job in [jobstart .. jobend]:
    Alldata()           ← Initialize accumulation buffers
    jobset(job)         ← Job setup (directories, material constants, Gauss tables)

    for step in [stepini .. stepall]:
        stepset(step)       ← Create step directory
        makemesh(step)      ← Generate global/local meshes

        if meshonly != 1:
            meshset()       ← Identify overlapping elements
            makematrix()    ← Assemble KG, KL, KGL
            boundary(step)  ← Apply boundary conditions
            initial(step)   ← Set initial conditions
            solve(step)     ← Static/dynamic solver
            getresult()     ← Separate G/L results
            if islocal == 1:
                finduGuL()  ← Compute coupled displacement & local stress
            if issave == 1:
                savedata(step)  ← Accumulate data & export VTU

    calnos()            ← Compare with analytical solution (SIF evaluation)
```

---

## Module Descriptions

### `core/state.py` — Shared State

Consolidates all global variables from the original script as module-level attributes.  
Each module accesses them via `import core.state as st` and references `st.variable_name`.

### `config/parameters.py` — Parameter Configuration

Calling `load_parameters(rGL_value)` sets all parameters in `core.state`.  
Edit this file to change simulation parameters.

### `config/fem_data.py` — FEM Data Loading

Loads from the `FEM_data/` directory:
- `nodeFEMv{version}.csv` — FEM node coordinates
- `elemFEMv{version}.dat` — FEM element connectivity
- `disFEMsolutionAllrGLV{version}step{step_label}.mat` — FEM displacement solution

### `matrix/make_KGL.py` — Coupling Matrix Assembly

Core of the S-version FEM. Composed of 4 functions:
- `makeKGL()` — Dispatcher
- `makeKGL1()` — Inverse mapping from s-global nodes to local elements
- `makeKGL3()` — Inverse mapping from local nodes to s-global elements
- `makeKGL6()` — KGL / MGL matrix assembly via numerical integration

---

## Logging

`utils/logger.py` manages simulation-wide logging.

```python
from utils.logger import logger

logger.info("Message")
logger.debug("Debug message")
```

- **With logzero**: Outputs to `logs/simulation.log` + console
- **Without logzero**: Console output only via standard `logging` module

---

## Output

### Result Directory Structure

All results are saved in the `results/` directory with parameter-based folder names:

```
results/
  v_400_rGL_2_aL_5_lL_15_HL_4/    # Named by simulation parameters
    paraview/
      vtu/
        step_00000.vtu               # VTU file for step 0
        step_00001.vtu               # VTU file for step 1
        ...
    excel/                          # Excel outputs (if enabled)
    s_*/                            # Step-specific directories
```

### VTU Files

Each calculation step exports a VTU file to `paraview/vtu/` containing:
- **Displacement** (3D vector)
- **Stress components** (Sxx, Syy, Sxy)
- **IsLocal flag** (0: Global, 1: Local)

Open these files in ParaView for visualization.

### Folder Naming Convention

Folders are automatically named based on key parameters:
- `v_<velocity>` — Crack velocity
- `rGL_<ratio>` — Global-local element size ratio
- `aL_<value>` — Initial crack step
- `lL_<value>` — Local mesh width
- `HL_<value>` — Local mesh height

---

## License

See [LICENSE](LICENSE) for details.