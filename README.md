# hS-IGA: 2D straight-crack implementation

This repository contains the Python implementation used for the two-dimensional straight-crack studies associated with the hS-IGA CMAME paper. It couples a B-spline-based global IGA discretisation with a locally refined finite-element discretisation. The code writes displacement and stress fields in VTU format for inspection in ParaView and includes J-integral/DSIF post-processing.

The detailed method, validation, and numerical results are documented in the paper; this README only records how to install and run the released 2D code.

## Branches

- main is the dynamic straight-crack implementation. It advances a prescribed crack path at a prescribed velocity using HHT-alpha time integration.
- static is the static benchmark implementation. Check out that branch to reproduce the static studies:

      git switch static

Use one branch at a time. The configuration files and output conventions differ between the two branches.

## Requirements

The automatic installer supports Ubuntu 22.04/24.04, including Ubuntu under WSL 2. It installs Python 3.10, Pipenv, and the locked Python dependencies. Internet access and sudo permission are required on a new machine. Other platforms may be usable, but are not covered by this release script.

## Installation

On a brand-new Ubuntu or WSL installation, Git is needed to obtain the repository once:

    sudo apt-get update
    sudo apt-get install -y git

Then clone the repository and run the installer. No prior Python, pip, virtual-environment, or Pipenv setup is needed:

    git clone https://github.com/Shibanuma-Lab/hS-IGA-2D-CMAME-2026.git
    cd hS-IGA-2D-CMAME-2026
    ./setup.sh

The script creates the project-local .venv from the locked Python 3.10 environment. It is safe to rerun and installs missing Ubuntu packages automatically. Confirm a completed installation with:

    ./setup.sh --check

If system dependencies are managed by an administrator, use ./setup.sh --skip-system-deps; it stops with a clear error if Python 3.10 or Pipenv is absent.

After setup, run the code directly with:

    pipenv run python main.py

## Running the dynamic code (main)

The entry point is main.py. For a quick smoke test, edit config/parameters.py and set:

    st.stepend = 1
    st.meshonly = 0
    st.calc_jintegral = 0

Then run:

    pipenv run python main.py

This processes dynamic steps 0 and 1 using the supplied HDF5 reference data. A successful run writes results/.../paraview/vtu/step_00001.vtu. For a production run, restore the desired step range and post-processing settings, then adjust the global--local mesh-size ratio (rGL_value), crack velocity (st.vlist), local mesh size (st.hL), and analysis mode (st.isdynamiclist) as needed. Set st.meshonly = 1 only when generating mesh and input files.

Outputs are written under results/; VTU files, when enabled by st.issave, are located in the corresponding paraview/vtu/ directory. The code also writes run diagnostics to logs/.

## FEM reference data

The repository includes compact HDF5 FEM reference fields under FEM_data/ for the supplied dynamic configurations. The default configuration uses the corresponding HDF5 directory automatically; no separate data download is required for the provided examples.

Each HDF5 data set contains the mesh together with displacement, velocity, and acceleration fields. To analyse a custom configuration, place its matching FEM reference data under FEM_data/ and set fem_reference_source, fem_mat_file, or fem_h5_dir in config/parameters.py as appropriate. Keep the data directory and its files together when moving or copying a case.

## Scope and limitations

- The released examples use prescribed crack geometry/path and, for dynamic analyses, prescribed crack velocity. They do not autonomously predict crack initiation, direction, velocity, or arrest.
- This is research software. Users should verify a configuration on a small case before launching a full parameter sweep.
- Results depend on the mesh, quadrature, material parameters, reference FEM data, and post-processing settings. These choices should be recorded when reporting or comparing results.

## License

This code is released under the [MIT License](LICENSE).
