# hS-IGA: 2D straight-crack implementation

This repository contains the Python implementation used for the two-dimensional straight-crack studies associated with the hS-IGA CMAME paper. It couples a NURBS-based global IGA discretisation with a locally refined finite-element discretisation. The code writes displacement and stress fields in VTU format for inspection in ParaView and includes J-integral/DSIF post-processing.

The 2D and 3D implementations are independent research codes: they use different discretisations, solvers, and input data. Their scientific formulation, validation, and numerical results are described in the paper; this README focuses on installing and running the code.

## Branches

- main is the dynamic straight-crack implementation. It advances a prescribed crack path at a prescribed velocity using HHT-alpha time integration.
- static is the static benchmark implementation. Check out that branch to reproduce the static studies:

      git switch static

Use one branch at a time. The configuration files and output conventions differ between the two branches.

## Requirements

- Python 3.10
- numpy, scipy, h5py, and openpyxl
- logzero is optional and provides enhanced logging

The supplied Pipfile.lock records the Python environment used for this release. Linux, macOS, and Windows are supported in principle; the examples below use a POSIX shell.

## Installation

    git clone https://github.com/Shibanuma-Lab/hS-IGA-2D-straight-crack.git
    cd hS-IGA-2D-straight-crack
    python -m pip install pipenv
    pipenv install --dev

Alternatively, create a virtual environment and install the four required packages with pip.

## Running the dynamic code (main)

The entry point is main.py:

    pipenv run python main.py

Before a production run, edit the load_parameters() call in main.py and the values in config/parameters.py. The usual controls are the global--local mesh-size ratio (rGL_value), crack velocity (st.vlist), local mesh size (st.hL), step range (st.stepini and st.stepend), and analysis mode (st.isdynamiclist). To generate only the mesh and input files, set st.meshonly = 1 in config/parameters.py.

Outputs are written under results/; VTU files, when enabled by st.issave, are located in the corresponding paraview/vtu/ directory. The code also writes run diagnostics to logs/.

## Reference data required by the dynamic example

The dynamic implementation interpolates boundary data from a reference FEM solution. These case-specific .mat or HDF5 files are deliberately not stored in Git because the complete collection is large. A run therefore requires the matching data set in FEM_data/; its selected source is controlled by st.fem_reference_source, st.fem_mat_file, and st.fem_h5_dir in config/parameters.py.

For a public archival release, obtain the accompanying data archive and unpack it into FEM_data/ before running the dynamic example. The archive must be versioned and cited separately from the source repository. Do not assume that the small tracked interpolated_bc_all.csv file replaces the full reference data set.

## Scope and limitations

- The released examples use prescribed crack geometry/path and, for dynamic analyses, prescribed crack velocity. They do not autonomously predict crack initiation, direction, velocity, or arrest.
- This is research software. Users should verify a configuration on a small case before launching a full parameter sweep.
- Results depend on the mesh, quadrature, material parameters, reference FEM data, and post-processing settings. These choices should be recorded when reporting or comparing results.

## License

This code is released under the [MIT License](LICENSE).
