# hS-IGA: 2D straight-crack implementation

This repository contains the Python implementation used for the two-dimensional straight-crack studies associated with the hS-IGA CMAME paper. It couples a NURBS-based global IGA discretisation with a locally refined finite-element discretisation. The code writes displacement and stress fields in VTU format for inspection in ParaView and includes J-integral/DSIF post-processing.

The detailed method, validation, and numerical results are documented in the paper; this README only records how to install and run the released 2D code.

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

## FEM reference data

The repository includes compact HDF5 FEM reference fields under FEM_data/ for the supplied dynamic configurations. The default configuration uses the corresponding HDF5 directory automatically; no separate data download is required for the provided examples.

Each HDF5 data set contains the mesh together with displacement, velocity, and acceleration fields. To analyse a custom configuration, place its matching FEM reference data under FEM_data/ and set fem_reference_source, fem_mat_file, or fem_h5_dir in config/parameters.py as appropriate. Keep the data directory and its files together when moving or copying a case.

## Scope and limitations

- The released examples use prescribed crack geometry/path and, for dynamic analyses, prescribed crack velocity. They do not autonomously predict crack initiation, direction, velocity, or arrest.
- This is research software. Users should verify a configuration on a small case before launching a full parameter sweep.
- Results depend on the mesh, quadrature, material parameters, reference FEM data, and post-processing settings. These choices should be recorded when reporting or comparing results.

## License

This code is released under the [MIT License](LICENSE).
