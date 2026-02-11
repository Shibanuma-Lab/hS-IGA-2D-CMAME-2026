"""
Load FEM reference data (node coordinates, element connectivity, displacement
solution) into ``core.state``.
"""

import numpy as np
from scipy.io import loadmat
from pathlib import Path

import core.state as st
from utils.logger import logger

# Resolve the FEM_data directory relative to this file's location
_FEM_DIR = Path(__file__).resolve().parent.parent / "FEM_data"


def load_fem_data(version: int = 1200, step_label: int = 200):
    """
    Load FEM reference data.

    Parameters
    ----------
    version : int
        Mesh density identifier (400 or 1200).
    step_label : int
        Step count embedded in the .mat filename.
    """
    node_path = _FEM_DIR / f"nodeFEMv{version}.csv"
    elem_path = _FEM_DIR / f"elemFEMv{version}.dat"
    mat_path  = _FEM_DIR / f"disFEMsolutionAllrGLV{version}step{step_label}.mat"

    logger.info("Loading FEM node file : %s", node_path)
    st.nodeFEM = np.loadtxt(str(node_path), delimiter=",")

    logger.info("Loading FEM elem file : %s", elem_path)
    st.elemFEM = np.loadtxt(str(elem_path), dtype=int) - 1   # 0-based

    logger.info("Loading FEM solution   : %s", mat_path)
    st.disFEMsolutionAll = loadmat(str(mat_path))["Expression1"][0][0][0]

    logger.info(
        "FEM data loaded — nodes=%d  elems=%d  solution_shape=%s",
        len(st.nodeFEM), len(st.elemFEM), st.disFEMsolutionAll.shape,
    )
