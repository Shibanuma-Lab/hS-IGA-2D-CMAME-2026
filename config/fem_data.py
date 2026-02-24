"""
Load FEM reference data into ``core.state`` from one unified MAT struct file.
"""

from pathlib import Path

import core.state as st
from utils.fem_struct_mat import load_fem_struct_mat
from utils.logger import logger

# Resolve the FEM_data directory relative to this file's location
_FEM_DIR = Path(__file__).resolve().parent.parent / "FEM_data"


def _resolve_mat_path(mat_file) -> Path:
    if mat_file is None:
        mat_file = getattr(st, "fem_mat_file", "FEM_data/uvaG2DAllFEM2D_v_400_a_20.mat")

    path = Path(mat_file)
    if path.is_absolute():
        return path

    cand_root = (_FEM_DIR.parent / path).resolve()
    if cand_root.exists():
        return cand_root
    return (Path.cwd() / path).resolve()


def load_fem_data(version=None, step_label=None, mat_file=None):
    """
    Load FEM reference data.

    Parameters
    ----------
    version : int, optional
        Legacy argument retained for backward compatibility.
    step_label : int, optional
        Legacy argument retained for backward compatibility.
    mat_file : str or Path, optional
        MAT struct path. If omitted, ``st.fem_mat_file`` is used.
    """
    if version is not None or step_label is not None:
        logger.info(
            "load_fem_data(version, step_label) is deprecated; "
            "using unified MAT source instead."
        )

    mat_path = _resolve_mat_path(mat_file)
    logger.info("Loading FEM data from MAT struct: %s", mat_path)

    fem = load_fem_struct_mat(mat_path, require_dynamic_fields=False)
    st.nodeFEM = fem["node"]
    st.elemFEM = fem["elem"]
    st.disFEMsolutionAll = fem["dis"]
    st.velFEMsolutionAll = fem["vel"]
    st.acceFEMsolutionAll = fem["acce"]
    st.fem_mat_file = str(mat_path)

    logger.info(
        "FEM data loaded — nodes=%d elems=%d dis=%s vel=%s acce=%s",
        len(st.nodeFEM),
        len(st.elemFEM),
        st.disFEMsolutionAll.shape,
        st.velFEMsolutionAll.shape,
        st.acceFEMsolutionAll.shape,
    )
