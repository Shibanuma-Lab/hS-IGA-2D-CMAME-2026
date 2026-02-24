"""
Load FEM reference data into ``core.state`` from one unified MAT struct file.
"""

import re
from pathlib import Path

import numpy as np

import core.state as st
from utils.fem_struct_mat import load_fem_struct_mat
from utils.logger import logger

# Resolve the FEM_data directory relative to this file's location
_FEM_DIR = Path(__file__).resolve().parent.parent / "FEM_data"


def _first_if_sequence(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            raise ValueError("Expected non-empty list/tuple/array.")
        return value[0]
    return value


def _auto_velocity_value() -> int:
    v = getattr(st, "v", None)
    if v is None:
        v = getattr(st, "vlist", None)
    v = _first_if_sequence(v)
    if v is None:
        raise ValueError("Cannot infer velocity for FEM mat file naming.")
    return int(float(v))


def _auto_crack_length_mm() -> int:
    crack = _first_if_sequence(getattr(st, "c_crack", None))
    if crack is None:
        raise ValueError("Cannot infer crack length for FEM mat file naming.")
    return int(float(crack) * 1000.0)


def build_fem_mat_filename(v=None, crack_length_mm=None, prefix=None) -> str:
    if prefix is None:
        prefix = str(getattr(st, "fem_mat_prefix", "uvaG2DAllFEM2D"))
    v_int = _auto_velocity_value() if v is None else int(float(v))
    a_mm = _auto_crack_length_mm() if crack_length_mm is None else int(crack_length_mm)
    return f"{prefix}_v_{v_int}_a_{a_mm}.mat"


def _best_mat_match_by_velocity(prefix: str, v_int: int, a_mm_target: int):
    pat = f"{prefix}_v_{v_int}_a_*.mat"
    candidates = sorted(_FEM_DIR.glob(pat))
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0].resolve()

    rx = re.compile(rf"^{re.escape(prefix)}_v_{int(v_int)}_a_(\d+)\.mat$")
    scored = []
    for c in candidates:
        m = rx.match(c.name)
        if m is None:
            continue
        aval = int(m.group(1))
        scored.append((abs(aval - int(a_mm_target)), aval, c.resolve()))

    if len(scored) == 0:
        return candidates[0].resolve()
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def resolve_fem_mat_path(mat_file=None) -> Path:
    if mat_file is None:
        mat_file = getattr(st, "fem_mat_file", "auto")

    if mat_file is None:
        mat_file = "auto"
    mat_file_str = str(mat_file).strip()

    if mat_file_str == "" or mat_file_str.lower() == "auto":
        prefix = str(getattr(st, "fem_mat_prefix", "uvaG2DAllFEM2D"))
        v_int = _auto_velocity_value()
        a_mm = _auto_crack_length_mm()

        exact = (_FEM_DIR / build_fem_mat_filename(v=v_int, crack_length_mm=a_mm, prefix=prefix)).resolve()
        if exact.exists():
            return exact

        alt = _best_mat_match_by_velocity(prefix, v_int, a_mm)
        if alt is not None and alt.exists():
            logger.warning(
                "Auto FEM mat exact name not found. Using nearest candidate: %s",
                alt,
            )
            return alt
        return exact

    path = Path(mat_file_str)
    if path.is_absolute():
        return path

    cand_root = (_FEM_DIR.parent / path).resolve()
    if cand_root.exists():
        return cand_root
    cand_cwd = (Path.cwd() / path).resolve()
    if cand_cwd.exists():
        return cand_cwd
    return cand_root


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

    mat_path = resolve_fem_mat_path(mat_file)
    logger.info("Loading FEM data from MAT struct: %s", mat_path)

    fem = load_fem_struct_mat(mat_path, require_dynamic_fields=False)
    st.nodeFEM = fem["node"]
    st.elemFEM = fem["elem"]
    st.disFEMsolutionAll = fem["dis"]
    st.velFEMsolutionAll = fem["vel"]
    st.acceFEMsolutionAll = fem["acce"]
    st.fem_mat_file_resolved = str(mat_path)
    if mat_file is not None:
        st.fem_mat_file = str(mat_file)

    logger.info(
        "FEM data loaded — nodes=%d elems=%d dis=%s vel=%s acce=%s",
        len(st.nodeFEM),
        len(st.elemFEM),
        st.disFEMsolutionAll.shape,
        st.velFEMsolutionAll.shape,
        st.acceFEMsolutionAll.shape,
    )
