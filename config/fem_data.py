"""
Load FEM reference data into ``core.state`` from MAT or H5 sources.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import core.state as st
from utils.fem_struct_h5 import FEMH5Projected2D, FEMStepFieldProxy
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
        raise ValueError("Cannot infer velocity for FEM reference file naming.")
    return int(float(v))


def _auto_crack_length_mm() -> int:
    crack = _first_if_sequence(getattr(st, "c_crack", None))
    if crack is None:
        raise ValueError("Cannot infer crack length for FEM reference file naming.")
    return int(float(crack) * 1000.0)


def build_fem_mat_filename(v=None, crack_length_mm=None, prefix=None) -> str:
    if prefix is None:
        prefix = str(getattr(st, "fem_mat_prefix", "uvaG2DAllFEM2D"))
    v_int = _auto_velocity_value() if v is None else int(float(v))
    a_mm = _auto_crack_length_mm() if crack_length_mm is None else int(crack_length_mm)
    return f"{prefix}_v_{v_int}_a_{a_mm}.mat"


def build_fem_h5_dirname(v=None, crack_length_mm=None, prefix=None) -> str:
    if prefix is None:
        prefix = str(getattr(st, "fem_h5_dir_prefix", "h5_export_V_"))
    v_int = _auto_velocity_value() if v is None else int(float(v))
    a_mm = _auto_crack_length_mm() if crack_length_mm is None else int(crack_length_mm)
    return f"{prefix}{v_int}_a_{a_mm}"


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


def _resolve_user_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()

    cand_root = (_FEM_DIR.parent / path).resolve()
    if cand_root.exists():
        return cand_root
    cand_cwd = (Path.cwd() / path).resolve()
    if cand_cwd.exists():
        return cand_cwd
    return cand_root


def _auto_h5_dir(v_int: int, a_mm: Optional[int] = None) -> Path:
    cfg = str(getattr(st, "fem_h5_dir", "auto")).strip()
    if cfg != "" and cfg.lower() != "auto":
        return _resolve_user_path(cfg)
    return (_FEM_DIR / build_fem_h5_dirname(v=v_int, crack_length_mm=a_mm)).resolve()


def _prefer_h5_for_current_case(a_mm: Optional[int] = None) -> bool:
    if a_mm is None:
        a_mm = _auto_crack_length_mm()

    # Auto rule requested by user:
    #   c_crack <= 10 mm  -> MAT
    #   c_crack >  10 mm  -> H5
    mat_max_mm = int(getattr(st, "fem_mat_max_crack_mm", 10))
    return int(a_mm) > mat_max_mm


def resolve_fem_reference_path(data_file=None, source_preference=None) -> Tuple[str, Path]:
    """
    Resolve FEM reference source path.

    Returns
    -------
    (kind, path)
      kind in {"mat", "h5"}
    """
    if data_file is None:
        data_file = getattr(st, "fem_mat_file", "auto")
    if source_preference is None:
        source_preference = getattr(st, "fem_reference_source", "auto")
    src_pref = str(source_preference).strip().lower()
    if src_pref not in ("auto", "mat", "h5"):
        raise ValueError(f"Unsupported fem reference source preference: {source_preference}")

    data_file_str = "auto" if data_file is None else str(data_file).strip()
    if data_file_str == "" or data_file_str.lower() == "auto":
        prefix = str(getattr(st, "fem_mat_prefix", "uvaG2DAllFEM2D"))
        v_int = _auto_velocity_value()
        a_mm = _auto_crack_length_mm()
        prefer_h5 = _prefer_h5_for_current_case(a_mm=a_mm)

        mat_exact = (_FEM_DIR / build_fem_mat_filename(v=v_int, crack_length_mm=a_mm, prefix=prefix)).resolve()
        h5_dir = _auto_h5_dir(v_int, a_mm=a_mm)

        if src_pref == "h5":
            return "h5", h5_dir
        if src_pref == "mat":
            if mat_exact.exists():
                return "mat", mat_exact
            alt = _best_mat_match_by_velocity(prefix, v_int, a_mm)
            if alt is not None and alt.exists():
                logger.warning("Auto FEM mat exact name not found. Using nearest candidate: %s", alt)
                return "mat", alt
            return "mat", mat_exact

        # auto
        if mat_exact.exists() and not prefer_h5:
            return "mat", mat_exact

        if prefer_h5:
            return "h5", h5_dir

        if mat_exact.exists():
            return "mat", mat_exact

        alt = _best_mat_match_by_velocity(prefix, v_int, a_mm)
        if alt is not None and alt.exists():
            logger.warning("Auto FEM mat exact name not found. Using nearest candidate: %s", alt)
            return "mat", alt

        if h5_dir.exists():
            logger.warning("Auto FEM mat source not found. Falling back to H5 source: %s", h5_dir)
            return "h5", h5_dir

        return "mat", mat_exact

    path = _resolve_user_path(data_file_str)
    if src_pref == "mat":
        return "mat", path
    if src_pref == "h5":
        return "h5", path

    if path.is_dir() or path.suffix.lower() == ".h5":
        return "h5", path
    return "mat", path


def resolve_fem_mat_path(mat_file=None) -> Path:
    """
    Backward-compatible resolver.

    Historically this returned a MAT file path; now it may return an H5
    directory when auto source selection routes to H5.
    """
    _, path = resolve_fem_reference_path(data_file=mat_file, source_preference=None)
    return path


def get_fem_step_field(step: int, field: str = "dis") -> np.ndarray:
    key = str(field).strip().lower()
    if key not in ("dis", "vel", "acce"):
        raise KeyError(f"Unsupported FEM step field: {field}")

    source_kind = str(getattr(st, "fem_reference_kind", "mat")).strip().lower()
    s = int(step)

    if source_kind == "h5":
        reader = getattr(st, "fem_h5_reader", None)
        if reader is None:
            raise RuntimeError("FEM H5 reader not initialized in state.")
        return np.asarray(reader.get_field(s, key), dtype=float)

    attr = {
        "dis": "disFEMsolutionAll",
        "vel": "velFEMsolutionAll",
        "acce": "acceFEMsolutionAll",
    }[key]
    arr = getattr(st, attr, None)
    if arr is None:
        raise RuntimeError(f"FEM field array not loaded: {attr}")
    return np.asarray(arr[s], dtype=float)


def get_fem_step_displacement(step: int) -> np.ndarray:
    return get_fem_step_field(step, "dis")


def load_fem_data(version=None, step_label=None, mat_file=None):
    """
    Load FEM reference data from MAT or H5 source.

    Parameters
    ----------
    version : int, optional
        Legacy argument retained for backward compatibility.
    step_label : int, optional
        Legacy argument retained for backward compatibility.
    mat_file : str or Path, optional
        FEM reference path. If omitted, ``st.fem_mat_file`` is used.
    """
    if version is not None or step_label is not None:
        logger.info(
            "load_fem_data(version, step_label) is deprecated; "
            "using unified FEM source instead."
        )

    source_kind, source_path = resolve_fem_reference_path(mat_file)
    logger.info("Loading FEM reference data (%s): %s", source_kind, source_path)
    st._fem_bilinear_bc_cache = None

    if source_kind == "h5":
        plane_z = getattr(st, "fem_h5_plane_z", 0.0)
        plane_tol = getattr(st, "fem_h5_plane_tol", None)
        reader = FEMH5Projected2D(source_path, plane_z=plane_z, plane_tol=plane_tol)

        stepall = int(getattr(st, "stepall", -1))
        if stepall > reader.max_step:
            raise ValueError(
                f"Requested stepall={stepall} exceeds H5 FEM max step={reader.max_step} in {source_path}"
            )

        st.nodeFEM = reader.node
        st.elemFEM = reader.elem
        st.disFEMsolutionAll = FEMStepFieldProxy(reader, "dis")
        st.velFEMsolutionAll = FEMStepFieldProxy(reader, "vel")
        st.acceFEMsolutionAll = FEMStepFieldProxy(reader, "acce")
        st.fem_h5_reader = reader
        st.fem_reference_kind = "h5"
        st.fem_mat_file_resolved = str(reader.h5_dir)
        if mat_file is not None:
            st.fem_mat_file = str(mat_file)

        logger.info(
            "FEM H5 projected to 2D — nodes=%d elems=%d z=%.6g dis=%s vel=%s acce=%s",
            len(st.nodeFEM),
            len(st.elemFEM),
            reader.selected_plane_z,
            st.disFEMsolutionAll.shape,
            st.velFEMsolutionAll.shape,
            st.acceFEMsolutionAll.shape,
        )
        return

    fem = load_fem_struct_mat(source_path, require_dynamic_fields=False)
    st.nodeFEM = fem["node"]
    st.elemFEM = fem["elem"]
    st.disFEMsolutionAll = fem["dis"]
    st.velFEMsolutionAll = fem["vel"]
    st.acceFEMsolutionAll = fem["acce"]
    st.fem_h5_reader = None
    st.fem_reference_kind = "mat"
    st.fem_mat_file_resolved = str(source_path)
    if mat_file is not None:
        st.fem_mat_file = str(mat_file)

    logger.info(
        "FEM MAT data loaded — nodes=%d elems=%d dis=%s vel=%s acce=%s",
        len(st.nodeFEM),
        len(st.elemFEM),
        st.disFEMsolutionAll.shape,
        st.velFEMsolutionAll.shape,
        st.acceFEMsolutionAll.shape,
    )
