"""
Helpers for loading FEM reference data from MATLAB struct files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from scipy.io import loadmat


def _normalize_step_array(arr: np.ndarray, nnode: int, name: str) -> np.ndarray:
    """
    Convert an array to shape (nstep, nnode, 2).

    Supported layouts include:
      (nstep, nnode, 2), (nnode, 2, nstep), (nnode, nstep, 2).
    """
    a = np.asarray(arr, dtype=float)
    if a.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape={a.shape}")

    node_axes = [i for i, s in enumerate(a.shape) if s == nnode]
    if len(node_axes) != 1:
        raise ValueError(f"Cannot identify node axis for {name}, shape={a.shape}, nnode={nnode}")
    node_axis = node_axes[0]

    comp_axes = [i for i, s in enumerate(a.shape) if s == 2]
    if len(comp_axes) == 0:
        raise ValueError(f"Cannot identify component axis (=2) for {name}, shape={a.shape}")
    comp_axis = comp_axes[0] if comp_axes[0] != node_axis else (comp_axes[1] if len(comp_axes) > 1 else -1)
    if comp_axis < 0:
        raise ValueError(f"Cannot identify component axis for {name}, shape={a.shape}")

    step_axes = [i for i in range(3) if i not in (node_axis, comp_axis)]
    if len(step_axes) != 1:
        raise ValueError(f"Cannot identify step axis for {name}, shape={a.shape}")
    step_axis = step_axes[0]

    out = np.moveaxis(a, [step_axis, node_axis, comp_axis], [0, 1, 2])
    if out.shape[2] < 2:
        raise ValueError(f"{name} component dimension < 2 after normalization, shape={out.shape}")
    return np.asarray(out[:, :, :2], dtype=float)


def load_fem_struct_mat(mat_file: Path | str, require_dynamic_fields: bool = True) -> Dict[str, np.ndarray]:
    """
    Load FEM mesh/field data from a MATLAB struct mat file.

    The file must contain key ``Expression1`` with at least:
      - elemFEM
      - nodeFEM
      - disFEMsolutionAll

    If ``require_dynamic_fields`` is True, it must also contain:
      - velFEMsolutionAll
      - acceFEMsolutionAll
    Otherwise missing velocity/acceleration are filled with zeros.
    """
    path = Path(mat_file)
    if not path.exists():
        raise FileNotFoundError(f"FEM mat file not found: {path}")

    raw = loadmat(str(path))
    if "Expression1" not in raw:
        raise KeyError(f"Missing key 'Expression1' in mat file: {path}")

    expr = raw["Expression1"]
    if not isinstance(expr, np.ndarray) or expr.size == 0 or expr.dtype.names is None:
        raise ValueError(f"Invalid 'Expression1' structure in mat file: {path}")

    names = set(expr.dtype.names)
    required = {"elemFEM", "nodeFEM", "disFEMsolutionAll"}
    missing = required - names
    if missing:
        raise KeyError(f"Missing fields in Expression1 of {path}: {sorted(missing)}")

    dynamic_required = {"velFEMsolutionAll", "acceFEMsolutionAll"}
    if require_dynamic_fields:
        missing_dyn = dynamic_required - names
        if missing_dyn:
            raise KeyError(f"Missing dynamic fields in Expression1 of {path}: {sorted(missing_dyn)}")

    rec = expr[0, 0]

    node = np.asarray(rec["nodeFEM"], dtype=float)
    if node.ndim != 2 or node.shape[1] < 2:
        raise ValueError(f"Invalid nodeFEM shape: {node.shape}")
    node = np.asarray(node[:, :2], dtype=float)

    elem = np.asarray(rec["elemFEM"], dtype=int)
    if elem.ndim != 2 or elem.shape[1] < 4:
        raise ValueError(f"Invalid elemFEM shape: {elem.shape}")
    elem = np.asarray(elem[:, :4], dtype=int)
    if int(np.min(elem)) >= 1:
        elem = elem - 1

    nnode = int(node.shape[0])
    dis = _normalize_step_array(rec["disFEMsolutionAll"], nnode, "disFEMsolutionAll")

    if "velFEMsolutionAll" in names:
        vel = _normalize_step_array(rec["velFEMsolutionAll"], nnode, "velFEMsolutionAll")
    else:
        vel = np.zeros_like(dis)

    if "acceFEMsolutionAll" in names:
        acce = _normalize_step_array(rec["acceFEMsolutionAll"], nnode, "acceFEMsolutionAll")
    else:
        acce = np.zeros_like(dis)

    nstep = dis.shape[0]
    if vel.shape[0] != nstep or acce.shape[0] != nstep:
        raise ValueError(
            f"Inconsistent step counts: dis={dis.shape}, vel={vel.shape}, acce={acce.shape}"
        )

    return {
        "node": node,
        "elem": elem,
        "dis": dis,
        "vel": vel,
        "acce": acce,
    }
