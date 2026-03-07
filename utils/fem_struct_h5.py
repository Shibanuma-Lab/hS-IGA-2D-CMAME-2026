"""
Helpers for loading FEM reference data from HDF5 exports (3D -> 2D projection).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - import is validated at runtime.
    h5py = None


_STEP_FILE_RX = re.compile(r"^step(\d+)\.h5$")


def _require_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "h5py is required to read H5 FEM reference data. "
            "Install it in the project environment, e.g. `pipenv run pip install h5py`."
        )


def _read_h5_fields(h5_file: Path, expected_fields: Tuple[str, ...]) -> Dict[str, np.ndarray]:
    """
    Read fields from one H5 file.

    Supported layouts:
    - scalar compound dataset ``Dataset1`` with named fields
    - root-level datasets with expected field names
    """
    _require_h5py()
    out: Dict[str, np.ndarray] = {}

    with h5py.File(str(h5_file), "r") as f:
        if "Dataset1" in f:
            raw = f["Dataset1"][()]
            if not isinstance(raw, np.void) or raw.dtype.names is None:
                raise ValueError(f"Invalid compound Dataset1 in {h5_file}")
            names = set(raw.dtype.names)
            missing = [k for k in expected_fields if k not in names]
            if missing:
                raise KeyError(f"Missing fields in {h5_file} Dataset1: {missing}")
            for k in expected_fields:
                out[k] = np.asarray(raw[k])
            return out

        names = set(f.keys())
        missing = [k for k in expected_fields if k not in names]
        if missing:
            raise KeyError(f"Missing datasets in {h5_file}: {missing}")
        for k in expected_fields:
            out[k] = np.asarray(f[k])
    return out


def _normalize_vector_array(arr: np.ndarray, nnode: int, name: str) -> np.ndarray:
    """Normalize one nodal vector field to shape (nnode, 2)."""
    a = np.asarray(arr, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={a.shape}")

    if a.shape[0] == nnode and a.shape[1] >= 2:
        return np.asarray(a[:, :2], dtype=float)
    if a.shape[1] == nnode and a.shape[0] >= 2:
        return np.asarray(a.T[:, :2], dtype=float)
    raise ValueError(f"Cannot normalize {name}, shape={a.shape}, nnode={nnode}")


def _pick_plane(z: np.ndarray, target_z: Optional[float], tol: Optional[float]) -> Tuple[float, np.ndarray]:
    zvals = np.asarray(z, dtype=float).ravel()
    if zvals.ndim != 1:
        raise ValueError("z array must be 1D")

    uniq = np.unique(np.round(zvals, 12))
    if target_z is None:
        z_sel = float(np.min(zvals))
    elif len(uniq) == 0:
        raise ValueError("No z values found in node array.")
    else:
        i = int(np.argmin(np.abs(uniq - float(target_z))))
        z_sel = float(uniq[i])

    if tol is None:
        if len(uniq) >= 2:
            gaps = np.diff(np.sort(uniq))
            min_gap = float(np.min(np.abs(gaps)))
            tol_eff = max(1e-12, min_gap * 1.0e-6)
        else:
            zspan = float(np.max(zvals) - np.min(zvals))
            tol_eff = max(1e-12, zspan * 1.0e-9 + 1e-12)
    else:
        tol_eff = float(tol)
    mask = np.abs(zvals - z_sel) <= tol_eff
    if not np.any(mask):
        raise ValueError(f"No nodes found on selected z-plane z={z_sel} (tol={tol_eff}).")
    return z_sel, mask


class FEMH5Projected2D:
    """
    Lazy H5 FEM reader that projects 3D exported data to one 2D z-plane.
    """

    def __init__(
        self,
        h5_source: Path | str,
        plane_z: Optional[float] = 0.0,
        plane_tol: Optional[float] = None,
    ):
        _require_h5py()

        src = Path(h5_source)
        if src.is_dir():
            self.h5_dir = src.resolve()
        elif src.is_file():
            self.h5_dir = src.parent.resolve()
        else:
            raise FileNotFoundError(f"H5 FEM source not found: {src}")

        self.mesh_file = self.h5_dir / "mesh_only.h5"
        if not self.mesh_file.exists():
            raise FileNotFoundError(f"mesh_only.h5 not found under: {self.h5_dir}")

        self.plane_z = plane_z
        self.plane_tol = plane_tol

        self.node = None
        self.elem = None
        self.nnode3d = 0
        self.nnode2d = 0
        self._plane_node_idx = None
        self._z_selected = None
        self._step_files = None
        self._step_min = None
        self._step_max = None
        self.nstep = 0

        self._cached_step = None
        self._cached_fields = None

        self._load_mesh_projection()
        self._discover_steps()

    # ------------------------------------------------------------------
    def _load_mesh_projection(self) -> None:
        fields = _read_h5_fields(self.mesh_file, ("nodeFEM", "elemFEM"))
        node3d = np.asarray(fields["nodeFEM"], dtype=float)
        elem3d = np.asarray(fields["elemFEM"], dtype=int)

        if node3d.ndim != 2 or node3d.shape[1] < 3:
            raise ValueError(f"Invalid nodeFEM shape in {self.mesh_file}: {node3d.shape}")
        if elem3d.ndim != 2 or elem3d.shape[1] < 4:
            raise ValueError(f"Invalid elemFEM shape in {self.mesh_file}: {elem3d.shape}")

        self.nnode3d = int(node3d.shape[0])

        elem0 = np.asarray(elem3d, dtype=int)
        if int(np.min(elem0)) >= 1:
            elem0 = elem0 - 1
        if int(np.min(elem0)) < 0 or int(np.max(elem0)) >= self.nnode3d:
            raise ValueError("elemFEM indices are out of node index range.")

        z_sel, plane_mask = _pick_plane(node3d[:, 2], self.plane_z, self.plane_tol)
        plane_idx = np.flatnonzero(plane_mask)
        old2new = np.full(self.nnode3d, -1, dtype=int)
        old2new[plane_idx] = np.arange(len(plane_idx), dtype=int)

        elem2d = []
        invalid = 0
        for conn in elem0:
            plane_nodes = []
            seen = set()
            for nid in conn:
                idx = int(nid)
                if idx < 0 or idx >= self.nnode3d:
                    continue
                if not plane_mask[idx]:
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                plane_nodes.append(idx)

            if len(plane_nodes) == 0:
                continue
            if len(plane_nodes) != 4:
                invalid += 1
                continue

            elem2d.append([int(old2new[i]) for i in plane_nodes])

        if len(elem2d) == 0:
            raise ValueError("No 2D projected elements built from H5 mesh.")
        if invalid > 0:
            raise ValueError(
                f"Failed to project {invalid} elements to 2D quads. "
                "Expected one z-plane face (4 nodes) per 3D element."
            )

        self._z_selected = float(z_sel)
        self._plane_node_idx = np.asarray(plane_idx, dtype=int)
        self.node = np.asarray(node3d[self._plane_node_idx, :2], dtype=float)
        self.elem = np.asarray(elem2d, dtype=int)
        self.nnode2d = int(self.node.shape[0])

    # ------------------------------------------------------------------
    def _discover_steps(self) -> None:
        files = {}
        for p in sorted(self.h5_dir.glob("step*.h5")):
            m = _STEP_FILE_RX.match(p.name)
            if m is None:
                continue
            files[int(m.group(1))] = p.resolve()

        if len(files) == 0:
            raise FileNotFoundError(f"No stepXXXXX.h5 files found under: {self.h5_dir}")

        self._step_files = files
        self._step_min = int(min(files.keys()))
        self._step_max = int(max(files.keys()))
        self.nstep = int(self._step_max + 1)

    # ------------------------------------------------------------------
    @property
    def available_steps(self):
        return sorted(self._step_files.keys())

    @property
    def max_step(self) -> int:
        return int(self._step_max)

    @property
    def min_step(self) -> int:
        return int(self._step_min)

    @property
    def selected_plane_z(self) -> float:
        return float(self._z_selected)

    # ------------------------------------------------------------------
    def _step_file(self, step: int) -> Path:
        s = int(step)
        if s not in self._step_files:
            raise FileNotFoundError(f"Step file not found for step={s} under {self.h5_dir}")
        return self._step_files[s]

    def _read_step_fields(self, step: int) -> Dict[str, np.ndarray]:
        h5_file = self._step_file(step)
        fields = _read_h5_fields(
            h5_file,
            ("disFEMsolution", "velFEMsolution", "acceFEMsolution"),
        )
        dis3 = _normalize_vector_array(fields["disFEMsolution"], self.nnode3d, "disFEMsolution")
        vel3 = _normalize_vector_array(fields["velFEMsolution"], self.nnode3d, "velFEMsolution")
        ac3 = _normalize_vector_array(fields["acceFEMsolution"], self.nnode3d, "acceFEMsolution")

        return {
            "dis": np.asarray(dis3[self._plane_node_idx, :2], dtype=float),
            "vel": np.asarray(vel3[self._plane_node_idx, :2], dtype=float),
            "acce": np.asarray(ac3[self._plane_node_idx, :2], dtype=float),
        }

    def load_step(self, step: int) -> Dict[str, np.ndarray]:
        s = int(step)
        if self._cached_step == s and self._cached_fields is not None:
            return self._cached_fields

        fields = self._read_step_fields(s)
        self._cached_step = s
        self._cached_fields = fields
        return fields

    def get_field(self, step: int, field: str) -> np.ndarray:
        key = str(field).strip().lower()
        alias = {
            "dis": "dis",
            "disp": "dis",
            "disfemsolution": "dis",
            "vel": "vel",
            "velocity": "vel",
            "velfemsolution": "vel",
            "acce": "acce",
            "acc": "acce",
            "acceleration": "acce",
            "accefemsolution": "acce",
        }
        if key not in alias:
            raise KeyError(f"Unsupported field name: {field}")
        fields = self.load_step(step)
        return np.asarray(fields[alias[key]], dtype=float)


class FEMStepFieldProxy:
    """
    Lightweight array-like proxy for lazy step access from ``FEMH5Projected2D``.
    """

    def __init__(self, reader: FEMH5Projected2D, field: str):
        self.reader = reader
        self.field = str(field).strip().lower()
        self.shape = (int(reader.nstep), int(reader.nnode2d), 2)
        self.ndim = 3
        self.dtype = np.dtype(float)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            if len(key) == 0:
                raise IndexError("Empty index is not valid.")
            step_sel = key[0]
            rest = key[1:]
        else:
            step_sel = key
            rest = ()

        if isinstance(step_sel, (int, np.integer)):
            arr = self.reader.get_field(int(step_sel), self.field)
            return arr[rest] if len(rest) > 0 else arr

        if isinstance(step_sel, slice):
            steps = range(*step_sel.indices(self.shape[0]))
            steps = list(steps)
            if len(steps) == 0:
                data = np.empty((0, self.shape[1], self.shape[2]), dtype=float)
            else:
                data = np.stack([self.reader.get_field(s, self.field) for s in steps], axis=0)
            return data[(slice(None),) + rest] if len(rest) > 0 else data

        idx = np.asarray(step_sel, dtype=int).ravel()
        if idx.size == 0:
            data = np.empty((0, self.shape[1], self.shape[2]), dtype=float)
        else:
            data = np.stack([self.reader.get_field(int(s), self.field) for s in idx], axis=0)
        return data[(slice(None),) + rest] if len(rest) > 0 else data
