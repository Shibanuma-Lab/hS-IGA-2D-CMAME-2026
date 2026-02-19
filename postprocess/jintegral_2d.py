"""
J-integral and dynamic SIF (DSIF) post-processing for 2D straight cracks.

This module supports two data sources:
1) saved step files under ``results/.../s_*/`` (preferred)
2) in-memory ``core.state`` arrays (fallback / quick tests)
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import core.state as st
from utils.shape_functions import GP, GW, shp, Dshp, enlarge


@dataclass
class StepData:
    node: np.ndarray
    elem: np.ndarray
    dis: np.ndarray
    vel: np.ndarray
    acce: np.ndarray


class JIntegral2D:
    """Domain-integral J and DSIF calculator for 2D local mesh results.

    For 2D, the J-domain weight function uses radial band parameters only
    (``Rj0``, ``Rj1``).
    """

    def __init__(
        self,
        step_start: int = 1,
        step_end: Optional[int] = None,
        Rj0: float = 1.5,
        Rj1: float = 1.515,
        result_dir: Optional[Path] = None,
        use_saved_files: bool = True,
        extend_symmetric: bool = True,
    ):
        self.step_start = int(step_start)
        self.step_end = int(step_end) if step_end is not None else None

        self.Rj0 = float(Rj0)
        self.Rj1 = float(Rj1)

        self.use_saved_files = bool(use_saved_files)
        self.extend_symmetric = bool(extend_symmetric)
        self.save_extended_debug = bool(int(getattr(st, "jintegral_save_extended", 1)))

        def _f(val, default):
            return float(default if val is None else val)

        def _i(val, default):
            return int(default if val is None else val)

        self.EE = _f(getattr(st, "EE", None), 2.06e11)
        self.nu = _f(getattr(st, "nu", None), 0.3)
        self.rho = _f(getattr(st, "rho", None), 7800.0)

        v_now = getattr(st, "v", None)
        if v_now is None:
            v_now = getattr(st, "vlist", 0.0)
        self.v = _f(v_now, 0.0)

        self.hL = _f(getattr(st, "hL", None), 0.05e-3)

        aL_now = getattr(st, "aL", None)
        if aL_now is None:
            aL_now = getattr(st, "aLlist", 5)
        self.aL = _i(aL_now, 5)

        self.dmat = _i(getattr(st, "dmat", None), 2)

        de_state = getattr(st, "de", None)
        if de_state is None:
            if self.dmat == 1:
                # plane stress
                self.de = (self.EE / (1.0 - self.nu ** 2)) * np.array(
                    [[1.0, self.nu, 0.0],
                     [self.nu, 1.0, 0.0],
                     [0.0, 0.0, 0.5 - 0.5 * self.nu]],
                    dtype=float,
                )
            else:
                # plane strain
                self.de = (self.EE / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))) * np.array(
                    [[1.0 - self.nu, self.nu, 0.0],
                     [self.nu, 1.0 - self.nu, 0.0],
                     [0.0, 0.0, 0.5 - self.nu]],
                    dtype=float,
                )
        else:
            self.de = np.asarray(de_state, dtype=float)

        self.result_dir = Path(result_dir) if result_dir is not None else self._default_result_dir()
        self._step_dir_cache: Optional[Dict[int, Path]] = None

        gp = GP(2)
        gw = GW(2)
        self.gauss_pts = [(float(xi), float(eta)) for eta in gp for xi in gp]
        self.gauss_w = [float(wx * wy) for wy in gw for wx in gw]
        self.nn = [shp(pt).ravel() for pt in self.gauss_pts]
        self.dnn = [Dshp(pt) for pt in self.gauss_pts]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _default_result_dir(self) -> Optional[Path]:
        dirname = getattr(st, "dirname", None)
        if dirname is None:
            return None
        return Path(dirname)

    def _step_dirs(self) -> Dict[int, Path]:
        if self._step_dir_cache is not None:
            return self._step_dir_cache

        out: Dict[int, Path] = {}
        if self.result_dir is None or not self.result_dir.exists():
            self._step_dir_cache = out
            return out

        pat = re.compile(r"_(\d+)$")
        for d in self.result_dir.iterdir():
            if not d.is_dir():
                continue
            m = pat.search(d.name)
            if m is None:
                continue
            out[int(m.group(1))] = d

        self._step_dir_cache = out
        return out

    def _step_dir(self, step: int) -> Optional[Path]:
        return self._step_dirs().get(int(step))

    def _debug_step_dir(self, step: int) -> Path:
        """
        Resolve output directory for J-integral debug data of one step.
        Prefer existing saved step folder ``results/.../s_<step>``.
        """
        d = self._step_dir(step)
        if d is not None:
            return d
        if self.result_dir is None:
            out = Path.cwd() / "jintegral_debug" / f"s_{int(step)}"
        else:
            out = self.result_dir / "jintegral_debug" / f"s_{int(step)}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _write_node_dat(path: Path, nodes: np.ndarray, title: str) -> None:
        arr = np.asarray(nodes, dtype=float)
        with open(path, "w") as f:
            f.write(f"# {title}\n")
            f.write("# NodeID x y\n")
            for i, row in enumerate(arr, start=1):
                f.write(f"{i} {row[0]:.12e} {row[1]:.12e}\n")

    @staticmethod
    def _write_elem_dat(path: Path, elems: np.ndarray, title: str) -> None:
        arr = np.asarray(elems, dtype=int)
        with open(path, "w") as f:
            f.write(f"# {title}\n")
            f.write("# ElemID n1 n2 n3 n4\n")
            for i, row in enumerate(arr, start=1):
                n1, n2, n3, n4 = row + 1
                f.write(f"{i} {n1} {n2} {n3} {n4}\n")

    @staticmethod
    def _write_vec_dat(path: Path, vec: np.ndarray, title: str) -> None:
        arr = np.asarray(vec, dtype=float)
        with open(path, "w") as f:
            f.write(f"# {title}\n")
            f.write("# NodeID value_x value_y\n")
            for i, row in enumerate(arr, start=1):
                f.write(f"{i} {row[0]:.12e} {row[1]:.12e}\n")

    def _write_extended_debug_dat(self, step: int, data: StepData) -> None:
        """Write mirrored local mesh + fields for J-integral debugging."""
        out_dir = self._debug_step_dir(step)
        self._write_node_dat(
            out_dir / "node_mirror.l.dat",
            data.node,
            "Mirrored local nodes used in J-integral",
        )
        self._write_elem_dat(
            out_dir / "elem_mirror.l.dat",
            data.elem,
            "Mirrored local elements used in J-integral (Q4, 1-based)",
        )
        self._write_vec_dat(
            out_dir / "u_mirror_gl.l.dat",
            data.dis,
            "Mirrored local displacement (G+L) used in J-integral",
        )
        self._write_vec_dat(
            out_dir / "v_mirror_gl.l.dat",
            data.vel,
            "Mirrored local velocity (G+L) used in J-integral",
        )
        self._write_vec_dat(
            out_dir / "a_mirror_gl.l.dat",
            data.acce,
            "Mirrored local acceleration (G+L) used in J-integral",
        )

    @staticmethod
    def _has_any(path_a: Path, path_b: Path) -> bool:
        return path_a.exists() or path_b.exists()

    def _step_dir_has_solution(self, step_dir: Path) -> bool:
        if not (step_dir / "node.l.dat").exists():
            return False
        if not (step_dir / "elem.l.dat").exists():
            return False
        if not self._has_any(step_dir / "u_gl.l.dat", step_dir / "u.l.dat"):
            return False
        if not self._has_any(step_dir / "v_gl.l.dat", step_dir / "v.l.dat"):
            return False
        if not self._has_any(step_dir / "a_gl.l.dat", step_dir / "a.l.dat"):
            return False
        return True

    @staticmethod
    def _load_table(path: Path, dtype=float) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        arr = np.loadtxt(path, comments="#", dtype=dtype)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _load_nodes(self, path: Path) -> np.ndarray:
        arr = self._load_table(path, dtype=float)
        if arr.shape[1] < 3:
            raise ValueError(f"Invalid node file format: {path}")
        return np.asarray(arr[:, 1:3], dtype=float)

    def _load_elem(self, path: Path) -> np.ndarray:
        arr = self._load_table(path, dtype=int)
        if arr.shape[1] >= 5:
            return np.asarray(arr[:, 1:5] - 1, dtype=int)
        if arr.shape[1] == 4:
            return np.asarray(arr[:, 0:4] - 1, dtype=int)
        raise ValueError(f"Invalid element file format: {path}")

    def _load_vec(self, path: Path) -> np.ndarray:
        arr = self._load_table(path, dtype=float)
        if arr.shape[1] < 3:
            raise ValueError(f"Invalid vector file format: {path}")
        return np.asarray(arr[:, 1:3], dtype=float)

    def get_data_from_files(self, step: int) -> StepData:
        step_dir = self._step_dir(step)
        if step_dir is None:
            raise FileNotFoundError(
                f"Step {step} directory not found under {self.result_dir}"
            )

        node = self._load_nodes(step_dir / "node.l.dat")
        elem = self._load_elem(step_dir / "elem.l.dat")

        dis_file = step_dir / "u_gl.l.dat"
        if not dis_file.exists():
            dis_file = step_dir / "u.l.dat"
        vel_file = step_dir / "v_gl.l.dat"
        if not vel_file.exists():
            vel_file = step_dir / "v.l.dat"
        acce_file = step_dir / "a_gl.l.dat"
        if not acce_file.exists():
            acce_file = step_dir / "a.l.dat"

        dis = self._load_vec(dis_file)
        vel = self._load_vec(vel_file)
        acce = self._load_vec(acce_file)

        return StepData(node=node, elem=elem, dis=dis, vel=vel, acce=acce)

    def get_data_from_state(self, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward-compatible state-based loader.
        Returns ``nodeL, elemL, disLG, velLG, acceLG``.
        """
        step0 = int(getattr(st, "stepini", 0))
        idx = int(step) - step0
        if idx < 0:
            raise IndexError(f"Step {step} is before stepini={step0}")

        node = np.asarray(st.nodeL2DAllMa2D[idx], dtype=float)
        elem = np.asarray(st.elemL, dtype=int)
        dis = np.asarray(st.disLG2DAllMa2D[idx], dtype=float)

        if len(st.velLG2DAllMa2D) > idx:
            vel = np.asarray(st.velLG2DAllMa2D[idx], dtype=float)
        elif len(st.velL2DAllMa2D) > idx:
            vel = np.asarray(st.velL2DAllMa2D[idx], dtype=float)
        else:
            vel = np.zeros_like(dis)

        if len(st.acceLG2DAllMa2D) > idx:
            acce = np.asarray(st.acceLG2DAllMa2D[idx], dtype=float)
        elif len(st.acceL2DAllMa2D) > idx:
            acce = np.asarray(st.acceL2DAllMa2D[idx], dtype=float)
        else:
            acce = np.zeros_like(dis)

        return node, elem, dis, vel, acce

    def _has_state_data(self, step: int) -> bool:
        step0 = int(getattr(st, "stepini", 0))
        idx = int(step) - step0
        return 0 <= idx < len(st.nodeL2DAllMa2D) and 0 <= idx < len(st.disLG2DAllMa2D)

    def _get_step_data(self, step: int) -> StepData:
        if self.use_saved_files and self._step_dir(step) is not None:
            try:
                return self.get_data_from_files(step)
            except (FileNotFoundError, ValueError):
                if not self._has_state_data(step):
                    raise

        if self._has_state_data(step):
            node, elem, dis, vel, acce = self.get_data_from_state(step)
            return StepData(node=node, elem=elem, dis=dis, vel=vel, acce=acce)

        raise RuntimeError(
            f"No available data for step {step}. "
            f"Checked saved files in {self.result_dir} and state buffers."
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def crack_tip_x(self, step: int) -> float:
        # In this 2D moving-window local mesh, the physical crack-tip position
        # advances with time step.
        return int(step) * self.hL

    def compute_weight_function(self, nodeL: np.ndarray, crack_tip_x: float) -> np.ndarray:
        """Compute nodal weight function ``q`` for 2D radial domain integral."""
        node = np.asarray(nodeL, dtype=float)
        x = node[:, 0] - float(crack_tip_x)
        y = node[:, 1]
        r = np.sqrt(x * x + y * y)
        qR = np.zeros_like(r)

        r0 = self.Rj0 * self.hL
        r1 = self.Rj1 * self.hL
        if r1 <= r0:
            qR[r <= r0] = 1.0
        else:
            in0 = r <= r0
            tr = (r > r0) & (r < r1)
            qR[in0] = 1.0
            qR[tr] = (r1 - r[tr]) / (r1 - r0)
        return qR

    def _force_q_near_tip(self, node: np.ndarray, elem: np.ndarray, q: np.ndarray, crack_tip_x: float) -> np.ndarray:
        """Set q=1 for nodes in elements touching the crack-tip node."""
        dist2 = (node[:, 0] - crack_tip_x) ** 2 + node[:, 1] ** 2
        tip_idx = int(np.argmin(dist2))
        eids = np.where(np.any(elem == tip_idx, axis=1))[0]
        if len(eids) == 0:
            return q
        q2 = q.copy()
        q2[np.unique(elem[eids].ravel())] = 1.0
        return q2

    def extend_symmetric_mesh(
        self,
        node: np.ndarray,
        elem: np.ndarray,
        dis: np.ndarray,
        vel: np.ndarray,
        acce: np.ndarray,
    ) -> StepData:
        """
        Mirror local mesh across y=0 for closed-loop J-domain integration.

        Symmetry used for Mode-I type fields:
        - x-components: even
        - y-components: odd
        """
        tol = max(1e-14, 1e-8 * self.hL)

        node_ext = node.tolist()
        dis_ext = dis.tolist()
        vel_ext = vel.tolist()
        acce_ext = acce.tolist()

        mirror_idx: Dict[int, int] = {}
        for i, (x, y) in enumerate(node):
            if abs(y) <= tol:
                mirror_idx[i] = i
                continue
            mirror_idx[i] = len(node_ext)
            node_ext.append([x, -y])
            dis_ext.append([dis[i, 0], -dis[i, 1]])
            vel_ext.append([vel[i, 0], -vel[i, 1]])
            acce_ext.append([acce[i, 0], -acce[i, 1]])

        elem_mirror = []
        for conn in elem:
            m = [mirror_idx[int(conn[0])], mirror_idx[int(conn[1])], mirror_idx[int(conn[2])], mirror_idx[int(conn[3])]]
            # Keep positive Jacobian for mirrored elements
            elem_mirror.append([m[3], m[2], m[1], m[0]])

        node_arr = np.asarray(node_ext, dtype=float)
        dis_arr = np.asarray(dis_ext, dtype=float)
        vel_arr = np.asarray(vel_ext, dtype=float)
        acce_arr = np.asarray(acce_ext, dtype=float)
        elem_arr = np.vstack([elem, np.asarray(elem_mirror, dtype=int)])

        return StepData(node=node_arr, elem=elem_arr, dis=dis_arr, vel=vel_arr, acce=acce_arr)

    # ------------------------------------------------------------------
    # J-integral / DSIF
    # ------------------------------------------------------------------
    def _dynamic_factor_AI(self) -> float:
        c1 = np.sqrt((1.0 - self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu)) * self.EE / self.rho)
        c2 = np.sqrt(self.EE / (2.0 * self.rho * (1.0 + self.nu)))

        beta1_sq = 1.0 - (self.v / c1) ** 2
        beta2_sq = 1.0 - (self.v / c2) ** 2
        if beta1_sq <= 0.0 or beta2_sq <= 0.0:
            return np.nan

        beta1 = np.sqrt(beta1_sq)
        beta2 = np.sqrt(beta2_sq)
        denom = 4.0 * beta1 * beta2 - (1.0 + beta2 ** 2) ** 2
        if abs(denom) < 1e-14:
            return np.nan
        return (beta1 * (1.0 - beta2 ** 2)) / denom

    def _effective_modulus(self) -> float:
        # plane stress vs plane strain
        if self.dmat == 1:
            return self.EE
        return self.EE / (1.0 - self.nu ** 2)

    def _calc_ki(self, J: float) -> float:
        AI = self._dynamic_factor_AI()
        if not np.isfinite(AI) or AI <= 0.0 or J <= 0.0:
            return 0.0
        Eeff = self._effective_modulus()
        return float(np.sqrt(Eeff * J / AI))

    def calc_J_single_step(self, step: int) -> Dict[str, float]:
        """Calculate J-integral and DSIF for one step."""
        data = self._get_step_data(step)

        if self.extend_symmetric:
            data = self.extend_symmetric_mesh(data.node, data.elem, data.dis, data.vel, data.acce)
            if self.save_extended_debug:
                self._write_extended_debug_dat(step, data)

        x_tip = self.crack_tip_x(step)
        q = self.compute_weight_function(data.node, x_tip)
        q = self._force_q_near_tip(data.node, data.elem, q, x_tip)

        active = np.where(np.sum(q[data.elem], axis=1) > 1e-12)[0]
        if len(active) == 0:
            return {"step": int(step), "J": 0.0, "J_static": 0.0, "J_dynamic": 0.0, "K_I": 0.0}

        J_static = 0.0
        J_dynamic = 0.0

        for eidx in active:
            conn = data.elem[eidx]
            xe = data.node[conn, :]           # (4,2)
            ue = data.dis[conn, :]            # (4,2)
            ae = data.acce[conn, :]           # (4,2)
            qe = q[conn]                      # (4,)

            for N, dN_par, w in zip(self.nn, self.dnn, self.gauss_w):
                jac = dN_par @ xe
                det_jac = float(np.linalg.det(jac))
                if abs(det_jac) < 1e-20:
                    continue

                dN = np.linalg.inv(jac) @ dN_par
                B = enlarge(dN)

                ue8 = ue.reshape(-1)
                strain = B @ ue8
                stress = self.de @ strain
                sxx, syy, txy = stress

                dux_dx = float(dN[0] @ ue[:, 0])
                duy_dx = float(dN[0] @ ue[:, 1])
                dq_dx = float(dN[0] @ qe)
                dq_dy = float(dN[1] @ qe)

                ax_gp = float(N @ ae[:, 0])
                ay_gp = float(N @ ae[:, 1])

                W = 0.5 * float(np.dot(strain, stress))
                # 2D local-mesh coordinate convention:
                # this sign matches the original 2D implementation.
                s1 = W - (sxx * dux_dx + txy * duy_dx)
                s2 = -(txy * dux_dx + syy * duy_dx)

                J_static += (s1 * dq_dx + s2 * dq_dy) * det_jac * w
                # Match 3D reference: dynamic term does not multiply q at GP.
                J_dynamic += self.rho * (ax_gp * dux_dx + ay_gp * duy_dx) * det_jac * w

        if not self.extend_symmetric:
            # If only upper half-mesh is used, mirror contribution.
            J_static *= 2.0
            J_dynamic *= 2.0

        J_total = J_static + J_dynamic
        K_I = self._calc_ki(J_total)

        return {
            "step": int(step),
            "J": float(J_total),
            "J_static": float(J_static),
            "J_dynamic": float(J_dynamic),
            "K_I": float(K_I),
        }

    # ------------------------------------------------------------------
    # Batch run / output
    # ------------------------------------------------------------------
    def _available_steps(self) -> List[int]:
        steps: List[int] = []

        if self.use_saved_files:
            for s, d in sorted(self._step_dirs().items()):
                if self._step_dir_has_solution(d):
                    steps.append(s)

        if len(steps) == 0 and len(st.nodeL2DAllMa2D) > 0:
            s0 = int(getattr(st, "stepini", 0))
            steps = [s0 + i for i in range(len(st.nodeL2DAllMa2D))]

        if len(steps) == 0:
            return []

        smin = self.step_start
        smax = self.step_end if self.step_end is not None else max(steps)
        return [s for s in steps if smin <= s <= smax]

    def run(self, output_file: Optional[Path] = None) -> List[Dict[str, float]]:
        """Run J-integral calculation for all selected steps."""
        steps = self._available_steps()
        if len(steps) == 0:
            raise RuntimeError(
                "No steps found for J-integral calculation. "
                "Check saved result files or state buffers."
            )

        results = [self.calc_J_single_step(step) for step in steps]

        if output_file is None:
            out_dir = self.result_dir if self.result_dir is not None else Path.cwd()
            rgl_val = getattr(st, "rGL", None)
            if rgl_val is None:
                rgl_val = getattr(st, "rGLlist", 0)
            rgl = int(rgl_val)
            output_file = out_dir / f"J_integral_2D_v{int(self.v)}_rGL{rgl}.csv"
        else:
            output_file = Path(output_file)
            if output_file.parent != Path("."):
                output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "J_total", "J_static", "J_dynamic", "K_I"])
            for row in results:
                writer.writerow([row["step"], row["J"], row["J_static"], row["J_dynamic"], row["K_I"]])

        return results


def calculate_jintegral_2d(
    step_start: int,
    step_end: Optional[int] = None,
    Rj0: float = 1.5,
    Rj1: float = 1.515,
    result_dir: Optional[Path] = None,
    output_file: Optional[Path] = None,
    use_saved_files: bool = True,
    extend_symmetric: bool = True,
) -> List[Dict[str, float]]:
    """Convenience wrapper for batch J-integral + DSIF calculation."""
    calc = JIntegral2D(
        step_start=step_start,
        step_end=step_end,
        Rj0=Rj0,
        Rj1=Rj1,
        result_dir=result_dir,
        use_saved_files=use_saved_files,
        extend_symmetric=extend_symmetric,
    )
    return calc.run(output_file=output_file)
