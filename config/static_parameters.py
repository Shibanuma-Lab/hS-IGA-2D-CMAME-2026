"""
Static benchmark configuration and sweep generation.
"""

from __future__ import annotations

import math

import numpy as np

import core.state as st


def _estimate_case_dof(case: dict, p: int, q: int, ndof: int = 2) -> int:
    """
    Estimate total DOF for one static case without building the full meshes.

    Matches runtime definition:
      dof = 2 * (len(nodeG) + len(nodeL))
      len(nodeG) = nPtsX * nPtsY,  nPtsX=nGx+p, nPtsY=nGy+q
      len(nodeL) = (nLr+1) * (HL+1), nLr=aL+lL
    """
    nGx = int(case["nGx"])
    nGy = int(case["nGy"])
    aL = int(case["aL"])
    lL = int(case["lL"])
    HL = int(case["HL"])

    nPtsX = nGx + int(p)
    nPtsY = nGy + int(q)
    n_global_nodes = nPtsX * nPtsY

    nLr = aL + lL
    n_local_nodes = (nLr + 1) * (HL + 1)

    return int(ndof) * int(n_global_nodes + n_local_nodes)


def _truncate_cases_by_dof(cases, max_dof: int, p: int, q: int, ndof: int = 2):
    """
    Keep cases in sweep order until estimated DOF exceeds ``max_dof``.
    The first exceeding case is not included.
    """
    kept = []
    for case in cases:
        dof_est = _estimate_case_dof(case, p=p, q=q, ndof=ndof)
        case_with_dof = dict(case)
        case_with_dof["dof_estimate"] = int(dof_est)
        if dof_est > int(max_dof):
            break
        kept.append(case_with_dof)
    return kept


def _adjust_global_divisions(nGxbf: int):
    """Match the Mathematica static-mesh rule: odd ``nGx`` and ``nGy=(nGx-1)/2``."""
    nGxbf = int(nGxbf)
    nGx = nGxbf + 1 if (nGxbf % 2 == 0) else nGxbf
    nGy = (nGx - 1) // 2
    return int(nGx), int(nGy)


def _nGy_from_nGx_exact(nGx: int) -> int:
    """
    Exact global Y-division rule used in integrated static campaigns:
    nGy = (nGx - 1) / 2 for odd nGx.
    """
    nGx = int(nGx)
    if nGx % 2 == 0:
        raise ValueError(f"nGx must be odd for exact campaign settings, got {nGx}")
    return int((nGx - 1) // 2)


def _make_case_with_counts(
    *,
    nhL: int,
    nGx: int,
    nGy: int,
    aL: int,
    lL: int,
    HL: int,
    rGL: float,
):
    """
    Build one static case from explicit element counts.
    """
    width = float(st.static_width)
    hL = 1.0 / float(int(nhL))
    hG = width / float(int(nGx))
    return {
        "nhL": int(nhL),
        "rGL": float(rGL),
        "hL": float(hL),
        "hG": float(hG),
        "nGx": int(nGx),
        "nGy": int(nGy),
        "aL": int(aL),
        "lL": int(lL),
        "HL": int(HL),
        "step": int(aL),
    }


def _make_case(nhL: int, rGL: int | float):
    stand_size = float(st.static_local_half_span)
    width = float(st.static_width)

    hL = 1.0 / float(nhL)
    aL = int(math.floor(stand_size / hL))
    lL = int(math.floor(stand_size / hL))
    HL = int(math.floor(stand_size / hL))

    hG_target = hL * float(rGL)
    nGxbf = int(math.floor(width / hG_target))
    nGx, nGy = _adjust_global_divisions(nGxbf)
    hG = width / float(nGx)

    return {
        "nhL": int(nhL),
        "rGL": float(rGL),
        "hL": float(hL),
        "hG": float(hG),
        "nGx": int(nGx),
        "nGy": int(nGy),
        "aL": int(aL),
        "lL": int(lL),
        "HL": int(HL),
        # Static benchmark uses a fixed crack length equal to the local left span.
        "step": int(aL),
    }


def _dedupe_cases_by_hg(cases, target_rgl: float):
    """
    Keep one case per unique global mesh size (nGx -> unique hG for fixed width).

    For duplicated ``nGx``, keep the case whose actual ``hG / hL`` is closest to
    the target ratio.
    """
    best_by_nGx = {}
    for case in cases:
        key = int(case["nGx"])
        ratio_err = abs((float(case["hG"]) / float(case["hL"])) - float(target_rgl))
        if key not in best_by_nGx:
            best_by_nGx[key] = (ratio_err, case)
            continue

        prev_err, prev_case = best_by_nGx[key]
        if ratio_err < prev_err:
            best_by_nGx[key] = (ratio_err, case)
        elif ratio_err == prev_err and int(case["nhL"]) < int(prev_case["nhL"]):
            best_by_nGx[key] = (ratio_err, case)

    # Preserve a stable sweep order from coarse -> fine local mesh.
    out = [v[1] for v in best_by_nGx.values()]
    out.sort(key=lambda c: int(c["nhL"]))
    return out


def load_static_parameters(sweep_mode: str = "fix_rGL"):
    """Populate ``core.state`` for the static crack benchmark."""

    st.analysis_mode = "static"
    st.static_sweep_mode = str(sweep_mode)
    st.save_vtu = 1

    st.p = 2
    st.q = 2
    st.dmat = 2
    st.EE = 200.0
    st.nu = 0.3
    st.rho = 7800.0
    st.thi = 1.0

    st.static_width = 2.0
    st.static_height = 1.0
    st.static_crack_tip_x = 1.0
    st.static_local_half_span = 0.5 * math.sqrt(1.01)

    st.ngpG = 2
    st.ngpL = st.p + 1
    st.ngpGL = st.p + 1
    st.static_kgl_ngpGL = 3
    st.hrefLlist = 1
    st.islocallist = 1
    st.isdynamiclist = 0
    st.islocal = 1
    st.isdynamic = 0

    st.inc = 1
    st.ismortar = 0
    st.nofix = 1
    st.abo = 0
    st.ini1x = 2
    st.zentai = 0

    st.postprocess = 0
    st.issave = 1
    st.printcheck = 0
    st.meshonly = 0
    st.debug_output = 0
    st.calc_jintegral = 0
    st.jintegral_compare_fem = 0
    st.interpolator_type = "bilinear"
    # Number of worker processes for static case sweep. Use 1 for serial run.
    st.static_parallel_jobs = 1
    # Stop generating further sweep cases once estimated DOF exceeds this cap.
    st.static_max_dof = int(1.0e5)
    # Memory control: use sparse matrices and skip mass matrices for static solve.
    st.static_use_sparse = 1
    st.static_skip_mass = 1
    st.static_linear_solver = "auto"   # "auto" | "direct" | "cg"
    st.static_iter_tol = 1.0e-10
    st.static_iter_maxiter = 50000
    st.static_iter_switch_dof = 120000
    st.static_release_memory_each_job = 1

    st.jobstart = 1

    if st.static_sweep_mode == "fix_rGL":
        fixed_rGL = 6
        st.static_parent_label = f"fix_rGL_{fixed_rGL:g}"
        raw_cases = [_make_case(nhL=nhL, rGL=fixed_rGL) for nhL in range(5, 10000, 4)]
        deduped = _dedupe_cases_by_hg(raw_cases, target_rgl=fixed_rGL)
        st.static_cases = _truncate_cases_by_dof(
            deduped, max_dof=st.static_max_dof, p=st.p, q=st.q, ndof=2
        )
    elif st.static_sweep_mode == "fix_hG":
        # Higher coupling quadrature is required for stable fix_hG trends.
        st.static_kgl_ngpGL = 5
        fixed_nG = 81
        nGx = int(fixed_nG)
        nGy = _nGy_from_nGx_exact(nGx)
        hG_fixed = st.static_width / float(nGx)
        st.static_parent_label = f"fix_hG_{hG_fixed:.10g}"
        cases = []
        for nhL in range(5, 10000, 4):
            case = _make_case(nhL=nhL, rGL=1.0)
            case["nGx"] = int(nGx)
            case["nGy"] = int(nGy)
            case["hG"] = float(hG_fixed)
            case["rGL"] = case["hG"] / case["hL"]
            cases.append(case)
        st.static_cases = _truncate_cases_by_dof(
            cases, max_dof=st.static_max_dof, p=st.p, q=st.q, ndof=2
        )
    elif st.static_sweep_mode == "fix_hL":
        fixed_nL = 20
        fixed_nhL = int(fixed_nL)
        fixed_half = int(fixed_nL // 2)
        hL_fixed = 1.0 / float(fixed_nL)
        st.static_parent_label = f"fix_hL_{hL_fixed:.10g}"
        raw_cases = []
        nGx = 3
        while True:
            nGy = _nGy_from_nGx_exact(nGx)
            if int(nGx) <= 0 or int(nGy) <= 0:
                break
            hG = float(st.static_width) / float(nGx)
            rGL = float(hG / hL_fixed)
            if rGL < 2.0:
                break
            raw_cases.append(
                _make_case_with_counts(
                    nhL=fixed_nhL,
                    nGx=int(nGx),
                    nGy=int(nGy),
                    aL=fixed_half,
                    lL=fixed_half,
                    HL=fixed_half,
                    rGL=float(rGL),
                )
            )
            nGx += 2
        st.static_cases = _truncate_cases_by_dof(
            raw_cases, max_dof=st.static_max_dof, p=st.p, q=st.q, ndof=2
        )
    else:
        raise ValueError(
            f"Unsupported static sweep mode: {st.static_sweep_mode}. "
            "Use 'fix_rGL', 'fix_hG', or 'fix_hL'."
        )

    st.jobend = len(st.static_cases)
    st.jobnamelist = st.static_parent_label

    st.mu = st.EE / (2.0 * (1.0 + st.nu))
    st.de = (st.EE / ((1.0 + st.nu) * (1.0 - 2.0 * st.nu))) * np.array(
        [
            [1.0 - st.nu, st.nu, 0.0],
            [st.nu, 1.0 - st.nu, 0.0],
            [0.0, 0.0, 0.5 - st.nu],
        ],
        dtype=float,
    )
    st.kappa = 3.0 - 4.0 * st.nu
