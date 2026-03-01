"""
Static benchmark configuration and sweep generation.
"""

from __future__ import annotations

import math

import numpy as np

import core.state as st


def _adjust_global_divisions(nGxbf: int):
    """Match the Mathematica static-mesh rule: odd ``nGx`` and ``nGy=(nGx-1)/2``."""
    nGxbf = int(nGxbf)
    nGx = nGxbf + 1 if (nGxbf % 2 == 0) else nGxbf
    nGy = (nGx - 1) // 2
    return int(nGx), int(nGy)


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
        raw_cases = [_make_case(nhL=nhL, rGL=fixed_rGL) for nhL in range(20, 10000, 4)]
        st.static_cases = _dedupe_cases_by_hg(raw_cases, target_rgl=fixed_rGL)
    elif st.static_sweep_mode == "fix_hG":
        fixed_nG = 81
        nGx, nGy = _adjust_global_divisions(fixed_nG)
        hG_fixed = st.static_width / float(nGx)
        st.static_parent_label = f"fix_hG_{hG_fixed:.10g}"
        cases = []
        for nhL in range(61, 160, 3):
            case = _make_case(nhL=nhL, rGL=1.0)
            case["nGx"] = int(nGx)
            case["nGy"] = int(nGy)
            case["hG"] = float(hG_fixed)
            case["rGL"] = case["hG"] / case["hL"]
            cases.append(case)
        st.static_cases = cases
    elif st.static_sweep_mode == "fix_hL":
        fixed_nhL = 20
        hL_fixed = 1.0 / float(fixed_nhL)
        st.static_parent_label = f"fix_hL_{hL_fixed:.10g}"
        st.static_cases = [_make_case(nhL=fixed_nhL, rGL=rGL) for rGL in range(2, 20)]
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
