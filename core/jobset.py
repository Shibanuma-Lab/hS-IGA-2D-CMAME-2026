"""
jobset – configure a single job: directories, time stepping, Hooke tensor,
Gauss-point arrays and shape-function tables.

Mathematica: jobset[job_] := Module[{}, ...]
"""

import math
from pathlib import Path

import numpy as np

import core.state as st
from utils.shape_functions import GP, GW, shp, Dshp


# ------------------------------------------------------------------
def getlist(value_or_list, job, list_name=""):
    """
    Mathematica: getlist[list_] := If[Head[list]==List, ..., list]
    """
    if isinstance(value_or_list, (list, tuple, np.ndarray)):
        if len(value_or_list) < job:
            raise ValueError(
                f"Short of list: {list_name} (len={len(value_or_list)}) for job={job}"
            )
        return value_or_list[job - 1]
    return value_or_list


# ------------------------------------------------------------------
def _compute_ctrlpts_from_target_length(
    target_len: float,
    hG: float,
    degree: int,
    fit_mode: str = "closest",
):
    """
    Compute number of elements/control points from target domain length.

    Returns
    -------
    (npts, nelem, actual_len)
    """
    if target_len <= 0.0:
        raise ValueError(f"target_len must be > 0, got {target_len}")
    if hG <= 0.0:
        raise ValueError(f"hG must be > 0, got {hG}")

    ratio = float(target_len) / float(hG)
    mode = str(fit_mode).strip().lower()
    if mode == "nearest":
        mode = "closest"

    if mode == "cover":
        nelem = max(1, int(math.ceil(ratio - 1.0e-12)))
        # Keep actual domain strictly larger than requested target.
        if nelem * hG <= target_len:
            nelem += 1
    elif mode == "closest":
        lower = max(1, int(math.floor(ratio)))
        upper = max(1, int(math.ceil(ratio)))
        candidates = sorted({lower, upper})
        # On exact tie, prefer the larger mesh for conservative coverage.
        nelem = min(
            candidates,
            key=lambda n: (abs(float(n) * float(hG) - float(target_len)), -int(n)),
        )
    else:
        raise ValueError(f"Unsupported global domain fit mode: {fit_mode}")

    npts = int(nelem + int(degree))
    actual_len = float(nelem) * float(hG)
    return npts, nelem, actual_len


# ------------------------------------------------------------------
def jobset(job):
    """Execute a full job-level setup."""

    if st.printcheck == 1:
        print("jobset_1")

    print("number of jobs:", int(st.jobend) - int(st.jobstart) + 1)
    print()

    # ---------------- getlist ----------------
    st.rGL      = getlist(st.rGLlist,      job, "rGLlist")
    st.aL       = getlist(st.aLlist,       job, "aLlist")
    st.lL       = getlist(st.lLlist,       job, "lLlist")
    st.HL       = getlist(st.HLlist,       job, "HLlist")
    st.hrefL    = getlist(st.hrefLlist,    job, "hrefLlist")
    st.v        = getlist(st.vlist,        job, "vlist")
    st.jobname  = getlist(st.jobnamelist,  job, "jobnamelist")
    st.islocal  = getlist(st.islocallist,  job, "islocallist")
    st.isdynamic = getlist(st.isdynamiclist, job, "isdynamiclist")

    # ---------------- stepall ----------------
    if st.stepend == -1:
        st.stepall = int(round(st.c_crack / st.hL))
    else:
        st.stepall = int(st.stepend)

    if st.meshonly == 1:
        st.stepall = int(st.stepini)
        st.jobend  = int(st.jobstart)

    print("num. of step:", st.stepall)

    # ---------------- directories ----------------
    # Build parameter-based folder name
    folder_name = f"v_{int(st.v)}_rGL_{st.rGL}_aL_{st.aL}_lL_{st.lL}_HL_{st.HL}"
    folder_tag = str(getattr(st, "result_folder_tag", "")).strip()
    if folder_tag != "":
        folder_name = f"{folder_name}_{folder_tag}"
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    st.dirname = results_dir / folder_name
    print("Result folder:", folder_name)
    print()

    if not st.dirname.exists():
        st.dirname.mkdir(parents=True, exist_ok=True)

    st.pvd = st.dirname / "paraview"
    if not st.pvd.exists():
        st.pvd.mkdir(exist_ok=True)
        (st.pvd / "vtu").mkdir(exist_ok=True)
        (st.pvd / "pvtu").mkdir(exist_ok=True)

    st.xld = st.dirname / "excel"
    if not st.xld.exists():
        st.xld.mkdir(exist_ok=True)

    # ---------------- scale / time step ----------------
    st.hG  = st.hL * st.rGL * math.sqrt(1.01)
    st.nLr = st.aL + st.lL

    if int(getattr(st, "domain_target_x_auto_from_crack", 1)) == 1:
        scale = float(getattr(st, "domain_target_x_from_crack_scale", 1.5))
        st.domain_target_x = scale * float(st.c_crack)

    if int(getattr(st, "auto_global_domain", 1)) == 1:
        fit_mode = str(getattr(st, "global_domain_fit_mode", "closest"))
        st.nPtsX, st.nelemX, st.domain_x_actual = _compute_ctrlpts_from_target_length(
            float(st.domain_target_x), float(st.hG), int(st.p), fit_mode=fit_mode
        )
        st.nPtsY, st.nelemY, st.domain_y_actual = _compute_ctrlpts_from_target_length(
            float(st.domain_target_y), float(st.hG), int(st.q), fit_mode=fit_mode
        )

        print(
            "Global domain auto-fit:",
            f"mode={fit_mode}",
            f"Lx_target={st.domain_target_x:.6e}",
            f"Ly_target={st.domain_target_y:.6e}",
            f"-> Lx={st.domain_x_actual:.6e}",
            f"Ly={st.domain_y_actual:.6e}",
            f"(nelemX={st.nelemX}, nelemY={st.nelemY})",
        )
    else:
        st.nelemX = int(st.nPtsX - st.p)
        st.nelemY = int(st.nPtsY - st.q)
        st.domain_x_actual = float(st.nelemX) * float(st.hG)
        st.domain_y_actual = float(st.nelemY) * float(st.hG)

    st.beta_rayleigh = (
        2.57
        * (st.hG if int(st.islocal) == 0 else st.hL)
        * math.sqrt(st.rho / st.EE)
    )

    if int(st.islocal) == 0:
        st.Delta_t = st.hG / float(st.v)
    else:
        st.Delta_t = (st.hL / float(st.v)) if float(st.v) > 0.0 else 1.0
    st.dt = st.Delta_t / float(st.inc)

    # ---------------- Hooke tensor ----------------
    if st.dmat == 1:
        # plane stress
        st.de = (st.EE / (1.0 - st.nu ** 2)) * np.array(
            [[1.0, st.nu, 0.0],
             [st.nu, 1.0, 0.0],
             [0.0, 0.0, 0.5 - 0.5 * st.nu]],
            dtype=float,
        )
    elif st.dmat == 2:
        # plane strain
        st.de = (st.EE / ((1.0 + st.nu) * (1.0 - 2.0 * st.nu))) * np.array(
            [[1.0 - st.nu, st.nu, 0.0],
             [st.nu, 1.0 - st.nu, 0.0],
             [0.0, 0.0, 0.5 - st.nu]],
            dtype=float,
        )
    else:
        raise ValueError(f"dmat must be 1 or 2, got {st.dmat}")

    st.dRho = np.array([[st.rho, 0.0], [0.0, st.rho]], dtype=float)

    # ---------------- Gauss points & shape functions ----------------
    st.XiEtaG  = np.array([(b, a) for a in GP(st.ngpG) for b in GP(st.ngpG)], dtype=float)
    st.weightG  = np.array([w1 * w2 for w1 in GW(st.ngpG) for w2 in GW(st.ngpG)], dtype=float)

    st.XiEtaL  = np.array([(b, a) for a in GP(st.ngpL) for b in GP(st.ngpL)], dtype=float)
    st.weightL  = np.array([w1 * w2 for w1 in GW(st.ngpL) for w2 in GW(st.ngpL)], dtype=float)

    st.XiEtaGL = np.array([(b, a) for a in GP(st.ngpGL) for b in GP(st.ngpGL)], dtype=float)
    st.weightGL = np.array([w1 * w2 for w1 in GW(st.ngpGL) for w2 in GW(st.ngpGL)], dtype=float)

    st.nnG  = np.array([shp(pt) for pt in st.XiEtaG],  dtype=float)
    st.DnnG = np.array([Dshp(pt) for pt in st.XiEtaG],  dtype=float)
    st.nnL  = np.array([shp(pt) for pt in st.XiEtaL],  dtype=float)
    st.DnnL = np.array([Dshp(pt) for pt in st.XiEtaL],  dtype=float)
    st.nnGL = np.array([shp(pt) for pt in st.XiEtaGL], dtype=float)
    st.DnnGL = np.array([Dshp(pt) for pt in st.XiEtaGL], dtype=float)

    # lowercase aliases used by makeKL / makeKGL etc.
    st.xi_etaG  = st.XiEtaG
    st.xi_etaL  = st.XiEtaL
    st.xi_etaGL = st.XiEtaGL

    if st.printcheck == 1:
        print("jobset_end")
