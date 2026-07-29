#!/usr/bin/env python3
"""
Run the two static full s-IGA reviewer-comparison cases.

The production hS-IGA path is not modified.  This driver reuses its global
quadratic IGA patch and static solver, while replacing the local Q4 Lagrange
correction mesh by an isolated quadratic IGA correction patch.

Cases
-----
1. hG = 2/21, nhL = 40
2. hG = 2/41, nhL = 80

The default local patch is a standard open-uniform quadratic B-spline patch:
no additional/repeated crack-tip knot is inserted, so it remains C1 at the
tip.  The previous C0 tip-knot-inserted formulation remains available as an
explicit comparison mode.  Only normalized stress yy is evaluated.  Direct total-field stress is
the recommended curve for a strict cross-formulation comparison.  L2 error,
SIF, and J-integral are deliberately skipped because their existing
implementations assume a nodal Q4 local mesh.  A combined global/local VTU
file is written for each case.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import core.state as st
from config.static_parameters import (
    _make_case_with_counts,
    _nGy_from_nGx_exact,
    load_static_parameters,
)
from core.alldata import Alldata
from core.jobset import jobset
from core.stepset import stepset
from full_siga.static_case import (
    apply_full_siga_static_boundary,
    assemble_global_local_iga_coupling,
    assemble_local_iga_stiffness,
    build_local_iga_mesh,
    compute_full_siga_normalized_stress_yy,
    read_case_result,
    setup_combined_mesh_state,
    write_full_siga_case_outputs,
    write_full_siga_parent_summaries,
    write_full_siga_vtu,
)
from mesh.global_mesh import makeGlobalMesh
from matrix.make_KG import makeKG
from postprocess.getresult import getresult
from solver.initial import initial
from solver.solve import solve


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_RESULTS_DIR = PROJECT_ROOT / "static_results"
DEFAULT_OUTPUT_NAMES = {
    "c1": "full_siga_static_comparison_c1",
    "c0": "full_siga_static_comparison",
}
CAMPAIGN_SCHEMA_VERSION = 1


def _validate_output_name(value: str) -> str:
    name = str(value).strip()
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "--output-name must be a non-empty relative path below static_results"
        )
    return name


def _case_label(nGx: int, nhL: int, local_tip_continuity: str) -> str:
    continuity = str(local_tip_continuity).lower()
    prefix = "full_siga" if continuity == "c0" else f"full_siga_{continuity}"
    return f"{prefix}_hG_2_over_{int(nGx)}_nhL_{int(nhL)}"


def build_target_cases(
    nGy_overrides: tuple[int, int] | None = None,
    local_tip_continuity: str = "c1",
) -> list[dict]:
    """Build the two meshes requested for the reviewer comparison."""
    continuity = str(local_tip_continuity).lower()
    if continuity not in {"c1", "c0"}:
        raise ValueError(
            "local_tip_continuity must be 'c1' or 'c0'"
        )
    cases = []
    targets = ((21, 40), (41, 80))
    for index, (nGx, nhL) in enumerate(targets):
        if nGy_overrides is None:
            nGy = _nGy_from_nGx_exact(nGx)
        else:
            nGy = int(nGy_overrides[index])
        half = nhL // 2
        hG = float(st.static_width) / float(nGx)
        hL = 1.0 / float(nhL)
        case = _make_case_with_counts(
            nhL=nhL,
            nGx=nGx,
            nGy=nGy,
            aL=half,
            lL=half,
            HL=half,
            rGL=hG / hL,
        )
        case["full_siga_tip_continuity"] = continuity.upper()
        case["full_siga_label"] = _case_label(
            nGx,
            nhL,
            continuity,
        )
        cases.append(case)
    return cases


def _estimated_full_siga_dof(
    case: dict,
    p: int,
    q: int,
    local_tip_continuity: str,
) -> int:
    global_cp = (int(case["nGx"]) + p) * (int(case["nGy"]) + q)
    extra_tip_cp = (p - 1) if str(local_tip_continuity).lower() == "c0" else 0
    local_cp_u = int(case["aL"]) + int(case["lL"]) + p + extra_tip_cp
    local_cp_v = int(case["HL"]) + q
    return 2 * (global_cp + local_cp_u * local_cp_v)


def configure_campaign(
    *,
    output_name: str,
    coupling_order: int,
    local_tip_continuity: str,
    nGy_overrides: tuple[int, int] | None = None,
) -> list[dict]:
    load_static_parameters(sweep_mode="fix_rGL")
    continuity = str(local_tip_continuity).lower()
    cases = build_target_cases(
        nGy_overrides=nGy_overrides,
        local_tip_continuity=continuity,
    )

    st.static_sweep_mode = "custom"
    st.static_parent_label = str(output_name)
    st.static_cases = cases
    st.static_parallel_jobs = 1
    st.static_kgl_ngpGL = int(coupling_order)
    st.full_siga_local_p = int(st.p)
    st.full_siga_local_q = int(st.q)
    st.full_siga_local_ngp = int(st.p) + 1
    st.full_siga_local_tip_continuity = continuity.upper()
    st.local_discretization = "iga"

    # The dedicated writer below replaces the legacy Q4 savedata/metrics path.
    st.issave = 0
    st.save_vtu = 0
    st.calc_jintegral = 0
    st.static_release_memory_each_job = 1

    st.jobstart = 1
    st.jobend = len(cases)
    st.jobnamelist = st.static_parent_label
    return cases


def _case_output_dir(output_name: str, case: dict) -> Path:
    return (
        STATIC_RESULTS_DIR
        / output_name
        / f"nGx_{int(case['nGx'])}_aL_{int(case['aL'])}"
    )


def _campaign_manifest(
    *,
    cases: list[dict],
    coupling_order: int,
    ligament_fixity: str,
    local_tip_continuity: str,
) -> dict:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "formulation": "full_s_iga_static_reviewer_comparison",
        "local_discretization": "quadratic_iga",
        "local_tip_continuity": str(local_tip_continuity).upper(),
        "ligament_fixity": str(ligament_fixity),
        "coupling_gauss_points_per_direction": int(coupling_order),
        "recommended_cross_formulation_stress_recovery": (
            "direct_total_field_D_times_BG_uG_plus_BL_uL"
        ),
        "cases": [
            {
                "nGx": int(case["nGx"]),
                "nGy": int(case["nGy"]),
                "nhL": int(case["nhL"]),
                "aL": int(case["aL"]),
                "lL": int(case["lL"]),
                "HL": int(case["HL"]),
            }
            for case in cases
        ],
    }


def _prepare_campaign_directory(output_name: str, manifest: dict) -> None:
    """
    Create or validate the campaign manifest before any case is overwritten.

    Reusing one parent directory with a different local fixity or coupling
    order could silently combine incompatible case JSON files.  The manifest
    turns that situation into an explicit error.
    """
    parent_dir = STATIC_RESULTS_DIR / output_name
    manifest_path = parent_dir / "full_siga_campaign_manifest.json"
    if manifest_path.is_file():
        with manifest_path.open() as stream:
            existing = json.load(stream)
        if existing != manifest:
            raise ValueError(
                f"{manifest_path.relative_to(PROJECT_ROOT)} belongs to a "
                "different campaign configuration. Choose a new "
                "--output-name instead of mixing results."
            )
        return

    existing_results = sorted(
        parent_dir.glob("*/full_siga_case_result.json")
    ) if parent_dir.is_dir() else []
    if existing_results:
        raise ValueError(
            f"{parent_dir.relative_to(PROJECT_ROOT)} contains full s-IGA "
            "case results but no campaign manifest. Choose a new "
            "--output-name to avoid mixing configurations."
        )

    parent_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)


def _matrix_diagnostics() -> dict:
    local_symmetry = st.KL - st.KL.T
    local_symmetry_error = (
        float(np.max(np.abs(local_symmetry.data)))
        if local_symmetry.nnz > 0
        else 0.0
    )
    coupled = sp.bmat(
        [[st.KG, st.KGL], [st.KGL.T, st.KL]],
        format="csr",
    )
    residual = np.asarray(coupled @ st.dis - st.force, dtype=float)
    free_residual = residual[np.asarray(st.freedof, dtype=int)]
    return {
        "local_stiffness_symmetry_max_abs": local_symmetry_error,
        "free_residual_max_abs": (
            float(np.max(np.abs(free_residual)))
            if len(free_residual) > 0
            else 0.0
        ),
        "global_stiffness_nnz": int(st.KG.nnz),
        "local_stiffness_nnz": int(st.KL.nnz),
        "coupling_stiffness_nnz": int(st.KGL.nnz),
        "coupling_cell_intersections": int(st.full_siga_coupling_intersections),
        "fixed_dof_count": int(len(np.unique(
            2 * st.ebc[:, 0].astype(int) + st.ebc[:, 1].astype(int) - 1
        ))),
        "free_dof_count": int(len(st.freedof)),
    }


def run_case(
    *,
    job: int,
    case: dict,
    ligament_fixity: str,
    write_vtu: bool,
) -> dict:
    os.chdir(PROJECT_ROOT)
    Alldata()
    st.jobstart = int(job)
    st.jobend = int(job)
    jobset(job)
    st.static_case_label = str(case["full_siga_label"])

    step = int(st.stepall)
    stepset(step)

    makeGlobalMesh(st.hG, st.nPtsX, st.nPtsY)
    st.nodeG = st.controlPts
    st.elemG = [conn[:] for conn in st.element]

    build_local_iga_mesh()
    setup_combined_mesh_state()

    print(
        "[FULL-SIGA] Mesh:",
        f"global CP={st.nnmG}",
        f"local CP={st.nnmL}",
        f"local spans={st.nemL}",
        f"total DOF={st.neq}",
    )
    print(
        "[FULL-SIGA] Assembling:",
        f"local degree={st.local_iga_p}x{st.local_iga_q}",
        f"coupling quadrature={st.static_kgl_ngpGL}x{st.static_kgl_ngpGL}",
    )
    makeKG()
    assemble_local_iga_stiffness()
    assemble_global_local_iga_coupling()

    apply_full_siga_static_boundary(ligament_fixity=ligament_fixity)
    initial(step)
    solve(step)
    getresult()

    result = compute_full_siga_normalized_stress_yy()
    result["diagnostics"] = _matrix_diagnostics()
    result["local_control_point_count"] = int(st.nnmL)
    result["local_element_count"] = int(st.nemL)
    result["local_tip_continuity"] = str(st.local_iga_tip_continuity)
    result["ligament_fixity"] = str(ligament_fixity)
    result["coupling_order"] = int(st.static_kgl_ngpGL)

    write_full_siga_case_outputs(st.dirname, result)
    if write_vtu:
        vtu_path = write_full_siga_vtu(st.dirname, step)
        print(f"[FULL-SIGA] VTU: {vtu_path.relative_to(PROJECT_ROOT)}")

    print(
        "[FULL-SIGA] Diagnostics:",
        json.dumps(result["diagnostics"], sort_keys=True),
    )
    os.chdir(PROJECT_ROOT)
    return result


def _discover_case_results(output_name: str) -> list[dict]:
    parent_dir = STATIC_RESULTS_DIR / output_name
    if not parent_dir.is_dir():
        return []
    results = []
    for path in sorted(parent_dir.glob("*/full_siga_case_result.json")):
        results.append(read_case_result(path))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run static full s-IGA local-patch comparisons for "
            "(hG=2/21, nhL=40) and (hG=2/41, nhL=80)."
        )
    )
    parser.add_argument(
        "--case",
        choices=("all", "coarse", "fine"),
        default="all",
        help=(
            "Case selection: coarse=2/21+40, fine=2/41+80 "
            "(default: all)."
        ),
    )
    parser.add_argument(
        "--coupling-order",
        type=int,
        default=3,
        help="Gauss points per direction for KGL intersections (default: 3).",
    )
    parser.add_argument(
        "--local-tip-continuity",
        choices=("c1", "c0"),
        default="c1",
        help=(
            "Local quadratic IGA continuity at the crack tip: c1 uses a "
            "plain open-uniform knot vector with no additional/repeated tip "
            "knot insertion "
            "(default); c0 inserts one additional tip knot."
        ),
    )
    parser.add_argument(
        "--ngy",
        nargs=2,
        type=int,
        metavar=("COARSE", "FINE"),
        help=(
            "Override the global y-element counts for the coarse/fine cases. "
            "Default uses the current campaign rule 10 20. For example, use "
            "'--ngy 11 20' only when matching an older hS-IGA baseline with "
            "nGy=11 in the coarse case."
        ),
    )
    parser.add_argument(
        "--ligament-fixity",
        choices=("normal", "xy"),
        default="normal",
        help=(
            "Local bottom constraint at tip/ligament: normal fixes uy only "
            "(physical symmetry, default); xy fixes both correction components."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Relative campaign directory below static_results. Defaults to "
            "full_siga_static_comparison_c1 for c1 and the existing "
            "full_siga_static_comparison directory for c0."
        ),
    )
    parser.add_argument(
        "--no-vtu",
        action="store_true",
        help="Do not write the combined global/local VTU files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print mesh sizes and output paths without assembling or solving.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute a case even when its output directory is non-empty.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if int(args.coupling_order) < 1:
        raise SystemExit("--coupling-order must be positive")
    if args.ngy is not None and any(int(value) < 1 for value in args.ngy):
        raise SystemExit("--ngy values must be positive")
    continuity = str(args.local_tip_continuity).lower()
    requested_output_name = (
        args.output_name
        if args.output_name is not None
        else DEFAULT_OUTPUT_NAMES[continuity]
    )
    try:
        output_name = _validate_output_name(requested_output_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cases = configure_campaign(
        output_name=output_name,
        coupling_order=int(args.coupling_order),
        local_tip_continuity=continuity,
        nGy_overrides=(
            (int(args.ngy[0]), int(args.ngy[1]))
            if args.ngy is not None
            else None
        ),
    )
    if args.case == "coarse":
        selected_jobs = [(1, cases[0])]
    elif args.case == "fine":
        selected_jobs = [(2, cases[1])]
    else:
        selected_jobs = list(enumerate(cases, start=1))

    print(f"Project root: {PROJECT_ROOT}")
    print("Comparison: full s-IGA (quadratic IGA in global and local patches)")
    print("Metrics: normalized stress yy only; L2/SIF/J are skipped")
    print("Recommended comparison curve: direct total-field stress")
    if continuity == "c1":
        print(
            "Additional diagnostic curve: generalized-reaction projection "
            "(not a pure ligament-traction recovery in C1)"
        )
    else:
        print(
            "Additional native curve: lumped generalized-reaction projection"
        )
    print(
        "Caution: native Q4 and IGA reaction curves use different recovery "
        "kernels; the legacy hS metric also divides by nominal hL"
    )
    print("VTU stress recovery: direct total field")
    if continuity == "c1":
        print(
            "Local tip continuity: C1 "
            "(plain knot vector; no additional tip repetition)"
        )
        print(
            "Tip BC limitation: no control point lies exactly at the crack "
            "tip, so the crack-face/ligament transition is not exact"
        )
    else:
        print("Local tip continuity: C0 (one additional tip knot inserted)")
    print(f"Ligament fixity: {args.ligament_fixity}")
    print(f"Output: static_results/{output_name}")

    if not args.dry_run:
        try:
            _prepare_campaign_directory(
                output_name,
                _campaign_manifest(
                    cases=cases,
                    coupling_order=int(args.coupling_order),
                    ligament_fixity=str(args.ligament_fixity),
                    local_tip_continuity=continuity,
                ),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    for job, case in selected_jobs:
        output_dir = _case_output_dir(output_name, case)
        estimated_dof = _estimated_full_siga_dof(
            case,
            int(st.p),
            int(st.q),
            continuity,
        )
        print(
            f"[CASE] {case['full_siga_label']}: "
            f"nGx/nGy={case['nGx']}/{case['nGy']}, nhL={case['nhL']}, "
            f"local spans={case['aL'] + case['lL']}x{case['HL']}, "
            f"estimated DOF={estimated_dof}, "
            f"output={output_dir.relative_to(PROJECT_ROOT)}"
        )
        if args.dry_run:
            continue
        if output_dir.is_dir() and any(output_dir.iterdir()) and not args.force:
            print(
                f"[SKIP] {output_dir.relative_to(PROJECT_ROOT)} is non-empty; "
                "use --force to recompute."
            )
            continue

        print(f"[RUN] {case['full_siga_label']}")
        run_case(
            job=job,
            case=case,
            ligament_fixity=str(args.ligament_fixity),
            write_vtu=not bool(args.no_vtu),
        )
        print(f"[DONE] {case['full_siga_label']}")

    if not args.dry_run:
        results = _discover_case_results(output_name)
        if results:
            write_full_siga_parent_summaries(
                STATIC_RESULTS_DIR / output_name,
                results,
            )
        print(
            f"[DONE] Full s-IGA comparison finished; "
            f"available case results={len(results)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
