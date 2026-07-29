#!/usr/bin/env python3
"""Run the dynamic global B-spline order-sensitivity calculation.

The default calculation is the requested cubic-global case at ``v=500``:

* baseline geometry from ``run_velocity_baseline_sweep.py``:
  ``rGL=8``, ``aL=20``, ``lL=10``, ``HL=15``, and a 10 mm crack;
* cubic global IGA in both parametric directions (``p=q=3``);
* four Gauss points for the global matrix integration.  This is required for
  the degree-six ``N^T N`` mass integrand of cubic B-splines;
* the local Q4 integration remains 3x3, and the coupling integration defaults
  to 3x3, consistent with the coupling-quadrature study.

The result folder contains an explicit degree/quadrature tag, so it can never
be confused with the quadratic baseline.  Normalized crack-line stress is
written by ``calnos()``, while normalized DSIF is written by the automatic
J-integral/FEM comparison in ``main.execution()``.

Examples
--------
Check the exact planned cubic case and the available FEM reference:

    python3 run_velocity_global_order_sweep.py --dry-run

Run the requested cubic calculation:

    python3 run_velocity_global_order_sweep.py

For a strictly controlled p=2/p=3 comparison, run both degrees with the same
current geometry, FEM source, time stepping, and output convention:

    python3 run_velocity_global_order_sweep.py --degrees 2 3

If an already verified quadratic result is available, pass it explicitly to
create comparison CSV files after the cubic run:

    python3 run_velocity_global_order_sweep.py \
        --p2-result-dir results/<verified-p2-directory>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

import core.state as st
from config.fem_data import resolve_fem_reference_path
from config.parameters import load_parameters
from main import execution


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class GlobalOrderCase:
    """One dynamic calculation with a prescribed global B-spline degree."""

    velocity: int
    rGL: int
    aL: int
    lL: int
    HL: int
    crack_mm: int
    stepend: int
    p: int
    q: int
    ngpG: int
    ngpL: int
    ngpGL: int
    fem_source: str
    fem_reference: str
    user_tag: str


def _ceil_mul(value: int, factor: float) -> int:
    return int(math.ceil(float(value) * float(factor)))


def _unique_positive_ints(values: Iterable[int], *, name: str) -> list[int]:
    result = list(dict.fromkeys(int(value) for value in values))
    if not result or any(value < 1 for value in result):
        raise ValueError(f"{name} must contain positive integers.")
    return result


def _result_tag(case: GlobalOrderCase) -> str:
    tag = f"p{case.p}q{case.q}_ngpG{case.ngpG}_ngpGL{case.ngpGL}"
    if case.user_tag:
        tag = f"{tag}_{case.user_tag}"
    return tag


def _base_folder_name(case: GlobalOrderCase) -> str:
    return (
        f"v_{case.velocity}_rGL_{case.rGL}_aL_{case.aL}_"
        f"lL_{case.lL}_HL_{case.HL}"
    )


def _result_folder_name(case: GlobalOrderCase) -> str:
    return f"{_base_folder_name(case)}_{_result_tag(case)}"


def _resolve_path_from_project(value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_tag(value: str) -> str:
    """Keep an optional rerun tag inside one result-folder component."""
    tag = str(value).strip()
    if tag and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", tag) is None:
        raise ValueError(
            "--tag may contain only letters, digits, '.', '_' and '-' and must "
            "not begin with punctuation."
        )
    return tag


def _build_cases(args: argparse.Namespace) -> list[GlobalOrderCase]:
    degrees = _unique_positive_ints(args.degrees, name="--degrees")
    if args.q is not None and len(degrees) != 1:
        raise ValueError("--q may be used only with one value in --degrees.")
    if args.q is not None and int(args.q) < 1:
        raise ValueError("--q must be positive when supplied.")

    if int(args.global_ngp) < 0:
        raise ValueError("--global-ngp must be zero (automatic) or positive.")
    if int(args.local_ngp) < 1 or int(args.coupling_ngp) < 1:
        raise ValueError("--local-ngp and --coupling-ngp must be positive.")
    if int(args.velocity) <= 0 or int(args.rgl) <= 0 or int(args.crack_mm) <= 0:
        raise ValueError("Velocity, rGL, and crack length must be positive.")
    if int(args.stepend) < -1:
        raise ValueError("--stepend must be -1 or a non-negative integer.")
    if int(args.stress_sample_column) < 1:
        raise ValueError("--stress-sample-column must be one-based and positive.")

    aL = _ceil_mul(int(args.rgl), float(args.al_ratio))
    lL = int(args.ll) if args.ll is not None else _ceil_mul(int(args.rgl), float(args.ll_ratio))
    HL = _ceil_mul(int(args.rgl), float(args.hl_ratio))
    if min(aL, lL, HL) < 1:
        raise ValueError("aL, lL, and HL must be positive.")

    cases: list[GlobalOrderCase] = []
    for degree in degrees:
        q = int(args.q) if args.q is not None else int(degree)
        ngpG = int(args.global_ngp) if int(args.global_ngp) > 0 else int(degree) + 1
        cases.append(
            GlobalOrderCase(
                velocity=int(args.velocity),
                rGL=int(args.rgl),
                aL=int(aL),
                lL=int(lL),
                HL=int(HL),
                crack_mm=int(args.crack_mm),
                stepend=int(args.stepend),
                p=int(degree),
                q=int(q),
                ngpG=int(ngpG),
                ngpL=int(args.local_ngp),
                ngpGL=int(args.coupling_ngp),
                fem_source=str(args.fem_source),
                fem_reference=str(args.fem_reference),
                user_tag=_validate_tag(args.tag),
            )
        )
    return cases


def _configure_state(case: GlobalOrderCase) -> None:
    """Reset the dynamic defaults and apply one degree-controlled case."""
    load_parameters(rGL_value=int(case.rGL))

    # A two-dimensional order study must raise both parametric degrees.
    st.p = int(case.p)
    st.q = int(case.q)

    # ngpG is used for both global stiffness and global mass.  For p=3,
    # four points integrate the degree-six N^T N mass term exactly.
    st.ngpG = int(case.ngpG)
    st.ngpL = int(case.ngpL)
    st.ngpGL = int(case.ngpGL)

    st.rGLlist = int(case.rGL)
    st.aLlist = int(case.aL)
    st.lLlist = int(case.lL)
    st.HLlist = int(case.HL)

    st.vlist = float(case.velocity)
    st.v = int(case.velocity)
    st.c_crack = float(case.crack_mm) * 1.0e-3
    st.stepend = int(case.stepend)
    st.stepall = (
        int(round(st.c_crack / st.hL))
        if int(case.stepend) < 0
        else int(case.stepend)
    )

    if int(getattr(st, "domain_target_x_auto_from_crack", 0)) == 1:
        scale = float(getattr(st, "domain_target_x_from_crack_scale", 1.5))
        st.domain_target_x = scale * float(st.c_crack)

    # Keep all requested dynamic post-processing enabled.
    st.issave = 1
    st.calc_jintegral = 1
    st.jintegral_compare_fem = 1
    st.result_folder_tag = _result_tag(case)
    st.jobstart = 1
    st.jobend = 1
    st.jobnamelist = (
        f"global_order_v{case.velocity}_rGL{case.rGL}_aL{case.aL}_"
        f"lL{case.lL}_HL{case.HL}_{_result_tag(case)}"
    )

    # Unlike the existing velocity runner, do not force its unavailable
    # h5_export_V_500_a_10 path.  With source='auto' and a 10 mm crack, the
    # repository's matching MAT reference is selected automatically.
    st.fem_reference_source = str(case.fem_source)
    st.fem_mat_file = str(case.fem_reference)
    st.jintegral_fem_mat_file = str(case.fem_reference)
    if str(case.fem_source).lower() == "h5" and str(case.fem_reference).lower() == "auto":
        st.fem_h5_dir = f"FEM_data/h5_export_V_{case.velocity}_a_{case.crack_mm}"
    elif str(case.fem_reference).lower() != "auto":
        st.fem_h5_dir = str(case.fem_reference)


def _write_summary(rows: Sequence[Dict[str, str]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "idx",
        "velocity",
        "rGL",
        "aL",
        "lL",
        "HL",
        "crack_mm",
        "stepend",
        "p",
        "q",
        "ngpG",
        "ngpL",
        "ngpGL",
        "folder",
        "status",
        "seconds",
        "fem_kind",
        "fem_path",
        "message",
    ]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _state_mesh_metadata() -> dict:
    return {
        "global_control_points": [
            int(getattr(st, "nPtsX", 0)),
            int(getattr(st, "nPtsY", 0)),
        ],
        "global_elements": [
            int(getattr(st, "nelemX", 0)),
            int(getattr(st, "nelemY", 0)),
        ],
        "global_dof": int(getattr(st, "neqG", 0)),
        "local_dof": int(getattr(st, "neqL", 0)),
        "total_dof": int(getattr(st, "neq", 0)),
        "hG": float(getattr(st, "hG", float("nan"))),
        "hL": float(getattr(st, "hL", float("nan"))),
        "dt": float(getattr(st, "dt", float("nan"))),
        "stepall": int(getattr(st, "stepall", 0)),
    }


def _write_case_manifest(
    destination: Path,
    case: GlobalOrderCase,
    *,
    fem_kind: str,
    fem_path: Path,
    status: str,
) -> None:
    payload = {
        "schema_version": 1,
        "study": "dynamic_global_bspline_order_sensitivity",
        "status": str(status),
        "case": asdict(case),
        "fem_reference": {"kind": str(fem_kind), "path": str(fem_path)},
        "mesh": _state_mesh_metadata(),
        "postprocess": {
            "normalized_dsif": "J_integral_2D_compare_hs_vs_FEM_*.csv / K_I_norm_hs_over_fem",
            "normalized_local_stress": "sigmanos_*.csv",
        },
    }
    with (destination / "global_order_case_manifest.json").open("w") as stream:
        json.dump(payload, stream, indent=2)


def _find_single_file(folder: Path, pattern: str) -> Path | None:
    matches = sorted(folder.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] Multiple '{pattern}' files in {folder}; using {matches[0].name}")
    return matches[0]


def _read_normalized_dsif(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            return values
        name_map = {name.strip(): name for name in reader.fieldnames if name is not None}
        step_key = name_map.get("Step")
        dsif_key = name_map.get("K_I_norm_hs_over_fem")
        if step_key is None or dsif_key is None:
            raise KeyError(f"Missing Step or K_I_norm_hs_over_fem in {path}")
        for row in reader:
            try:
                step = int(float(row[step_key]))
                value = float(row[dsif_key])
            except (KeyError, TypeError, ValueError):
                continue
            if step >= 1 and math.isfinite(value):
                values[step] = value
    return values


def _read_normalized_stress(path: Path) -> np.ndarray:
    values = np.genfromtxt(path, delimiter=",", dtype=float)
    if values.size == 0:
        return np.empty((0, 0), dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return np.asarray(values, dtype=float)


def _relative_delta(reference: float, value: float) -> float:
    if not math.isfinite(reference) or not math.isfinite(value) or reference == 0.0:
        return float("nan")
    return (value - reference) / reference


def _write_rows(destination: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _write_order_comparison(
    *,
    p2_dir: Path,
    p3_dir: Path,
    stress_sample_column: int,
) -> list[Path]:
    """Write directly plot-ready p=2 versus p=3 metric CSV files."""
    if stress_sample_column < 1:
        raise ValueError("--stress-sample-column must be one-based and positive.")
    if not p2_dir.is_dir() or not p3_dir.is_dir():
        raise FileNotFoundError("Both p=2 and p=3 result directories must exist for comparison.")

    outputs: list[Path] = []
    p2_dsif_file = _find_single_file(p2_dir, "J_integral_2D_compare_hs_vs_FEM_*.csv")
    p3_dsif_file = _find_single_file(p3_dir, "J_integral_2D_compare_hs_vs_FEM_*.csv")
    if p2_dsif_file is not None and p3_dsif_file is not None:
        p2_dsif = _read_normalized_dsif(p2_dsif_file)
        p3_dsif = _read_normalized_dsif(p3_dsif_file)
        rows = []
        for step in sorted(set(p2_dsif) | set(p3_dsif)):
            p2_value = p2_dsif.get(step, float("nan"))
            p3_value = p3_dsif.get(step, float("nan"))
            rows.append(
                [
                    step,
                    p2_value,
                    p3_value,
                    p3_value - p2_value,
                    _relative_delta(p2_value, p3_value),
                ]
            )
        output = p3_dir / "global_order_comparison_normalized_dsif.csv"
        _write_rows(
            output,
            ["step", "p2", "p3", "p3_minus_p2", "relative_difference_to_p2"],
            rows,
        )
        outputs.append(output)
    else:
        print("[WARN] DSIF comparison skipped: a J-integral comparison CSV is missing.")

    p2_stress_file = _find_single_file(p2_dir, "sigmanos_*.csv")
    p3_stress_file = _find_single_file(p3_dir, "sigmanos_*.csv")
    if p2_stress_file is None or p3_stress_file is None:
        print("[WARN] Normalized-stress comparison skipped: sigmanos CSV is missing.")
        return outputs

    p2_stress = _read_normalized_stress(p2_stress_file)
    p3_stress = _read_normalized_stress(p3_stress_file)
    sample_index = int(stress_sample_column) - 1
    nsteps = max(p2_stress.shape[0], p3_stress.shape[0])
    sample_rows = []
    for row_index in range(nsteps):
        p2_value = (
            float(p2_stress[row_index, sample_index])
            if row_index < p2_stress.shape[0] and sample_index < p2_stress.shape[1]
            else float("nan")
        )
        p3_value = (
            float(p3_stress[row_index, sample_index])
            if row_index < p3_stress.shape[0] and sample_index < p3_stress.shape[1]
            else float("nan")
        )
        sample_rows.append(
            [
                row_index + 1,
                p2_value,
                p3_value,
                p3_value - p2_value,
                _relative_delta(p2_value, p3_value),
            ]
        )
    output = p3_dir / "global_order_comparison_normalized_stress_sample.csv"
    _write_rows(
        output,
        [
            "step",
            "p2",
            "p3",
            "p3_minus_p2",
            "relative_difference_to_p2",
        ],
        sample_rows,
    )
    outputs.append(output)

    p2_final = p2_stress[-1] if p2_stress.shape[0] else np.array([], dtype=float)
    p3_final = p3_stress[-1] if p3_stress.shape[0] else np.array([], dtype=float)
    final_rows = []
    for index in range(max(len(p2_final), len(p3_final))):
        p2_value = float(p2_final[index]) if index < len(p2_final) else float("nan")
        p3_value = float(p3_final[index]) if index < len(p3_final) else float("nan")
        final_rows.append(
            [
                index + 1,
                p2_value,
                p3_value,
                p3_value - p2_value,
                _relative_delta(p2_value, p3_value),
            ]
        )
    output = p3_dir / "global_order_comparison_normalized_stress_final_step.csv"
    _write_rows(
        output,
        [
            "sample_index",
            "p2",
            "p3",
            "p3_minus_p2",
            "relative_difference_to_p2",
        ],
        final_rows,
    )
    outputs.append(output)

    metadata = {
        "p2_result_dir": str(p2_dir),
        "p3_result_dir": str(p3_dir),
        "stress_sample_column_one_based": int(stress_sample_column),
        "caution": (
            "Interpret these as an order comparison only when both directories "
            "use identical geometry, local mesh, time stepping, FEM reference, "
            "and J-integral settings."
        ),
    }
    output = p3_dir / "global_order_comparison_manifest.json"
    with output.open("w") as stream:
        json.dump(metadata, stream, indent=2)
    outputs.append(output)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic global B-spline order-sensitivity cases."
    )
    parser.add_argument("--velocity", type=int, default=500, help="Crack velocity (default: 500).")
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=[3],
        help="Global x-degree(s) to calculate (default: 3). Use '2 3' for a controlled pair.",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=None,
        help="Optional global y-degree for a single calculation (default: q=p).",
    )

    parser.add_argument("--rgl", type=int, default=8, help="Baseline rGL (default: 8).")
    parser.add_argument("--al-ratio", type=float, default=2.5, help="aL/rGL (default: 2.5).")
    parser.add_argument("--ll-ratio", type=float, default=1.2, help="lL/rGL (default: 1.2).")
    parser.add_argument("--ll", type=int, default=None, help="Optional absolute local-right span.")
    parser.add_argument("--hl-ratio", type=float, default=1.8, help="HL/rGL (default: 1.8).")
    parser.add_argument("--crack-mm", type=int, default=10, help="Crack length in mm (default: 10).")
    parser.add_argument("--stepend", type=int, default=-1, help="-1 uses all 200 default steps.")

    parser.add_argument(
        "--global-ngp",
        type=int,
        default=0,
        help="Global Gauss order; 0 uses p+1 (default: 0).",
    )
    parser.add_argument("--local-ngp", type=int, default=3, help="Local Q4 Gauss order (default: 3).")
    parser.add_argument(
        "--coupling-ngp",
        type=int,
        default=3,
        help="Coupling-matrix Gauss order (default: 3).",
    )

    parser.add_argument(
        "--fem-source",
        choices=("auto", "mat", "h5"),
        default="auto",
        help="FEM reference source (default: auto; resolves v=500/a=10 to the available MAT file).",
    )
    parser.add_argument(
        "--fem-reference",
        type=str,
        default="auto",
        help="FEM MAT file or H5 directory; default is automatic resolution.",
    )
    parser.add_argument(
        "--p2-result-dir",
        type=Path,
        default=None,
        help="Existing verified p=2 result directory for automatic comparison CSVs.",
    )
    parser.add_argument(
        "--stress-sample-column",
        type=int,
        default=4,
        help="One-based sigmanos sample column for the step-by-step comparison (default: 4).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Optional suffix for an intentionally distinct rerun, e.g. smoke_test.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without solving or writing files.")
    parser.add_argument("--force", action="store_true", help="Allow recomputation in an existing tagged folder.")
    parser.add_argument(
        "--fail-on-missing-fem",
        action="store_true",
        help="Treat a missing FEM reference as an error instead of a skipped calculation.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results") / "velocity_global_order_sweep_summary.csv",
        help="Summary CSV path (default: results/velocity_global_order_sweep_summary.csv).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        cases = _build_cases(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    os.chdir(PROJECT_ROOT)
    print(f"Project root: {PROJECT_ROOT}")
    print(
        "Baseline geometry:",
        f"v={args.velocity}",
        f"rGL={args.rgl}",
        f"aL={cases[0].aL}",
        f"lL={cases[0].lL}",
        f"HL={cases[0].HL}",
        f"crack={args.crack_mm}mm",
        f"stepend={args.stepend}",
    )
    print(
        "Degrees:",
        [f"p={case.p}, q={case.q}, ngpG={case.ngpG}, ngpL={case.ngpL}, ngpGL={case.ngpGL}" for case in cases],
    )
    print(f"FEM: source={args.fem_source}, reference={args.fem_reference}")

    rows: list[Dict[str, str]] = []
    result_dirs_by_degree: dict[int, Path] = {}

    for index, case in enumerate(cases, start=1):
        folder_name = _result_folder_name(case)
        folder_path = RESULTS_DIR / folder_name
        result_dirs_by_degree[int(case.p)] = folder_path
        row: Dict[str, str] = {
            "idx": str(index),
            "velocity": str(case.velocity),
            "rGL": str(case.rGL),
            "aL": str(case.aL),
            "lL": str(case.lL),
            "HL": str(case.HL),
            "crack_mm": str(case.crack_mm),
            "stepend": str(case.stepend),
            "p": str(case.p),
            "q": str(case.q),
            "ngpG": str(case.ngpG),
            "ngpL": str(case.ngpL),
            "ngpGL": str(case.ngpGL),
            "folder": folder_name,
            "status": "",
            "seconds": "",
            "fem_kind": "",
            "fem_path": "",
            "message": "",
        }

        try:
            _configure_state(case)
            fem_kind, fem_path = resolve_fem_reference_path(
                data_file=str(case.fem_reference),
                source_preference=str(case.fem_source),
            )
            row["fem_kind"] = str(fem_kind)
            row["fem_path"] = str(fem_path)
        except Exception as exc:
            row["status"] = "failed_preflight"
            row["message"] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(cases)}] FAIL preflight p={case.p}: {row['message']}")
            rows.append(row)
            continue

        print(
            f"[{index}/{len(cases)}] PLAN p={case.p}, q={case.q} -> {folder_name} "
            f"(FEM {fem_kind}: {fem_path})"
        )
        if not Path(fem_path).exists():
            message = f"FEM reference not found: {fem_path}"
            if args.fail_on_missing_fem:
                row["status"] = "failed_missing_fem"
                row["message"] = message
                print(f"[{index}/{len(cases)}] FAIL p={case.p}: {message}")
            else:
                row["status"] = "skip_missing_fem"
                row["message"] = message
                print(f"[{index}/{len(cases)}] SKIP p={case.p}: {message}")
            rows.append(row)
            continue

        if args.dry_run:
            row["status"] = "planned"
            row["message"] = "dry-run"
            rows.append(row)
            continue

        if folder_path.exists() and not args.force:
            row["status"] = "skip_existing"
            row["message"] = "result folder already exists"
            print(f"[{index}/{len(cases)}] SKIP p={case.p}: result folder already exists")
            rows.append(row)
            continue

        started = time.time()
        print(f"[{index}/{len(cases)}] RUN  p={case.p}, q={case.q} -> {folder_name}")
        try:
            execution()
            elapsed = time.time() - started
            row["status"] = "done"
            row["seconds"] = f"{elapsed:.2f}"
            row["message"] = "ok"
            _write_case_manifest(
                folder_path,
                case,
                fem_kind=str(fem_kind),
                fem_path=Path(fem_path),
                status="done",
            )
            print(f"[{index}/{len(cases)}] DONE p={case.p} ({elapsed:.2f}s)")
        except Exception as exc:
            elapsed = time.time() - started
            row["status"] = "failed"
            row["seconds"] = f"{elapsed:.2f}"
            row["message"] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(cases)}] FAIL p={case.p}: {row['message']}")
            traceback.print_exc()
        finally:
            os.chdir(PROJECT_ROOT)
        rows.append(row)

    if not args.dry_run:
        summary_path = _resolve_path_from_project(args.summary)
        _write_summary(rows, summary_path)
        try:
            summary_label = summary_path.relative_to(PROJECT_ROOT)
        except ValueError:
            summary_label = summary_path
        print(f"Summary: {summary_label}")

        p3_dir = result_dirs_by_degree.get(3)
        p2_dir = (
            _resolve_path_from_project(args.p2_result_dir)
            if args.p2_result_dir is not None
            else result_dirs_by_degree.get(2)
        )
        if p2_dir is not None and p3_dir is not None and p2_dir.is_dir() and p3_dir.is_dir():
            try:
                p3_case = next((case for case in cases if int(case.p) == 3), None)
                if p3_case is not None and not p2_dir.name.startswith(_base_folder_name(p3_case)):
                    print(
                        "[WARN] The supplied p=2 directory name does not match the "
                        "current geometry tuple (v/rGL/aL/lL/HL). Verify that it is "
                        "a controlled baseline before interpreting the comparison."
                    )
                comparison_outputs = _write_order_comparison(
                    p2_dir=p2_dir,
                    p3_dir=p3_dir,
                    stress_sample_column=int(args.stress_sample_column),
                )
                for output in comparison_outputs:
                    print(f"Comparison: {output.relative_to(PROJECT_ROOT)}")
            except Exception as exc:
                print(f"[WARN] Could not write p=2/p=3 comparison: {type(exc).__name__}: {exc}")
        elif p3_dir is not None and args.p2_result_dir is None and 2 not in result_dirs_by_degree:
            print("[INFO] No p=2 directory supplied; p=3 outputs are ready for later comparison.")

    failed = sum(1 for row in rows if row["status"].startswith("failed"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
