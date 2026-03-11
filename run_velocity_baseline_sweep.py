#!/usr/bin/env python3
"""
Run baseline-parameter dynamic cases over a velocity sweep.

Baseline (default same as SFEM setting):
- rGL = 6
- aL = ceil(2.5 * rGL)
- lL = 15 (local elements)
- HL = ceil(1.8 * rGL)
- c_crack = 10 mm
- stepend = -1 (stepall = round(c_crack / hL))

Default velocity sweep:
- v = 200, 300, 400, ..., 1500

The script checks FEM reference source existence for each velocity before
running. Missing FEM references are skipped by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import core.state as st
from config.fem_data import resolve_fem_reference_path
from config.parameters import load_parameters
from main import execution


@dataclass(frozen=True)
class VelocityCase:
    v: int
    rGL: int
    aL: int
    lL: int
    HL: int
    crack_mm: int
    stepend: int


def _ceil_mul(value: int, factor: float) -> int:
    return int(math.ceil(float(value) * float(factor)))


def _folder_name(case: VelocityCase) -> str:
    return (
        f"v_{int(case.v)}_rGL_{int(case.rGL)}_aL_{int(case.aL)}_"
        f"lL_{int(case.lL)}_HL_{int(case.HL)}"
    )


def _parse_velocity_list(raw: str) -> List[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Velocity list is empty.")
    out = [int(float(v)) for v in vals]
    if any(v <= 0 for v in out):
        raise ValueError("All velocities must be positive.")
    return list(dict.fromkeys(out))


def _fem_h5_dir_for_case(case: VelocityCase) -> str:
    """FEM H5 directory naming for velocity sweep cases."""
    return f"FEM_data/h5_export_V_{int(case.v)}_a_{int(case.crack_mm)}"


def _build_cases(
    velocities: Sequence[int],
    rgl: int,
    aL_ratio: float,
    lL_const: int,
    HL_ratio: float,
    crack_mm: int,
    stepend: int,
) -> List[VelocityCase]:
    aL = _ceil_mul(rgl, aL_ratio)
    lL = int(lL_const)
    HL = _ceil_mul(rgl, HL_ratio)

    return [
        VelocityCase(
            v=int(v),
            rGL=int(rgl),
            aL=int(aL),
            lL=int(lL),
            HL=int(HL),
            crack_mm=int(crack_mm),
            stepend=int(stepend),
        )
        for v in velocities
    ]


def _configure_state_for_case(case: VelocityCase) -> None:
    load_parameters(rGL_value=int(case.rGL))

    # Baseline geometry/control parameters
    st.rGLlist = int(case.rGL)
    st.aLlist = int(case.aL)
    st.lLlist = int(case.lL)
    st.HLlist = int(case.HL)

    # Velocity
    st.vlist = float(case.v)
    st.v = int(case.v)

    # Crack length and stepping
    st.c_crack = float(case.crack_mm) * 1.0e-3
    st.stepend = int(case.stepend)
    st.stepall = int(round(st.c_crack / st.hL)) if int(st.stepend) < 0 else int(st.stepend)

    # Keep target x-domain consistent when auto-from-crack mode is on.
    if int(getattr(st, "domain_target_x_auto_from_crack", 0)) == 1:
        st.domain_target_x = 1.5 * float(st.c_crack)

    # Single job
    st.jobstart = 1
    st.jobend = 1
    st.jobnamelist = (
        f"velsweep_v{int(case.v)}_rGL{int(case.rGL)}_"
        f"aL{int(case.aL)}_lL{int(case.lL)}_HL{int(case.HL)}"
    )

    # Force H5-only FEM reference for both BC interpolation and J-integral.
    st.fem_reference_source = "h5"
    st.fem_h5_dir = _fem_h5_dir_for_case(case)

    # FEM/J-integral reference paths by naming convention (auto)
    st.fem_mat_file = "auto"
    st.jintegral_fem_mat_file = "auto"


def _write_summary(rows: Sequence[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "idx",
        "v",
        "rGL",
        "aL",
        "lL",
        "HL",
        "crack_mm",
        "stepend",
        "folder",
        "status",
        "seconds",
        "fem_kind",
        "fem_path",
        "message",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def run_cases(
    project_root: Path,
    cases: Sequence[VelocityCase],
    dry_run: bool,
    force: bool,
    fail_on_missing_fem: bool,
) -> List[Dict[str, str]]:
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []

    for i, case in enumerate(cases, start=1):
        folder = _folder_name(case)
        folder_path = results_dir / folder

        row = {
            "idx": str(i),
            "v": str(case.v),
            "rGL": str(case.rGL),
            "aL": str(case.aL),
            "lL": str(case.lL),
            "HL": str(case.HL),
            "crack_mm": str(case.crack_mm),
            "stepend": str(case.stepend),
            "folder": folder,
            "status": "",
            "seconds": "",
            "fem_kind": "",
            "fem_path": "",
            "message": "",
        }

        if folder_path.exists() and not force:
            row["status"] = "skip_existing"
            row["message"] = "result folder already exists"
            print(f"[{i:03d}/{len(cases):03d}] SKIP v={case.v} -> {folder}")
            rows.append(row)
            continue

        if dry_run:
            row["status"] = "planned"
            row["message"] = "dry-run"
            print(f"[{i:03d}/{len(cases):03d}] PLAN v={case.v} -> {folder}")
            rows.append(row)
            continue

        t0 = time.time()
        print(f"[{i:03d}/{len(cases):03d}] RUN  v={case.v} -> {folder}")

        try:
            os.chdir(project_root)
            _configure_state_for_case(case)

            fem_kind, fem_path = resolve_fem_reference_path(data_file="auto", source_preference=None)
            row["fem_kind"] = str(fem_kind)
            row["fem_path"] = str(fem_path)

            if not Path(fem_path).exists():
                msg = f"FEM reference not found: {fem_path}"
                if fail_on_missing_fem:
                    raise FileNotFoundError(msg)
                row["status"] = "skip_missing_fem"
                row["message"] = msg
                print(f"[{i:03d}/{len(cases):03d}] SKIP v={case.v}: {msg}")
                rows.append(row)
                continue

            execution()

            elapsed = time.time() - t0
            row["status"] = "done"
            row["seconds"] = f"{elapsed:.2f}"
            row["message"] = "ok"
            print(f"[{i:03d}/{len(cases):03d}] DONE v={case.v} ({elapsed:.2f}s)")
        except Exception as exc:
            elapsed = time.time() - t0
            row["status"] = "failed"
            row["seconds"] = f"{elapsed:.2f}"
            row["message"] = f"{type(exc).__name__}: {exc}"
            print(f"[{i:03d}/{len(cases):03d}] FAIL v={case.v}: {row['message']}")
            traceback.print_exc()
        finally:
            os.chdir(project_root)

        rows.append(row)

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run baseline-parameter dynamic cases for a velocity sweep."
    )
    parser.add_argument("--velocities", type=str, default=None, help="Comma list, e.g. '200,300,...'.")
    parser.add_argument("--v-start", type=int, default=200, help="Velocity start (inclusive).")
    parser.add_argument("--v-end", type=int, default=1500, help="Velocity end (inclusive).")
    parser.add_argument("--v-step", type=int, default=100, help="Velocity increment.")

    parser.add_argument("--rgl", type=int, default=6, help="Baseline rGL.")
    parser.add_argument("--al-ratio", type=float, default=2.5, help="aL ratio to rGL.")
    parser.add_argument("--ll", type=int, default=15, help="Baseline constant lL (local elements).")
    parser.add_argument("--hl-ratio", type=float, default=1.8, help="HL ratio to rGL.")

    parser.add_argument("--crack-mm", type=int, default=10, help="Crack length in mm.")
    parser.add_argument("--stepend", type=int, default=-1, help="Step end (-1 means auto by c_crack/hL).")

    parser.add_argument("--dry-run", action="store_true", help="Only show plan.")
    parser.add_argument("--force", action="store_true", help="Run even if result folder exists.")
    parser.add_argument(
        "--fail-on-missing-fem",
        action="store_true",
        help="Fail case when FEM reference is missing (default: skip).",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results") / "velocity_baseline_sweep_summary.csv",
        help="Output CSV summary path.",
    )

    args = parser.parse_args()

    if args.velocities is not None:
        velocities = _parse_velocity_list(args.velocities)
    else:
        if args.v_step <= 0:
            raise ValueError("--v-step must be positive.")
        velocities = list(range(int(args.v_start), int(args.v_end) + 1, int(args.v_step)))

    if not velocities:
        raise ValueError("No velocity cases to run.")

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    cases = _build_cases(
        velocities=velocities,
        rgl=int(args.rgl),
        aL_ratio=float(args.al_ratio),
        lL_const=int(args.ll),
        HL_ratio=float(args.hl_ratio),
        crack_mm=int(args.crack_mm),
        stepend=int(args.stepend),
    )

    print(f"Project root: {project_root}")
    print(
        "Baseline params:",
        f"rGL={args.rgl}",
        f"aL=ceil({args.al_ratio}*rGL)",
        f"lL={args.ll}",
        f"HL=ceil({args.hl_ratio}*rGL)",
        f"c_crack={args.crack_mm}mm",
        f"stepend={args.stepend}",
    )
    print(f"Velocities: {velocities}")
    print(f"Cases: {len(cases)}")

    rows = run_cases(
        project_root=project_root,
        cases=cases,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        fail_on_missing_fem=bool(args.fail_on_missing_fem),
    )

    summary = args.summary
    _write_summary(rows, summary)

    cnt_done = sum(1 for r in rows if r["status"] == "done")
    cnt_skip_exist = sum(1 for r in rows if r["status"] == "skip_existing")
    cnt_skip_fem = sum(1 for r in rows if r["status"] == "skip_missing_fem")
    cnt_failed = sum(1 for r in rows if r["status"] == "failed")
    cnt_plan = sum(1 for r in rows if r["status"] == "planned")

    print("--------------------------------------------------")
    print(f"Summary file: {summary}")
    print(
        "Counts:",
        f"done={cnt_done}",
        f"skip_existing={cnt_skip_exist}",
        f"skip_missing_fem={cnt_skip_fem}",
        f"failed={cnt_failed}",
        f"planned={cnt_plan}",
    )

    return 0 if cnt_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
