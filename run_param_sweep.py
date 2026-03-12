#!/usr/bin/env python3
"""
Automated parameter sweep runner for hS-IGA-2D.

Sweep groups (sequential sensitivity checks):
1) rGL sweep
2) aL sweep
3) lL sweep
4) HL sweep

Rule:
- Skip a case when its result folder already exists under ``results/``.
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
class SweepCase:
    group: str
    label: str
    rGL: int
    aL: int
    lL: int
    HL: int


def _ceil_mul(value: int, factor: float) -> int:
    return int(math.ceil(float(value) * float(factor)))


def _folder_name(v: float, case: SweepCase) -> str:
    return f"v_{int(v)}_rGL_{case.rGL}_aL_{case.aL}_lL_{case.lL}_HL_{case.HL}"


def _fem_h5_dir_for_case(v: float, crack_mm: int) -> str:
    """FEM H5 directory naming for parameter sweep cases."""
    return f"FEM_data/h5_export_V_{int(v)}_a_{int(crack_mm)}"


def _rgl_sweep_lL(rgl: int) -> int:
    """
    lL rule for rGL-sweep cases.

    Keep the ratio-based rule by default, but force rGL=2/3 to lL=5 to
    ensure enough local nodes ahead of crack tip for stress-error extraction.
    """
    if int(rgl) in (2, 3):
        return 5
    return _ceil_mul(int(rgl), 1.2)


def build_sweep_cases(base_rGL: int = 6, ll_values: Sequence[int] | None = None) -> List[SweepCase]:
    """
    Build the 4 sweep groups in order.
    """
    rgl_values = [2, 3, 4, 5, 6, 8, 10]
    aL_factors = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    # lL is controlled by ratio to rGL (same style as aL).
    lL_factors = [0.8, 1.0, 1.2, 1.4, 1.8, 2.2]
    HL_factors = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4]

    all_cases: List[SweepCase] = []

    # 1) rGL sweep
    for r in rgl_values:
        all_cases.append(
            SweepCase(
                group="rGL",
                label=f"rGL={r}",
                rGL=int(r),
                aL=_ceil_mul(r, 2.5),
                lL=_rgl_sweep_lL(r),
                HL=_ceil_mul(r, 1.8),
            )
        )

    # Fixed baseline for other groups
    base_aL = _ceil_mul(base_rGL, 2.5)
    base_lL = _ceil_mul(base_rGL, 1.2)
    base_HL = _ceil_mul(base_rGL, 1.8)

    # 2) aL sweep
    for f in aL_factors:
        all_cases.append(
            SweepCase(
                group="aL",
                label=f"aL=ceil({f:.1f}*rGL)",
                rGL=int(base_rGL),
                aL=_ceil_mul(base_rGL, f),
                lL=base_lL,
                HL=base_HL,
            )
        )

    # 3) lL sweep
    if ll_values is not None:
        # Optional absolute override from CLI (kept for targeted reruns).
        iter_ll = [(f"lL={int(ll)}", int(ll)) for ll in ll_values]
    else:
        iter_ll = [
            (f"lL=ceil({f:.1f}*rGL)", _ceil_mul(base_rGL, f))
            for f in lL_factors
        ]

    for label, ll in iter_ll:
        all_cases.append(
            SweepCase(
                group="lL",
                label=label,
                rGL=int(base_rGL),
                aL=base_aL,
                lL=int(ll),
                HL=base_HL,
            )
        )

    # 4) HL sweep
    for f in HL_factors:
        all_cases.append(
            SweepCase(
                group="HL",
                label=f"HL=ceil({f:.1f}*rGL)",
                rGL=int(base_rGL),
                aL=base_aL,
                lL=base_lL,
                HL=_ceil_mul(base_rGL, f),
            )
        )

    return all_cases


def _configure_state_for_case(v: float, case: SweepCase) -> None:
    """
    Reset parameters and set one single-case job configuration.
    """
    load_parameters(rGL_value=case.rGL)

    st.vlist = float(v)
    st.v = int(float(v))

    st.rGLlist = int(case.rGL)
    st.aLlist = int(case.aL)
    st.lLlist = int(case.lL)
    st.HLlist = int(case.HL)

    st.jobstart = 1
    st.jobend = 1
    st.jobnamelist = (
        f"sweep_v{int(v)}_rGL{case.rGL}_aL{case.aL}_lL{case.lL}_HL{case.HL}"
    )

    # Force H5-only FEM reference for both BC interpolation and J-integral.
    crack_mm = int(round(float(st.c_crack) * 1000.0))
    st.fem_reference_source = "h5"
    st.fem_h5_dir = _fem_h5_dir_for_case(v=v, crack_mm=crack_mm)

    # Keep unified automatic FEM source resolution entry points.
    st.fem_mat_file = "auto"
    st.jintegral_fem_mat_file = "auto"


def run_sweep(
    project_root: Path,
    v: float,
    cases: Sequence[SweepCase],
    dry_run: bool = False,
    force: bool = False,
) -> List[Dict[str, str]]:
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []

    for i, case in enumerate(cases, start=1):
        folder = _folder_name(v, case)
        folder_path = results_dir / folder
        key = f"{case.group}:{case.label}"

        row = {
            "idx": str(i),
            "group": case.group,
            "label": case.label,
            "v": str(int(v)),
            "rGL": str(case.rGL),
            "aL": str(case.aL),
            "lL": str(case.lL),
            "HL": str(case.HL),
            "folder": folder,
            "status": "",
            "seconds": "",
            "message": "",
        }

        if folder_path.exists() and not force:
            row["status"] = "skip_existing"
            row["message"] = "result folder already exists"
            print(f"[{i:03d}/{len(cases):03d}] SKIP {key} -> {folder}")
            rows.append(row)
            continue

        if dry_run:
            row["status"] = "planned"
            row["message"] = "dry-run"
            print(f"[{i:03d}/{len(cases):03d}] PLAN {key} -> {folder}")
            rows.append(row)
            continue

        print(f"[{i:03d}/{len(cases):03d}] RUN  {key} -> {folder}")
        t0 = time.time()
        try:
            os.chdir(project_root)
            _configure_state_for_case(v=v, case=case)

            # Validate FEM reference source before starting expensive solve.
            _, fem_ref_path = resolve_fem_reference_path("auto")
            if not fem_ref_path.exists():
                raise FileNotFoundError(
                    f"FEM reference source not found for this case: {fem_ref_path}"
                )

            execution()
            elapsed = time.time() - t0
            row["status"] = "done"
            row["seconds"] = f"{elapsed:.2f}"
            row["message"] = "ok"
            print(f"[{i:03d}/{len(cases):03d}] DONE {key} ({elapsed:.2f}s)")
        except Exception as exc:
            elapsed = time.time() - t0
            row["status"] = "failed"
            row["seconds"] = f"{elapsed:.2f}"
            row["message"] = f"{type(exc).__name__}: {exc}"
            print(f"[{i:03d}/{len(cases):03d}] FAIL {key}: {row['message']}")
            traceback.print_exc()
        finally:
            os.chdir(project_root)

        rows.append(row)

    return rows


def write_summary_csv(rows: Sequence[Dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "idx",
        "group",
        "label",
        "v",
        "rGL",
        "aL",
        "lL",
        "HL",
        "folder",
        "status",
        "seconds",
        "message",
    ]
    with open(output_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _parse_groups(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return ["rGL", "aL", "lL", "HL"]
    groups = [x.strip() for x in raw.split(",") if x.strip()]
    valid = {"rGL", "aL", "lL", "HL"}
    for g in groups:
        if g not in valid:
            raise ValueError(f"Unknown group '{g}', valid groups: {sorted(valid)}")
    return groups


def _parse_int_list(raw: str) -> List[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("List cannot be empty.")
    out = [int(v) for v in vals]
    if any(v <= 0 for v in out):
        raise ValueError("All list values must be positive integers.")
    # Preserve input order while removing duplicates.
    return list(dict.fromkeys(out))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run automated parameter-condition sweeps.")
    parser.add_argument("--v", type=float, default=500.0, help="Crack velocity value for this sweep.")
    parser.add_argument("--base-rgl", type=int, default=6, help="Baseline rGL for non-rGL groups.")
    parser.add_argument(
        "--groups",
        type=str,
        default="all",
        help="Groups to run: all or comma list from {rGL,aL,lL,HL}.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned/skip actions.")
    parser.add_argument("--force", action="store_true", help="Run even if result folder exists.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Run only first N cases after filtering (0 means all).",
    )
    parser.add_argument(
        "--ll-values",
        type=str,
        default=None,
        help="Override lL sweep values as comma list, e.g. '4,6,8,9'.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    groups = _parse_groups(args.groups)
    ll_values = _parse_int_list(args.ll_values) if args.ll_values is not None else None
    all_cases = build_sweep_cases(base_rGL=int(args.base_rgl), ll_values=ll_values)
    selected = [c for c in all_cases if c.group in set(groups)]
    if args.max_cases > 0:
        selected = selected[: int(args.max_cases)]

    print(f"Project root: {project_root}")
    print(f"Velocity v: {int(args.v)}")
    print(f"Groups: {groups}")
    print(f"Cases generated: {len(all_cases)}")
    print(f"Cases selected: {len(selected)}")

    rows = run_sweep(
        project_root=project_root,
        v=float(args.v),
        cases=selected,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
    )

    summary = project_root / "results" / f"param_sweep_v{int(args.v)}_summary.csv"
    write_summary_csv(rows, summary)

    cnt_done = sum(1 for r in rows if r["status"] == "done")
    cnt_skip = sum(1 for r in rows if r["status"] == "skip_existing")
    cnt_fail = sum(1 for r in rows if r["status"] == "failed")
    cnt_plan = sum(1 for r in rows if r["status"] == "planned")
    print("--------------------------------------------------")
    print(f"Summary file: {summary}")
    print(
        f"Counts: done={cnt_done}, skip_existing={cnt_skip}, "
        f"failed={cnt_fail}, planned={cnt_plan}"
    )
    return 0 if cnt_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
