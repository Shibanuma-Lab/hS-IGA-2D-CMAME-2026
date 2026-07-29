#!/usr/bin/env python3
"""
Run the static coupling-quadrature sensitivity study.

The default campaign reproduces the exact ``fix rGL=4`` paper mesh family,

    nhL in range(10, 10000, 4),

with

    nGx = nhL / 2,
    nGy = (nGx - 1) / 2,
    aL = lL = HL = nhL / 2.

The explicit construction is intentional: it avoids deriving local element
counts from the slightly enlarged physical local-patch span, which would not
reproduce the paper mesh family at sufficiently fine resolutions.

and performs only the two additional coupling-quadrature calculations requested
for the sensitivity study:

    2 x 2 and 4 x 4.

The existing paper results are the 3 x 3 baseline and are therefore not
recomputed by default.  Passing ``--orders 3`` explicitly remains available
for an optional verification run.

The usual per-order static CSV files are written below
``static_results/quadrature_sensitivity_fix_rGL_4_paper_mesh/ngp_NxN/``.
Combined CSV files containing an additional ``gauss_order`` column, together
with an auditable ``mesh_recipe.csv``, are created in
``static_results/quadrature_sensitivity_fix_rGL_4_paper_mesh/``.

Examples
--------
Inspect the planned cases without solving:

    python3 run_static_quadrature_sweep.py --dry-run

Run both additional quadrature orders, saving only the paper metrics:

    python3 run_static_quadrature_sweep.py --metrics-only

Run the two additional orders one at a time:

    python3 run_static_quadrature_sweep.py --orders 2 --metrics-only
    python3 run_static_quadrature_sweep.py --orders 4 --metrics-only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import core.state as st
from config.static_parameters import (
    _dedupe_cases_by_hg,
    _make_case,
    _make_case_with_counts,
    _nGy_from_nGx_exact,
    _truncate_cases_by_dof,
    load_static_parameters,
)
from main import execution


PROJECT_ROOT = Path(__file__).resolve().parent
STATIC_RESULTS_DIR = PROJECT_ROOT / "static_results"
DEFAULT_OUTPUT_NAME = "quadrature_sensitivity_fix_rGL_4_paper_mesh"
FIXED_RGL = 4.0
PAPER_MESH_FAMILY = "paper_exact"
LEGACY_MESH_FAMILY = "legacy_floor_span"
MANIFEST_FILENAME = "quadrature_campaign_manifest.json"
SUMMARY_FILES = (
    "dof_l2_norm.csv",
    "normalized_sif.csv",
    "normalized_stress_yy.csv",
)


def _unique_positive_orders(values: Iterable[int]) -> list[int]:
    orders = list(dict.fromkeys(int(value) for value in values))
    if not orders or any(order < 1 for order in orders):
        raise ValueError("Every Gauss order must be a positive integer.")
    return orders


def _validate_output_name(value: str) -> str:
    """Keep all campaign output safely below ``static_results``."""
    name = str(value).strip()
    path = Path(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "--output-name must be a non-empty relative path below static_results."
        )
    return name


def _build_paper_exact_rgl4_cases(
    *,
    nhl_start: int,
    nhl_stop: int,
    nhl_step: int,
) -> list[dict]:
    """Build the published exact ``rGL=4`` mesh family from integer counts."""
    if nhl_start % 4 != 2:
        raise ValueError(
            "The paper rGL=4 family requires --nhl-start = 4k+2 "
            "(for example 10)."
        )
    if nhl_step % 4 != 0:
        raise ValueError(
            "The paper rGL=4 family requires --nhl-step to be a multiple "
            "of 4 (default: 4)."
        )

    cases: list[dict] = []
    for nhl in range(nhl_start, nhl_stop, nhl_step):
        half = int(nhl) // 2
        nGx = int(half)
        nGy = _nGy_from_nGx_exact(nGx)
        case = _make_case_with_counts(
            nhL=int(nhl),
            nGx=nGx,
            nGy=nGy,
            aL=half,
            lL=half,
            HL=half,
            rGL=FIXED_RGL,
        )
        actual_rgl = float(case["hG"]) / float(case["hL"])
        if abs(actual_rgl - FIXED_RGL) > 1.0e-12:
            raise RuntimeError(
                f"Paper mesh case nhL={nhl} has rGL={actual_rgl}, "
                f"expected {FIXED_RGL}"
            )
        case["mesh_family"] = PAPER_MESH_FAMILY
        cases.append(case)
    return cases


def _campaign_manifest(
    *,
    mesh_family: str,
    nhl_start: int,
    nhl_stop: int,
    nhl_step: int,
    dof_cap: int,
    max_cases: int,
) -> dict:
    return {
        "schema_version": 1,
        "study": "static_coupling_quadrature_sensitivity",
        "fixed_nominal_rGL": FIXED_RGL,
        "mesh_family": str(mesh_family),
        "nhL_range": [int(nhl_start), int(nhl_stop), int(nhl_step)],
        "dof_cap": int(dof_cap),
        "max_cases": int(max_cases),
    }


def _prepare_campaign_directory(output_name: str, manifest: dict) -> Path:
    """Create or validate the mesh-family manifest before calculating."""
    campaign_dir = STATIC_RESULTS_DIR / output_name
    manifest_path = campaign_dir / MANIFEST_FILENAME
    if manifest_path.is_file():
        with manifest_path.open() as stream:
            existing = json.load(stream)
        if existing != manifest:
            raise ValueError(
                f"{manifest_path.relative_to(PROJECT_ROOT)} belongs to a "
                "different mesh recipe. Use a new --output-name instead of "
                "mixing quadrature results."
            )
        return campaign_dir

    if campaign_dir.is_dir() and any(campaign_dir.iterdir()):
        raise ValueError(
            f"{campaign_dir.relative_to(PROJECT_ROOT)} already contains "
            "results without a mesh-family manifest. Use a new --output-name "
            "to preserve those results."
        )

    campaign_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
    return campaign_dir


def _write_mesh_recipe(output_name: str, cases: Sequence[dict]) -> None:
    """Write the planned integer mesh counts once for auditability."""
    destination = STATIC_RESULTS_DIR / output_name / "mesh_recipe.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "case_name",
                "mesh_family",
                "hG",
                "hL",
                "rGL",
                "nGx",
                "nGy",
                "nhL",
                "aL",
                "lL",
                "HL",
                "dof_estimate",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    f"nGx_{int(case['nGx'])}_aL_{int(case['aL'])}",
                    case.get("mesh_family", LEGACY_MESH_FAMILY),
                    case["hG"],
                    case["hL"],
                    case["rGL"],
                    case["nGx"],
                    case["nGy"],
                    case["nhL"],
                    case["aL"],
                    case["lL"],
                    case["HL"],
                    case["dof_estimate"],
                ]
            )


def _build_fix_rgl_cases(
    *,
    mesh_family: str,
    nhl_start: int,
    nhl_stop: int,
    nhl_step: int,
    dof_cap: int,
    max_cases: int,
) -> list[dict]:
    if nhl_start <= 0:
        raise ValueError("--nhl-start must be positive.")
    if nhl_stop <= nhl_start:
        raise ValueError("--nhl-stop must be greater than --nhl-start.")
    if nhl_step <= 0:
        raise ValueError("--nhl-step must be positive.")
    if dof_cap <= 0:
        raise ValueError("--dof-cap must be positive.")
    if max_cases < 0:
        raise ValueError("--max-cases cannot be negative.")

    if mesh_family == PAPER_MESH_FAMILY:
        deduped = _build_paper_exact_rgl4_cases(
            nhl_start=nhl_start,
            nhl_stop=nhl_stop,
            nhl_step=nhl_step,
        )
    elif mesh_family == LEGACY_MESH_FAMILY:
        raw_cases = [
            _make_case(nhL=nhl, rGL=FIXED_RGL)
            for nhl in range(nhl_start, nhl_stop, nhl_step)
        ]
        deduped = _dedupe_cases_by_hg(raw_cases, target_rgl=FIXED_RGL)
        for case in deduped:
            case["mesh_family"] = LEGACY_MESH_FAMILY
    else:
        raise ValueError(f"Unsupported mesh family: {mesh_family}")
    cases = _truncate_cases_by_dof(
        deduped,
        max_dof=dof_cap,
        p=int(st.p),
        q=int(st.q),
        ndof=2,
    )
    if max_cases > 0:
        cases = cases[:max_cases]
    return cases


def _configure_order(
    *,
    order: int,
    output_name: str,
    mesh_family: str,
    nhl_start: int,
    nhl_stop: int,
    nhl_step: int,
    dof_cap: int,
    max_cases: int,
    metrics_only: bool,
) -> list[dict]:
    # Reset all static defaults before preparing each independent batch.
    load_static_parameters(sweep_mode="fix_rGL")

    cases = _build_fix_rgl_cases(
        mesh_family=mesh_family,
        nhl_start=nhl_start,
        nhl_stop=nhl_stop,
        nhl_step=nhl_step,
        dof_cap=dof_cap,
        max_cases=max_cases,
    )

    # Both values are set so the generic GL tables and the static KGL assembly
    # use the same order.  ngpG and ngpL are intentionally left unchanged.
    st.ngpGL = int(order)
    st.static_kgl_ngpGL = int(order)

    st.static_sweep_mode = "custom"
    st.static_parallel_jobs = 1
    st.static_max_dof = int(dof_cap)
    st.static_parent_label = f"{output_name}/ngp_{order}x{order}"
    st.static_cases = cases
    st.jobstart = 1
    st.jobend = len(cases)
    st.jobnamelist = st.static_parent_label

    # Metrics are computed and written independently of savedata().  This mode
    # avoids large DAT/VTU output while retaining all three requested CSVs.
    if metrics_only:
        st.issave = 0
        st.save_vtu = 0

    return cases


def _print_batch(order: int, cases: Sequence[dict], dof_cap: int) -> None:
    if not cases:
        print(f"[SKIP] {order}x{order}: no case satisfies DOF cap={dof_cap}")
        return

    first = cases[0]
    last = cases[-1]
    print(
        f"[BATCH] coupling quadrature={order}x{order}, cases={len(cases)}, "
        f"mesh_family={first.get('mesh_family', LEGACY_MESH_FAMILY)}, "
        f"first(nhL={first['nhL']}, nGx={first['nGx']}, "
        f"dof={first['dof_estimate']}), "
        f"last(nhL={last['nhL']}, nGx={last['nGx']}, "
        f"dof={last['dof_estimate']})"
    )


def _output_dir(output_name: str, order: int) -> Path:
    return STATIC_RESULTS_DIR / output_name / f"ngp_{order}x{order}"


def _has_existing_results(output_dir: Path) -> bool:
    if not output_dir.is_dir():
        return False
    return any(output_dir.iterdir())


def _discover_complete_orders(output_name: str) -> list[int]:
    """Find every completed ``ngp_NxN`` batch from current or earlier runs."""
    campaign_dir = STATIC_RESULTS_DIR / output_name
    if not campaign_dir.is_dir():
        return []

    orders: list[int] = []
    for child in campaign_dir.iterdir():
        match = re.fullmatch(r"ngp_(\d+)x(\d+)", child.name)
        if match is None or match.group(1) != match.group(2):
            continue
        if all((child / filename).is_file() for filename in SUMMARY_FILES):
            orders.append(int(match.group(1)))
    return sorted(set(orders))


def _write_combined_summaries(output_name: str, orders: Sequence[int]) -> None:
    """Combine available standard summaries and add the swept Gauss order."""
    campaign_dir = STATIC_RESULTS_DIR / output_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    for filename in SUMMARY_FILES:
        combined_header: list[str] | None = None
        combined_rows: list[list[str]] = []

        for order in orders:
            source = _output_dir(output_name, order) / filename
            if not source.is_file():
                continue

            with source.open(newline="") as stream:
                reader = csv.reader(stream)
                try:
                    header = next(reader)
                except StopIteration:
                    continue

                if combined_header is None:
                    combined_header = ["gauss_order", "quadrature"] + header
                elif combined_header[2:] != header:
                    raise ValueError(
                        f"Incompatible columns while combining {source}: {header}"
                    )

                for row in reader:
                    combined_rows.append([str(order), f"{order}x{order}", *row])

        if combined_header is None:
            continue

        destination = campaign_dir / filename.replace(
            ".csv", "_all_orders.csv"
        )
        with destination.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(combined_header)
            writer.writerows(combined_rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the coupling-matrix Gauss order for the static nominal "
            "fix-rGL=4 cases."
        )
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[2, 4],
        help=(
            "Gauss points per parametric direction (default: 2 4; "
            "the existing paper results provide the 3x3 baseline)."
        ),
    )
    parser.add_argument(
        "--dof-cap",
        type=int,
        default=100_000,
        help="Maximum estimated total DOF per case (default: 100000).",
    )
    parser.add_argument(
        "--mesh-family",
        choices=(PAPER_MESH_FAMILY, LEGACY_MESH_FAMILY),
        default=PAPER_MESH_FAMILY,
        help=(
            "Mesh-count rule: paper_exact reproduces the published "
            "nhL=10,14,18,... sequence (default); legacy_floor_span "
            "reproduces the earlier nhL-to-count floor rule."
        ),
    )
    parser.add_argument(
        "--nhl-start",
        type=int,
        default=10,
        help="Start of the nhL range, inclusive (default: 10).",
    )
    parser.add_argument(
        "--nhl-stop",
        type=int,
        default=10_000,
        help="Stop of the nhL range, exclusive (default: 10000).",
    )
    parser.add_argument(
        "--nhl-step",
        type=int,
        default=4,
        help="Step of the nhL range (default: 4).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Keep only the first N cases per order; 0 keeps all (default: 0).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help=(
            "Relative campaign directory below static_results "
            f"(default: {DEFAULT_OUTPUT_NAME})."
        ),
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Save the requested CSV metrics but skip per-step DAT and VTU files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the batches without running the solver or writing output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute an order even when its output directory is non-empty.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        orders = _unique_positive_orders(args.orders)
        output_name = _validate_output_name(args.output_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Static mesh family: {args.mesh_family}, nominal fix rGL=4")
    print("Existing paper baseline: coupling quadrature=3x3 (not recomputed by default)")
    print(
        "nhL range: "
        f"range({args.nhl_start}, {args.nhl_stop}, {args.nhl_step})"
    )
    print(f"Coupling quadrature orders: {orders}")
    print(f"DOF cap: {args.dof_cap}")
    print(f"Output: static_results/{output_name}")
    print(f"Metrics only: {bool(args.metrics_only)}")

    if not args.dry_run:
        try:
            _prepare_campaign_directory(
                output_name,
                _campaign_manifest(
                    mesh_family=str(args.mesh_family),
                    nhl_start=int(args.nhl_start),
                    nhl_stop=int(args.nhl_stop),
                    nhl_step=int(args.nhl_step),
                    dof_cap=int(args.dof_cap),
                    max_cases=int(args.max_cases),
                ),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    recipe_written = False
    for order in orders:
        try:
            cases = _configure_order(
                order=order,
                output_name=output_name,
                mesh_family=str(args.mesh_family),
                nhl_start=int(args.nhl_start),
                nhl_stop=int(args.nhl_stop),
                nhl_step=int(args.nhl_step),
                dof_cap=int(args.dof_cap),
                max_cases=int(args.max_cases),
                metrics_only=bool(args.metrics_only),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        _print_batch(order, cases, int(args.dof_cap))
        if not args.dry_run and not recipe_written:
            _write_mesh_recipe(output_name, cases)
            recipe_written = True
        if args.dry_run or not cases:
            continue

        order_dir = _output_dir(output_name, order)
        if _has_existing_results(order_dir) and not args.force:
            print(
                f"[SKIP] {order}x{order}: {order_dir.relative_to(PROJECT_ROOT)} "
                "is non-empty (use --force to recompute this order)."
            )
            continue

        print(
            f"[RUN] {order}x{order} -> "
            f"{order_dir.relative_to(PROJECT_ROOT)}"
        )
        execution()
        available_orders = _discover_complete_orders(output_name)
        _write_combined_summaries(output_name, available_orders)
        print(f"[DONE] coupling quadrature={order}x{order}")

    if not args.dry_run:
        available_orders = _discover_complete_orders(output_name)
        if available_orders:
            _write_combined_summaries(output_name, available_orders)
        print(
            "[DONE] Quadrature-sensitivity campaign finished. "
            f"Available complete orders: {available_orders}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
