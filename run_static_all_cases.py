#!/usr/bin/env python3
"""
Run all requested static sweeps and special static cases in one command.

Requested campaign:
1) fix rGL: rGL in [2, 4, 6, 8], nhL in range(5, 10000, 4)
2) fix hG : nGx in [21, 41, 61, 81], nhL in range(5, 10000, 4)
3) fix hL : nhL in [20, 40, 80, 160], nGx = 3,5,7,... until rGL<2
4) special cases:
   a) nominal fix rGL=4 with (nGx, nhL) = (11,20), (21,40), (41,80)
   b) fix hG=2/21 with nhL = [20, 40, 80]
   c) fix hL=1/80 with nGx = [11, 21, 41]
"""

from __future__ import annotations

import argparse
from typing import Iterable, List, Sequence

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


def _ensure_static_defaults():
    if getattr(st, "static_local_half_span", None) is None:
        load_static_parameters(sweep_mode="fix_rGL")


def _fmt_val(v: float) -> str:
    return f"{float(v):.10g}"


def _is_valid_global_divisions(nGx: int, nGy: int) -> bool:
    """
    Minimal validity for global IGA knot construction:
      nPtsX - p = nGx > 0, nPtsY - q = nGy > 0
    """
    return int(nGx) > 0 and int(nGy) > 0


def _make_case_with_fixed_ngx(
    nhL: int,
    nGx_raw: int,
    *,
    nominal_rgl: float | None = None,
    exact_local_counts: bool = False,
):
    """
    Build one case from nhL and a fixed global x-element count.
    """
    _ensure_static_defaults()
    base = _make_case(nhL=int(nhL), rGL=1.0)
    nGx = int(nGx_raw)
    nGy = _nGy_from_nGx_exact(nGx)
    hG = float(st.static_width) / float(nGx)

    if exact_local_counts:
        half = int(nhL) // 2
        aL = half
        lL = half
        HL = half
    else:
        aL = int(base["aL"])
        lL = int(base["lL"])
        HL = int(base["HL"])

    if nominal_rgl is None:
        rGL = float(hG / float(base["hL"]))
    else:
        rGL = float(nominal_rgl)

    return _make_case_with_counts(
        nhL=int(nhL),
        nGx=int(nGx),
        nGy=int(nGy),
        aL=int(aL),
        lL=int(lL),
        HL=int(HL),
        rGL=float(rGL),
    )


def _prep_batch(parent_label: str, cases: Sequence[dict], max_dof: int) -> List[dict]:
    """
    Set static defaults and prepare one custom batch.
    """
    # Reinitialize all static defaults for each batch.
    load_static_parameters(sweep_mode="fix_rGL")

    st.static_sweep_mode = "custom"
    st.static_parallel_jobs = 1
    st.static_parent_label = str(parent_label)
    st.static_max_dof = int(max_dof)
    st.static_kgl_ngpGL = 5 if str(parent_label).startswith("fix_hG_") else 3

    trimmed = _truncate_cases_by_dof(
        list(cases),
        max_dof=int(max_dof),
        p=int(st.p),
        q=int(st.q),
        ndof=2,
    )

    st.static_cases = trimmed
    st.jobstart = 1
    st.jobend = len(trimmed)
    st.jobnamelist = st.static_parent_label
    return trimmed


def _run_batch(parent_label: str, cases: Sequence[dict], max_dof: int, dry_run: bool):
    trimmed = _prep_batch(parent_label=parent_label, cases=cases, max_dof=max_dof)
    if len(trimmed) == 0:
        print(f"[SKIP] {parent_label}: no case under DOF cap={max_dof}")
        return

    first = trimmed[0]
    last = trimmed[-1]
    print(
        f"[BATCH] {parent_label}: cases={len(trimmed)}, "
        f"first(nGx={first['nGx']}, nGy={first['nGy']}, aL/lL/HL={first['aL']}/{first['lL']}/{first['HL']}, "
        f"hG={_fmt_val(first['hG'])}, hL={_fmt_val(first['hL'])}, dof={first['dof_estimate']}), "
        f"last(nGx={last['nGx']}, nGy={last['nGy']}, aL/lL/HL={last['aL']}/{last['lL']}/{last['HL']}, "
        f"hG={_fmt_val(last['hG'])}, hL={_fmt_val(last['hL'])}, dof={last['dof_estimate']})"
    )
    if dry_run:
        return
    execution()


def _build_fix_rgl_cases(rgl: int, nhL_values: Iterable[int]) -> List[dict]:
    _ensure_static_defaults()
    raw = [_make_case(nhL=int(nhL), rGL=float(rgl)) for nhL in nhL_values]
    return _dedupe_cases_by_hg(raw, target_rgl=float(rgl))


def _build_fix_hg_cases(nGx: int, nhL_values: Iterable[int]) -> List[dict]:
    _ensure_static_defaults()
    return [
        _make_case_with_fixed_ngx(int(nhL), int(nGx))
        for nhL in nhL_values
    ]


def _build_fix_hl_cases(nhL: int) -> List[dict]:
    _ensure_static_defaults()
    out = []
    half = int(nhL) // 2
    hL_fixed = 1.0 / float(int(nhL))
    width = float(st.static_width)
    nGx = 3
    while True:
        nGy = _nGy_from_nGx_exact(nGx)
        if not _is_valid_global_divisions(nGx, nGy):
            print(
                f"[INFO] fix_hL nhL={int(nhL)} stops at nGx={int(nGx)} "
                f"(invalid global divisions: nGx={nGx}, nGy={nGy})."
            )
            break
        hG = width / float(nGx)
        rGL = float(hG / hL_fixed)
        if rGL < 2.0:
            print(
                f"[INFO] fix_hL nhL={int(nhL)} stops at nGx={int(nGx)} "
                f"(rGL={rGL:.6f} < 2.0)."
            )
            break
        out.append(
            _make_case_with_counts(
                nhL=int(nhL),
                nGx=nGx,
                nGy=nGy,
                aL=int(half),
                lL=int(half),
                HL=int(half),
                rGL=rGL,
            )
        )
        nGx += 2
    return out


def main():
    parser = argparse.ArgumentParser(description="Run all static sweep/special campaigns.")
    parser.add_argument(
        "--dof-cap",
        type=int,
        default=int(1.0e5),
        help="Maximum DOF cap per batch (default: 100000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print batch/case counts after DOF truncation, do not run solver.",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        choices=["fix_rGL", "fix_hG", "fix_hL", "special"],
        default="fix_rGL",
        help="Start campaign from this group (default: fix_rGL).",
    )
    args = parser.parse_args()

    dof_cap = int(args.dof_cap)
    dry_run = bool(args.dry_run)
    start_from = str(args.start_from)

    nhL_sweep = range(10, 10000, 4)
    group_order = {"fix_rGL": 0, "fix_hG": 1, "fix_hL": 2, "special": 3}
    start_rank = group_order[start_from]

    # 1) fix rGL groups
    if start_rank <= group_order["fix_rGL"]:
        for rgl in (2, 4, 6, 8):
            label = f"fix_rGL_{int(rgl)}"
            cases = _build_fix_rgl_cases(rgl=int(rgl), nhL_values=nhL_sweep)
            _run_batch(label, cases, max_dof=dof_cap, dry_run=dry_run)

    # 2) fix hG groups (by nGx list)
    if start_rank <= group_order["fix_hG"]:
        for nGx in (21, 41, 61, 81):
            hG = 2.0 / float(nGx)
            label = f"fix_hG_{_fmt_val(hG)}"
            cases = _build_fix_hg_cases(nGx=int(nGx), nhL_values=nhL_sweep)
            _run_batch(label, cases, max_dof=dof_cap, dry_run=dry_run)

    # 3) fix hL groups (by nhL list)
    if start_rank <= group_order["fix_hL"]:
        for nhL in (20, 40, 80, 160):
            hL = 1.0 / float(nhL)
            label = f"fix_hL_{_fmt_val(hL)}"
            cases = _build_fix_hl_cases(nhL=int(nhL))
            _run_batch(label, cases, max_dof=dof_cap, dry_run=dry_run)

    # 4a) special: nominal fix rGL=4, three explicit pairs
    if start_rank <= group_order["special"]:
        special_a = [
            _make_case_with_fixed_ngx(20, 11, nominal_rgl=4.0, exact_local_counts=True),
            _make_case_with_fixed_ngx(40, 21, nominal_rgl=4.0, exact_local_counts=True),
            _make_case_with_fixed_ngx(80, 41, nominal_rgl=4.0, exact_local_counts=True),
        ]
        _run_batch("special_fix_rGL_4", special_a, max_dof=dof_cap, dry_run=dry_run)

        # 4b) special: fix hG = 2/21, hL list = [1/20, 1/40, 1/80]
        special_b = [
            _make_case_with_fixed_ngx(20, 21, exact_local_counts=True),
            _make_case_with_fixed_ngx(40, 21, exact_local_counts=True),
            _make_case_with_fixed_ngx(80, 21, exact_local_counts=True),
        ]
        _run_batch("special_fix_hG_2_over_21", special_b, max_dof=dof_cap, dry_run=dry_run)

        # 4c) special: fix hL = 1/80, hG list = [2/11, 2/21, 2/41]
        special_c = [
            _make_case_with_fixed_ngx(80, 11, exact_local_counts=True),
            _make_case_with_fixed_ngx(80, 21, exact_local_counts=True),
            _make_case_with_fixed_ngx(80, 41, exact_local_counts=True),
        ]
        _run_batch("special_fix_hL_1_over_80", special_c, max_dof=dof_cap, dry_run=dry_run)

    print("[DONE] All requested static campaigns completed.")


if __name__ == "__main__":
    main()
