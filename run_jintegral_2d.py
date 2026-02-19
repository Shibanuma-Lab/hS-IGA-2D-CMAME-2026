#!/usr/bin/env python3
"""
Offline runner for J-integral / DSIF post-processing.

Usage examples:
  python3 run_jintegral_2d.py
  python3 run_jintegral_2d.py --result-dir results/v_400_rGL_2_aL_5_lL_15_HL_4
  python3 run_jintegral_2d.py --result-dir ... --step-start 50 --step-end 80
  python3 run_jintegral_2d.py --result-dir ... --Rj0 1.5 --sweep-rj1 "2.01*Rj0,3.01*Rj0,4.01*Rj0"
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import core.state as st
from config.parameters import load_parameters
from core.jobset import jobset
from postprocess.jintegral_2d import calculate_jintegral_2d


def _infer_meta_from_result_dir(result_dir: Path) -> None:
    """
    Infer key parameters from result folder name:
    v_<v>_rGL_<rGL>_aL_<aL>_lL_<lL>_HL_<HL>
    """
    pat = re.compile(
        r"^v_(?P<v>\d+)_rGL_(?P<rgl>\d+)_aL_(?P<aL>\d+)_lL_(?P<lL>\d+)_HL_(?P<HL>\d+)$"
    )
    m = pat.match(result_dir.name)
    if m is None:
        return

    st.v = int(m.group("v"))
    st.rGL = int(m.group("rgl"))
    st.aL = int(m.group("aL"))
    st.lL = int(m.group("lL"))
    st.HL = int(m.group("HL"))
    st.nLr = int(st.aL + st.lL)


def _safe_eval_numeric_expr(expr: str, names: Dict[str, float]) -> float:
    """Evaluate a simple numeric expression safely (no function calls)."""
    node = ast.parse(expr, mode="eval")

    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError(f"Unsupported constant in expression: {expr}")
        if isinstance(n, ast.Name):
            if n.id in names:
                return float(names[n.id])
            raise ValueError(f"Unknown name '{n.id}' in expression: {expr}")
        if isinstance(n, ast.UnaryOp):
            v = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +v
            if isinstance(n.op, ast.USub):
                return -v
            raise ValueError(f"Unsupported unary operator in expression: {expr}")
        if isinstance(n, ast.BinOp):
            a = _eval(n.left)
            b = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Pow):
                return a ** b
            raise ValueError(f"Unsupported binary operator in expression: {expr}")
        raise ValueError(f"Unsupported expression: {expr}")

    out = float(_eval(node))
    if not np.isfinite(out):
        raise ValueError(f"Expression is not finite: {expr}")
    return out


def _parse_sweep_rj1(spec: str, Rj0: float) -> List[float]:
    """
    Parse comma-separated Rj1 expressions.
    Example:
      "2.01*1.5,3.01*1.5,4.01*Rj0"
    """
    values: List[float] = []
    for token in spec.split(","):
        expr = token.strip()
        if expr == "":
            continue
        val = _safe_eval_numeric_expr(expr, {"Rj0": float(Rj0)})
        values.append(float(val))

    if len(values) == 0:
        raise ValueError("--sweep-rj1 is empty. Provide comma-separated values/expressions.")

    deduped: List[float] = []
    seen = set()
    for v in values:
        key = round(v, 14)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


def _slug_float(x: float) -> str:
    s = f"{float(x):.8f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _write_sweep_files(
    result_dir: Path,
    v: int,
    rgl: int,
    cases: List[Dict[str, object]],
) -> Dict[str, Path]:
    combined = result_dir / f"J_integral_2D_v{v}_rGL{rgl}_Rj1_sweep_all.csv"
    summary = result_dir / f"J_integral_2D_v{v}_rGL{rgl}_Rj1_sweep_summary.csv"

    # Wide-long combined output: one row per (Rj1, step)
    with open(combined, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rj0", "Rj1", "Step", "J_total", "J_static", "J_dynamic", "K_I"])
        for c in cases:
            rj0 = float(c["Rj0"])
            rj1 = float(c["Rj1"])
            rows = c["rows"]
            for row in rows:
                writer.writerow([rj0, rj1, row["step"], row["J"], row["J_static"], row["J_dynamic"], row["K_I"]])

    # Path-independence summary relative to first case (reference contour)
    ref = cases[0]
    ref_by_step = {int(r["step"]): r for r in ref["rows"]}

    with open(summary, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Rj0",
                "Rj1",
                "n_steps",
                "J_mean",
                "J_min",
                "J_max",
                "K_mean",
                "K_min",
                "K_max",
                "J_rel_mean_vs_ref",
                "J_rel_max_vs_ref",
                "K_rel_mean_vs_ref",
                "K_rel_max_vs_ref",
            ]
        )

        for c in cases:
            rows = c["rows"]
            J = np.array([float(r["J"]) for r in rows], dtype=float)
            K = np.array([float(r["K_I"]) for r in rows], dtype=float)

            row_by_step = {int(r["step"]): r for r in rows}
            common_steps = sorted(set(ref_by_step.keys()) & set(row_by_step.keys()))

            if len(common_steps) == 0:
                j_rel_mean = np.nan
                j_rel_max = np.nan
                k_rel_mean = np.nan
                k_rel_max = np.nan
            else:
                eps = 1e-14
                j_rel = []
                k_rel = []
                for s in common_steps:
                    j_ref = float(ref_by_step[s]["J"])
                    j_now = float(row_by_step[s]["J"])
                    k_ref = float(ref_by_step[s]["K_I"])
                    k_now = float(row_by_step[s]["K_I"])
                    j_rel.append(abs(j_now - j_ref) / max(abs(j_ref), eps))
                    k_rel.append(abs(k_now - k_ref) / max(abs(k_ref), eps))
                j_rel = np.array(j_rel, dtype=float)
                k_rel = np.array(k_rel, dtype=float)
                j_rel_mean = float(np.mean(j_rel))
                j_rel_max = float(np.max(j_rel))
                k_rel_mean = float(np.mean(k_rel))
                k_rel_max = float(np.max(k_rel))

            writer.writerow(
                [
                    float(c["Rj0"]),
                    float(c["Rj1"]),
                    len(rows),
                    float(np.mean(J)),
                    float(np.min(J)),
                    float(np.max(J)),
                    float(np.mean(K)),
                    float(np.min(K)),
                    float(np.max(K)),
                    j_rel_mean,
                    j_rel_max,
                    k_rel_mean,
                    k_rel_max,
                ]
            )

    return {"combined": combined, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate J-integral / DSIF from saved results.")
    parser.add_argument("--result-dir", type=Path, default=None, help="Result root directory")
    parser.add_argument("--rgl", type=int, default=2, help="rGL value used for default path init")
    parser.add_argument("--step-start", type=int, default=2, help="Start step")
    parser.add_argument("--step-end", type=int, default=None, help="End step (default: auto)")
    parser.add_argument("--Rj0", type=float, default=None, help="Inner radial weight parameter")
    parser.add_argument("--Rj1", type=float, default=None, help="Outer radial weight parameter")
    parser.add_argument(
        "--sweep-rj1",
        type=str,
        default=None,
        help=(
            "Comma-separated Rj1 list/expressions, e.g. "
            "'2.01*1.5,3.01*1.5,4.01*Rj0'. "
            "When set, batch mode runs all contours and writes comparison CSVs."
        ),
    )
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--no-extend", action="store_true", help="Disable symmetric mesh extension")
    args = parser.parse_args()

    # Initialize default parameters and default result path format
    load_parameters(rGL_value=args.rgl)
    jobset(1)

    result_dir = args.result_dir.resolve() if args.result_dir is not None else Path(st.dirname)
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    _infer_meta_from_result_dir(result_dir)
    st.dirname = result_dir

    Rj0 = float(st.jintegral_Rj0 if args.Rj0 is None else args.Rj0)

    if args.sweep_rj1 is None:
        Rj1 = float(st.jintegral_Rj1 if args.Rj1 is None else args.Rj1)
        if Rj1 <= Rj0:
            raise ValueError(f"Invalid contour: Rj1 ({Rj1}) must be > Rj0 ({Rj0}).")

        output = args.output
        if output is None:
            output = result_dir / f"J_integral_2D_v{int(st.v)}_rGL{int(st.rGL)}.csv"

        results = calculate_jintegral_2d(
            step_start=int(args.step_start),
            step_end=args.step_end,
            Rj0=Rj0,
            Rj1=Rj1,
            result_dir=result_dir,
            output_file=output,
            use_saved_files=True,
            extend_symmetric=(not args.no_extend),
        )

        print(f"[JINT] result_dir: {result_dir}")
        print(f"[JINT] output:     {output}")
        print(f"[JINT] steps:      {len(results)}")
        return

    if args.output is not None:
        print("[JINT] Note: --output is ignored in --sweep-rj1 mode.")

    Rj1_list = _parse_sweep_rj1(args.sweep_rj1, Rj0=Rj0)
    for rj1 in Rj1_list:
        if rj1 <= Rj0:
            raise ValueError(f"Invalid contour in sweep: Rj1 ({rj1}) must be > Rj0 ({Rj0}).")

    print(f"[JINT] result_dir: {result_dir}")
    print(f"[JINT] mode:       sweep-rj1 ({len(Rj1_list)} contours)")

    cases: List[Dict[str, object]] = []
    for rj1 in Rj1_list:
        out_case = result_dir / (
            f"J_integral_2D_v{int(st.v)}_rGL{int(st.rGL)}_Rj1_{_slug_float(rj1)}.csv"
        )
        rows = calculate_jintegral_2d(
            step_start=int(args.step_start),
            step_end=args.step_end,
            Rj0=Rj0,
            Rj1=float(rj1),
            result_dir=result_dir,
            output_file=out_case,
            use_saved_files=True,
            extend_symmetric=(not args.no_extend),
        )
        cases.append({"Rj0": Rj0, "Rj1": float(rj1), "rows": rows, "output": out_case})
        print(f"[JINT] contour Rj1={rj1:.8g}: steps={len(rows)}, file={out_case.name}")

    out_files = _write_sweep_files(result_dir, int(st.v), int(st.rGL), cases)
    print(f"[JINT] combined:   {out_files['combined']}")
    print(f"[JINT] summary:    {out_files['summary']}")


if __name__ == "__main__":
    main()
