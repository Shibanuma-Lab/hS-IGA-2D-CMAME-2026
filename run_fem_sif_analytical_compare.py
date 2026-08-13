#!/usr/bin/env python3
"""
Compare FEM-reference DSIF from J-integral with Broberg analytical SIF.

Default case:
  python3 run_fem_sif_analytical_compare.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import core.state as st
from config.parameters import load_parameters
from core.calnos import analytical_sif
from postprocess.jintegral_2d import calculate_jintegral_2d_fem_reference


def _parse_steps(spec: str) -> List[int]:
    steps = []
    for token in str(spec).split(","):
        item = token.strip()
        if item == "":
            continue
        steps.append(int(item))
    if len(steps) == 0:
        raise ValueError("No steps were provided.")
    return steps


def _infer_case_from_path(path: Path) -> Dict[str, Optional[int]]:
    """
    Infer velocity/crack length from names such as:
      h5_export_V_500_a_50
      h5_export_v500_a_50
    """
    rx = re.compile(r"h5_export_[Vv]_?(?P<v>\d+)(?:_a_(?P<a>\d+))?")
    m = rx.search(path.name)
    if m is None:
        return {"velocity": None, "crack_length_mm": None}
    a = m.group("a")
    return {
        "velocity": int(m.group("v")),
        "crack_length_mm": None if a is None else int(a),
    }


def _default_output_dir(fem_ref: Path) -> Path:
    return Path("results") / "fem_sif_validation" / fem_ref.name


def _write_compare_csv(rows: List[Dict[str, float]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Step",
                "crack_length_m",
                "crack_length_mm",
                "J_total_FEM",
                "J_static_FEM",
                "J_dynamic_FEM",
                "K_I_FEM",
                "K_I_analytical",
                "K_I_FEM_over_analytical",
                "K_I_rel_error",
                "K_I_abs_rel_error",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    int(r["step"]),
                    r["crack_length_m"],
                    r["crack_length_mm"],
                    r["J_total_FEM"],
                    r["J_static_FEM"],
                    r["J_dynamic_FEM"],
                    r["K_I_FEM"],
                    r["K_I_analytical"],
                    r["K_I_FEM_over_analytical"],
                    r["K_I_rel_error"],
                    r["K_I_abs_rel_error"],
                ]
            )


def _build_compare_rows(fem_rows: List[Dict[str, float]], velocity: float, sigma_inf: float) -> List[Dict[str, float]]:
    out = []
    for row in fem_rows:
        step = int(row["step"])
        crack_length = step * float(st.hL)
        k_analytical = analytical_sif(
            crack_length=crack_length,
            V=velocity,
            sigma_inf=sigma_inf,
            hL=st.hL,
        )
        k_fem = float(row["K_I"])
        if abs(k_analytical) < 1e-14:
            ratio = np.nan
            rel_error = np.nan
            abs_rel_error = np.nan
        else:
            ratio = k_fem / k_analytical
            rel_error = (k_fem - k_analytical) / k_analytical
            abs_rel_error = abs(rel_error)

        out.append(
            {
                "step": step,
                "crack_length_m": float(crack_length),
                "crack_length_mm": float(crack_length * 1000.0),
                "J_total_FEM": float(row["J"]),
                "J_static_FEM": float(row["J_static"]),
                "J_dynamic_FEM": float(row["J_dynamic"]),
                "K_I_FEM": k_fem,
                "K_I_analytical": float(k_analytical),
                "K_I_FEM_over_analytical": float(ratio),
                "K_I_rel_error": float(rel_error),
                "K_I_abs_rel_error": float(abs_rel_error),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate J-integral FEM-reference DSIF against analytical Broberg SIF."
    )
    parser.add_argument(
        "--fem-ref",
        type=Path,
        default=Path("FEM_data/h5_export_V_500_a_50"),
        help="FEM reference source (.mat, .h5 file, or H5 directory).",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="50,125,250,500,1000",
        help="Comma-separated step numbers to evaluate.",
    )
    parser.add_argument("--velocity", type=float, default=None, help="Crack velocity V [m/s].")
    parser.add_argument("--crack-length-mm", type=float, default=None, help="Total crack length [mm].")
    parser.add_argument("--sigma-inf", type=float, default=None, help="Far-field stress [Pa].")
    parser.add_argument("--rgl", type=int, default=2, help="rGL value for parameter initialization.")
    parser.add_argument("--Rj0", type=float, default=None, help="Inner radial weight parameter.")
    parser.add_argument("--Rj1", type=float, default=None, help="Outer radial weight parameter.")
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["mathematica", "standard"],
        default=None,
        help="J-integral algebra/scaling convention.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Analytical comparison CSV.")
    parser.add_argument("--fem-output", type=Path, default=None, help="Raw FEM J/K CSV.")
    parser.add_argument("--extend-symmetric", action="store_true", help="Enable symmetric mesh extension.")
    args = parser.parse_args()

    fem_ref = args.fem_ref.resolve()
    if not fem_ref.exists():
        raise FileNotFoundError(f"FEM reference source not found: {fem_ref}")

    inferred = _infer_case_from_path(fem_ref)
    velocity = args.velocity if args.velocity is not None else inferred["velocity"]
    if velocity is None:
        raise ValueError("Velocity could not be inferred. Please pass --velocity.")
    crack_length_mm = args.crack_length_mm
    if crack_length_mm is None:
        crack_length_mm = inferred["crack_length_mm"]

    steps = _parse_steps(args.steps)

    load_parameters(rGL_value=args.rgl)
    st.v = float(velocity)
    st.vlist = float(velocity)
    if crack_length_mm is not None:
        st.c_crack = float(crack_length_mm) * 1.0e-3
        st.stepall = int(round(st.c_crack / st.hL))
    else:
        st.stepall = max(steps)

    st.fem_reference_source = "h5" if fem_ref.is_dir() or fem_ref.suffix.lower() == ".h5" else "mat"
    st.fem_h5_dir = str(fem_ref)
    if args.scheme is not None:
        st.jintegral_scheme = args.scheme

    Rj0 = float(st.jintegral_Rj0 if args.Rj0 is None else args.Rj0)
    Rj1 = float(st.jintegral_Rj1 if args.Rj1 is None else args.Rj1)
    if Rj1 <= Rj0:
        raise ValueError(f"Invalid contour: Rj1 ({Rj1}) must be > Rj0 ({Rj0}).")

    sigma_inf = float(st.SigmaInfinity if args.sigma_inf is None else args.sigma_inf)
    output_dir = _default_output_dir(fem_ref)
    output_file = args.output
    if output_file is None:
        output_file = output_dir / f"FEM_SIF_vs_analytical_v{int(velocity)}.csv"
    fem_output = args.fem_output
    if fem_output is None:
        fem_output = output_dir / f"J_integral_2D_FEM_v{int(velocity)}_selected_steps.csv"

    fem_rows = calculate_jintegral_2d_fem_reference(
        fem_reference_file=fem_ref,
        step_start=min(steps),
        step_end=max(steps),
        Rj0=Rj0,
        Rj1=Rj1,
        result_dir=output_dir,
        output_file=fem_output,
        extend_symmetric=bool(args.extend_symmetric),
        steps=steps,
    )
    compare_rows = _build_compare_rows(fem_rows, velocity=float(velocity), sigma_inf=sigma_inf)
    _write_compare_csv(compare_rows, output_file)

    print(f"[SIF] FEM ref:    {fem_ref}")
    print(f"[SIF] scheme:     {st.jintegral_scheme}")
    print(f"[SIF] steps:      {','.join(str(s) for s in steps)}")
    print(f"[SIF] fem output: {fem_output}")
    print(f"[SIF] compare:    {output_file}")
    print("[SIF] Step  K_FEM/K_analytical  abs_rel_error")
    for row in compare_rows:
        print(
            f"[SIF] {int(row['step']):4d}  "
            f"{row['K_I_FEM_over_analytical']:.8e}  "
            f"{row['K_I_abs_rel_error']:.8e}"
        )


if __name__ == "__main__":
    main()
