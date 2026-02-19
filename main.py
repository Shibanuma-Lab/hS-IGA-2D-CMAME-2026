#!/usr/bin/env python3
"""
main.py – entry point for the hS-IGA-2D straight crack simulation.

Replaces the monolithic ``execution()`` function and the outer for-loop
from the original single-file script.
"""

from datetime import datetime

# ---- Shared state (replaces all 'global' declarations) ----
import core.state as st

# ---- Configuration ----
from config.parameters import load_parameters
from config.fem_data import load_fem_data

# ---- Core logic ----
from core.alldata import Alldata
from core.jobset import jobset
from core.stepset import stepset
from core.calnos import calnos

# ---- Mesh ----
from mesh.local_mesh import makemesh
from mesh.meshset import meshset

# ---- Matrix assembly ----
from matrix.assemble import makematrix

# ---- Boundary conditions ----
from boundary.apply import boundary

# ---- Solver ----
from solver.initial import initial
from solver.solve import solve

# ---- Post-processing ----
from postprocess.getresult import getresult
from postprocess.find_uG_uL import finduGuL
from postprocess.savedata import savedata
from postprocess.debug_output import write_debug_info
from postprocess.jintegral_2d import calculate_jintegral_2d


def run_jintegral_postprocess():
    """Run J-integral / DSIF post-processing for the current job."""
    if int(getattr(st, "calc_jintegral", 0)) != 1:
        return
    if int(st.islocal) != 1:
        print("[JINT] Skip: J-integral requires local mesh (islocal=1).")
        return

    step_start_cfg = int(getattr(st, "jintegral_step_start", -1))
    step_end_cfg = int(getattr(st, "jintegral_step_end", -1))
    # Auto mode includes the full range from static initialization.
    step_start = 0 if step_start_cfg < 0 else step_start_cfg
    step_end = int(st.stepall) if step_end_cfg < 0 else step_end_cfg
    if step_start > step_end:
        step_start = step_end

    out_file = st.dirname / f"J_integral_2D_v{int(st.v)}_rGL{int(st.rGL)}.csv"

    print("[JINT] Calculating J-integral / DSIF ...")
    print(
        f"[JINT] steps={step_start}..{step_end}, "
        f"Rj0={st.jintegral_Rj0}, Rj1={st.jintegral_Rj1}"
    )

    results = calculate_jintegral_2d(
        step_start=step_start,
        step_end=step_end,
        Rj0=float(st.jintegral_Rj0),
        Rj1=float(st.jintegral_Rj1),
        result_dir=st.dirname,
        output_file=out_file,
        use_saved_files=bool(int(getattr(st, "jintegral_use_saved_files", 1))),
        extend_symmetric=bool(int(getattr(st, "jintegral_extend_symmetric", 1))),
    )
    print(f"[JINT] Done. {len(results)} steps written to: {out_file}")


def execution():
    """
    Main simulation loop.

    Flow (matches original Mathematica → Python translation):
        for job in [jobstart .. jobend]:
            Alldata()
            jobset(job)
            for step in [stepini .. stepall]:
                stepset(step)
                makemesh(step)
                if meshonly != 1:
                    meshset()
                    makematrix()
                    boundary(step)
                    initial(step)
                    solve(step)
                    getresult()
                    if islocal == 1: finduGuL()
                    if issave  == 1: savedata(step)
            calnos()
    """
    for job in range(int(st.jobstart), int(st.jobend) + 1):
        Alldata()
        jobset(job)

        for step in range(int(st.stepini), int(st.stepall) + 1):
            print("--------------------------------------------------")
            print("step:", step)
            print(f"Job{job}_Step{step}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            if step >= 95:
                print(f"[DIAG] === Entering critical region: step {step} ===")

            stepset(step)
            makemesh(step)

            if int(st.meshonly) != 1:
                meshset()
                makematrix()
                boundary(step)
                initial(step)
                
                # Write debug info (mesh, BC, initial conditions) if enabled
                write_debug_info(step)
                
                solve(step)
                getresult()
                if int(st.islocal) == 1:
                    finduGuL()
                if int(st.issave) == 1:
                    savedata(step)

        calnos()
        run_jintegral_postprocess()


# ====================================================================
# Script entry point
# ====================================================================
if __name__ == "__main__":
    # 1) Load parameters (rGL value can be changed here)
    load_parameters(rGL_value=2)

    # 2) Load FEM reference data
    load_fem_data(version=st.v, step_label=st.step_label_fem)

    # 3) Run simulation
    execution()
