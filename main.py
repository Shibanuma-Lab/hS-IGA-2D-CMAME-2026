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
