"""
stepset – create step-level directory and set CWD.

Mathematica: stepset[step_] := Module[{}, ...]
"""

import os
import shutil
from pathlib import Path

import core.state as st


def stepset(step):
    """Create the step directory under ``st.dirname`` and chdir there."""

    if st.printcheck == 1:
        print("setting_1")

    def num(ndata: int) -> str:
        if ndata == -1:
            return f"i_{st.jobname}_00000"
        return f"s_{st.jobname}_{str(int(ndata)).zfill(5)}"

    droot = Path(st.dirname)

    if getattr(st, "analysis_mode", "dynamic") == "static":
        droot.mkdir(parents=True, exist_ok=True)
        os.chdir(droot)
        st.dirnamestep = str(droot)
        if st.printcheck == 1:
            print("setting_end")
        return

    target = droot / num(step)
    if target.exists() and target.is_dir():
        shutil.rmtree(target)

    droot.mkdir(parents=True, exist_ok=True)
    os.chdir(droot)

    target.mkdir(parents=True, exist_ok=False)
    st.dirnamestep = str(target)

    if st.printcheck == 1:
        print("setting_end")
