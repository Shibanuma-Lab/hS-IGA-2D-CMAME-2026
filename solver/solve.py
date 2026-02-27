"""
solve – dispatcher for static / dynamic solve.
"""

import core.state as st
from solver.static import solvestatic
from solver.dynamic import solvedynamic


def solve(step):
    """Select and run the appropriate solver for *step*."""
    st.step = step
    if getattr(st, "analysis_mode", "dynamic") == "static" or int(getattr(st, "isdynamic", 1)) == 0:
        solvestatic()
    elif step == 0:
        solvestatic()
    else:
        solvedynamic()
