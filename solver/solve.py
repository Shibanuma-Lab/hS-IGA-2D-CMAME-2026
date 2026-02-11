"""
solve – dispatcher for static / dynamic solve.
"""

import core.state as st
from solver.static import solvestatic
from solver.dynamic import solvedynamic


def solve(step):
    """Select and run the appropriate solver for *step*."""
    st.step = step
    if step == 0:
        solvestatic()
    else:
        solvedynamic()
