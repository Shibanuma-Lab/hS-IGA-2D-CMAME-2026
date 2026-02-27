"""
boundary – dispatcher for Dirichlet / Neumann boundary conditions.
"""

import core.state as st
from boundary.boundary_fem import boundaryFEM
from boundary.boundary_static import boundary_static
from boundary.boundary_sfem import boundarysFEM


def boundary(step):
    """Apply boundary conditions for the current step."""
    if getattr(st, "analysis_mode", "dynamic") == "static":
        boundary_static(step)
        return
    if int(st.islocal) == 0:
        boundaryFEM(step)
    elif int(st.islocal) == 1:
        boundarysFEM(step)
    else:
        raise ValueError(f"islocal must be 0 or 1, got {st.islocal}")
