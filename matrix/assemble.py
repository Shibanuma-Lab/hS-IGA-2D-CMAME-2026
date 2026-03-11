"""
makematrix – dispatcher that selects which stiffness/mass matrices to build.
"""

import core.state as st
from matrix.make_KG import makeKG
from matrix.make_KL import makeKL
from matrix.make_KGL import makeKGL


def _global_signature():
    control_pts = getattr(st, "controlPts", None)
    if control_pts is None:
        return None
    return (
        str(getattr(st, "dirname", "")),
        id(control_pts),
        int(getattr(st, "nPtsX", 0)),
        int(getattr(st, "nPtsY", 0)),
        int(getattr(st, "p", 0)),
        int(getattr(st, "q", 0)),
        int(getattr(st, "nelem", 0)),
    )


def _need_global_assembly() -> bool:
    """
    Global IGA mesh/material are step-invariant within one job in current flow.
    Rebuild KG/MG when global mesh signature changes (or if missing).
    """
    current_sig = _global_signature()
    if current_sig is None:
        return True

    cached_sig = getattr(st, "_kg_cache_signature", None)
    if cached_sig != current_sig:
        return True

    if getattr(st, "KG", None) is None:
        return True

    return False


def makematrix():
    """Build stiffness / mass matrices depending on ``st.islocal``."""
    need_kg = _need_global_assembly()

    if int(st.islocal) == 0:
        if need_kg:
            makeKG()
            st._kg_cache_signature = _global_signature()
    elif int(st.islocal) == 1:
        if need_kg:
            makeKG()
            st._kg_cache_signature = _global_signature()
        makeKL()
        makeKGL()
