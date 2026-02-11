"""
makematrix – dispatcher that selects which stiffness/mass matrices to build.
"""

import core.state as st
from matrix.make_KG import makeKG
from matrix.make_KL import makeKL
from matrix.make_KGL import makeKGL


def makematrix():
    """Build stiffness / mass matrices depending on ``st.islocal``."""
    if int(st.islocal) == 0:
        makeKG()
    elif int(st.islocal) == 1:
        makeKG()
        makeKL()
        makeKGL()
