"""
Analytical mode-I crack-tip fields for the static benchmark.
"""

from __future__ import annotations

import math
from typing import Tuple


def polar_from_tip(point, crack_tip_x: float) -> Tuple[float, float]:
    """Return ``(r, theta)`` for a point measured from ``(crack_tip_x, 0)``."""
    x = float(point[0]) - float(crack_tip_x)
    y = float(point[1])
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    if theta < 0.0:
        theta += 2.0 * math.pi
    return r, theta


def exact_mode_i_displacement(point, crack_tip_x: float, mu: float, kappa: float):
    """
    Analytical near-tip displacement for the unit-amplitude static benchmark.
    """
    r, theta = polar_from_tip(point, crack_tip_x)
    if r == 0.0:
        return (0.0, 0.0)

    factor = (1.0 / (2.0 * float(mu))) * math.sqrt(r / (2.0 * math.pi))
    cos_half = math.cos(theta / 2.0)
    sin_half = math.sin(theta / 2.0)

    u1 = factor * cos_half * (float(kappa) - 1.0 + 2.0 * sin_half ** 2)
    u2 = factor * sin_half * (float(kappa) + 1.0 - 2.0 * cos_half ** 2)
    return (u1, u2)


def exact_mode_i_stress_yy(point, crack_tip_x: float):
    """
    Analytical ``sigma_yy`` for the unit-amplitude static benchmark.
    """
    r, theta = polar_from_tip(point, crack_tip_x)
    if r == 0.0:
        return 1.0e2

    return (
        math.sqrt(1.0 / (2.0 * math.pi * r))
        * math.cos(theta / 2.0)
        * (1.0 + math.sin(theta / 2.0) * math.sin(1.5 * theta))
    )
