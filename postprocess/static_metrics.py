"""
Static benchmark post-processing: L2 norm, normalized stress yy, and normalized SIF.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

import core.state as st
from utils.interpolator import BilinearQuadInterpolator
from utils.shape_functions import Dshp, GP, GW, shp
from utils.static_crack import exact_mode_i_stress_yy


def _zero_snap(x: float, scale: float) -> float:
    return 0.0 if abs(x) < 1.0e-9 * scale else float(x)


def _l2_quadrature():
    """
    Return tensor-product Gauss points/weights for static L2 integration.

    Uses the same GP/GW source as assembly so L2 and shape-function integration
    stay fully consistent.
    """
    ngp = int(getattr(st, "static_l2_ngp", 8))
    gp1 = np.asarray(GP(ngp), dtype=np.float64)
    gw1 = np.asarray(GW(ngp), dtype=np.float64)
    gauss_points = np.array([(b, a) for a in gp1 for b in gp1], dtype=np.float64)
    weights = np.outer(gw1, gw1).reshape(-1).astype(np.float64)
    return gauss_points, weights


def _build_structured_field(nodes: np.ndarray, values: np.ndarray, decimals: int = 14):
    """
    Build a tensor-grid field table for fast bilinear interpolation.
    Returns ``(x_grid, y_grid, field_2d)`` or ``None`` if the mesh is not structured.
    """
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or len(nodes) != len(values):
        return None

    xr = np.round(nodes[:, 0], decimals=decimals)
    yr = np.round(nodes[:, 1], decimals=decimals)
    x_grid = np.unique(xr)
    y_grid = np.unique(yr)
    nx = len(x_grid)
    ny = len(y_grid)
    if nx * ny != len(nodes):
        return None

    ix = np.searchsorted(x_grid, xr)
    iy = np.searchsorted(y_grid, yr)
    field = np.full((ny, nx), np.nan, dtype=float)
    field[iy, ix] = values
    if np.isnan(field).any():
        return None

    return x_grid.astype(float), y_grid.astype(float), field


def _interp_structured_bilinear(x_grid, y_grid, field_2d, xq, yq):
    """Vectorized bilinear interpolation on a structured grid."""
    xq = np.asarray(xq, dtype=float)
    yq = np.asarray(yq, dtype=float)

    ix = np.searchsorted(x_grid, xq, side="right") - 1
    iy = np.searchsorted(y_grid, yq, side="right") - 1
    ix = np.clip(ix, 0, len(x_grid) - 2)
    iy = np.clip(iy, 0, len(y_grid) - 2)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]

    tx = np.divide(xq - x0, x1 - x0, out=np.zeros_like(xq), where=np.abs(x1 - x0) > 0.0)
    ty = np.divide(yq - y0, y1 - y0, out=np.zeros_like(yq), where=np.abs(y1 - y0) > 0.0)

    v00 = field_2d[iy, ix]
    v10 = field_2d[iy, ix + 1]
    v11 = field_2d[iy + 1, ix + 1]
    v01 = field_2d[iy + 1, ix]

    return (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + tx * ty * v11
        + (1.0 - tx) * ty * v01
    )


def _exact_u_legacy_vec(x, y, crack_tip_x, mu, kappa):
    """Vectorized legacy analytical displacement field."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xpolar = x - float(crack_tip_x)
    ypolar = y
    r = np.hypot(xpolar, ypolar)

    theta = np.empty_like(xpolar)
    mask_zero = xpolar == 0.0
    mask_pos = xpolar > 0.0
    mask_neg = ~(mask_zero | mask_pos)
    theta[mask_zero] = math.pi / 2.0
    theta[mask_pos] = np.arctan(ypolar[mask_pos] / xpolar[mask_pos])
    theta[mask_neg] = math.pi + np.arctan(ypolar[mask_neg] / xpolar[mask_neg])

    factor = np.zeros_like(r)
    nonzero = r > 0.0
    factor[nonzero] = (1.0 / (2.0 * float(mu))) * np.sqrt(r[nonzero] / (2.0 * math.pi))

    half = 0.5 * theta
    s = np.sin(half)
    c = np.cos(half)
    u1 = factor * c * (float(kappa) - 1.0 + 2.0 * s * s)
    u2 = factor * s * (float(kappa) + 1.0 - 2.0 * c * c)
    return u1, u2


def _compute_l2_norm_fallback():
    """Legacy robust path (generic interpolator search for each quadrature point)."""
    width = float(st.static_width)
    height = float(st.static_height)
    crack_tip_x = float(st.static_crack_tip_x)
    mu = float(st.mu)
    kappa = float(st.kappa)
    h_back = float(st.hL) * 0.5
    n_back_x = int(width / h_back)
    n_back_y = int(height / h_back)

    node_back_x = h_back * np.arange(0, n_back_x + 1, dtype=float)
    node_back_y = h_back * np.arange(0, n_back_y + 1, dtype=float)
    node_back = np.array(
        [[_zero_snap(x, height), _zero_snap(y, height)] for y in node_back_y for x in node_back_x],
        dtype=float,
    )
    elem_back = []
    for iy in range(n_back_y):
        for ix in range(n_back_x):
            n0 = iy * (n_back_x + 1) + ix
            n1 = n0 + 1
            n2 = n0 + (n_back_x + 1) + 1
            n3 = n0 + (n_back_x + 1)
            elem_back.append([n0, n1, n2, n3])
    elem_back = np.asarray(elem_back, dtype=int)

    uGx = BilinearQuadInterpolator(st.nodeG, st.elemGI, st.disG2D[:, 0], name="static_meshG_x")
    uGy = BilinearQuadInterpolator(st.nodeG, st.elemGI, st.disG2D[:, 1], name="static_meshG_y")
    uLx = BilinearQuadInterpolator(st.nodeL, st.elemL, st.disLG2D[:, 0], name="static_meshL_x")
    uLy = BilinearQuadInterpolator(st.nodeL, st.elemL, st.disLG2D[:, 1], name="static_meshL_y")

    gauss_points, weights = _l2_quadrature()
    min_lx = float(np.min(st.nodeL[:, 0]))
    max_lx = float(np.max(st.nodeL[:, 0]))
    min_ly = float(np.min(st.nodeL[:, 1]))
    max_ly = float(np.max(st.nodeL[:, 1]))

    ut_nt_nu = 0.0
    ut_nt_nu_theo = 0.0

    for conn in elem_back:
        elem_nodes = node_back[conn]
        for gp_idx, (xi, eta) in enumerate(gauss_points):
            dshp = Dshp(np.array([xi, eta], dtype=float))
            jac = dshp @ elem_nodes
            det_jac = float(np.linalg.det(jac))
            phys = (shp(np.array([xi, eta], dtype=float)).ravel() @ elem_nodes).astype(float)
            if min_lx <= phys[0] <= max_lx and min_ly <= phys[1] <= max_ly:
                ui = np.array([uLx(phys), uLy(phys)], dtype=float)
            else:
                ui = np.array([uGx(phys), uGy(phys)], dtype=float)

            ux_ref, uy_ref = _exact_u_legacy_vec(
                np.array([phys[0]], dtype=float),
                np.array([phys[1]], dtype=float),
                crack_tip_x,
                mu,
                kappa,
            )
            u_exact = np.array([ux_ref[0], uy_ref[0]], dtype=float)

            diff = ui - u_exact
            w = float(weights[gp_idx])
            ut_nt_nu += w * float(np.dot(diff, diff)) * det_jac
            ut_nt_nu_theo += w * float(np.dot(u_exact, u_exact)) * det_jac

    if ut_nt_nu_theo <= 0.0:
        return float("nan")
    return math.sqrt(ut_nt_nu / ut_nt_nu_theo)


def compute_l2_norm():
    """
    Compute the legacy static L2 norm over the whole plate.

    Fast path:
      structured-grid bilinear interpolation + batched quadrature evaluation.
    Fallback path:
      original point-by-point generic interpolation.
    """
    width = float(st.static_width)
    height = float(st.static_height)
    crack_tip_x = float(st.static_crack_tip_x)
    mu = float(st.mu)
    kappa = float(st.kappa)
    h_back = float(st.hL) * 0.5
    n_back_x = int(width / h_back)
    n_back_y = int(height / h_back)

    g_x = _build_structured_field(st.nodeG, st.disG2D[:, 0])
    g_y = _build_structured_field(st.nodeG, st.disG2D[:, 1])
    l_x = _build_structured_field(st.nodeL, st.disLG2D[:, 0])
    l_y = _build_structured_field(st.nodeL, st.disLG2D[:, 1])

    # If any mesh is not structured, preserve legacy behavior.
    if g_x is None or g_y is None or l_x is None or l_y is None:
        return _compute_l2_norm_fallback()

    gx_grid, gy_grid, g_field_x = g_x
    _, _, g_field_y = g_y
    lx_grid, ly_grid, l_field_x = l_x
    _, _, l_field_y = l_y

    gauss_points, weights = _l2_quadrature()
    half = 0.5 * h_back
    det_jac = half * half
    dx = gauss_points[:, 0] * half
    dy = gauss_points[:, 1] * half
    weighted_det = weights * det_jac
    row_weights = np.tile(weighted_det, n_back_x)

    min_lx = float(np.min(st.nodeL[:, 0]))
    max_lx = float(np.max(st.nodeL[:, 0]))
    min_ly = float(np.min(st.nodeL[:, 1]))
    max_ly = float(np.max(st.nodeL[:, 1]))

    centers_x = (np.arange(n_back_x, dtype=float) + 0.5) * h_back
    centers_y = (np.arange(n_back_y, dtype=float) + 0.5) * h_back

    ut_nt_nu = 0.0
    ut_nt_nu_theo = 0.0

    for cy in centers_y:
        xq_mat = centers_x[:, None] + dx[None, :]
        yq_mat = cy + dy[None, :]
        yq_mat = np.broadcast_to(yq_mat, xq_mat.shape)

        xq = xq_mat.reshape(-1)
        yq = yq_mat.reshape(-1)
        use_local = (xq >= min_lx) & (xq <= max_lx) & (yq >= min_ly) & (yq <= max_ly)

        ux_num = np.empty_like(xq)
        uy_num = np.empty_like(yq)

        if np.any(use_local):
            idx = np.where(use_local)[0]
            ux_num[idx] = _interp_structured_bilinear(lx_grid, ly_grid, l_field_x, xq[idx], yq[idx])
            uy_num[idx] = _interp_structured_bilinear(lx_grid, ly_grid, l_field_y, xq[idx], yq[idx])
        if np.any(~use_local):
            idx = np.where(~use_local)[0]
            ux_num[idx] = _interp_structured_bilinear(gx_grid, gy_grid, g_field_x, xq[idx], yq[idx])
            uy_num[idx] = _interp_structured_bilinear(gx_grid, gy_grid, g_field_y, xq[idx], yq[idx])

        ux_ref, uy_ref = _exact_u_legacy_vec(xq, yq, crack_tip_x, mu, kappa)
        diff_sq = (ux_num - ux_ref) ** 2 + (uy_num - uy_ref) ** 2
        exact_sq = ux_ref ** 2 + uy_ref ** 2

        ut_nt_nu += float(np.dot(row_weights, diff_sq))
        ut_nt_nu_theo += float(np.dot(row_weights, exact_sq))

    if ut_nt_nu_theo <= 0.0:
        return float("nan")
    return math.sqrt(ut_nt_nu / ut_nt_nu_theo)


def compute_normalized_stress_yy():
    """Evaluate ``sigma_yy / sigma_yy_exact`` on the local bottom edge ahead of the tip."""
    start = int(st.aL) + 1
    stop = int(st.aL) + int(st.lL)
    if stop <= start:
        return np.empty(0, dtype=float)

    stress_ref = np.array(
        [exact_mode_i_stress_yy(pt, st.static_crack_tip_x) for pt in st.nodeL[start:stop]],
        dtype=float,
    )
    stress_sol = (-1.0) * np.asarray(st.rfL2D[start:stop, 1], dtype=float) / float(st.hL)
    out = np.full_like(stress_ref, np.nan, dtype=float)
    mask = np.abs(stress_ref) > 0.0
    out[mask] = stress_sol[mask] / stress_ref[mask]
    return out


def compute_normalized_sif():
    """
    Compute the static J-integral and the corresponding normalized SIF.

    The static benchmark uses the unit-amplitude analytical near-tip field,
    so the normalized SIF equals the computed ``K_I`` itself.
    """
    xi_eta = np.array([(b, a) for a in GP(2) for b in GP(2)], dtype=float)
    weights = np.array([w2 * w1 for w1 in GW(2) for w2 in GW(2)], dtype=float)
    Dnn = [Dshp(pt) for pt in xi_eta]

    tip_node = int(st.aL)
    node_tip = np.asarray(st.nodeL, dtype=float) - np.array([float(st.static_crack_tip_x), 0.0], dtype=float)

    tip_elems = np.where(np.any(st.elemL == tip_node, axis=1))[0]
    q1i = np.unique(st.elemL[tip_elems].ravel()) if len(tip_elems) > 0 else np.array([tip_node], dtype=int)

    if st.nodeLx is None or len(st.nodeLx) < 3:
        hL_real = float(st.hL)
    else:
        hL_real = float(st.nodeLx[2] - st.nodeLx[1])

    qi = np.zeros(len(st.nodeL), dtype=float)
    rj0 = float(st.jintegral_Rj0)
    rj1 = float(st.jintegral_Rj1)
    q1i_set = set(int(v) for v in q1i.tolist())
    for i, pt in enumerate(node_tip):
        if i in q1i_set:
            qi[i] = 1.0
            continue
        r = math.hypot(float(pt[0]), float(pt[1]))
        if r <= rj0 * hL_real + 1.0e-8:
            qi[i] = 1.0
        elif abs(rj0 - rj1) <= 1.0e-15:
            qi[i] = 0.0
        elif r < rj1 * hL_real:
            qi[i] = (r - rj1 * hL_real) / ((rj0 - rj1) * hL_real)
        else:
            qi[i] = 0.0

    meas = float(qi[tip_node]) if 0 <= tip_node < len(qi) else 0.0
    if meas <= 1.0e-12:
        return {"J": float("nan"), "normalized_sif": float("nan")}

    j_acc = 0.0
    qe = np.array([float(np.sum(qi[conn])) for conn in st.elemL], dtype=float)
    active_elems = np.where(qe > 1.0e-8)[0]

    for e in active_elems:
        conn = st.elemL[e]
        enode = node_tip[conn]
        disp = np.asarray(st.disLG2D[conn], dtype=float)
        qi_e = qi[conn]

        for gp_idx, dnn in enumerate(Dnn):
            jac = np.asarray(dnn, dtype=float) @ enode
            det_jac = float(np.linalg.det(jac))
            bb = np.linalg.inv(jac) @ np.asarray(dnn, dtype=float)

            eps = np.array(
                [
                    bb[0] @ disp[:, 0],
                    bb[1] @ disp[:, 1],
                    bb[0] @ disp[:, 1] + bb[1] @ disp[:, 0],
                ],
                dtype=float,
            )
            sigma = st.de @ eps
            du = bb[0] @ disp
            dq = np.array([bb[0] @ qi_e, bb[1] @ qi_e], dtype=float)
            energy = 0.5 * float(np.dot(eps, sigma))
            term = (
                ((sigma[0] * du[0] + sigma[2] * du[1] - energy) * dq[0])
                + ((sigma[2] * du[0] + sigma[1] * du[1]) * dq[1])
            )
            j_acc += term * det_jac * float(weights[gp_idx])

    jint = (2.0 * j_acc) / meas
    jint = float(jint)
    if jint < 0.0 and abs(jint) < 1.0e-10:
        jint = 0.0

    sif = math.sqrt(max(0.0, (float(st.EE) * jint) / (1.0 - float(st.nu) ** 2)))
    return {"J": jint, "normalized_sif": sif}


def compute_static_metrics():
    """Compute all requested scalar/vector metrics for the current static case."""
    stress = compute_normalized_stress_yy()
    sif = compute_normalized_sif()
    return {
        "case_name": str(st.static_case_label),
        "hG": float(st.static_case_hG),
        "hL": float(st.static_case_hL),
        "rGL": float(st.rGL),
        "nGx": int(st.static_case_nGx),
        "nGy": int(st.static_case_nGy),
        "nhL": int(st.static_case_nhL),
        "aL": int(st.aL),
        "lL": int(st.lL),
        "HL": int(st.HL),
        "dof": int((len(st.nodeL) + len(st.nodeG)) * 2),
        "l2_norm": float(compute_l2_norm()),
        "normalized_stress_yy": stress,
        "normalized_sif": float(sif["normalized_sif"]),
        "j_integral": float(sif["J"]),
    }


def write_case_metric_files(case_dir: Path, metrics):
    """Write per-case scalar/vector metric CSV files inside the case directory."""
    case_dir = Path(case_dir)
    with open(case_dir / "static_case_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["case_name", "hG", "hL", "rGL", "nGx", "nGy", "nhL", "aL", "lL", "HL", "dof", "l2_norm", "normalized_sif", "j_integral"]
        )
        writer.writerow(
            [
                metrics["case_name"],
                metrics["hG"],
                metrics["hL"],
                metrics["rGL"],
                metrics["nGx"],
                metrics["nGy"],
                metrics["nhL"],
                metrics["aL"],
                metrics["lL"],
                metrics["HL"],
                metrics["dof"],
                metrics["l2_norm"],
                metrics["normalized_sif"],
                metrics["j_integral"],
            ]
        )

    with open(case_dir / "normalized_stress_yy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["case_name", "dof"] + [f"point_{i + 1}" for i in range(len(metrics["normalized_stress_yy"]))]
        writer.writerow(header)
        writer.writerow([metrics["case_name"], metrics["dof"], *metrics["normalized_stress_yy"].tolist()])


def write_static_summary_files(parent_dir: Path, metrics_rows):
    """Write the requested summary CSV files to the sweep parent folder."""
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(metrics_rows, key=lambda r: (r["dof"], r["hG"], r["hL"], r["case_name"]))

    with open(parent_dir / "dof_l2_norm.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_name", "hG", "hL", "rGL", "nGx", "nGy", "nhL", "aL", "lL", "HL", "dof", "l2_norm"])
        for row in rows:
            writer.writerow(
                [
                    row["case_name"],
                    row["hG"],
                    row["hL"],
                    row["rGL"],
                    row["nGx"],
                    row["nGy"],
                    row["nhL"],
                    row["aL"],
                    row["lL"],
                    row["HL"],
                    row["dof"],
                    row["l2_norm"],
                ]
            )

    with open(parent_dir / "normalized_sif.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_name", "hG", "hL", "rGL", "nGx", "nGy", "nhL", "dof", "normalized_sif", "j_integral"])
        for row in rows:
            writer.writerow(
                [
                    row["case_name"],
                    row["hG"],
                    row["hL"],
                    row["rGL"],
                    row["nGx"],
                    row["nGy"],
                    row["nhL"],
                    row["dof"],
                    row["normalized_sif"],
                    row["j_integral"],
                ]
            )

    max_len = max((len(row["normalized_stress_yy"]) for row in rows), default=0)
    with open(parent_dir / "normalized_stress_yy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["case_name", "hG", "hL", "rGL", "nGx", "nGy", "nhL", "dof"]
            + [f"point_{i + 1}" for i in range(max_len)]
        )
        for row in rows:
            stress = row["normalized_stress_yy"].tolist()
            stress += [float("nan")] * (max_len - len(stress))
            writer.writerow(
                [
                    row["case_name"],
                    row["hG"],
                    row["hL"],
                    row["rGL"],
                    row["nGx"],
                    row["nGy"],
                    row["nhL"],
                    row["dof"],
                    *stress,
                ]
            )
