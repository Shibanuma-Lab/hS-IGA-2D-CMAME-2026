"""
savedata – accumulate per-step results and (optionally) export VTU files.
"""

import numpy as np
import core.state as st
from pathlib import Path


def savedata(step):
    """Append current step data to the accumulation buffers in ``st``."""
    if int(st.islocal) == 0:
        st.disG2DAllMa2D.append(st.disG2D)
        st.disVis2DAllMa2D.append(st.disVis)
        st.velG2DAllMa2D.append(st.velG2D)
        st.acceG2DAllMa2D.append(st.acceG2D)
        st.rfG2DAllMa2D.append(st.rfG2D)
    else:
        st.disG2DAllMa2D.append(st.disG2D)
        st.disL2DAllMa2D.append(st.disL2D)
        st.disVis2DAllMa2D.append(st.disVis)
        st.disLG2DAllMa2D.append(st.disLG2D)
        st.velG2DAllMa2D.append(st.velG2D)
        st.velL2DAllMa2D.append(st.velL2D)
        st.velLG2DAllMa2D.append(st.velLG2D)
        st.acceG2DAllMa2D.append(st.acceG2D)
        st.acceL2DAllMa2D.append(st.acceL2D)
        st.acceLG2DAllMa2D.append(st.acceLG2D)
        st.rfG2DAllMa2D.append(st.rfG2D)
        st.rfMG2DAllMa2D.append(st.rfMG2D)
        st.rfL2DAllMa2D.append(st.rfL2D)
        st.rfML2DAllMa2D.append(st.rfML2D)
        st.stressViaAllMa2D.append(st.stressVis)
        st.stressLAllMa2D.append(st.stressL)

    # ---- Save per-step result data for offline post-processing ----
    _write_step_result_dat()

    # ---- VTU export ----
    if int(getattr(st, "save_vtu", 1)) != 1:
        return

    if int(st.islocal) == 0:
        stress_vis = np.vstack([st.sigmaXX, st.sigmaYY, st.sigmaXY]).T
        stressVTU = stress_vis
        dispVTU = np.column_stack([st.disVis[:, 0], st.disVis[:, 1], np.zeros(len(st.disVis))])
        elemVTU = st.elemVis
        islocalList = np.zeros(len(st.nodeVis), dtype=int)
        pointsVTU = np.column_stack([st.nodeVis[:, 0], st.nodeVis[:, 1], np.zeros(len(st.nodeVis))])
    else:
        stressVTU = np.vstack([st.stressVis, st.stressL])
        dispVTU = np.vstack([
            np.column_stack([st.disVis[:, 0], st.disVis[:, 1], np.zeros(len(st.disVis))]),
            np.column_stack([st.disLG2D[:, 0], st.disLG2D[:, 1], np.zeros(len(st.disLG2D))]),
        ])
        offset = len(st.nodeVis)
        elemVTU = np.vstack([st.elemVis, st.elemL + offset])
        islocalList = np.concatenate([
            np.zeros(len(st.nodeVis), dtype=int),
            np.ones(len(st.nodeL), dtype=int),
        ])
        pointsVTU = np.vstack([
            np.column_stack([st.nodeVis[:, 0], st.nodeVis[:, 1], np.zeros(len(st.nodeVis))]),
            np.column_stack([st.nodeL[:, 0], st.nodeL[:, 1], np.zeros(len(st.nodeL))]),
        ])

    offsets = np.arange(1, len(elemVTU) + 1) * 4
    types   = np.full(len(elemVTU), 9, dtype=int)

    # ---- Export VTU file ----
    _write_vtu(step, pointsVTU, elemVTU, dispVTU, stressVTU, islocalList, offsets, types)


def _write_step_result_dat():
    """Write per-step nodal results to *.dat files under ``st.dirnamestep``."""
    step_dir = Path(st.dirnamestep)
    step_dir.mkdir(parents=True, exist_ok=True)

    # Global nodal fields
    _write_vector_dat(step_dir / "u.g.dat", st.disG2D, "Global displacement")
    _write_vector_dat(step_dir / "v.g.dat", st.velG2D, "Global velocity")
    _write_vector_dat(step_dir / "a.g.dat", st.acceG2D, "Global acceleration")

    if int(st.islocal) != 1:
        return

    # Local fields (L only)
    _write_vector_dat(step_dir / "u.l.dat", st.disL2D, "Local displacement (L only)")
    _write_vector_dat(step_dir / "v.l.dat", st.velL2D, "Local velocity (L only)")
    _write_vector_dat(step_dir / "a.l.dat", st.acceL2D, "Local acceleration (L only)")

    # Local total fields (G + L)
    _write_vector_dat(step_dir / "u_gl.l.dat", st.disLG2D, "Local displacement (G+L)")
    _write_vector_dat(step_dir / "v_gl.l.dat", st.velLG2D, "Local velocity (G+L)")
    _write_vector_dat(step_dir / "a_gl.l.dat", st.acceLG2D, "Local acceleration (G+L)")

    # Mesh data for offline J-integral / DSIF post-processing
    node_file = step_dir / "node.l.dat"
    if not node_file.exists():
        _write_node_dat(node_file, st.nodeL, "Local mesh nodes")

    elem_file = step_dir / "elem.l.dat"
    if not elem_file.exists():
        _write_elem_dat(elem_file, st.elemL, "Local mesh elements (Q4, 1-based)")


def _write_vector_dat(path, vec2d, title):
    """Write NodeID + 2-component vector field."""
    arr = np.asarray(vec2d, dtype=float)
    with open(path, "w") as f:
        f.write(f"# {title}\n")
        f.write("# NodeID value_x value_y\n")
        for i, row in enumerate(arr, start=1):
            f.write(f"{i} {row[0]:.12e} {row[1]:.12e}\n")


def _write_node_dat(path, nodes, title):
    arr = np.asarray(nodes, dtype=float)
    with open(path, "w") as f:
        f.write(f"# {title}\n")
        f.write("# NodeID x y\n")
        for i, row in enumerate(arr, start=1):
            f.write(f"{i} {row[0]:.12e} {row[1]:.12e}\n")


def _write_elem_dat(path, elems, title):
    arr = np.asarray(elems, dtype=int)
    with open(path, "w") as f:
        f.write(f"# {title}\n")
        f.write("# ElemID n1 n2 n3 n4\n")
        for i, row in enumerate(arr, start=1):
            # local connectivity is stored as 0-based; export as 1-based
            n1, n2, n3, n4 = row + 1
            f.write(f"{i} {n1} {n2} {n3} {n4}\n")


def _write_vtu(step, points, cells, disp, stress, islocal, offsets, types):
    """Write VTU file for the current step."""
    if getattr(st, "analysis_mode", "dynamic") == "static":
        vtu_dir = Path(st.dirname)
    else:
        vtu_dir = st.pvd / "vtu"
        vtu_dir.mkdir(parents=True, exist_ok=True)
    vtu_file = vtu_dir / f"step_{step:05d}.vtu"
    
    npts = len(points)
    ncells = len(cells)
    
    with open(vtu_file, 'w') as f:
        # VTU header
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{npts}" NumberOfCells="{ncells}">\n')
        
        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for pt in points:
            f.write(f'          {pt[0]:.10e} {pt[1]:.10e} {pt[2]:.10e}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        
        # Cells
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for cell in cells:
            f.write(f'          {cell[0]} {cell[1]} {cell[2]} {cell[3]}\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for off in offsets:
            f.write(f'          {off}\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for typ in types:
            f.write(f'          {typ}\n')
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')
        
        # Point Data
        f.write('      <PointData>\n')
        
        # Displacement
        f.write('        <DataArray type="Float64" Name="Displacement" NumberOfComponents="3" format="ascii">\n')
        for d in disp:
            f.write(f'          {d[0]:.10e} {d[1]:.10e} {d[2]:.10e}\n')
        f.write('        </DataArray>\n')
        
        # Stress (Sxx, Syy, Sxy)
        f.write('        <DataArray type="Float64" Name="Stress_xx" format="ascii">\n')
        for s in stress:
            f.write(f'          {s[0]:.10e}\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Float64" Name="Stress_yy" format="ascii">\n')
        for s in stress:
            f.write(f'          {s[1]:.10e}\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Float64" Name="Stress_xy" format="ascii">\n')
        for s in stress:
            f.write(f'          {s[2]:.10e}\n')
        f.write('        </DataArray>\n')
        
        # IsLocal flag
        f.write('        <DataArray type="Int32" Name="IsLocal" format="ascii">\n')
        for il in islocal:
            f.write(f'          {il}\n')
        f.write('        </DataArray>\n')
        
        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')
