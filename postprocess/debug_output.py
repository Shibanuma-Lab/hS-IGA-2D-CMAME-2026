"""
debug_output - Export detailed mesh, boundary condition, and initial condition
information for each step (for debugging purposes).
"""

import numpy as np
from pathlib import Path

import core.state as st


def write_debug_info(step):
    """
    Write detailed debugging information for the current step.
    
    Outputs:
    - Global IGA mesh: node.g.dat, elem.g.dat, bc.g.dat, init.u.g.dat, init.v.g.dat, init.a.g.dat
    - Local mesh: node.l.dat, elem.l.dat, bc.l.dat, init.u.l.dat, init.v.l.dat, init.a.l.dat
    - Visualization mesh: node.vis.dat, elem.vis.dat
    """
    
    if int(st.debug_output) != 1:
        return
    
    # Get the current step directory
    step_dir = Path(st.dirnamestep)
    
    # ===================== Global IGA Mesh =====================
    # node.g.dat: Node ID, x, y
    if hasattr(st, 'controlPts') and st.controlPts is not None:
        node_g_path = step_dir / "node.g.dat"
        with open(node_g_path, 'w') as f:
            f.write("# Global IGA Control Points\n")
            f.write("# NodeID  X  Y\n")
            for i, pt in enumerate(st.controlPts):
                f.write(f"{i+1}  {pt[0]:.12e}  {pt[1]:.12e}\n")
    
    # elem.g.dat: Element ID, node IDs
    if hasattr(st, 'element') and st.element is not None:
        elem_g_path = step_dir / "elem.g.dat"
        with open(elem_g_path, 'w') as f:
            f.write("# Global IGA Elements\n")
            f.write("# ElemID  NodeIDs...\n")
            for i, elem in enumerate(st.element):
                node_ids = ' '.join(str(int(nid)) for nid in elem)
                f.write(f"{i+1}  {node_ids}\n")
    
    # bc.g.dat: Boundary conditions for global mesh
    if hasattr(st, 'ebc') and st.ebc is not None and len(st.ebc) > 0:
        bc_g_path = step_dir / "bc.g.dat"
        with open(bc_g_path, 'w') as f:
            f.write("# Global Boundary Conditions (Dirichlet)\n")
            f.write("# NodeID  Direction(1=x,2=y)  Value\n")
            for bc in st.ebc:
                node_id_0based = int(bc[0])
                node_id = node_id_0based + 1  # Convert to 1-based indexing for Mathematica compatibility
                direction = int(bc[1])
                value = float(bc[2])
                f.write(f"{node_id}\t{direction}\t{value:.12e}\n")
    
    # init.u.g.dat, init.v.g.dat, init.a.g.dat: Initial conditions
    if hasattr(st, 'disini') and st.disini is not None:
        _write_initial_conditions_global(step_dir, st.disini, "init.u.g.dat", "Displacement")
    if hasattr(st, 'Vini') and st.Vini is not None:
        _write_initial_conditions_global(step_dir, st.Vini, "init.v.g.dat", "Velocity")
    if hasattr(st, 'Aini') and st.Aini is not None:
        _write_initial_conditions_global(step_dir, st.Aini, "init.a.g.dat", "Acceleration")
    
    # ===================== Local Mesh =====================
    if int(st.islocal) == 1:
        # node.l.dat: Node ID, x, y
        if hasattr(st, 'nodeL') and st.nodeL is not None:
            node_l_path = step_dir / "node.l.dat"
            with open(node_l_path, 'w') as f:
                f.write("# Local Mesh Nodes\n")
                f.write("# NodeID  X  Y\n")
                for i, pt in enumerate(st.nodeL):
                    f.write(f"{i+1}  {pt[0]:.12e}  {pt[1]:.12e}\n")
        
        # elem.l.dat: Element ID, node IDs
        if hasattr(st, 'elemL') and st.elemL is not None:
            elem_l_path = step_dir / "elem.l.dat"
            with open(elem_l_path, 'w') as f:
                f.write("# Local Mesh Elements (Q4)\n")
                f.write("# ElemID  Node1  Node2  Node3  Node4\n")
                for i, elem in enumerate(st.elemL):
                    # elemL is 0-based, convert to 1-based
                    node_ids = ' '.join(str(int(nid)+1) for nid in elem)
                    f.write(f"{i+1}  {node_ids}\n")
        
        # bc.l.dat: Boundary conditions for local mesh
        # Extract local BCs from st.ebc (node IDs >= nG indicate local nodes)
        if hasattr(st, 'ebc') and st.ebc is not None and len(st.ebc) > 0:
            bc_l_path = step_dir / "bc.l.dat"
            nG = int(st.nG) if hasattr(st, 'nG') else 0
            
            with open(bc_l_path, 'w') as f:
                f.write("# Local Boundary Conditions (Dirichlet)\n")
                f.write("# NodeID  Direction(1=x,2=y)  Value\n")
                
                # Filter for local nodes (global node numbering >= nG means local)
                for bc in st.ebc:
                    node_id_global = int(bc[0])  # This is global numbering (0-based)
                    if node_id_global >= nG:
                        # Convert to local node ID (1-based)
                        node_id_local = node_id_global - nG + 1
                        direction = int(bc[1])
                        value = float(bc[2])
                        f.write(f"{node_id_local}\t{direction}\t{value:.12e}\n")
        
        # init.u.l.dat, init.v.l.dat, init.a.l.dat: Initial conditions for local
        nG = int(st.nG) if hasattr(st, 'nG') else 0
        if hasattr(st, 'disini') and st.disini is not None and len(st.disini) > nG * 2:
            _write_initial_conditions_local(step_dir, st.disini, nG, "init.u.l.dat", "Displacement")
        if hasattr(st, 'Vini') and st.Vini is not None and len(st.Vini) > nG * 2:
            _write_initial_conditions_local(step_dir, st.Vini, nG, "init.v.l.dat", "Velocity")
        if hasattr(st, 'Aini') and st.Aini is not None and len(st.Aini) > nG * 2:
            _write_initial_conditions_local(step_dir, st.Aini, nG, "init.a.l.dat", "Acceleration")
    
    # ===================== Visualization Mesh =====================
    # node.vis.dat: Node ID, x, y
    if hasattr(st, 'nodeVis') and st.nodeVis is not None:
        node_vis_path = step_dir / "node.vis.dat"
        with open(node_vis_path, 'w') as f:
            f.write("# Visualization Mesh Nodes\n")
            f.write("# NodeID  X  Y\n")
            for i, pt in enumerate(st.nodeVis):
                f.write(f"{i+1}  {pt[0]:.12e}  {pt[1]:.12e}\n")
    
    # elem.vis.dat: Element ID, node IDs
    if hasattr(st, 'elemVis') and st.elemVis is not None:
        elem_vis_path = step_dir / "elem.vis.dat"
        with open(elem_vis_path, 'w') as f:
            f.write("# Visualization Mesh Elements (Q4)\n")
            f.write("# ElemID  Node1  Node2  Node3  Node4\n")
            for i, elem in enumerate(st.elemVis):
                # elemVis is 0-based, convert to 1-based
                node_ids = ' '.join(str(int(nid)+1) for nid in elem)
                f.write(f"{i+1}  {node_ids}\n")


def _write_initial_conditions_global(step_dir, data, filename, label):
    """Write global initial conditions (displacement/velocity/acceleration)."""
    file_path = step_dir / filename
    
    # data is a 1D array with [ux1, uy1, ux2, uy2, ...]
    data = np.asarray(data).ravel()
    nG = int(st.nG) if hasattr(st, 'nG') else len(data) // 2
    
    with open(file_path, 'w') as f:
        f.write(f"# Global Initial {label}\n")
        f.write("# NodeID  X-direction  Y-direction\n")
        
        # Output all nodes with x and y values on the same line
        for i in range(nG):
            ux = data[2*i] if 2*i < len(data) else 0.0
            uy = data[2*i+1] if 2*i+1 < len(data) else 0.0
            
            # Write all values (including zeros) for completeness
            f.write(f"{i+1}\t{ux:.12e}\t{uy:.12e}\n")


def _write_initial_conditions_local(step_dir, data, nG, filename, label):
    """Write local initial conditions (displacement/velocity/acceleration)."""
    file_path = step_dir / filename
    
    # data is a 1D array with [global part, local part]
    # Local part starts at index nG*2
    data = np.asarray(data).ravel()
    nL = (len(data) - nG * 2) // 2
    
    with open(file_path, 'w') as f:
        f.write(f"# Local Initial {label}\n")
        f.write("# NodeID  X-direction  Y-direction\n")
        
        # Output all local nodes with x and y values on the same line
        for i in range(nL):
            idx_x = nG * 2 + 2*i
            idx_y = nG * 2 + 2*i + 1
            
            ux = data[idx_x] if idx_x < len(data) else 0.0
            uy = data[idx_y] if idx_y < len(data) else 0.0
            
            # Write all values (including zeros) for completeness
            f.write(f"{i+1}\t{ux:.12e}\t{uy:.12e}\n")
