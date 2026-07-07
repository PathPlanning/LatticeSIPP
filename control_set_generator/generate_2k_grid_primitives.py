import numpy as np
import os
import matplotlib.pyplot as plt

# Global constant: base number of ticks per 1 unit of distance (grid cell size), the base speed
TICKS_PER_UNIT = 10.0

def get_2k_neighbors(k: int = 2):
    """
    Recursively generates directional offset vectors for 2^k grid connectivity.
    
    k = 2 -> 4-connectivity (orthogonal moves: up, right, left, down)
    k = 3 -> 8-connectivity (adds diagonal transitions)
    k = 4 -> 16-connectivity, etc.
    """
    assert k >= 2
    # Base directions for 4-connectivity
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  
    
    for _ in range(k - 2):
        new_dirs = []
        n = len(directions)
        for i in range(n):
            v = directions[i]
            u = directions[(i + 1) % n]
            new_dirs.append(v)  # Keep the original vector
            new_dirs.append((v[0] + u[0], v[1] + u[1]))  # Add intermediate vector (vector sum)
        directions = new_dirs
    return directions

def get_analytic_collision(dx: int, dy: int) -> list:
    """
    Computes precise grid cell ray-intersections using 2D cross products.
    Uses integer logic to determine the sequence of visited cells and exact 
    algebraic fractions to calculate entrance (t_in) and exit (t_out) timestamps.
    """
    L = np.hypot(dx, dy)
    if L == 0:
        return []
        
    total_time = L * TICKS_PER_UNIT
    
    # Project into the first quadrant to simplify cross-product boundary logic
    abs_dx = abs(dx)
    abs_dy = abs(dy)
    sign_x = 1 if dx >= 0 else -1
    sign_y = 1 if dy >= 0 else -1
    
    intervals = []
    
    # ci, cj represent the current cell coordinates in the projected quadrant
    ci, cj = 0, 0
    s_current = 0.0  # Path progression factor scaled from 0.0 to 1.0
    
    # Traverse grid until the goal cell (abs_dy, abs_dx) is reached
    while cj < abs_dx or ci < abs_dy:
        # Compute the cross product scaled by 2 to maintain pure integer logic.
        # Measures if the ray passes left/above or right/below the grid corner (cj + 0.5, ci + 0.5)
        diff = (2 * cj + 1) * abs_dy - (2 * ci + 1) * abs_dx
        
        if diff < 0:
            # Ray passes to the right/below the corner -> intersects vertical boundary (X-axis step)
            s_next = (2 * cj + 1) / (2.0 * abs_dx)
            intervals.append((ci * sign_y, cj * sign_x, s_current * total_time, s_next * total_time))
            cj += 1
            
        elif diff > 0:
            # Ray passes to the left/above the corner -> intersects horizontal boundary (Y-axis step)
            s_next = (2 * ci + 1) / (2.0 * abs_dy)
            intervals.append((ci * sign_y, cj * sign_x, s_current * total_time, s_next * total_time))
            ci += 1
            
        else:
            # Corner-case: Ray hits the grid corner vertex exactly!
            s_next = (2 * cj + 1) / (2.0 * abs_dx)  # X and Y scales match perfectly here
            intervals.append((ci * sign_y, cj * sign_x, s_current * total_time, s_next * total_time))
            cj += 1
            ci += 1
            
        s_current = s_next
        
    # Append the final segment inside the destination target cell (up to s = 1.0)
    assert 1.0 - s_current > 1e-8
    intervals.append((ci * sign_y, cj * sign_x, s_current * total_time, 1.0 * total_time))
        
    return intervals

def generate_and_save_2k_primitives(k: int, file_path: str):
    """
    Generates straight-line primitives for 2^k grid connectivity.
    Uses a single dummy heading (0.0) since omnidirectional grid search 
    does not restrict subsequent moves based on arrival orientation.
    
    Returns the generated list of directional vectors for inline notebook plotting.
    """
    directions = get_2k_neighbors(k)
    
    # --- METADATA ---
    # We define exactly ONE dummy angle (0.0). 
    # This keeps the state space flat (x, y) for traditional grid planners.
    with open(file_path, "w") as f:
        f.write("=== METADATA ===\n")
        f.write("NumHeadings: 1\n")
        f.write("Headings(rad):\n")
        f.write("0.000000\n")
        f.write("---\n\n")
    
    prim_id = 0
    with open(file_path, "a") as f:
        for (dj, di) in directions:  # dj = X offset, di = Y offset
            length = np.hypot(dj, di)
            if length == 0:
                continue
                
            execution_time = length * TICKS_PER_UNIT
            
            f.write("=== PRIMITIVE ===\n")
            f.write(f"ID: {prim_id}\n")
            f.write("StartHeadingNum: 0\n") 
            f.write(f"Goal: i={di} j={dj} heading=0\n")  # Orientation is discarded
            f.write(f"Length: {length:.6f}\n")
            f.write(f"ExecutionTime(ticks): {execution_time:.6f}\n")
            f.write("TotalHeadingChange: 0.0000\n")
            f.write("BendingEnergy: 0.0000\n")
            
            f.write("---\nTrajectory(x,y):\n")
            # Generate linear trajectory endpoints for visualization tools
            for s in np.linspace(0, 1, num=5):
                x = s * dj
                y = s * di
                f.write(f"{x:.3f} {y:.3f}\n")
                
            f.write("---\nCollisionFootprint(i, j, t_in, t_out):\n")
            # Execute exact cell occupancy analytics
            footprint = get_analytic_collision(dx=dj, dy=di)
            for (i, j, t_in, t_out) in footprint:
                f.write(f"{i} {j} {t_in:.6f} {t_out:.6f}\n")
                
            f.write("=== END ===\n\n")
            prim_id += 1
            
    print(f"Successfully generated {prim_id} primitives for 2^{k} connectivity into: {file_path}")
    return directions
