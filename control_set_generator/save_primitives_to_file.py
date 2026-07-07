"""
Contains utility functions for generating, manipulating (symmetries), and saving 
kinematic primitives to a standardized text format.
"""

from typing import List, Tuple, Optional
import copy
import matplotlib
import sys
import numpy as np

# Adjust path to point to tools/common/
sys.path.append("../common/")
from DO_structs import *
from DO_graphics import *

# Global constant: base number of ticks per 1 unit of distance (grid cell size)
TICKS_PER_UNIT = 10.0 


def write_metadata(file: str, theta_discrete: Theta) -> None:
    """
    Writes the heading discretization metadata to the beginning of the file.
    This must be called once before appending any primitives.
    """
    with open(file, "w") as f:
        f.write("=== METADATA ===\n")
        f.write(f"NumHeadings: {theta_discrete.theta_amount}\n")
        f.write("Headings(rad):\n")
        for angle in theta_discrete.angles:
            f.write(f"{angle:.6f}\n")
        f.write("---\n\n")


def get_prim_collision(traj: ShortTrajectory, total_time_ticks: int) -> List[Tuple[int, int, float, float]]:
    """
    Calculates a highly precise collision footprint without temporal gaps.
    Time is scaled to perfectly fit into 'total_time_ticks'.
    """
    if traj.length <= 1e-6:
        return []
        
    time_factor = total_time_ticks / traj.length
    
    ds = 0.0001 
    xs = list(traj.sample_x(ds=ds))
    ys = list(traj.sample_y(ds=ds))
    
    # Calculate exact cumulative chord length to avoid generator inaccuracies
    s_vals = [0.0]
    for k in range(1, len(xs)):
        dx = xs[k] - xs[k-1]
        dy = ys[k] - ys[k-1]
        s_vals.append(s_vals[-1] + np.hypot(dx, dy))
        
    # Normalize 's' so the final point STRICTLY matches analytical primitive length
    L_actual = s_vals[-1]
    if L_actual > 0:
        scale = traj.length / L_actual
        s_vals = [s * scale for s in s_vals]

    cell_intervals = {}
    prev_cell = None
    
    for k in range(len(xs)):
        t_current = s_vals[k] * time_factor
        
        i = int(round(ys[k]))
        j = int(round(xs[k]))
        curr_cell = (i, j)
        
        if prev_cell is None:
            prev_cell = curr_cell
            cell_intervals[curr_cell] = [0.0, 0.0]
            
        elif curr_cell != prev_cell:
            # CELL BOUNDARY CROSSED
            t_prev = s_vals[k-1] * time_factor
            t_cross = (t_prev + t_current) / 2.0
            
            # Close previous cell interval (eliminates micro-gaps)
            cell_intervals[prev_cell][1] = max(cell_intervals[prev_cell][1], t_cross)
            
            # Open new cell interval
            if curr_cell not in cell_intervals:
                cell_intervals[curr_cell] = [t_cross, t_cross]
                
            prev_cell = curr_cell
            
        cell_intervals[curr_cell][1] = max(cell_intervals[curr_cell][1], t_current)
        
    # GUARANTEE: The end of the last interval perfectly equals total_time_ticks
    if prev_cell is not None:
        cell_intervals[prev_cell][1] = float(total_time_ticks)
        
    return [(i, j, t_start, t_end) for (i, j), [t_start, t_end] in cell_intervals.items()]


def save_primitive(file: str, prim: ShortTrajectory, prim_id: int, theta_discrete: Theta, ticks_per_unit: float = TICKS_PER_UNIT) -> None: 
    """
    Saves a primitive to file. Fixes the discrete time first, 
    then passes it for exact footprint calculation.
    """
    
    total_time_ticks = int(round(prim.length * ticks_per_unit))
    
    with open(file, "a") as f:  # append to file!
        f.write("=== PRIMITIVE ===\n")
        f.write(f"ID: {prim_id}\n")
        f.write(f"StartHeadingNum: {theta_discrete.num_angle(prim.start.theta)}\n") 
        
        goal_i = int(round(prim.goal.y))
        goal_j = int(round(prim.goal.x))
        goal_h = theta_discrete.num_angle(prim.goal.theta)
        f.write(f"Goal: i={goal_i} j={goal_j} heading={goal_h}\n")

        f.write(f"Length: {prim.length:.6f}\n")
        f.write(f"ExecutionTime(ticks): {total_time_ticks}\n")
        f.write(f"TotalHeadingChange: {prim.total_heading_change():.4f}\n")
        f.write(f"BendingEnergy: {prim.bending_energy():.4f}\n")
        
        f.write("---\nTrajectory(x,y):\n")  
        for x, y in zip(prim.sample_x(ds=0.1), prim.sample_y(ds=0.1)):
            f.write(f"{x:.3f} {y:.3f}\n")  
        
        f.write("---\nCollisionFootprint(i, j, t_in, t_out):\n")
        for i, j, t_in, t_out in get_prim_collision(prim, total_time_ticks):
            f.write(f"{i} {j} {t_in:.6f} {t_out:.6f}\n")

        f.write("=== END ===\n\n")


def central_symmetry(prim: ShortTrajectory) -> List[ShortTrajectory]:
    """
    Generates 4-fold rotational copies (0, 90, 180, 270 degrees) of the primitive.
    """
    x, y, theta, k = prim.goal.x, prim.goal.y, prim.goal.theta, prim.goal.k
    prim_0 = copy.deepcopy(prim)

    prim_90 = copy.deepcopy(prim)
    prim_90.start.theta += np.pi/2
    prim_90.goal = State(-y, x, theta + np.pi/2, k)
    
    prim_180 = copy.deepcopy(prim)
    prim_180.start.theta += np.pi 
    prim_180.goal = State(-x, -y, theta + np.pi, k)
    
    prim_270 = copy.deepcopy(prim)
    prim_270.start.theta += 3 * np.pi / 2
    prim_270.goal = State(y, -x, theta + 3 * np.pi / 2, k)

    return [prim_0, prim_90, prim_180, prim_270]


def y_axis_symmetry(prim: ShortTrajectory) -> ShortTrajectory:
    """
    Generates a mirrored copy of the primitive across the Y-axis (x -> -x).
    Automatically maps headings using the universal transformation (pi - theta)
    and handles parameter sign inversion for parametric curves.
    """
    assert (prim.start.x == 0 and prim.start.y == 0)
    sym = copy.deepcopy(prim)
    
    # Helper to reflect and normalize any angle across the Y-axis
    def reflect_angle(angle: float) -> float:
        reflected = np.pi - angle
        # Normalize strictly to [-pi, pi)
        return float((reflected + np.pi) % (2 * np.pi) - np.pi)

    new_start_theta = reflect_angle(prim.start.theta)
    new_goal_theta = reflect_angle(prim.goal.theta)
    
    # Invert initial curvature by the trajectory state
    k0_sym = -prim.k0
    goal_k_sym = -prim.goal.k
    
    # Dynamically instantiate states using the original object types
    sym.start = type(prim.start)(0.0, 0.0, new_start_theta, k0_sym)
    sym.goal = type(prim.goal)(-prim.goal.x, prim.goal.y, new_goal_theta, goal_k_sym)
        
    # Reflect parametric spiral coefficients
    sym.set_coef_params(-prim.a, -prim.b, -prim.c, prim.length)
        
    return sym
