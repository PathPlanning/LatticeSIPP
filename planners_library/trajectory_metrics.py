import math
import sys
import os

sys.path.append(os.path.abspath('../common'))
from DO_structs import *

"""
MODULE: trajectory_metrics.py
DESCRIPTION:
This module provides a suite of evaluation metrics to analyze the geometric 
and kinematic quality of generated paths. It computes angularity, cumulative 
heading changes, and bending energy, properly accounting for the differences 
between kinematically smooth trajectories and piecewise-linear (angular) paths.
"""

# Relates to Metric 1 (Angularity). 
# Defines the temporal step size (in ticks) used to sample discrete points 
# along the simulated continuous trajectory to calculate the circumscribed radius.
CURVATURE_SAMPLE_TICKS = 2

# Relates to Metric 3 (Bending Energy). 
# Used to approximate the curvature penalty at sharp junctions in angular paths 
# (where actual turning radius is zero and curvature is theoretically infinite). 
# The value 0.5 is deliberately chosen because a circle with a radius of 0.5 
# fits exactly within a single grid cell (cell side length of 1). 
# Within our discrete grid framework, a turn of this minimal radius is 
# practically indistinguishable from pivoting in place.
MIN_BEND_RADIUS = 0.5

def get_prim_ids(path):
    """
    Extracts only the primitive IDs from the raw path 
    (ignores waiting actions which are recorded as floats < 0).
    """
    return [int(act) for act in path if act >= 0]

def compute_curvature(prim_ids, control_set):
    """
    Metric 1: Angularity (Sum of 1/R).
    Calculates the sum of the inverse radii of circumscribed circles for triangles 
    formed by sequentially sampled points along the trajectory.
    This is a purely geometric measure of path jaggedness and is not normalized by length.
    
    :param prim_ids: List of primitive IDs in the robot's trajectory.
    :param control_set: The ControlSet containing the actual primitive objects.
    """
    if len(prim_ids) < 1:
        return 0.0

    # Create a dummy robot solely to traverse its trajectory
    robot = DynamicObstacle(control_set)
    # We only care about the shape of the path, so start coordinates are set to 0
    robot.set_parametres(R=None, start_i=0, start_j=0, start_theta=None, path=prim_ids)

    # Sample points along the robot's trajectory
    points = []
    t = 0
    while True:
        p = robot.get_position(t)
        if p is None:
            break
        points.append(p)
        t += CURVATURE_SAMPLE_TICKS  # Advance to the next tick

    # Slide a window of 3 points to calculate the sum of 1/R
    total_curvature = 0.0
    for i in range(len(points) - 2):
        Ax, Ay = points[i]
        Bx, By = points[i+1]
        Cx, Cy = points[i+2]

        c = math.hypot(Bx - Ax, By - Ay)
        a = math.hypot(Cx - Bx, Cy - By)
        b = math.hypot(Cx - Ax, Cy - Ay)

        if a == 0 or b == 0 or c == 0:
            continue

        # Calculate triangle area via cross product
        area = 0.5 * abs((Bx - Ax)*(Cy - Ay) - (By - Ay)*(Cx - Ax))  
        # If the area is near zero, the points are collinear -> radius is infinite -> 1/R = 0
        if area < 1e-6:                                              
            continue 

        # Formula for circumscribed circle radius: R = (a*b*c) / (4*S) 
        # Therefore, the curvature is: 1/R = (4*S) / (a*b*c)
        total_curvature += (4 * area) / (a * b * c)

    return total_curvature

def get_angular_delta_theta(p1, p2):
    """
    Calculates the rotation angle at the junction of two line-segment 
    primitives (used primarily for angular/piecewise-linear paths).
    """
    # Direction vectors of the segment primitives (relative to their local origins)
    v1x, v1y = p1.goal.j - p1.start.j, p1.goal.i - p1.start.i  
    v2x, v2y = p2.goal.j - p2.start.j, p2.goal.i - p2.start.i
    
    dot = v1x*v2x + v1y*v2y  # Dot product
    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    
    if mag1 > 0 and mag2 > 0:
        # Guard against float rounding errors to prevent math.acos domain errors
        cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))  
        return math.acos(cos_theta)
    return 0.0

def compute_heading_change(prim_ids, control_set, is_angular):
    """
    Metric 2: Total heading change in radians along the path.
    
    Smooth paths: Sum of the precalculated 'total_heading_change' for each primitive 
                  (since their junctions are geometrically smooth, there are no instantaneous angle changes there).
    Angular paths: Sum of the junction angles between straight segments 
                   (since 'total_heading_change' is 0 along the straight segments themselves, only the junctions contribute).
    """
    if not prim_ids:
        return 0.0

    if not is_angular:
        return sum(control_set.get_primitive(pid).total_heading_change for pid in prim_ids)

    total_angle = 0.0
    for i in range(len(prim_ids) - 1):
        p1 = control_set.get_primitive(prim_ids[i])
        p2 = control_set.get_primitive(prim_ids[i+1])
        
        total_angle += get_angular_delta_theta(p1, p2)
            
    return total_angle

def compute_bending_energy(prim_ids, control_set, is_angular):
    """
    Metric 3: Bending Energy (Integral of squared curvature along the trajectory).
    
    Smooth paths: A simple sum of the primitives' internal bending energies 
                  (the junction is a single point/set of measure 0 with finite curvature, which does not affect the integral).
    Angular paths: The energy at junctions is calculated via (Angle / MIN_BEND_RADIUS). 
                   Since energy is 0 along the straight segments, the junctions are points of infinite curvature 
                   (turning radius is zero). To avoid infinite energy, we approximate the turn using a minimal feasible radius.
    """
    if not prim_ids:
        return 0.0

    energy = 0.0
    for i in range(len(prim_ids)):
        prim = control_set.get_primitive(prim_ids[i])
        # Add the internal energy of the primitive (for 2^k grids this is likely 0, but added for architectural safety)
        energy += prim.bending_energy
        
        # For angular paths, we add a "penalty" energy for the sharp junctions
        if is_angular and i < len(prim_ids) - 1:
            next_prim = control_set.get_primitive(prim_ids[i+1])
            
            delta_theta = get_angular_delta_theta(prim, next_prim)  
            
            # Derivation:
            # If the turn occurs along a circular arc of radius R_min by an angle delta_theta, 
            # the curvature is constant: kappa = 1 / R_min.
            # The arc length over which the turn occurs is: L = R_min * delta_theta.
            # Therefore, the bending energy at the junction is:
            # E_junction = kappa^2 * L = (1 / R_min)^2 * (R_min * delta_theta) = delta_theta / R_min
            energy += delta_theta / MIN_BEND_RADIUS
                
    return energy
