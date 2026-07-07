import os
import random
import math
import copy
import numpy as np
from typing import Dict, List, Optional

from DO_structs import DiscreteState, ControlSet

"""
MODULE: random_obstacles.py
DESCRIPTION:
Automates the generation of dynamic obstacles with kinematically feasible trajectories
using a State Lattice Search (DFS-based) to guarantee a minimum trajectory length.

GENERATION MODES:
1. Random Walk ('random'):
   - Initialization: Selects a uniformly random valid starting coordinate and heading.
   - Expansion: Iteratively applies randomly selected, kinematically valid motion 
     primitives that match the current state's heading.
   - Fallback (DFS Backtracking): If a trajectory branch hits a dead-end (e.g., 
     colliding with map boundaries before reaching the target length), the algorithm 
     backtracks to the previous state and explores alternative primitive branches.

2. Circular Orbit ('circle'):
   - Initialization: Selects a random starting coordinate. The initial heading is 
     calculated to be nearly perpendicular to the radial vector pointing from the 
     start position to the map's center (creating a tangential starting trajectory).
   - Expansion: Evaluates the bundle of available primitives and greedily prioritizes 
     those that minimize the deviation from the original orbital radius.
   - Fallback (DFS Backtracking): Relies on DFS to recover from boundary collisions, 
     trying the next-best primitives to maintain the circular flow around the center.
"""

class DynamicObstacleGenerator:
    def __init__(self, control_set: ControlSet, task_map, prefix_to_save: str = "./random_obstacles/"):
        random.seed(239239)
        self.cs = control_set
        self.task_map = task_map
        self.theta_config = control_set.theta
        self.prefix_to_save = prefix_to_save
        
        self.center_i = self.task_map.height / 2
        self.center_j = self.task_map.width / 2
        
        if not os.path.exists(self.prefix_to_save):
            os.makedirs(self.prefix_to_save)

    def _get_circular_start_theta(self, i: int, j: int) -> int:
        """Finds the heading (theta) that is most perpendicular to the radial vector from the center."""
        di = i - self.center_i
        dj = j - self.center_j
        
        # Tangent vector calculation (perpendicular to the radius).
        # Yields two directional options: clockwise and counter-clockwise orbits.
        target_angles = [
            math.atan2(dj, -di), # Option 1
            math.atan2(-dj, di)  # Option 2
        ]
        target_angle = random.choice(target_angles)
        
        # Determine the closest discrete heading index in theta_config
        best_theta = 0
        min_diff = float('inf')
        for idx, angle in enumerate(self.theta_config.angles):
            diff = abs(math.atan2(math.sin(angle - target_angle), math.cos(angle - target_angle)))
            if diff < min_diff:
                min_diff = diff
                best_theta = idx
        return best_theta

    def _is_primitive_in_bounds(self, state: DiscreteState, prim, radius: int) -> bool:
        R_ceil = int(math.ceil(radius))
        for k in range(prim.U):
            cell_i = state.i + prim.collision_in_i[k]
            cell_j = state.j + prim.collision_in_j[k]
            if not (self.task_map.in_bounds(cell_i - R_ceil, cell_j - R_ceil) and
                    self.task_map.in_bounds(cell_i + R_ceil, cell_j + R_ceil)):
                return False
        return True

    def _find_path_dfs(self, current_state: DiscreteState, radius: int, target_len: int, 
                       gen_type: str, orbital_radius: float, current_path: List[int]) -> Optional[List[int]]:
        """Recursive search for a path of target length (State Lattice Search approach)."""
        if len(current_path) >= target_len:
            return current_path

        bundle = self.cs.get_primitives_heading(current_state.theta)
        
        # Filter valid primitives (collision-free w.r.t map boundaries and obstacle radius)
        valid_options = []
        for prim in bundle:
            if self._is_primitive_in_bounds(current_state, prim, radius):
                # Calculate primitive "cost" for heuristic sorting
                if gen_type == 'circle':
                    new_i, new_j = current_state.i + prim.goal.i, current_state.j + prim.goal.j
                    cost = abs(math.hypot(new_i - self.center_i, new_j - self.center_j) - orbital_radius)
                else:
                    cost = random.random() # For random generation, assign random cost for shuffling
                valid_options.append((cost, prim))

        # Sort: prioritize the most suitable primitives (e.g., minimal deviation from orbital radius)
        valid_options.sort(key=lambda x: x[0])
        
        # Introduce stochastic variance into the top choices to diversify random walks
        """
        if gen_type == 'random' or len(valid_options) > 5:
            top_slice = valid_options[:5]
            random.shuffle(top_slice)
            valid_options[:5] = top_slice   
        """
        for _, prim in valid_options:
            # Execute the step / apply the primitive
            next_state = DiscreteState(
                i=current_state.i + prim.goal.i,
                j=current_state.j + prim.goal.j,
                theta=prim.goal.theta
            )
            
            # Recursive depth-first call
            res = self._find_path_dfs(next_state, radius, target_len, gen_type, 
                                      orbital_radius, current_path + [prim.id])
            if res is not None:
                return res
                
        return None # Dead-end reached, backtrack

    def generate_batch(self, size_distribution: Dict[int, int], min_length: int = 20, gen_type: str = 'random'):
        tasks = []
        for radius, count in size_distribution.items():
            tasks.extend([radius] * count)
        random.shuffle(tasks)

        for idx, radius in enumerate(tasks):
            # 1. Start position extraction
            R_ceil = int(math.ceil(radius))
            start_i = random.randint(R_ceil, self.task_map.height - R_ceil - 1)
            start_j = random.randint(R_ceil, self.task_map.width - R_ceil - 1)
            
            # 2. Start heading resolution
            if gen_type == 'circle':
                start_theta = self._get_circular_start_theta(start_i, start_j)
            else:
                start_theta = random.randint(0, self.theta_config.theta_amount - 1)
            
            start_state = DiscreteState(i=start_i, j=start_j, theta=start_theta)
            orbital_radius = math.hypot(start_i - self.center_i, start_j - self.center_j)

            # 3. DFS pathfinding (guarantees the specified minimum trajectory length)
            trajectory_ids = self._find_path_dfs(start_state, radius, min_length, gen_type, orbital_radius, [])
            
            if trajectory_ids:
                self._save_to_file(idx + 1, start_state, radius, trajectory_ids)
            else:
                print(f"⚠️ Failed to find a valid path for object {idx+1}")
    
    def _save_to_file(self, file_idx, start_state, radius, trajectory_ids):
        filename_txt = os.path.join(self.prefix_to_save, f"dynamic_obstacle_{file_idx}.txt")
        
        with open(filename_txt, 'w', encoding='utf-8') as f:
            f.write(f"======= Dynamic obstacle description ======\n")
            f.write(f"R: {radius}\n")
            f.write(f"Start: i={start_state.i} j={start_state.j} theta={start_state.theta}\n")
            f.write(f"PrimitivesIDs: {' '.join(map(str, trajectory_ids))}\n")
