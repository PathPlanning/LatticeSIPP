"""
This module defines the core data structures and functions required for A* search 
on graph structures (e.g., State Lattice or Mesh graphs).
"""

import numpy as np
from heapq import heappop, heappush
import bisect
import glob
import math
import copy, time
from abc import ABC, abstractmethod
from enum import IntFlag

from typing import List, Tuple, Any, Union, Optional, Set
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self 

from DO_structs import DiscreteState, DynamicObstacle, ObstacleSet

# Generic type for a graph vertex
VertexType = Union[DiscreteState]


class Map:  
    """
    Represents the discrete workspace (occupancy grid).
    Stored as a binary matrix where '1' indicates an obstacle and '0' indicates free space.
    
    Coordinate System Note:
    The map is stored in matrix format (row i, column j), where 'i' increases downwards 
    and 'j' increases to the right. 
    
    When using motion primitives generated in a Cartesian frame (where Y usually points up),
    a coordinate transformation may be required. Typically, if the primitives assume Y-up, 
    but the map 'i' is down, the heading angles on the map will appear mirrored (clockwise vs counter-clockwise).
    This can be resolved by negating the starting heading (-theta) during the search initialization.
    """
    
    def __init__(self) -> None:
        self.cells = None 
        self.width = 0
        self.height = 0
             
    
    def read_from_movingai_file(self, file: str):
        with open(file) as f:
            f.readline()
            f.readline()
            f.readline()
            assert f.readline() == "map\n"  # skip MovingAI-format lines
            self.convert_string_to_cells(f.read())


    def convert_string_to_cells(self, cell_str: str, obs: bool = True) -> Self:
        """
        Parses a string representation of the map into a binary matrix.
        
        Args:
            cell_str: String containing the map ('.' for free, '#' or '@' for obstacles).
            obs: If False, ignores obstacles and treats the entire map as traversable.
        """
        
        # Normalize line endings
        cell_str = cell_str.replace("\r\n", "\n")
        cell_lines = cell_str.split("\n")
        
        cells_list = []

        for line in cell_lines:
            if len(line) == 0:
                continue
            # Ignore lines that are likely metadata (do not start with map characters)
            if line[0] not in ['.', '@', '#', 'T']:
                continue
            
            cells_row = []
            for char in line:
                if char == '.':
                    cells_row.append(0)
                elif (char == '#') or (char == '@') or (char == 'T'):
                    cells_row.append(1 if obs else 0)
                elif (char == ' '):
                    continue
                else:
                    raise Exception(f"Unknown character '{char}' in map string.")
                
            cells_list.append(cells_row)
            
        self.cells = np.array(cells_list, dtype=np.int8)
        self.width = self.cells.shape[1]
        self.height = self.cells.shape[0]
        
        return self


    def in_bounds(self, i: int, j: int) -> bool:
        """Checks if coordinates (i, j) are within map boundaries."""
        return (0 <= i < self.height) and (0 <= j < self.width)


    def traversable(self, i: int, j: int) -> bool:
        """Checks if cell (i, j) is free (not an obstacle)."""
        return (self.cells[i][j] == 0)            
    
    
    def get_slice(self, si: int, fi: int, sj: int, fj: int) -> Self:
        """
        Crops the map to the specified bounds [si, fi) and [sj, fj).
        Useful for performing search on a sub-region.
        """
        self.cells = self.cells[si:fi, sj:fj]
        self.width = self.cells.shape[1]
        self.height = self.cells.shape[0]
        return self


class DynamicEnvironment:
    """
    Manages the spatial-temporal map for dynamic path planning.
    """
    def __init__(self, task_map, robot_radius=0.0):
        self.task_map: Map = task_map
        self.robot_radius = robot_radius
        self.obstacles = []
        
        # Grid-based lookup: cell (i, j) --> list of time intervals [(t1, t2), ...] representing
        # the periods when the cell is occupied by dynamic obstacles:
        self.occupied_intervals = [
            [[] for _ in range(self.task_map.width)] 
                for _ in range(self.task_map.height)
        ]
        # Complementary lookup: (i, j) --> list of safe (free) time intervals.
        # Essential for algorithms like SIPP to identify traversable time windows:
        self.safe_intervals = [
            [[] for _ in range(self.task_map.width)] 
                for _ in range(self.task_map.height)
        ]

    def get_dynamic_obstacles_set(self, is_copy=True):
        res = ObstacleSet()
        if is_copy:
            res.obstacles = copy.deepcopy(self.obstacles)
        else:
            res.obstacles = self.obstacles
        return res
    
    def load_obstacles_from_dir(self, file_pattern: str, control_set, max_items: int = 1000):
        """
        Populates the environment with a group of dynamic obstacles sharing a common 
        control_set. This method can be called multiple times to batch-load distinct 
        subsets of obstacles, each governed by a specific control set.
        """
        # The file order returned by glob.glob() is OS-dependent and not guaranteed.
        # We enforce sorting to ensure a consistent, deterministic order across all platforms.
        file_list = sorted(glob.glob(file_pattern))
        for i, file_path in enumerate(file_list):
            if i >= max_items:
                return
            self.obstacles.append(DynamicObstacle(control_set).load_from_file(file_path))
            
    def compile_safe_intervals(self):
        """
        Constructs Safe Intervals by projecting the exact precomputed swept-volumes 
        of obstacle primitives onto the grid, dilated by the combined collision radius.
        """
        
        for obs in self.obstacles:
            curr_i, curr_j = obs.start.i, obs.start.j
            curr_t = 0.0
            
            # Minkowski sum radius: dilate the footprint once to treat the ego-robot as a point
            total_R = obs.R + self.robot_radius
            R_ceil = int(np.ceil(total_R))
            
            for ID in obs.path:
                if ID >= 0:
                    prim = obs.control_set.list_all_prims[ID]
                    
                    # Iterate through the precomputed swept-volume of the primitive
                    for k in range(prim.U):
                        base_i = curr_i + prim.collision_in_i[k]
                        base_j = curr_j + prim.collision_in_j[k]
                        t_in = curr_t + prim.t_in[k]
                        t_out = curr_t + prim.t_out[k]
                        
                        # Dilate the specific cell by the configuration space radius
                        for di in range(-R_ceil, R_ceil + 1):
                            for dj in range(-R_ceil, R_ceil + 1):
                                if math.hypot(di, dj) <= total_R:
                                    i, j = base_i + di, base_j + dj
                                    if self.task_map.in_bounds(i, j):
                                        self.occupied_intervals[i][j].append((t_in, t_out))
                    
                    # Update state (prim.goal stores relative coordinate displacements)
                    curr_i += prim.goal.i
                    curr_j += prim.goal.j
                    curr_t += prim.exec_ticks
                    
                else:
                    # ID < 0 represents a dwell/wait action
                    duration = -ID
                    for di in range(-R_ceil, R_ceil + 1):
                        for dj in range(-R_ceil, R_ceil + 1):
                            if math.hypot(di, dj) <= total_R:
                                i, j = curr_i + di, curr_j + dj
                                if self.task_map.in_bounds(i, j):
                                    self.occupied_intervals[i][j].append((curr_t, curr_t + duration))
                    
                    curr_t += duration

        # Merge overlaps and invert to generate strictly disjoint Safe Intervals
        for i in range(self.task_map.height):
            for j in range(self.task_map.width):
                merged = self._merge_intervals(self.occupied_intervals[i][j])
                self.occupied_intervals[i][j] = merged
                self.safe_intervals[i][j] = self._invert_to_safe(merged)

    def _merge_intervals(self, intervals: list) -> list:
        """Merges overlapping temporal intervals in O(N log N)."""
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        
        merged = [intervals[0]]
        for current in intervals[1:]:
            previous = merged[-1]
            if current[0] <= previous[1] + 1e-5:
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)
        return merged

    def _invert_to_safe(self, occupied: list) -> list:
        """Inverts a list of occupied intervals into Safe Intervals."""
        safe = []
        t = 0.0  # end of the last occupied interval; potential start of the next safe interval...
        for t1, t2 in occupied:
            if t1 > t + 1e-5:
                safe.append((t, t1))
            t = max(t, t2)
        safe.append((t, float('inf')))
        return safe

    def get_safe_intervals(self, i: int, j: int) -> list:
        return self.safe_intervals[i][j]

    class Safety(IntFlag):
        STATIC = 1
        DYNAMIC = 2

    def is_cell_safe(self, i: int, j: int, t_in: float, t_out: float,
                     obstacle_check: Safety = Safety.STATIC | Safety.DYNAMIC) -> bool:
        """
        O(log N) verification to determine if a cell is collision-free during a specific time window.
        """

        if self.Safety.STATIC in obstacle_check:
            # 1. Map bounds and static obstacle check
            R_ceil = int(math.ceil(self.robot_radius))
            for di in range(-R_ceil, R_ceil + 1):
                for dj in range(-R_ceil, R_ceil + 1):
                    if math.hypot(di, dj) <= self.robot_radius:
                        if not self.task_map.in_bounds(i+di, j+dj) or not self.task_map.traversable(i+di, j+dj):
                            return False

        if self.Safety.DYNAMIC in obstacle_check:
            # 2. Dynamic obstacle check via Safe Intervals
            intervals = self.get_safe_intervals(i, j)
            if not intervals:
                return False
                
            # Binary search for the correct interval
            idx_bisect = bisect.bisect_right(intervals, t_in, key=lambda x: x[0]) - 1
            
            if idx_bisect >= 0:
                safe_start, safe_end = intervals[idx_bisect]
                if safe_start <= t_in and t_out <= safe_end:
                    return True
            return False
        
        return True

    def is_primitive_safe(self, start_i: int, start_j: int, start_t: float, prim,
                          obstacle_check: Safety = Safety.STATIC | Safety.DYNAMIC) -> bool:
        """
        Validates the entire swept-volume of a motion primitive against dynamic obstacles.
        """
        for k in range(prim.U):
            cell_i = start_i + prim.collision_in_i[k]
            cell_j = start_j + prim.collision_in_j[k]
            abs_t_in = start_t + prim.t_in[k]
            abs_t_out = start_t + prim.t_out[k]
            
            # If even a single cell is occupied during its time window, the primitive is invalid
            if not self.is_cell_safe(cell_i, cell_j, abs_t_in, abs_t_out, obstacle_check):
                return False
                
        return True


class SearchNode:
    def __init__(self, f, g, state, parent=None, action=None):
        self.f = f
        self.g = g
        self.state = state
        self.parent = parent
        self.action = action
        # Tie-breaking: in case of equal f-values, prefer the node with the higher g-value (deeper in the search tree)
        self.tie_breaker = -g 

    def __lt__(self, other):
        # This method is invoked by heapq to compare and order nodes within the priority queue
        if self.f == other.f:
            return self.tie_breaker < other.tie_breaker
        return self.f < other.f
    
      
class SearchTreePQD:
    def __init__(self) -> None:
        self._open: List[SearchNode] = []  # Priority queue containing SearchNodes
        self._closed: Set[Any] = set()     # Set of expanded states (graph vertices)
        self._enc_open_duplicates = 0       

    def open_is_empty(self) -> bool:
        return len(self._open) == 0

    def add_to_open(self, node: 'SearchNode') -> None:
        """Adds a node to the OPEN list. Duplicate detection is deferred (lazy approach)."""
        heappush(self._open, node)

    def was_expanded(self, state: Any) -> bool:
        """Checks if the given STATE has already been expanded (present in CLOSED)."""
        return state in self._closed

    def add_to_closed(self, state: Any) -> None:
        """Marks the given STATE as expanded by adding it to the CLOSED set."""
        self._closed.add(state)

    def get_best_node_from_open(self) -> Optional['SearchNode']:
        """
        Extracts the best node from the priority queue. 
        If the node's state has already been expanded (present in CLOSED), it is discarded.
        """
        while self._open:
            best_node = heappop(self._open)
            
            # If this state is already in CLOSED, a shorter path to it was found earlier. 
            # It is a duplicate and can be safely ignored.
            if self.was_expanded(best_node.state):
                self._enc_open_duplicates += 1
                continue
                
            return best_node            
            
        return None

    @property
    def number_of_expanded_nodes(self) -> int:
        """The actual number of unique states that were successfully expanded."""
        return len(self._closed)


class BaseSearchSpace(ABC):
    """
    Universal interface defining the environment and state transitions for A* search.
    """
    
    @abstractmethod
    def start_state(self) -> Any:
        """
        Returns the initial state from which the search begins.
        Crucially, it must return an instance of the specific state type used in the 
        underlying search graph/space.
        """
        pass

    @abstractmethod
    def get_successors(self, state: Any) -> List[Tuple[Any, float, Any]]:
        """
        Generates valid successors for a given state.
        Returns a list of tuples: (next_state, transition_cost, action_taken),
        where action_taken is the payload to be recorded in the final path (e.g., primitive ID).
        """
        pass
        
    @abstractmethod
    def is_goal(self, state: Any) -> bool:
        pass
        
    @abstractmethod
    def heuristic(self, state: Any) -> float:
        pass

    @abstractmethod
    def reconstruct_path(self, goal_node: SearchNode) -> Any:
        """
        Domain-specific reconstruction of the path from the goal node back to the start.
        Returns a fully formatted trajectory ready for execution or visualization.
        """
        pass


def astar(search_space: BaseSearchSpace, w: float = 1.0, max_time_sec=180.0):
    """
    Standard A* search algorithm running on top of any graph topology that
    implements the BaseSearchSpace interface.
    
    Supports Weighted A* (WA*) through the inflation parameter 'w'. 
    By default, w = 1.0, which resolves to standard, optimal A* search.
    """

    time_start = time.perf_counter()
    ast = SearchTreePQD()
    steps = 0

    start_state = search_space.start_state()
    if not start_state:
        return False, None, steps, None, ast

    h_start = search_space.heuristic(start_state)
    start_node = SearchNode(f=w*h_start, g=0.0, state=start_state)
    ast.add_to_open(start_node)

    while not ast.open_is_empty():
        if time.perf_counter() - time_start > max_time_sec:  # Operational time limit exceeded
            break
        
        current = ast.get_best_node_from_open()
        if current is None:
            break
        
        # Extracting a node guarantees that the shortest path to this state has been found
        ast.add_to_closed(current.state)

        if search_space.is_goal(current.state):
            final_trajectory = search_space.reconstruct_path(current)
            return True, final_trajectory, steps, current.g, ast
            
        # Retrieve neighboring transitions specific to the current search space topology
        for next_state, cost, action in search_space.get_successors(current.state):
            # print("!", next_state.i, next_state.j, cost, action)
            if ast.was_expanded(next_state):
                continue
                
            g_new = current.g + cost
            h_new = search_space.heuristic(next_state)
            
            new_node = SearchNode(
                f=g_new + w * h_new, 
                g=g_new, 
                state=next_state, 
                parent=current, 
                action=action
            )
            
            ast.add_to_open(new_node)
            
        steps += 1
        
    return False, None, steps, None, ast

