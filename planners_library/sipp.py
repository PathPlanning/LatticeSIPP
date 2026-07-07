import copy
import math
import sys, os
from typing import Optional
sys.path.append(os.path.abspath('../common'))
from DO_searching import BaseSearchSpace, SearchNode, DynamicEnvironment
from DO_structs import ControlSet, DiscreteState

"""
DESCRIPTION:
Implements the Safe Interval Path Planning (SIPP) algorithm integrated with a 
kinematic State Lattice framework. 

THEORETICAL INSIGHT ON SIPP VALIDITY IN A* / WA*:
In SIPP, the arrival time `self.t` effectively serves as the g-value (path cost).
Unlike traditional static graphs where vertices and edges are predefined and invariant, 
the successors here are not determined uniquely upfront; instead, they depend dynamically 
on the parent's incoming g-value. 

This might initially seem to violate the core assumptions of heuristic search algorithms. 
However, the mathematical guarantees of A* and Weighted A* (WA*) remain completely intact. 
Because A*/WA* operate deterministically, a node is only expanded at the exact moment 
its optimal (or bounded sub-optimal for WA*) g-value is successfully found. 
Once a node is popped from the OPEN list, its time `t` becomes strictly fixed. Consequently, 
the local edge-generation space from this node becomes instantly deterministic and static. 
Thus, the dynamic graph expansion is mathematically equivalent to exploring a pre-defined 
static graph, preserving all completeness and suboptimality bounds.

SIPP defines a search state uniquely as a tuple: (i, j, theta, interval_id). 
The arrival time `t` represents the optimal cost to reach this specific safe interval, 
mapping perfectly onto standard graph duplicate detection mechanisms.
"""

class SIPPState:
    def __init__(self, i: int, j: int, theta: int, interval_id: int, t: float):
        self.i = i
        self.j = j
        self.theta = theta
        self.interval_id = interval_id
        self.t = t  # Earliest possible arrival time into this safe interval
    
    def __eq__(self, other: 'SIPPState') -> bool:
        # CRITICAL: The arrival time 't' is EXCLUDED from state equality checks.
        # States are equivalent if they share the same configuration and safe interval.
        return (self.i == other.i) and (self.j == other.j) and \
               (self.theta == other.theta) and (self.interval_id == other.interval_id)
       
    def __hash__(self) -> int:
        return hash((self.i, self.j, self.theta, self.interval_id))


class SIPPSearchSpace(BaseSearchSpace):
    def __init__(self, start: DiscreteState, start_tick: int, goal: DiscreteState,
                 control_set: ControlSet, env: DynamicEnvironment,
                 R: float = 0.0, A: int = 0, position_only: bool = False):
        self.start = start
        self.start_tick = start_tick
        self.goal = goal
        self.control_set = control_set
        self.env = env
        self.R = R
        self.A = A
        self.position_only = position_only
        if position_only:  # If final heading is ignored, maximize the angular tolerance window
            self.A = self.control_set.theta.theta_amount

        self.start_sipp_state = None

    def start_state(self):
        if not self.env.is_cell_safe(self.start.i, self.start.j, None, None, self.env.Safety.STATIC):
            return None
        for ind, (t1, t2) in enumerate(self.env.get_safe_intervals(self.start.i, self.start.j)):
            if t1 <= self.start_tick < t2:
                self.start_sipp_state = SIPPState(i=self.start.i, j=self.start.j, theta=self.start.theta, interval_id=ind, t=self.start_tick)
                return self.start_sipp_state
        return None

    def _find_earliest_safe_departure(self, state: SIPPState, prim, t_dep_min: float, t_dep_max: float) -> Optional[float]:
        """
        Performs a continuous time-domain search for the earliest valid departure time (t_dep) 
        using motion primitive 'prim' from 'state', bounded within [t_dep_min, t_dep_max].
        """
        invalid_intervals = []

        # 1. Aggregate ALL invalid (colliding) departure intervals for this primitive's swept footprint
        for k in range(prim.U):
            cell_i = state.i + prim.collision_in_i[k]
            cell_j = state.j + prim.collision_in_j[k]

            # Fetch the OCCUPIED temporal intervals (when dynamic obstacles block the cell)
            obs_intervals = self.env.occupied_intervals[cell_i][cell_j]

            for o_start, o_end in obs_intervals:
                # Map the obstacle's occupancy interval to the forbidden departure window for the agent
                inv_start = o_start - prim.t_out[k]
                inv_end = o_end - prim.t_in[k]

                # If the collision window overlaps with our feasible execution window, record it
                if inv_end > t_dep_min and inv_start < t_dep_max:
                    invalid_intervals.append((inv_start, inv_end))

        # If no dynamic obstacles intersect the footprint, depart at the earliest possible time
        if not invalid_intervals:
            return t_dep_min

        # 2. Sort the invalid departure intervals chronologically by their start times.
        # In a compiled language (e.g., C++), this corresponds to std::sort over std::vector<std::pair<double, double>> (highly efficient).
        invalid_intervals.sort(key=lambda x: x[0])

        # 3. Scan sequentially for the first available collision-free temporal gap
        current_t_dep = t_dep_min
        
        for inv_start, inv_end in invalid_intervals:
            # If the current scheduled departure is strictly before the next blocked window
            # (using a epsilon tolerance of 1e-5 against floating-point precision issues), a valid gap is found.
            if current_t_dep <= inv_start - 1e-5:
                return current_t_dep

            # Otherwise, the scheduled departure falls inside or is blocked by the collision interval.
            # We must defer the departure until the obstacle clears the cell.
            current_t_dep = max(current_t_dep, inv_end + 1e-5)

            # If the required wait forces the departure past the maximum allowed boundary, execution fails
            if current_t_dep > t_dep_max:
                return None

        # Final check: if we are still within the permissible window after evaluating all obstacles, proceed
        if current_t_dep <= t_dep_max:
            return current_t_dep

        return None

    def get_successors(self, state: SIPPState):
        successors = []
        
        # Retrieve the safe interval boundaries for the agent's current grid position
        current_intervals = self.env.get_safe_intervals(state.i, state.j)
        safe_start, safe_end = current_intervals[state.interval_id]
        
        # If the start orientation is unconstrained, evaluate the unaligned any-angle primitive bundle
        primitives_bundle = None
        if self.position_only and (state == self.start_sipp_state):
            primitives_bundle = self.control_set.any_theta_bundle
        else:
            primitives_bundle = self.control_set.get_primitives_heading(state.theta)
            
        for prim in primitives_bundle:
            next_i = state.i + prim.goal.i
            next_j = state.j + prim.goal.j
            next_theta = prim.goal.theta
            exec_time = prim.exec_ticks
            
            # Static map layer validation
            if not self.env.is_primitive_safe(state.i, state.j, 0.0, prim, obstacle_check=self.env.Safety.STATIC):
                continue
                
            # Inspect all available safe intervals inside the target grid cell
            next_intervals = self.env.get_safe_intervals(next_i, next_j)
            
            for m, (m_start, m_end) in enumerate(next_intervals):
                # Compute the valid time window to initiate the transition (t_dep).
                # Minimum departure time is constrained by our current time and the opening of the target interval.
                t_dep_min = max(state.t, m_start - exec_time)
                
                # Maximum departure time is constrained by the closure of our current interval 
                # (accounting for footprint clearance times) and the closure of the target interval.
                t_dep_max = min(safe_end - prim.t_out[0], m_end - exec_time)
                
                # If the departure window is inverted, transitioning into interval 'm' via this primitive is impossible
                if t_dep_min > t_dep_max + 1e-5:
                    continue
                    
                valid_t_dep = self._find_earliest_safe_departure(state, prim, t_dep_min, t_dep_max)
                    
                if valid_t_dep is not None:
                    next_t = valid_t_dep + exec_time
                    wait_time = valid_t_dep - state.t
                    
                    next_state = SIPPState(next_i, next_j, next_theta, m, next_t)
                    transition_cost = wait_time + exec_time
                    action = (wait_time, prim.id)
                    successors.append((next_state, transition_cost, action))

        return successors

    def is_goal(self, state: SIPPState) -> bool:
        return (self.goal.i - state.i) ** 2 + (self.goal.j - state.j) ** 2 <= self.R ** 2 and \
                self.control_set.theta.num_dist(self.goal.theta, state.theta) <= self.A

    def heuristic(self, state: SIPPState) -> float:
        TICS_PER_UNIT = 10
        return math.hypot((state.i - self.goal.i), (state.j - self.goal.j)) * TICS_PER_UNIT

    def reconstruct_path(self, goal_node: SearchNode):
        path = []
        current = goal_node
        while current.parent is not None:
            wait_time, prim_id = current.action
            
            # Since path reconstruction traverses the search tree backward from goal to start:
            # 1. Append the motion primitive first
            path.append(prim_id)
            # 2. If a waiting period occurred, append it after the primitive 
            # (which correctly places it BEFORE the primitive once the entire sequence is reversed)
            if wait_time > 0:
                path.append(-wait_time)
                
            current = current.parent

        if self.start_tick > 0:
            path.append(-self.start_tick)  # Prepend an initial wait action if planning started at a delayed tick
            
        # Reverse the accumulated sequence to obtain the chronological path
        return path[::-1]
