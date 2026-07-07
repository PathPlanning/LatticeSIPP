import copy
import math
import sys, os
sys.path.append(os.path.abspath('../common'))
from DO_searching import BaseSearchSpace, SearchNode, DynamicEnvironment
from DO_structs import DiscreteState, ControlSet

"""
DESCRIPTION:
A naive and purely illustrative implementation of a dynamic state lattice search.
This approach explicitly models time (t) as a discrete, independent dimension in the search space.

WARNING (Performance bottleneck):
Waiting in place is implemented by generating a successor state advanced by exactly 1 tick. 
Because time increments linearly step-by-step, this causes a massive explosion in the state space 
branching factor, making the search notoriously slow. It is provided here strictly for educational 
purposes to demonstrate how 4D search spaces $(i, j, \theta, t)$ function at a base level. 
For practical, high-performance applications, interval-based approaches (SIPP) are highly recommended.
"""

class DynamicState:
    def __init__(self, i, j, theta, t):
        self.i = i
        self.j = j
        self.theta = theta
        self.t = t
    
    def __eq__(self, other: 'DynamicState') -> bool:
        return (self.i == other.i) and (self.j == other.j) and (self.theta == other.theta) and (self.t == other.t)
       
    def __hash__(self) -> int:
        # Time 't' is explicitly modeled as part of the search space! It acts as a 4th dimension.
        # States with identical (i, j, theta) but different 't' are strictly distinct, 
        # as dynamic obstacles alter the feasibility or cost of occupying a cell at different times.
        return hash((self.i, self.j, self.theta, self.t))


class StateLatticeDynamic(BaseSearchSpace):
    def __init__(self, start: DiscreteState, start_tick: int, goal: DiscreteState,
                 control_set: ControlSet, env: DynamicEnvironment,
                 R: float = 0.0, A: int = 0):
        self.start = start
        # Collisions in the interval [0, start_tick) are ignored; the planning phase strictly begins at start_tick.
        self.start_tick = start_tick  
        self.goal = goal
        self.control_set = control_set
        self.env = env
        self.R = R
        self.A = A

    def start_state(self):
        return DynamicState(i=self.start.i, j=self.start.j, theta=self.start.theta, t=self.start_tick)

    def get_successors(self, state: DynamicState):
        # List of tuples: (successor_state, transition_cost, action_ID)
        # Action ID mapping: ID >= 0 denotes a motion primitive; ID < 0 denotes a waiting action in ticks.
        successors = []  
        
        # 1. Generate successors by applying kinematically feasible motion primitives
        for prim in self.control_set.get_primitives_heading(state.theta):  
            di = prim.goal.i
            dj = prim.goal.j
            if self.env.is_primitive_safe(state.i, state.j, state.t, prim):
                next_state = DynamicState(state.i + di, state.j + dj, prim.goal.theta, state.t + prim.exec_ticks)
                successors.append((next_state, prim.exec_ticks, prim.id))

        # 2. Generate a waiting successor (advance time by 1 tick)
        if self.env.is_cell_safe(state.i, state.j, state.t, state.t + 1):
            wait_state = DynamicState(state.i, state.j, state.theta, state.t + 1)
            successors.append((wait_state, 1, -1)) 
            
        return successors

    def is_goal(self, state: DynamicState) -> bool:
        # Validates if the current state falls within the acceptable spatial radius (R) 
        # and angular tolerance (A) of the goal state.
        return (self.goal.i - state.i) ** 2 + (self.goal.j - state.j) ** 2 <= self.R ** 2 and \
                self.control_set.theta.num_dist(self.goal.theta, state.theta) <= self.A

    def heuristic(self, state: DynamicState) -> float:
        TICS_PER_UNIT = 10
        # Euclidean distance scaled to temporal units (admissible heuristic)
        return math.hypot((state.i - self.goal.i), (state.j - self.goal.j)) * TICS_PER_UNIT  

    def reconstruct_path(self, goal_node: SearchNode):
        path = []
        current = goal_node
        while current.parent is not None:
            path.append(current.action)
            current = current.parent
            
        if self.start_tick > 0:
            # If planning started at a delayed tick, prepend an initial wait action
            path.append(-self.start_tick)  
            
        return path[::-1]
