"""
This file defines the core data structures.
"""

import numpy as np
from scipy.integrate import quad
import copy
import glob

from typing import List, Tuple, Optional, Union  # Type hinting imports to ensure code clarity and reduce errors.
try:
    from typing import Self               # 'Self' type is available in typing module since Python 3.11. 
except ImportError:
    from typing_extensions import Self    # For older versions, we use typing_extensions.


class State:
    """
    Represents the 4-dimensional continuous state of a mobile agent: 
    coordinates (x, y), heading angle (theta), and curvature (k).
    """
    
    def __init__(self, x: float, y: float, theta: float, k: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.k = k


class DiscreteState:
    """
    Represents a discrete state: two integer coordinates (i, j) and a discrete heading index (theta).
    """
    
    def __init__(self, i: int, j: int, theta: int) -> None:
        self.i = i
        self.j = j
        self.theta = theta
        
    def __eq__(self, other: 'DiscreteState') -> bool:
        return (self.i == other.i) and (self.j == other.j) and (self.theta == other.theta)
       
    def __hash__(self) -> int:
        return hash((self.i, self.j, self.theta))  # Hashing function. Returns an integer used for addressing the DiscreteState in a hash table (e.g., the CLOSED set).
        

class ShortTrajectory:
    """
    Class for storing and generating a short feasible trajectory between two states.
    
    A short trajectory is defined by a fixed 'start' state and a set of internal parameters.
    We also store a 'goal' state, which is the ideal target. The optimization process (Newton's method)
    adjusts the parameters so that the actual 'final_state' of the trajectory matches the 'goal'.
    """
    
    def __init__(self, start: State, goal: State) -> None:
        # Fixed endpoints
        self.start = start
        self.goal = goal
        self.k0 = self.start.k  # Initial curvature is fixed by the start state
        
        # Parametrization 1: Polynomial curvature coefficients (a, b, c) and length
        self.a = None
        self.b = None
        self.c = None
        self.length = None

        # Parametrization 2: Curvature values at specific points and log_length
        self.log_length = None
        self.k1 = None
        self.k2 = None
        self.kf = None

        # Vectorize coordinate functions to allow batch processing of 's' (arc length) values
        self.vect_x = np.vectorize(self.x)
        self.vect_y = np.vectorize(self.y)

    def set_coef_params(self, a: float, b: float, c: float, length: float) -> Self:
        assert length >= 0, "Length cannot be negative!"
        
        # Set params for parametrization 1
        self.length = length  
        self.a = a
        self.b = b
        self.c = c
        
        # Calculate params for parametrization 2 (derived from 1)
        self.log_length = np.log(length)
        self.k1 = self.k(1/3 * length)
        self.k2 = self.k(2/3 * length)
        self.kf = self.goal.k  # Convenience of param 2: final curvature is explicitly the goal curvature

        return self

    def set_curve_params(self, k1: float, k2: float, log_length: float) -> Self:
        # Set params for parametrization 2
        self.log_length = log_length
        self.k1 = k1
        self.k2 = k2
        self.kf = self.goal.k
        
        # Calculate params for parametrization 1 (Polynomial coefficients)
        self.length = np.exp(log_length)
        vect_s = np.array([0, 1/3 * self.length, 2/3 * self.length, self.length]).reshape(-1, 1)
        mat_s = np.hstack((vect_s ** 0, vect_s ** 1, vect_s ** 2, vect_s ** 3))
        params = np.linalg.inv(mat_s) @ np.array([self.k0, self.k1, self.k2, self.kf])
        k0_check, a, b, c = params
        self.a, self.b, self.c = a, b, c
        assert abs(k0_check - self.k0) < 1e-4, "Error in parametrization conversion: calculated k0 mismatch."
                                                                                               
        return self

    def k(self, s: float) -> float:
        assert s >= 0, "Parameter s must be non-negative!"
        return self.k0 + self.a * s + self.b * s**2 + self.c * s**3
    
    def theta(self, s: float) -> float:
        assert s >= 0, "Parameter s must be non-negative!"
        theta0, k0 = self.start.theta, self.k0
        a, b, c = self.a, self.b, self.c
        return theta0 + k0 * s + a/2 * s**2 + b/3 * s**3 + c/4 * s**4

    def x(self, s: float) -> float:
        """Returns x-coordinate at arc length s (via numerical integration)."""
        assert s >= 0, "Parameter s must be non-negative!"
        x0 = self.start.x
        return x0 + quad(lambda x: np.cos(self.theta(x)), 0, s, limit=200, limlst=10)[0]
    
    def y(self, s: float) -> float:
        assert s >= 0, "Parameter s must be non-negative!"
        y0 = self.start.y
        return y0 + quad(lambda x: np.sin(self.theta(x)), 0, s, limit=200, limlst=10)[0]

    def total_heading_change(self) -> float:
        """
        Calculates the total accumulated change in heading angle along the trajectory.
        Dividing this value by the length yields the AOL (Angle-Over-Length) metric, 
        used to evaluate path "wobbliness" (see Bench-MR: https://eric-heiden.com/publication/2021-benchmr-ral-icra/2021-benchmr-ral-icra.pdf).
        """ 
        return quad(lambda s: abs(self.k(s)), 0, self.length, limit=200, limlst=10)[0]
    
    def bending_energy(self) -> float:
        """
        Integral of k(s)^2 for s=0...length. Represents bending energy, indicating how jerky (lacking smoothness)
        the curve is and how difficult it is to follow.
        """ 
        return quad(lambda s: (self.k(s)) ** 2, 0, self.length, limit=200, limlst=10)[0]
    
    def sample_x(self, ds: float = 0.02) -> np.ndarray:
        num = int(self.length / ds)
        return self.vect_x(np.linspace(0, self.length, num=num, endpoint=True))
    
    def sample_y(self, ds: float = 0.02) -> np.ndarray:
        num = int(self.length / ds)
        return self.vect_y(np.linspace(0, self.length, num=num, endpoint=True))

    def state(self, s: float) -> State:
        return State(self.x(s), self.y(s), self.theta(s), self.k(s))

    def final_state(self) -> State:
        return self.state(self.length)
    
    
class Primitive:
    """
    Class representing a generated motion primitive between two discrete states.
    Includes time intervals for continuous collision checking.
    """
    
    def __init__(self, start: DiscreteState, goal: DiscreteState) -> None:
        self.start = start
        self.goal = goal
        
    def set_description(self, x_coords: np.ndarray, y_coords: np.ndarray, length: float, 
                              i_coords: Union[np.ndarray, None], j_coords: Union[np.ndarray, None], 
                              t_in: Union[np.ndarray, None], t_out: Union[np.ndarray, None], 
                              exec_ticks: float, total_heading_change: Union[float, None],
                              bending_energy: Union[float, None]) -> Self:
        """
        Stores detailed information about the primitive, including continuous time intervals.
        """
        self.x_coords = np.concatenate(([self.start.j], x_coords, [self.goal.j])).astype(float)  # Enforce strict boundary conditions by wrapping the coordinates
        self.y_coords = np.concatenate(([self.start.i], y_coords, [self.goal.i])).astype(float)
        self.length = length
        self.collision_in_i = i_coords
        self.collision_in_j = j_coords
        self.t_in = t_in
        self.t_out = t_out
        self.exec_ticks = exec_ticks
        self.total_heading_change = total_heading_change
        self.bending_energy = bending_energy
        self.U = len(i_coords)

        self._precompute()
        return self

    def _precompute(self):
        """
        Precomputes the normalized cumulative arc length to enable O(1) execution for spatial queries via the get_location() method.
        """
        # Compute the normalized cumulative arc length.
        dx = np.diff(self.x_coords)
        dy = np.diff(self.y_coords)
        # Calculate Euclidean distances between consecutive sequence nodes.
        segment_lengths = np.sqrt(dx**2 + dy**2)
        cumulative_arc_length = np.insert(np.cumsum(segment_lengths), 0, 0.0)  # [0, s1, s2, ..., sn-1], where si = distance to (x_coord[i], y_coord[i]) from start
        total_discrete_length = cumulative_arc_length[-1]

        if total_discrete_length == 0.0:
            self.normalized_arc_length = np.zeros_like(cumulative_arc_length)
        else:
            # Map the cumulative chordal length strictly to a [0, 1] parametric domain
            self.normalized_arc_length = cumulative_arc_length / total_discrete_length

    def get_location(self, p: float) -> tuple[float, float]:
        """
        Evaluates the spatial (x, y) coordinates along the primitive's trajectory at a specific normalized path parameter 'p'.
        """      
        assert 0.0 <= p <= 1.0, "Parameter 'p' must represent a normalized fraction of the traversed primitive length!"
        # Execute 1D linear interpolation mapped from parameter p.
        x_interp = float(np.interp(p, self.normalized_arc_length, self.x_coords))
        y_interp = float(np.interp(p, self.normalized_arc_length, self.y_coords))
        return x_interp, y_interp

    def set_id(self, id: int) -> None:
        self.id = id
        
    def __eq__(self, other: 'Primitive') -> bool:  # we suppose there is no more than one primitive between two specific states
        return (self.start == other.start) and (self.goal == other.goal)
    
    def __hash__(self) -> int:
        return hash((self.start.i, self.start.j, self.start.theta, self.goal.i, self.goal.j, self.goal.theta))


class ControlSet:
    """
    Manages a collection of generated primitives (the Control Set).
    """

    def __init__(self) -> None:
        # Theta configuration is now dynamically loaded from the primitive file
        self.theta: Optional[Theta] = None
        self.theta_amount: int = 0
        self.control_set: List[List[Primitive]] = []  # bundle of primitives for each heading
        self.list_all_prims: List[Optional[Primitive]] = [None] * 1000  # if it is not enough, list will be extended later
        self.any_theta_bundle: List[Primitive] = []

    def get_primitive(self, id: int) -> Primitive:
        return self.list_all_prims[id]
    
    def get_primitives_heading(self, heading: int) -> List[Primitive]:
        return self.control_set[heading % self.theta_amount]
    
    def _sort_primitives_left_to_right(self):
        """
        Sorts primitives in each heading bundle from Rightmost to Leftmost.
        Uses the 2D cross product of the initial heading vector and the displacement vector.
        """
        for heading in range(self.theta_amount):
            bundle = self.get_primitives_heading(heading)
            
            # Reference heading vector
            angle_rad = self.theta.angles[heading]
            vx = np.cos(angle_rad)
            vy = np.sin(angle_rad)
            
            def sort_key(prim):
                # Displacement vector from start to goal (for specific primitive)
                dx = prim.goal.j - prim.start.j
                dy = prim.goal.i - prim.start.i
                
                # Z-component of the 2D cross product: vx * dy - vy * dx.
                # Measures the signed distance from the displacement vector (dx,dy) to the heading ray (contain vector v).
                # Positive Z indicates the displacement vector lies to the LEFT of the heading.
                cross_prod = vx * dy - vy * dx
                
                # Sort from Right (large -Z) to Left (large +Z) by negating the cross product.
                # Secondary key: angular displacement (tie-breaking).
                return (cross_prod, self.theta.num_dist(heading, prim.goal.theta))

            # In-place sort of the primitive bundle
            bundle.sort(key=sort_key)

    def load_primitives(self, file: str) -> 'ControlSet':    
        """
        Parses metadata and primitives from the text format.
        """  
        with open(file, "r") as f:
            lines = f.readlines()
            
        idx = 0
        
        # 1. PARSE METADATA
        if idx < len(lines) and lines[idx].strip() == "=== METADATA ===":
            idx += 1
            num_headings = 0
            angles = []
            
            while idx < len(lines) and lines[idx].strip() != "---":
                line = lines[idx].strip()
                if line.startswith("NumHeadings:"):
                    num_headings = int(line.split()[1])
                    idx += 1
                elif line.startswith("Headings(rad):"):
                    idx += 1
                    for _ in range(num_headings):
                        angles.append(float(lines[idx].strip()))
                        idx += 1
                else:
                    idx += 1
                    
            idx += 1  # Skip the closing "---"
            
            # Initialize dynamic Theta
            self.theta = Theta(angles)
            self.theta_amount = self.theta.theta_amount
            self.control_set = [[] for _ in range(self.theta_amount)]
        else:
            raise ValueError("Invalid format: File must start with === METADATA ===")

        # 2. PARSE PRIMITIVES
        curr_prim_x, curr_prim_y = [], []
        i_col, j_col, t_in, t_out = [], [], [], []
        theta, goal, length = None, None, None
        exec_ticks, total_heading_change, prim_id = None, None, None
        
        while idx < len(lines):
            line = lines[idx].strip()
            idx += 1
            
            if line == "=== PRIMITIVE ===":
                curr_prim_x, curr_prim_y = [], []
                i_col, j_col, t_in, t_out = [], [], [], []
                continue

            if line.startswith("ID:"):
                prim_id = int(line.split()[1])
            elif line.startswith("StartHeadingNum:"):
                theta = int(line.split()[1])
            elif line.startswith("Goal:"):
                parts = line.split()
                goal_i = int(parts[1].split('=')[1])
                goal_j = int(parts[2].split('=')[1])
                goal_h = int(parts[3].split('=')[1])
                goal = DiscreteState(goal_i, goal_j, goal_h) 
            elif line.startswith("Length:"):
                length = float(line.split()[1])
            elif line.startswith("ExecutionTime(ticks):"):
                exec_ticks = float(line.split()[1])
            elif line.startswith("TotalHeadingChange:"):  # for metric AOL (anlge-over-length)
                total_heading_change = float(line.split()[1])
            elif line.startswith("BendingEnergy:"):
                bending_energy = float(line.split()[1])
            elif line == "Trajectory(x,y):":
                while idx < len(lines) and not lines[idx].startswith("---"):
                    pts = lines[idx].split()
                    curr_prim_x.append(float(pts[0]))
                    curr_prim_y.append(float(pts[1]))
                    idx += 1
            elif line == "CollisionFootprint(i, j, t_in, t_out):":
                while idx < len(lines) and not lines[idx].startswith("==="):
                    pts = lines[idx].split()
                    i_col.append(int(pts[0]))
                    j_col.append(int(pts[1]))
                    t_in.append(float(pts[2]))
                    t_out.append(float(pts[3]))
                    idx += 1
            elif line == "=== END ===":
                prim = Primitive(DiscreteState(0, 0, theta), goal)
                prim.set_description(np.array(curr_prim_x), np.array(curr_prim_y), length,
                                        np.array(i_col), np.array(j_col), 
                                        np.array(t_in), np.array(t_out),
                                        exec_ticks=exec_ticks, 
                                        total_heading_change=total_heading_change,
                                        bending_energy=bending_energy)
                prim.set_id(prim_id)
                
                self.control_set[theta].append(prim)
                if prim_id < len(self.list_all_prims):
                    self.list_all_prims[prim_id] = prim
                else:
                    self.list_all_prims.extend([None] * (prim_id - len(self.list_all_prims) + 1))
                    self.list_all_prims[prim_id] = prim

        # Aftre loading sort all bundles!!!
        self._sort_primitives_left_to_right()

        for theta in range(self.theta.theta_amount):
            for prim in self.get_primitives_heading(theta):
                self.any_theta_bundle.append(prim)

        return self
    
        
class Theta:
    """
    Handles dynamic heading angle discretization based on loaded metadata.
    """
    def __init__(self, angles: List[float]) -> None:
        self.theta_amount = len(angles)
        # Normalize all angles to [-pi, pi) upon initialization
        self.angles = np.array([self.correct_angle(a) for a in angles])
         
    def correct_angle(self, angle: float) -> float:
        """
        Normalizes an angle to the interval [-pi, pi).
        """
        angle %= (2 * np.pi)
        if angle > np.pi:
            angle -= 2 * np.pi
        return float(angle)
    
    def __getitem__(self, ind: int) -> float:
        """Allows array-like access: theta_obj[i] returns the i-th discrete angle."""
        return self.angles[ind % self.theta_amount]

    def num_angle(self, angle: float) -> int:
        """Maps a continuous angle to the nearest discrete heading index."""
        EPS = 1e-6
        angle = self.correct_angle(angle)
        
        for i in range(self.theta_amount): 
            if (angle - EPS <= self.angles[i] <= angle + EPS):      
                return i
                
        raise ValueError(f"Angle {angle} is not a valid discrete heading in this configuration!")
        
    def dist(self, angle1: float, angle2: float) -> int:
        """Shortest distance (in number of steps) between two continuous angles."""
        l1 = self.num_angle(angle1)
        l2 = self.num_angle(angle2)
        return self.num_dist(l1, l2)
    
    def num_dist(self, l1: int, l2: int) -> int:
        """Shortest distance (in number of steps) between two discrete heading indices."""
        l1 %= self.theta_amount
        l2 %= self.theta_amount
        return min(abs(l1 - l2), self.theta_amount - abs(l1 - l2))


class DynamicObstacle:
    def __init__(self, control_set):
        self.R = None   # integer radius of obstacle
        self.start = None
        self.path = []  # list of IDs: ID >= 0 represents a motion primitive; ID = -x represents a wait duration of x ticks
        self.control_set = control_set

    def _build_timeline(self):
        self.segments = []  # lists of (t1, t2, prim, <action>): do <action> in time interval [t1, t2)
        tick = 0
        prim = None
        for ID in self.path:
            if ID >= 0:
                prim = self.control_set.list_all_prims[ID]
                self.segments.append((tick, tick + prim.exec_ticks, prim, "execution"))  # execuate primitive 'prim'
                tick += prim.exec_ticks
            else:
                duration = -ID
                self.segments.append((tick, tick + duration, prim, "waiting"))  # wait in the end of 'prim'
                tick += duration

    def set_parametres(self, R: int, start_i: int, start_j: int, start_theta: int, path: List[int]):
        self.R = R
        self.start = DiscreteState(start_i, start_j, start_theta)
        self.path = path
        self._build_timeline()
        return self
    
    def load_from_file(self, file):
        with open(file, 'r') as f:
            assert "Dynamic obstacle description" in f.readline(), "Incorrect description file for dynamic obstacle!"
            for line in f.readlines():
                if line.startswith("R:"):
                    self.R = int(line.split(":")[1].strip())
                elif line.startswith("Start:"):
                    parts = line.split()[1:]
                    start_i, start_j, start_theta = int(parts[0].split('=')[1]), int(parts[1].split('=')[1]), int(parts[2].split('=')[1])
                    self.start = DiscreteState(start_i, start_j, start_theta)
                elif line.startswith("PrimitivesIDs:"):
                    self.path = [int(x) if (float(x) >= 0) else float(x) for x in line.split()[1:]]
        self._build_timeline()
        return self
    
    def save_to_file(self, file):
        with open(file, 'w', encoding='utf-8') as f:
            f.write(f"======= Dynamic obstacle description ======\n")
            f.write(f"R: {self.R}\n")
            f.write(f"Start: i={self.start.i} j={self.start.j} theta={self.start.theta}\n")
            f.write(f"PrimitivesIDs: {' '.join(map(str, self.path))}\n")
    
    def get_position(self, timestamp):
        """
        Retrieves the dynamic obstacle's (x, y) position at a given timestamp (float value in discrete time steps).
        """
        x, y = self.start.j, self.start.i  # current position (before next segment)
        for (t1, t2, prim, action) in self.segments:
            if t1 <= timestamp < t2:
                if action == "execution":
                    p = (timestamp - t1) / (t2 - t1)  # Interpolation factor: normalized progress along the primitive (0.0 to 1.0).
                    px, py = prim.get_location(p)     # Assumes constant velocity as primitives represent elementary motion segments.
                    return x + px, y + py
                elif action == "waiting":
                    return (x, y)                     # Dwell state: obstacle remains stationary at the primitive's goal configuration.
                else:
                    raise Exception(f"Incorrect <action> in self.segments! Found {action}")
            else:
                if action == "execution":
                    x += prim.goal.j
                    y += prim.goal.i
        return None


class ObstacleSet:
    def __init__(self):
        self.obstacles = []

    def load_dynamic_obstacles(self, file_pattern, control_set, max_items: int = 1000):
        files = sorted(glob.glob(file_pattern))
        for i, filename in enumerate(files):
            if i >= max_items:
                break
            self.obstacles.append(DynamicObstacle(control_set).load_from_file(filename))
        return self
