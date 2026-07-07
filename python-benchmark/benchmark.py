import sys
import os
import time
import random
import csv
import math
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

# Algorithm and data structure imports
sys.path.append(os.path.abspath('../common'))
try:
    from DO_structs import *
    from DO_searching import *
except ImportError:
    print("⚠️ Import Error: Could not find modules in ../common")

sys.path.append(os.path.abspath('../planners_library'))
try:
    from sipp import *
    from trajectory_metrics import *
except ImportError:
    print("⚠️ Import Error: Could not find modules in ../planners_library")


# =====================================================================
# 🛠 BENCHMARK & ENVIRONMENT SETTINGS
# =====================================================================
MAP_NAME = "Denver_1_256"
MAP_FILE = f"../maps/{MAP_NAME}.map"
SCEN_FILE = f"../maps/{MAP_NAME}.map.scen"
DYN_OBSTACLES_PATTERN = f"../dynamic_obstacles/{MAP_NAME}_random/dynamic_obstacle_*.txt"
OBSTACLES_CONTROL_SET = '../data/control_set.txt'

BATCHES = [0, 50, 150, 320]            # Batch sizes for the number of dynamic obstacles
THETA_SAMPLES_PER_SCEN = 1             # Number of random start/goal heading pairs to generate per MovingAI scenario 
MAX_SCENARIOS = 1000                   # Limit the number of scenarios to process (None to process all)
NUM_WORKERS = 60                       # Number of parallel worker processes
ROBOT_RADIUS = 1                       # Ego-robot physical radius

# =====================================================================
# 🧠 ALGORITHM CONFIGURATION (Easily extensible!)
# =====================================================================
# id:           Name of the algorithm (used in the output CSV).
# file:         Path to the primitives configuration file.
# is_angular:   Whether the algorithm searches for a piecewise-linear path (True for 2^k grids; affects metric calculations).
# needs_theta:  Whether the real starting/goal heading is required (True for smooth paths, False for 2^k grids where heading is implicitly 0).

ALGORITHMS_CONFIG = [
    {'id': 'SIPP_Standard', 'file': '../data/control_set.txt',     'is_angular': False, 'needs_theta': True},
    {'id': 'SIPP_BigCS',    'file': '../data/extended_set.txt',    'is_angular': False, 'needs_theta': True},
    {'id': 'SIPP_2^2',      'file': '../data/primitives_2k_2.txt', 'is_angular': True,  'needs_theta': False},
    {'id': 'SIPP_2^3',      'file': '../data/primitives_2k_3.txt', 'is_angular': True,  'needs_theta': False},
    {'id': 'SIPP_2^4',      'file': '../data/primitives_2k_4.txt', 'is_angular': True,  'needs_theta': False},
    {'id': 'SIPP_2^5',      'file': '../data/primitives_2k_5.txt', 'is_angular': True,  'needs_theta': False},
]

# ID of the baseline algorithm. If this algorithm fails (success=0), 
# we skip executing the remaining algorithms for that specific task to save time.
FAST_FAIL_ALG_ID = 'SIPP_Standard'


# =====================================================================
# ⚙️ CHILD PROCESS (WORKER) LOGIC
# =====================================================================
_worker_cs_dict = {}
_worker_environment = None

def init_worker(map_file, dyn_pattern, num_obstacles, r_radius):
    """
    Initializer for the multiprocessing pool.
    Loads the map, environment, and ALL required Control Sets into 
    the worker process's memory exactly once upon startup.
    """
    global _worker_cs_dict, _worker_environment
    
    # 1. Load all Control Sets defined in ALGORITHMS_CONFIG
    for cfg in ALGORITHMS_CONFIG:
        alg_id = cfg['id']
        cs = ControlSet()
        cs.load_primitives(cfg['file'])
        _worker_cs_dict[alg_id] = cs
    
    # 2. Initialize the static map and dynamic environment
    task_map = Map()
    task_map.read_from_movingai_file(map_file)
    
    # Control set defining the kinematic constraints of the dynamic obstacles
    cs_for_env = ControlSet().load_primitives(OBSTACLES_CONTROL_SET)
    
    _worker_environment = DynamicEnvironment(task_map, r_radius)
    _worker_environment.load_obstacles_from_dir(dyn_pattern, cs_for_env, max_items=num_obstacles)
    _worker_environment.compile_safe_intervals()

def format_raw_path(path):
    """Formats the raw path list for CSV output: waits (floats) to 3 decimals, IDs as integers."""
    if not path: return ""
    return " ".join([f"{float(act):.3f}" if act < 0 else f"{int(act)}" for act in path])

def run_task(task_id, start_i, start_j, start_theta, goal_i, goal_j, goal_theta):
    """Core pathfinding execution function for a specific task (runs within the worker pool)."""
    global _worker_cs_dict, _worker_environment
    
    results = []
    fast_fail_triggered = False
    
    for cfg in ALGORITHMS_CONFIG:
        alg_id = cfg['id']
        
        # If the baseline algorithm failed, immediately populate the remaining algorithms with zeroed results
        if fast_fail_triggered:
            results.append({
                'task_id': task_id, 'alg': alg_id,
                'start': f"{start_i},{start_j},0", 'goal': f"{goal_i},{goal_j},0",
                'success': 0, 'cost': -1, 'steps': 0, 'time_sec': 0.0,
                'curvature': 0.0, 'heading_change': 0.0, 'bending_energy': 0.0,
                'path': ""
            })
            continue

        # Prepare states based on the algorithm's specific kinematic requirements
        s_theta = start_theta if cfg['needs_theta'] else 0
        g_theta = goal_theta if cfg['needs_theta'] else 0
        
        start_state = DiscreteState(start_i, start_j, s_theta)
        goal_state = DiscreteState(goal_i, goal_j, g_theta)
        
        # Execute Search
        try:
            cs = _worker_cs_dict[alg_id]
            
            # Since we are comparing algorithms with smooth control sets against 2^k connectivity algorithms 
            # that operate purely in 2D space (ignoring headings), we set position_only=True. 
            # This ensures all algorithms ignore initial and final orientations for a fair comparison.
            search_space = SIPPSearchSpace(start_state, 0.0, goal_state, cs, _worker_environment, R=0, A=0, position_only=True)
            
            t_start = time.perf_counter()
            success, path, steps, cost, _ = astar(search_space)
            t_end = time.perf_counter()
            time_sec = t_end - t_start

        except Exception as e:
            print(f"⚠️ Error in {alg_id} (task {task_id}): {e}")
            success, path, steps, cost, time_sec = False, [], 0, -1, 0.0

        # Calculate geometric and kinematic metrics (if a valid path was found)
        curv, head_change, b_energy = 0.0, 0.0, 0.0
        if success:
            prim_ids = get_prim_ids(path)
            curv = compute_curvature(prim_ids, cs)
            head_change = compute_heading_change(prim_ids, cs, cfg['is_angular'])
            b_energy = compute_bending_energy(prim_ids, cs, cfg['is_angular'])

        # Store the result payload
        results.append({
            'task_id': task_id,
            'alg': alg_id,
            'start': f"{start_i},{start_j},{s_theta}",
            'goal': f"{goal_i},{goal_j},{g_theta}",
            'success': int(success),
            'cost': cost if success else -1,
            'steps': steps,
            'time_sec': round(time_sec, 5),
            'curvature': round(curv, 5),
            'heading_change': round(head_change, 5),
            'bending_energy': round(b_energy, 5),
            'path': format_raw_path(path) if success else ""
        })
        
        # Trigger Fast-Fail if the baseline algorithm fails to find a solution
        if alg_id == FAST_FAIL_ALG_ID and not success:
            fast_fail_triggered = True

    return results


# =====================================================================
# 🚀 MAIN LOOP (ORCHESTRATION)
# =====================================================================
def parse_movingai_scen(scen_path):
    """Parses standard MovingAI scenario files to extract start and goal coordinates."""
    scenarios = []
    with open(scen_path, 'r') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split('\t')
            if len(parts) >= 8:
                # Extracts: start_i, start_j, goal_i, goal_j
                scenarios.append((int(parts[5]), int(parts[4]), int(parts[7]), int(parts[6]))) 
    return scenarios

def main():
    random.seed(239)
    print(f"🚀 Starting benchmark for map: {MAP_NAME}")
    
    # 1. Load Scenarios
    base_scenarios = parse_movingai_scen(SCEN_FILE)
    if MAX_SCENARIOS: base_scenarios = base_scenarios[:MAX_SCENARIOS]
    
    tasks = []
    task_id_counter = 1
    for start_i, start_j, goal_i, goal_j in base_scenarios:
        for _ in range(THETA_SAMPLES_PER_SCEN):
            tasks.append((task_id_counter, start_i, start_j, random.randint(0, 15), goal_i, goal_j, random.randint(0, 15)))
            task_id_counter += 1
            
    print(f"🎯 Generated {len(tasks)} pathfinding tasks.")
    print(f"⚙️ Active algorithms: {[cfg['id'] for cfg in ALGORITHMS_CONFIG]}")
    
    # 2. Execute by Batches (Iterating over different dynamic obstacle densities)
    for num_obstacles in BATCHES:
        print(f"\n=============================================")
        print(f"🔥 Launching batch: {num_obstacles} dynamic obstacles")
        print(f"=============================================")
        
        results_file = f"benchmark_{MAP_NAME}_obs_{num_obstacles}.csv"
        all_results = []
        completed = 0
        
        with ProcessPoolExecutor(
            max_workers=NUM_WORKERS, 
            initializer=init_worker, 
            initargs=(MAP_FILE, DYN_OBSTACLES_PATTERN, num_obstacles, ROBOT_RADIUS)
        ) as executor:
            
            future_to_task = {executor.submit(run_task, *task): task[0] for task in tasks}
            
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    all_results.extend(future.result())
                except Exception as exc:
                    print(f"❌ Worker exception occurred: {exc}")
                    
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"⏳ Progress: {completed}/{len(tasks)} tasks completed...")

        # 3. Save to CSV (Single clean write block per batch)
        all_results.sort(key=lambda x: (x['task_id'], x['alg']))
        print(f"💾 Saving results to {results_file}...")
        
        if all_results:
            fieldnames = ['task_id', 'alg', 'start', 'goal', 'success', 'cost', 'steps', 
                          'time_sec', 'curvature', 'heading_change', 'bending_energy', 'path']
            
            with open(results_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                writer.writeheader()
                for row in all_results:
                    writer.writerow(row)
                    
        print(f"✅ Batch {num_obstacles} finished successfully!")

if __name__ == '__main__':
    main()
