"""
Multi-Map Sequential Benchmarking Orchestrator
==============================================
This script automates the sequential execution of pathfinding benchmarks across multiple 
maps with varying dynamic obstacle setups. It dynamically modifies global configuration 
variables in the main benchmark script (benchmark.py), executes it, and organizes the
resulting CSV files into map-specific subdirectories.

Execution Architecture:
-----------------------
- Orchestrator Level: Runs sequentially. It uses `subprocess.run` to execute one map configuration
  at a time, blocking until the entire sequence for that map finishes.
- Benchmark Level: Runs in parallel. The underlying benchmark script utilizes a `ProcessPoolExecutor`
  to spin up multiple worker processes across available CPU cores, maximizing performance.

How to Run (Recommended Docker/Background Command):
--------------------------------------------------
To run this script in the background without output buffering (for example, in a Docker container), use:
    PYTHONUNBUFFERED=1 nohup python3 runner.py > benchmark_log.txt 2>&1 &

Process Lifecycle & Termination Notes:
-------------------------------------
- Using `docker stop`: Container shutdown via `docker stop` sends a SIGTERM signal. Results are 
  committed to CSV files only at the end of each finished obstacle batch. Stopping mid-batch 
  will lose unwritten progress for that active batch.
- Using `docker start`: The script WILL NOT automatically resume upon container restart unless 
  explicitly declared in the Docker ENTRYPOINT/CMD. It will restart from the first map.
- Manual Termination: To kill the background execution cleanly without stopping the entire 
  Docker container, enter the container and execute:
    pkill -f runner.py && pkill -f temp_runner.py
"""

import os
import re
import subprocess
import glob
import shutil

# Specify the name of your main benchmark script here
ORIGINAL_SCRIPT = "benchmark.py" 
TEMP_SCRIPT = "temp_runner.py"

# Configurations for the 3 sequential runs
RUNS = [
    {
        "MAP_NAME": "Denver_1_256",
        "DYN_OBSTACLES_PATTERN": "../dynamic_obstacles/{MAP_NAME}_random/dynamic_obstacle_*.txt",
        "BATCHES": "[0, 50, 150, 320]"
    },
    {
        "MAP_NAME": "empty_64_64",
        "DYN_OBSTACLES_PATTERN": "../dynamic_obstacles/{MAP_NAME}_circle/dynamic_obstacle_*.txt",
        "BATCHES": "[0, 5, 10, 20, 50, 80]"
    },
    {
        "MAP_NAME": "arena",
        "DYN_OBSTACLES_PATTERN": "../dynamic_obstacles/{MAP_NAME}_random/dynamic_obstacle_*.txt",
        "BATCHES": "[0, 5, 10, 20, 35, 50]"
    }
]

def main():
    if not os.path.exists(ORIGINAL_SCRIPT):
        print(f"❌ Error: File '{ORIGINAL_SCRIPT}' not found!")
        return

    # Read the original source code of the benchmark script
    with open(ORIGINAL_SCRIPT, 'r', encoding='utf-8') as f:
        original_code = f.read()

    for i, run_cfg in enumerate(RUNS, 1):
        map_name = run_cfg["MAP_NAME"]
        print(f"\n{'='*50}")
        print(f"🚀 Run {i}/{len(RUNS)} — Map: {map_name}")
        print(f"{'='*50}")
        
        # 1. Replace global variables using regular expressions
        # Looks for the pattern 'MAP_NAME = "anything"' (handles single or double quotes)
        modified_code = re.sub(r'MAP_NAME\s*=\s*["\'].*?["\']', f'MAP_NAME = "{map_name}"', original_code)
        
        modified_code = re.sub(r'DYN_OBSTACLES_PATTERN\s*=\s*f["\'].*?["\']', 
                               f'DYN_OBSTACLES_PATTERN = f"{run_cfg["DYN_OBSTACLES_PATTERN"]}"', 
                               modified_code)
                               
        modified_code = re.sub(r'BATCHES\s*=\s*\[.*?\]', f'BATCHES = {run_cfg["BATCHES"]}', modified_code)
        
        # 2. Save the modified code to a temporary executable file
        with open(TEMP_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(modified_code)
            
        # 3. Create a target directory for the results
        os.makedirs(map_name, exist_ok=True)
        
        # 4. Execute the temporary script and wait for its completion
        print(f"▶️ Executing pathfinding benchmark...")
        try:
            # check=True will raise an exception if the benchmark script fails critically
            subprocess.run(["python3", TEMP_SCRIPT], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Run for {map_name} failed. Moving to the next one.")
            continue
            
        # 5. Collect all generated CSV files and move them to the target directory
        csv_files = glob.glob(f"benchmark_{map_name}_obs_*.csv")
        if not csv_files:
            print("⚠️ No result files found. Algorithms might have failed to find solutions.")
            
        for csv_file in csv_files:
            target_path = os.path.join(map_name, csv_file)
            shutil.move(csv_file, target_path)
            
        print(f"✅ Results successfully saved to directory: ./{map_name}/")

    # 6. Clean up the temporary file
    if os.path.exists(TEMP_SCRIPT):
        os.remove(TEMP_SCRIPT)
        
    print("\n🎉 All benchmarks completed successfully!")

if __name__ == '__main__':
    main()
