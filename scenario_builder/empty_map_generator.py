import random
import math

"""
MODULE: empty_map_generator.py
DESCRIPTION: 
Generates benchmark environments and queries strictly following the MovingAI format.
Produces an obstacle-free grid map (.map) and a corresponding scenario file (.scen) 
with randomly sampled valid start-goal pairs and their ground-truth Octile distances.
Useful for baseline testing of grid-based pathfinding algorithms (A*, SIPP, etc.).
"""

# --- CONFIGURATION ---
W = 64            # Map width
H = 64            # Map height
NUM_TESTS = 250   # Number of test queries in the scenario
# ---------------------

def generate_moving_ai_files(w, h, num_tests):
    map_name = f"empty_{w}_{h}.map"
    scen_name = f"{map_name}.scen"

    # 1. Map file (.map) generation
    # Format specification: type, height, width, map header + character grid ('.' for traversable)
    map_header = [
        "type octile",
        f"height {h}",
        f"width {w}",
        "map"
    ]
    
    with open(map_name, "w") as f:
        f.write("\n".join(map_header) + "\n")
        for _ in range(h):
            f.write("." * w + "\n")
    
    print(f"Map file successfully generated: {map_name}")

    # 2. Scenario file (.scen) generation
    # Line format: bucket map_name map_width map_height start_x start_y goal_x goal_y distance
    scen_lines = ["version 1"]
    used_pairs = set()

    while len(used_pairs) < num_tests:
        # Sample random coordinates uniformly
        sx, sy = random.randint(0, w - 1), random.randint(0, h - 1)
        gx, gy = random.randint(0, w - 1), random.randint(0, h - 1)

        # Ensure start and goal are distinct
        if (sx, sy) == (gx, gy):
            continue
            
        # Ensure the query pair is unique regardless of direction
        pair = tuple(sorted([(sx, sy), (gx, gy)]))
        if pair not in used_pairs:
            used_pairs.add(pair)
            
            # Ground-truth Octile Distance calculation for an obstacle-free grid
            dx = abs(sx - gx)
            dy = abs(sy - gy)
            # Formula: (sqrt(2)-1) * min(dx, dy) + max(dx, dy)
            dist = (math.sqrt(2) - 1) * min(dx, dy) + max(dx, dy)
            
            # The 'bucket' (first integer) is conventionally used to group queries by difficulty/length; 
            # assigned to 1 here for simplicity.
            scen_lines.append(f"1\t{map_name}\t{w}\t{h}\t{sx}\t{sy}\t{gx}\t{gy}\t{dist:.8f}")

    with open(scen_name, "w") as f:
        f.write("\n".join(scen_lines) + "\n")
    
    print(f"Scenario file successfully generated: {scen_name} ({len(used_pairs)} queries)")

if __name__ == "__main__":
    generate_moving_ai_files(W, H, NUM_TESTS)
