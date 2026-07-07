"""
Provides headless, parallelized rendering capabilities for dynamic spatial-temporal 
simulations. Optimized for server-side execution, this module bypasses GUI constraints 
by utilizing the 'Agg' Matplotlib backend within isolated multiprocessing workers, 
ensuring fast and memory-safe compilation of high-resolution videos and GIFs.
"""

import os
import tempfile
import concurrent.futures
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v3 as iio
from tqdm.auto import tqdm
import shutil

from DO_graphics import draw_task_map, draw_task_map_fast

def _render_single_frame(args):
    """
    Isolated worker function to render a single simulation frame.
    Designed to run in a separate process to avoid GIL bottlenecks and memory leaks.
    """
    # CRITICAL: Switch matplotlib backend inside the child process 
    # to avoid GUI thread issues on headless servers.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 

    (t, frame_index, temp_dir, task_map, obstacle_set, 
     robot, show_path, fast_draw, dpi, scale, style_kwargs) = args
    
    # Extract style configurations
    robot_line_width = style_kwargs.get('robot_line_width', 2.0)
    obs_line_width = style_kwargs.get('obs_line_width', 1.5)
    robot_line_marker = style_kwargs.get('robot_line_marker', '*')
    robot_line_segments_alternation = style_kwargs.get('robot_line_segments_alternation', None)
    robot_line_alpha = style_kwargs.get('robot_line_alpha', 0.9)
    
    # 0. Render the static background environment
    if fast_draw:
        ax = draw_task_map_fast(task_map, None, None, dpi=dpi, scale=scale, theta_config=None)
    else:
        ax = draw_task_map(task_map, None, None, dpi=dpi, scale=scale, theta_config=None)
        
    fig = ax.figure
    ax.set_title(f"Spatial-Temporal Simulation of Dynamic Agents | Time: {t:.2f}")
    
    # 1. Agent rendering logic (consistent with interactive visualizer)
    def add_agent_to_plot(agent, color, is_robot=False):
        if not agent: return
        pos = agent.get_position(t)
        
        # If the agent is not present at current time 't', do not render it
        if pos is None: return 
        
        style_cfg = {
            'robot': {
                'body_color': 'red',
                'edge_color': 'darkgreen',
                'body_alpha': 1.0,
                'body_lw': 2.0,
                'z_order': 30,
                'path_color': 'red',
                'path_alpha': robot_line_alpha,
                'path_lw': robot_line_width,
                'marker': robot_line_marker
            },
            'obstacle': {
                'body_color': '#A0A0A0',
                'edge_color': '#505050',
                'body_alpha': 0.8,
                'body_lw': 1.5,
                'inner_radius': 0.5,
                'inner_alpha': 0.6,
                'z_order': 20,
                'path_color': color,
                'path_alpha': 0.35,
                'path_lw': obs_line_width,
                'marker': ''
            }
        }
        
        style = style_cfg['robot'] if is_robot else style_cfg['obstacle']
        
        # 1.1 Render the main physical body
        main_body = patches.Circle(
            (pos[0], pos[1]), 
            radius=agent.R, 
            facecolor=style['body_color'], 
            edgecolor=style['edge_color'],
            lw=style['body_lw'],
            alpha=style['body_alpha'],
            zorder=style['z_order']
        )
        ax.add_patch(main_body)
        
        # 1.2 Render the inner colored identifier for obstacles
        if not is_robot:
            inner_marker = patches.Circle(
                (pos[0], pos[1]), 
                radius=style['inner_radius'], 
                facecolor=color, 
                edgecolor='none',
                alpha=style['inner_alpha'],
                zorder=style['z_order'] + 1
            )
            ax.add_patch(inner_marker)
        
        # 1.3 Pre-compute and render the full continuous trajectory
        if show_path and is_robot and (robot_line_segments_alternation is not None):
            curr_x, curr_y = agent.start.j, agent.start.i
            ind = 0
            for prim_id in agent.path:
                if prim_id >= 0:
                    prim = agent.control_set.list_all_prims[prim_id]
                    ax.plot(
                        [curr_x + x for x in prim.x_coords], [curr_y + y for y in prim.y_coords], 
                        color=robot_line_segments_alternation.split(',')[ind % 2], 
                        alpha=style['path_alpha'], 
                        lw=style['path_lw'], 
                        zorder=style['z_order'] - 50,
                        marker=style['marker'],
                        markevery=10
                    )
                    curr_x += prim.goal.j
                    curr_y += prim.goal.i
                    ind += 1
        elif show_path and hasattr(agent, 'path') and hasattr(agent, 'control_set'):
            path_x, path_y = [], []
            curr_x, curr_y = agent.start.j, agent.start.i 
            
            for prim_id in agent.path:
                if prim_id >= 0:
                    prim = agent.control_set.list_all_prims[prim_id]
                    path_x.extend([curr_x + x for x in prim.x_coords])
                    path_y.extend([curr_y + y for y in prim.y_coords])
                    curr_x += prim.goal.j
                    curr_y += prim.goal.i
            
            if path_x and path_y:
                ax.plot(
                    path_x, path_y, 
                    color=style['path_color'], 
                    alpha=style['path_alpha'], 
                    lw=style['path_lw'], 
                    zorder=style['z_order'] - 50,
                    marker=style['marker'],
                    markevery=10
                )

    # 2. Populate dynamic obstacles
    num_obstacles = len(obstacle_set.obstacles)
    colors = plt.get_cmap('hsv', max(2, num_obstacles + 1))
    for i, obs in enumerate(obstacle_set.obstacles):
        add_agent_to_plot(obs, color=colors(i))
        
    # 3. Populate the primary ego-robot
    if robot:
        add_agent_to_plot(robot, color='lime', is_robot=True)
        
    # Save frame to disk
    filename = os.path.join(temp_dir, f"frame_{frame_index:05d}.png")
    fig.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    
    # Strictly close figure to free memory within the worker
    plt.close(fig) 
    
    return filename


def render_simulation_parallel(output_file, task_map, obstacle_set, robot=None,
                               max_time=100.0, dt=1.0, fps=20, 
                               show_path=False, fast_draw=True, dpi=150, scale=1.0, workers=None,
                               robot_line_width=2.0, obs_line_width=1.5, robot_line_marker='*',
                               robot_line_segments_alternation=None, robot_line_alpha=0.9):
    """
    Renders a dynamic simulation into a video/GIF file using multiprocessing.
    Ideal for server-side execution and high-resolution outputs.

    Args:
        output_file (str): Path to save the media (e.g., 'output.mp4' or 'output.gif').
        task_map, obstacle_set: Core environment and agent definitions.
        robot (DynamicAgent, optional): The ego-agent.
        max_time (float): Total duration of the simulation to render.
        dt (float): Time step increment per frame.
        fps (int): Frames per second for the output media.
        show_path (bool): If True, pre-renders the full trajectory.
        fast_draw (bool): Toggles optimized background rendering.
        dpi (int), scale (float): Figure resolution and scaling.
        workers (int, optional): Number of parallel processes. Defaults to max CPU cores - 1.
        
        # Style parameters matching the interactive visualizer
        robot_line_width (float): Thickness of the robot's trajectory line.
        obs_line_width (float): Thickness of obstacle trajectory lines.
        robot_line_marker (str): Matplotlib marker style for the robot's path.
        robot_line_segments_alternation (str, optional): Comma-separated colors for alternating segments.
        robot_line_alpha (float): Transparency of the robot's path.
    """
    
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)
        
    time_steps = [t * dt for t in range(int(max_time / dt) + 1)]
    total_frames = len(time_steps)
    
    print(f"🚀 Starting parallel rendering...")
    print(f"📊 Total frames: {total_frames} | CPU Workers: {workers}")

    temp_dir = os.path.join(os.getcwd(), "temp_frames_for_video")
    os.makedirs(temp_dir, exist_ok=True)

    # Bundle visual styles to pass them cleanly to the pool
    style_kwargs = {
        'robot_line_width': robot_line_width,
        'obs_line_width': obs_line_width,
        'robot_line_marker': robot_line_marker,
        'robot_line_segments_alternation': robot_line_segments_alternation,
        'robot_line_alpha': robot_line_alpha
    }

    try:
        tasks = []
        for i, t in enumerate(time_steps):
            tasks.append((t, i, temp_dir, task_map, obstacle_set, 
                          robot, show_path, fast_draw, dpi, scale, style_kwargs))
            
        generated_files = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            generated_files = list(tqdm(executor.map(_render_single_frame, tasks), 
                                        total=total_frames, desc="Rendering frames"))
            
        generated_files.sort()
        
        print("🎬 Assembling media file (Streaming Mode + HEVC)...")
    
        import imageio 
        
        if output_file.endswith('.mp4'):
            # Using FFMPEG backend directly. Pass compression flags via output_params
            with imageio.get_writer(
                output_file, 
                format='FFMPEG', 
                mode='I', 
                fps=fps, 
                codec='libx265', 
                pixelformat='yuv420p', 
                macro_block_size=None,
                # FFMPEG compression and video layout configuration:
                # - crf 23: Constant Rate Factor (0-51). 23 is the default sweet spot for H.265 visual quality vs file size.
                # - preset medium: Balanced encoding speed vs compression efficiency trade-off.
                # - vf scale=...: Video filter forcing canvas dimensions to be even integers. 
                #   Strictly required by the yuv420p pixel format to prevent 'chroma subsampling' errors with arbitrary map sizes.
                output_params=[
                    '-crf', '23', 
                    '-preset', 'medium', 
                    '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2'
                ]
            ) as writer:
                for frame_path in tqdm(generated_files, desc="Saving MP4"):
                    frame_data = iio.imread(frame_path) 
                    if frame_data.shape[-1] == 4:  # Strip alpha channel
                        frame_data = frame_data[..., :3]
                    writer.append_data(frame_data)
        else:
            # GIF assembly
            with imageio.get_writer(output_file, mode='I', fps=fps, loop=0) as writer:
                for frame_path in tqdm(generated_files, desc="Saving GIF"):
                    frame_data = iio.imread(frame_path)
                    if frame_data.shape[-1] == 4:
                        frame_data = frame_data[..., :3]
                    writer.append_data(frame_data)
        print(f"✅ Success! Render saved to: {output_file}")

    finally:
        # Guaranteed cleanup block
        if os.path.exists(temp_dir):
            print("🧹 Cleaning up temporary frames...")
            shutil.rmtree(temp_dir, ignore_errors=True)
