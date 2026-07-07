"""
This module contains core functions for visualizing path planning results:
rendering grids, trajectories, and the discrete workspace (map) with dynamic obstacles.
"""

import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap
from typing import Optional
import numpy as np
from typing import Optional, List, Union, Tuple
from IPython.display import clear_output

from DO_structs import ShortTrajectory, Primitive, DiscreteState, Theta
from DO_searching import Map


def draw_grid(ax: matplotlib.axes.Axes, 
              xs: int = -8, ys: int = -8, 
              xf: int = 8, yf: int = 8, 
              tick_step: int = 1) -> None:
    """ 
    Draws a grid and sets coordinate axis ticks.
    
    Args:
        ax: The matplotlib axes to draw on.
        xs, ys, xf, yf: Coordinates defining the bounding box of the grid.
        tick_step: Step size for axis ticks.
        
    Note: The grid is drawn such that integer coordinates represent cell centers 
    (grid lines are at half-integer coordinates).
    """
    
    # Fix for the warning: explicitly tell matplotlib to adjust the plot box, not the data limits
    ax.set_aspect('equal', adjustable='box') 
    
    ax.set(xlim=(xs, xf), xticks=np.arange(xs, xf, tick_step),
           ylim=(ys, yf), yticks=np.arange(ys, yf, tick_step))
    
    # Horizontal grid lines
    for y in np.arange(ys + 0.5, yf, 1):
        ax.axhline(y, color='grey', alpha=0.45)
        
    # Vertical grid lines
    for x in np.arange(xs - 0.5, xf, 1):
        ax.axvline(x, color='grey', alpha=0.45)


def plot_arrow(x: float, y: float, theta: float, 
               length: float = 1.0, width: float = 0.5, 
               fc: str = "r", ec: str = "k", 
               ax: Optional[matplotlib.axes.Axes] = None) -> None:
    """
    Draws an arrow indicating direction/heading.
    
    Args:
        x, y, theta: Position and heading angle (in radians).
        length, width: Dimensions of the arrow head.
        fc, ec: Face color and edge color.
        ax: Axes to draw on (uses plt if None).
    """
    
    board = plt if (ax is None) else ax
    board.arrow(x, y, length * np.cos(theta), length * np.sin(theta),
                fc=fc, ec=ec, head_width=width, head_length=width)


def show_trajectory(traj: ShortTrajectory, 
                    col: str = 'r',
                    linestyle: str = '-',
                    arrow: bool = True, 
                    ax: Optional[matplotlib.axes.Axes] = None) -> None:
    """
    Visualizes a trajectory.
    
    Args:
        traj: The ShortTrajectory object.
        col: Color string.
        linestyle: Style of the line.
        arrow: Whether to draw an arrow at the final state.
        ax: Axes to draw on.
    """
    
    board = plt if (ax is None) else ax
    
    xc = traj.sample_x()
    yc = traj.sample_y()
    board.plot(xc, yc, color=col, linestyle=linestyle)
    
    if arrow:
        final = traj.goal
        plot_arrow(final.x, final.y, final.theta, fc=col, ax=ax)


def draw_task_map(task_map: Map, 
                  start: Union[DiscreteState, None], 
                  goal: Union[DiscreteState, None], 
                  dpi: int = 200, 
                  scale: float = 1.0, 
                  R: float = 0, 
                  theta_config: Optional[Theta] = None,
                  ax_existing=None) -> matplotlib.axes.Axes:
    """
    Original function to draw the map, obstacles, start, and goal.
    
    Method:
        Iterates through every cell in the map. If a cell is an obstacle, 
        it adds a matplotlib.patches.Rectangle to the plot.
    """
    
    w, h = 10 * (task_map.width / max(task_map.width, task_map.height)), 10 * (task_map.height / max(task_map.width, task_map.height))
        
    if ax_existing is None:
        fig = plt.figure(figsize=(scale * w, scale * h), dpi=dpi)
        ax = fig.add_subplot(111)
    else:
        ax = ax_existing

    # ===== Draw Map Grid =====
    xs, ys = 0, 0
    xf, yf = task_map.width, task_map.height
    _d = int((xf - xs) / 30) + 1  
    
    # Invert Y axis so 'i' increases downwards
    ax.set(xlim=(xs-0.5, xf-0.5), xticks=np.arange(0, xf, _d),        
           ylim=(ys-0.5, yf-0.5)[::-1], yticks=np.arange(0, yf, _d))  
    # ax.axis('equal')     
    ax.set_aspect('equal')  # adjust box of coordinates

    for y in np.arange(ys-0.5, yf, 1):
        ax.axhline(y, color='grey', alpha=0.45)
    for x in np.arange(xs-0.5, xf, 1):
        ax.axvline(x, color='grey', alpha=0.45)

    # ===== Draw Obstacles (Vector Method) =====
    for i in range(task_map.height):
        for j in range(task_map.width):
            if not task_map.traversable(i, j):
                # i corresponds to y (inverted), j corresponds to x
                ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, color='g', alpha=0.25))

    # ===== Draw Start and Goal =====
    if goal:  # Goal
        ax.add_patch(patches.Circle((goal.j, goal.i), R, color='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((goal.j-0.5, goal.i-0.5), 1, 1, color='r', alpha=1))
        if theta_config:
            plot_arrow(goal.j, goal.i, theta_config.angles[goal.theta], fc='red', ax=ax)
    if start:  # Start
        ax.add_patch(patches.Rectangle((start.j-0.5, start.i-0.5), 1, 1, color='g', alpha=1))
        if theta_config:
            plot_arrow(start.j, start.i, theta_config.angles[start.theta], fc='green', ax=ax)
    return ax


def draw_task_map_fast(task_map: Map, 
                       start: Union[DiscreteState, None], 
                       goal: Union[DiscreteState, None], 
                       dpi: int = 200, 
                       scale: float = 1.0, 
                       R: float = 0, 
                       theta_config: Optional[Theta] = None,
                       ax_existing: Optional[matplotlib.axes.Axes] = None) -> matplotlib.axes.Axes:
    """
    Optimized function to draw the map using Raster Graphics (imshow).
    Matches the visual style of the vector-based draw_task_map (green obstacles).

    Args:
        task_map: The Map object containing the binary grid data.
        start: Start state.
        goal: Goal state.
        dpi: Dots per inch.
        scale: Scaling factor for figure size.
        R: Goal radius.
        theta_config: Theta configuration.
        ax_existing: Optional existing axes to draw on.
    """
    
    # 1. Figure Setup
    aspect = task_map.width / task_map.height
    base_h = 8
    base_w = base_h * aspect
    
    if ax_existing is None:
        fig = plt.figure(figsize=(scale * base_w, scale * base_h), dpi=dpi)
        ax = fig.add_subplot(111)
    else:
        ax = ax_existing

    # 2. Draw Map (Optimized)
    # Mask free cells (0) so they are completely transparent
    obstacle_mask = np.ma.masked_where(task_map.cells == 0, task_map.cells)
    
    # Use a solid green colormap to match 'color="g"' from the original function.
    # We set alpha=0.4 to match the visual weight of the original's alpha=0.25 
    # (raster pixels often need slightly higher alpha to look as dense as vector patches).
    cmap_green = ListedColormap(['green'])
    
    ax.imshow(obstacle_mask, cmap=cmap_green, origin='upper', alpha=0.4, 
              extent=[-0.5, task_map.width - 0.5, task_map.height - 0.5, -0.5])

    # 3. Configure Axes (Smart Ticks)
    # MaxNLocator automatically skips numbers to prevent overlap
    locator_x = ticker.MaxNLocator(nbins=20, integer=True)
    locator_y = ticker.MaxNLocator(nbins=20, integer=True)
    
    ax.xaxis.set_major_locator(locator_x)
    ax.yaxis.set_major_locator(locator_y)
    
    # Invert Y axis limits to match the coordinate system (0 at top)
    ax.set_xlim(-0.5, task_map.width - 0.5)
    ax.set_ylim(task_map.height - 0.5, -0.5) 

    # Fast grid drawing (instead of iterating lines)
    ax.grid(which='major', color='grey', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_aspect('equal')
    
    # 4. Draw Start and Goal (Vector elements are fine here as there are only 2)
    if goal:  # Goal
        ax.add_patch(patches.Circle((goal.j, goal.i), R, color='red', alpha=0.2))
        ax.add_patch(patches.Rectangle((goal.j-0.5, goal.i-0.5), 1, 1, color='r', alpha=1))
        if theta_config:
            plot_arrow(goal.j, goal.i, theta_config.angles[goal.theta], fc='red', ax=ax)
    if start:  # Start
        ax.add_patch(patches.Rectangle((start.j-0.5, start.i-0.5), 1, 1, color='g', alpha=1))
        if theta_config:
            plot_arrow(start.j, start.i, theta_config.angles[start.theta], fc='green', ax=ax)

    return ax


from PIL import Image
from IPython.display import display

def create_dynamic_visualizer(task_map, obstacle_set, robot=None,
                              show_path=False, fast_draw=False, dpi=200, scale=1.0, make_gif=False,
                              robot_line_width=2.0, obs_line_width=1.5, robot_line_marker='*',
                              robot_line_segments_alternation=None, robot_line_alpha=0.9,
                              jupyter_display=True):
    """
    Initializes a spatial-temporal visualization context for dynamic agents 
    (robots and moving obstacles) overlaying a static environmental map.

    Args:
        task_map: The static grid/environment map.
        obstacle_set: Collection of dynamic obstacles with predictable trajectories.
        robot (DynamicAgent, optional): The primary ego-agent being planned for.
        show_path (bool): If True, renders the full continuous trajectory for all agents.
        fast_draw (bool): Toggles optimized background rendering.
        dpi (int): Resolution of the Matplotlib figure.
        scale (float): Scaling factor for the visualization.
        make_gif (bool): If True, captures frames and returns a save_gif callback.
        robot_line_width (float): Thickness of the robot's trajectory line.
        obs_line_width (float): Thickness of the obstacles' trajectory lines.
        robot_line_marker (str): Matplotlib marker style for the robot's path.
        robot_line_segments_alternation (str, optional): Comma-separated colors (e.g., 'teal,orange') for alternating path segments.
        robot_line_alpha (float): Transparency of the robot's trajectory.
        jupyter_display (bool): If True, aggressively updates the Jupyter Notebook cell display in real-time. 
                                Set to False for silent background rendering (e.g., generating static snapshots).

    Returns:
        update_fn (callable): Function to update agent positions at a given timestamp 't'.
        save_gif (callable, optional): Function to compile and save the captured frames (if make_gif=True).
    """
    
    # 0. Render the static background environment
    if fast_draw:
        ax = draw_task_map_fast(task_map, None, None, dpi=dpi, scale=scale, theta_config=None)
    else:
        ax = draw_task_map(task_map, None, None, dpi=dpi, scale=scale, theta_config=None)
        
    fig = ax.figure
    ax.set_title("Spatial-Temporal Simulation of Dynamic Agents")
    
    # Data structures to maintain graphical handles: (circle_patch, path_line_patch, agent_reference)
    agent_patches = [] 
    frames = []  # Initialize frame buffer for GIF generation
    
    # 1. Helper function to inject an agent into the Matplotlib axes
    def add_agent_to_plot(agent, color, is_robot=False):
        if not agent: return
        
        # --- Visualization Configuration ---
        # Centralized styling parameters for rapid tweaking and academic consistency
        style_cfg = {
            'robot': {
                'body_color': 'red',
                'edge_color': 'darkgreen',
                'body_alpha': 1.0,
                'body_lw': 2.0,
                'z_order': 30,                   # Robot is strictly on top
                'path_color': 'red',
                'path_alpha': robot_line_alpha,  # Configurable alpha for the ego-agent
                'path_lw': robot_line_width,
                'marker': robot_line_marker
            },
            'obstacle': {
                'body_color': '#A0A0A0',   # Solid, darker gray for physical presence
                'edge_color': '#505050',   # Dark gray border
                'body_alpha': 0.8,
                'body_lw': 1.5,
                'inner_radius': 0.5,       # Radius of the identity marker
                'inner_alpha': 0.6,        # Transparency of the identity marker
                'z_order': 20,             # Obstacles above paths, below robot
                'path_color': color,       # Unique color tied to the obstacle
                'path_alpha': 0.35,        # Dimmer, thinner paths to reduce visual clutter
                'path_lw': obs_line_width,
                'marker': ''               # No marker for obstacles
            }
        }
        
        style = style_cfg['robot'] if is_robot else style_cfg['obstacle']
        graphics = []  # Store all visual patches associated with this agent
        
        # 1.1 Render the main physical body of the agent
        main_body = patches.Circle(
            (agent.start.j, agent.start.i), 
            radius=agent.R, 
            facecolor=style['body_color'], 
            edgecolor=style['edge_color'],
            lw=style['body_lw'],
            alpha=style['body_alpha'],
            zorder=style['z_order']
        )
        ax.add_patch(main_body)
        graphics.append(main_body)
        
        # 1.2 Render the inner colored identifier for obstacles
        if not is_robot:
            inner_marker = patches.Circle(
                (agent.start.j, agent.start.i), 
                radius=style['inner_radius'], 
                facecolor=color, 
                edgecolor='none',  # No border for the inner marker
                alpha=style['inner_alpha'],
                zorder=style['z_order'] + 1  # Strictly above the main body
            )
            ax.add_patch(inner_marker)
            graphics.append(inner_marker)
        
         # 1.3 Path Rendering Logic
        if show_path and is_robot and (robot_line_segments_alternation is not None):
            # Render alternating color segments for the robot's primitives
            curr_x, curr_y = agent.start.j, agent.start.i
            ind = 0
            for prim_id in agent.path:
                if prim_id >= 0:
                    prim = agent.control_set.list_all_prims[prim_id]
                    path_line, = ax.plot(
                        [curr_x + x for x in prim.x_coords], [curr_y + y for y in prim.y_coords], 
                        color=robot_line_segments_alternation.split(',')[ind % 2], 
                        alpha=style['path_alpha'], 
                        lw=style['path_lw'], 
                        zorder=style['z_order'] - 50,  # Paths remain strictly below all agents
                        marker=style['marker'],
                        markevery=10
                    )
                    graphics.append(path_line)
                    curr_x += prim.goal.j
                    curr_y += prim.goal.i
                    ind += 1

        elif show_path and hasattr(agent, 'path') and hasattr(agent, 'control_set'):
            # Pre-compute and render the full continuous trajectory from primitive coordinates
            path_x, path_y = [], []
            curr_x, curr_y = agent.start.j, agent.start.i 
            
            for prim_id in agent.path:
                if prim_id >= 0:
                    prim = agent.control_set.list_all_prims[prim_id]
                    path_x.extend([curr_x + x for x in prim.x_coords])
                    path_y.extend([curr_y + y for y in prim.y_coords])
                    curr_x += prim.goal.j
                    curr_y += prim.goal.i
            
            # Render the continuous trajectory line
            if path_x and path_y:
                path_line, = ax.plot(
                    path_x, path_y, 
                    color=style['path_color'], 
                    alpha=style['path_alpha'], 
                    lw=style['path_lw'], 
                    zorder=style['z_order'] - 50,  # Paths remain strictly below all agents
                    marker=style['marker'],
                    markevery=10
                )
                graphics.append(path_line)
            
        # Bind the list of graphic elements to the agent instance
        agent_patches.append((graphics, agent))

    # 2. Populate dynamic obstacles
    # Generate a distinct color palette for obstacles
    num_obstacles = len(obstacle_set.obstacles)
    colors = plt.get_cmap('hsv', max(2, num_obstacles + 1))
    
    for i, obs in enumerate(obstacle_set.obstacles):
        add_agent_to_plot(obs, color=colors(i))
        
    # 3. Populate the primary ego-robot
    if robot:
        add_agent_to_plot(robot, color='lime', is_robot=True)

    # --- Real-time Rendering Setup ---
    # plt.close(fig) # Suppress duplicate static output in Jupyter
    # display_handle = display(fig, display_id=True)
    plt.close(fig) # Suppress standard static output in Jupyter
    
    display_handle = None
    if jupyter_display:
        display_handle = display(fig, display_id=True)

    # 4. Fast execution loop update function
    def update(t):
        """
        Updates the graphical coordinates and visibility of all agents for timestamp t.
        """
        ax.set_title(f"Simulation Time: {t:.1f}")
        
        for graphics, agent in agent_patches:
            pos = agent.get_position(t)
            
            if pos is not None:
                # Agent is active: update positions and ensure visibility
                for item in graphics:
                    item.set_visible(True)
                    # Move circles (main body and inner marker)
                    if isinstance(item, patches.Circle):
                        item.set_center((pos[0], pos[1]))
            else:
                # Agent's timeline ended: hide all associated graphical elements
                for item in graphics:
                    item.set_visible(False)
        
        # Execute an in-place update of the Jupyter display cell ONLY if enabled
        if display_handle:
            display_handle.update(fig)

        # Buffer frames for post-simulation GIF compilation
        if make_gif:
            fig.canvas.draw()  # Force canvas render before capturing buffer
            rgba_buffer = fig.canvas.buffer_rgba()
            image_array = np.asarray(rgba_buffer)
            im = Image.fromarray(image_array).convert('RGB')
            frames.append(im)

    # 5. GIF compilation and storage function
    def save_gif(filename="dynamic_sim.gif", fps=15, final_pause=0.5):
        """
        Compiles captured simulation frames into a looping GIF.
        """
        if len(frames) == 0:
            print("Warning: No frames captured to save.")
            return

        durations = int(1000 / fps)
        pause_frames = int(final_pause * fps)
        
        # Duplicate the terminal frame to induce a visual pause before looping
        frames_to_save = frames + [frames[-1]] * pause_frames
            
        print(f"Saving GIF... Total frames compiled: {len(frames_to_save)}")

        frames_to_save[0].save(
            filename, 
            save_all=True, 
            append_images=frames_to_save[1:], 
            optimize=True, 
            duration=durations, 
            loop=0
        )
        print(f"Simulation successfully saved to: {filename}")

    # Return bindings based on expected utility
    if make_gif:
        return update, save_gif
    else:
        return update
