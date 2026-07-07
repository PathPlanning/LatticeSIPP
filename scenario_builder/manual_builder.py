import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import traceback
import copy
import os

from DO_structs import DiscreteState, ControlSet
from DO_graphics import draw_task_map, draw_task_map_fast

"""
MODULE: manual_builder.py (TrajectoryBuilder Class)
DESCRIPTION:
An interactive Jupyter notebook widget for manual construction of kinematically 
feasible trajectories for dynamic obstacles. It uses a predefined ControlSet 
(motion primitives) to build paths step-by-step, providing visual feedback 
via forward kinematics projections (primitive fan) and saves the trajectory history.
"""

class TrajectoryBuilder:
    def __init__(self, control_set: ControlSet, task_map, start_state: DiscreteState, fast_draw: bool = False, prefix_to_save: str = "./"):
        self.cs = control_set
        self.task_map = task_map
        self.start_state = copy.deepcopy(start_state)
        self.current_state = copy.deepcopy(start_state)
        self.theta_config = control_set.theta
        self.fast_draw = fast_draw
        self.prefix_to_save = prefix_to_save
        
        self.history = []
        self.trajectory_ids = []
        
        self.out_plot = widgets.Output()
        self.box_primitives = widgets.HBox()
        self.box_controls = widgets.HBox()
        
        self.ax = None
        self._init_controls()
        self.update_ui()

    def _init_controls(self):
        btn_undo = widgets.Button(description="Undo", button_style='warning', icon='undo')
        btn_undo.on_click(self._on_undo)
        
        btn_wait_10 = widgets.Button(description="Wait 10", button_style='info')
        btn_wait_10.on_click(lambda b: self._on_wait(10))
        
        btn_wait_50 = widgets.Button(description="Wait 50", button_style='info')
        btn_wait_50.on_click(lambda b: self._on_wait(50))
        
        btn_save = widgets.Button(description="Save Trajectory", button_style='success', icon='save')
        btn_save.on_click(self._on_save)
        
        self.box_controls.children = [btn_undo, btn_wait_10, btn_wait_50, btn_save]

    def _create_primitive_buttons(self):
        # Retrieves the action bundle for the current heading. 
        # Note: Primitives are visually sorted right-to-left (due to the image y-axis pointing downwards)
        bundle = self.cs.get_primitives_heading(self.current_state.theta) 

        buttons = []
        for i, prim_template in enumerate(bundle):
            btn = widgets.Button(description=f"P {i+1} (ID:{prim_template.id})", 
                                 layout=widgets.Layout(width='auto'))
            btn.on_click(lambda b, p=prim_template: self._on_prim_select(p))
            buttons.append(btn)
            
        self.box_primitives.children = buttons

    def _on_prim_select(self, prim_template):
        try:
            self.history.append(('prim', prim_template, copy.deepcopy(self.current_state)))
            self.trajectory_ids.append(prim_template.id)

            self.current_state.i += prim_template.goal.i
            self.current_state.j += prim_template.goal.j
            self.current_state.theta = prim_template.goal.theta
            
            self.update_ui()
            
        except Exception as e:
            with self.out_plot:
                print(f"ON-CLICK EVENT ERROR: {e}")
                import traceback
                traceback.print_exc()

    def _on_wait(self, ticks):
        try:
            self.history.append(('wait', ticks, copy.deepcopy(self.current_state)))
            self.trajectory_ids.append(-ticks)
            self.update_ui()
        except Exception as e:
            with self.out_plot:
                print(f"WAIT EVENT ERROR: {e}")

    def _on_undo(self, b):
        try:
            if not self.history:
                return
            action_type, data, prev_state = self.history.pop()
            self.trajectory_ids.pop()
            self.current_state = prev_state
            self.update_ui()
        except Exception as e:
            with self.out_plot:
                print(f"UNDO EVENT ERROR: {e}")

    def _on_save(self, b):
        with self.out_plot:
            try:
                # 1. Probe for an available filename index to prevent overwriting
                file_idx = 1
                while os.path.exists(f"{self.prefix_to_save}dynamic_obstacle_{file_idx}.txt") or \
                      os.path.exists(f"{self.prefix_to_save}dynamic_obstacle_{file_idx}.png"):
                    file_idx += 1
                
                filename_txt = f"{self.prefix_to_save}dynamic_obstacle_{file_idx}.txt"
                filename_png = f"{self.prefix_to_save}dynamic_obstacle_{file_idx}.png"
                
                # 2. Dump trajectory metadata and execution sequence to text file
                with open(filename_txt, 'w', encoding='utf-8') as f:
                    f.write(f"======= Dynamic obstacle description ======\n")
                    f.write("R: 1\n")
                    f.write(f"Start: i={self.start_state.i} j={self.start_state.j} theta={self.start_state.theta}\n")
                    f.write(f"PrimitivesIDs: {' '.join(map(str, self.trajectory_ids))}\n")
                self.ax.figure.savefig(filename_png, transparent=False, bbox_inches="tight", facecolor='white')
                
                # 3. Display an aesthetically pleasing status report in the UI
                print(f"\n--- ✅ Trajectory Saved ---")
                print(f"FILE: {filename_txt} with IMAGE: {filename_png}")
                print(f"💡 Note: The obstacle radius (R) can be changed manually in the file (defaulted to 1)")
                
            except Exception as e:
                print(f"FILE SAVING ERROR: {e}")
                import traceback
                traceback.print_exc()

    def update_ui(self):
        self._create_primitive_buttons()
        
        with self.out_plot:
            clear_output(wait=True)
            try:
                # Retrieve the figure object directly from the rendering function
                fig = self._draw_scene() 
                
                # Force immediate display update
                from IPython.display import display as ipy_display
                ipy_display(fig)
                plt.close(fig)
                
            except Exception as e:
                print(f"RENDERING ERROR: {e}")
                traceback.print_exc()

    def _draw_scene(self):
        if self.fast_draw:
            self.ax = draw_task_map_fast(self.task_map, self.start_state, self.current_state, 
                                    theta_config=self.theta_config)
        else:
            self.ax = draw_task_map(self.task_map, self.start_state, self.current_state, 
                               theta_config=self.theta_config)
            
        fig = self.ax.figure
        
        # 3. ALTERNATING COLORS FOR TRAJECTORY HISTORY
        colors = ['r', 'g']
        prim_count = 0
        
        for action_type, data, prev_state in self.history:
            if action_type == 'prim':
                prim_template = data
                # j-coordinate grows to the right (equivalent to Cartesian x)
                shifted_x = [x + prev_state.j for x in prim_template.x_coords]  
                # i-coordinate grows downwards (equivalent to Cartesian -y)
                shifted_y = [y + prev_state.i for y in prim_template.y_coords]  
                
                current_color = colors[prim_count % 2]
                self.ax.plot(shifted_x, shifted_y, color=current_color, linestyle='-', linewidth=1.8)
                prim_count += 1

        # 4. SEMI-TRANSPARENT RED DASHED LINES FOR THE PRIMITIVE FAN (FORWARD KINEMATICS)
        bundle = self.cs.get_primitives_heading(self.current_state.theta)
        for prim_template in bundle:
            shifted_x = [x + self.current_state.j for x in prim_template.x_coords]
            shifted_y = [y + self.current_state.i for y in prim_template.y_coords]
            
            # 'r--' denotes red color (r) and dashed line style (--)
            self.ax.plot(shifted_x, shifted_y, 'r--', linewidth=1.2, alpha=0.4)
            
        return fig

    def display(self):
        display(widgets.VBox([self.out_plot, self.box_primitives, self.box_controls]))
