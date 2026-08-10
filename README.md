# Path Planning with Motion Primitives in Dynamic Environments: SIPP on Lattices

[![Conference](https://img.shields.io/badge/Conference-ICR_2026-blue)](https://icr.nw.ru/2026/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![Docker](https://img.shields.io/badge/Docker-Supported-2496ED.svg)](https://www.docker.com/)

This repository contains the official implementation and experimental framework for our paper presented at the **XI International Conference on Interactive Collaborative Robotics (ICR 2026)**.

## 📖 About The Project

Path planning for non-holonomic robots in environments with dynamic, moving obstacles is a challenging task. This project provides a comprehensive, open-source framework that bridges **Safe Interval Path Planning (SIPP)** with a **State Lattice** representation. 

By defining the state space as discrete poses $(x, y, \theta)$ connected by kinematically feasible motion primitives aligned with a regular grid, we effectively handle system dynamics and sensor noise. While the individual concepts of SIPP and State Lattices are well-known, combining them robustly for dynamic grid environments (such as standard [MovingAI](https://movingai.com/benchmarks/grids.html) maps) requires careful implementation. This repository provides a complete end-to-end ecosystem: from motion primitive generation and custom scenario building to parallelized server-side rendering and deep benchmarking.

## 🔍 Visual Summary: Impact of Control Sets

The following benchmark summary demonstrates the algorithm solving the exact same task (identical map, start/goal states, and dynamic obstacles) using different control sets. Notice how increasing the expressiveness of the motion primitives (from a minimal kinematically feasible set to expanded $2^k$ sets) dramatically alters the resulting trajectory and execution timing.

<p align="center">
  <video src="https://github.com/user-attachments/assets/35c8c031-d350-4a1c-8daa-e325de2a36d0" width="800" controls autoplay loop muted playsinline></video>
  <br>
  <em>Visualization of SIPP execution across varying control sets.</em>
</p>

---

## 📂 Repository Structure

The repository is modularly designed, separating core algorithms from data generation and visualization tools:

* **`common/`**: Shared Python modules, including fundamental data structures, basic A* implementations, and Matplotlib visualization utilities.
* **`control_set_generator/`**: Scripts and Jupyter notebooks to generate custom motion primitives, calculate their motion profiles (time occupancy per grid cell), and tune/visualize their kinematics.
* **`data/`**: Pre-generated motion primitive control sets. Contains 6 default sets used in the paper's experiments.
* **`dynamic_obstacles/`**: Trajectory and dimensional data for dynamic obstacles, tailored for specific maps.
* **`maps/`**: Grid maps in the standard MovingAI format.
* **`media/`**: Auto-generated assets, including trajectory plots, GIFs, and MP4 animations.
* **`planners_library/`**: The core SIPP implementations running on motion primitives. Includes interactive Jupyter notebooks demonstrating the search process and trajectory generation.
* **`python-benchmark/`**: The benchmarking suite for large-scale comparative tests. Contains scripts to run massive evaluations and notebooks to aggregate results into publication-ready graphs and LaTeX tables.
* **`scenario_builder/`**: Tools and interactive notebooks for placing dynamic obstacles on chosen maps, generating their moving trajectories either manually or procedurally.
* **`server_side/`**: A containerized environment (Dockerfile and setup guide) optimized for high-performance, multi-core remote servers. Ideal for offloading the heavy parallelized rendering of complex multi-frame video animations.

---

## 🚀 Research Workflow: Getting Started

This repository provides everything needed to reproduce our results or adapt the planner for your own research. We recommend the following workflow:

1.  **Generate a Control Set:** Open the notebooks in `control_set_generator/` to define your robot's kinematics and generate new motion primitive sets.
2.  **Select a Map:** Download a grid from MovingAI (or use the provided ones) and place it in the `maps/` directory.
3.  **Build a Scenario:** Use `scenario_builder/` to spawn dynamic obstacles on your map and define their behavior.
4.  **Run the Planner:** Navigate to `planners_library/` and use the interactive notebooks to test the SIPP algorithm on your newly created scenario.
5.  **Render Animations (Optional):** If you are generating long, complex videos, deploy the `server_side/` Docker container to a powerful machine to render frames in parallel.
6.  **Benchmark & Analyze:** Use `python-benchmark/` to run large-scale automated tests across multiple maps and control sets, automatically generating charts and tables for your paper.

---

## 🎥 Complex Environment Demos

Here are two examples of the algorithm operating in highly constrained and large-scale scenarios:

<table align="center">
  <tr>
    <td align="center"><b>Tight Navigation (Empty Map)</b></td>
    <td align="center"><b>Large-Scale Search (Denver)</b></td>
  </tr>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/e81dd55a-c80e-4f4f-8390-4cadec6c8d21" width="400" controls autoplay loop muted playsinline></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/42a46162-a6c8-460a-b118-26eb123ca8ab" width="400" controls autoplay loop muted playsinline></video>
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>A large-footprint agent skillfully squeezing between dynamically moving obstacles.</em>
    </td>
    <td align="center">
      <em>Extensive pathfinding over a massive MovingAI map requiring prolonged trajectory calculations.</em>
    </td>
  </tr>
</table>

---

## 📄 Citation

If you use this code, ideas, or visualizations in your research, please cite our ICR 2026 paper:

**APA:**
> Agranovskiy, M. (2027). Path Planning with Motion Primitives in Dynamic Environments: SIPP on Lattices. In: Ronzhin, A., Gribova, V., Meshcheryakov, R. (eds) Interactive Collaborative Robotics. ICR 2026. Lecture Notes in Computer Science(), vol 16790. Springer, Cham. https://doi.org/10.1007/978-3-032-34387-1_29

**BibTeX:**
```bibtex
@InProceedings{10.1007/978-3-032-34387-1_29,
  author="Agranovskiy, Marat",
  title="Path Planning with Motion Primitives in Dynamic Environments: SIPP on Lattices",
  booktitle="Interactive Collaborative Robotics",
  year="2027",
  publisher="Springer Nature Switzerland",
  pages="406--420",
}
```
