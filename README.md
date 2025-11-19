Two-Link Robot Manipulator – Optimal Control Methods
📚 Import Libraries

This project uses the following Python libraries:

matplotlib – for plotting and visualization

NumPy – for numerical computations

cvxpy – for convex optimization and MPC

sympy – for symbolic mathematics

📝 Project Description

This project implements and compares different optimal control methods applied to a two-link robot manipulator.
The main goals include:

Modeling the dynamics of a two-link robotic arm

Implementing optimal control laws

Running linear MPC

Performing trajectory tracking

Visualizing system behavior and generating animations

Each task corresponds to a specific assignment and can be run independently unless stated otherwise.

📁 Code Structure & Description
dynamics.py

Contains two core functions that define the robot manipulator dynamics.
These functions are called by other scripts throughout the project.

cost.py

Includes cost-related functions used for optimization and control tasks.

solver_linear_mpc.py

Contains the linear MPC solver function used by the MPC task.

🧪 Task Files
Task1.py

Contains the code for Assignment Task 1

Plots will run sequentially

Works independently

Task2.py

Contains the code for Assignment Task 2

Plots run one by one

Works independently

Task3 (tracking).py

Implements Assignment Task 3

First runs Task2, then executes the Task 3 tracking results

Requires waiting for the plots/output to finish

Task4 (MPC).py

Implements Assignment Task 4 (Model Predictive Control)

First runs Task2, then executes Task 4 MPC computations

Visualization/plots follow afterward

Task5 (animation).py

Generates an animation of the manipulator

Runs Task2, then Task3, then produces the animation

Requires waiting for the full simulation to complete

🗂️ Extra (Not Required for Project)

Folder: extra/
Contains additional practice scripts used for learning robot manipulator concepts.
These files are not necessary for the main project.

🚀 How to Run the Project

Ensure all required libraries are installed.

Run any task file individually to generate results.

For Task 3, Task 4, and Task 5, allow time for dependent tasks to finish executing.
