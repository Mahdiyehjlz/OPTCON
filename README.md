# Two-Link Robot Manipulator – Optimal Control Methods

## 📚 Import Libraries
This project uses the following Python libraries:

- **matplotlib** – for plotting and visualization  
- **NumPy** – for numerical computations  
- **cvxpy** – for convex optimization and MPC  
- **sympy** – for symbolic mathematics  

---

## 📝 Project Description
This project implements and compares different **optimal control methods** applied to a **two-link robot manipulator**.  
The tasks include:

- Modeling manipulator dynamics  
- Implementing optimal control strategies  
- Running linear Model Predictive Control (MPC)  
- Performing trajectory tracking  
- Visualizing results and generating animations  

---

## 📁 Code Structure & Description

### `dynamics.py`
Defines the dynamics of the two-link robot manipulator.  
Contains two functions that are used across multiple scripts.

### `cost.py`
Includes cost-related functions used in optimization and control tasks.

### `solver_linear_mpc.py`
Contains the linear MPC solver function.

---

## 🧪 Task Files

### `Task1.py`
- Implements Assignment Task 1  
- Plots run one by one  
- Works independently  

### `Task2.py`
- Implements Assignment Task 2  
- Plots run one by one  
- Works independently  

### `Task3 (tracking).py`
- Implements Assignment Task 3  
- Automatically runs **Task2** first  
- Then executes tracking results  
- Requires waiting for full output  

### `Task4 (MPC).py`
- Implements Assignment Task 4 (Model Predictive Control)  
- Automatically runs **Task2** first  
- Then runs the MPC computations  

### `Task5 (animation).py`
- Generates an animation of the manipulator  
- Runs **Task2**, then **Task3**  
- Produces animation after computations finish  

---

## 🚀 How to Run
1. Install all required libraries.  
2. Run any task file independently (except where dependencies are mentioned).  
3. For Task 3, Task 4, and Task 5, allow time for dependent tasks to finish.  

---

