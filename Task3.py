import numpy as np
import matplotlib.pyplot as plt
import cost as cst
import dynamics as d

dt=0.01
tf=2

Ts=int(tf/dt)

ns=4
ni=1

from Task2 import xx_star, uu_star
x_traj=xx_star
u_traj=uu_star


# Cost matrices
Q = np.eye(ns)  # State cost matrix
R = np.eye(ni)  # Input cost matrix
QT = np.eye(ns)  # Terminal state cost matrix

# Initialize variables
x = np.zeros((ns, Ts))
u = np.zeros((ni, Ts))

At = np.zeros((ns, ns, Ts))
Bt = np.zeros((ns, ni, Ts))

A = np.zeros((ns, ns, Ts))
B = np.zeros((ns, ni, Ts))

PP = np.zeros((ns, ns, Ts))
KK = np.zeros((ni, ns, Ts - 1))

# Compute A and B matrices from dynamics
A, B = d.dynamics(x_traj[:, :], u_traj[:, :], Ts)[1:]


# Solve Riccati equation
PP[:, :, -1] = QT

for tt in reversed(range(Ts - 1)):
    QQt = Q
    RRt = R
    AAt = A[:, :, tt]
    BBt = B[:, :, tt]
    PPtp = PP[:, :, tt + 1]

    PP[:, :, tt] = QQt + AAt.T @ PPtp @ AAt - \
                   (AAt.T @ PPtp @ BBt) @ np.linalg.inv((RRt + BBt.T @ PPtp @ BBt)) @ (BBt.T @ PPtp @ AAt)

# Compute gain matrix KK
for tt in range(Ts - 1):
    QQt = Q
    RRt = R
    AAt = A[:, :, tt]
    BBt = B[:, :, tt]
    PPtp = PP[:, :, tt + 1]

    KK[:, :, tt] = -np.linalg.inv(RRt + BBt.T @ PPtp @ BBt) @ (BBt.T @ PPtp @ AAt)

# Simulate the system
#x[:, 0] = x_traj[:, 0]
x[:, 0] = [0.2, 0.2,0.2,0.2]

for tt in range(Ts - 1):
    u[:, tt] = u_traj[:, tt] + KK[:, :, tt] @ (x[:, tt] - x_traj[:, tt])
    x[:, tt + 1] = A[:, :, tt] @ x[:, tt] + B[:, :, tt] @ u[:, tt]

# Plot results
time = np.arange(Ts)  # Time index

# Create the figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle("Tracking Performance", fontsize=16)

# Plot x[0] vs x_ref[0]
axs[0].plot(time, x_traj[0, :], label="x_ref[0] (Reference)", linestyle='--', color='blue')
axs[0].plot(time, x[0, :], label="x[0] (Actual)", linestyle='-', color='orange')
axs[0].set_title("State Trajectory: x[0]")
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("x[0]")
axs[0].legend()
axs[0].grid()

# Plot x[1] vs x_ref[1]
axs[1].plot(time, x_traj[1, :], label="x_ref[1] (Reference)", linestyle='--', color='blue')
axs[1].plot(time, x[1, :], label="x[1] (Actual)", linestyle='-', color='orange')
axs[1].set_title("State Trajectory: x[1]")
axs[1].set_xlabel("Time Step")
axs[1].set_ylabel("x[1]")
axs[1].legend()
axs[1].grid()

# Plot u[0] vs u_ref[0]
axs[2].plot(time, u_traj[0, :], label="u_ref[0] (Reference)", linestyle='--', color='blue')
axs[2].plot(time, u[0, :], label="u[0] (Actual)", linestyle='-', color='orange')
axs[2].set_title("Control Input: u[0]")
axs[2].set_xlabel("Time Step")
axs[2].set_ylabel("u[0]")
axs[2].legend()
axs[2].grid()

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
