import numpy as np
import matplotlib.pyplot as plt
import cost as cst
import dynamics as d
import solver_linear_mpc as s

# Simulation Parameters
tf = 2
dt = 0.01
Ts = int(tf / dt)  #T simulation
ns = 4
ni = 1

T_pred = 10  # Prediction horizon

# Cost matrices
Q = np.diag([1000, 1000, 10, 10])  # Increase state cost
R = np.eye(ni) * 0.01  # Reduce control cost
QT = np.eye(ns)  # Terminal state cost matrix


# Initialize Variables
xx = np.zeros((ns, Ts, T_pred))
uu = np.zeros((ni, Ts, T_pred))

# Generate reference trajectories
from Task2 import uu_star, xx_star
x_traj = xx_star
u_traj = uu_star

dfx = np.zeros((ns, ns, Ts))
dfu = np.zeros((ns, ni, Ts))

A = np.zeros((ns, ns, Ts))
B = np.zeros((ns, ni, Ts))

At = np.zeros((ns, ns, Ts))
Bt = np.zeros((ns, ni, Ts))


dfx, dfu = d.dynamics(x_traj[:, :], u_traj[:,:], Ts)[1:]
At= dfx # Extract the first slice (time step 0) from dfx
Bt = dfu # Extract the first slice (time step 0) from dfu
print("Linearized dynamics (dfx):", At[:,  0])
print("Linearized control (dfu):", Bt[:, 0])

# Input and State Bounds
u1max = 40
u1min = -u1max
x_bounds = np.array([[np.pi/2, -np.pi/2],  # x1 max/min
                     [np.pi/2, -np.pi/2],  # x2 max/min
                     [2, -2],  # x3 max/min
                     [2, -2]])  # x4 max/min


# MPC Trajectory Initialization
xx_real_mpc = np.zeros((ns, Ts))
uu_real_mpc = np.zeros((ni, Ts))

xx_mpc = np.zeros((ns, T_pred, Ts))
#uu_mpc = np.zeros((ni, T_pred, Tsim))


xx_real_mpc[:, 0] = [0.2, 0.2, 0.2, 0.2]
uu_real_mpc[:,0]=-1


print(xx_real_mpc[:,0].shape)
print(uu_real_mpc[:,0].shape)

# Main Loop
# Lists to save xx_real_mpc and uu_real_mpc at each iteration
xx_real_mpc_history = []
uu_real_mpc_history = []
    

for tt in range(Ts - T_pred):


  x_traj[:,tt] = xx_star[:,tt] 
  u_traj[:,tt]  = uu_star[:,tt] 
  dfx,dfu=d.dynamics(x_traj[:,tt],u_traj[:,tt],Ts)[1:]
  #dfx,dfu=d.dynamics(xx[:,:,tt],uu[:,:,tt],Ts)[1:]

  At[:, :, tt] = dfx[:, :, tt] 
  Bt[:, :, tt] = dfu[:, :, tt]

  #print(At)
  #print(Bt)
  print(f"trajectory{x_traj[:,tt]}")


  # Update Trajectory with MPC
  print(f"Iteration {tt}: Updating Trajectory")
  xx_t_mpc = xx_real_mpc[:,tt]  # Current state
  #xx_real_mpc changed tp x_traj
  # Solve MPC

  uu_real_mpc[:, tt] , xx_mpc[:, :, tt] = s.solver_linear_mpc(
      At, Bt, Q, R, QT, xx_t_mpc,x_traj,u_traj, tt) [:2]
        #u1_max=u1max, u1_min=u1min, 
        #x1_min=x_bounds[0, 1], x1_max=x_bounds[0, 0], 
        #x2_min=x_bounds[1, 1], x2_max=x_bounds[1, 0], 
        #x3_max=x_bounds[2, 0], T_pred=T_pred
      #)[:2]
  print(f"iteration{tt}")

  #x_next=d.dynamics(xx_real_mpc[:,tt],uu_real_mpc[:,tt],Ts)[0]
  #xx_real_mpc[:,tt+1]=x_next[:,tt]
  xx_real_mpc [:,tt+1] = At[:,:,tt] @ xx_real_mpc[:,tt] + Bt[:,:,tt] @ uu_real_mpc[:,tt]
  print(f"Updated state: {xx_real_mpc[:, tt+1]}")
  print(f"Applied control: {uu_real_mpc[:, tt]}")  
    # Save xx_real_mpc and uu_real_mpc at each iteration
  xx_real_mpc_history.append(xx_real_mpc[:, tt].copy())
  uu_real_mpc_history.append(uu_real_mpc[:, tt].copy())

# Convert lists to numpy arrays for easier plotting
xx_real_mpc_history = np.array(xx_real_mpc_history).T  # Transpose to match (ns, Ts)
uu_real_mpc_history = np.array(uu_real_mpc_history).T  # Transpose to match (ni, Ts)

# Plotting
plt.figure(figsize=(12, 10))

# Plot each state in a separate subplot and compare with reference trajectory
state_labels = ['x1', 'x2', 'x3', 'x4']
for i in range(ns):
    plt.subplot(ns, 1, i + 1)
    plt.plot(xx_real_mpc_history[i, :], label=f'MPC State x{i+1}')
    plt.plot(x_traj[i, :Ts], '--', label=f'Reference x{i+1}')
    plt.xlabel('Time Step')
    plt.ylabel(f'State x{i+1}')
    plt.title(f'State x{i+1} vs Reference Trajectory')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Plot control input
plt.figure(figsize=(12, 4))
plt.plot(uu_real_mpc_history[0, :], label='MPC Control u1')
plt.plot(u_traj[0, :Ts], '--', label='Reference u1')
plt.xlabel('Time Step')
plt.ylabel('Control Input (u1)')
plt.title('Control Input (u1) vs Reference Trajectory')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()