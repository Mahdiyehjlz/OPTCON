import numpy as np
import matplotlib.pyplot as plt
import cost as cst
import dynamics as d


def ref(Ts, theta1_start, theta1_end):
    """
    Generate reference trajectories for the flexible robotic arm system.
    
    Args:
        t (np.array): Time array.
        theta1_start (float): Initial angle of Link 1.
        theta1_end (float): Final angle of Link 1.
        theta2_start (float): Initial angle of Link 2.
        theta2_end (float): Final angle of Link 2.

    Returns:
        x_ref (np.array): Reference state trajectory (4 x N), where N is the number of time steps.
        u_ref (np.array): Reference input torque trajectory (1 x N).
    """
    # System parameters
    m1 = 2.0
    m2 = 2.0
    l1 = 1.5
    r1 = 0.75
    r2 = 0.75
    g = 9.81

    t = np.linspace(0, 1, Ts)

    theta1 = np.zeros((1, Ts))
    theta2 = np.zeros((1, Ts))

    theta1_eq2=np.pi/7
    theta1eq3=np.pi/4

    t_start_transition = 2/10
    t_end_transition_first=3/10
    t_start_transition_second=5/10
    t_end_transition_second=6/10
    t_start_transition_third=7/10
    t_end_transition_third=8/10
    t_end_transition = 9/10 

    for i in range(Ts):
        s = t[i]

        if  s < t_start_transition:
            theta1[0, i] = theta1_start
            theta2[0, i] = -theta1_start

        elif t_start_transition < s  < t_end_transition_first:
            s_transition = (s - t_start_transition) / (t_end_transition_first - t_start_transition)
            theta1[0, i] = theta1_start + (theta1_eq2 - theta1_start) * (3*s_transition**2 - 2*s_transition**3)
            theta2[0, i] = -theta1_start + (-theta1_eq2 + theta1_start) * (3*s_transition**2 - 2*s_transition**3)

        elif t_end_transition_first < s  < t_start_transition_second:
            theta1[0, i] = theta1_eq2
            theta2[0, i] = -theta1_eq2

        elif t_start_transition_second < s  < t_end_transition_second:
            s_transition = (s - t_start_transition_second) / (t_end_transition_second - t_start_transition_second)
            theta1[0, i] = theta1_eq2 + (theta1eq3 - theta1_eq2) * (3*s_transition**2 - 2*s_transition**3)
            theta2[0, i] = -theta1_eq2 + (-theta1eq3 + theta1_eq2) * (3*s_transition**2 - 2*s_transition**3)
        
        elif t_end_transition_second < s  < t_start_transition_third:
            theta1[0, i] = theta1eq3
            theta2[0, i] = -theta1eq3  

        elif t_start_transition_third < s  < t_end_transition_third:             
            s_transition = (s - t_start_transition_third) / (t_end_transition_third - t_start_transition_third)
            theta1[0, i] = theta1eq3 + (theta1_end - theta1eq3) * (3*s_transition**2 - 2*s_transition**3)
            theta2[0, i] = -theta1eq3 + (-theta1_end + theta1eq3) * (3*s_transition**2 - 2*s_transition**3)
        
        else:
            theta1[0, i] = theta1_end
            theta2[0, i] = -theta1_end 

    theta1_dot = np.zeros_like(theta1)
    theta2_dot = np.zeros_like(theta2)


    u_ref = np.zeros((1, Ts))
    for tt in range(Ts):
        u_ref[:, tt] = g * (m1 * r1 + m2 * l1) * np.sin(theta1[:,tt]) + g * m2 * r2 * np.sin(theta1[:, tt] + theta2[:, tt])


    x_ref = np.vstack((theta1, theta2, theta1_dot, theta2_dot))

    return x_ref, u_ref




tf = 2
dt = 0.01
Ts = int(tf/dt)
ns = 4
ni = 1
max_iters = 20


#########################
#Cost matrix
Q=np.diag([100, 100, 1, 1])   #State cost
R=np.eye(ni)    #Control cost
S=np.zeros((ni, ns))
QT=np.diag([10, 10, 1, 1])

St = np.repeat(S[:, :, np.newaxis], Ts, axis=2)
QTt = np.repeat(QT[:, :, np.newaxis], Ts, axis=2)
Rt = np.repeat(R[:, :, np.newaxis], Ts, axis=2)
Qt = np.repeat(Q[:, :, np.newaxis], Ts, axis=2)

# Initial and Fial values for theta1
theta1_start = 0.0
theta1_end = np.pi/3 


# Generate reference trajectories
xx_ref, uu_ref = ref(Ts, theta1_start, theta1_end)

##################
#Initializing
##################
x= np.zeros((ns, Ts, max_iters))   # state seq.
u= np.zeros((ni, Ts, max_iters))  # input seq.

q= np.zeros((ns, Ts))  # Gradient of the stage cost with respect to x
r= np.zeros((ni, Ts-1))  # Gradient of the stage cost with respect to u

lmbd= np.zeros((ns, Ts, max_iters)) # lambdas - costate seq.

deltau = np.zeros((ni,Ts, max_iters)) # Du - descent direction
dJ = np.zeros((ni,Ts, max_iters))     # DJ - gradient of J wrt u

JJ = np.zeros(max_iters)      # collect cost
descent = np.zeros(max_iters) # collect descent direction
descent_arm = np.zeros(max_iters) # collect descent direction

x_traj= np.zeros((ns, Ts,max_iters)) #Initiliaize 
u_traj= np.zeros((ni, Ts-1,max_iters))

dfx = np.zeros((ns,ns, Ts))
dfu = np.zeros((ns,ni, Ts))
dfx_temp = np.zeros((ns,ns, Ts))
dfu_temp = np.zeros((ns,ni, Ts))

At= np.zeros((ns,ns, Ts))
Bt= np.zeros((ns,ni, Ts))

sigma = np.zeros((ni,Ts))

PP = np.zeros((ns,ns,Ts))
pp = np.zeros((ns,Ts))
KK = np.zeros((ni,ns,Ts-1))

########################
#Initial gues
#######################
x_init = np.repeat(xx_ref[:,0].reshape(-1,1), Ts, axis=1) #Shape 4xts
u_init = np.repeat(uu_ref.reshape(-1,1), Ts, axis=1)

########################
#Main
#########################
k=0 #start iter

stepsize_0=10
stepsize=stepsize_0

######################### save armijo 
armijo_stepsizes_per_iteration = []

for k in range(max_iters-1):
  print(f"'''''''Newton Iteration {k}:''''''''")
  
  #################################
  # calculate cost used in Armijo
  #################################
  '''
  Calculate to monitor convergence. 
  The cost is for the diffrence between the reference and the current trajectory
  in this k iteration
  '''
 
  JJ[k] = 0
  for t in range(Ts-1):
    #print("Shapes",Q.shape,R.shape,x[:,t, k].shape,u[:,t,k].shape, x_ref[:,t].shape, u_ref[:,t].shape)
    temp_cost = cst.stage_cost(Q,R,x[:,t, k],u[:,t,k], xx_ref[:,t], uu_ref[:, t])[0]
    JJ[k] += temp_cost

  temp_cost = cst.term_cost(QT,x[:,-1,k], xx_ref[:,-1])[0]
  JJ[k] += temp_cost
  
  print(f"Calculated cost {JJ[k]}:")
  ###########################
  #Decent direction
  ##########################
  print("''''' Find Gradients'''''")
  '''
  We find the terminal gradient and place it in the last time step for the current iteration k
  '''
  lmbd_temp = cst.term_cost(QT,x[:,-1,k], xx_ref[:,-1])[1] #4x1 gradient lambda terminal
  lmbd[:,-1,k] = lmbd_temp.squeeze()
 
  dfx, dfu = d.dynamics(x[:,:,k], u[:,:,k],Ts)[1:]
  
  
  for tt in reversed(range(Ts-1)):  # integration backward in time because we need q to calc lamb backwards
   
   qt, rt = cst.stage_cost(Q,R,x[:,tt, k], u[:,tt,k], xx_ref[:,tt], uu_ref[:,tt])[1:]
  
   q[:, tt] = qt.squeeze()  # Store the gradient with respect to x
   r[:, tt] = rt.squeeze()  # Store the gradient with respect to u
   
   At[:,:,tt]=dfx[:,:,tt]
   Bt[:,:,tt]=dfu[:,:,tt]
  
  
  print("''''' Backwards co state eq and compute Q,R,S QT'''''")
  for tt in reversed(range(Ts-1)):
   
    lmbd_temp = At[:,:,tt].T@lmbd[:,tt+1,k][:,None] + q[:,tt][:,None]  #4x1 costate equation [:,None] comvert (4,) to (4,1)
    
    lmbd[:,tt,k] = lmbd_temp.squeeze()
 
  ############
  #Riccatti
  #############
  print("''''''''Ricatti'''''''")
  PP[:,:,-1] = QTt[:,:,-1]
  pp[:,-1] = lmbd[:,-1,k]

  for tt in reversed(range(Ts-1)):
   QQt = Qt[:,:,tt]
   qqt = q[:,tt][:,None]
   RRt = Rt[:,:,tt]
   rrt = r[:,tt][:,None]
   SSt = St[:,:,tt]
   AAt = At[:,:,tt]
   BBt = Bt[:,:,tt]
   PPtp = PP[:,:,tt+1]
   pptp = pp[:, tt+1][:,None]
   
   MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt) #1x1
  
   mmt = rrt + BBt.T @ pptp #1x1

   PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt ) + QQt #4x4
   ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt #4x1

   PP[:,:,tt] = PPt
   pp[:,tt] = ppt.squeeze()
   
 
  for tt in range(Ts-1):
    QQt = Qt[:,:,tt]
    qqt = q[:,tt][:,None]
    RRt = Rt[:,:,tt]
    rrt = r[:,tt][:,None]
    AAt = At[:,:,tt]
    BBt = Bt[:,:,tt]
    SSt = St[:,:,tt]

    PPtp = PP[:,:,tt+1]
    pptp = pp[:,tt+1][:,None]

    # Check positive definiteness
    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt) #1x1
    mmt = rrt + BBt.T @ pptp #1X1

    KK[:,:,tt] = -MMt_inv@(SSt+BBt.T@PPtp@AAt)#1X4
    sigma_t = -MMt_inv@mmt #1X1
    sigma[:,tt] = sigma_t.squeeze()

  
  print("K and sigma found")
  
  ######################################
  #Find step size
  #####################################
  descent[k] = 0
  descent_arm[k] = 0

  for tt in reversed(range(Ts - 1)):
   
   dJ_temp = Bt[:,:,tt].T@lmbd[:,tt+1,k][:,None] + r[:,tt][:,None]     # gradient of J wrt u
   deltau[:, tt,k] =KK[:,:,tt]@(xx_ref[:,tt]-x[:, tt,k])+ sigma[:,tt]
 
   dJ[:,tt,k] = dJ_temp.squeeze()
   
   descent[k] += deltau[:,tt,k].T@deltau[:,tt,k]
   
   descent_arm[k] += dJ[:,tt,k].T@deltau[:,tt,k]
  
  print("descent",descent[k])
  print("descent arm",descent_arm[k])
  
  ###########################
  #Armijo rule
  ##########################
  print("''''''''''starting armijo rule''''''''''''''''''")

  c = 0.5
  beta = 0.7
  armijo_maxiters = 30 # number of Armijo iterations
  term_cond = 1e-6 #tolerance to archive of decent direction
  xx_temp = np.zeros((ns,Ts))
  uu_temp = np.zeros((ni,Ts))
  xx_temp[:,0] = xx_ref[:,0]
  stepsizes = []  # list of stepsizes
  costs_armijo = []
 

  for i in range(armijo_maxiters):
    print("''''''''''''''''''''''''''''''''")
    print("Iteration in armijo",i)

    uu_temp[:,:], xx_temp[:,:] = d.temp_dynamics(xx_ref[:,0], stepsize,deltau[:,:,k], Ts,u[:,:,k])
    
    JJ_temp=0
    for tt in range(Ts-1):
      temp_cost = cst.stage_cost(Q,R,xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
      JJ_temp += temp_cost

    temp_cost = cst.term_cost(QT,xx_temp[:,-1], xx_ref[:,-1])[0]
    JJ_temp += temp_cost
    
    stepsizes.append(stepsize)      # save the stepsize
    costs_armijo.append(np.min([JJ_temp, 100*JJ[k]]))    # save the cost associated to the stepsize

    if JJ_temp > JJ[k]  + c*stepsize*descent_arm[k]:
      # update the stepsize
      stepsize = beta*stepsize
      print("stepsize new",stepsize)
    
    else:
      print('Armijo stepsize = {:.3e}'.format(stepsize))
      break
      
            
    if i == armijo_maxiters -1:
      print("WARNING: no stepsize was found with armijo rule!")
  
  # Store the chosen stepsize for this Newton iteration
  armijo_stepsizes_per_iteration.append(stepsize)

  ###########################
  # Update Trajectory
  ###########################
  print("'''''''Update trajectory''''''''")
  # Set the initial state for the new trajectory
  
  x_next = np.zeros((ns, Ts))
  x_next[:, 0] = xx_ref[:, 0]
  x[:,0,k] = xx_ref[:, 0]

  # Iterate over time steps to update the trajectory
  
  # Update the control using the feedback gain and feedforward term
  uu_temp[:, tt] = u[:, tt, k] + KK[:, :, tt] @ (xx_temp[:, tt] - x[:, tt, k]) + stepsize * sigma[:, tt]

  # Update the state using the system dynamics
  next_state_trajectory = d.dynamics(x[:, tt], u[:, tt], Ts)[0]
  next_state = next_state_trajectory[:, tt]
  xx_temp[:, tt + 1] = next_state
  # Assuming d.dynamics returns the next state

  # Handle the last time step (copy the second-to-last control input)
  uu_temp[:, -1] = uu_temp[:, -2]

  # Store the updated trajectories for the next iteration
  x[:, :, k + 1] = xx_temp.copy()
  u[:, :, k + 1] = uu_temp.copy()

  ############################
  # Termination condition
  ############################

  if descent[k] <= term_cond:
    max_iters = k
    break
  
xx_star = x[:,:,max_iters-1]
uu_star = u[:,:,max_iters-1]
uu_star[:,-1] = uu_star[:,-2] # for plotting purposes

############################
# REFERENCE plots
# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot x_ref (states) in the first subplot
ax1.plot(np.arange(Ts), xx_ref[0, :], 'b', label=r'$\theta_1(t)$')
ax1.plot(np.arange(Ts), xx_ref[1, :], 'r', label=r'$\theta_2(t)$')

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('States')
ax1.legend()
ax1.set_title('Reference State Trajectories')
ax1.grid(True)

# Plot u_ref (input torque) in the second subplot
ax2.plot(np.arange(Ts), uu_ref[0,:], label=r'$u(t)$', color='red')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Torque (Nm)')
ax2.legend()
ax2.set_title('Reference Input Torque')
ax2.grid(True)

plt.tight_layout()
plt.show()

############################
# Plots
############################

################  STEPSIZE   #############
# Plot the chosen stepsize for each Newton iteration
# Plot the chosen stepsize for each Newton iteration
plt.figure('Armijo Step Sizes per Newton Iteration')
plt.plot(np.arange(len(armijo_stepsizes_per_iteration)), armijo_stepsizes_per_iteration, marker='o', linestyle='-', color='b')
plt.xlabel('Newton Iteration (k)')
plt.ylabel('Chosen Armijo Step Size')
plt.title('Chosen Armijo Step Size vs. Newton Iteration')
plt.grid(True)
plt.show()
############### COST AND DESCENT #######################

plt.figure('descent direction')
plt.plot(np.arange(max_iters), descent[:max_iters])
plt.xlabel('$k$')
plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)


plt.figure('cost')
plt.plot(np.arange(max_iters), JJ[:max_iters])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

############ optimal trajectory #######################

tt_hor = np.linspace(0, tf, Ts)

# Create subplots for all states and control input
fig, axs = plt.subplots(ns + ni, 1, sharex='all', figsize=(10, 12))

# Plot x1 (theta1)
axs[0].plot(tt_hor, xx_star[0, :], linewidth=2, label='Optimal $x_1$')
axs[0].plot(tt_hor, xx_ref[0, :], 'g--', linewidth=2, label='Reference $x_1$')
axs[0].grid()
axs[0].set_ylabel('$x_1$')
axs[0].legend()

# Plot x2 (theta2)
axs[1].plot(tt_hor, xx_star[1, :], linewidth=2, label='Optimal $x_2$')
axs[1].plot(tt_hor, xx_ref[1, :], 'g--', linewidth=2, label='Reference $x_2$')
axs[1].grid()
axs[1].set_ylabel('$x_2$')
axs[1].legend()

# Plot x3 (theta1_dot)
axs[2].plot(tt_hor, xx_star[2, :], linewidth=2, label='Optimal $x_3$')
axs[2].plot(tt_hor, xx_ref[2, :], 'g--', linewidth=2, label='Reference $x_3$')
axs[2].grid()
axs[2].set_ylabel('$x_3$')
axs[2].legend()

# Plot x4 (theta2_dot)
axs[3].plot(tt_hor, xx_star[3, :], linewidth=2, label='Optimal $x_4$')
axs[3].plot(tt_hor, xx_ref[3, :], 'g--', linewidth=2, label='Reference $x_4$')
axs[3].grid()
axs[3].set_ylabel('$x_4$')
axs[3].legend()

# Plot control input (u)
axs[4].plot(tt_hor, uu_star[0, :], 'r', linewidth=2, label='Optimal $u$')
axs[4].plot(tt_hor, uu_ref[0, :], 'r--', linewidth=2, label='Reference $u$')
axs[4].grid()
axs[4].set_ylabel('$u$')
axs[4].set_xlabel('Time (s)')
axs[4].legend()

plt.tight_layout()
plt.show()