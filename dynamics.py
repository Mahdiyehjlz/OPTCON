import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Time step for discretization
dt = 0.01
ni = 1  # Number of inputs
ns = 4  # Number of states

# Define symbolic variables
m1, m2, l1, l2, r1, r2, I1, I2, g, f1, f2 = sp.symbols('m1 m2 l1 l2 r1 r2 I1 I2 g f1 f2')
theta1, theta2, dtheta1, dtheta2 = sp.symbols('theta1 theta2 dtheta1 dtheta2')
u_sym = sp.symbols('u_sym')

# Define the mass (M) matrix
M11 = I1 + I2 + m1 * r1**2 + m2 * (l1**2 + r2**2) + 2 * m2 * l1 * r2 * sp.cos(theta2)
M12 = I2 + m2 * (r2**2 + l1 * r2 * sp.cos(theta2))
M21 = M12
M22 = I2 + m2 * r2**2
M = sp.Matrix([[M11, M12], [M21, M22]])

# Define the Coriolis (C) matrix
C1 = -m2 * l1 * r2 * dtheta2 * sp.sin(theta2) * (2 * dtheta1 + dtheta2)
C2 = m2 * l1 * r2 * sp.sin(theta2) * dtheta1**2
C = sp.Matrix([[C1], [C2]])

# Define the gravity (G) vector
G1 = g * (m1 * r1 + m2 * l1) * sp.sin(theta1) + m2 * g * r2 * sp.sin(theta1 + theta2)
G2 = m2 * g * r2 * sp.sin(theta1 + theta2)
G = sp.Matrix([[G1], [G2]])

# Define the friction (F) vector
F = sp.Matrix([[f1, 0], [0, f2]])

# Control input
tau = sp.Matrix([[u_sym], [0]])

# State vector
v = sp.Matrix([[dtheta1], [dtheta2]])

# Calculate the inverse of the mass matrix
M_inv = M.inv()

# Solve for accelerations
ddtheta = M_inv @ (tau - C - F @ v - G)

# Define the state vector and its derivative
x_sym = sp.Matrix([[theta1], [theta2], [dtheta1], [dtheta2]])
dx_sym = sp.Matrix([[dtheta1], [dtheta2], [ddtheta[0]], [ddtheta[1]]])

# Discretize the dynamics using the Euler method
x_next = x_sym + dt * dx_sym

# Compute the Jacobians of the discretized system
dfx_sym = x_next.jacobian(x_sym)  # Jacobian with respect to the state
dfu_sym = x_next.jacobian([u_sym])  # Jacobian with respect to the control input

# Substitute numerical values for the parameters
subs = {
    'm1': 2.0, 'm2': 2.0, 'l1': 1.5, 'l2': 1.5, 'r1': 0.75, 'r2': 0.75,
    'I1': 1.5, 'I2': 1.5, 'g': 9.81, 'f1': 0.1, 'f2': 0.1
}

# Substitute the values into the symbolic expressions
x_next_subs = x_next.subs(subs)
dfx_subs = dfx_sym.subs(subs)
dfu_subs = dfu_sym.subs(subs)

# Lambdify the substituted expressions for numerical evaluation
x_next_lambdified = sp.lambdify((theta1, theta2, dtheta1, dtheta2, u_sym), x_next_subs, 'numpy')
dfx_lambdified = sp.lambdify((theta1, theta2, dtheta1, dtheta2, u_sym), dfx_subs, 'numpy')
dfu_lambdified = sp.lambdify((theta1, theta2, dtheta1, dtheta2, u_sym), dfu_subs, 'numpy')


def dynamics(x, u, Ts):
    """
    Compute the next state and Jacobians using the discretized dynamics.

    Args:
        x: Current state (4, Ts)
        u: Current control input (1, Ts)
        Ts: Number of time steps

    Returns:
        x_next: Next state (4, Ts)
        dfx: Jacobian with respect to the state (4, 4, Ts)
        dfu: Jacobian with respect to the control input (4, 1, Ts)
    """
    Ts = int(Ts)
    ns = x.shape[0]  # Get the number of states dynamically
    ni = u.shape[0]  # Get the number of inputs dynamically
    dfx_numm = np.zeros((ns, ns, Ts-1))  # Corrected shape: Ts-1
    dfu_numm = np.zeros((ns, ni, Ts-1))  # Corrected shape: Ts-1
    x_nextnu = np.zeros((ns, Ts))
    
    for tt in range(Ts - 1):
        # Extract COLUMN vectors (2D arrays)
        #x = np.atleast_2d(x)  # Converts 1D to 2D if necessary
        #x_col = x[:, tt].reshape(-1, 1)
        #x_col = x[:, tt].reshape(-1, 1)  # Reshape to (4, 1) - Column vector
        #u = np.atleast_2d(x)  # Converts 1D to 2D if necessary
        #u_col = u[:, tt].reshape(-1, 1)  # Reshape to (1, 1) - Column vector

        #print(f"x_col shape: {x_col.shape}, x_col: {x_col}")
        #print(f"u_col shape: {u_col.shape}, u_col: {u_col}")
        x_col = np.atleast_2d(x).reshape(-1, 1)  # Ensure correct shape
        u_col = np.atleast_2d(u).reshape(-1, 1)


        # Now access elements safely
        x_nextnum = np.array(x_next_lambdified(x_col[0, 0], x_col[1, 0], x_col[2, 0], x_col[3, 0], u_col[0, 0])).reshape(-1)
         
        # Call lambdified functions with column vectors
        #x_nextnum = np.array(x_next_lambdified(x_col[0,0], x_col[1,0], x_col[2,0], x_col[3,0], u_col[0,0])).reshape(-1) #Pass single values to the lambdified functions and reshape the output
        dfx_num = np.array(dfx_lambdified(x_col[0,0], x_col[1,0], x_col[2,0], x_col[3,0], u_col[0,0]))
        dfu_num = np.array(dfu_lambdified(x_col[0,0], x_col[1,0], x_col[2,0], x_col[3,0], u_col[0,0]))

        x_nextnu[:, tt] = x_nextnum
        dfx_numm[:, :, tt] = dfx_num
        dfu_numm[:, :, tt] = dfu_num

    # Calculate the last state using the last control input (important!)
    #x_col_last = x[:, -1].reshape(-1, 1)
    x = np.atleast_2d(x).T  # Ensures x is at least (N, T)
    x_col_last = x[:, -1].reshape(-1, 1)
    
    u = np.atleast_2d(x)
    u_col_last = u[:, -1].reshape(-1, 1)
    
    # Check if x_col_last has at least 4 rows before accessing indices
    print(x.shape)
    print(x_col_last.shape)
    x_nextnum_last = np.array(x_next_lambdified(x_col_last[0, 0], x_col_last[1, 0], x_col_last[2, 0], x_col_last[3, 0], u_col_last[0, 0])).reshape(-1)
    #x_nextnum_last = np.array(x_next_lambdified(x_col_last[0,0], x_col_last[1,0], x_col_last[2,0], x_col_last[3,0], u_col_last[0,0])).reshape(-1)
    x_nextnu[:, -1] = x_nextnum_last

    return x_nextnu, dfx_numm, dfu_numm


def temp_dynamics(x0, stepsize, deltau, Ts, uu):
    """
    Simulate the system dynamics with a perturbed control input.

    Args:
        x0: Initial state (4,)
        stepsize: Step size for the perturbation
        deltau: Perturbation in the control input (1, Ts)
        Ts: Number of time steps
        uu: Nominal control input (1, Ts)

    Returns:
        uu_temp: Perturbed control input (1, Ts)
        xx_temp: Resulting state trajectory (4, Ts)
    """
    xx_temp = np.zeros((ns, Ts))
    uu_temp = np.zeros((ni, Ts))

    xx_temp[:, 0] = x0

    for tt in range(Ts - 1):
        # Update control input with step size and perturbation
        uu_temp[:, tt] = uu[:, tt] + stepsize * deltau[:, tt]

        # Get current state and control values
        theta1_val = xx_temp[0, tt]
        theta2_val = xx_temp[1, tt]
        dtheta1_val = xx_temp[2, tt]
        dtheta2_val = xx_temp[3, tt]
        u_sym_val = uu_temp[0, tt]

        # Compute the next state
        x_nextnum = np.array(x_next_lambdified(theta1_val, theta2_val, dtheta1_val, dtheta2_val, u_sym_val))
        xx_temp[:, tt + 1] = x_nextnum.squeeze()

    return uu_temp, xx_temp


# Test the dynamics function
Ts = 200  # Number of time steps
x_t = np.zeros((4, Ts))  # Initial state (all zeros)
u_t = np.zeros((1, Ts))  # Initial control input (all zeros)

# Compute the next state and Jacobians
x_next, dfx_t, dfu_t = dynamics(x_t, u_t, Ts)

# Print the results
print("Next state (x_next):")
print(x_next[:, 0])

print("Jacobian with respect to states (dfx):")
print(dfx_t[:, :, 0])

print("Jacobian with respect to control input (dfu):")
print(dfu_t[:, :, 0])


