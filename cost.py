import numpy as np

def stage_cost(Q, R, x_t, u_t, x_ref, u_ref):
    """
    Compute the stage cost and its gradients.

    Args:
        Q: State cost matrix (ns, ns)
        R: Control cost matrix (ni, ni)
        x_t: Current state (ns,)
        u_t: Current control input (ni,)
        x_ref: Reference state (ns,)
        u_ref: Reference control input (ni,)

    Returns:
        ll: Stage cost (scalar)
        lx: Gradient of stage cost w.r.t. state (ns,)
        lu: Gradient of stage cost w.r.t. control (ni,)
    """
    # Ensure inputs are 2D arrays for matrix operations
    x_t = np.array(x_t).reshape(-1, 1)
    u_t = np.array(u_t).reshape(-1, 1)
    x_ref = np.array(x_ref).reshape(-1, 1)
    u_ref = np.array(u_ref).reshape(-1, 1)

    # Compute state and control deviations
    delta_x = x_t - x_ref
    delta_u = u_t - u_ref

    # Compute stage cost
    ll = 0.5 * (delta_x.T @ Q @ delta_x + delta_u.T @ R @ delta_u).item()

    # Compute gradients
    lx = (Q @ delta_x).squeeze()  # Ensure lx is a 1D array
    lu = (R @ delta_u).squeeze()  # Ensure lu is a 1D array

    return ll, lx, lu

def term_cost(Q_T, x_T, x_ref_T):
    """
    Compute the terminal cost and its gradient.

    Args:
        Q_T: Terminal state cost matrix (ns, ns)
        x_T: Terminal state (ns,)
        x_ref_T: Reference terminal state (ns,)

    Returns:
        llT: Terminal cost (scalar)
        lTx: Gradient of terminal cost w.r.t. state (ns,)
    """
    # Ensure inputs are 2D arrays for matrix operations
    x_T = np.array(x_T).reshape(-1, 1)
    x_ref_T = np.array(x_ref_T).reshape(-1, 1)

    # Compute state deviation
    delta_x_T = x_T - x_ref_T

    # Compute terminal cost
    llT = 0.5 * (delta_x_T.T @ Q_T @ delta_x_T).item()

    # Compute gradient
    lTx = Q_T @ delta_x_T

    return llT, lTx.squeeze()


#test
Q = np.diag([100, 100, 1, 1])  # State cost matrix
R = np.eye(1)                  # Control cost matrix
x_t = np.array([0.1, 0.2, 0.3, 0.4])  # Current state
u_t = np.array([0.5])          # Current control input
x_ref = np.array([0.0, 0.0, 0.0, 0.0])  # Reference state
u_ref = np.array([0.0])        # Reference control input

ll, lx, lu = stage_cost(Q, R, x_t, u_t, x_ref, u_ref)
print("Stage Cost:", ll)
print("Gradient w.r.t. State:", lx)
print("Gradient w.r.t. Control:", lu)

Q_T = np.diag([1, 1, 1, 1])  # Terminal state cost matrix
x_T = np.array([0.1, 0.2, 0.3, 0.4])  # Terminal state
x_ref_T = np.array([0.0, 0.0, 0.0, 0.0])  # Reference terminal state

llT, lTx = term_cost(Q_T, x_T, x_ref_T)
print("Terminal Cost:", llT)
print("Gradient w.r.t. State:", lTx)