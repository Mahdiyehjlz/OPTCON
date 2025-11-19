import numpy as np
import cvxpy as cp

def solver_linear_mpc(AA, BB, QQ, RR, QQf, xxt,x_ref,u_ref,t, u1_max=60, u1_min=-60, x1_max=np.pi, x1_min=-np.pi, x2_max=np.pi, x2_min=-np.pi, x3_max=10, x3_min=-10, x4_max=10, x4_min=-10, T_pred=10):
    """
    Solve the constrained Linear MPC problem.

    Args:
        AA: State transition matrix (ns, ns, T_pred)
        BB: Control input matrix (ns, ni, T_pred)
        QQ: State cost matrix (ns, ns)
        RR: Control cost matrix (ni, ni)
        QQf: Terminal state cost matrix (ns, ns)
        xxt: Initial state (ns,)
        u1_max: Maximum control input (default: 1)
        u1_min: Minimum control input (default: -1)
        x1_max: Maximum theta1 (default: 20)
        x1_min: Minimum theta1 (default: -20)
        x2_max: Maximum theta2 (default: 20)
        x2_min: Minimum theta2 (default: -20)
        x3_max: Maximum dtheta1 (default: 20)
        x3_min: Minimum dtheta1 (default: -20)
        x4_max: Maximum dtheta2 (default: 20)
        x4_min: Minimum dtheta2 (default: -20)
        T_pred: Prediction horizon (default: 5)

    Returns:
        u_t: Optimal control input at time t (ni,)
        xx_mpc: Predicted state trajectory (ns, T_pred)
        uu_mpc: Predicted control input (ni, T_pred)
    """
    ns, ni = BB.shape[0], BB.shape[1]  # Number of states and inputs

    # Define optimization variables
    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))

    # Define cost function and constraints
    cost = 0
    constraints = [xx_mpc[:,0]==xxt]  

    for tt in range(T_pred - 1):
       
        cost += cp.quad_form(xx_mpc[:, tt]-x_ref[:,t+tt], QQ) + cp.quad_form(uu_mpc[:, tt]-u_ref[:,t+tt], RR)
        constraints += [
            xx_mpc[:, tt + 1] == AA[ :,:, tt] @ xx_mpc[:, tt] + BB[:, :, tt] @ uu_mpc[:, tt],  # Dynamics
            uu_mpc[:, tt] <= u1_max,  # Control input constraints
            uu_mpc[:, tt] >= u1_min,
            xx_mpc[0, tt] <= x1_max,  # State constraints
            xx_mpc[0, tt] >= x1_min,
            xx_mpc[1, tt] <= x2_max,
            xx_mpc[1, tt] >= x2_min,
            xx_mpc[2, tt] <= x3_max,
            xx_mpc[2, tt] >= x3_min,
            xx_mpc[3, tt] <= x4_max,
            xx_mpc[3, tt] >= x4_min,
        ]

    # Add terminal cost
    cost += cp.quad_form(xx_mpc[:, T_pred - 1]-x_ref[:,t+T_pred-1], QQf)
    constraints += [xx_mpc[:, 0] == xxt]  # Initial state constraint

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve()

    # Check for infeasibility
    if problem.status == "infeasible":
        raise ValueError("Constrained MPC problem is infeasible! Check your constraints.")

    return uu_mpc[:, 0].value, xx_mpc.value, uu_mpc.value