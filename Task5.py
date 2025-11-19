import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Task3 import x_traj, u_traj  # Import the trajectories from Task3test

def animate(xx_star, xx_ref, dt):
    """
    Animates the dynamics of a two-link robot arm.
    
    Parameters:
        - xx_star: Optimal state trajectory (4, T) (theta1, theta2, etc.)
        - xx_ref: Reference trajectory (4, T) (theta1_ref, theta2_ref, etc.)
        - dt: Sampling time in seconds
        
    Returns:
        None
    """
    TT = xx_star.shape[1]  # Number of time steps
    l1, l2 = 1.0, 1.0      # Lengths of the robot arm links

    # Set up the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-2.5, 2.5)  # Adjust limits as needed for the arm's reach
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal', adjustable='box')

    # Plot elements
    optimal_line1, = ax.plot([], [], 'b-', lw=3, label="Optimal Link 1")
    optimal_line2, = ax.plot([], [], 'g-', lw=3, label="Optimal Link 2")
    reference_line1, = ax.plot([], [], 'b--', lw=2, label="Reference Link 1")
    reference_line2, = ax.plot([], [], 'g--', lw=2, label="Reference Link 2")
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax.legend()
    ax.set_title("Flexible Robot Arm Animation")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Initial setup function for the animation
    def init():
        optimal_line1.set_data([], [])
        optimal_line2.set_data([], [])
        reference_line1.set_data([], [])
        reference_line2.set_data([], [])
        time_text.set_text('')
        return optimal_line1, optimal_line2, reference_line1, reference_line2, time_text

    # Update function for each frame of the animation
    def update(frame):
        # Extract optimal joint angles for the current frame
        theta1_opt, theta2_opt = xx_star[0, frame], xx_star[1, frame]
        theta1_ref, theta2_ref = xx_ref[0, frame], xx_ref[1, frame]

        # Optimal trajectory positions
        x1_opt, y1_opt = l1 * np.cos(theta1_opt), l1 * np.sin(theta1_opt)
        x2_opt, y2_opt = x1_opt + l2 * np.cos(theta1_opt + theta2_opt), y1_opt + l2 * np.sin(theta1_opt + theta2_opt)

        # Reference trajectory positions
        x1_ref, y1_ref = l1 * np.cos(theta1_ref), l1 * np.sin(theta1_ref)
        x2_ref, y2_ref = x1_ref + l2 * np.cos(theta1_ref + theta2_ref), y1_ref + l2 * np.sin(theta1_ref + theta2_ref)

        # Update optimal trajectory lines
        optimal_line1.set_data([0, x1_opt], [0, y1_opt])
        optimal_line2.set_data([x1_opt, x2_opt], [y1_opt, y2_opt])

        # Update reference trajectory lines
        reference_line1.set_data([0, x1_ref], [0, y1_ref])
        reference_line2.set_data([x1_ref, x2_ref], [y1_ref, y2_ref])

        # Update time text
        time_text.set_text(f'time = {frame*dt:.2f}s')

        return optimal_line1, optimal_line2, reference_line1, reference_line2, time_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=TT, init_func=init, blit=True, interval=dt*1000)

    # Display the animation
    plt.show()

# Parameters
dt = 0.01  # Time step
tf = 2     # Final time
Ts = int(tf / dt)  # Number of time steps

# Use the trajectories from Task3test
xx_star = x_traj  # Optimal state trajectory (4, T)
xx_ref = x_traj   # Reference trajectory (4, T)

# Call the animate function
animate(xx_star, xx_ref, dt)