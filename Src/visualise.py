import numpy as np
import matplotlib.pyplot as plt

def plot_roots_with_unit_circle(roots, f_name):
    """
    Plot the roots on the complex plane with a unit circle reference.

    Parameters:
        roots (array-like): An array (or list) of complex numbers representing the roots.
    """
    # Ensure roots is a NumPy array
    roots = np.array(roots)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the unit circle
    unit_circle = plt.Circle((0, 0), 1, color='r', fill=False, linestyle='--')
    ax.add_artist(unit_circle)
    
    # Plot the roots as blue crosses
    ax.scatter(roots.real, roots.imag, color='b', marker='.')

    for r in roots:
        ax.plot([0, r.real], [0, r.imag], color='blue', linestyle=':', linewidth=1)
    
    # Draw x and y axes through the origin
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')

    root_str = "\n".join([str(np.round(r, 4)) for r in roots])
    magnitudes = ", ".join([str(np.round(r, 4)) for r in np.abs(roots)])
    title = "\n".join(["Roots:", root_str, "Magnitudes of roots:", magnitudes])
    ax.set_title(title)
    
    # Determine axis limits: use the maximum of 1 and the absolute value of any root
    max_val = max(1, np.max(np.abs(roots)))
    margin = 0.5
    ax.set_xlim([-max_val - margin, max_val + margin])
    ax.set_ylim([-max_val - margin, max_val + margin])
    
    # Ensure the plot has equal aspect ratio
    ax.set_aspect('equal', adjustable='datalim')
    
    # Add a grid and legend
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(f_name, bbox_inches='tight')
    plt.show()

def plot_contribution_from_roots(rradius, rangle, bradius, bangle, lags):
    """
    Plots the contribution of a pair of complex conjugate roots
    or a real root (indicated by rangle and bangle equal 0)
    to the autocovariance over a series of lags.

    Parameters:
        rradius (float):
            The magnitude (radius) of the complex conjugate roots
            or of the real root.
        rangle (float) [radians]:
            The angle associated with the conjugate roots,
            if the root is real, this is 0.
        bradius (float):
            The radius of the coefficient in the solution
            to a difference Yule-Walker equation associated with the root(s).
        bangle (float):
            The angle of the coefficient in the solution
            to a difference Yule-Walker equation associated with the root(s).
        lags (int):
            The number of discrete time steps (lags)
            over which to calculate and display the contribution.
    """
    if (rangle == 0) != (bangle == 0):
        raise ValueError("Invalid input: rangle and bangle must be either both zero or both nonzero.")
    
    if rangle == 0:
        ampl_scaling = 1
    else:
        ampl_scaling = 2
    
    x_axis = np.arange(0, lags)
    y = ampl_scaling * bradius * np.pow(rradius, x_axis) * np.cos(rangle*x_axis + bangle)
    plt.scatter(x_axis, y)

    for i in range(len(x_axis)):
        ymin_pt = np.min([0, y[i]])
        ymax_pt = np.max([0, y[i]])
        plt.vlines(x = x_axis[i], ymin = ymin_pt, ymax = ymax_pt, linewidth = 2)

    x_axis = np.arange(0, lags, 0.1)
    y = ampl_scaling * bradius * np.pow(rradius, x_axis) * np.cos(rangle*x_axis + bangle)
    plt.fill_betweenx(y, x_axis, color = 'orange', alpha = 0.3)
    
    plt.axhline(y = 0, c = 'black')
    plt.axvline(x = 0, c = 'black')
    plt.grid(True)
    plt.show()
