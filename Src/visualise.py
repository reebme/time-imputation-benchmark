import numpy as np
import matplotlib.pyplot as plt
import cmath

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

def plot_contribution_from_roots(rradius, rangle, bradius, bangle, lags, f_name):
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
        fname (string):
            Name of the file to save the plot in.
    """
    # For a real root: rangle is 0 or pi, and bangle must be 0
    if (rangle in [0, np.pi]) and (bangle != 0):
        raise ValueError("For a real root (rangle = 0 or pi), bangle must be 0.")

    # For a complex-conjugate root pair: rangle not in [0, pi], so bangle must be nonzero
    elif (rangle not in [0, np.pi]) and (bangle == 0):
        raise ValueError("For complex roots, bangle must be nonzero.")
    
    if rangle in [0, np.pi]:
        ampl_scaling = 1
        roots_str = " ".join([str(np.round(cmath.rect(rradius, rangle), 3))])
    else:
        ampl_scaling = 2
        roots_str = " ".join([str(np.round(cmath.rect(rradius, rangle), 3)), 
                                str(np.round(cmath.rect(rradius, -rangle), 3))])
   
    plt.figure(figsize = (10,7))

    y_str = (
        f"f(x) = "
        f"{ampl_scaling} * "
        f"{np.round(np.abs(bradius), 3)} * "
        f"{np.round(rradius, 3)}^x * "
        f"cos({np.round(rangle, 3)}* x + {np.round(bangle, 3)})"
    )

 
    x_axis = np.arange(0, lags)
    y = ampl_scaling * bradius * np.pow(rradius, x_axis) * np.cos(rangle*x_axis + bangle)
    plt.scatter(x_axis, y)

    for i in range(len(x_axis)):
        ymin_pt = np.min([0, y[i]])
        ymax_pt = np.max([0, y[i]])
        plt.vlines(x = x_axis[i], ymin = ymin_pt, ymax = ymax_pt, linewidth = 2)

    x_axis = np.arange(0, lags, 0.05)
    y = ampl_scaling * bradius * np.pow(rradius, x_axis) * np.cos(rangle*x_axis + bangle)
    plt.fill_between(x_axis, y, color = 'orange', alpha = 0.3)
    #plt.plot(x_axis, y, color = "orange")
    
    plt.axhline(y = 0, c = 'black')
    plt.axvline(x = 0, c = 'black')
    plt.grid(True)

    rc = " ".join(["Contribution of the root(s)", roots_str, "to the autocovariance"])
    title = "\n".join([rc, y_str])
    plt.title(title)

    plt.savefig(f_name, bbox_inches='tight')

    plt.show()
