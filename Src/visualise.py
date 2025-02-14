import numpy as np
import matplotlib.pyplot as plt

def plot_roots_with_unit_circle(roots):
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
    
    plt.show()
