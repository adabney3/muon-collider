import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twiss_plots import read_outx_file, plot_two_columns, plot_multiple_columns
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from beam_size import read_outx_with_headers, calculate_and_plot_beam_sizes
from phase_advance import compute_phase_advance

def calculate_particle_coordinates(df, params, phi=0, n_particles=100):
    """
    Calculate particle coordinates x(s) and x'(s)
    
    x(s) = sqrt(epsilon) * sqrt(beta(s)) * cos(psi(s) + phi)
    x'(s) = -sqrt(epsilon) * (alpha(s) * cos(psi(s) + phi) + sin(psi(s) + phi)) / sqrt(beta(s))
    
    """
    epsilon_x = params.get('EX', 0)
    epsilon_y = params.get('EY', 0)
    
    # If phi is a scalar, generate multiple particles with different phases
    if np.isscalar(phi):
        phi_array = np.linspace(0, 2*np.pi, n_particles)
    else:
        phi_array = np.array(phi)
    
    # Calculate for horizontal plane
    x_coords = []
    xp_coords = []
    
    for i, row in df.iterrows():
        beta_x = row['BETX']
        alpha_x = row['ALFX']
        psi_x = row['PSIX']
        
        # Calculate x and x' for each particle at this s position
        x_at_s = np.sqrt(epsilon_x) * np.sqrt(beta_x) * np.cos(psi_x + phi_array)
        xp_at_s = -np.sqrt(epsilon_x) * (alpha_x * np.cos(psi_x + phi_array) + 
                                          np.sin(psi_x + phi_array)) / np.sqrt(beta_x)
        
        x_coords.append(x_at_s)
        xp_coords.append(xp_at_s)
    
    # Calculate for vertical plane
    y_coords = []
    yp_coords = []
    
    for i, row in df.iterrows():
        beta_y = row['BETY']
        alpha_y = row['ALFY']
        psi_y = row['PSIY']
        
        y_at_s = np.sqrt(epsilon_y) * np.sqrt(beta_y) * np.cos(psi_y + phi_array)
        yp_at_s = -np.sqrt(epsilon_y) * (alpha_y * np.cos(psi_y + phi_array) + 
                                          np.sin(psi_y + phi_array)) / np.sqrt(beta_y)
        
        y_coords.append(y_at_s)
        yp_coords.append(yp_at_s)
    
    return {
        'x': np.array(x_coords),
        'xp': np.array(xp_coords),
        'y': np.array(y_coords),
        'yp': np.array(yp_coords),
        'phi': phi_array
    }

def plot_emittance_evolution(df, params, filename='twiss_IR_v09.outx'):
    """Plot emittance ellipses at different s positions"""
    
    # Select a few positions to plot
    n_plots = 6
    indices = np.linspace(0, len(df)-1, n_plots, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    epsilon_x = params.get('EX', 0)
    
    # Generate particle coordinates
    coords = calculate_particle_coordinates(df, params, n_particles=100)
    
    for idx, ax_idx in enumerate(indices):
        row = df.iloc[ax_idx]
        s_pos = row['S']
        name = row['NAME']
        
        # Get coordinates at this position
        x = coords['x'][ax_idx, :]
        xp = coords['xp'][ax_idx, :]
        
        # Plot particles
        axes[idx].plot(x * 1e3, xp * 1e3, 'b.', markersize=3, alpha=0.6)
        axes[idx].set_xlabel('x [mm]')
        axes[idx].set_ylabel("x' [mrad]")
        axes[idx].set_title(f's = {s_pos:.2f} m\n{name}')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axis('equal')
        
        # Add emittance ellipse
        beta_x = row['BETX']
        alpha_x = row['ALFX']
        gamma_x = (1 + alpha_x**2) / beta_x
        
        # Ellipse parameters
        width = 2 * np.sqrt(epsilon_x * beta_x) * 1e3  # mm
        height = 2 * np.sqrt(epsilon_x * gamma_x) * 1e3  # mrad
        
        # Rotation angle
        if beta_x != gamma_x:
            angle = np.degrees(0.5 * np.arctan2(-2*alpha_x, gamma_x - beta_x))
        else:
            angle = 0
        
        ellipse = Ellipse((0, 0), width, height, angle=angle,
                         fill=False, edgecolor='red', linewidth=2)
        axes[idx].add_patch(ellipse)
        
        # Add text with Twiss parameters
        text = f'β={beta_x:.1f} m\nα={alpha_x:.2f}\nε={epsilon_x*1e6:.2f} μm'
        axes[idx].text(0.05, 0.95, text, transform=axes[idx].transAxes,
                      verticalalignment='top', fontsize=8,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plot_phase_space_at_location(df, params, s_location=None, element_name=None):
    """Plot phase space at a specific location"""
    
    if element_name:
        row = df[df['NAME'] == element_name].iloc[0]
    elif s_location is not None:
        idx = (df['S'] - s_location).abs().argmin()
        row = df.iloc[idx]
    else:
        # Use IP (middle of the beamline)
        idx = len(df) // 2
        row = df.iloc[idx]
    
    s_pos = row['S']
    name = row['NAME']
    
    # Generate coordinates at this location
    coords = calculate_particle_coordinates(df, params, n_particles=1000)
    
    # Find index for this s position
    idx = df[df['S'] == s_pos].index[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Horizontal phase space
    x = coords['x'][idx, :] * 1e3
    xp = coords['xp'][idx, :] * 1e3
    ax1.plot(x, xp, 'b.', markersize=2, alpha=0.5)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel("x' [mrad]")
    ax1.set_title(f'Horizontal Phase Space at s={s_pos:.2f} m\n{name}')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Vertical phase space
    y = coords['y'][idx, :] * 1e3
    yp = coords['yp'][idx, :] * 1e3
    ax2.plot(y, yp, 'r.', markersize=2, alpha=0.5)
    ax2.set_xlabel('y [mm]')
    ax2.set_ylabel("y' [mrad]")
    ax2.set_title(f'Vertical Phase Space at s={s_pos:.2f} m\n{name}')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    filename = 'twiss_IR_v09.outx'
    
    print("Reading TWISS data...")
    params, df = read_outx_with_headers(filename)
    
    print("\nTWISS Parameters:")
    print(f"  Particle: {params.get('PARTICLE', 'N/A')}")
    print(f"  Energy: {params.get('ENERGY', 'N/A')} GeV")
    print(f"  Horizontal emittance (EX): {params.get('EX', 0)*1e6:.6f} μm")
    print(f"  Vertical emittance (EY): {params.get('EY', 0)*1e6:.6f} μm")

    df = compute_phase_advance(df)

    plot_emittance_evolution(df, params, filename)

    plot_phase_space_at_location(df, params, element_name='IP1')
