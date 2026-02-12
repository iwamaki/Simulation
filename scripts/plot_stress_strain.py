import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_stress_strain(data_file, output_dir=None, show=True):
    """
    Plot stress-strain curve from LAMMPS simulation data.
    
    Args:
        data_file (str): Path to stress_strain.txt file
        output_dir (str): Directory to save the plot image. If None, saved to same dir as data_file
        show (bool): Whether to display the plot
    """
    
    # Read data
    data = np.loadtxt(data_file, skiprows=1)
    strain = data[:, 0]
    stress = data[:, 1]
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot
    ax.plot(strain * 100, stress, 'b-', linewidth=1.5, label='Stress-Strain')
    
    # Formatting
    ax.set_xlabel('Strain (%)', fontsize=12)
    ax.set_ylabel('Stress (GPa)', fontsize=12)
    ax.set_title('Cu(100) - Al(100) Interface Tensile Test at 300K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add statistics
    max_stress = np.max(stress)
    max_stress_strain = strain[np.argmax(stress)] * 100
    mean_stress = np.mean(stress)
    
    textstr = f'Max Stress: {max_stress:.2f} GPa @ {max_stress_strain:.3f}%\nMean Stress: {mean_stress:.2f} GPa'
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    if output_dir is None:
        output_dir = os.path.dirname(data_file)
    
    output_file = os.path.join(output_dir, 'stress_strain_curve.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    if show:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot stress-strain curve from LAMMPS data")
    parser.add_argument("data_file", help="Path to stress_strain.txt file")
    parser.add_argument("--output-dir", help="Output directory for plot image")
    parser.add_argument("--no-show", action="store_true", help="Don't display the plot")
    
    args = parser.parse_args()
    
    plot_stress_strain(args.data_file, args.output_dir, show=not args.no_show)