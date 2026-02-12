import os
import matplotlib.pyplot as plt
import glob
import numpy as np

def load_data(file_path):
    """
    Load stress-strain data from LAMMPS fix print output.
    Format: strain stress (skipped 1 header line)
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Skip header line starting with #
            for line in lines:
                if line.strip().startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        strain = float(parts[0])
                        stress = float(parts[1]) # GPa
                        data.append([strain, stress])
                    except ValueError:
                        continue
    except FileNotFoundError:
        return np.array([])
        
    return np.array(data)

def main():
    # スクリプトがあるディレクトリを基準にする場合
    sim_dir = os.path.dirname(__file__) 
    # あるいは単にカレントディレクトリにする場合
    # sim_dir = "."
    
    job_dirs = glob.glob(os.path.join(sim_dir, "*"))
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    styles = ['-', '--', '-.', ':']
    
    max_stress_info = []

    for i, job_dir in enumerate(sorted(job_dirs)):
        job_name = os.path.basename(job_dir)
        data_file = os.path.join(job_dir, "stress_strain.txt")
        
        if os.path.exists(data_file):
            print(f"Loading data for {job_name}...")
            data = load_data(data_file)
            
            if len(data) > 0:
                # Calculate Max Stress
                max_stress_idx = np.argmax(data[:, 1])
                max_stress = data[max_stress_idx, 1]
                max_strain = data[max_stress_idx, 0]
                
                max_stress_info.append((job_name, max_stress, max_strain))
                
                # Plot
                label = f"{job_name} (Max: {max_stress:.2f} GPa)"
                plt.plot(data[:, 0]*100, data[:, 1], label=label, 
                         color=colors[i % len(colors)], linestyle=styles[i % len(styles)])
            else:
                print(f"Warning: No valid data found in {data_file}")
        else:
            print(f"Warning: stress_strain.txt not found in {job_dir}")

    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (GPa)")
    plt.title("Comparison of Cu-Al Interface Strength for Different Orientations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(sim_dir, "comparison_plot.png")
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")
    
    # Print summary
    print("\n--- Summary ---")
    max_stress_info.sort(key=lambda x: x[1], reverse=True)
    for name, stress, strain in max_stress_info:
        print(f"{name}: Max Stress = {stress:.3f} GPa at Strain = {strain*100:.2f}%")

if __name__ == "__main__":
    main()
