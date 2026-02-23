import os
import matplotlib.pyplot as plt
import glob
import numpy as np
import argparse

def load_data(file_path):
    """
    Load stress-strain data from LAMMPS fix print output.
    Format: strain stress (skipped header lines starting with #)
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
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
    parser = argparse.ArgumentParser(description="Compare stress-strain results from multiple simulations.")
    parser.add_argument("dir", nargs="?", default=".", help="Root directory to search for results (default: .)")
    parser.add_argument("--pattern", default="**/stress_strain.txt", help="Glob pattern to find data files (default: **/stress_strain.txt)")
    parser.add_argument("--no-recursive", action="store_true", help="Do not search subdirectories recursively")
    parser.add_argument("--output", "-o", default="comparison_plot.png", help="Output PNG filename (default: comparison_plot.png)")
    parser.add_argument("--title", "-t", default=None, help="Plot title (optional)")
    parser.add_argument("--max-strain", type=float, default=None, help="Maximum strain to plot (0.0 to 1.0)")
    parser.add_argument("--max-y", type=float, default=None, help="Maximum stress for y-axis (GPa)")
    parser.add_argument("--min-y", type=float, default=None, help="Minimum stress for y-axis (GPa)")
    parser.add_argument("--no-label-max", action="store_true", help="Do not include max stress in legend labels")
    parser.add_argument("--sort", choices=["name", "stress"], default="name", help="Sort order in legend and summary (default: name)")

    args = parser.parse_args()

    # Search for files
    if args.no_recursive and args.pattern == "**/stress_strain.txt":
        args.pattern = "*/stress_strain.txt"

    search_path = os.path.join(args.dir, args.pattern)
    data_files = glob.glob(search_path, recursive=not args.no_recursive)
    
    if not data_files:
        print(f"No files found matching pattern: {search_path}")
        return

    print(f"Found {len(data_files)} files.")

    results = []

    for data_file in data_files:
        # Generate a label from the directory structure
        # Relative to args.dir, and take the parent folder name
        rel_path = os.path.relpath(data_file, args.dir)
        parent_dir = os.path.dirname(rel_path)
        
        # If the file is in a 'data' folder, use the parent of that folder as the label prefix if needed,
        # but usually the folder name where the file resides is most descriptive.
        # Example: simulations/CuAl/data/Cu100_Al100_300K/stress_strain.txt -> Cu100_Al100_300K
        label = os.path.basename(parent_dir) if parent_dir else "root"
        if label == "data":
             label = os.path.basename(os.path.dirname(parent_dir))
        
        data = load_data(data_file)
        
        if len(data) > 0:
            # Calculate Max Stress
            max_stress_idx = np.argmax(data[:, 1])
            res_max_stress = data[max_stress_idx, 1]
            res_max_strain = data[max_stress_idx, 0]
            results.append({
                'label': label,
                'data': data,
                'max_stress': res_max_stress,
                'max_strain': res_max_strain,
                'path': data_file
            })
        else:
            print(f"Warning: No valid data found in {data_file}")

    # Sorting
    if args.sort == "stress":
        results.sort(key=lambda x: x['max_stress'], reverse=True)
    else:
        results.sort(key=lambda x: x['label'])

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Use a better color cycle if many lines
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    styles = ['-', '--', '-.', ':']
    
    for i, res in enumerate(results):
        data = res['data']
        label = res['label']
        if not args.no_label_max:
            label += f" (Max: {res['max_stress']:.2f} GPa)"
            
        x = data[:, 0] * 100 # Convert to %
        y = data[:, 1]
        
        if args.max_strain is not None:
            mask = data[:, 0] <= args.max_strain
            x = x[mask]
            y = y[mask]
            
        plt.plot(x, y, label=label, 
                 color=colors[i % len(colors)], 
                 linestyle=styles[(i // len(colors)) % len(styles)])

    plt.xlabel("Strain (%)")
    plt.ylabel("Stress (GPa)")
    if args.max_strain is not None:
        plt.xlim(left=0, right=args.max_strain * 100)
    if args.min_y is not None or args.max_y is not None:
        plt.ylim(bottom=args.min_y, top=args.max_y)
    if args.title:
        plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save output
    # If args.output is a path, use it. Otherwise, save in args.dir
    if os.path.isabs(args.output) or os.path.dirname(args.output):
        output_path = args.output
    else:
        output_path = os.path.join(args.dir, args.output)
        
    plt.savefig(output_path, dpi=300)
    print(f"Comparison plot saved to {output_path}")
    
    # Print summary
    print("\n--- Summary (Sorted by {0}) ---".format(args.sort))
    summary_list = sorted(results, key=lambda x: x['max_stress'], reverse=True)
    for res in summary_list:
        print(f"{res['label']}: Max Stress = {res['max_stress']:.3f} GPa at Strain = {res['max_strain']*100:.2f}%")

if __name__ == "__main__":
    main()
