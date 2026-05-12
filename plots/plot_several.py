import json
import os
import math
import matplotlib.pyplot as plt

def plot_multiple_errs(log_configs, plot_name, save_dir, exact_energy=None, max_iter=None, y_lim=None, min_iter=0):
    """
    Plots multiple VMC convergence logs on a single figure using continuous lines.
    """
    # 1. LaTeX Typography Setup
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 1.5
    })

    plt.figure(figsize=(8, 6))

    def sanitize(arr):
        if arr is None:
            return None
        return [float('nan') if x is None else x for x in arr]

    # 2. Iterate through each log configuration
    for config in log_configs:
        path = config["path"]
        label = config["label"]
        color = config.get("color", "black") 

        if not os.path.exists(path):
            print(f"Warning: File {path} not found. Skipping.")
            continue

        with open(path, "r") as f:
            data = json.load(f)

        energy_data = data.get("Energy", {})
        iters = energy_data.get("iters")
        energies = energy_data.get("Mean")
        
        # Complex data handling
        if isinstance(energies, dict) and "real" in energies:
            energies = energies["real"]

        # Sanitize missing data 
        iters = sanitize(iters)
        energies = sanitize(energies)

        # Slice based on iterations
        if iters is not None:
            iters = iters[min_iter:max_iter]
        if energies is not None:
            energies = energies[min_iter:max_iter]

        if iters is not None and energies is not None:
            # --- Detect Collapse ---
            collapse_idx = None
            for i, e in enumerate(energies):
                if math.isnan(e):
                    collapse_idx = i
                    break
            
            # --- Plot clean lines ONLY (removed marker arguments) ---
            plt.plot(
                iters, energies, 
                label=label, color=color, 
                alpha=0.85, linestyle='-', linewidth=2
            )

            # --- Annotate the collapse ---
            if collapse_idx is not None and collapse_idx > 0:
                last_valid_iter = iters[collapse_idx - 1]
                last_valid_energy = energies[collapse_idx - 1]

                # The 'x' marker is explicitly preserved here for the crash point
                plt.plot(last_valid_iter, last_valid_energy, marker='x', color='red', markersize=10, markeredgewidth=2)
                
                plt.annotate(
                    r'Collapsed', 
                    xy=(last_valid_iter, last_valid_energy), 
                    xytext=(last_valid_iter + 50, last_valid_energy + 0.2), 
                    color='red',
                    fontsize=12,
                    arrowprops=dict(arrowstyle="->", color='red', lw=1.0)
                )

    # 3. Add Exact Ground State
    if exact_energy is not None:
        plt.axhline(exact_energy, color="black", linestyle="--", linewidth=2, label="Exact GS Energy")

    if y_lim is not None:
        plt.ylim(y_lim)
    
    plt.xlabel("VMC Iteration")
    plt.ylabel(r"Mean Energy $\langle H \rangle$")
    plt.title(f"{plot_name}")
    
    plt.legend(frameon=True, edgecolor='black', fancybox=False)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name.replace(' ', '_')}.pdf") 
    
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches="tight", dpi=300) 
    plt.close()

if __name__ == "__main__": 

    my_runs = [
        {
            "path": "/home/ilya/FermiNQS/outputs/2026-05-05/19-48-09/combined_optimization_results.log", 
            "label": r"hidden\_size = 64", 
            "color": "tab:blue"
        },
        {
            "path": "/home/ilya/FermiNQS/outputs/2026-05-06/10-20-24/optimization_results.log", 
            "label": r"hidden\_size = 32", 
            "color": "tab:orange"
        },
        {
            "path": "/home/ilya/FermiNQS/outputs/2026-05-06/14-50-13/optimization_results.log", 
            "label": r"hidden\_size = 16", 
            "color": "tab:green"
        }
    ]

    plot_name = "Mean energy convergence for N=4 and varied hidden size"
    save_dir = "plots/thesis"

    exact_GS = 8.0
    max_iter = 1500
    zoom_range = (7.8, 13.0) 
    
    plot_multiple_errs(
        log_configs=my_runs, 
        plot_name=plot_name, 
        save_dir=save_dir, 
        exact_energy=exact_GS, 
        max_iter=max_iter, 
        min_iter=0, 
        y_lim=zoom_range
    )