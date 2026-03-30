import json
import os
import matplotlib.pyplot as plt

def plot_err(log_path, plot_name, save_dir, exact_energy=None, max_iter=None, y_lim=None):
    with open(log_path, "r") as f:
        data = json.load(f)
    
    iters = data["Energy"]["iters"]
    energies = data["Energy"]["Mean"]
    errors = data["Energy"]["Sigma"]
    
    # Slicing for max_iter
    if max_iter is not None: 
        iters = iters[:max_iter]
        energies = energies[:max_iter]
        errors = errors[:max_iter]

    # Handle NetKet/Complex data structures
    if isinstance(energies, dict) and "real" in energies:
        energies = energies["real"]
    if isinstance(errors, dict) and "real" in errors:
        errors = errors["real"]
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(iters, energies, yerr=errors, label="VMC Energy", capsize=2, elinewidth=1, fmt='-o', markersize=2)
 
    if exact_energy is not None:
        plt.axhline(exact_energy, color="red", linestyle="--", label="Exact GS Energy")

    # --- Zoom Implementation ---
    if y_lim is not None:
        # y_lim should be a tuple or list: (bottom, top)
        plt.ylim(y_lim)
    
    plt.xlabel("VMC Iteration")
    plt.ylabel(r"Energy $\langle H \rangle$")
    plt.title(f"Convergence: {plot_name}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__": 
    log_path = "outputs/2026-03-30/22-20-57/optimization_results.log"
    plot_name = "LR(0.02)_replica_13-14-38_zoomed"
    save_dir = "plots"
    exact_GS = 11.0
    max_iter = 200
    
    # Define your zoom window here. 
    # For example, if your GS is 11.0, you might want to see 10.9 to 11.5
    zoom_range = (0.0, 100.0) 
    
    plot_err(log_path, plot_name, save_dir, exact_GS, max_iter=max_iter, y_lim=zoom_range)