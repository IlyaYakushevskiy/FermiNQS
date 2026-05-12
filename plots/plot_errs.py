import json
import os
import matplotlib.pyplot as plt

def plot_err(log_path, plot_name, save_dir, exact_energy=None, max_iter=None, y_lim=None, min_iter=None):
    with open(log_path, "r") as f:
        data = json.load(f)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    # 1. Safely extract data using .get() to avoid KeyErrors
    energy_data = data.get("Energy", {})
    iters = energy_data.get("iters")
    energies = energy_data.get("Mean")
    errors = energy_data.get("Sigma")
    
    # 2. Handle NetKet/Complex data structures
    if isinstance(energies, dict) and "real" in energies:
        energies = energies["real"]
    if isinstance(errors, dict) and "real" in errors:
        errors = errors["real"]
    
    # NEW: Define a helper to convert None to IEEE 754 NaN for any array
    def sanitize(arr):
        if arr is None:
            return None
        return [float('nan') if x is None else x for x in arr]

    # 3. Sanitize arrays first, then slice
    iters = sanitize(iters)
    energies = sanitize(energies)
    errors = sanitize(errors)

    if iters is not None:
        iters = iters[min_iter:max_iter]
    if energies is not None:
        energies = energies[min_iter:max_iter]
    if errors is not None:
        errors = errors[min_iter:max_iter]
   
    plt.figure(figsize=(8, 5))
    
    # 4. Plot data if available. Matplotlib handles yerr=None natively.
    if iters is not None and energies is not None:
        plt.errorbar(iters, energies, yerr=errors, label="Rhat", capsize=2, elinewidth=1, fmt='-o', markersize=2)
    else:
        print(f"Warning: Missing iteration or energy data for {plot_name}. Plotting available exact GS only.")
 
    if exact_energy is not None:
        plt.axhline(exact_energy, color="red", linestyle="--", label="Exact GS Energy")

    # --- Zoom Implementation ---
    if y_lim is not None:
        # y_lim should be a tuple or list: (bottom, top)
        plt.ylim(y_lim)
    
    plt.xlabel("VMC Iteration")
    plt.ylabel(r"Mean Energy $\langle H \rangle$")
    plt.title(f"Convergence: {plot_name}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__": 
    log_path = "outputs/2026-05-05/19-48-09/combined_optimization_results.log"
    plot_name = "N=4 , n = 64"
    save_dir = "plots"
    exact_GS = 8.0

    max_iter = 1500
    
    # Define your zoom window here. 
    # For example, if your GS is 11.0, you might want to see 10.9 to 11.5
    zoom_range = (7, 15) 
    
    plot_err(log_path, plot_name, save_dir, exact_GS, max_iter=max_iter, min_iter=0, y_lim=zoom_range)