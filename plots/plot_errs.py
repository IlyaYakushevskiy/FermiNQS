import json
import os
import matplotlib.pyplot as plt

def plot_err(log_path, plot_name, save_dir):
    with open(log_path, "r") as f:
        data = json.load(f)
    
    iters = data["Energy"]["iters"]
    energies = data["Energy"]["Mean"]
    errors = data["Energy"]["Sigma"]
    
    
    if isinstance(energies, dict) and "real" in energies:
        energies = energies["real"]
        
  
    if isinstance(errors, dict) and "real" in errors:
        errors = errors["real"]
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(iters, energies, yerr=errors, label="VMC Energy", capsize=2)
 
    
    plt.xlabel("VMC Iteration")
    plt.ylabel(r"Energy $\langle H \rangle$")
    plt.title(f"Convergence: {plot_name}")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()