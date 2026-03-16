import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    log_path = "outputs/2026-03-16/15-12-01/optimization_results.log"
    
    with open(log_path, "r") as f:
        data = json.load(f)
    
    iters = data["Energy"]["iters"]
    energies = data["Energy"]["Mean"]
    errors = data["Energy"]["Sigma"]
    
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(iters, energies, yerr=errors, label="VMC Energy", capsize=2)
    plt.axhline(1.5, color="red", linestyle="--", label="Exact Ground State")
    
    plt.xlabel("VMC Iteration")
    plt.ylabel(r"Energy $\langle H \rangle$")
    plt.title("QHO Convergence")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("plots/qho_convergence.png", bbox_inches="tight")

if __name__ == "__main__":
    main()