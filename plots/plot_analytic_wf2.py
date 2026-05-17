import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import flax
import netket as nk
from flax import nnx

from src.system import System
from src.ansatz import FermiSets

def qho_1d_wf(n, x):
    norms = {
        0: jnp.power(jnp.pi, -0.25),
        1: jnp.power(4 * jnp.pi, -0.25),
        2: 1.0 / (jnp.sqrt(8) * jnp.power(jnp.pi, 0.25))
    }
    hermite = {
        0: 1.0,
        1: 2.0 * x,
        2: 4.0 * jnp.square(x) - 2.0
    }
    return norms[n] * hermite[n] * jnp.exp(-0.5 * jnp.square(x))

def get_slater_matrices(x_batch, N):
    base_orbitals = [(0,0), (1,0), (0,1)]
    top_choices = [(2,0), (1,1), (0,2)]
    
    def evaluate_orbital(nx, ny, r):
        return qho_1d_wf(nx, r[:, 0]) * qho_1d_wf(ny, r[:, 1])

    matrices = []
    for top in top_choices:
        orbitals = base_orbitals + [top]
        M = jnp.stack([
            jnp.stack([evaluate_orbital(nx, ny, x_batch[:, j, :]) for j in range(N)], axis=-1)
            for (nx, ny) in orbitals
        ], axis=1)
        matrices.append(M)
    
    return matrices

def compute_overlap_squared(log_psi_nn, x_batch, N):
    matrices = get_slater_matrices(x_batch, N)
    overlaps = {}
    total_s2 = 0.0
    
    state_names = ["State A (2,0)", "State B (1,1)", "State C (0,2)"]
    
    for name, M in zip(state_names, matrices):
        sign, log_psi_exact = jnp.linalg.slogdet(M) 
        ratio = sign * jnp.exp(log_psi_exact - log_psi_nn)
        
        s2 = jnp.square(jnp.abs(jnp.mean(ratio))) / jnp.mean(jnp.square(jnp.abs(ratio)))
        overlaps[name] = float(s2) 
        total_s2 += s2
        
    overlaps["Total_Manifold"] = float(total_s2)
    return overlaps

def plot_wf(system, vstate, plot_path, step=None):
    N = system.N
    dim = system.dim
    n_dots = N - 1 
    
    energy_stats = vstate.expect(system.H)

    samples = vstate.sample() 
    x_mcmc_flat = samples.reshape(-1, N * dim)
    x_mcmc = samples.reshape(-1, N, dim)
    
    log_psi_mcmc = vstate.log_value(x_mcmc_flat)
    overlaps = compute_overlap_squared(log_psi_mcmc, x_mcmc, N)

    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{lc}")
    print(r"\hline")
    print(r"Metric & Value \\")
    print(r"\hline")
    print(rf"VMC Energy & ${energy_stats.mean.real:.4f} \pm {energy_stats.error_of_mean:.4f}$ \\")
    print(r"\hline")
    print(rf"Overlap: State A (2,0) & {overlaps['State A (2,0)']:.4f} \\")
    print(rf"Overlap: State B (1,1) & {overlaps['State B (1,1)']:.4f} \\")
    print(rf"Overlap: State C (0,2) & {overlaps['State C (0,2)']:.4f} \\")
    print(r"\hline")
    print(rf"\textbf{{Total Overlap}} & \textbf{{{overlaps['Total_Manifold']:.4f}}} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Energy and squared overlap of the trained NQS with the exact ground state manifold.}")
    print(r"\label{tab:results}")
    print(r"\end{table}")

    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.5, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    grid_2d = np.stack([X.ravel(), Y.ravel()], axis=-1)
    batch_size = grid_2d.shape[0]
    
    R = 1.0
    angles = jnp.array([jnp.pi / 2 + 2 * jnp.pi * k / n_dots for k in range(n_dots)])
    ngon_coords = jnp.stack([R * jnp.cos(angles), R * jnp.sin(angles)], axis=-1)

    fixed_coords = jnp.vstack([
        jnp.array([[0.0, 0.0]]),  
        ngon_coords               
    ])

    full_configs = jnp.tile(fixed_coords, (batch_size, 1, 1))
    full_configs = full_configs.at[:, 0, :].set(grid_2d) 

    full_configs_flat = jnp.reshape(full_configs, [-1, N * dim])
    log_psi_nn_grid = vstate.log_value(full_configs_flat)

    log_mag_nn = jnp.real(log_psi_nn_grid)
    log_mag_shifted = log_mag_nn - jnp.nanmax(log_mag_nn)
    prob_density_nn = jnp.exp(2.0 * log_mag_shifted).reshape(100, 100)

    matrices = get_slater_matrices(full_configs, N)
    state_densities = []
    
    for M in matrices:
        sign, log_det = jnp.linalg.slogdet(M)
        prob = jnp.square(jnp.abs(sign * jnp.exp(log_det)))
        prob = prob / jnp.max(prob)
        state_densities.append(prob.reshape(100, 100))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    c0 = axes[0].contourf(X, Y, prob_density_nn, levels=50, cmap="magma")
    fig.colorbar(c0, ax=axes[0])
    title_nn = "NN Density: $|\Psi_{VMC}|^2$" if step is None else f"NN Density Step {step}"
    axes[0].set_title(title_nn, fontweight="bold")

    titles = ["Analytic State A: (2,0)", "Analytic State B: (1,1)", "Analytic State C: (0,2)"]
    for i in range(3):
        ax = axes[i+1]
        c = ax.contourf(X, Y, state_densities[i], levels=50, cmap="magma")
        fig.colorbar(c, ax=ax)
        ax.set_title(titles[i])

    fixed_plot_coords = fixed_coords[1:] 
    for ax in axes:
        ax.scatter(fixed_plot_coords[:, 0], fixed_plot_coords[:, 1], 
                   color='white', marker='*', s=150, edgecolor='black', label='Fixed Particles')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.grid(True, alpha=0.3)
        if ax == axes[0]: 
            ax.legend()

    plt.tight_layout()

    if plot_path:
        os.makedirs(plot_path, exist_ok=True)
        filename = "psi_comparison.png" if step is None else f"psi_comparison_step_{step}.png"
        out_file = os.path.join(plot_path, filename)
        plt.savefig(out_file, bbox_inches="tight")
    else:
        plt.show()
    
    plt.close(fig)

if __name__ == "__main__": 
    N = 4
    dim = 2
    
    checkpoint_dir = "/home/iyakus/scratch/FermiNQS/outputs/2026-05-14/18-38-51/checkpoints"
    plot_directory = "./plots"
    os.makedirs(plot_directory, exist_ok=True)

    system = System(N=N, dim=dim, mass=1.0, potential="qho_no_inter")
    
    ansatz = FermiSets(
        dim=dim,
        rngs=nnx.Rngs(43),
        N=N, 
        hidden_units=16,
        out_units=20,
        log=None
    )
    
    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=0.1, n_chains=32, sweep_size=128) 
    vstate = nk.vqs.MCState(sampler, ansatz, n_samples=1024, n_discard_per_chain=100)

    for step in range(0, 1000, 50):
        model_path = os.path.join(checkpoint_dir, f"step_{step}.mpack")
        
        if not os.path.exists(model_path):
            continue
            
        with open(model_path, "rb") as file:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())

        plot_wf(system, vstate, plot_path=plot_directory, step=step)