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
    """Single particle QHO 1D wavefunction."""
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
    """
    Constructs the 3 Slater matrices for N=4, dim=2.
    x_batch expects shape: (batch, N, dim)
    """
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
    """
    Computes total overlap squared with the ground state manifold.
    log_psi_nn: (batch,)
    x_batch: (batch, N, dim)
    """
    matrices = get_slater_matrices(x_batch, N)
    total_s2 = 0.0
    
    for M in matrices:
        sign, log_psi_exact = jnp.linalg.slogdet(M)
        ratio = sign * jnp.exp(log_psi_exact - log_psi_nn)
        s2 = jnp.square(jnp.abs(jnp.mean(ratio))) / jnp.mean(jnp.square(jnp.abs(ratio)))
        total_s2 += s2
        
    return total_s2

def plot_wf(system, vstate, N, plot_path=None, step=None):
    """
    Plots the Neural Network probability density vs the Analytical probability density.
    Can be called from a callback during training or standalone.
    """
    dim = system.dim
    n_dots = N - 1 
    
    # 1. Create the grid
    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.5, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    grid_2d = np.stack([X.ravel(), Y.ravel()], axis=-1)
    batch_size = grid_2d.shape[0]
    
    # 2. Define fixed coordinates
    R = 1.0
    angles = jnp.array([jnp.pi / 2 + 2 * jnp.pi * k / n_dots for k in range(n_dots)])
    ngon_coords = jnp.stack([R * jnp.cos(angles), R * jnp.sin(angles)], axis=-1)

    fixed_coords = jnp.vstack([
        jnp.array([[0.0, 0.0]]),  # Active particle placeholder
        ngon_coords               # Fixed particles
    ])

    full_configs = jnp.tile(fixed_coords, (batch_size, 1, 1))
    full_configs = full_configs.at[:, 0, :].set(grid_2d) 

    # 3. Evaluate Neural Network Density
    # NetKet usually expects a flattened array (batch, N * dim) for evaluation
    full_configs_flat = jnp.reshape(full_configs, [-1, N * dim])
    log_psi_nn = vstate.log_value(full_configs_flat)
    
    log_mag_nn = jnp.real(log_psi_nn)
    # Shift to prevent overflow when exponentiating
    log_mag_shifted = log_mag_nn - jnp.nanmax(log_mag_nn)
    prob_density_nn = jnp.exp(2.0 * log_mag_shifted).reshape(100, 100)

    # 4. Evaluate Analytical Density
    # For a degenerate ground state, a uniform superposition is a fair visual comparison
    matrices = get_slater_matrices(full_configs, N)
    psi_ana_cmplx = jnp.zeros(batch_size, dtype=jnp.complex64)
    
    for M in matrices:
        sign, log_det = jnp.linalg.slogdet(M)
        # We can safely exponentiate directly here because QHO analytic log-dets 
        # around the origin for N=4 don't overflow standard float range.
        psi_ana_cmplx += sign * jnp.exp(log_det)
    
    prob_density_ana = jnp.square(jnp.abs(psi_ana_cmplx))
    # Normalize analytical density to [0, 1] for visual parity with NN
    prob_density_ana = prob_density_ana / jnp.max(prob_density_ana)
    prob_density_ana = prob_density_ana.reshape(100, 100)

    # 5. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    c0 = axes[0].contourf(X, Y, prob_density_nn, levels=50, cmap="magma")
    fig.colorbar(c0, ax=axes[0])
    title_nn = "NN Density: $|\Psi_{VMC}|^2$" if step is None else f"NN Density Step {step}"
    axes[0].set_title(title_nn)

    c1 = axes[1].contourf(X, Y, prob_density_ana, levels=50, cmap="magma")
    fig.colorbar(c1, ax=axes[1])
    axes[1].set_title(r"Analytic Superposition Density: $|\Psi_{exact}|^2$")

    fixed_plot_coords = fixed_coords[1:] 
    for ax in axes:
        ax.scatter(fixed_plot_coords[:, 0], fixed_plot_coords[:, 1], 
                   color='white', marker='*', s=150, edgecolor='black', label='Fixed Particles')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if plot_path:
        os.makedirs(plot_path, exist_ok=True)
        filename = "psi_comparison.png" if step is None else f"psi_comparison_step_{step}.png"
        plt.savefig(os.path.join(plot_path, filename), bbox_inches="tight")
        print(f"Plot saved to {os.path.join(plot_path, filename)}")
    else:
        plt.show()
    
    plt.close(fig) # Prevent memory leaks if called in a loop


if __name__ == "__main__": 
    N = 4
    dim = 2
    model_path = "/home/ilya/FermiNQS/outputs/2026-05-12/19-42-14/checkpoints/step_650.mpack"

    # Initialize System and Ansatz
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
    vstate = nk.vqs.MCState(sampler, ansatz, n_samples=10**3, n_discard_per_chain=100)

    # Load trained parameters
    with open(model_path, "rb") as file:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())

    # Run the plot
    plot_wf(system, vstate, N, plot_path=".")