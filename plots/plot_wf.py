import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
import netket as nk
from flax import nnx

from src.system import System
from src.ansatz import FermiSets

def main():
    N = 5
    dim = 2
    
    # 1. Initialize System and Ansatz
    system = System(N=N, dim=dim, mass=1.0, potential="qho_no_inter")
    ansatz = FermiSets(dim=dim, rngs=nnx.Rngs(42), N=N, hidden_units=32)
    
    sampler = nk.sampler.MetropolisGaussian(system.hi, 
                                            sigma=0.1,
                                            n_chains=1024,
                                            sweep_size=128) 
    
    vstate = nk.vqs.MCState(sampler, ansatz, n_samples=10**4, n_discard_per_chain=100)

    # 2. Load trained parameters
    
    mpack_path = "outputs/2026-04-20/22-53-56/optimization_results.mpack"
    with open(mpack_path, "rb") as file:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())

    # 3. Create the 2D grid for the active particle
    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.5, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    grid_2d = np.stack([X.ravel(), Y.ravel()], axis=-1) # Shape: (10000, 2)
    batch_size = grid_2d.shape[0]

    # 4. Define fixed coordinates for the 5 particles
    # Index 0 is a placeholder for the active particle.
    fixed_coords = jnp.array([
        [0.0, 0.0],   # Particle 0 (Active, to be overwritten)
        [-1.0, 1.0],  # Particle 1 (Fixed)
        [-1.0, -1.0], # Particle 2 (Fixed)
        [1.0, -1.0],  # Particle 3 (Fixed)
        [1.0, 1.0]    # Particle 4 (Fixed)
    ])

    # Tile the configuration for the whole batch
    full_configs = jnp.tile(fixed_coords, (batch_size, 1, 1)) # Shape: (10000, 5, 2)
    
    # Overwrite the coordinates of Particle 0 with the grid points
    full_configs = full_configs.at[:, 0, :].set(grid_2d) 

    # 5. Evaluate the wave function

    log_psi = vstate.log_value(full_configs)

    # Separate magnitude and phase
    log_mag = jnp.real(log_psi)
    phase = jnp.imag(log_psi)

    # Use nanmax so the 4 singularities don't poison the maximum value
    max_val = jnp.nanmax(log_mag)
    log_mag_shifted = log_mag - max_val

    # 6. Calculate the components
    real_psi = (jnp.exp(log_mag_shifted) * jnp.cos(phase)).reshape(100, 100)
    prob_density = jnp.exp(2.0 * log_mag_shifted).reshape(100, 100)

    # 7. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    
    # Plot Real part (Fixed the centering keyword)
    # Using vmin=-1 and vmax=1 ensures white is exactly 0.0
    c0 = axes[0].contourf(X, Y, real_psi, levels=50, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(c0, ax=axes[0])
    axes[0].set_title(r"Real Part: $\Re(\Psi)$")

    # Plot Probability Density
    c2 = axes[1].contourf(X, Y, prob_density, levels=50, cmap="magma")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title(r"Probability Density: $|\Psi|^2$")

    # Apply styling and plot fixed particles
    fixed_plot_coords = fixed_coords[1:] 
    for ax in axes:
        ax.scatter(fixed_plot_coords[:, 0], fixed_plot_coords[:, 1], 
                   color='white', marker='*', s=150, edgecolor='black', label='Fixed Particles')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig("plots/the_long_one_32h.png", bbox_inches="tight")
    print("ploted and saved")
    plt.show()

if __name__ == "__main__":
    main()