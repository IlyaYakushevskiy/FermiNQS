import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
import netket as nk
from flax import nnx

from src.system import System
from src.ansatz import FermiSets

def main():
    
    system = System(N=2, dim=1, mass=1.0, potential="qho_no_inter")
    ansatz = FermiSets(dim=1, rngs=nnx.Rngs(42), N=2, hidden_units=4)
    
    sampler = nk.sampler.MetropolisGaussian(system.hi, 
                                            sigma=0.1,
                                            n_chains=1024,
                                            sweep_size=32) 
    
    vstate = nk.vqs.MCState(sampler, ansatz, n_samples=100000, n_discard_per_chain=100)

    mpack_path = "outputs/2026-03-30/22-20-57/optimization_results.mpack"
    with open(mpack_path, "rb") as file:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())
        

    x = np.linspace(-4, 4, 100)
    X1, X2 = np.meshgrid(x, x)

    x1_flat = X1.ravel()
    x2_flat = X2.ravel()
    

    # Shape must be (batch_size, N, dim) -> (10000, 2, 1)
    grid_configs = np.stack([x1_flat, x2_flat], axis=-1) # Shape: (10000, 2)
    full_configs = np.expand_dims(grid_configs, axis=-1) # Shape: (10000, 2, 1)

    log_psi = vstate.log_value(full_configs)
    

    log_mag = jnp.real(log_psi)
    phase = jnp.imag(log_psi)

    log_mag_shifted = log_mag - jnp.max(log_mag)
    

    Z = (jnp.exp(log_mag_shifted) * jnp.cos(phase)).reshape(100, 100)
    
    # 6. Plotting
    plt.figure(figsize=(8, 6))

    contour = plt.contourf(X1, X2, Z, levels=50, cmap="RdBu_r")
    plt.colorbar(contour, label=r"Amplitude $\Psi(x_1, x_2)$")

    plt.plot([-4, 4], [-4, 4], 'k--', alpha=0.5, label=r"Nodal line $x_1 = x_2$")
    
    plt.title("Antisymmetric 2-Particle 1D Ground State")
    plt.xlabel(r"Particle 1 position ($x_1$)")
    plt.ylabel(r"Particle 2 position ($x_2$)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("plots/antisymmetry_2particles_1d.png", bbox_inches="tight")


if __name__ == "__main__":
    main()