import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
import netket as nk
from flax import nnx

from src.system import System
from src.ansatz import FermiSets

def main( N , dim):

    system = System(N=N, dim=dim, mass=1.0, potential="qho_no_inter")
    ansatz = FermiSets(dim=2, rngs=nnx.Rngs(42), N=N, hidden_units=16)
    
    sampler = nk.sampler.MetropolisGaussian(system.hi, 
                                            sigma=0.1,
                                            n_chains=16,
                                            sweep_size=32) 
    
    vstate = nk.vqs.MCState(sampler, ansatz, n_samples=100000, n_discard_per_chain=100)

    mpack_path = "outputs/2026-03-26/19-00-52/optimization_results.mpack"
    with open(mpack_path, "rb") as file:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())

    batch_size = 10000 # grid is 100x100 , use batch for vectorisation of grid

    x = np.linspace(-1.5,1.5,100)
    X1, X2 = np.meshgrid(x,x)
    grid_configuration = np.stack([X1.ravel(), X2.ravel()] , axis= -1)# Shape: (10000, 2)
    active_particle_batch = np.expand_dims(grid_configuration, axis = 1 )# Shape: (10000, 1, 2)

    fixed_coords = np.array([
    [-1.0, 1.0], # Coordinates for Particle 1
    [-1.0, -1.0],   # Coordinates for Particle 2
    [1.0, -1.0],
    [1.0, 1.0]
    ])

    
    fixed_batch = np.broadcast_to(fixed_coords, (batch_size, N - 1, dim)) # Shape: (10000, 2, 2)
    X = np.concatenate([fixed_batch, active_particle_batch], axis=1)

    log_psi = vstate.log_value(X)
    log_psi = jnp.nan_to_num(log_psi, nan=0.0, posinf=100.0, neginf=-100.0) #
    

    log_mag = jnp.real(log_psi)
    phase = jnp.imag(log_psi)

    log_mag_shifted = log_mag - jnp.max(log_mag)
    
    print(phase)
    Z = (jnp.exp(log_mag_shifted) * jnp.sin(phase)).reshape(100, 100) #real of Psi
    
    # 6. Plotting
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X1, X2, Z, levels=50, cmap="RdBu_r")
    plt.scatter([-1, -1, 1, 1], [1, -1, -1, 1], color='white', marker='*', s=200, label='Fixed Particles')
    
    plt.colorbar(contour, label=r"Amplitude $\Psi(x_1, x_2)$")
    
    plt.title("Proabability amplitude of 1 active particle and 4 fixed")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("plots/antisymmetry_5particles_2d_phase.png", bbox_inches="tight")


if __name__ == "__main__":
    main(5,2)