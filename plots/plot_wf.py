import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
import netket as nk
from flax import nnx

from src.system import System
from src.ansatz import FermiSets

def main():
    # 1. Re-initialize the exact architecture (N=1, dim=2)
    system = System(N=5, dim=2, mass=1.0, potential= "qho_no_inter")
    ansatz = FermiSets(dim=2, rngs=nnx.Rngs(42), N=5 , hidden_units= 16)
    
    sampler = nk.sampler.MetropolisGaussian(system.hi, 
                                            sigma=0.1,
                                            n_chains=16,
                                            sweep_size=32) 
    
    vstate = nk.vqs.MCState(sampler, ansatz, n_samples=10**4, n_discard_per_chain=100)
    
    # EVERY particle is sitting at (0,0) !!! 
    fixed_config = jnp.zeros((5, 2)) # TODO read from actual cfg in future N , dim 



    mpack_path = "outputs/2026-03-26/19-00-52/optimization_results.mpack"
    with open(mpack_path, "rb") as file:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())
        
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    grid_2d = np.stack([X.ravel(), Y.ravel()], axis=-1)#flattens 100x100 -> (10000, 2)
    full_configs = jnp.tile(fixed_config, (grid_2d.shape[0], 1, 1)) # (10000, 5, 2) tensor of 10000 particle configurations
    full_configs = full_configs.at[:, 0, :].set(grid_2d) # setting coordinates for particle 0
    
    log_psi = vstate.log_value(full_configs)
    psi = jnp.exp(log_psi)
    
    x_coords = grid_2d[:, 0] 
    operated_psi = x_coords * psi
    
    #  the real part of the amplitude, reshaped back to the 100x100 grid
    Z = jnp.real(operated_psi).reshape(100, 100)
    
    plt.figure(figsize=(8, 6))
    
    
    contour = plt.contourf(X, Y, Z, levels=50, cmap="coolwarm", center=0)
    plt.colorbar(contour, label=r"Amplitude of $\hat{x}\Psi(x,y)$")
    
    plt.title(r"Action of the Position Operator on the 2D Ground State")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    
    plt.savefig("plots/operator_action_19-00-52_2particles.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()