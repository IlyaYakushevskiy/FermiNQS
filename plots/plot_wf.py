import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
import netket as nk
from flax import nnx

# Adjust these imports based on your folder structure
from src.system import System
from src.ansatz import Gaussian

def main():
    # 1. Re-initialize the exact architecture (N=1, dim=2)
    system = System(N=1, dim=2, mass=1.0)
    ansatz = Gaussian(dim=2, rngs=nnx.Rngs(42), N=1)
    
    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=0.1)
    vstate = nk.vqs.MCState(sampler, ansatz, n_samples=100)
    
    # 2. Load the trained parameters (update this path to your actual run)
    mpack_path = "outputs/YOUR_RUN_FOLDER/optimization_results.mpack"
    with open(mpack_path, "rb") as file:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())
        
    # 3. Create a 2D Cartesian grid (Calculus 3 style)
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the grid into a list of coordinates
    # NetKet expects shape: (batch_size, N, dim) -> (10000, 1, 2)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1).reshape(-1, 1, 2)
    
    # 4. Evaluate the wave function
    log_psi = vstate.log_value(grid_points)
    
    # Convert from log-space back to linear amplitude
    psi = jnp.exp(log_psi)
    
    # 5. Apply the x operator
    # Extract just the x-coordinates from the grid. 
    # grid_points[:, 0, 0] gives all batches, the 1st particle, the 1st dimension (x)
    x_coords = grid_points[:, 0, 0]
    
    # Element-wise multiplication: x * Psi
    operated_psi = x_coords * psi
    
    # 6. Prepare data for plotting
    # We plot the real part of the amplitude, reshaped back to the 100x100 grid
    Z = jnp.real(operated_psi).reshape(100, 100)
    
    # 7. Generate the plot
    plt.figure(figsize=(8, 6))
    
    # A diverging colormap (coolwarm) is perfect here to show positive and negative amplitudes
    contour = plt.contourf(X, Y, Z, levels=50, cmap="coolwarm", center=0)
    plt.colorbar(contour, label=r"Amplitude of $\hat{x}\Psi(x,y)$")
    
    plt.title(r"Action of the Position Operator on the 2D Ground State")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    
    plt.savefig("operator_action.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()