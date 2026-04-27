import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
import netket as nk
from flax import nnx

from src.system import System
from src.ansatz import FermiSets

import os
import re
import matplotlib.image as mpimg
import matplotlib.animation as animation

def nu_antisymmetric(x, dim, N): 
    if dim == 2: 
        x_reshaped = x.reshape(-1, N, dim)

        z = x_reshaped[:, :, 0] + 1j * x_reshaped[:, :, 1]
        idx_i, idx_j = jnp.tril_indices(z.shape[1], k=-1)

        diff = z[:, idx_i] - z[:, idx_j]

        diff_sq = jnp.square(jnp.real(diff)) + jnp.square(jnp.imag(diff))
        a = 1.0 
        r_test = diff / jnp.sqrt(diff_sq + a**2)
        y = jnp.prod(r_test, axis=1)

        return y

    else:
        raise NotImplementedError 

def plot_wf ( plot_name : str, plot_path : str, plot_title : str, vstate : nk.vqs.MCState, system : System ): 

    
    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.5, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    grid_2d = np.stack([X.ravel(), Y.ravel()], axis=-1) # Shape: (10000, 2)
    batch_size = grid_2d.shape[0]
    
    fixed_coords = jnp.array([
        [0.0, 0.0],   # Particle 0 (Active, to be overwritten)
        [-1.0, 1.0],  # Particle 1 (Fixed)
        [-1.0, -1.0], # Particle 2 (Fixed)
        [1.0, -1.0],  # Particle 3 (Fixed)
        [1.0, 1.0]    # Particle 4 (Fixed)
    ])

    fixed_coords_flipped = jnp.array([
        [0.0, 0.0],   # Particle 0 (Active, to be overwritten)
        [-1.0, 1.0],  # Particle 1 (Fixed)
        [-1.0, -1.0], # Particle 2 (Fixed)
        [1.0, 1.0],  # Particle 3 (Fixed)
        [1.0, -1.0]    # Particle 4 (Fixed)
    ])


    full_configs = jnp.tile(fixed_coords, (batch_size, 1, 1)) # Shape: (10000, 5, 2)
    full_configs = full_configs.at[:, 0, :].set(grid_2d) 
    full_configs = jnp.reshape(full_configs, [-1,10])

    full_configs_flipped = jnp.tile(fixed_coords_flipped, (batch_size, 1, 1)) # Shape: (10000, 5, 2)
    full_configs_flipped = full_configs_flipped.at[:, 0, :].set(grid_2d) 
    full_configs_flipped = jnp.reshape(full_configs_flipped, [-1,10])

    log_psi = vstate.log_value(full_configs) # log is in form log(r)+iθ 

    log_mag = jnp.real(log_psi)
    phase = jnp.imag(log_psi)

    max_val = jnp.nanmax(log_mag)
    log_mag_shifted = log_mag - max_val #is normalisation , we force max value of exponent be 0

    real_psi = (jnp.exp(log_mag_shifted) * jnp.cos(phase)).reshape(100, 100)
    img_psi = (jnp.exp(log_mag_shifted) * jnp.sin(phase)).reshape(100, 100)
    prob_density = jnp.exp(2.0 * log_mag_shifted).reshape(100, 100)
    nu_outputs = nu_antisymmetric( x = full_configs , dim = system.dim, N = system.N)

    real_nu =  jnp.real(nu_outputs)
    img_nu = jnp.imag(nu_outputs)
    real_nu = jnp.real(nu_outputs).reshape(100, 100)
    img_nu = jnp.imag(nu_outputs).reshape(100, 100)

    ##quick antisymmetry check, printing to output log 
    nu_outputs_flipped = nu_antisymmetric( x = full_configs_flipped , dim = system.dim, N = system.N)

    print( f"For step {plot_name} Norm of nu(x) + nu(-x): " , jnp.linalg.norm( nu_outputs + nu_outputs_flipped)) 

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(plot_title, fontsize=10, y=1.02)
    c0 = axes[0].contourf(X, Y, real_psi, levels=50, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(c0, ax=axes[0])
    axes[0].set_title(r"Real Part: $\Re(\Psi)$")

    c1 = axes[1].contourf(X, Y, prob_density, levels=50, cmap="magma")
    fig.colorbar(c1, ax=axes[1])
    axes[1].set_title(r"Probability Density: $|\Psi|^2$")

    c2 = axes[2].contourf(X, Y, real_nu, levels=50, cmap="RdBu_r")
    fig.colorbar(c2, ax=axes[2])
    axes[2].set_title(r"Antisymmetric Part: $\Re(\eta)$")

    c3 = axes[3].contourf(X, Y, img_nu, levels=50, cmap="RdBu_r")
    fig.colorbar(c3, ax=axes[3])
    axes[3].set_title(r"Antisymmetric Part: $\Im(\eta)$")

    fixed_plot_coords = fixed_coords[1:] 
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
        ax.scatter(fixed_plot_coords[:, 0], fixed_plot_coords[:, 1], 
                   color='white', marker='*', s=150, edgecolor='black', label='Fixed Particles')
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig( f"{plot_path}/{plot_name}", bbox_inches="tight")
    print(f"ploted {plot_name} and saved")

def animate_training_plots(plot_dir: str, output_path: str, fps: int = 5):
    """
    Reads a directory of training plot PNGs and stitches them into a movie.
    """
    # 1. Grab all PNG files
    valid_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
    
    if not valid_files:
        print(f"Error: No PNG files found in {plot_dir}")
        return

    # 2. Sort numerically by step number (CRITICAL)
    # This looks for "step_10", "step_100", etc., and sorts by the integer
    def extract_step(filename):
        match = re.search(r'step_(\d+)', filename)
        return int(match.group(1)) if match else -1

    valid_files.sort(key=extract_step)
    
    print(f"Found {len(valid_files)} frames. First frame: {valid_files[0]}, Last: {valid_files[-1]}")

    # 3. Setup the Matplotlib Figure
    # We make it large to maintain the resolution of your 1x4 subplot grid
    fig, ax = plt.subplots(figsize=(18, 5)) 
    
    # We turn off the axis lines because your PNGs already contain their own axes and labels!
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.axis('off') 

    frames = []
    print("Loading images into memory...")
    for filename in valid_files:
        img_path = os.path.join(plot_dir, filename)
        img = mpimg.imread(img_path)
        
        # imshow returns an AxesImage object. ArtistAnimation requires a list of these.
        im = ax.imshow(img, animated=True)
        frames.append([im])

    print("Generating animation...")
    # interval is in milliseconds. 1000/fps gives the correct delay between frames.
    ani = animation.ArtistAnimation(fig, frames, interval=1000/fps, blit=True, repeat_delay=2000)

    try:
        # Saving as MP4 requires ffmpeg to be installed on your system
        ani.save(output_path, writer='ffmpeg', fps=fps)
        print(f"Success! Saved animation to {output_path}")
    except Exception as e:
        print(f"\nFailed to save as MP4. Do you have 'ffmpeg' installed on your system?")
        print(f"Error details: {e}")
        
        # Fallback to GIF which requires no external dependencies (uses Pillow)
        gif_path = output_path.replace('.mp4', '.gif')
        print(f"\nFalling back to GIF format...")
        ani.save(gif_path, writer='pillow', fps=fps)
        print(f"Success! Saved animation to {gif_path}")

    plt.close(fig)


def main():
    N = 5
    dim = 2
    
    # 1. Initialize System and Ansatz
    system = System(N=N, dim=dim, mass=1.0, potential="qho_no_inter")
    #ansatz = FermiSets(dim=dim, rngs=nnx.Rngs(43), N=N, hidden_units=16, out_units= 20, log = None)

    ansatz = FermiSets(
            dim= dim,
            rngs= nnx.Rngs(43),
            N = N, 
            hidden_units= 16,
            out_units = 20,
            log= None
        )
    
    sampler = nk.sampler.MetropolisGaussian(system.hi, 
                                            sigma=0.1,
                                            n_chains=32,
                                            sweep_size=128) 
    
    vstate = nk.vqs.MCState(sampler, ansatz, n_samples=10**3, n_discard_per_chain=100)

    # 2. Load trained parameters
    
    mpack_path = "/home/ilya/FermiNQS/outputs/2026-04-23/18-02-04/optimization_results.mpack"
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
        [3.2, 1.0]    # Particle 4 (Fixed)
    ])

    # Tile the configuration for the whole batch
    full_configs = jnp.tile(fixed_coords, (batch_size, 1, 1)) # Shape: (10000, 5, 2)
    
    
    # Overwrite the coordinates of Particle 0 with the grid points
    full_configs = full_configs.at[:, 0, :].set(grid_2d) 

    full_configs = jnp.reshape(full_configs, [-1,10])

    # 5. Evaluate the wave function

    log_psi = vstate.log_value(full_configs)

    # Separate magnitude and phase
    log_mag = jnp.real(log_psi)
    phase = jnp.imag(log_psi)

    # Use nanmax so the 4 singularities don't poison the maximum value
    max_val = jnp.nanmax(log_mag)
    log_mag_shifted = log_mag - max_val #is normalisation , we force max value of exponent be 0

    # 6. Calculate the components
    real_psi = (jnp.exp(log_mag_shifted) * jnp.cos(phase)).reshape(100, 100)
    img_psi = (jnp.exp(log_mag_shifted) * jnp.sin(phase)).reshape(100, 100)
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

    plt.savefig("plots/the_img_part.png", bbox_inches="tight")
    print("ploted and saved")
    plt.show()

if __name__ == "__main__":
    #main()
    plot_directory = "/home/ilya/FermiNQS/outputs/2026-04-27/21-59-37/plots"
    output_movie = "/home/ilya/FermiNQS/outputs/2026-04-27/21-59-37/training_evolution.mp4"
    
    animate_training_plots(plot_dir=plot_directory, output_path=output_movie, fps=10)
