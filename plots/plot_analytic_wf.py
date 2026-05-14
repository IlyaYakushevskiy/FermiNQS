import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import flax
import netket as nk
from flax import nnx

from src.system import System
from src.ansatz import FermiSets
import jax.scipy.special as jsp

import os
import re

def qho_1d_wf(n, x):
    """Single particle QHO 1D wavefunction."""
    # Normalization: (2^n * n! * sqrt(pi))^(-1/2)
    norms = {
        0: jnp.power(jnp.pi, -0.25),
        1: jnp.power(4 * jnp.pi, -0.25),
        2: 1.0 / (jnp.sqrt(8) * jnp.power(jnp.pi, 0.25))
    }
    # Hermite polynomials: H0=1, H1=2x, H2=4x^2-2
    hermite = {
        0: 1.0,
        1: 2.0 * x,
        2: 4.0 * jnp.square(x) - 2.0
    }
    return norms[n] * hermite[n] * jnp.exp(-0.5 * jnp.square(x))

def get_slater_matrices(x_batch, N):
    """
    Constructs the 3 Slater matrices for N=4, dim=2.
    x_batch: (batch, 4, 2)
    """
    # Define the orbital indices for the 3 degenerate states
    # Shell 0: (0,0) | Shell 1: (1,0), (0,1) | Shell 2: (2,0), (1,1), (0,2)
    base_orbitals = [(0,0), (1,0), (0,1)]
    top_choices = [(2,0), (1,1), (0,2)]
    
    def evaluate_orbital(nx, ny, r):
        # r is (batch, 2) -> returns (batch,)
        return qho_1d_wf(nx, r[:, 0]) * qho_1d_wf(ny, r[:, 1])

    matrices = []
    for top in top_choices:
        orbitals = base_orbitals + [top]
        # Build matrix M_ij = phi_i(r_j)
        # Shape: (batch, N, N)
        M = jnp.stack([
            jnp.stack([evaluate_orbital(nx, ny, x_batch[:, j, :]) for j in range(N)], axis=-1)
            for (nx, ny) in orbitals
        ], axis=1)
        matrices.append(M)
    
    return matrices # List of 3 matrices, each (batch, 4, 4)

### this code is for 2D case
def slater_determinant( phis : jnp.array, rs : jnp.array ):  
    """input are array of fct functions and coordinates
    phi: array of lambda functions """ 

    A = jnp.array([phis.shape, phis.shape])
    for r_i in range(len(rs)): 
        A[r_i] = phis(rs[r_i]) # rhs is an array 

    slater = jnp.linalg.det(A)
    return slater


def single_part_wf( r ): 
    #we looking for gs, but we have to figure lowest n for given N particles 
    #r is 2d array
    phi_x = 1 / jnp.norm() * pi**(-0.25 )  * jnp.polynomial.hermite.Hermite
    return phi_x * phi_y 

def total_wf(): 
    #TODO 
    return 0
    

def compute_overlap_squared(log_psi_nn, x_batch, N):
    """
    Computes total overlap squared with the ground state manifold.
    """
    matrices = get_slater_matrices(x_batch, N)
    total_s2 = 0.0
    
    for M in matrices:
        # log_psi_exact = log(det(M))
        sign, log_psi_exact = jnp.linalg.slogdet(M)
        
        # ratio = exp(log_psi_exact - log_psi_nn) * sign
        # We use the sign because the determinant can be negative
        ratio = sign * jnp.exp(log_psi_exact - log_psi_nn)
        
        s2 = jnp.square(jnp.abs(jnp.mean(ratio))) / jnp.mean(jnp.square(jnp.abs(ratio)))
        total_s2 += s2
        
    return total_s2


def plot_wf( N, vstate : ): 

    n_dots = N -1 
    x = np.linspace(-3.0, 3.0, 100)
    y = np.linspace(-3.5, 3.0, 100)
    X, Y = np.meshgrid(x, y)
    grid_2d = np.stack([X.ravel(), Y.ravel()], axis=-1) # Shape: (10000, 2)
    batch_size = grid_2d.shape[0]
    
    R = 1.0

    angles = jnp.array([jnp.pi / 2 + 2 * jnp.pi * k / n_dots  for k in range(n_dots)])
    ngon_coords = jnp.stack([R * jnp.cos(angles), R * jnp.sin(angles)], axis=-1)

    fixed_coords = jnp.vstack([
        jnp.array([[0.0, 0.0]]),  
        ngon_coords           # fixed particles 
    ])

    full_configs = jnp.tile(fixed_coords, (batch_size, 1, 1)) # Shape: (10000, 5, 2)
    full_configs = full_configs.at[:, 0, :].set(grid_2d) 
    full_configs = jnp.reshape(full_configs, [-1,system.N * system.dim])

    
    psi = total_wf #pehaps one needs to return log 
    eigen_energy = ... #TODO for overlap 


if __name__ == "main": 
    N = 4
    model_path = "/home/ilya/FermiNQS/outputs/2026-05-12/19-42-14/checkpoints/step_650.mpack"