## here i describe physics of the system and the Hamiltonian
import netket as nk 
import jax 
import jax.numpy as jnp

class System(): 

    def __init__(self, N : int, dim : int , mass, potential : str , **kwargs):

        self.N = N 
        self.dim = dim
        self.mass = mass 
        self.potential = potential


        geometry = nk.experimental.geometry.FreeSpace(d =dim ) ##PBC are not implemented 
        self.hi = nk.experimental.hilbert.Particle(N=N, geometry=geometry) 

         
        #redundant
        #self.states = self.hi.random_state(jax.random.key(0), 1) # continious hilbert 

        self.Ekin = nk.operator.KineticEnergy(self.hi, mass = 1.0) #this part stays const

        if self.potential == "qho_no_inter":
            def v(x):
                return 0.5 * jnp.sum(x**2, axis=-1) # potential is 1/2 * m w^2 * x^2 -> hbar * w = 1 , GS is 1/2 * hbar w * dim * particles
        elif self.potential == "qho_aniso":
            # anisotropic trap v = (x^2 + omega_y^2 y^2)/2: breaks rotational symmetry
            # ([H, L_z] != 0 — lz_proj_K must be 0), spectrum still separable:
            # E(nx, ny) = (nx + 1/2) + omega_y (ny + 1/2), see exact_trap_gs_energy in main.py
            if dim != 2:
                raise ValueError("qho_aniso is 2D only")
            self.omega_y = float(kwargs.get("omega_y") or 1.5)
            # weights follow the flattened layout (x1, y1, x2, y2, ...)
            w = jnp.tile(jnp.array([1.0, self.omega_y**2]), N)
            def v(x):
                return 0.5 * jnp.sum(w * x**2, axis=-1)
        elif self.potential == "dot_gauss":
            # interacting dot: harmonic trap + Gaussian repulsion lam * exp(-r_ij^2/(2 s^2)).
            # Gaussian chosen over bare Coulomb deliberately: exact ED reference
            # (tools/ed_dot.py), no coalescence cusp, identical operator in ED and VMC.
            lam = float(kwargs.get("int_strength") or 2.0)
            s_int = float(kwargs.get("int_range") or 1.0)
            self.int_strength, self.int_range = lam, s_int
            idx_i, idx_j = jnp.tril_indices(N, k=-1)
            def v(x):
                xr = x.reshape(x.shape[:-1] + (N, dim))
                trap = 0.5 * jnp.sum(xr**2, axis=(-2, -1))
                d = xr[..., idx_i, :] - xr[..., idx_j, :]
                r2 = jnp.sum(d**2, axis=-1)
                return trap + lam * jnp.sum(jnp.exp(-r2 / (2.0 * s_int**2)), axis=-1)
        else:
            raise ValueError(f"unknown potential '{potential}'")


        self.Epot = nk.operator.PotentialEnergy(self.hi, v)

        self.H =  self.Ekin + self.Epot
        
 