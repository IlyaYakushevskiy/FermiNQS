## here i describe physics of the system and the Hamiltonian
import netket as nk 
import jax 
import jax.numpy as jnp

class System(): 

    def __init__(self, N : int, dim : int , mass , **kwargs):

        self.N = N 
        self.dim = dim
        self.mass = mass 


        geometry = nk.experimental.geometry.FreeSpace(d =dim ) ##PBC are not implemented 
        self.hi = nk.experimental.hilbert.Particle(N=N, geometry=geometry) 
        #redundant
        #self.states = self.hi.random_state(jax.random.key(0), 1) # continious hilbert 

        self.Ekin = nk.operator.KineticEnergy(self.hi, mass = 1.0) #this part stays const, thus implemented at init

        def v(x): 
            return 0.5 * jnp.sum(x**2, axis=-1) # potential is 1/2 * hbar w * x^2 -> hbar w = 1 
       
        self.Epot = nk.operator.PotentialEnergy(self.hi, v)

        self.H =  self.Ekin + self.Epot
        
 