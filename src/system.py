## here i describe physics of the system and Hamiltonian
import netket as nk 
import jax 

class System(): 

    def __init__(self, N : int, **kwargs):

        self.N = N 




        geo = nk.experimental.geometry.FreeSpace(d =1 ) ##PBC are not implemented 
        self.hi = nk.experimental.hilbert.Particle(N, geo) 
        self.states = self.hi.random_state(jax.random.key(0), 1) #for cont
        