## init NN 
import flax 
from flax import nnx 
import jax.numpy as jnp
import jax


class DeepSetsNN(nnx.Module): 
    """
    Simplest way to symmetrise the Ansatz. Pooling function must be defined. 
    
    """
    def __init__(self , pool_fct_name : str = None ):
        self.pool_fct_name = pool_fct_name


        if self.pool_fct_name ==  None: 
                self.pool_fct = jnp.log( jnp.sum( jnp.exp))
        else: 
            print("no other pool fct defined ")

    
        

    def __call__(self, x : jax.Array):
        
        
        return x
    
