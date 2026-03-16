## init NN 
import flax 
from flax import nnx 
import jax.numpy as jnp
import jax
from jax.nn.initializers import normal

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
    
class Gaussian(nnx.Module): 
     
    """
    We know that GS of QHO is a Gaussian, parametrised with covariance matrix 
    the sum of (x_i)^2 in exponent is just a dot product X^T * X, hence : 

    The wavefunction is given by the formula: :math:`\Psi(x) = \exp(\sum_{ij} x_i \Sigma_{ij} x_j)`.
    The (positive definite) :math:`\Sigma_{ij} = AA^T` matrix is stored as
    non-positive definite matrix A.
    """
    def __init__(self, dim: int, rngs: nnx.Rngs , N:int,  std: float = 1.0,  ): 

        self.N = N
        initializer = jax.nn.initializers.normal(std)

        inital_A = initializer( rngs.params() , (dim * N ,dim * N ), jnp.float64)

        self.A = nnx.Param(inital_A)

    def __call__(self, X : jax.Array): 

        A_matrix = self.A.value
        Sigma = jnp.dot(A_matrix.T , A_matrix)
        #super weired op, but basically it's optimised (X.T @ Sigma @ X)
        exponent = -0.5 * jnp.einsum("...i,ij,...j", X , Sigma, X)

        return exponent #nk expects log , don't exponentiate 