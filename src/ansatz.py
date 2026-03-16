## init NN 
import flax 
from flax import nnx 
import jax.numpy as jnp
import jax
from jax.nn.initializers import normal

class DeepSetsNN(nnx.Module): 
    """
    Simplest way to symmetrise the Ansatz. Implemented from paper: 

    Zaheer, Manzil, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov,
    and Alexander Smola. 2018. “Deep Sets.” doi:10.48550/arXiv.1703.06114.

    """
    def __init__(self , dim: int , N: int, rngs: nnx.Rngs, pool_fct_name : str = None, L: float = None , hidden_units: int = 8 ):

        self.pool_fct_name = pool_fct_name
        self.dim = dim 
        self.N = N 
        self.L = L
        self.hidden_units = hidden_units

        #pbc ignored for now 

        ###PHI 

        self.phi_dense1 = nnx.Linear(in_features= dim , out_features= hidden_units, rngs= rngs) #dim * 2 if PBC map x -> (sin(x), cos(x))
        self.phi_dense2 = nnx.Linear(in_features= hidden_units, out_features= hidden_units, rngs=rngs  )
        
        ### RHO

        self.rho_dense1 = nnx.Linear(in_features=hidden_units, out_features=hidden_units, rngs=rngs)
        self.rho_dense2 = nnx.Linear(in_features=hidden_units, out_features=1, rngs=rngs)

        # if self.pool_fct_name ==  None: 
        #         self.pool_fct = jnp.log( jnp.sum( jnp.exp))
        # else: 
        #     print("no other pool fct defined ")

    def __call__(self, x : jax.Array):

        x_reshaped = x.reshape(-1, self.N, self.dim) #-1 inferes the batch size automatically 

        y = self.phi_dense1(x_reshaped)
        y = nnx.gelu(y)
        y = self.phi_dense2(y)

        # pooling, enforcing symmetrisation 
        y = jnp.sum(y, axis=1)
        y = self.rho_dense1(y)
        y = nnx.gelu(y)
        y = self.rho_dense2(y)

        logNNoutput = y.squeeze() 

        #zeroing the tails of "gaussian" (for QHO like systems)
        logPsi = logNNoutput + (-0.5 * jnp.sum(x**2, axis=-1)) 

        return logPsi
    
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
        #super weird op, but basically it's optimised (X.T @ Sigma @ X)
        exponent = -0.5 * jnp.einsum("...i,ij,...j", X , Sigma, X)

        return exponent #nk expects log , don't exponentiate 