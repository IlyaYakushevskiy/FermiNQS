## init NN 
import flax 
from flax import nnx 
import jax.numpy as jnp
import jax
from jax.nn.initializers import normal

class DeepSetsNN(nnx.Module): 
    """
    Simplest way to symmetrise the Ansatz. Implemented from the paper: 

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
        

class FermiSets(nnx.Module):
    """
     Implemented from the paper 
     Fu, Liang. 2025. “A Minimal and Universal Representation of Fermionic Wavefunctions 
     (Fermions = Bosons + One).” doi:10.48550/arXiv.2510.11431.
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

    def nu_antisymmetric(self, x): 
            x_reshaped = x.reshape(-1, self.N, self.dim)
            #x is (batch, N, dim)
            if self.dim == 1: 

                #N = x.shape[-1]
                #using broadcasting, x[..., :, None] has shape (batch, N, 1) and x[..., None, :] (batch, 1, N), so diff_matrix has shape (batch, N, N)
                #diff_matrix = x[..., :, None] - x[..., None, :]
                #return jnp.sign(jnp.prod(jnp.diff(x, axis=1), axis=-1)) # sign of product of differences

                #TODO the very naive approach, to be vectorised 
        
                batch_size = x_reshaped.shape[0]
                y = jnp.ones((batch_size, 1))

                for i in range(self.N):
                    r_i = x_reshaped[:, i, :]
                    for j in range(i): 
                        
                        r_j = x_reshaped[:, j, :]

                        y = y * ( r_i - r_j )

                y = y.squeeze()


                return jnp.log(y.astype(jnp.complex64)) ## casted to complex, bc log in Re{} is undef

            else:
                raise NotImplementedError
    

    def __call__(self, x : jax.Array):

        #x is (batch, N_particles, dim)
        x_reshaped = x.reshape(-1, self.N, self.dim) #-1 inferes the batch size automatically 

        y = self.phi_dense1(x_reshaped)
        y = nnx.gelu(y)
        y = self.phi_dense2(y)

        y = jnp.sum(y, axis=1) # pooling "layer", enforcing symmetrisation 

        y = self.rho_dense1(y)
        y = nnx.gelu(y)
        y = self.rho_dense2(y)

        logNNoutput = y.squeeze() 
        #zeroing the tails of "gaussian" (for QHO like systems)
        log_psi_boson = logNNoutput + (-0.5 * jnp.sum(x**2, axis=(-2,-1))) 
        log_antisymmetric = self.nu_antisymmetric(x)

        logPsi = log_psi_boson + log_antisymmetric

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