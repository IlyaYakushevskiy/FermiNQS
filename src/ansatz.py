## init NN 
import flax 
from flax import nnx 
import jax.numpy as jnp
import jax
from jax.nn.initializers import normal
import timeit
import logging 

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
     
    def __init__(self , dim: int , N: int, rngs: nnx.Rngs, log : logging.Logger,  pool_fct_name : str = None, L: float = None , hidden_units: int = 8 ):

        self.pool_fct_name = pool_fct_name
        self.dim = dim 
        self.N = N 
        self.L = L
        self.hidden_units = hidden_units
        self.log = log

        #pbc ignored for now 

        ###PHI 

        self.phi_dense1 = nnx.Linear(in_features= dim , out_features= hidden_units, dtype=jnp.complex128, param_dtype=jnp.complex128, rngs=rngs) #dim * 2 if PBC map x -> (sin(x), cos(x))
        self.phi_dense2 = nnx.Linear(in_features= hidden_units, out_features= hidden_units, dtype=jnp.complex128, param_dtype=jnp.complex128, rngs=rngs  )
        
        ### RHO

        self.rho_dense1 = nnx.Linear(in_features=hidden_units, out_features=hidden_units, dtype=jnp.complex128, param_dtype=jnp.complex128, rngs=rngs)
        self.rho_dense2 = nnx.Linear(in_features=hidden_units, out_features=1, dtype=jnp.complex128, param_dtype=jnp.complex128, rngs=rngs)

        ### Psi layer, combining symmetric and antisymmetric features
        self.Psi_dense1 = nnx.Linear(in_features=hidden_units+ 3 , out_features=hidden_units+3, dtype=jnp.complex128, param_dtype=jnp.complex128, rngs=rngs) # +3 for Re{} and Im{} of the Log(nu)
        self.Psi_dense2 = nnx.Linear(in_features=hidden_units+ 3 , out_features=1, dtype=jnp.complex128, param_dtype=jnp.complex128, rngs=rngs)


    def nu_antisymmetric(self, x): 
            #print("starting est poly")
            x_reshaped = x.reshape(-1, self.N, self.dim)
            #rows, cols = jnp.tril_indices(self.N, k=-1)
            
            #x is (batch, N, dim)
            if self.dim == 1:                 
                # r = x_reshaped[..., 0]
                # #all pairwise differences with broadcasting 
                # diff = r[:, :, None] - r[:, None, :] # Shape: (batch, N, N) 

                # valid_diffs = diff[:, rows, cols] #lower triangular
            
                # log_diffs = jnp.log(valid_diffs.astype(complex))

                
                # return jnp.sum(log_diffs, axis=1)

                #non-vectorised code
                batch_size = x_reshaped.shape[0]
                y = jnp.zeros((batch_size, 1))

                for i in range(self.N):
                    r_i = x_reshaped[:, i, :]
                    for j in range(i): 
                        
                        r_j = x_reshaped[:, j, :]

                        #log this part instead ,then we're talking sums 
                        diff = ( r_i - r_j) 
                        log_diff = jnp.log(diff.astype(jnp.complex128))
                        y = y + log_diff
                #print("finished est poly ")
                return y.reshape(-1)
         
            elif self.dim == 2: 

                batch_size = x_reshaped.shape[0]
                
                y = jnp.zeros((batch_size, ), dtype= jnp.complex128) # must be 1D ! 

                for i in range(self.N): 

                    r_i = x_reshaped[:, i , : ]

                    z_i = r_i[ : , 0] + 1j * r_i[ : , 1] 

                    for j in range(i): 
                        r_j = x_reshaped[:, j, :]
                        z_j = r_j[:, 0] + 1j * r_j[:, 1]

                        diff = (z_i - z_j)

                        y = y + jnp.log(diff)
                return y 

            else:
                raise NotImplementedError
    
    def eval_psi0(self, x, nu): #nu is passed not as functor but as value 
        #x is (batch, N_particles, dim)
        x_reshaped = x.reshape(-1, self.N, self.dim) #-1 inferes the batch size automatically 


        y = self.phi_dense1(x_reshaped)
        y = nnx.tanh(y)
        y = self.phi_dense2(y)

        y = jnp.sum(y, axis=1) # s_j = Sigma_i->N ( phi_j (x_i))  Not to confuse , this is still j sums of N 

        y = self.rho_dense1(y)
        y = nnx.tanh(y)

        log_nu_real = jnp.real(nu)
        log_nu_imag = jnp.imag(nu)
        safe_real = jnp.clip(log_nu_real, a_min=-15.0, a_max=15.0)[:, None]
        phase_cos = jnp.cos(log_nu_imag)[:, None]
        phase_sin = jnp.sin(log_nu_imag)[:, None]

        log_feat_concat = jnp.concatenate([y, safe_real, phase_cos, phase_sin], axis=-1)

        logPsi = self.Psi_dense1(log_feat_concat)
        logPsi = jnp.tanh(logPsi)
        logPsi = self.Psi_dense2(logPsi) #TODO try complex 

        logPsi = logPsi.squeeze() 

        return logPsi

    def __call__(self, x : jax.Array):

        nu = self.nu_antisymmetric(x)

        # feeding dummy tensor to check antisymmetry
        # idx = jnp.arange(x.shape[-2]) 
        # idx = idx.at[0].set(1).at[1].set(0) # Swapping particle 0 and particle 1
        # x_swapped = x[..., idx, :]

        # nu_minus = self.nu_antisymmetric(x_swapped)

        # self.log.info("nu(x) - Pnu(x) = " , jnp.abs(nu + nu_minus)) ##TODO make printing with call back function, jax does not allow log.info
  
        log_psi0_plus = self.eval_psi0(x, nu)
        log_psi0_minus = self.eval_psi0(x, nu + 1j * jnp.pi) # nu + 1j * jnp.pi is a swap ( nu -> -nu) in complex space 

        stacked_logs = jnp.stack([log_psi0_plus, log_psi0_minus], axis=-1)
        weights = jnp.array([0.5, -0.5])
        log_psi_final = jax.nn.logsumexp(stacked_logs, axis=-1, b=weights) ##TODO this fct only accepts real, change it to complex 

        #jax.debug.print("log_psi_boson = {} and log_antisymmetric = {}", log_psi_boson,log_antisymmetric)
        return log_psi_final
    

    
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
    

class GaussianFermions(nnx.Module): 
     
    """
    We know that GS of QHO is a Gaussian, parametrised with covariance matrix 
    the sum of (x_i)^2 in exponent is just a dot product X^T * X, hence : 

    The wavefunction is given by the formula: :math:`\Psi(x) = \exp(\sum_{ij} x_i \Sigma_{ij} x_j)`.
    The (positive definite) :math:`\Sigma_{ij} = AA^T` matrix is stored as
    non-positive definite matrix A.
    """
    def __init__(self, dim: int, rngs: nnx.Rngs , N:int,  std: float = 1.0,  ): 

        self.N = N
        self.dim = dim
        initializer = jax.nn.initializers.normal(std)

        inital_A = initializer( rngs.params() , (dim * N ,dim * N ), jnp.float64)

        self.A = nnx.Param(inital_A)
    
    def nu_antisymmetric(self, x): 
            x_reshaped = x.reshape(-1, self.N, self.dim)
            #x is (batch, N, dim)
            if self.dim == 1:              
                batch_size = x_reshaped.shape[0]
                y = jnp.zeros((batch_size, 1))

                for i in range(self.N):
                    r_i = x_reshaped[:, i, :]
                    for j in range(i): 
                        r_j = x_reshaped[:, j, :]
                        #log this part instead ,then we're talking sums 
                        diff = ( r_i - r_j) 
                        log_diff = jnp.log(diff.astype(jnp.complex128))
                        y = y + log_diff

                y = y.squeeze()
                return y

            elif self.dim == 2: 
                return 0
            else:
                raise NotImplementedError

    def __call__(self, X : jax.Array): 

        A_matrix = self.A.value
        Sigma = jnp.dot(A_matrix.T , A_matrix)
        #super weird op, but basically it's optimised (X.T @ Sigma @ X)
        exponent = -0.5 * jnp.einsum("...i,ij,...j", X , Sigma, X)

        if self.dim ==  1:
            X_reshaped = X.reshape(-1, self.N, 1)
            log_nu = self.nu_antisymmetric(X_reshaped)
        return exponent + log_nu #nk expects log , don't exponentiate 