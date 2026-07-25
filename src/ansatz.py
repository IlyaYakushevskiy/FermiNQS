## init NN 
import flax 
from flax import nnx 
import jax.numpy as jnp
import jax
from jax.nn.initializers import normal
import timeit
import logging 


class FermiSets(nnx.Module):
    """
     Implemented from the paper 
     Fu, Liang. 2025. “A Minimal and Universal Representation of Fermionic Wavefunctions 
     (Fermions = Bosons + One).” doi:10.48550/arXiv.2510.11431.
    """
     
    def __init__(self , dim: int , N: int, rngs: nnx.Rngs, log : logging.Logger,  pool_fct_name : str = None, L: float = None , hidden_units: int = 8, out_units: int = 10, lz_proj_K: int = 0, pair_hidden: int = 0, backflow_hidden: int = 0 ):

        self.dim = dim
        self.N = N
        self.L = L
        self.hidden_units = hidden_units
        self.log = log
        self.out_units = out_units
        # Signature backflow (RESEARCH_LOG 2026-07-24): the plain ansatz outsources ALL
        # antisymmetry to the FIXED Vandermonde eta = prod(z_i - z_j), whose nodal surface
        # is holomorphic and immovable — the representability wall at N=6 (07-23). Backflow
        # replaces the raw coords in eta with z_tilde_i = z_i + Delta_i, where Delta_i is a
        # permutation-EQUIVARIANT DeepSets map (per-particle feature + symmetric pool). Two
        # consequences: (i) antisymmetry survives EXACTLY (equivariance -> eta(z_tilde) still
        # sign-flips under swaps; structural, no regularizer), (ii) Delta may depend on the
        # conjugates z-bar (real input features), so eta(z_tilde) carries anti-holomorphic
        # nodal content the fixed Vandermonde forbids. Output layer zero-init: z_tilde = z at
        # init, so eta reduces to the baseline Vandermonde and gradients grow the deformation.
        # 0 disables (old arch). Acts on the SIGNATURE ONLY; the symmetric xi embedding still
        # sees raw coords, isolating "does a deformable node help" from everything else.
        self.backflow_hidden = backflow_hidden
        # Pair-feature stream (RESEARCH_LOG 2026-07-15): Deep Sets over unordered pairs,
        # exchange-even bounded features, sum-pooled into the symmetric embedding. Gives the
        # Psi head linear access to sum_pairs [log(1+r^2) - log(r^2+eps)] = -log T (T = |eta|^2),
        # the singular prefactor needed for triangle-area sign structures. 0 disables (old arch).
        self.pair_hidden = pair_hidden
        # L_z-sector projection: average over lz_proj_K rotations, keeping only
        # L_z = 0 (mod K) components. Kills the holomorphic trap family
        # (eta * holomorphic-symmetric has L_z = N(N-1)/2 + d > 0) exactly.
        # 0 or 1 disables. 2D only.
        self.lz_proj_K = lz_proj_K
        if lz_proj_K and lz_proj_K > 1 and dim != 2:
            raise ValueError("lz_proj_K is only implemented for dim=2")

        #pbc ignored for now 

        ###PHI 

        self.phi_dense1 = nnx.Linear(in_features= dim , out_features= hidden_units, rngs= rngs) #dim * 2 if PBC map x -> (sin(x), cos(x))
        self.phi_dense2 = nnx.Linear(in_features= hidden_units, out_features= hidden_units, rngs=rngs  )

        
        
        ### RHO

        self.rho_dense1 = nnx.Linear(in_features=hidden_units, out_features=hidden_units, rngs=rngs)

        ### PAIR stream (optional)
        if pair_hidden:
            n_pair_feats = 5 if dim == 2 else 3
            # live (default) init: the stream must participate in basin selection from step 0
            # (zero-init gating parked the run in the 6.0 trap before the stream woke up).
            # Stability comes from the BOUNDED features in pair_features, not from gating —
            # unbounded features with this init blew up VMC at step 0 (2026-07-15 shakedown).
            self.pair_dense1 = nnx.Linear(in_features=n_pair_feats, out_features=pair_hidden, rngs=rngs)
            self.pair_dense2 = nnx.Linear(in_features=pair_hidden, out_features=pair_hidden, rngs=rngs)

        ### SIGNATURE BACKFLOW stream (optional): equivariant coordinate deformation for eta
        if backflow_hidden:
            self.bf_dense1 = nnx.Linear(in_features=dim, out_features=backflow_hidden, rngs=rngs)
            # input is [per-particle feat, symmetric pool] -> 2*backflow_hidden
            self.bf_dense2 = nnx.Linear(in_features=2 * backflow_hidden, out_features=backflow_hidden, rngs=rngs)
            # zero-init output => z_tilde = z at init (eta == baseline Vandermonde), grads still flow
            self.bf_out = nnx.Linear(
                in_features=backflow_hidden, out_features=dim, rngs=rngs,
                kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros,
            )

        ### Psi layer, combining symmetric and antisymmetric features
        psi_in = hidden_units + pair_hidden + 2  # +2 for Re{} and Im{} of eta
        self.Psi_dense1 = nnx.Linear(in_features= psi_in , out_features=psi_in*2, rngs=rngs)
        #self.Psi_dense2 = nnx.Linear(in_features=(hidden_units+ 2)*2 , out_features=(hidden_units+ 2)*2, rngs=rngs)
        #extra layer when not using SR
        #self.Psi_dense_extra = nnx.Linear(in_features=(hidden_units+2)*2 , out_features=(hidden_units+2)*2, rngs=rngs)
        self.Psi_dense2 = nnx.Linear(in_features=psi_in*2 , out_features=out_units, rngs=rngs)


    def _backflow_coords(self, x_reshaped):
        # equivariant DeepSets backflow: Delta_i = MLP(h_i, mean_j h_j), h_i = gelu(W x_i).
        # Per-particle h_i + symmetric pool -> Delta permutes with the particles, so the
        # coincidence set of z_tilde stays permutation-symmetric and eta(z_tilde) stays
        # exactly antisymmetric. Real-valued features make Delta depend on z-bar.
        h = nnx.gelu(self.bf_dense1(x_reshaped))                 # (batch, N, H)
        g = jnp.mean(h, axis=1, keepdims=True)                   # (batch, 1, H) invariant pool
        g = jnp.broadcast_to(g, h.shape)
        d = nnx.gelu(self.bf_dense2(jnp.concatenate([h, g], axis=-1)))
        d = self.bf_out(d)                                       # (batch, N, dim), 0 at init
        return x_reshaped + d

    def eta_antisymmetric(self, x):
            x_reshaped = x.reshape(-1, self.N, self.dim)
            #x is (batch, N, dim)
            if self.backflow_hidden:
                x_reshaped = self._backflow_coords(x_reshaped)
            if self.dim == 1:
                # regularized real Vandermonde, same construction as dim==2 below.
                # NOTE: previously this returned log(prod diff) and __call__ negated the
                # log — which is the reciprocal, not a sign flip — structurally breaking
                # 1D antisymmetry (see RESEARCH_LOG.md 2026-07-14). Raw bounded product
                # makes the -eta flip exact.
                x1 = x_reshaped[:, :, 0]
                idx_i, idx_j = jnp.tril_indices(self.N, k=-1)
                diff = x1[:, idx_i] - x1[:, idx_j]

                a = 1.0
                r_test = diff / jnp.sqrt(diff**2 + a**2)
                y = jnp.prod(r_test, axis=1)
                return y

            elif self.dim == 2:

                batch_size = x_reshaped.shape[0]

                #trying Attila's regularisation fct
                z = x_reshaped[:, :, 0] + 1j * x_reshaped[:, :, 1]
                idx_i, idx_j = jnp.tril_indices(z.shape[1], k=-1)
                diff = z[:, idx_i] - z[:, idx_j]
                diff_sq = jnp.square(jnp.abs(diff))
                    
                a = 1.0
                
                r_test = diff / jnp.sqrt(diff_sq + a**2)
                
                y = jnp.prod(r_test, axis=1)

                return y


            else:
                raise NotImplementedError
            
    def safe_complex_logsumexp(self, x, b=None, eps=1e-12):
        """
        Assuming that's what actually returns nan in my code, every time innersum is close to machine preciosion 
        we evaluate log(0) = nan which corrupts all the code, trying to prevent speifically this situation 
        """
       
        x_real = jnp.real(x)
        x_max = jnp.max(x_real, axis=-1, keepdims=True)

        shifted_x = x - x_max

        if b is not None:
            exp_sum = jnp.sum(b * jnp.exp(shifted_x), axis=-1)
        else:
            exp_sum = jnp.sum(jnp.exp(shifted_x), axis=-1)

        safe_exp_sum = jnp.where(
            jnp.abs(exp_sum) < eps,
            eps + 0j, 
            exp_sum
        )

        out = jnp.log(safe_exp_sum) + jnp.squeeze(x_max, axis=-1)
    
        return out
    
    def pair_features(self, x_reshaped):
        # exchange-EVEN features of unordered pairs: invariant under i<->j within a pair
        # (all even in d -> -d), and sum-pooling makes the result permutation-symmetric,
        # so antisymmetry keeps coming solely from the +-eta flip.
        idx_i, idx_j = jnp.tril_indices(self.N, k=-1)
        d = x_reshaped[:, idx_i, :] - x_reshaped[:, idx_j, :]  # (batch, n_pairs, dim)
        r2 = jnp.sum(d**2, axis=-1)
        eps = 1e-3
        # all features BOUNDED on the sampled region (raw r^2/quadrupoles at random init
        # produced |log psi| ~ 30 and sigma^2 ~ 1e2 at step 0 — instant blow-up):
        # log(r^2+eps) is the load-bearing feature: -log T = sum [log(1+r^2) - log(r^2+eps)]
        # is a linear readout of the pooled vector. Normalized quadrupoles keep orientation.
        if self.dim == 2:
            feats = [jnp.log1p(r2),
                     (d[..., 0] ** 2 - d[..., 1] ** 2) / (1.0 + r2),
                     2.0 * d[..., 0] * d[..., 1] / (1.0 + r2),
                     jnp.log(r2 + eps),
                     1.0 / (r2 + 1.0)]
        else:
            feats = [jnp.log1p(r2), jnp.log(r2 + eps), 1.0 / (r2 + 1.0)]
        return jnp.stack(feats, axis=-1)

    def eval_psi0(self, x, eta):
        #x is (batch, N_particles, dim)
        x_reshaped = x.reshape(-1, self.N, self.dim) #-1 inferes the batch size automatically

        y = self.phi_dense1(x_reshaped)
        y = nnx.gelu(y)
        y = self.phi_dense2(y)
        y = jnp.sum(y, axis=1)

        y = self.rho_dense1(y)
        y = nnx.gelu(y)

        if self.pair_hidden:
            p = self.pair_dense1(self.pair_features(x_reshaped))
            p = nnx.gelu(p)
            p = self.pair_dense2(p)
            p = jnp.sum(p, axis=1)  # symmetric pool over pairs
            y = jnp.concatenate([y, p], axis=-1)

        
        log_eta_real = jnp.real(eta)[:, None]
        log_eta_imag = jnp.imag(eta)[:, None]

        #y + 2 real features + 2 imag features
        log_feat_concat = jnp.concatenate([y, log_eta_real, log_eta_imag], axis=-1)

        logPsi = self.Psi_dense1(log_feat_concat)
        logPsi = nnx.gelu(logPsi)

        # logPsi = self.Psi_dense_extra(logPsi)
        # logPsi = nnx.gelu(logPsi)

        logPsi = self.Psi_dense2(logPsi) 
     

        logPsireal, logPsiphase = jnp.split(logPsi, 2, axis= -1)

        logPsi_comp = logPsireal + 1j * logPsiphase #log psi = log(R) + log(phase)

        #logPsi_comp = jax.nn.logsumexp(logPsi_comp,axis=-1) 
        logPsi_comp = self.safe_complex_logsumexp(logPsi_comp) 
        #logPsi_comp = logPsi_comp.squeeze() 

        return logPsi_comp

    def __call__(self, x : jax.Array):
        if not self.lz_proj_K or self.lz_proj_K <= 1:
            return self._logpsi_base(x)

        # project onto L_z = 0 (mod K): psi_proj(x) = (1/K) sum_k psi(R(2 pi k / K) x).
        # rotations commute with particle permutations, so antisymmetry is preserved.
        import math
        K = self.lz_proj_K
        xr = x.reshape(-1, self.N, 2)
        logs = []
        for k in range(K):
            th = 2.0 * math.pi * k / K
            c, s = math.cos(th), math.sin(th)
            xrot = jnp.stack(
                [c * xr[..., 0] - s * xr[..., 1], s * xr[..., 0] + c * xr[..., 1]],
                axis=-1,
            )
            logs.append(self._logpsi_base(xrot.reshape(x.shape)))
        stacked = jnp.stack(logs, axis=-1)
        return self.safe_complex_logsumexp(stacked) - jnp.log(K)

    def _logpsi_base(self, x : jax.Array): #og forward pass without the L_z projection 

        eta = self.eta_antisymmetric(x)
        log_psi0_plus = self.eval_psi0(x, eta)
        log_psi0_minus = self.eval_psi0(x, -eta) # eta is a raw bounded product in both dims, so -eta IS the exact exchange flip
        
        stacked_logs = jnp.stack([log_psi0_plus, log_psi0_minus], axis=-1)
        weights = jnp.array([0.5, -0.5])
        
        #log_psi_nn = jax.nn.logsumexp(stacked_logs, axis=-1, b=weights)

        log_psi_nn = self.safe_complex_logsumexp(stacked_logs, b=weights)
        

        log_gaussian_factor = -0.5 * jnp.sum(jnp.square(x), axis=-1)

        logPsi = log_psi_nn + log_gaussian_factor

        #logPsireal = jnp.real(logPsi) 
        #logPsicompl = jnp.imag(logPsi) 

        #Psi = logPsireal + jnp.log(jnp.cos(logPsicompl)+ 0j )
        return logPsi
        

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
    
    def eta_antisymmetric(self, x): 
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

                        ##assumption : diff close to machine precision cause nan in energy 
                        eps = 1e-7
                        #jnp.where( condition, if true, if false)  , preserve sign, fix min magnitude 
                        safe_diff = jnp.where(
                            jnp.abs(diff) < eps, 
                            eps * jnp.sign(diff),
                            diff
                        )

                        log_diff = jnp.log(safe_diff.astype(jnp.complex64))

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
            log_eta = self.eta_antisymmetric(X_reshaped)
        return exponent + log_eta #nk expects log , don't exponentiate


class SlaterNN(nnx.Module):
    """
    QUEUE.md P2 baseline: N learned single-particle orbitals phi_k(x_i) (shared MLP,
    dim -> hidden_units -> N), Slater determinant via jnp.linalg.slogdet of the N x N
    orbital matrix M[i,k] = phi_k(x_i). Antisymmetry under particle exchange is exact
    by construction (swapping two rows of M flips det's sign) -- no signature encoder,
    no L_z projection needed. Purpose: isolate whether "antisymmetry by construction"
    alone avoids the holomorphic-trap pathology that FermiSets needs L_z projection to
    escape (see RESEARCH_LOG "holomorphic trap" entries).

    Orbitals are real-valued, so slogdet's sign is discrete (+-1) -- the resulting
    phase (0 or pi) is a locally-constant field, same as any real-orbital Slater
    determinant in VMC (PauliNet/FermiNet-style): its gradient is zero a.e. away from
    the nodal surface, which is expected and does not break VMC_SR's mode="complex"
    training (only log|psi|'s gradient carries signal there).
    """
    def __init__(self, dim: int, N: int, rngs: nnx.Rngs, hidden_units: int = 64):
        self.dim = dim
        self.N = N
        self.hidden_units = hidden_units

        self.orb_dense1 = nnx.Linear(in_features=dim, out_features=hidden_units, rngs=rngs)
        self.orb_dense2 = nnx.Linear(in_features=hidden_units, out_features=hidden_units, rngs=rngs)
        self.orb_dense3 = nnx.Linear(in_features=hidden_units, out_features=N, rngs=rngs)

    def __call__(self, x: jax.Array):
        x_reshaped = x.reshape(-1, self.N, self.dim)  # (batch, N, dim)

        y = self.orb_dense1(x_reshaped)
        y = nnx.gelu(y)
        y = self.orb_dense2(y)
        y = nnx.gelu(y)
        orbitals = self.orb_dense3(y)  # (batch, N, N): orbitals[b, i, k] = phi_k(x_i)

        sign, logabsdet = jnp.linalg.slogdet(orbitals)
        phase = jnp.where(sign < 0, jnp.pi, 0.0)

        log_gaussian_factor = -0.5 * jnp.sum(jnp.square(x), axis=-1)

        logPsi = logabsdet + log_gaussian_factor + 1j * phase
        return logPsi 