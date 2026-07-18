"""
Projector/deflation penalty against the KNOWN degenerate lazy subspace {holo, antiholo}
(RESEARCH_LOG 2026-07-16). Supersedes the first-order (D) and second-order (Laplacian, L)
differential holomorphy penalties in src/holomorphy_penalty.py: both of those are LOCAL
quantities and were shown empirically to be satisfiable by remixing WITHIN the exactly
degenerate {holo, antiholo} subspace (any reflection-symmetric potential has this
degeneracy) without any energy change -- a plateaued N=4 test scored as high as the
actual analytic GS on both D and L while sitting at the trap energy E=10.

This penalty is instead a GLOBAL quantity: the standard Choo-Carleo-style overlap
estimator |<psi|phi>|^2 / (<psi|psi><phi|phi>) for phi in {holo, antiholo}, which is zero
only when psi has NO component along either basis vector -- it cannot be satisfied by
moving between them.

|<psi|phi>|^2/(<psi|psi><phi|phi>) = r1 * r2, where
  r1 = E_{x~|psi|^2}[phi(x)/psi(x)]     (normalization by <psi|psi> is automatic: x is
                                          drawn from the NORMALIZED |psi|^2/<psi|psi>)
  r2 = E_{y~|phi|^2}[psi(y)/phi(y)]     (ditto for <phi|phi>)
holo and antiholo are known analytically (det{1,z,...,z^(N-1)} / its conjugate, times the
Gaussian envelope) so y-samples can be drawn EXACTLY and for free: |holo(x)|^2 *
exp(-sum|z_i|^2) is precisely the eigenvalue density of the complex Ginibre ensemble
(|Vandermonde|^2 * exp(-sum|z_i|^2) is the textbook GinUE joint eigenvalue density) --
no MCMC needed, and |antiholo|^2 = |holo|^2 (same distribution, conjugate function), so
the SAME Ginibre samples serve both terms.

Caveat carried over from the holomorphy-penalty implementation: the x-side gradient here
treats the current training-chain samples as FIXED (i.e. it omits the score-function
correction term that accounts for |psi|^2 itself depending on parameters -- the same term
netket's own VMC energy gradient includes via the log-derivative trick). This makes the
auxiliary force an approximate/heuristic gradient of the overlap, not the exact one -- an
annealed, secondary nudge on top of the exact SR energy step, same spirit and same
limitation as HolomorphyPenalty. Acceptable here because the physics is still driven by
the (untouched, exact) energy gradient; this force only needs to point roughly the right
direction (away from known lazy states), not be exact.
"""
import logging

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def sample_ginibre_positions(N: int, n_samples: int, seed: int = 0):
    """Exact samples from |Vandermonde(z)|^2 * exp(-sum|z_i|^2) via the eigenvalues of
    an N x N complex Ginibre matrix (i.i.d. standard complex Gaussian entries) --
    matches |holo(x)|^2 = |antiholo(x)|^2 exactly, no MCMC needed."""
    rng = np.random.default_rng(seed)
    xs = np.zeros((n_samples, N, 2))
    for i in range(n_samples):
        M = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0)
        eigs = np.linalg.eigvals(M)
        xs[i, :, 0] = eigs.real
        xs[i, :, 1] = eigs.imag
    return jnp.asarray(xs.reshape(n_samples, N * 2))


def log_holo(x, N):
    xr = x.reshape(-1, N, 2)
    z = xr[..., 0] + 1j * xr[..., 1]
    Md = jnp.stack([z**k for k in range(N)], axis=-1)
    det = jnp.linalg.det(Md)
    env = -0.5 * jnp.sum(xr[..., 0] ** 2 + xr[..., 1] ** 2, axis=-1)
    return jnp.log(det.astype(jnp.complex128)) + env


def log_antiholo(x, N):
    xr = x.reshape(-1, N, 2)
    zb = xr[..., 0] - 1j * xr[..., 1]
    Md = jnp.stack([zb**k for k in range(N)], axis=-1)
    det = jnp.linalg.det(Md)
    env = -0.5 * jnp.sum(xr[..., 0] ** 2 + xr[..., 1] ** 2, axis=-1)
    return jnp.log(det.astype(jnp.complex128)) + env


def _safe_mean_exp(a):
    """mean(exp(a)) for complex a, shifted by max(Re a) to avoid overflow (same pattern
    as FermiSets.safe_complex_logsumexp elsewhere in this codebase)."""
    shift = jnp.max(jnp.real(a))
    return jnp.exp(shift) * jnp.mean(jnp.exp(a - shift))


class DeflationPenalty:
    def __init__(self, template_model, N: int, n_ginibre: int, mu0: float, decay: float,
                 aux_lr: float, log: logging.Logger, seed: int = 123):
        self.graphdef, self.state_template = nnx.split(template_model)
        self.N = N
        self.mu0 = mu0
        self.decay = decay
        self.aux_lr = aux_lr
        self.log = log
        self.y = sample_ginibre_positions(N, n_ginibre, seed=seed)
        self.log_holo_y = log_holo(self.y, N)
        self.log_antiholo_y = log_antiholo(self.y, N)

    def _overlap_penalty(self, params_dict, x_batch):
        state = self.state_template
        state.replace_by_pure_dict(params_dict)
        m = nnx.merge(self.graphdef, state)

        logpsi_x = m(x_batch)
        log_holo_x = log_holo(x_batch, self.N)
        log_antiholo_x = log_antiholo(x_batch, self.N)
        r1_holo = _safe_mean_exp(log_holo_x - logpsi_x)
        r1_antiholo = _safe_mean_exp(log_antiholo_x - logpsi_x)

        logpsi_y = m(self.y)
        r2_holo = _safe_mean_exp(logpsi_y - self.log_holo_y)
        r2_antiholo = _safe_mean_exp(logpsi_y - self.log_antiholo_y)

        O_holo = jnp.real(r1_holo * r2_holo)
        O_antiholo = jnp.real(r1_antiholo * r2_antiholo)
        return O_holo + O_antiholo

    def __call__(self, step: int, log_data: dict, driver) -> bool:
        mu = self.mu0 * (self.decay**step)
        vstate = driver.state
        x = jnp.asarray(vstate.samples).reshape(-1, self.N * 2)
        params = vstate.parameters
        val, grad = jax.value_and_grad(self._overlap_penalty)(params, x)
        if mu > 1e-10:
            # DESCENT on overlap (opposite sign from the D-ascent in HolomorphyPenalty --
            # we want LESS lazy-subspace content, not more of some derivative quantity).
            vstate.parameters = jax.tree_util.tree_map(
                lambda p, g: p - self.aux_lr * mu * g, params, grad
            )
        log_data["Deflation_overlap"] = float(val)
        log_data["Deflation_penalty_mu"] = mu
        return True
