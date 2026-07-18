"""
Training-time auxiliary force that pushes FermiSets away from the holomorphic-trap
family WITHOUT any L_z projection and without a known target wavefunction
(RESEARCH_LOG 2026-07-16, "penalize the LLL state" idea).

D(x) = sum_i |d(log(psi_nn/eta_ideal))/d zbar_i|^2  (tools/holomorphy_defect.py) is the
Cauchy-Riemann violation of the network's antisymmetric factor against the ideal complex
Vandermonde. It is ~0 exactly for trap-family states and measurably larger elsewhere
(validated against 4 existing checkpoints: isotropic/dot-interacting traps ~0.27-0.29,
aniso trap ~0.91, a non-trap-non-GS state ~1.05 -- a real but not enormous separation,
strongest for the isotropic/complex-Vandermonde family specifically).

This callback takes an ANNEALED gradient-ASCENT step on mean(D) after each driver step
(added on top of, not instead of, the ordinary SR energy step), using the current
training-chain samples so it costs one extra jax.grad pass per iteration.
"""
import jax
import jax.numpy as jnp
from flax import nnx

from tools.holomorphy_defect import holomorphy_defect_batch


class HolomorphyPenalty:
    def __init__(self, template_model, N: int, mu0: float, decay: float, aux_lr: float, log):
        self.graphdef, self.state_template = nnx.split(template_model)
        self.N = N
        self.mu0 = mu0
        self.decay = decay
        self.aux_lr = aux_lr
        self.log = log

    def _mean_D(self, params_dict, x_batch):
        state = self.state_template
        state.replace_by_pure_dict(params_dict)
        m = nnx.merge(self.graphdef, state)
        return jnp.mean(holomorphy_defect_batch(m, x_batch))

    def __call__(self, step: int, log_data: dict, driver) -> bool:
        mu = self.mu0 * (self.decay**step)
        vstate = driver.state
        x = jnp.asarray(vstate.samples).reshape(-1, self.N * 2)
        params = vstate.parameters
        val, grad = jax.value_and_grad(self._mean_D)(params, x)
        if mu > 1e-10:
            vstate.parameters = jax.tree_util.tree_map(
                lambda p, g: p + self.aux_lr * mu * g, params, grad
            )
        log_data["Holomorphy_defect"] = float(val)
        log_data["Holomorphy_penalty_mu"] = mu
        return True
