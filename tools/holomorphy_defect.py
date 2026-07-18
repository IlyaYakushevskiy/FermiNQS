"""
Holomorphy-defect diagnostic (RESEARCH_LOG 2026-07-16, "penalize the LLL state" idea).

The holomorphic trap family is psi = eta(z) * g(xi) with g a SYMMETRIC, HOLOMORPHIC
function of z (depends on z, not zbar) -- literally the lowest-Landau-level / Laughlin
m=1 family. This is a property of the ansatz's optimization dynamics, not of any
particular potential's rotational symmetry, so it can be measured (and later penalized)
without any L_z projection and without knowing the exact target ground state.

Define h(x) = psi_nn(x) / eta(x), where psi_nn is the network's antisymmetric factor
BEFORE the analytic Gaussian envelope is multiplied in (the envelope is common to every
state we care about -- trap or GS -- and is itself non-holomorphic, so it must be
factored out or it swamps the signal). h is always permutation-symmetric (both psi_nn
and eta flip sign under any particle swap). The trap family has h holomorphic in each
z_i; the true GS does not (Fu's own flagged hard case: psi/eta is non-smooth there).

D(x) = sum_i |d(log h)/d zbar_i|^2   (Wirtinger anti-holomorphic derivative)

is exactly 0 at any trap-family state and should be > 0 at genuine 2D (non-Vandermonde)
sign structures. Computed via ordinary real-valued forward-mode autodiff (Cauchy-Riemann
violation), no complex-step tricks needed.

Usage: python tools/holomorphy_defect.py <ckpt.mpack> [--hidden 64] [--out 10]
                                          [--potential qho_no_inter] [--omega-y 1.0]
                                          [--samples 20000]
Note: this always evaluates the UNPROJECTED base wavefunction (_logpsi_base equivalent),
even if the checkpoint was trained with lz_proj_K > 0 -- deliberately, since the point is
to characterize the raw network's own antisymmetric factor, not the projected physical
state. Pass --lz-proj-K only to load a checkpoint's params with matching architecture
shapes if pair_hidden/lz_proj_K affect parameter count (lz_proj_K doesn't add params, so
this normally doesn't matter).
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import flax
from flax import nnx
import netket as nk

from src.ansatz import FermiSets
from src.system import System

jax.config.update("jax_enable_x64", True)

N_DIM = 2  # dim=2 only; eta_antisymmetric's complex-z construction is dim-specific


def eta_ideal(x, N):
    """UNREGULARIZED complex Vandermonde prod_{i<j}(z_i - z_j) -- exactly holomorphic,
    unlike the architecture's own bounded eta_antisymmetric (diff/sqrt(|diff|^2+a^2) is
    itself non-holomorphic away from collisions, since |diff|^2 depends on zbar too --
    dividing by the network's OWN regularized eta would contaminate D with that mismatch
    rather than measuring the network's holomorphy). This matches the plain-det analytic
    probes already used in overlap_check.py (holo/antiholo/gs/excx), so h = psi_nn /
    eta_ideal tests exactly "is psi_nn = (plain Vandermonde) * (holomorphic symmetric)",
    the literal trap-family definition."""
    xr = x.reshape(-1, N, 2)
    z = xr[..., 0] + 1j * xr[..., 1]
    idx_i, idx_j = jnp.tril_indices(N, k=-1)
    diff = z[:, idx_i] - z[:, idx_j]
    return jnp.prod(diff, axis=1)


def log_h_components(model, x, N):
    """x: (N*dim,) single sample. Returns (Re log h, Im log h), h = psi_nn / eta_ideal,
    psi_nn = the antisymmetric factor BEFORE the Gaussian envelope."""
    xb = x[None, :]
    eta = model.eta_antisymmetric(xb)[0]
    log_psi0_plus = model.eval_psi0(xb, eta[None])[0]
    log_psi0_minus = model.eval_psi0(xb, -eta[None])[0]
    stacked = jnp.stack([log_psi0_plus, log_psi0_minus])[None, :]
    weights = jnp.array([0.5, -0.5])
    log_psi_nn = model.safe_complex_logsumexp(stacked, b=weights)[0]

    eta_id = eta_ideal(xb, N)[0]
    eps = 1e-12
    safe_eta = jnp.where(jnp.abs(eta_id) < eps, eps + 0j, eta_id)
    log_h = log_psi_nn - jnp.log(safe_eta)
    return jnp.real(log_h), jnp.imag(log_h)


def holomorphy_defect_single(model, x, dim=N_DIM):
    """D(x) = sum_i |d(log h)/d zbar_i|^2 via the Cauchy-Riemann combination of the
    real Jacobian of (Re log h, Im log h) w.r.t. each particle's (x_i, y_i)."""
    N = x.shape[0] // dim
    fn = lambda xx: log_h_components(model, xx, N)
    jac_r, jac_i = jax.jacfwd(fn)(x)  # each (N*dim,)
    jac_r = jac_r.reshape(N, dim)
    jac_i = jac_i.reshape(N, dim)
    dFr_dx, dFr_dy = jac_r[:, 0], jac_r[:, 1]
    dFi_dx, dFi_dy = jac_i[:, 0], jac_i[:, 1]
    re_part = 0.5 * (dFr_dx - dFi_dy)
    im_part = 0.5 * (dFi_dx + dFr_dy)
    return jnp.sum(re_part**2 + im_part**2)


def holomorphy_defect_batch(model, x_batch, dim=N_DIM):
    return jax.vmap(lambda x: holomorphy_defect_single(model, x, dim))(x_batch)


def laplacian_defect_single(model, x, dim=N_DIM):
    """L(x) = sum_i |d^2(log h)/(d z_i d zbar_i)|^2, the MIXED Wirtinger second
    derivative -- equal to (1/4) * the ordinary 2D Laplacian w.r.t. particle i's own
    (x_i, y_i), since d/dz d/dzbar = (1/4)(d^2/dx^2 + d^2/dy^2).

    Upgrade over the first-order D (RESEARCH_LOG 2026-07-16, N=4 no-projection test):
    D alone is zero for f(z) (holomorphic) OR g(zbar) (antiholomorphic) individually,
    but a plain first-order defect can be satisfied cheaply by mixing INTO the exactly
    degenerate antiholomorphic mirror of the trap (same energy for any reflection-
    symmetric potential) rather than by finding genuinely 2D structure -- observed
    directly: a penalized N=4 run plateaued at E=10 (the trap energy) with D and the
    opposite-chirality D' BOTH elevated and nearly equal, consistent with exactly this
    degenerate remix, not real progress. log h harmonic (Laplacian 0) for EITHER pure
    chirality (standard fact: holomorphic and antiholomorphic functions are both
    harmonic) -- so L is a strictly higher-order probe than D, sensitive to curvature/
    nonlinearity in the z-zbar coupling rather than just "any nonzero zbar-dependence".
    NOTE (checked 2026-07-16 with a toy example, log(2x) = log(z+zbar)): L is NOT
    guaranteed zero for every superposition of holo+antiholo PIECES (only for each pure
    piece individually) -- log of a sum is not additive under the Laplacian the way log
    of a pure holomorphic/antiholomorphic function is. So validate empirically against
    real checkpoints before trusting it (see task in QUEUE.md); it is a strictly more
    discriminating quantity than D, not a proven perfect fix.
    """
    N = x.shape[0] // dim
    fn_r = lambda xx: log_h_components(model, xx, N)[0]
    fn_i = lambda xx: log_h_components(model, xx, N)[1]
    hess_r = jax.hessian(fn_r)(x)
    hess_i = jax.hessian(fn_i)(x)
    total = 0.0
    for i in range(N):
        ix, iy = dim * i, dim * i + 1
        lap_r = hess_r[ix, ix] + hess_r[iy, iy]
        lap_i = hess_i[ix, ix] + hess_i[iy, iy]
        total = total + (lap_r**2 + lap_i**2) / 16.0
    return total


def laplacian_defect_batch(model, x_batch, dim=N_DIM):
    return jax.vmap(lambda x: laplacian_defect_single(model, x, dim))(x_batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=int, default=10)
    ap.add_argument("--pair-hidden", type=int, default=0)
    ap.add_argument("--potential", type=str, default="qho_no_inter")
    ap.add_argument("--omega-y", type=float, default=1.0)
    ap.add_argument("--int-strength", type=float, default=2.0)
    ap.add_argument("--int-range", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=20_000)
    ap.add_argument("--n-chains", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--also-raw-gaussian", action="store_true",
                     help="also report D under a naive N(0,1) proposal (includes"
                          " collision tails outside the physical |psi|^2 support --"
                          " kept only for comparison/debugging, not the main number).")
    args = ap.parse_args()

    N = args.N
    if args.potential == "qho_aniso":
        system = System(N=N, dim=2, mass=1.0, potential="qho_aniso", omega_y=args.omega_y)
    elif args.potential == "dot_gauss":
        system = System(N=N, dim=2, mass=1.0, potential="dot_gauss",
                         int_strength=args.int_strength, int_range=args.int_range)
    else:
        system = System(N=N, dim=2, mass=1.0, potential="qho_no_inter")

    model = FermiSets(dim=2, N=N, rngs=nnx.Rngs(42), log=logging.getLogger(),
                       hidden_units=args.hidden, out_units=args.out,
                       lz_proj_K=0, pair_hidden=args.pair_hidden)

    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=args.sigma,
                                             n_chains=args.n_chains, sweep_size=32)
    vstate = nk.vqs.MCState(sampler, model, n_samples=max(args.n_chains, 4096), seed=1,
                             n_discard_per_chain=200, chunk_size=4096)

    with open(args.ckpt, "rb") as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
    print(f"loaded {args.ckpt}")
    model = vstate.model  # unprojected FermiSets with trained params, same nnx graph

    E = vstate.expect(system.H)
    print(f"sanity energy (should match the checkpoint's known training/validation energy): {E}")

    def report(x, label):
        Ds = []
        CH = 2000
        for start in range(0, x.shape[0], CH):
            Ds.append(holomorphy_defect_batch(model, x[start:start + CH]))
        D = jnp.concatenate(Ds)
        print(f"[{label}] D = E[sum_i |d(log h)/d zbar_i|^2]  over {D.shape[0]} samples")
        print(f"  mean = {float(jnp.mean(D)):.6e}  median = {float(jnp.median(D)):.6e}  "
              f"std = {float(jnp.std(D)):.6e}  [min,max] = "
              f"[{float(jnp.min(D)):.3e}, {float(jnp.max(D)):.3e}]")

    # physical samples: draw from |psi|^2 via the trained model's own Metropolis sampler
    # (matches the training/eval distribution; avoids the collision tail that a naive
    # N(0,1) proposal would include, where log(eta_ideal) is genuinely singular and
    # inflates D with an artifact of the reference function, not of the network).
    vstate.n_samples = args.samples
    vstate.reset()
    vstate.sample()  # extra decorrelation sweep beyond n_discard_per_chain
    x_phys = jnp.asarray(vstate.samples).reshape(-1, N * 2)
    report(x_phys, "physical |psi|^2 samples")

    if args.also_raw_gaussian:
        key, sub = jax.random.split(key)
        x_raw = jax.random.normal(sub, (args.samples, N * 2), dtype=jnp.float64)
        report(x_raw, "naive N(0,1) proposal (includes collision tail, for reference only)")


if __name__ == "__main__":
    main()
