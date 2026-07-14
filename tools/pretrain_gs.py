"""
Phase-1 diagnostic (RESEARCH_LOG.md 2026-07-14): supervised-fit the CURRENT FermiSets
architecture to the analytic N=3 2D ground state, then hand the checkpoint to VMC.

  target: log psi_GS(x) = log|det[1,x,y]| + i*pi*[det<0] - 0.5*sum r^2   (E = 5, L_z = 0)

The fit minimizes Var(Re delta) + mean(1 - cos(Im delta)) with delta = log psi_NN - log psi_GS
(variance instead of MSE because the overall normalization is irrelevant; 1-cos because the
phase target is 0/pi and a global phase is irrelevant). Samples near the GS nodal surface
(collinear configurations, |det| < eps) are masked out — log|psi_GS| diverges there.

Output: outputs/pretrained/gs_N3_2d_h<hidden>.mpack, compatible with ansatz.pretrained_path.

Usage: python tools/pretrain_gs.py [--steps 6000] [--batch 4096] [--hidden 64] [--out 10]
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import netket as nk  # noqa: F401  x64 side effect, must precede jnp use
import jax
import jax.numpy as jnp
import flax
from flax import nnx
import optax

from src.system import System
from src.ansatz import FermiSets

N, DIM = 3, 2
NODE_EPS = 1e-3  # mask samples closer to the GS node than this |det|


def log_gs_target(x):
    xr = x.reshape(-1, N, DIM)
    ones = jnp.ones_like(xr[..., 0])
    det = jnp.linalg.det(jnp.stack([ones, xr[..., 0], xr[..., 1]], axis=-1))
    logmag = jnp.log(jnp.abs(det)) - 0.5 * jnp.sum(x**2, axis=-1)
    phase = jnp.pi * (det < 0)
    return logmag + 1j * phase, jnp.abs(det)


def make_batch(key, n):
    """75% x ~ N(0,1)^6, 25% with one pair squeezed together (collision neighborhood)."""
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (n, N * DIM), dtype=jnp.float64)
    n_aug = n // 4
    aug = x[:n_aug].reshape(n_aug, N, DIM)
    delta = 0.15 * jax.random.normal(k2, (n_aug, DIM), dtype=jnp.float64)
    aug = aug.at[:, 1, :].set(aug[:, 0, :] + delta)
    return jnp.concatenate([aug.reshape(n_aug, N * DIM), x[n_aug:]], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    model = FermiSets(dim=DIM, N=N, rngs=nnx.Rngs(42), log=logging.getLogger(),
                      hidden_units=args.hidden, out_units=args.out)
    graphdef, state = nnx.split(model)

    tx = optax.adam(optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.1))
    opt_state = tx.init(state)

    @jax.jit
    def step(state, opt_state, x):
        target, absdet = log_gs_target(x)
        mask = (absdet > NODE_EPS).astype(jnp.float64)
        wsum = jnp.sum(mask)

        def loss_fn(state):
            m = nnx.merge(graphdef, state)
            d = m(x) - target
            re, im = jnp.real(d), jnp.imag(d)
            re_mean = jnp.sum(mask * re) / wsum
            amp = jnp.sum(mask * (re - re_mean) ** 2) / wsum
            phase = jnp.sum(mask * (1.0 - jnp.cos(im))) / wsum
            return amp + phase, (amp, phase)

        (loss, (amp, phase)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state)
        updates, opt_state = tx.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)
        return state, opt_state, loss, amp, phase

    key = jax.random.PRNGKey(0)
    for i in range(args.steps):
        key, sub = jax.random.split(key)
        x = make_batch(sub, args.batch)
        state, opt_state, loss, amp, phase = step(state, opt_state, x)
        if i % 500 == 0 or i == args.steps - 1:
            print(f"step {i:5d}  loss={float(loss):.5f}  amp={float(amp):.5f}  phase={float(phase):.5f}")

    nnx.update(model, state)

    # serialize in the vstate.variables format that main.py's pretrained_path loader expects
    system = System(N=N, dim=DIM, mass=1.0, potential="qho_no_inter")
    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=0.35, n_chains=64, sweep_size=32)
    vstate = nk.vqs.MCState(sampler, model, n_samples=16384, seed=1, chunk_size=4096)

    E = vstate.expect(system.H)
    print(f"\nVMC energy of the pretrained state: {E}  (target: 5.0)")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "pretrained")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gs_N{N}_{DIM}d_h{args.hidden}.mpack")
    with open(out_path, "wb") as f:
        f.write(flax.serialization.to_bytes(vstate.variables))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
