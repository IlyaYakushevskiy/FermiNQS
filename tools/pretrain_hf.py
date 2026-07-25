"""
Hartree-Fock / Slater-determinant pretraining for FermiSets — the production-viable
successor of tools/pretrain_gs.py (RESEARCH_LOG.md 2026-07-14, "Outcome A" follow-up).

Target: log psi_HF(x) = log|det[phi_j(r_i)]| + i*pi*[det<0], with phi_j the N lowest
2D harmonic-oscillator orbitals (aufbau filling, deterministic within degenerate shells).
No exact wavefunction is used anywhere: for the non-interacting benchmark HF happens to be
exact, but the same pipeline applies to interacting systems — there, replace the identity
orbital coefficients with the C matrix from a self-consistent HF solve (see `make_log_hf`).

Output: outputs/pretrained/hf_N<N>_2d_h<hidden>.mpack, compatible with ansatz.pretrained_path.

Usage: python tools/pretrain_hf.py [--N 3] [--steps 30000] [--batch 8192] [--hidden 64] [--out 10]
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
from main import exact_qho_gs_energy

DIM = 2
NODE_EPS = 1e-3  # mask samples closer to the HF nodal surface than this |det|


def hermite(n, x):
    """Physicists' Hermite polynomial H_n via recurrence (small n only)."""
    h0, h1 = jnp.ones_like(x), 2.0 * x
    if n == 0:
        return h0
    for k in range(1, n):
        h0, h1 = h1, 2.0 * x * h1 - 2.0 * k * h0
    return h1


def aufbau(N):
    """The N lowest 2D-oscillator orbitals (n_x, n_y), deterministic within shells."""
    orbs = sorted((nx + ny, nx, ny) for nx in range(N + 1) for ny in range(N + 1))[:N]
    return [(nx, ny) for (_, nx, ny) in orbs]


def make_log_hf(N, coeffs=None):
    """
    log of the HF Slater determinant (unnormalized) and |det| for node masking.
    `coeffs`: optional (N_basis, N) orbital-coefficient matrix from an SCF solve;
    None means identity (non-interacting HF = the bare oscillator orbitals).
    """
    orbitals = aufbau(N)

    def log_hf(x):
        xr = x.reshape(-1, N, DIM)
        basis = jnp.stack(
            [hermite(nx, xr[..., 0]) * hermite(ny, xr[..., 1]) for nx, ny in orbitals],
            axis=-1,
        )  # (batch, N particles, N basis fns); envelope factored out below
        M = basis if coeffs is None else basis @ coeffs
        det = jnp.linalg.det(M)
        logmag = jnp.log(jnp.abs(det)) - 0.5 * jnp.sum(xr**2, axis=(-2, -1))
        return logmag + 1j * jnp.pi * (det < 0), jnp.abs(det)

    return log_hf, orbitals


def make_batch(key, n, N):
    """75% x ~ N(0,1), 25% with one pair squeezed together (collision neighborhood)."""
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (n, N * DIM), dtype=jnp.float64)
    n_aug = n // 4
    aug = x[:n_aug].reshape(n_aug, N, DIM)
    delta = 0.15 * jax.random.normal(k2, (n_aug, DIM), dtype=jnp.float64)
    aug = aug.at[:, 1, :].set(aug[:, 0, :] + delta)
    return jnp.concatenate([aug.reshape(n_aug, N * DIM), x[n_aug:]], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--pair-hidden", type=int, default=0)
    ap.add_argument("--backflow-hidden", type=int, default=0)
    args = ap.parse_args()
    N = args.N

    log_hf, orbitals = make_log_hf(N)
    print(f"HF target: N={N} spinless fermions, occupied orbitals (n_x,n_y) = {orbitals}")

    model = FermiSets(dim=DIM, N=N, rngs=nnx.Rngs(42), log=logging.getLogger(),
                      hidden_units=args.hidden, out_units=args.out,
                      pair_hidden=args.pair_hidden,
                      backflow_hidden=args.backflow_hidden)
    graphdef, state = nnx.split(model)

    tx = optax.adam(optax.cosine_decay_schedule(args.lr, args.steps, alpha=0.1))
    opt_state = tx.init(state)

    @jax.jit
    def step(state, opt_state, x):
        target, absdet = log_hf(x)
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
        x = make_batch(sub, args.batch, N)
        state, opt_state, loss, amp, phase = step(state, opt_state, x)
        if i % 2000 == 0 or i == args.steps - 1:
            print(f"step {i:6d}  loss={float(loss):.5f}  amp={float(amp):.5f}  phase={float(phase):.5f}")

    nnx.update(model, state)

    system = System(N=N, dim=DIM, mass=1.0, potential="qho_no_inter")
    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=0.35, n_chains=64, sweep_size=32)
    vstate = nk.vqs.MCState(sampler, model, n_samples=16384, seed=1, chunk_size=4096)

    E = vstate.expect(system.H)
    e_exact = exact_qho_gs_energy(N, DIM, "fermion")
    print(f"\nVMC energy of the HF-pretrained state: {E}  (exact GS = {e_exact}, HF = exact here)")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "outputs", "pretrained")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_p{args.pair_hidden}" if args.pair_hidden else ""
    suffix += f"_bf{args.backflow_hidden}" if args.backflow_hidden else ""
    out_path = os.path.join(out_dir, f"hf_N{N}_2d_h{args.hidden}{suffix}.mpack")
    with open(out_path, "wb") as f:
        f.write(flax.serialization.to_bytes(vstate.variables))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
