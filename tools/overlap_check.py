"""
Diagnostic: load a FermiSets N=3 2D checkpoint and report its energy and squared
overlaps with the three relevant analytic eigenstates:

  gs        det[1, x, y]   * exp(-sum r^2/2)   E = 5   (true ground state, L_z = 0)
  holo      det[1, z, z^2] * exp(...)          E = 6   (Vandermonde trap, L_z = +3)
  antiholo  det[1, zb, zb^2] * exp(...)        E = 6   (conjugate mirror,  L_z = -3)

Overlaps via importance sampling from q = N(0,1)^(N*dim).

Usage: python tools/overlap_check.py <checkpoint.mpack> [--hidden 64] [--out 10] [--samples 400000]
See RESEARCH_LOG.md 2026-07-14 for why these three states.
"""
import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import netket as nk
import jax
import jax.numpy as jnp
import flax
from flax import nnx

from src.system import System
from src.ansatz import FermiSets

N, DIM = 3, 2


def log_gs(x):
    xr = x.reshape(-1, N, DIM)
    ones = jnp.ones_like(xr[..., 0])
    Md = jnp.stack([ones, xr[..., 0], xr[..., 1]], axis=-1).astype(jnp.complex128)
    return jnp.log(jnp.linalg.det(Md)) - 0.5 * jnp.sum(x**2, axis=-1)


def log_holo(x):
    xr = x.reshape(-1, N, DIM)
    z = xr[..., 0] + 1j * xr[..., 1]
    Md = jnp.stack([jnp.ones_like(z), z, z**2], axis=-1)
    return jnp.log(jnp.linalg.det(Md)) - 0.5 * jnp.sum(x**2, axis=-1)


def log_antiholo(x):
    xr = x.reshape(-1, N, DIM)
    zb = xr[..., 0] - 1j * xr[..., 1]
    Md = jnp.stack([jnp.ones_like(zb), zb, zb**2], axis=-1)
    return jnp.log(jnp.linalg.det(Md)) - 0.5 * jnp.sum(x**2, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help=".mpack checkpoint (vstate.variables serialization)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=int, default=10)
    ap.add_argument("--samples", type=int, default=400_000)
    args = ap.parse_args()

    system = System(N=N, dim=DIM, mass=1.0, potential="qho_no_inter")
    model = FermiSets(dim=DIM, N=N, rngs=nnx.Rngs(42), log=logging.getLogger(),
                      hidden_units=args.hidden, out_units=args.out)
    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=0.35, n_chains=64, sweep_size=32)
    vstate = nk.vqs.MCState(sampler, model, n_samples=8192, seed=1, chunk_size=4096)

    with open(args.ckpt, "rb") as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
    print(f"loaded {args.ckpt}")

    E = vstate.expect(system.H)
    print(f"energy: {E}   (gs = 5.0, holo/antiholo = 6.0)")

    funcs = {"nn": lambda x: vstate.log_value(x),
             "gs": log_gs, "holo": log_holo, "antiholo": log_antiholo}
    names = list(funcs)
    S = {(a, b): 0.0 + 0j for a in names for b in names}

    key = jax.random.PRNGKey(7)
    CH = 50_000
    shift = None
    for c in range(args.samples // CH):
        key, sub = jax.random.split(key)
        x = jax.random.normal(sub, (CH, N * DIM), dtype=jnp.float64)
        logq = -0.5 * jnp.sum(x**2, axis=-1)  # constants cancel in the normalized ratio
        logs = {n: f(x) for n, f in funcs.items()}
        if shift is None:
            shift = {n: float(jnp.max(jnp.real(v))) for n, v in logs.items()}
        for a in names:
            for b in names:
                w = jnp.exp(jnp.conj(logs[a] - shift[a]) + (logs[b] - shift[b]) - logq)
                S[(a, b)] += complex(jnp.sum(w))

    print()
    for b in ["gs", "holo", "antiholo"]:
        ov2 = abs(S[("nn", b)]) ** 2 / (S[("nn", "nn")].real * S[(b, b)].real)
        print(f"|<nn|{b}>|^2 = {ov2:.6f}")
    ov2 = abs(S[("gs", "holo")]) ** 2 / (S[("gs", "gs")].real * S[("holo", "holo")].real)
    print(f"[sanity] |<gs|holo>|^2 = {ov2:.2e} (should be ~0)")


if __name__ == "__main__":
    main()
