"""
Quick diagnostic (2026-07-20): what is the trained N=6 K=3 checkpoint actually converging
to? It plateaus around E~16.3-16.7, well below the E=21 holomorphic trap (correctly
excluded by L_z projection) but well above the exact GS (E=14.0). This script measures
overlap against a few analytic candidates instead of guessing from the energy alone.

Candidates, all exact eigenstates of the non-interacting N=6 2D QHO (energies in hbar*omega):
  gs     - true ground state: shells n=0,1,2 fully filled, E=14, L_z=0. Built from the
           plain monomial basis {1,x,y,x^2,xy,y^2}: since this spans the FULL degree<=2
           polynomial subspace (= shells 0+1+2 together), the determinant is identical
           (up to a constant) to the one built from true Hermite/Laguerre eigenstates -
           the same trick used in tools/pretrain_hf.py.
  holo   - the original holomorphic trap: top-m (p=0) orbital of every shell 0..5,
           i.e. {1,z,z^2,z^3,z^4,z^5}. E=21, L_z=15. Excluded exactly by projection,
           just here as a sanity check that it's ~0.
  e16a   - shells 0,1 full + shell-2's m=0 orbital (the p=1 "radial node" state,
           propto (1-r^2)) + shell-3's m=+-3 pair (top orbitals z^3, z^3bar).
           E = 1*1 + 2*2 + 1*3 + 2*4 = 16, L_z = 0 + 0 + 0 + (3-3) = 0.
  e16b   - same shell-2 orbital, but shell-3's m=+-1 pair instead (p=1 states,
           propto (2-r^2)*z and (2-r^2)*z_bar). Also E=16, L_z=0.
Both e16a/e16b are members of the SAME 4-dimensional degenerate (E=16, L_z=0) eigenspace
(see RESEARCH_LOG 2026-07-20) - the first excited-configuration energy level that has any
L_z=0 content at all above the E=14 ground state (E=15,17 manifolds are pure odd-L_z by a
parity argument, only even excitation levels 16,18,... can touch L_z=0). Since L_z=0
survives EVERY K-projection tested (K=3,4,6), this is the natural "next-easiest" family a
projected ansatz could fall back on if the true GS's genuinely-2D nodal structure remains
hard to reach - unlike the original trap, however, there is no single dominant member: it's
a whole manifold, so overlap with e16a/e16b individually is only a lower bound on overlap
with the manifold as a whole.

Usage: python tools/overlap_check_n6.py <checkpoint.mpack> [--hidden 64] [--out 10]
       [--lz-proj-K 3] [--samples 400000]
"""
import argparse
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

N, DIM = 6, 2


def log_env(x):
    xr = x.reshape(-1, N, DIM)
    return -0.5 * jnp.sum(xr[..., 0] ** 2 + xr[..., 1] ** 2, axis=-1)


def _det_logpsi(x, orbitals):
    """orbitals: list of N callables f(xr) -> (batch,) complex, xr = x.reshape(-1,N,DIM)."""
    xr = x.reshape(-1, N, DIM)
    cols = [f(xr) for f in orbitals]
    M = jnp.stack(cols, axis=-1).astype(jnp.complex128)  # (batch, N, N)
    sign, logdet = jnp.linalg.slogdet(M)
    return logdet + jnp.log(sign.astype(jnp.complex128)) + log_env(x)


def _r2(xr):
    return xr[..., 0] ** 2 + xr[..., 1] ** 2


def _z(xr):
    return xr[..., 0] + 1j * xr[..., 1]


def _zb(xr):
    return xr[..., 0] - 1j * xr[..., 1]


def log_gs(x):
    xr = x.reshape(-1, N, DIM)
    x1, y1 = xr[..., 0], xr[..., 1]
    orbitals = [
        lambda xr: jnp.ones_like(xr[..., 0]),
        lambda xr: xr[..., 0],
        lambda xr: xr[..., 1],
        lambda xr: xr[..., 0] ** 2,
        lambda xr: xr[..., 0] * xr[..., 1],
        lambda xr: xr[..., 1] ** 2,
    ]
    return _det_logpsi(x, orbitals)


def log_holo(x):
    orbitals = [lambda xr, k=k: _z(xr) ** k for k in range(N)]
    return _det_logpsi(x, orbitals)


def log_e16a(x):
    orbitals = [
        lambda xr: jnp.ones_like(xr[..., 0]),
        lambda xr: xr[..., 0],
        lambda xr: xr[..., 1],
        lambda xr: (1 - _r2(xr)).astype(jnp.complex128),
        lambda xr: _z(xr) ** 3,
        lambda xr: _zb(xr) ** 3,
    ]
    return _det_logpsi(x, orbitals)


def log_e16b(x):
    orbitals = [
        lambda xr: jnp.ones_like(xr[..., 0]),
        lambda xr: xr[..., 0],
        lambda xr: xr[..., 1],
        lambda xr: (1 - _r2(xr)).astype(jnp.complex128),
        lambda xr: (2 - _r2(xr)) * _z(xr),
        lambda xr: (2 - _r2(xr)) * _zb(xr),
    ]
    return _det_logpsi(x, orbitals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out", type=int, default=10)
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--lz-proj-K", type=int, default=3)
    args = ap.parse_args()

    system = System(N=N, dim=DIM, mass=1.0, potential="qho_no_inter")
    model = FermiSets(dim=DIM, N=N, rngs=nnx.Rngs(42), log=logging.getLogger(),
                      hidden_units=args.hidden, out_units=args.out,
                      lz_proj_K=args.lz_proj_K)
    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=0.35, n_chains=64, sweep_size=32)
    vstate = nk.vqs.MCState(sampler, model, n_samples=8192, seed=1, chunk_size=4096)

    with open(args.ckpt, "rb") as f:
        vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
    print(f"loaded {args.ckpt}", flush=True)
    print("(energy already known from training log; skipping vstate.expect to avoid GPU OOM)", flush=True)

    funcs = {"nn": lambda x: vstate.log_value(x),
             "gs": log_gs, "holo": log_holo, "e16a": log_e16a, "e16b": log_e16b}
    names = list(funcs)
    S = {(a, b): 0.0 + 0j for a in names for b in names}

    key = jax.random.PRNGKey(7)
    CH = 2048
    n_chunks = args.samples // CH
    shift = None
    for c in range(n_chunks):
        key, sub = jax.random.split(key)
        x = jax.random.normal(sub, (CH, N * DIM), dtype=jnp.float64)
        logq = -0.5 * jnp.sum(x**2, axis=-1)
        logs = {n: f(x) for n, f in funcs.items()}
        if shift is None:
            shift = {n: float(jnp.max(jnp.real(v))) for n, v in logs.items()}
        for a in names:
            for b in names:
                w = jnp.exp(jnp.conj(logs[a] - shift[a]) + (logs[b] - shift[b]) - logq)
                S[(a, b)] += complex(jnp.sum(w))
        print(f"chunk {c+1}/{n_chunks} done", flush=True)

    print()
    for b in ["gs", "holo", "e16a", "e16b"]:
        ov2 = abs(S[("nn", b)]) ** 2 / (S[("nn", "nn")].real * S[(b, b)].real)
        print(f"|<nn|{b}>|^2 = {ov2:.6f}")
    print()
    for pair in [("gs", "holo"), ("gs", "e16a"), ("gs", "e16b"), ("e16a", "e16b")]:
        ov2 = abs(S[pair]) ** 2 / (S[(pair[0], pair[0])].real * S[(pair[1], pair[1])].real)
        print(f"[sanity] |<{pair[0]}|{pair[1]}>|^2 = {ov2:.2e}")


if __name__ == "__main__":
    main()
