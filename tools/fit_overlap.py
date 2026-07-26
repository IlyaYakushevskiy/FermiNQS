"""Squared overlap between a supervised-fit checkpoint and the exact ground state.

The supervised assay reports amplitude/phase loss, which localises WHERE a fit fails
(modulus vs sign structure) but is the fit's own objective rather than an observable.
This reports the physical quantity instead:

    F = |<psi_NN | psi_GS>|^2 / (<psi_NN|psi_NN> <psi_GS|psi_GS>)

for any N, with the exact non-interacting ground state taken from the same aufbau
Slater determinant the fit targeted (`tools.pretrain_hf.make_log_hf`).

Estimator: importance sampling from q = N(0,1)^(N*dim), the same scheme as
tools/overlap_check.py. Reported with a bootstrap error, because the estimator's
variance grows with dimension and a silently bad estimate at N=6 would be worse than
no estimate at all. If the bootstrap error is large compared with the value, say so
rather than quoting the number.

NOTE on energy: do NOT read the VMC energy of a supervised fit as representability.
The masked fit ignores the near-collision region, so fitted states routinely show
sigma^2 of 1e3-1e11 (RESEARCH_LOG 2026-07-24, 2026-07-26).

Usage:
  python tools/fit_overlap.py --N 6 --ckpt outputs/pretrained/hf_N6_2d_h64.mpack
  python tools/fit_overlap.py --all
"""
import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import netket as nk  # noqa: F401  x64 side effect
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import nnx

from src.system import System
from src.ansatz import FermiSets
from tools.pretrain_hf import make_log_hf

DIM = 2


def fit_overlap(N, ckpt, hidden=64, out_units=10, lz_proj_K=0, pair_sig_hidden=0,
                n_samples=400_000, chunk=50_000, seed=0, n_boot=200, orbitals=None):
    # MUST match the occupation the checkpoint was FIT to. At an open shell the aufbau
    # default is a different, orthogonal member of the degenerate manifold, and comparing
    # against it returns F = 0 for a perfectly good fit.
    log_hf, _ = make_log_hf(N, orbitals=orbitals)
    model = FermiSets(dim=DIM, N=N, rngs=nnx.Rngs(42), log=logging.getLogger(),
                      hidden_units=hidden, out_units=out_units,
                      lz_proj_K=lz_proj_K, pair_sig_hidden=pair_sig_hidden)
    system = System(N=N, dim=DIM, mass=1.0, potential="qho_no_inter")
    vs = nk.vqs.MCState(nk.sampler.MetropolisGaussian(system.hi, sigma=0.35, n_chains=16),
                        model, n_samples=512, seed=1)
    with open(ckpt, "rb") as f:
        loaded = flax.serialization.from_bytes(vs.variables, f.read())
    nnx.update(model, loaded["params"])

    # accumulate the three Monte Carlo sums in chunks; keep per-chunk values for bootstrap
    nn_nn, gs_gs, nn_gs = [], [], []
    key = jax.random.PRNGKey(seed)
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        key, sub = jax.random.split(key)
        x = jax.random.normal(sub, (m, N * DIM), dtype=jnp.float64)
        log_q = -0.5 * jnp.sum(x**2, axis=-1)
        a = model(x)                      # log psi_NN
        b, _ = log_hf(x)                  # log psi_GS
        nn_nn.append(np.asarray(jnp.exp(jnp.conj(a) + a - log_q).real))
        gs_gs.append(np.asarray(jnp.exp(jnp.conj(b) + b - log_q).real))
        nn_gs.append(np.asarray(jnp.exp(jnp.conj(a) + b - log_q)))
        done += m
    nn_nn = np.concatenate(nn_nn); gs_gs = np.concatenate(gs_gs); nn_gs = np.concatenate(nn_gs)

    def F(idx):
        return abs(nn_gs[idx].mean())**2 / (nn_nn[idx].mean() * gs_gs[idx].mean())

    all_idx = np.arange(len(nn_nn))
    val = F(all_idx)
    rng = np.random.default_rng(1)
    boot = np.array([F(rng.integers(0, len(all_idx), len(all_idx))) for _ in range(n_boot)])
    return float(val), float(boot.std())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int)
    ap.add_argument("--ckpt")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out-units", type=int, default=10)
    ap.add_argument("--lz-proj-K", type=int, default=0)
    ap.add_argument("--pair-sig-hidden", type=int, default=0)
    ap.add_argument("--samples", type=int, default=400_000)
    ap.add_argument("--orbitals", default=None,
                    help='occupation the checkpoint was fit to, e.g. "0,0;0,1;1,0;1,1"')
    ap.add_argument("--all", action="store_true",
                    help="scan outputs/pretrained/hf_N*_2d_h64*.mpack (plain fits only)")
    args = ap.parse_args()

    if args.all:
        rows = []
        for ck in sorted(glob.glob("outputs/pretrained/hf_N*_2d_h64*.mpack")):
            base = os.path.basename(ck)
            if any(t in base for t in ("_ps", "_K", "_p32", "_bf")):
                continue  # architecture / projected variants need their own flags
            N = int(base.split("_")[1][1:])
            v, e = fit_overlap(N, ck, hidden=args.hidden, out_units=args.out_units,
                               n_samples=args.samples)
            rows.append((base, N, v, e))
            print(f"{base:44s} N={N}  F = {v:.4f} +- {e:.4f}")
        return

    occ = None
    if args.orbitals:
        occ = [tuple(int(v) for v in o.split(",")) for o in args.orbitals.split(";")]
    v, e = fit_overlap(args.N, args.ckpt, hidden=args.hidden, out_units=args.out_units,
                       lz_proj_K=args.lz_proj_K, pair_sig_hidden=args.pair_sig_hidden,
                       n_samples=args.samples, orbitals=occ)
    print(f"F = |<psi_fit|psi_GS>|^2 = {v:.4f} +- {e:.4f}   ({args.ckpt})")


if __name__ == "__main__":
    main()
