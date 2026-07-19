"""
QUEUE.md P1 -- wall-clock complexity ablation. Forward-pass-ONLY timing (no training):
pins the actual O(K*N^2) (FermiSets + L_z projection) vs O(N^3) (bare Slater slogdet)
crossover empirically. K per N is chosen via tools/lz_margin.py's choose_K(N,
margin_target=3) -- NOT a fixed K=6 -- matching how the ablation is meant to be run
per QUEUE.md P1.

The Slater side deliberately does NOT build a trained/accurate ansatz (that's the
separate P2 SlaterNN task) -- correct forward-pass cost is all this needs, and
jnp.linalg.slogdet on a batch of N x N complex matrices IS that cost (the per-particle
orbital evaluation is O(N) and negligible next to O(N^3) at the N here).

Run: python tools/timing_ablation.py
"""
import logging
import time

import jax
import jax.numpy as jnp
from flax import nnx

from src.ansatz import FermiSets
from tools.lz_margin import choose_K

BATCH = 4096
HIDDEN_UNITS = 64
OUT_UNITS = 10
NS = [3, 6, 10, 15, 20, 30, 50, 100, 200]
N_WARMUP = 5
N_REPEAT = 50


def time_call(fn):
    for _ in range(N_WARMUP):
        out = fn()
        jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(N_REPEAT):
        out = fn()
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    return (t1 - t0) / N_REPEAT


def main():
    log = logging.getLogger("timing_ablation")
    key = jax.random.PRNGKey(0)

    print(f"{'N':>4} {'K':>4} {'FermiSets ms/fwd':>18} {'Slater slogdet ms':>20} {'ratio (Slater/Fermi)':>22}")
    for N in NS:
        K = choose_K(N, margin_target=3)

        model = FermiSets(
            dim=2, N=N, rngs=nnx.Rngs(0), log=log,
            hidden_units=HIDDEN_UNITS, out_units=OUT_UNITS, lz_proj_K=K,
        )
        key, xkey = jax.random.split(key)
        x = jax.random.normal(xkey, (BATCH, N * 2))

        @nnx.jit
        def fwd(m, x):
            return m(x)

        t_fermi = time_call(lambda: fwd(model, x))

        key, mkey1, mkey2 = jax.random.split(key, 3)
        mat = (jax.random.normal(mkey1, (BATCH, N, N))
               + 1j * jax.random.normal(mkey2, (BATCH, N, N)))

        @jax.jit
        def slogdet_fn(m):
            return jnp.linalg.slogdet(m)

        t_slater = time_call(lambda: slogdet_fn(mat))

        ratio = t_slater / t_fermi
        print(f"{N:>4} {K:>4} {t_fermi*1000:>18.4f} {t_slater*1000:>20.4f} {ratio:>22.3f}")


if __name__ == "__main__":
    main()
