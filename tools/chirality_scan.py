"""Verify the 'petal-flip' hypothesis: across the energy spikes of a from-scratch
FermiSets run, does the state flip chirality between the two DEGENERATE trap states

  holo      det[1, z, z^2, ..., z^(N-1)] * exp(-sum r^2/2)   E = N(N+1)/2, L_z = +N(N-1)/2
  antiholo  det[1, zb, zb^2, ..., zb^(N-1)] * exp(...)        E = N(N+1)/2, L_z = -N(N-1)/2

For every checkpoint in a run directory, importance-sample |<psi|holo>|^2 and
|<psi|antiholo>|^2 (same estimator as tools/overlap_check.py) and also the local
angular-momentum expectation <L_z> = < -i sum_k (x_k d_yk - y_k d_xk) ln psi >.
Print a per-checkpoint table; the hypothesis predicts holo/antiholo dominance swaps
(and <L_z> changes sign) at the iterations where the energy spikes.

Usage:
  python tools/chirality_scan.py <run_dir> --N 4 --hidden 256 --out 20 [--lz-proj-K 0]
"""
import argparse, glob, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import netket as nk
import jax
import jax.numpy as jnp
import flax
from flax import nnx

from src.system import System
from src.ansatz import FermiSets


def make_orbital_logdet(N, conj):
    """log det of the Vandermonde-style matrix [z^0..z^(N-1)] (holo) or conj (antiholo),
    times the isotropic trap envelope exp(-sum r^2 / 2)."""
    def f(x):
        xr = x.reshape(-1, N, 2)
        z = xr[..., 0] + ((-1j) if conj else (1j)) * xr[..., 1]
        cols = [z ** k for k in range(N)]
        M = jnp.stack(cols, axis=-1)  # (batch, N, N)
        env = -0.5 * jnp.sum(xr[..., 0] ** 2 + xr[..., 1] ** 2, axis=-1)
        return jnp.log(jnp.linalg.det(M)) + env
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--out", type=int, default=20)
    ap.add_argument("--lz-proj-K", type=int, default=0)
    ap.add_argument("--samples", type=int, default=300_000)
    args = ap.parse_args()
    N = args.N

    system = System(N=N, dim=2, mass=1.0, potential="qho_no_inter")
    model = FermiSets(dim=2, N=N, rngs=nnx.Rngs(42), log=logging.getLogger(),
                      hidden_units=args.hidden, out_units=args.out, lz_proj_K=args.lz_proj_K)
    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=0.35, n_chains=64, sweep_size=32)
    vstate = nk.vqs.MCState(sampler, model, n_samples=8192, seed=1, chunk_size=4096)

    log_holo = make_orbital_logdet(N, conj=False)
    log_anti = make_orbital_logdet(N, conj=True)

    # local L_z estimator: O_Lz(R) = -i sum_k (x_k d/dy_k - y_k d/dx_k) ln psi(R)
    graphdef, state = nnx.split(model)
    def logpsi_single(params, x):
        m = nnx.merge(graphdef, params)
        return m(x.reshape(1, -1))[0]
    def lz_local(params, x):
        g = jax.grad(lambda xx: logpsi_single(params, xx), holomorphic=False)
        # complex output -> use jacrev on real+imag; simpler: grad of real and imag parts
        gr = jax.jacrev(lambda xx: jnp.real(logpsi_single(params, xx)))(x)
        gi = jax.jacrev(lambda xx: jnp.imag(logpsi_single(params, xx)))(x)
        gz = gr + 1j * gi                      # d ln psi / d coord
        xr = x.reshape(N, 2)
        gzr = gz.reshape(N, 2)
        # -i sum (x d_y - y d_x) ln psi
        val = -1j * jnp.sum(xr[:, 0] * gzr[:, 1] - xr[:, 1] * gzr[:, 0])
        return val

    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "checkpoints", "step_*.mpack")),
                   key=lambda p: int(p.split("step_")[1].split(".")[0]))
    # energies from the log
    with open(os.path.join(args.run_dir, "optimization_results.log")) as f:
        d = json.load(f)
    Emean = d["Energy"]["Mean"]; Emean = Emean["real"] if isinstance(Emean, dict) else Emean
    Eiters = np.array(d["Energy"]["iters"]); Emean = np.array([np.nan if v is None else v for v in Emean])

    key0 = jax.random.PRNGKey(7)
    CH = 50_000
    print(f"{'step':>5} {'E(log)':>8} {'|<holo>|^2':>11} {'|<anti>|^2':>11} {'<L_z>':>8} {'chirality':>10}")
    rows = []
    for cp in ckpts:
        step = int(cp.split("step_")[1].split(".")[0])
        with open(cp, "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
        _, st = nnx.split(vstate.model)
        # overlaps via importance sampling
        S = {k: 0.0 + 0j for k in ["nn_nn", "h_h", "a_a", "nn_h", "nn_a"]}
        key = key0
        shift = None
        for c in range(args.samples // CH):
            key, sub = jax.random.split(key)
            x = jax.random.normal(sub, (CH, N * 2), dtype=jnp.float64)
            logq = -0.5 * jnp.sum(x ** 2, axis=-1)
            lnn = vstate.log_value(x); lh = log_holo(x); la = log_anti(x)
            if shift is None:
                shift = {"nn": float(jnp.max(jnp.real(lnn))),
                         "h": float(jnp.max(jnp.real(lh))),
                         "a": float(jnp.max(jnp.real(la)))}
            lnn = lnn - shift["nn"]; lh = lh - shift["h"]; la = la - shift["a"]
            S["nn_nn"] += complex(jnp.sum(jnp.exp(jnp.conj(lnn) + lnn - logq)))
            S["h_h"] += complex(jnp.sum(jnp.exp(jnp.conj(lh) + lh - logq)))
            S["a_a"] += complex(jnp.sum(jnp.exp(jnp.conj(la) + la - logq)))
            S["nn_h"] += complex(jnp.sum(jnp.exp(jnp.conj(lnn) + lh - logq)))
            S["nn_a"] += complex(jnp.sum(jnp.exp(jnp.conj(lnn) + la - logq)))
        ov_h = abs(S["nn_h"]) ** 2 / (S["nn_nn"].real * S["h_h"].real)
        ov_a = abs(S["nn_a"]) ** 2 / (S["nn_nn"].real * S["a_a"].real)
        # <L_z> over a fresh |psi|^2 sample from the vstate
        samp = vstate.samples.reshape(-1, N * 2)
        idx = jax.random.choice(jax.random.PRNGKey(step), samp.shape[0], (2048,), replace=False)
        xs = samp[idx]
        lz_vals = jax.vmap(lambda xx: lz_local(st, xx))(xs)
        lz = float(jnp.real(jnp.mean(lz_vals)))
        E = float(Emean[Eiters == step][0]) if np.any(Eiters == step) else np.nan
        chir = "holo+" if ov_h > ov_a else "anti-"
        print(f"{step:>5} {E:>8.3f} {ov_h:>11.4f} {ov_a:>11.4f} {lz:>8.2f} {chir:>10}")
        rows.append((step, E, ov_h, ov_a, lz))
    np.save(os.path.join(args.run_dir, "chirality_scan.npy"), np.array(rows))
    print(f"\nsaved {os.path.join(args.run_dir, 'chirality_scan.npy')}")


if __name__ == "__main__":
    main()
