"""Where the network walks: single-particle density snapshots along a training run,
next to the analytic states it could be walking towards.

One particle is swept over the plane while the other N-1 are pinned at fixed
(deliberately asymmetric) positions, and |psi|^2 is plotted on that slice; each
panel is normalised to its own maximum, so the figure compares *shapes*, i.e.
nodal structure, not amplitudes. The top row is the network at a sequence of
checkpoints, the bottom row the analytic references of Section "the lazy trap":
the true ground state det[1, x, y] and the holomorphic Vandermonde trap
det[1, z, z^2] (the Laughlin-like state).

The antiholomorphic mirror is deliberately absent: it is the exact complex
conjugate of the holomorphic state, so |psi|^2 is identical for the two at every
configuration and no density plot can separate them (that separation needs the
phase, or Re psi as in the mirror-flip figure). Its fidelity is still reported
on stdout, since the two do split the network's weight.

The energy printed under each snapshot is read from the run's own
optimization_results.log (validation chain where available), and the fidelities
are estimated by importance sampling from N(0,1), the same estimator as
tools/overlap_check.py.

Usage:
  python plots/plot_density_walk.py outputs/2026-07-14/12-20-48 \
      --steps 0 50 150 350 --out thesis/n3_trap_density_walk.png
  python plots/plot_density_walk.py outputs/2026-07-14/15-36-30 \
      --steps 0 50 150 750 --lz-proj-K 6 --out thesis/n3_lz_density_walk.png
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import jax
import jax.numpy as jnp
import flax
from flax import nnx
import netket as nk

from src.system import System
from src.ansatz import FermiSets
from tools.overlap_check import log_gs, log_holo, log_antiholo

N, DIM = 3, 2
GRID = 120
EXTENT = 3.0
# Chosen by scanning pinnings for the largest L1 separation between the ground-state
# and holomorphic-trap densities on this slice (0.96 of a possible 2.0), so that the
# reader can see by eye which reference the network has matched. Asymmetric on purpose.
FIXED = np.array([[1.69, 0.62], [0.46, 0.39]])

REFS = [("true ground state", r"$\det[1,x,y]$, $E_0=5$", log_gs),
        ("holomorphic trap", r"$\det[1,z,z^2]$, $E=6$", log_holo)]
# reported numerically but not drawn: identical density to the holomorphic state
EXTRA_FID = [("antiholomorphic mirror", log_antiholo)]


def _grid_configs():
    ax = np.linspace(-EXTENT, EXTENT, GRID)
    X, Y = np.meshgrid(ax, ax)
    swept = np.stack([X.ravel(), Y.ravel()], axis=-1)
    cfg = np.tile(np.vstack([[[0.0, 0.0]], FIXED]), (swept.shape[0], 1, 1))
    cfg[:, 0, :] = swept
    return X, Y, jnp.asarray(cfg.reshape(-1, N * DIM))


def _density(log_psi):
    """log psi on the grid -> |psi|^2 normalised to unit maximum."""
    lm = jnp.real(log_psi)
    d = jnp.exp(2.0 * (lm - jnp.nanmax(lm)))
    return np.asarray(d).reshape(GRID, GRID)


def _fidelities(vstate, n_samples, seed):
    """|<psi|phi>|^2 for each analytic reference, importance-sampled from N(0,1)."""
    x = jax.random.normal(jax.random.PRNGKey(seed), (n_samples, N * DIM))
    log_q = -0.5 * jnp.sum(x**2, axis=-1)
    logs = {"nn": vstate.log_value(x)}
    for key, _, fn in REFS:
        logs[key] = fn(x)
    for key, fn in EXTRA_FID:
        logs[key] = fn(x)

    def braket(a, b):
        return jnp.mean(jnp.exp(jnp.conj(logs[a]) + logs[b] - log_q))

    nn_nn = braket("nn", "nn").real
    out = {}
    for key in [k for k, _, _ in REFS] + [k for k, _ in EXTRA_FID]:
        out[key] = float(abs(braket("nn", key)) ** 2 / (nn_nn * braket(key, key).real))
    return out


def _energy_at(run_dir, step):
    """Validation energy at `step` from the run's own log, else the training estimate."""
    path = os.path.join(run_dir, "optimization_results.log")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    for block in ("Validation_Energy", "Energy"):
        b = data.get(block)
        if not b or not b.get("iters"):
            continue
        it = np.asarray(b["iters"], dtype=float)
        mean = b["Mean"]
        if isinstance(mean, dict):
            mean = mean["real"]
        j = int(np.argmin(np.abs(it - step)))
        if abs(it[j] - step) <= 1:
            return float(np.real(mean[j]))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="run directory containing checkpoints/")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--out-units", type=int, default=10)
    ap.add_argument("--lz-proj-K", type=int, default=0,
                    help="must match the run's ansatz.lz_proj_K")
    ap.add_argument("--fid-samples", type=int, default=200_000)
    ap.add_argument("--no-fidelity", action="store_true")
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()

    system = System(N=N, dim=DIM, mass=1.0, potential="qho_no_inter")
    model = FermiSets(dim=DIM, N=N, rngs=nnx.Rngs(42), log=logging.getLogger(),
                      hidden_units=args.hidden, out_units=args.out_units,
                      lz_proj_K=args.lz_proj_K)
    sampler = nk.sampler.MetropolisGaussian(system.hi, sigma=0.35, n_chains=64,
                                            sweep_size=32)
    vstate = nk.vqs.MCState(sampler, model, n_samples=4096, seed=1, chunk_size=4096)

    X, Y, cfg = _grid_configs()

    panels = []
    for step in args.steps:
        ckpt = os.path.join(args.run_dir, "checkpoints", f"step_{step}.mpack")
        if not os.path.exists(ckpt):
            raise SystemExit(f"missing checkpoint: {ckpt}")
        with open(ckpt, "rb") as f:
            vstate.variables = flax.serialization.from_bytes(vstate.variables, f.read())
        dens = _density(vstate.log_value(cfg))
        E = _energy_at(args.run_dir, step)
        fid = None if args.no_fidelity else _fidelities(vstate, args.fid_samples, 7)
        panels.append((step, dens, E, fid))
        msg = f"step {step:4d}  E = {E if E is None else round(E, 4)}"
        if fid:
            msg += "".join(f"   F[{k}] = {v:.3f}" for k, v in fid.items())
        print(msg)

    ref_dens = [_density(fn(cfg)) for _, _, fn in REFS]

    plt.rcParams.update({
        "text.usetex": not args.no_usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
    })

    ncol = max(len(panels), len(REFS) + 1)
    fig = plt.figure(figsize=(3.0 * ncol, 6.6))
    gs = gridspec.GridSpec(2, ncol, figure=fig, hspace=0.28, wspace=0.16)
    levels = np.linspace(0.0, 1.0, 51)

    def draw(ax, dens, title, subtitle):
        m = ax.contourf(X, Y, dens, levels=levels, cmap="magma", vmin=0, vmax=1)
        ax.scatter(FIXED[:, 0], FIXED[:, 1], marker="*", s=110, color="white",
                   edgecolor="black", linewidth=0.4, zorder=5)
        ax.set_title(title + "\n" + r"{\small " + subtitle + "}"
                     if not args.no_usetex else f"{title}\n{subtitle}")
        ax.set_aspect("equal")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.grid(True, alpha=0.15, color="white")
        return m

    for i, (step, dens, E, fid) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        sub = "" if E is None else rf"$E = {E:.3f}$"
        if fid:
            sub += rf", $\mathcal{{F}}_{{\rm holo}} = {fid['holomorphic trap']:.2f}$"
            sub += rf", $\mathcal{{F}}_{{\rm GS}} = {fid['true ground state']:.2f}$"
        mappable = draw(ax, dens, rf"network, iteration ${step}$", sub)
        if i == 0:
            ax.set_ylabel(r"$y_1$")

    for i, ((name, formula, _), dens) in enumerate(zip(REFS, ref_dens)):
        ax = fig.add_subplot(gs[1, i])
        draw(ax, dens, name, formula)
        ax.set_xlabel(r"$x_1$")
        if i == 0:
            ax.set_ylabel(r"$y_1$")

    cax = fig.add_subplot(gs[1, len(REFS)])
    cax.axis("off")
    cb = fig.colorbar(mappable, ax=cax, fraction=0.35, aspect=12,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label(r"$|\psi|^2$ (panel maximum $=1$)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
