"""The opening figure of the Results chapter: the N=3 from-scratch trap.

Three stacked panels sharing the iteration axis, all from one run
(outputs/2026-07-14/12-20-48, the canonical N=3 benchmark started from random
parameters):
  top    — energy, zoomed so both the exact GS (E=5) and the trap (E=6) are on
           screen, with the independent-chain validation points overlaid;
  middle — variance of the local energy on a log scale: it collapses by two
           orders of magnitude while the run sits on the *wrong* state, which is
           the point of the figure (low variance is not evidence of the GS);
  bottom — sampler diagnostics (acceptance and R_hat), to pre-empt the reading
           that the plateau is a sampling artefact.

Usage:
  python plots/plot_n3_trap.py outputs/2026-07-14/12-20-48/optimization_results.log \
      --out thesis/n3_trap_convergence.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plots.plot_publication import _series


def plot_n3_trap(log_path, out_path, exact=5.0, trap=6.0, usetex=True):
    with open(log_path) as f:
        data = json.load(f)

    it, E, sig = _series(data["Energy"])
    var = np.asarray(data["Energy"]["Variance"], dtype=float)
    rhat = np.asarray(data["Energy"]["R_hat"], dtype=float)

    val = data.get("Validation_Energy")
    vit = vE = vsig = None
    if val and val.get("iters"):
        vit, vE, vsig = _series(val)

    acc_it = np.asarray(data["acceptance"]["iters"], dtype=float)
    acc = np.asarray(data["acceptance"]["value"], dtype=float)

    plt.rcParams.update({
        "text.usetex": usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.2, 1.0], "hspace": 0.10})

    # --- top: energy ---
    ax1.errorbar(it, E, yerr=sig, color="C0", lw=1.0, elinewidth=0.5,
                 capsize=0, alpha=0.85, label=r"training energy $\pm 1\sigma$")
    if vit is not None:
        ax1.errorbar(vit, vE, yerr=vsig, fmt="s", ms=5, color="C1",
                     capsize=3, zorder=5, label="validation (independent chain)")
    ax1.axhline(trap, color="gray", ls=":", lw=1.4,
                label=rf"holomorphic trap, $E = {trap:g}$")
    ax1.axhline(exact, color="red", ls="--", lw=1.4,
                label=rf"exact GS, $E_0 = {exact:g}$")
    ax1.set_ylim(exact - 0.25, 7.8)
    ax1.set_ylabel(r"Energy $\langle H \rangle$ [$\hbar\omega$]")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, ls=":", alpha=0.6)

    # annotate the gap that never closes
    ax1.annotate("", xy=(it[-1] * 0.62, exact), xytext=(it[-1] * 0.62, trap),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.0))
    ax1.text(it[-1] * 0.64, 0.5 * (exact + trap),
             r"$\Delta E = 1\,\hbar\omega$" "\n" r"never closes",
             va="center", ha="left", fontsize=10)

    # --- middle: local-energy variance ---
    ax2.semilogy(it, var, color="C2", lw=1.0)
    ax2.set_ylabel(r"$\mathrm{Var}[E_{\mathrm{loc}}]$")
    ax2.grid(True, which="both", ls=":", alpha=0.6)
    ax2.text(0.98, 0.9,
             "variance falls two decades onto the wrong eigenstate",
             transform=ax2.transAxes, ha="right", va="top", fontsize=10)

    # --- bottom: sampler diagnostics ---
    ax3.plot(acc_it, acc, color="C4", lw=1.0, label="acceptance")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("acceptance")
    ax3.set_xlabel("VMC iteration")
    ax3.grid(True, ls=":", alpha=0.6)
    ax3b = ax3.twinx()
    ax3b.plot(it, rhat, color="C5", lw=1.0, label=r"$\hat R$")
    ax3b.axhline(1.05, color="C5", ls=":", lw=0.9)
    ax3b.set_ylim(1.0, max(1.2, float(np.nanmax(rhat)) * 1.02))
    ax3b.set_ylabel(r"$\hat R$")
    lines = ax3.get_lines()[:1] + ax3b.get_lines()[:1]
    ax3.legend(lines, [l.get_label() for l in lines],
               loc="upper right", framealpha=0.9, ncol=2)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")

    # numbers quoted in the text, so they cannot drift from the figure
    tail = slice(-100, None)
    print(f"final training E   = {E[-1]:.4f}")
    print(f"plateau mean E     = {np.nanmean(E[tail]):.4f} "
          f"+- {np.nanmean(sig[tail]):.4f} (last 100 iters)")
    print(f"variance start/end = {var[0]:.3g} -> {np.nanmean(var[tail]):.3g}")
    print(f"R_hat range (tail) = {np.nanmin(rhat[tail]):.3f}-{np.nanmax(rhat[tail]):.3f}")
    print(f"acceptance (tail)  = {np.nanmean(acc[-100:]):.3f}")
    if vit is not None:
        print(f"last validation    = {vE[-1]:.4f} +- {vsig[-1]:.4f} at iter {vit[-1]:.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log_path")
    p.add_argument("--out", required=True)
    p.add_argument("--exact", type=float, default=5.0)
    p.add_argument("--trap", type=float, default=6.0)
    p.add_argument("--no-usetex", action="store_true")
    a = p.parse_args()
    plot_n3_trap(a.log_path, a.out, exact=a.exact, trap=a.trap,
                 usetex=not a.no_usetex)
