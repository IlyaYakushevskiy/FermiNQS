"""Publication-quality VMC convergence figure.

Two stacked panels sharing the iteration axis:
  top    — energy zoomed to the physically interesting window (exact GS line,
           optional trap-state line, independent-chain validation points);
  bottom — relative error |E - E_exact|/E_exact on a log scale, so both the
           initial transient and the final plateau are readable in one figure.

Usage:
  python plots/plot_publication.py <optimization_results.log> \
      --exact 5.0 --trap 6.0 --trap-label "holomorphic trap" \
      --ymax 6.2 --out plots/qho_lz_N3_convergence_pub.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


def _series(block):
    """Extract (iters, mean, sigma) from a NetKet log block, real parts only."""
    it = np.asarray(block["iters"], dtype=float)
    mean = block["Mean"]
    if isinstance(mean, dict):
        mean = mean["real"]
    mean = np.asarray([np.nan if m is None else m for m in mean], dtype=float)
    sig = block.get("Sigma")
    if isinstance(sig, dict):
        sig = sig["real"]
    sig = None if sig is None else np.asarray(
        [np.nan if s is None else s for s in sig], dtype=float)
    return it, mean, sig


def plot_publication(log_path, exact, out_path, title=None, ymin=None, ymax=None,
                     trap=None, trap_label="trap state", crit=1e-3):
    with open(log_path) as f:
        data = json.load(f)

    it, E, sig = _series(data["Energy"])
    val = data.get("Validation_Energy")
    vit = vE = vsig = None
    if val and val.get("iters"):
        vit, vE, vsig = _series(val)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

    # --- top: zoomed energy ---
    ax1.errorbar(it, E, yerr=sig, color="C0", lw=1.0, elinewidth=0.5,
                 capsize=0, alpha=0.8, label=r"training energy $\pm 1\sigma$")
    if vit is not None:
        ax1.errorbar(vit, vE, yerr=vsig, fmt="s", ms=5, color="C1",
                     capsize=3, zorder=5, label="validation (independent chain)")
    ax1.axhline(exact, color="red", ls="--", lw=1.2,
                label=rf"exact GS, $E_0 = {exact:g}$")
    if trap is not None:
        ax1.axhline(trap, color="gray", ls=":", lw=1.2,
                    label=rf"{trap_label}, $E = {trap:g}$")
    lo = ymin if ymin is not None else exact - 0.05 * abs(exact)
    hi = ymax if ymax is not None else exact + 0.30 * abs(exact)
    ax1.set_ylim(lo, hi)
    ax1.set_ylabel(r"Energy $\langle H \rangle$ [$\hbar\omega$]")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, ls=":", alpha=0.6)
    if title:
        ax1.set_title(title)

    # --- bottom: log relative error ---
    rel = np.abs(E - exact) / abs(exact)
    ax2.semilogy(it, rel, color="C0", lw=1.0, alpha=0.8)
    if vit is not None:
        vrel = np.abs(vE - exact) / abs(exact)
        ax2.semilogy(vit, vrel, "s", ms=5, color="C1", zorder=5)
    if crit is not None:
        ax2.axhline(crit, color="green", ls="--", lw=1.0,
                    label=rf"criterion $10^{{{int(np.log10(crit))}}}$")
        ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_xlabel("VMC iteration")
    ax2.set_ylabel(r"$|E - E_0|/E_0$")
    ax2.grid(True, which="both", ls=":", alpha=0.6)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log_path")
    p.add_argument("--exact", type=float, required=True,
                   help="exact/reference ground-state energy")
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--title", default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None,
                   help="top of the zoom window (e.g. 6.2)")
    p.add_argument("--trap", type=float, default=None,
                   help="energy of the trap/excited state to mark")
    p.add_argument("--trap-label", default="trap state")
    p.add_argument("--crit", type=float, default=1e-3,
                   help="relative-error criterion line (0 to disable)")
    a = p.parse_args()
    plot_publication(a.log_path, a.exact, a.out, title=a.title, ymin=a.ymin,
                     ymax=a.ymax, trap=a.trap, trap_label=a.trap_label,
                     crit=a.crit or None)
