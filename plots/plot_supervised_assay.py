"""Supervised expressivity assay: can the ansatz be FIT to the exact ground state?

Two panels:
  left  — phase loss against training step, one curve per run. The phase term runs
          from 0 (sign structure reproduced) to 1 (uncorrelated with the target), so
          this panel shows directly whether a fit is descending or pinned.
  right — the converged phase loss against N. Open shells (N=4,5) are shown as two
          points, one per member of the degenerate ground-state manifold, since the
          aufbau filling picks one member arbitrarily and a representability claim
          must not rest on that choice.

Usage:
  python plots/plot_supervised_assay.py --out thesis/supervised_assay.png
"""
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# (label, N, log file, style) — grouped by N, one entry per target/variant
RUNS = [
    (r"$N=3$",                    3, "logs/pretrain_n3_h64_control.log",   "-"),
    (r"$N=4$, $(0,2)$",           4, "logs/pretrain_n4_h64_aufbau.log",    "-"),
    (r"$N=4$, $(1,1)$",           4, "logs/pretrain_n4_h64_m11.log",       "--"),
    (r"$N=5$, $(0,2)(1,1)$",      5, "logs/pretrain_n5_h64_aufbau.log",    "-"),
    (r"$N=5$, $(0,2)(2,0)$",      5, "logs/pretrain_n5_h64_alt.log",       "--"),
    (r"$N=6$, hidden $64$",       6, "logs/pretrain_n6_h64.log",           "-"),
    (r"$N=6$, hidden $128$",      6, "logs/pretrain_n6_h128.log",          ":"),
    # the class the METHOD uses: the raw-ansatz rows above do not bound it, since
    # P_K(psi) can be the ground state while psi itself is nowhere near it
    (r"$N=3$, projected $K=6$",   3, "logs/pretrain_n3_h64_K6_control.log", "-."),
    (r"$N=6$, projected $K=6$",   6, "logs/pretrain_n6_h64_K6.log",         "-."),
]
# runs that vary WIDTH rather than the degenerate member — drawn differently on the
# right panel so "two points" never silently means two different things
WIDTH_VARIANT = {r"$N=6$, hidden $128$"}
PROJECTED = {r"$N=3$, projected $K=6$", r"$N=6$, projected $K=6$"}

STEP_RE = re.compile(r"step\s+(\d+)\s+loss=([\d.eE+-]+)\s+amp=([\d.eE+-]+)\s+phase=([\d.eE+-]+)")


def read_log(path):
    steps, amp, phase = [], [], []
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                amp.append(float(m.group(3)))
                phase.append(float(m.group(4)))
    return np.array(steps), np.array(amp), np.array(phase)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-usetex", action="store_true")
    args = ap.parse_args()

    plt.rcParams.update({
        "text.usetex": not args.no_usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
    })

    cmap = {3: "C2", 4: "C0", 5: "C1", 6: "C3"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.55, 1]})

    summary = {}
    for label, N, path, ls in RUNS:
        if not os.path.exists(path):
            print(f"skip (missing): {path}")
            continue
        steps, amp, phase = read_log(path)
        if len(steps) == 0:
            continue
        ax1.plot(steps, phase, ls, color=cmap[N], lw=1.4, label=label)
        final = float(np.mean(phase[-3:]))          # tail mean: the last points are noisy
        summary.setdefault(N, []).append((label, final, float(np.mean(amp[-3:]))))
        print(f"{label:24s} phase(final,tail-mean)={final:.3f}  amp={np.mean(amp[-3:]):.3f}")

    ax1.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax1.text(200, 1.012, "uncorrelated with the target", fontsize=9, color="gray")
    ax1.set_xlabel("supervised step")
    ax1.set_ylabel("phase loss")
    ax1.set_ylim(0, 1.10)
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
               framealpha=0.9, handlelength=2.4)

    for N, entries in sorted(summary.items()):
        members = [e for e in entries
                   if e[0] not in WIDTH_VARIANT and e[0] not in PROJECTED]
        widths = [e for e in entries if e[0] in WIDTH_VARIANT]
        projected = [e for e in entries if e[0] in PROJECTED]
        vals = [e[1] for e in members]
        if len(vals) > 1:
            ax2.plot([N, N], [min(vals), max(vals)], "-", color=cmap[N], lw=1.6, zorder=1)
        ax2.plot([N] * len(vals), vals, "o", color=cmap[N], ms=8, zorder=2,
                 mfc="white" if len(vals) > 1 else cmap[N], mew=1.6)
        for _, v, _ in widths:
            ax2.plot([N], [v], "P", color=cmap[N], ms=9, zorder=3)
        for _, v, _ in projected:
            ax2.plot([N], [v], "s", color=cmap[N], ms=8, mfc="none", mew=2.0, zorder=4)
    xs = sorted(summary)
    ax2.plot(xs, [np.mean([e[1] for e in summary[N]
                           if e[0] not in WIDTH_VARIANT and e[0] not in PROJECTED])
                  for N in xs], "-", color="gray", lw=1.0, zorder=0)
    from matplotlib.lines import Line2D
    ax2.legend(handles=[
        Line2D([], [], ls="", marker="o", color="gray", label="ground-state target"),
        Line2D([], [], ls="", marker="o", mfc="white", mec="gray", color="gray",
               label="2nd degenerate member"),
        Line2D([], [], ls="", marker="P", color="gray", label=r"hidden $128$"),
        Line2D([], [], ls="", marker="s", mfc="none", mec="gray", color="gray",
               mew=2.0, label=r"$L_z$-projected, $K=6$"),
    ], loc="lower right", framealpha=0.9)
    ax2.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax2.set_xlabel("$N$")
    ax2.set_ylabel("converged phase loss")
    ax2.set_xticks(xs)
    ax2.set_ylim(0, 1.10)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.set_title(r"$N=4,5$ are open shells: both members shown", fontsize=9)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=250)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
