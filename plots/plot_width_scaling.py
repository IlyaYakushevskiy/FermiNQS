"""N=4 QHO width-scaling summary figure for the Results section.

For each network width (hidden_units) we harvested every FermiSets QHO run in the
backlog and recorded, per run, whether it (i) stayed in the Laughlin trap (E~10),
(ii) collapsed via SR instability (E crosses below the variational bound / NaN),
or (iii) stably escaped the trap and held a physical energy < trap. This figure
plots, per width, the *record stable energy* (deepest physically valid final energy
that held without collapsing) together with the exact GS and the trap energy.

The message: capacity breaks the trap only partially (16->64), the record saturates
around E~8.3 at 64 units, and larger widths (128-512, incl. 80 GB A100 runs) do not
push closer to the true GS -- they become collapse-dominated or regress to the trap.
Data are hard-coded from the harvest (scratchpad/harvest_scaling2.py) so the figure
is reproducible without re-reading the whole outputs tree.
"""
import numpy as np
import matplotlib.pyplot as plt

EXACT_GS = 8.0
TRAP = 10.0

# width : (n_runs, n_escaped_stable, best_stable_final, collapse_fraction)
# best_stable_final = NaN where no seed escaped the trap without collapsing.
data = {
    16:  dict(n=2,  esc=0,  best=np.nan, coll=0.0),
    32:  dict(n=3,  esc=0,  best=np.nan, coll=2/3),
    64:  dict(n=46, esc=12, best=8.36,   coll=5/46),
    128: dict(n=3,  esc=0,  best=np.nan, coll=2/3),
    256: dict(n=17, esc=1,  best=8.71,   coll=11/17),
    300: dict(n=4,  esc=0,  best=np.nan, coll=1/4),
    512: dict(n=1,  esc=0,  best=np.nan, coll=0.0),
}

widths = sorted(data)
best = np.array([data[w]["best"] for w in widths])
coll = np.array([data[w]["coll"] for w in widths])
nrun = np.array([data[w]["n"] for w in widths])

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], "font.size": 12,
    "axes.labelsize": 14, "legend.fontsize": 11,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
})

fig, ax = plt.subplots(figsize=(8, 5))

# reference lines
ax.axhline(EXACT_GS, color="red", ls="--", lw=1.3, label=r"exact GS, $E_0 = 8$")
ax.axhline(TRAP, color="gray", ls=":", lw=1.3, label=r"Laughlin trap, $E = 10$")

# widths that produced a stable escape: plot the record energy
esc_mask = np.isfinite(best)
ax.plot(np.array(widths)[esc_mask], best[esc_mask], "o-", color="C0", ms=9,
        lw=1.6, zorder=5, label="record stable energy")

# widths where NO seed escaped: mark on the trap line
trap_mask = ~esc_mask
ax.scatter(np.array(widths)[trap_mask], np.full(trap_mask.sum(), TRAP),
           marker="x", s=90, color="gray", zorder=6,
           label="no stable escape (all trapped/collapsed)")

# annotate collapse fraction above each point
for w in widths:
    y = data[w]["best"] if np.isfinite(data[w]["best"]) else TRAP
    ax.annotate(rf"{int(round(data[w]['coll']*data[w]['n']))}/{data[w]['n']} coll.",
                (w, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="C3")

ax.set_xscale("log", base=2)
ax.set_xticks(widths)
ax.set_xticklabels([str(w) for w in widths])
ax.set_xlabel(r"network width (hidden units)")
ax.set_ylabel(r"Energy $\langle H \rangle$ [$\hbar\omega$]")
ax.set_ylim(7.6, 10.4)
ax.set_title(r"N=4 QHO: capacity does not scale to convergence")
ax.legend(loc="lower right", framealpha=0.95)
ax.grid(True, which="both", ls=":", alpha=0.5)

out = "plots/n4_width_scaling.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
print(f"wrote {out}")
