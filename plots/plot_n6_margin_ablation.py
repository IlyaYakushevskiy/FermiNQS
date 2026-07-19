"""
QUEUE.md P0 writeup figure: N=6 QHO validation-energy convergence for K=3, K=4, K=6
(L_z projection margin ablation) plus the SlaterNN baseline, vs the exact GS (14.0)
and the unprojected holomorphic-trap energy (21.0). Concatenates each K's original
run with its same-day continuation (checkpoint-resumed, iteration count offset).
"""
import json
import matplotlib.pyplot as plt

RUNS = {
    "K=3 (margin=0)": [
        "outputs/2026-07-18/20-36-14/optimization_results.log",
        "outputs/2026-07-19/07-46-05/optimization_results.log",
    ],
    "K=4 (margin=1)": [
        "outputs/2026-07-18/21-48-29/optimization_results.log",
        "outputs/2026-07-19/09-00-54/optimization_results.log",
    ],
    "K=6 (margin=3)": [
        "outputs/2026-07-16/09-32-11/optimization_results.log",
        "outputs/2026-07-16/11-45-06/optimization_results.log",
    ],
    "SlaterNN (no projection)": [
        "outputs/2026-07-19/11-15-58/optimization_results.log",
    ],
}

# fixed hue order per the palette (categorical slots 1,2,3,4)
COLORS = {
    "K=3 (margin=0)": "#2a78d6",
    "K=4 (margin=1)": "#008300",
    "K=6 (margin=3)": "#e87ba4",
    "SlaterNN (no projection)": "#eda100",
}

EXACT_GS = 14.0
TRAP_E = 21.0

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

for label, paths in RUNS.items():
    iters_all, mean_all = [], []
    offset = 0
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        ve = data["Validation_Energy"]
        its = [i + offset for i in ve["iters"]]
        means = ve["Mean"]["real"]
        iters_all.extend(its)
        mean_all.extend(means)
        offset = iters_all[-1] if iters_all else 0
    ax.plot(iters_all, mean_all, color=COLORS[label], linewidth=2, label=label,
             solid_capstyle="round")

ax.axhline(EXACT_GS, color="#52514e", linestyle="--", linewidth=1.5, zorder=0)
ax.text(ax.get_xlim()[1] if False else 5, EXACT_GS + 0.3, "exact GS = 14.0",
        color="#52514e", fontsize=9)
ax.axhline(TRAP_E, color="#8a8a86", linestyle=":", linewidth=1.5, zorder=0)
ax.text(5, TRAP_E + 0.3, "holomorphic trap = 21.0", color="#8a8a86", fontsize=9)

ax.set_xlabel("VMC iteration (cumulative, original + checkpoint-resumed continuation)")
ax.set_ylabel("Validation energy")
ax.set_title("N=6 QHO: L_z-projection margin ablation vs SlaterNN baseline")
ax.legend(loc="upper right", frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("plots/n6_margin_ablation_vs_slater.png")
print("saved plots/n6_margin_ablation_vs_slater.png")
