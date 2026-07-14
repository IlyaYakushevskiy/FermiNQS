"""
Stage 0 of the benchmark protocol (see CLAUDE.md, "Benchmark experiment"):
structural sanity checks for the FermiSets 2D ansatz. No training, runs in <1 min.

Every check here tests a property that must hold EXACTLY by construction
(up to float error) — a failure means a bug in the ansatz, not a bad hyperparameter.

Run:  python tests/stage0_sanity.py
Exit code 0 = all pass.
"""
import sys
import logging

import netket  # noqa: F401  -- imported for the x64 side effect, matches training env
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, ".")
from src.ansatz import FermiSets
from main import exact_qho_gs_energy

log = logging.getLogger("stage0")
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


N, DIM = 3, 2
model = FermiSets(dim=DIM, N=N, rngs=nnx.Rngs(42), log=log, hidden_units=64, out_units=10)

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (128, N * DIM), dtype=jnp.float64)
logpsi = model(x)

# --- 0. output shape and finiteness on generic configs
check("output shape is (batch,)", logpsi.shape == (128,), f"got {logpsi.shape}")
check("finite on random configs", bool(jnp.all(jnp.isfinite(logpsi))))

# batch=1 must not collapse to a scalar via stray squeezes
logpsi_b1 = model(x[:1])
check("batch=1 keeps shape (1,)", logpsi_b1.shape == (1,), f"got {logpsi_b1.shape}")

# --- 1. antisymmetry: psi(swap_ij x) == -psi(x) for every particle pair
xr = x.reshape(-1, N, DIM)
worst = 0.0
for i in range(N):
    for j in range(i):
        xs = xr.at[:, [i, j], :].set(xr[:, [j, i], :]).reshape(-1, N * DIM)
        ratio = jnp.exp(model(xs) - logpsi)  # must equal -1 exactly
        worst = max(worst, float(jnp.max(jnp.abs(ratio + 1.0))))
check("antisymmetry under all pair swaps", worst < 1e-6, f"max |psi'/psi + 1| = {worst:.2e}")

# --- 2. full-cycle permutation (composition of transpositions, even parity -> +1)
xc = jnp.roll(xr, shift=1, axis=1).reshape(-1, N * DIM)  # 3-cycle = even for N=3
ratio = jnp.exp(model(xc) - logpsi)
worst = float(jnp.max(jnp.abs(ratio - 1.0)))
check("even permutation leaves psi invariant", worst < 1e-6, f"max |psi'/psi - 1| = {worst:.2e}")

# --- 3. collision suppression: |psi| -> 0 as two particles merge, no NaN
base = jax.random.normal(jax.random.PRNGKey(1), (1, N, DIM), dtype=jnp.float64)
prev = None
monotone = True
finite = True
for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 0.0]:
    xcol = base.at[0, 1, :].set(base[0, 0, :] + eps).reshape(1, -1)
    lp = model(xcol)
    finite &= bool(jnp.all(jnp.isfinite(lp)))
    mag = float(jnp.real(lp[0]))
    if prev is not None and eps > 0:
        monotone &= mag <= prev + 1e-6
    if eps > 0:
        prev = mag
check("no NaN/Inf at and near collisions", finite)
check("|psi| decreases toward collision", monotone)

# --- 4. gradient finiteness near collision (what the kinetic energy needs)
xnear = base.at[0, 1, :].set(base[0, 0, :] + 1e-5).reshape(1, -1)
g = jax.grad(lambda z: jnp.real(model(z)).sum())(xnear)
check("finite gradient near collision", bool(jnp.all(jnp.isfinite(g))))

# --- 5. far-field: gaussian envelope keeps things finite
xfar = 10.0 * jax.random.normal(jax.random.PRNGKey(2), (16, N * DIM), dtype=jnp.float64)
check("finite in the far field (|x|~10)", bool(jnp.all(jnp.isfinite(model(xfar)))))

# --- 6. signature encoder oddness: eta(swap x) == -eta(x), exact
eta = model.eta_antisymmetric(x)
xs = xr.at[:, [0, 1], :].set(xr[:, [1, 0], :]).reshape(-1, N * DIM)
eta_s = model.eta_antisymmetric(xs)
worst = float(jnp.max(jnp.abs(eta_s + eta)))
check("eta is odd under exchange", worst < 1e-12, f"max |eta' + eta| = {worst:.2e}")

# --- 6b. 1D FermiSets (regularized-eta fix of 2026-07-14): antisymmetry must now be exact
model1d = FermiSets(dim=1, N=4, rngs=nnx.Rngs(7), log=log, hidden_units=16, out_units=10)
x1 = jax.random.normal(jax.random.PRNGKey(3), (64, 4), dtype=jnp.float64)
lp1 = model1d(x1)
check("1D: finite on random configs", bool(jnp.all(jnp.isfinite(lp1))))
x1r = x1.reshape(-1, 4, 1)
worst = 0.0
for i in range(4):
    for j in range(i):
        xs1 = x1r.at[:, [i, j], :].set(x1r[:, [j, i], :]).reshape(-1, 4)
        ratio = jnp.exp(model1d(xs1) - lp1)
        worst = max(worst, float(jnp.max(jnp.abs(ratio + 1.0))))
check("1D: antisymmetry under all pair swaps", worst < 1e-6, f"max |psi'/psi + 1| = {worst:.2e}")
eta1 = model1d.eta_antisymmetric(x1)
xs1 = x1r.at[:, [0, 1], :].set(x1r[:, [1, 0], :]).reshape(-1, 4)
worst = float(jnp.max(jnp.abs(model1d.eta_antisymmetric(xs1) + eta1)))
check("1D: eta is odd under exchange", worst < 1e-12, f"max |eta' + eta| = {worst:.2e}")

# --- 6c. L_z-projected model (lz_proj_K=6): antisymmetry + discrete rotation invariance
import math
modelp = FermiSets(dim=2, N=3, rngs=nnx.Rngs(42), log=log, hidden_units=64, out_units=10,
                   lz_proj_K=6)
lpp = modelp(x)
check("proj: finite on random configs", bool(jnp.all(jnp.isfinite(lpp))))
xs = xr.at[:, [0, 1], :].set(xr[:, [1, 0], :]).reshape(-1, N * DIM)
worst = float(jnp.max(jnp.abs(jnp.exp(modelp(xs) - lpp) + 1.0)))
check("proj: antisymmetry under swap", worst < 1e-6, f"max |psi'/psi + 1| = {worst:.2e}")
th = 2.0 * math.pi / 6
c, s = math.cos(th), math.sin(th)
xrot = jnp.stack([c * xr[..., 0] - s * xr[..., 1], s * xr[..., 0] + c * xr[..., 1]],
                 axis=-1).reshape(-1, N * DIM)
worst = float(jnp.max(jnp.abs(jnp.exp(modelp(xrot) - lpp) - 1.0)))
check("proj: invariant under 2pi/6 rotation", worst < 1e-6, f"max |psi'/psi - 1| = {worst:.2e}")

# --- 7. exact reference energies used as the benchmark target
for n, d, expect in [(1, 2, 1.0), (3, 2, 5.0), (6, 2, 14.0), (4, 2, 8.0), (5, 1, 12.5)]:
    got = exact_qho_gs_energy(n, d, "fermion")
    check(f"exact E(N={n}, d={d}) == {expect}", abs(got - expect) < 1e-12, f"got {got}")

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
    sys.exit(1)
print("Stage 0: all checks passed.")
