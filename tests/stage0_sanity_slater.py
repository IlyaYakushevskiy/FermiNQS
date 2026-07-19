"""
Stage 0 sanity checks for SlaterNN (QUEUE.md P2 baseline). No training, runs in
seconds. Mirrors tests/stage0_sanity.py's structure/spirit for FermiSets: every check
here must hold EXACTLY by construction (up to float error) -- a failure means a bug in
the ansatz, not a bad hyperparameter.

Run: python tests/stage0_sanity_slater.py
Exit code 0 = all pass.
"""
import sys
import logging

import netket  # noqa: F401  -- x64 side effect, matches training env
import jax
import jax.numpy as jnp
from flax import nnx

sys.path.insert(0, ".")
from src.ansatz import SlaterNN

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


for N in (3, 6):
    DIM = 2
    print(f"\n--- N={N}, dim={DIM} ---")
    model = SlaterNN(dim=DIM, N=N, rngs=nnx.Rngs(42), hidden_units=64)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (128, N * DIM), dtype=jnp.float64)
    logpsi = model(x)

    check(f"N={N} output shape is (batch,)", logpsi.shape == (128,), f"got {logpsi.shape}")
    check(f"N={N} finite on random configs", bool(jnp.all(jnp.isfinite(logpsi))))

    logpsi_b1 = model(x[:1])
    check(f"N={N} batch=1 keeps shape (1,)", logpsi_b1.shape == (1,), f"got {logpsi_b1.shape}")

    # antisymmetry: psi(swap_ij x) == -psi(x) for every particle pair (exact by
    # construction -- determinant sign flips under row swap)
    xr = x.reshape(-1, N, DIM)
    worst = 0.0
    for i in range(N):
        for j in range(i):
            xs = xr.at[:, [i, j], :].set(xr[:, [j, i], :]).reshape(-1, N * DIM)
            ratio = jnp.exp(model(xs) - logpsi)  # must equal -1 exactly
            worst = max(worst, float(jnp.max(jnp.abs(ratio + 1.0))))
    check(f"N={N} antisymmetry under all pair swaps", worst < 1e-6,
          f"max |psi'/psi + 1| = {worst:.2e}")

    # full N-cycle: parity (-1)^(N-1) -> even for odd N, odd for even N
    xc = jnp.roll(xr, shift=1, axis=1).reshape(-1, N * DIM)
    expected_sign = 1.0 if (N - 1) % 2 == 0 else -1.0
    ratio_cycle = jnp.exp(model(xc) - logpsi)
    worst_cycle = float(jnp.max(jnp.abs(ratio_cycle - expected_sign)))
    check(f"N={N} N-cycle permutation parity", worst_cycle < 1e-6,
          f"expected sign {expected_sign:+.0f}, max |ratio - expected| = {worst_cycle:.2e}")

    # collision: two particles at the same point -> determinant exactly singular
    # (two identical rows), no NaN -- this is the correct/expected behavior for a bare
    # Slater determinant (no regularization needed/used, unlike FermiSets' eta). LU
    # elimination on an exactly-singular float32/64 matrix mostly underflows to -inf
    # but not always bit-for-bit (rounding in the elimination steps), so check "far
    # below the non-collision scale" rather than a fixed absolute threshold.
    x_collide = xr.at[:, 1, :].set(xr[:, 0, :]).reshape(-1, N * DIM)
    logpsi_collide = model(x_collide)
    logabs_collide = jnp.real(logpsi_collide)
    logabs_normal = jnp.real(logpsi)
    collision_ceiling = float(jnp.min(logabs_normal)) - 5.0
    check(f"N={N} collision far below non-collision scale, not NaN",
          bool(jnp.all(logabs_collide < collision_ceiling)),
          f"collision max={float(jnp.max(logabs_collide)):.2f}, "
          f"non-collision min={float(jnp.min(logabs_normal)):.2f}")
    check(f"N={N} collision has no NaN", not bool(jnp.any(jnp.isnan(logpsi_collide))))

print("\n" + "=" * 50)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    sys.exit(0)
