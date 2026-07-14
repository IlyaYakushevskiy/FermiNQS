# Research log — FermiNQS

Idea bank: what was tried, what happened, what's known-dead, what's still open.
Each session appends; never delete "dead ends" — they exist so nobody retries them.

---

## 2026-07-14 — Benchmark setup + post-mortem of ~100 h of cluster/PC runs

**TL;DR: the FermiSets 2D ansatz does not fail randomly — every from-scratch run converges to the
same wrong eigenstate, the fully holomorphic state with E = N(N+1)/2, because that state's
antisymmetric factor is literally the ansatz's own signature encoder η. Confirmed by direct
overlap measurement on a checkpoint. Secondary problem: SR/warm-start instability.**

### Problem 1 (main): the holomorphic trap

Survey of all 746 parsed runs (`EXPERIMENTAL DATA/outputs_cluster-18-05`, `outputs_from_pc_18-05`,
local `outputs/`; params from each run's `main.log`, energies from validation lines / final report):

| N (2D) | E_exact (GS) | E_holo = N(N+1)/2 | from-scratch runs land at | best warm-started |
|-------:|-------------:|------------------:|--------------------------:|------------------:|
| 3      | 5.0          | 6                 | 6.00 – 6.18               | 5.97              |
| 4      | 8.0          | 10                | 10.03 – 10.08             | 8.36              |
| 5      | 11.0         | 15                | 15.02 – 15.13             | 14.55             |
| 6      | 14.0         | 21                | 21.0 – 21.2               | 17.6              |

E_holo is the energy of det{1, z, z², …, z^(N−1)} × Gaussian — the "ν=1-like" state whose
antisymmetric factor is the complex Vandermonde ∏(zᵢ−zⱼ). **That is exactly the signature
encoder η in `FermiSets.eta_antisymmetric` (dim=2).** So "η × smooth symmetric factor" — the
laziest thing the network can do — is already an exact excited eigenstate. (The a=1.0
regularization `diff/√(|diff|²+1)` doesn't change this: the correction ∏√(|z_ij|²+1) is a
positive symmetric function the ρ-network absorbs for free.)

Direct proof (2026-07-14 N=3 benchmark run, checkpoint step_350, stopped at iter ~373):

- checkpoint energy: **6.0012 ± 0.002**, σ² = 0.024, R̂ = 1.008 (clean eigenstate, wrong one)
- |⟨ψ_NN | GS det{1,x,y}⟩|² = **1e-6** — zero ground-state content
- |⟨ψ_NN | det{1,z,z²}⟩|²    = **0.85** — dominantly the holomorphic state
- |⟨GS | holo⟩|² = 1e-5 (orthogonality sanity check of the estimator)

(Method: importance sampling from N(0,1)^6, 400k samples; script preserved in session scratchpad,
trivially re-creatable: build MCState, `flax.serialization.from_bytes` the .mpack, compare log-values
against analytic determinants.)

**Why it gets stuck (mechanism):** every eigenstate is a stationary point of the Rayleigh
quotient, so the energy gradient vanishes there. To descend from E_holo to the GS the network
must produce anti-holomorphic (z̄) orbital content, which in the parity-graded representation
means f(ξ, η) must depend on η in a genuinely odd, non-multiplicative way — not ψ ≈ η·g(ξ).
Fu's paper itself flags this as the hard case: the ratio ψ/η is discontinuous at collisions
(their z̄₁−z̄₂ vs z₁−z₂ example), and the exact f may be non-differentiable (their η/D example).
So the GS sits at the representationally hard corner of the construction while an excited
eigenstate sits at the trivially easy corner — the optimizer finds the easy eigenstate and the
gradient toward the GS is (near-)zero there. This is not a sampling problem, not a learning-rate
problem, and more capacity alone doesn't fix it (hidden_units 16→512 all land on the same plateau).

Consistent detail: warm-started/nudged runs (N=4: 8.36–9.3; N=5: 14.55–14.94; N=6: 17.6–19.1)
do get below the plateau but stall between the two eigenvalues and become unstable — consistent
with entering the non-smooth-f regime where SR conditioning degrades.

### Problem 2: optimization/SR instability

- Catastrophic blow-ups scattered through the data: E → 100.2, 133.9, 283.3, 1120, 6509,
  2.1e5, −1.2e6, −1.1e15, −5.6e56. Concentrated in: warm-started runs, Adam with hidden=512,
  sgd with lr ≥ 0.08 or ≤ 0.002 warm-started. This is what motivated the `make_safe_solver`
  bilinear-form guard (never triggered in the from-scratch N=3 benchmark: bilinear form ~8–16
  vs threshold 900).
- Sampler mixing during training is marginal: R̂ ≈ 1.07–1.17 throughout (target < 1.05), though
  a *converged* state samples cleanly (R̂ = 1.008 at checkpoint evaluation). So mixing is a
  transient-phase nuisance, not the cause of Problem 1. Note the `SamplerExchangeRule` swap moves
  cannot help with wrong-state convergence at all: |ψ|² is permutation-symmetric, so swaps are
  always accepted and carry no information about the sign structure.

### Corroborated code bugs (found by reading, confirmed by data)

- **1D FermiSets is structurally broken** (`-eta` applied to a *log*; correct op is `eta + iπ`):
  the 22 N=2 1D runs landed at E = 8.3–15.6 vs exact 2.0. All 1D FermiSets data is invalid.
- **`optmizer:` typo** in `qho_fermisets_2d_3N.yaml` / `_5N.yaml`: Hydra silently kept the
  default — those runs used sgd, not momentum. (Verified via Hydra compose.)
- `GaussianFermions` dim=2 crashes (NameError); `main.py` hardcodes `n_dim=2` in the sampler rule.

### Dead ends (do not retry)

- Rerunning FermiSets-as-is from scratch on any 2D N with different lr / optimizer / hidden_units /
  n_samples: 700+ runs show the outcome is always E = N(N+1)/2. The plateau is structural.
- Fixing Problem 1 via sampler tweaks (sigma, exchange_prob, sweep_size): sampling is not the
  bottleneck; the trained states have R̂ ≈ 1.
- More vmc_iters on a plateaued run (3000- and 5000-iter runs plateau identically).

### Open hypotheses / next experiments (respect the stop rule: pick ≤ 3)

1. **Break the multiplicative shortcut at the architecture level.** Feed η into f oddly but
   richly: e.g. gate the symmetric features with odd functions of η (ξ ⊗ [Re η, Im η, |η|²·Re η, …])
   or give the Ψ-network an η-conditioned backflow on ξ, instead of appending 2 scalars to the
   pooled vector (current `eval_psi0`). Rationale: the GS needs f odd in η with strong ξ–η mixing.
2. **Deflate the known trap state.** For benchmarks the trap is analytic (det{1,z,…}); minimize
   E + λ|⟨ψ|ψ_holo⟩|²/⟨ψ|ψ⟩⟨holo|holo⟩ (penalty estimable by MC alongside the energy), or
   initialize orthogonal to it. Diagnostic value even if not a production method: if the network
   *still* can't reach 5.0 when the E=6 state is walled off, the representation itself (not the
   optimizer) is the limit for smooth NNs.
3. ~~**Anti-holomorphic augmentation of η.**~~ **RETRACTED (2026-07-14, same session):** feeding η̄
   is mathematically redundant — f already receives (Re η, Im η) as independent reals, so its linear
   sector spans αη + βη̄ automatically. Also the η̄-trap state det{1,z̄,…} is degenerate with the
   η-trap (same E = N(N+1)/2), so no encoder choice avoids the trap. Superseded by the coding plan
   below (L_z projection + feature engineering).

Negative-result framing for the thesis (if the above don't reach 5.0): the parity-graded
representation is universal in principle, but for the 2D harmonic gas the energy landscape has an
exponentially attractive excited eigenstate built from the signature encoder itself; universality
of representation ≠ trainability. That is a publishable observation, with the overlap measurement
as the smoking gun.

---

## 2026-07-14 — Coding plan: escape the holomorphic trap (P1) + stabilize training (P2)

### Theory recap that drives the design

η holomorphic is NOT the restriction: f gets (Re η, Im η) as independent reals, so already its
linear sector spans αη + βη̄; the η̄-trap is degenerate with the η-trap anyway, so no encoder choice
helps. What pins the optimizer to E = N(N+1)/2:
1. η·(smooth symmetric) is an exact eigenstate → stationary point (zero gradient).
2. The GS needs a symmetric prefactor singular at collisions: for N=3, GS antisym factor
   A = det[1,x,y] (signed triangle area) = η·S(ξ)/T(ξ) with S = Aη̄, T = |η|² symmetric.
   Fu's smoothing f_ε = ηS/(T+ε) is, in log space, `log η + log S − log(T+ε)` — trivially
   learnable IF the net can see log T. Currently it can't: η enters as 2 scalars appended after
   single-particle pooling; T = ∏|z_ij|² is pairwise.
3. |ψ_holo|² suppresses pair coalescence (~|z_ij|²), so samples never visit where the sign
   structures differ → vanishing gradient signal on top of stationarity.
4. Symmetry lever: the trap has L_z = N(N−1)/2 (e.g. 3 for N=3); every closed-shell GS has
   L_z = 0. Rotation-averaging (L_z=0 projection) annihilates the trap state EXACTLY.

### Phase 0 — hygiene + Problem 2 (src/train.py, main.py; ~1 h)

- 0.1 `make_safe_solver`: remove the `jax.debug.print` spam; replace zero-out with a trust-region
  rescale — if δᵀb > c, scale the update by c/(δᵀb) (keep direction, cap SR step norm) instead of
  throwing the step away.
- 0.2 Config guard in `main.py`: warn/fail on unknown `trainer.*` keys (the `optmizer` typo class).
- 0.3 Auto-rollback: callback tracks best validation energy; on NaN or E > best + 5σ, stop driver,
  reload last checkpoint, halve lr, resume (outer retry loop in `Trainer.__call__`, max 2 retries).
  This addresses the blow-up family (E → 1e2…1e56) seen in warm-started runs.
- 0.4 Sampler: log acceptance; tune `sigma` during burn-in to acceptance ∈ [0.3, 0.6] (R̂ was
  1.07–1.17 during training). Fix `n_dim=2` hardcode (use cfg.system.dim); resample idx_j ≠ idx_i
  in the exchange rule.
- 0.5 Fix 1D FermiSets sign flip (`eta + iπ`, not `−eta`) so Stage-0 tests can cover 1D too.

### Phase 1 — GS-pretrain diagnostic on the CURRENT architecture (highest info/hour; ~1–2 h)

New `tools/pretrain_gs.py`: supervised fit of log ψ_NN to the analytic N=3 GS
(log|det[1,x,y]| − Σr²/2, phase 0/π), batches x ~ N(0,1)⁶ ∪ near-collision oversampling,
loss = MSE(Re Δ) + (1 − cos Im Δ), Adam, ~5k steps; save `.mpack` compatible with
`ansatz.pretrained_path`. Then run the benchmark from that init.
- Outcome A (stays at 5.0, polishes): trap is pure basin-finding → deflation/projection will work.
- Outcome B (drifts back to 6.0): representation leaks near the GS (non-smooth-f regime) →
  Phase 2 features are load-bearing, not optional.
- The achievable supervised fidelity of the current architecture is itself a result for the thesis.

### Phase 2 — architecture v2 (src/ansatz.py, new `fermi_sets_v2`, keep v1 for A/B; ~2–3 h)

- 2.1 Pair stream: pooled two-body DeepSets Σ_{i<j} χ(|z_ij|², log(|z_ij|²+a²)) appended to ξ,
  plus explicit even feature log(T_reg + ε), T_reg = |η_reg|². Gives the net the "denominator".
- 2.2 Odd feature set fed to the Ψ-net (flipped by the ± symmetrization): u₁ = η_reg (2 reals) and
  u₂ = η_reg/(T_reg + ε) (2 reals, bounded by 1/2√ε — the f_ε structure as an input feature).
  Keep raw (non-log) form for odd features: the −η flip is only valid in raw space (the 1D bug).
- 2.3 Optional flag: L_z = 0 projection wrapper — logsumexp over K~8 rotated copies of the input
  (rotations commute with permutations, so antisymmetry survives). Kills the trap exactly for
  closed shells. CAVEAT: near the trap configuration the projection of η·g(ξ→invariant) ≈ 0, so
  log ψ can start very negative — verify init numerics in Stage 0 before training.
- 2.4 Extend `tests/stage0_sanity.py`: parametrize over models; add antisymmetry + rotation tests
  for v2; add a mini supervised-fidelity smoke test (can v2 fit the GS better than v1 offline?).

### Phase 3 — deflation penalty (only if Phases 1–2 don't reach 5.0; ~3–4 h, hardest)

Minimize E + λ·|⟨ψ|φ_holo⟩|²/(⟨ψ|ψ⟩⟨φ|φ⟩) with φ_holo analytic and FIXED:
- Pre-sample 100k configs from |φ_holo|² once (Metropolis; note |φ_holo|² is exactly the complex
  Ginibre eigenvalue density — eigenvalues of a random Gaussian matrix give exact samples for free).
- O = r₁r₂ with r₁ = E_{x∼|ψ|²}[φ/ψ], r₂ = E_{y∼|φ|²}[ψ/φ]; gradient à la Choo–Carleo penalty
  VMC. Needs a custom step (bypass nk.driver: vstate.expect_and_grad for H + manual penalty force).
- Anneal λ: ~2 (order of the gap) → 0; success = E < 5.5 that persists at λ = 0.

### Phase 4 — protocol runs + diagnostics

- Promote the overlap test to `tools/overlap_check.py` (checkpoint path as arg); report the triple
  (E, |⟨ψ|GS⟩|², |⟨ψ|holo⟩|²) — also add the η̄-mirror det{1,z̄,z̄²} to account for the missing
  0.15 weight in today's measurement.
- Every variant runs the same benchmark (N=3, target E ≤ 5.005) under the CLAUDE.md protocol;
  results appended here. Success at N=3 → scale to N=6 (next closed shell, E=14, L_z=0).

**Stop rule mapping: the "3 ideas" are (i) Phase-1 pretrain diagnostic, (ii) Phase-2 v2
architecture (features + optional projection), (iii) Phase-3 deflation. Hyperparameter re-tuning
of any variant does not count as a new idea but also must not exceed ~2 attempts per variant.**

---

## 2026-07-14 (later) — Phase 0 + Phase 1 execution

### Phase 0 landed (all verified on a 60-iter shakedown)

- Trust-region SR step (rescale onto the boundary instead of zeroing), debug-print removed
  (`src/train.py make_safe_solver`).
- `BlowupGuard` callback + auto-rollback loop in `Trainer.__call__`: on NaN or E > best + margin,
  reload last good checkpoint, halve lr, rebuild driver (momentum state reset); max 2 retries.
  Config keys: `trainer.auto_rollback / rollback_margin / max_retries`.
- Sampler sigma auto-tuning at startup targeting acceptance 0.35–0.6: on the benchmark it moved
  sigma 0.10 → 0.35 (acceptance 0.86 → 0.59). Config: `sampler.tune_sigma`. Acceptance now logged
  alongside each validation step.
- Config-typo guard in `main.py` (`validate_config`): unknown keys in system/ansatz/sampler/trainer
  now raise. Verified it rejects the historical `optmizer` configs — **those YAMLs now fail fast
  instead of silently running sgd; fix the typo if you rerun them.**
- 1D FermiSets eta rewritten as the regularized real Vandermonde (same construction as 2D, raw
  bounded product, vectorized) — the `-eta` flip is now exact in both dims. Stage-0 extended with
  1D checks: antisymmetry exact to 1.5e-11. Exchange-rule fixes: `n_dim` from config, distinct
  swap indices, `sampler.exchange_prob` configurable.

### Phase 1: supervised GS-fit diagnostic (`tools/pretrain_gs.py`, `tools/overlap_check.py`)

Fit target: analytic N=3 GS, log|det[1,x,y]| + iπ[det<0] − r²/2. Loss Var(Re Δ) + mean(1−cos Im Δ),
masking |det| < 1e-3 (GS node = collinear configs — note: codim-1, much larger than the collision
set where η vanishes; the sign structure the net must learn lives mostly in ξ-space).

- 6k steps, batch 4096, lr 1e-3:  |⟨nn|GS⟩|² = **0.36**, E = 13.0, σ² = 150.
- 20k steps, batch 8192, lr 2e-3: |⟨nn|GS⟩|² = **0.84**, E = 12.7, σ² = 1.1e3, loss still ↓ slowly.

Reading: the architecture CAN mostly represent the GS (0.84 and not saturated) — representability
is not a hard wall at N=3. The energy of the fitted state is still bad (12.7) because the 16%
residual is high-energy roughness (kinetic term punishes log-amplitude noise of std ~0.5).
Checkpoint: `outputs/pretrained/gs_N3_2d_h64.mpack` (earlier weaker fit: `..._v1.mpack`).

### Phase 1 decisive run: VMC released on the 84%-GS state — **OUTCOME A, confirmed**

`main.py +experiment=qho_fermisets_2d_3N_bench ansatz.pretrained_path=.../gs_N3_2d_h64.mpack
trainer.vmc_iters=400` (run dir `outputs/2026-07-14/14-21-35`). Final energy **5.0505 ± 0.0067**
(1.0% rel. error) — the FIRST run in this project below 5.97. Overlap trajectory
(`tools/overlap_check.py` per checkpoint):

| step | E          | \|⟨ψ\|GS⟩\|² | \|⟨ψ\|holo⟩\|² |
|-----:|-----------:|-------------:|---------------:|
| 0    | 12.70      | 0.845        | 1.1e-3         |
| 100  | 5.128      | 0.972        | 1.3e-3         |
| 250  | 5.049      | 0.990        | 4.3e-4         |
| 350  | 5.049      | **0.993**    | **4e-6**       |

**Verdict: the holomorphic trap is a basin-finding problem, not a representability problem.**
VMC actively *amplifies* GS weight once inside the basin and drives trap weight to zero. The
energy stalled at ~5.05 within 400 iters (σ² = 0.35, not yet an eigenstate; the last 0.7% of
non-GS weight costs ~0.05 ħω) — polishing further needs more iterations and/or lr schedule, and
possibly the Phase-2 pair features for the final digits (the cusp-adjacent structure is where the
architecture is weakest — its supervised fit also converged slowly).

**Honest caveat (user raised it, correctly): pretraining on the exact GS is a diagnostic, not a
method — real systems have no exact target.** Production-viable versions this verdict now
licenses, none of which need the exact answer:
1. **HF/mean-field pretraining** (what FermiNet/PauliNet do): HF orbitals exist for any system and
   already contain the anti-holomorphic content; supervised-fit machinery from
   `tools/pretrain_gs.py` carries over — only the target function changes.
2. **L_z = 0 projection** (Phase 2.3): needs only the symmetry sector of the target state.
3. **Deflation** (Phase 3): needs only the trap state, which is known by construction for this
   ansatz class (the encoder itself times an envelope).

Next session: pick between (1) as the quickest thesis-grade method and (2) as the most elegant;
also try simply continuing the 5.05 run (more iters, lower lr floor) to see where polishing stops.

### HF pretraining — the answer-free pipeline (`tools/pretrain_hf.py`)

Generic Slater-determinant pretraining: target = det[φ_j(r_i)] with φ_j the N lowest 2D-oscillator
orbitals via aufbau filling (deterministic within degenerate shells, works for any N incl. open
shells). Nothing exact is used: for `qho_no_inter` HF happens to coincide with the exact GS, but
the pipeline carries over to interacting potentials by supplying an orbital-coefficient matrix
from an SCF solve (`make_log_hf(N, coeffs=C)` hook — SCF solver itself not yet written).

- Pretrain result (N=3, 30k steps, ~4 min on the 2080): |⟨ψ|GS⟩|² = **0.951**, E = 5.89, σ² = 41 —
  a better init than the Phase-1 exact-GS fit (0.845), at no extra cost.
- Full pipeline run (pretrain → benchmark VMC, 1000 iters): launched, then **interrupted near
  step 0** — user pivoted to the fully from-scratch attempt below, and the outcome here is
  near-certain from Phase 1 (init has 0.951 GS overlap; Phase 1 converged from 0.845). Rerun
  anytime: the `.mpack` and command are above. For the QHO benchmark HF = exact GS anyway, so
  this run's only new content was pipeline packaging.

### L_z = 0 sector projection — the from-scratch, answer-free attempt (`lz_proj_K`)

User requirement: solve the benchmark from scratch with NO QHO-derived pretraining. Mechanism:
project the ansatz onto the L_z ≡ 0 (mod K) sector by averaging over K rotated copies,
psi_proj(x) = (1/K) Σ_k psi(R(2πk/K) x), inside `FermiSets.__call__` (`lz_proj_K` config key,
K=6, 2D only; rotations commute with permutations so antisymmetry is exact — verified in Stage 0
along with exact 2π/6-rotation invariance).

Why this kills the trap structurally: every lazy state eta·(holomorphic symmetric of degree d)
has L_z = N(N−1)/2 + d > 0 — for N=3 the trap (d=0) has L_z=3, eta·Σz (d=1) has L_z=4 at E=7.
K=6 annihilates all |L_z| in 1..5, so the ENTIRE lazy family is unrepresentable; the closed-shell
GS (L_z=0) survives untouched. Projected-sector spectrum: 5.0 (GS), 7.0 (next), leakage only from
|L_z| ≥ 6 (E ≥ 9). The only physics knowledge used is the target's symmetry sector — legitimate
for any closed-shell system, interacting or not.

Cost: ~6x per wavefunction evaluation. Risks flagged in advance: (i) at init the projection
suppresses the amplitude (the eta·g_even component is annihilated) — watch for log-floor
saturation in `safe_complex_logsumexp`; (ii) the network must now build z̄ content to represent
ANYTHING low-energy — if optimization stalls near E≈7 (the sector's lazy-ish region) instead of
descending to 5, the verdict is that projection alone fixes representability-bias but not
trainability, and the Phase-2 pair features become the complement.

Run: `+experiment=qho_fermisets_2d_3N_bench_lz` (800 iters, from scratch, seed 42) — launched
2026-07-14 ~15:30, results to be appended.
