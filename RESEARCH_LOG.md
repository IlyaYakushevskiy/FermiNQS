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

Run: `+experiment=qho_fermisets_2d_3N_bench_lz` (800 iters, from scratch, seed 42, run dir
`outputs/2026-07-14/15-36-30`, ~15 s/it ≈ 3.9 h total on the 2080) — **RESULT: solved, modulo
the last decimal.**

- Trajectory: E = 10.37 (step 0) → 5.17 (step 50) → plateau oscillating 5.01–5.03 from ~step 150;
  best validations 5.0132 ± 0.0048 (step 600), 5.0173 ± 0.0039 (step 750, σ² = 0.062);
  final training-chain estimate **5.0090 ± 0.0076**. Neither init-suppression nor an E≈7 stall
  materialized — the network descended straight into the GS basin, from scratch, no pretraining.
- Step-750 checkpoint, evaluated WITH the projection (overlap_check `--lz-proj-K 6`):
  E = 5.0149 ± 0.0039, |⟨ψ|GS⟩|² = **0.9988**, holo = antiholo = 2.5e-5.
- Same checkpoint WITHOUT the projection (raw network): E = 7.41, GS weight 0.06, antiholo 0.38 —
  i.e. the raw network is nowhere near the GS; **the projection is load-bearing at evaluation
  time, not just a training regularizer.** (This also bit `tools/overlap_check.py`, which
  originally rebuilt the model unprojected — fixed with the `--lz-proj-K` flag; always match the
  training config when loading checkpoints.)
- Benchmark criteria: variance ↓ (5.7 → 0.06) ✓; train/val agreement ✓; guard silent ✓; energy
  criterion |E−5|/5 ≤ 1e-3 **narrowly missed** — clean readings sit at 0.2–0.35% (the final
  5.0090 ± 0.0076 touches 5.005 within ~1.2σ but the validation plateau is ~5.015–5.02).
  The residual ~0.3% is a real variational gap, consistent with the architecture's known weak
  spot (slow supervised convergence near nodal/cusp structure) — next levers: Phase-2 pair
  features, longer run with finer lr floor, or larger hidden size.

**Bottom line for the thesis: the holomorphic trap is fully circumvented from scratch by
symmetry projection alone (2 of the 3 sanctioned ideas used: pretrain-diagnostic, projection;
deflation never needed). From-scratch error on the canonical benchmark: 20% (old, all 746 runs)
→ 0.2–0.3% (this run), with 99.9% GS fidelity.**

---

## 2026-07-15 — Does the L_z trick generalize? (discussion, no runs)

User challenge: "L_z projection cheats the particular symmetric benchmark." Assessment:

**Not cheating in the standard sense** — the projection uses [H, L_z] = 0 + the target sector,
a property of the Hamiltonian, not of the solution (same class as fixing N or S_z; cf.
point-group symmetrization in quantum chemistry, momentum/point-group projection in lattice NQS
à la Choo–Neupert–Carleo, angular-momentum projection in nuclear structure). **But** the fair
core of the objection stands: the escape worked by making the trap *unrepresentable*, so we have
NOT shown FermiSets can escape lazy eigenstates when no symmetry forbids them.

**Key counter-observation**: the trap is exactly stationary only because η × symmetric is an
*exact eigenstate of the non-interacting QHO* (Vandermonde × Gaussian). With any interaction on,
that state stops being an eigenstate → nonzero gradient there → plain VMC can slide off. Open
question: does a soft metastable basin remain, or does the pathology evaporate with the
fine-tuning that created it? (Aside: the holomorphic states are LLL/Laughlin-type — the ansatz's
bias would be a *feature* for rotating traps / FQHE regimes.)

**Where the trick transfers**: any rotationally symmetric 2D system, interacting included —
2D quantum dots (harmonic + Coulomb; [H,L_z]=0 survives interactions; GS sector can shift to
"magic" L at strong coupling / Wigner regime → scan sectors via phases e^(−2πikl/K));
harmonium/Taut atom (analytic interacting reference at special ω, but N=2 polarized GS is
degenerate m=±1); 2D HEG (analogue = total-momentum projection). **Where it dies**: generic
geometries (molecules, disorder) — there HF pretraining (`tools/pretrain_hf.py`) is the
production escape, exactly the FermiNet/PauliNet pipeline.

**Proposed next benchmark (settles the objection with data)** — N=3 spin-polarized 2D quantum
dot, Coulomb λ/r (new `qho_coulomb` branch in `src/system.py`), three arms:
  1. from scratch, NO projection — falsification test: if VMC finds the GS unaided, the trap was
     a QHO artifact; if it hangs near holomorphic-ish states, the trap is a soft basin and the
     mitigations earn their keep;
  2. with lz_proj_K — does projection still deliver with interactions on;
  3. with HF pretraining — production pipeline (SCF `coeffs` hook already in pretrain_hf.py).
Ground truth: small CI/ED code in the oscillator basis (N=3, few dozen orbitals — minutes on
CPU), gives exact reference energies at any coupling. Status: QUEUED as future work
(user decision 2026-07-15) — the no-symmetry falsification test below runs first, since it
answers the same "is the trap a QHO artifact?" question with far less machinery.

---

## 2026-07-15 — No-symmetry benchmark: anisotropic trap (falsification test of the trap hypothesis)

**Design** (documented before running, per protocol). User question: does FermiSets work at all
without rotational symmetry to lean on? Cleanest test found: the **anisotropic harmonic trap**
V = ½(x² + ω_y² y²), ω_y = 1.5 — breaks rotational symmetry completely ([H, L_z] ≠ 0; the L_z
projection is not just unavailable, it would be *wrong*), yet remains exactly solvable:
single-particle levels E(nₓ, n_y) = (nₓ + ½) + ω_y(n_y + ½).

**System: N=3 spinless fermions, dim=2, ω_y = 1.5.** Levels 1.25, 2.25, 2.75 | gap 0.5 | next 3.25
→ **exact GS energy E = 6.25, non-degenerate** (ω_y = 2 would be degenerate at the Fermi level —
avoided deliberately; a degeneracy guard now raises on such configs). Exact GS wavefunction:
det{1, x, y} × ∏ᵢ exp(−(xᵢ² + 1.5 yᵢ²)/2) — same collinear nodal topology as the solved isotropic
benchmark, just a squeezed envelope. The ansatz itself needs NO changes: FermiSets has no
hardcoded isotropy (learned envelope; η is a valid signature encoder for any 2D system).

**Hypothesis under test**: the holomorphic trap was exactly stationary only because
η × symmetric happened to be an exact QHO eigenstate. In the anisotropic trap it is NOT an
eigenstate → nonzero gradient there → from-scratch VMC (no projection, no pretraining) should
slide off and find the GS. Outcomes:
- **A (trap = QHO artifact)**: E → 6.25 from scratch, GS overlap ≈ 1. Then FermiSets needs no
  crutch in generic geometries, and L_z projection is understood as the fix for the
  pathologically symmetric benchmark specifically.
- **B (trap = soft basin)**: stall at some plateau above 6.25 with high overlap onto
  det{1,z,z²} × squeezed envelope (the lazy analogue, no longer an eigenstate). Then
  trap-escape machinery (HF pretraining) is needed generally, and the ω_y → 1 crossover
  (how stall time diverges as symmetry is restored) becomes the interesting follow-up.

**Code**: `qho_aniso` branch in `src/system.py` (+ `omega_y` config key), general
`exact_trap_gs_energy(N, omegas)` in `main.py` (aufbau fill of anisotropic levels, raises on
Fermi-level degeneracy), config `configs/experiment/qho_fermisets_2d_3N_aniso.yaml`
(copy of the canonical bench: seed 42, momentum lr 0.01, 4096 samples, 512 chains, 800 iters,
**lz_proj_K: 0**), stage-0 checks for the new energies, `tools/overlap_check.py --omega-y`
(anisotropic envelope in the reference states). Cost without projection: ~2.5 s/it → full run
~35 min. Status at time of writing: RUNNING — result appended below when done.

**RESULT (run `outputs/2026-07-15/13-32-36`, 800 iters, ~3.6 s/it): Outcome B — and sharper
than predicted. The trap generalizes; holomorphy was never the point.**

- Trajectory: 9.1 → plateau 6.75–6.77 from ~step 150; final training estimate 6.7519 ± 0.0043;
  validation flat at 6.75–6.76 for the last 400 steps; σ² settles at 0.06–0.13 (not → 0).
  Exact GS = 6.25: stalled 8% high. Guard silent, R̂ ≈ 1.06, acceptance 0.50.
- Step-750 checkpoint (overlap_check `--omega-y 1.5`): E = 6.7547 ± 0.0036 (σ² = 0.086, R̂ = 1.004),
  |⟨ψ|GS⟩|² = **1.5e-5**, |⟨ψ|holo⟩|² = |⟨ψ|antiholo⟩|² = 0.216 (equal — conjugation-symmetric
  state), |⟨ψ|det{1,x,x²}⟩|² = **0.9989**.
- Identification: det{1, x, x²} × envelope is the **exact first excited state** of the ω_y=1.5
  trap — promote (0,1) → (2,0), E = 1.25 + 2.25 + 3.25 = 6.75, non-degenerate (next excitation
  7.25). The run parked exactly on it, nailing its energy to 0.07% — same accuracy scale as the
  L_z-projected GS run. The Hermite lower-order terms cancel inside the determinant, so
  det{1,x,x²} is exact, and its antisymmetric factor is the REAL 1D Vandermonde ∏(xᵢ−xⱼ).

**Refined mechanism (supersedes "holomorphic trap" as the general statement):** from-scratch
FermiSets converges to the lowest eigenstate whose sign structure is a *product of pairwise
differences* — the sortable / Vandermonde-type sign structures that are representationally easy
for f(ξ, ±η):
  - isotropic trap → det{1,z,z²}: complex Vandermonde ∏(zᵢ−zⱼ) = literally η (E=6, L_z=3);
  - anisotropic trap → det{1,x,x²}: real Vandermonde ∏(xᵢ−xⱼ) = the classically-easy 1D
    ("sortable") antisymmetry embedded in 2D (E=6.75).
The true GS det{1,x,y} has the signed-triangle-area sign structure — NOT a product of pairwise
terms, the genuinely-2D case requiring singular/non-smooth f — and the network avoids it in
every geometry tested. So the trap is a property of the **parity-graded ansatz + local
optimization**, not of the QHO's rotational symmetry. The user's suspicion that L_z projection
"cheats the symmetric benchmark" is answered by data: the projection is indeed a symmetric-case
fix, but the pathology it fixes generalizes to non-symmetric systems, where from-scratch
training fails identically (0 GS overlap). Escape machinery is a *required* production
ingredient, not a benchmark crutch: HF pretraining (general geometries) / symmetry projection
(when available) / pair-feature architectures (root-cause fix, Phase 2).

Deliberately NOT run: the HF-pretrained arm for this trap. For a non-interacting system HF is
exact, so "HF pretraining" here would be pretraining on the exact GS — near-certain outcome
(already demonstrated at 0.951 init overlap isotropically) and no new information per GPU-hour.
The place where the HF arm is a real test is the interacting dot (QUEUED above).

Open follow-ups: (i) ω_y → 1 crossover — how the stall state morphs between det{1,x,x²} and
det{1,z,z²} as symmetry is restored (cheap, diagnostic); (ii) pair-feature architecture to make
triangle-area sign structures representable (the root fix); (iii) the interacting-dot 3-arm
study. New probe: `excx` in tools/overlap_check.py (det{1,x,x²}, any ω).

---

## 2026-07-15 — Pair-feature architecture (Phase 2 / sanctioned idea #1, the root-cause attempt)

**Design (before running).** Both benchmarks show the same failure: f(ξ, ±η) cannot build the
singular symmetric prefactor S/T (T = |η|²) that converts the pairwise-product η into
triangle-area-type states. Fix: add a **pair stream** to the symmetric embedding — Deep Sets
over the N(N−1)/2 unordered pairs, features per pair (all exchange-even, so ξ stays symmetric
and antisymmetry still comes solely from the ±η flip):
  dim=2: [r², Re(Δz)², Im(Δz)², log(r²+1e-3), 1/(r²+1)]     (Δz = Δx + iΔy)
  dim=1: [Δx², log(Δx²+1e-3), 1/(Δx²+1)]
MLP per pair → sum-pool → concat with the per-particle pooled vector before the Ψ head.

**The load-bearing feature is log(r²+ε):** −log T = −Σ_pairs [log r²ᵢⱼ − log(r²ᵢⱼ+1)] is then a
LINEAR function of the pooled pair vector, so log ψ_GS = log η + log S − log T becomes directly
expressible in log space (S smooth symmetric). ε = 1e-3 softens collisions (Fu's f_ε analogue);
the discrepancy lives where |ψ|² → 0, so finite ε should not cap fidelity at the 1e-3 level.
1/(r²+1) matches η's own a=1 regularizer.

Config knob `ansatz.pair_hidden` (0 = off, fully backward compatible — old checkpoints/configs
unchanged). Benchmark configs: `qho_fermisets_2d_3N_bench_pair.yaml` (ISOTROPIC, from scratch,
**NO projection, no pretraining** — the historical always-fails-to-6.0 setting) and
`qho_fermisets_2d_3N_aniso_pair.yaml` (same for the 6.75 trap). pair_hidden: 32,
otherwise identical hyperparameters to the canonical bench.

**Decision rule**: isotropic-pair from scratch first. If E descends below the 6.0 plateau with
growing GS overlap → root cause addressed → run aniso-pair to confirm generality. If it stalls
at 6.0 again → pair features insufficient at this size/feature set; that exhausts sanctioned
idea #1 and the negative result stands (projection/pretraining remain the escape routes).
Note: eigenstates remain parameter-space stationary points under ANY architecture — the
hypothesis is about basin geometry from random init, not about removing the stationary point.

Status: implementing; results appended below.

**RESULT: NEGATIVE — pair features do not change basin selection. Killed at ~step 100 by
decision rule; sanctioned idea #1 exhausted.**

Shakedown history (all documented failure modes, all fixed or fatal):
- v1 unbounded features (raw r², quadrupoles ~50 on sampled configs): instant blow-up at
  step 0 (σ² ≈ 3.6e2, complex-garbage energies, rollback exhausted). Root cause: feature
  scale, not physics. FIX: bounded features log1p(r²), quadrupoles/(1+r²).
- v2 zero-init gating of pair_dense2: stable, but early dynamics are then EXACTLY the plain
  architecture → parks in the 6.0 trap before the stream wakes (gradient ≈ 0 at eigenstate,
  nothing to wake it). Design error, reverted.
- v3 bounded features + live init (committed 1e4b131): stable, Stage 0/1 clean.
Full run `outputs/2026-07-15/17-20-12`: step 0 E = 6.611 (σ²=4.1) → step 50 E = 6.0004
(σ²=1.5e-2) — full-scale trap capture, identical to the 746 historical runs. Killed there:
the pre-registered risk is confirmed — **eigenstates are parameter-space stationary points
under ANY architecture; representability (which the pair stream does add) is irrelevant once
the optimizer reaches the trap, because no gradient signal remains.** Basin selection from
random init is governed by which states are easy EARLY, and η×smooth is always easiest.

Verdict for the thesis: the trap is an optimization-dynamics property, not (only) a
representability gap. Architecture work cannot fix it alone; state selection must come from
outside the energy gradient: symmetry projection (works, proven), pretraining (works,
proven), or deflation (untested, idea #2, not spent). OPTIONAL cheap salvage (not run, GPU
prioritized to the dot): pair features + lz_proj_K on the isotropic bench to attack the
residual 0.3% variational gap — representability is exactly what the pair stream fixes, and
inside the projected sector the trap is unrepresentable so basin selection is moot.

---

## 2026-07-15 — INTERACTING DOT: design, ED reference (arms 1–2 wired)

**System**: N=3 spinless fermions, 2D, H = Σ(−½∇² + ½r²) + λ Σ_{i<j} exp(−r²ᵢⱼ/(2s²)),
λ = 2.0, s = 1.0. **Gaussian repulsion chosen over bare Coulomb deliberately**: (i) two-body
integrals in the oscillator basis are numerically exact (integrand smooth → Gauss-Legendre
converges to machine precision), (ii) no coalescence cusp, so a smooth NN can represent the
GS — trap failures are attributable to optimization, not cusp representability (Coulomb would
confound the two), (iii) ED and VMC use the IDENTICAL operator → internally exact benchmark.
Bare Coulomb = follow-up via Gaussian expansion of 1/r or Anisimovas–Matulis elements.

**ED reference** (`tools/ed_dot.py`: full CI, Cartesian oscillator basis, parity-block
diagonalization, Slater–Condon):
- Validation: λ=0 → 5.0 exactly; N=2 ED vs INDEPENDENT radial-ODE solve of the relative
  problem (m=1): +5e-5 (S=6) → +2.6e-6 (S=10), variational from above ✓. The validation
  CAUGHT a real bug: doubles-excitation phase had a spurious factor (−1) — with it, N=2 ED
  landed 5e-3 BELOW the ODE (impossible variationally). Fixed & verified. Lesson: never
  trust CI signs without an independent cross-check.
- **N=3, λ=2, s=1: E_GS = 6.21066** (shells ≤7, basis-converged ~8e-5), parity block (1,1)
  (= adiabatically connected to the non-interacting closed shell), first excited 7.016
  (x↔y doublet, rotational symmetry) → **gap 0.81, non-degenerate. Valid benchmark.**
  Interaction shift +1.21 on E, ~24% of E_GS — genuinely interacting regime.

**Arms** (plain ansatz pair_hidden=0, matching historical failure conditions):
1. `dot_gauss_2d_3N_scratch` — from scratch, no projection: does the trap survive when
   η×symmetric is NOT an eigenstate of the interacting H? (The aniso trap answered "the
   network finds ANOTHER easy eigenstate" — here, with interactions, no Vandermonde-type
   det is an eigenstate at all. Open question.)
2. `dot_gauss_2d_3N_lz` — lz_proj_K=6 (Gaussian interaction is rotation-invariant →
   [H,L_z]=0 survives → trick carries over verbatim), 400 iters (~6× eval cost).
3. HF-pretrain arm: NOT wired yet — needs an SCF solver (Fock matrix from the same ED
   integral tables; genuine test here since HF ≠ exact with interactions). Future work.

Status: arm 1 launching (run dir appended below when done).

### Arm 1 result (2026-07-15, run `outputs/2026-07-15/17-30-08`) — NEGATIVE: trap survives interactions

Killed at step 378/800 (user call, correct): energy flat for >100 iters.
- Training E (last 100 iters): 7.0204, σ² ≈ 0.053. Validation (independent chain, step 350):
  **7.0225 ± 0.0031** — i.e. the ED **first excited state 7.016**, not the GS 6.21066.
- Overlap probes (`tools/overlap_check.py` on step_350, probes are the NON-interacting
  states, so quantitative deformation expected): **|⟨nn|holo⟩|² = 0.849**, antiholo 0.063,
  excx 0.170 (not independent — excx has holo/antiholo components), GS content 0.0014.
- **Reading**: with interactions, η×symmetric is no longer an eigenstate — and the trap
  doesn't care. From-scratch VMC converges to the interaction-deformed holomorphic state
  (85% overlap with the λ=0 holo state, energy pinned to the interacting first-excited
  doublet). Third system (iso QHO, aniso trap, interacting dot), same mechanism: the basin
  is selected by the ansatz's easiest sign structure (pairwise product), NOT by the
  Hamiltonian's spectrum or symmetry. Escape machinery is a REQUIRED production ingredient.
- Small variance (0.053) despite not being an exact eigenstate: the deformed-holo state is
  a near-eigenstate of the interacting H — variance alone would NOT have flagged this
  failure. The ED reference (energy 13% off GS) is what catches it. Lesson reaffirmed:
  never benchmark without an independent reference value.

### Arm 2 (lz_proj_K=6) launched 2026-07-15, run `outputs/2026-07-15/17-57-24` (result appended below when done).
Clean start: e_ref 6.21066 wired, step-0 validation 12.203 ± 0.050 (random init, no NaN).

### Arm 2 result — POSITIVE: projection solves the interacting dot from scratch

400 iters completed clean (no NaN, no guard trips). ED reference E_GS = 6.21066.
- Trajectory: 12.20 (step 0) → 6.269 (step 50) → plateau ~6.212–6.225 from step ~100.
- Final training-chain estimate: **6.2165 ± 0.0044** (rel. excess **9.4e-4**, within 1.3σ
  of ED). Trailing-100 training mean 6.2164, σ² ≈ 0.087. Validation plateau (independent
  chain) 6.212–6.225 ± 0.004 — train/val agree ✓.
- Step-350 checkpoint, `overlap_check --lz-proj-K 6`: holo = antiholo ≈ 1e-5 (trap
  annihilated ✓), |⟨ψ|gs_noninter⟩|² = 0.876 — the 12% deficit IS the interaction
  deformation (probe is the λ=0 GS; interaction shift is 24% of E). NOTE: the tool's
  energy line uses the NON-interacting H — ignore it for dot checkpoints; the energy
  test is the ED comparison above. Follow-up tooling idea: overlap vs the ED
  wavefunction itself (expand ED GS on the oscillator basis, evaluate on MC samples).
- **A/B, same seed/hyperparams, only lz_proj_K differs: 0 → converges to the FIRST
  EXCITED state 7.0225 (deformed-holo trap); 6 → GS at 0.09% rel. error.** With the QHO
  result (5.009/5.015, fidelity 0.9988): symmetry projection is established as the
  from-scratch trap escape for BOTH non-interacting and interacting rotationally
  symmetric systems. The energy criterion 1e-3 is MET on the final training estimate
  here (validation plateau reads ~2e-3); the residual is the same architecture-level
  variational gap seen on the QHO.
- Publication figures: `plots/plot_publication.py` (new reusable two-panel script:
  zoomed energy + log rel-error), `plots/qho_lz_N3_convergence_pub.png`,
  `plots/dot_gauss_lz_N3_convergence_pub.png`.

**Conceptual framing settled with the aniso result (for the thesis write-up): the trick
as implemented is projection onto the trivial rep of the DISCRETE rotation group C_K —
it needs [H, R(2π/K)] = 0, not full U(1) symmetry. The general principle is symmetry
projection onto any sector that separates the GS from the ansatz's lazy states: for the
aniso trap, y-parity does this (GS det{1,x,y} has (Σnx,Σny) parity (1,1); the trap
det{1,x,x²} has (1,0)) — a ×2-cost reflection average, UNTESTED (idea, not run). Systems
with no usable symmetry at all (disorder, generic geometry) still need pretraining.**
