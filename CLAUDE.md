# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Thesis codebase: "Efficient neural network wave functions for interacting fermions." Variational Monte Carlo
(VMC) training of neural-network wavefunction ansätze for particles (bosons/fermions) trapped in a quantum
harmonic oscillator (QHO), built on NetKet + JAX/Flax NNX. Results are checked against the analytic QHO
ground-state energy.

**Research goal** (paper: `papers/boson+one.md`, Fu 2025 arXiv:2510.11431): test whether the "parity-graded"
representation ψ(R) = [f(ξ(R), η(R)) − f(ξ(R), −η(R))]/2 — a symmetric Deep-Sets embedding ξ plus a
low-dimensional antisymmetric "signature encoder" η — can replace Slater determinants for fermions in **2D**.
1D is not scientifically interesting (antisymmetry there is classically easy via sorting/the real
Vandermonde); the entire thesis question lives in the `FermiSets` ansatz with `dim=2`, where η₁+iη₂ is the
complex Vandermonde ∏(zᵢ−zⱼ) (implemented with the regularized factor `diff/sqrt(|diff|²+a²)`, still a valid
signature encoder per the paper's Eq. 10).

## Setup and running

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional, if logging to Weights & Biases
wandb login

python main.py +experiment=<experiment_name>
```

Experiments are Hydra configs under `configs/experiment/*.yaml`, layered on top of the defaults in
`configs/train.yaml` (each experiment file only overrides what it needs, via `# @package _global_`). To tweak
hyperparameters, either edit/add a file in `configs/experiment/` or override on the CLI
(`python main.py +experiment=qho_fermisets_2d_5N trainer.lr=0.005`).

There is no test suite, linter, or CI configured in this repo.

### Local hardware (default target)

Development runs on a single consumer GPU: nvidia-smi reports an RTX 2080 with **8 GB VRAM**, under WSL2
(the `cuda_executor.cc` driver-version warnings at JAX startup are cosmetic — `jax.devices()` returns the
CUDA device). NetKet enables float64 by default, which is ~1/32 throughput on consumer GPUs — fine for the
small benchmark networks, but do not scale `n_samples` past ~4096 or `hidden_units` past ~128 without
re-estimating memory: with minSR (`use_ntk=True`) the kernel matrix is O(n_samples²) and the backward pass
is chunked by `chunk_size`. The cluster (below) is NOT available in Claude Code sessions.

### Cluster (SLURM, science cluster)

`jobs/install.job` documents the interactive-node environment setup (conda env, CUDA modules,
`jax[cuda12]`/netket/hydra-core install). `jobs/run.job` is the batch submission script
(`sbatch jobs/run.job`), which loads the same conda env and runs `python3 main.py +experiment=<name>`.
Paths in these jobs are hardcoded to `/home/iyakus/scratch/FermiNQS` — keep that in mind if adapting them.

## Architecture

**`main.py`** is the Hydra entrypoint. Per run it:
1. Builds the physical `System` (Hamiltonian) and picks an ansatz based on `cfg.ansatz.model`.
2. Computes the exact analytic QHO ground-state energy (`exact_qho_gs_energy`) as the ground truth to
   compare against — this is the key correctness signal for any given run.
3. Builds a `MetropolisSampler` using a custom transition rule, `SamplerExchangeRule` (defined inline in
   `main.py`), which mixes Gaussian drift moves with random pairwise particle-position exchanges — the
   exchange moves are what make sampling ergodic for fermionic (antisymmetric) wavefunctions.
4. Hands everything to `Trainer` (`src/train.py`) and runs it, then plots the energy-vs-iteration error
   curve via `plots/plot_errs.py`.

**`src/system.py`** wraps NetKet's continuous-space `Particle` Hilbert space + kinetic/potential energy
operators into a `System`. Currently only `potential="qho_no_inter"` (non-interacting harmonic trap) is
implemented; adding a new potential means adding a branch here that defines `v(x)`.

**`src/ansatz.py`** defines the trial wavefunctions, all `flax.nnx.Module`s that output **log** psi (NetKet
convention — never exponentiate inside `__call__`):
- `Gaussian` / `GaussianFermions` — the known-analytic Gaussian ground state, parametrized via
  `Sigma = A^T A` for positive-definiteness. `GaussianFermions` additionally multiplies in a Jastrow-like
  antisymmetric factor (`eta_antisymmetric`, log-space product of pairwise coordinate differences) — mainly
  used as a sanity-check baseline against the exact solution.
- `DeepSetsNN` — permutation-symmetric (bosonic) ansatz: per-particle encoder → sum-pool → decoder.
- `FermiSets` — from Fu (2025) "Fermions = Bosons + One" (arXiv:2510.11431): builds antisymmetry by
  evaluating a symmetric network at `+eta` and `-eta` (the antisymmetric log-factor) and combining with a
  signed log-sum-exp, rather than an explicit Slater determinant.
- All ansätze wrap sums/products of exponentials through `safe_complex_logsumexp` (or equivalent manual
  epsilon-clamping in `eta_antisymmetric`) specifically to avoid `log(0) = nan` when particle positions
  coincide or amplitudes cancel — **this is a recurring numerical-stability failure mode in this codebase,
  not just defensive boilerplate**. If you see NaNs during training, check these paths first.

**`src/train.py`**'s `Trainer` wraps a NetKet `VMC`/`VMC_SR` driver:
- `optimizer="adam"` uses plain `nk.driver.VMC` with optax Adam (+ grad clipping); any other optimizer
  (`sgd`, `momentum`) uses `nk.driver.VMC_SR` (stochastic reconfiguration / natural gradient, `use_ntk=True`
  i.e. "minSR" kernel trick, `mode="complex"`).
- `make_safe_solver` wraps the SR linear solver so that if the natural-gradient update's bilinear form
  blows up past a threshold, the update is zeroed out for that step rather than corrupting the parameters —
  a stability guard against ill-conditioned SR steps, tuned via the `max_bilinear_form` constant.
- When `trainer.validation: true`, `validation_callback` runs periodically (every 50 steps): checkpoints
  params to `<hydra_output_dir>/checkpoints/step_N.mpack` and evaluates energy on a freshly-reseeded
  `MCState` (decorrelated from the training sampler chain) as an independent validation estimate.
- `pretrained_path` (set in an experiment config's `ansatz:` block) loads a previous checkpoint's
  `.mpack` params to warm-start a run — used for sweeps that continue from an earlier converged state.
  Note experiment configs currently hardcode absolute paths to prior checkpoints; these need manual updating
  between sweeps (see the `#DONT FORGET TO REFRESH` comments in configs).

**Outputs**: Hydra writes each run to `outputs/<date>/<time>/`, containing `optimization_results.log` (JSON
lines, NetKet's logger) and `checkpoints/`. `plots/` contains both plotting utility modules
(`plot_errs.py`, `plot_wf.py`, etc.) and the generated PNGs themselves (not separated into a build artifact
directory — treat existing PNGs in `plots/` as historical output, not something to clean up).

## Benchmark experiment (canonical)

**System: N=3 spinless fermions, dim=2, non-interacting harmonic trap. Exact ground-state energy E = 5.0 ħω.**
Config: `configs/experiment/qho_fermisets_2d_3N_bench.yaml`. This is THE benchmark — all bug hunting and
optimization work is measured against it before anything else is tried. A second, no-symmetry variant
exists: `qho_fermisets_2d_3N_aniso.yaml` (anisotropic trap `qho_aniso`, ω_y=1.5, exact GS E = 6.25,
`lz_proj_K` must be 0) — used 2026-07-15 to prove the trap generalizes beyond the QHO (see RESEARCH_LOG).

Why this system and not the others in `configs/experiment/`:
- **Closed shell.** In the 2D trap, shells k = nₓ+n_y have degeneracy k+1; N=3 fills shells 0 and 1
  completely, giving a **non-degenerate** ground state. N=4 and N=5 (the previously-run configs) are open
  shells with degenerate ground states — flat directions in the loss that confound convergence diagnostics
  and make wavefunction-overlap comparisons ill-defined. A benchmark must not have that ambiguity.
- **Smallest nontrivial 2D fermion system.** N=2 is also open-shell (degenerate); N=3 is the minimum where
  the complex-Vandermonde signature encoder does real work. The scaling follow-up is N=6 (next closed
  shell, E = 14.0), only after N=3 is nailed.
- **Fits the 8 GB GPU** with headroom: ~19k parameters, 4096 samples, 512 chains.

**Protocol** — three stages, in order, never skipping:
1. **Stage 0 — structural checks** (`python tests/stage0_sanity.py`, <1 min, must pass before any training):
   exact antisymmetry under all pair swaps, even-permutation invariance, η oddness, collision suppression
   without NaN, finite gradients near collisions, far-field finiteness, exact-energy reference values.
2. **Stage 1 — pipeline shakedown** (~minutes): a short run to verify the full loop end-to-end —
   `python main.py +experiment=qho_fermisets_2d_3N_bench trainer.vmc_iters=60 trainer.n_samples=1024 sampler.n_chains=256`
   — checking: no NaN in energy, energy decreasing from the initial value, checkpoint written at step 50,
   JSON log + error plot produced, acceptance rate sane.
3. **Stage 2 — the benchmark run** (`python main.py +experiment=qho_fermisets_2d_3N_bench`, ~1–2 h locally).

**Success criteria for Stage 2** (all required):
- Relative energy error |E − 5.0|/5.0 ≤ 1e-3 on the **validation** energy (independent chain), with the MC
  error bar consistent with the deviation; stretch goal 3e-4.
- Variance of the local energy decreasing over training (an exact eigenstate has zero variance — energy
  looking right while variance stays large is the classic silent-bug signature).
- Training and validation energies agree within error bars (disagreement = sampler autocorrelation bug).
- The safe-solver bilinear-form guard (`make_safe_solver`) not triggering in late training (frequent
  zero-outs = ill-conditioned SR, treat as failure to diagnose, not noise to ignore).

**Stop rule** (from `claude_code.md`): at most ~3 deliberate hyperparameter/architecture adjustments from
the documented config. If the criteria are still unmet, write up the negative result and the stress-test
summary instead of trying "one more idea."

**Known bugs found during benchmark setup (2026-07-14)** — relevant when interpreting old runs:
- `qho_fermisets_2d_3N.yaml` and `qho_fermisets_2d_5N.yaml` misspell `optimizer` as `optmizer`; Hydra merges
  the typo key silently, so those historical runs used the `train.yaml` default **sgd**, not momentum.
- `FermiSets` dim=1: `__call__` passes `-eta` where eta is a *log*, so the minus branch computes the
  reciprocal, not the sign flip (the correct op, noted in the code comment, is `eta + iπ`). 1D antisymmetry
  is structurally broken. The dim=2 path is correct (eta is the raw product there) — verified by Stage 0.
- `GaussianFermions` with dim=2 crashes (`log_eta` referenced but only assigned in the dim==1 branch;
  `eta_antisymmetric` returns a stub 0 for dim==2).
- `main.py` hardcodes `n_dim=2` in `SamplerExchangeRule` (marked `##TEMP`) — any 1D run with this sampler
  reshapes incorrectly. Harmless for the 2D benchmark.

## Tools

- `tests/stage0_sanity.py` — structural ansatz checks (antisymmetry, collisions, NaN); run before
  any training after touching `src/ansatz.py`.
- `tools/overlap_check.py <ckpt.mpack>` — energy + squared overlaps of an N=3 2D checkpoint with
  the analytic GS, the holomorphic trap state, its conjugate mirror, and `excx` = det{1,x,x²}
  (the real-Vandermonde state; exact first excited state of the anisotropic trap). **Pass
  `--lz-proj-K` matching the training config** — a checkpoint trained with projection evaluates
  to a completely different (wrong) state without it. For `qho_aniso` checkpoints also pass
  `--omega-y` matching `system.omega_y`.
- `tools/pretrain_hf.py --N <n>` — answer-free HF/Slater pretraining (aufbau oscillator orbitals,
  SCF-coefficient hook for future interacting systems); writes an `.mpack` for
  `ansatz.pretrained_path`. `tools/pretrain_gs.py` is its N=3-only predecessor fitting the exact GS
  (diagnostic use only).

## Research log

`RESEARCH_LOG.md` is the idea bank: findings, dead ends, and open hypotheses from past sessions.
**Read it before proposing experiments** — in particular, the "holomorphic trap" entry (2026-07-14)
explains why every from-scratch 2D FermiSets run converges to E = N(N+1)/2 instead of the exact GS,
and lists approaches that are already known not to work. Append observations there; never delete
dead ends.

## Conventions specific to this repo

- Ansätze always operate on flattened particle coordinates `x` of shape `(batch, N*dim)` and internally
  `reshape(-1, N, dim)`.
- Wavefunctions are complex (`logPsi = log(R) + 1j*phase`); NetKet's `mode="complex"` in `VMC_SR` must match
  how the ansatz splits real/imaginary log-amplitude components.
- `dim=2` fermionic antisymmetry (`FermiSets`/`GaussianFermions`) is implemented via complex-plane pairwise
  differences (`z = x + 1j*y`); `dim=1` uses an explicit double loop over particle pairs. Neither is
  vectorized for large N — this is a known perf gap, not an oversight, if you're asked to scale N up.
