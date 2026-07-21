# Project queue

**STRATEGIC PIVOT (2026-07-17, user decision)**: thesis focus is the Laughlin/LLL trap
identification + L_z projection as the rigorous fix + systematic ablations — NOT further
investment in the deflation-penalty regularizer. Reasoning: the penalty result is real
but epistemically weak for a thesis (took 3 attempts to find a working formulation, its
gradient is an explicit approximation missing the score-function term, calibrated by
trial-and-error, single-ish seed/single system) — whereas the projection result has a
hard representability guarantee and is validated across 4 systems. Deflation work is now
"finish what's already running in the background, mention as one future-work paragraph,
don't invest further." Ablations (K-vs-N, symmetry, interaction, complexity-vs-Slater)
are the new priority.

Explicit, prioritized backlog. Convention going forward: **always draw the next piece of
work from here**, top to bottom, unless the user redirects. Move items down (not delete)
when superseded; append new ideas at the bottom of their priority tier. Completed items
move to RESEARCH_LOG.md with results, and get struck through here with a pointer.

Priority order = top of file. "In progress" items show their run dir / PID so a fresh
session can check status without re-deriving anything.

---

## P0 — K-vs-N margin ablation (cheap, high value, directly supports the thesis narrative)

Settled analytically (`tools/lz_margin.py`, RESEARCH_LOG 2026-07-16): required K does NOT
scale with N (checked to N=210), but a FIXED K is not safe for all N (K=6 fails completely
at N=21,28,36,45...). N=6 conveniently gives a same-system A/B/C test for free (K=6
margin=3 already validated; K=3 or K=5 margin=0 predicted total failure; K=4 margin=1 thin).

- [x] **RESOLVED 2026-07-19 (result reframed, not the planned A/B/C)**: K3-cont/K4-cont
      finished (+500 iters each from checkpoint, ~1000 cumulative). K=3: 17.208±0.063.
      K=4: 17.458±0.070. K=6 (same cumulative budget, 2026-07-16): 17.408±0.051. **All
      three plateau in the same 17.2-17.5 band** — margin made no visible difference in
      this budget. The "K=6 validated" claim below was actually the N=3 system, not N=6
      — no tested K has converged N=6 QHO to 14.0 yet. Live hypothesis (user's own,
      posed independently): L_z=12 survives projection under K=3, K=4, AND K=6
      (LCM=12) and could be a shared intermediate attractor — NOT yet confirmed,
      `tools/ed_dot.py` only blocks by parity not L_z. See RESEARCH_LOG.md 2026-07-19
      for full writeup and next-step options. Superseded by SlaterNN baseline result
      (P2): SlaterNN nails N=6 QHO to 0.004% in 800 iters from scratch — FermiSets
      hasn't beaten bare Slater on any system at N=6 yet, accuracy or speed.
      **Figure**: `plots/n6_margin_ablation_vs_slater.png` (`plots/plot_n6_margin_ablation.py`,
      re-runnable) — all three K trajectories track together, still slowly descending
      (NOT a hard plateau) at 900-1000 cumulative iters, vs SlaterNN's clean drop to
      14.0 by step ~50. Caveat for the writeup: since K=3/4/6 are still descending, not
      flat, "stuck" is not proven yet — could still be an iteration-budget question,
      same as SlaterNN needed 0 extra tuning but FermiSets might just be slower here.
- [x] **CLOSED 2026-07-20 (user decision): do not pursue N=10.** N=10 K=6 scaling
      attempt (2026-07-19) FAILED with catastrophic SR instability (two auto-rollback
      retries, both re-diverged, final E≈3.67e28 — full diagnosis in RESEARCH_LOG.md
      2026-07-19). Was only worth revisiting if N=6 (K<N) showed FermiSets converging in
      practice — it doesn't (see next bullet), so N=10 is closed, no further GPU time.
- [x] **CLOSED 2026-07-20: K=3 extended to 10 batches (5000 iters), then killed —
      confirmed noisy plateau, not real convergence.** Trajectory 17.0→16.7 (batches 1-4,
      real descent) → noisy 16.3-16.7 band with no further net progress (batches 5-10,
      one batch even ticked slightly positive). K=6 never restarted — SlaterNN already
      solves this system to 0.004% from scratch, no point grinding K=3/K=6 further to
      confirm what the trend already shows. **Overlap diagnostic**
      (`tools/overlap_check_n6.py`, RESEARCH_LOG.md 2026-07-20) falsifies the natural
      "second lazy trap" guess: the batch-10 checkpoint has 31% overlap with the true
      GS (not ~0), 0.18% with the excluded holo trap (correctly gone), and only ~0.8-0.9%
      with each of two candidate `E=16, L_z=0` "next-easiest" states — so it's not stuck
      on a single wrong eigenstate, it's a genuine partial superposition (real GS content
      plus a long thin tail across many sector eigenstates). Projection solves *which
      sector*, not *how hard the GS's non-factorizable sign structure is to reach inside
      it*. This is the headline finding for the thesis's honest limits section.
- [ ] **PAUSED 2026-07-18 22:57 (user shut down machine, jobs killed on request)** — status: (historical, superseded by resolution above)
      - `qho_fermisets_2d_6N_lz_k3` (K=3, margin=0) **completed all 500/500 iters cleanly**
        (`outputs/2026-07-18/20-36-14`). Final: **18.198−0.061j ± 0.065** [σ²≈17].
        **Does NOT match the naive prediction** ("total failure, reproduces E→21 trap") —
        it settled around 18, well below the trap (21) and well above GS (14), with no
        sign of the trap-energy plateau at all in the validation trace (26.0 → 21.1 →
        20.5 → 20.0 → 21.2 → 19.5 → 18.6 → 18.5 → 18.5 → 18.2, i.e. monotonic-ish descent
        very similar in shape to the K=6 run's own trajectory). **Needs interpretation
        tomorrow, not yet understood**: margin=0 means L_z=15 (the trap) technically
        survives the K=3 projection sector, so the naive "total failure" prediction assumed
        the trap would be reproduced exactly — instead something else happened. Possible
        explanations to check: (a) K=3 still excludes OTHER lazy-family members (d=1,2 etc.
        survive too since margin=0 only describes d=0), so the sector isn't as unconstrained
        as assumed and partial projection benefit still applies; (b) 500 iters may just not
        be enough to fully settle into L_z=15 specifically (same "N=6 needs more iters"
        pattern seen in the unprojected scaling runs); (c) re-derive margin(N,K) logic —
        double check it's not off-by-one for what "total failure" should look like in
        practice. Do not update the thesis narrative from this run until this is resolved.
      - `qho_fermisets_2d_6N_lz_k4` (K=4, margin=1) **killed mid-run at iter 353/800**
        (`outputs/2026-07-18/21-48-29`, checkpoint `step_350.mpack` saved). Trend at kill
        time: energy climbing 20.2 → 20.6-20.7 over the last ~10 iters, σ² falling
        (7.9→5.0) — inconclusive, nowhere near settled. **Resume tomorrow** via
        `ansatz.pretrained_path=.../21-48-29/checkpoints/step_350.mpack`, remaining
        ~450 iters (or just relaunch from scratch if a clean run is preferred — cheap
        either way for N=6/K=4).
      - Console logs preserved: `outputs/n6_lz_k3_console.log` (complete),
        `outputs/n6_lz_k4_console.log` (partial, ends at iter 353).
      - **Next session**: (1) resolve the K=3 interpretation question above, (2) resume/
        rerun K=4 to completion, (3) only then write the K-vs-N ablation table and move to
        P1. Lesson reconfirmed: `setsid nohup ... & disown` does NOT survive a machine
        shutdown (obviously) — always expect to relaunch from checkpoint after a planned
        machine-off, this is normal, not a bug.

---

## P1 — wall-clock complexity ablation (cheap, no training, supports the O(K·N²) vs O(N³) claim) — DONE

- [x] **DONE 2026-07-19** (`tools/timing_ablation.py`, RESEARCH_LOG.md same date). N=3..30
      (this thesis's actual trainable range): NO crossover, FermiSets 15-65x slower —
      wall-clock there is network-width-bound (hidden_units=64, K=4-7 repeats), not
      N-scaling-bound; real finding, must state plainly. Extended to N=50/100/200:
      crossover found between N=50 (Slater ~2x faster) and N=100 (FermiSets ~3x faster,
      ~6.4x by N=200) — confirms the O(K·N²) vs O(N³) shape empirically, just outside
      this thesis's own trainable N. Caveat: bare `slogdet` only, no orbital NN (P2 would
      add constant cost to the Slater side too, could shift the crossover later).

---

## P2 — Slater-determinant baseline, accuracy comparison

Recommended design: **bare NN orbitals + `slogdet`, no backflow** first — isolates "does
antisymmetry-by-construction avoid the holomorphic trap at all" cleanly. Backflow
(FermiNet/PauliNet-style) only as a follow-up if bare Slater underperforms on the dot.

- [x] **Built 2026-07-19**: `SlaterNN` in `src/ansatz.py` — shared per-particle MLP
      (dim -> hidden_units -> N) gives orbitals[i,k] = phi_k(x_i), `jnp.linalg.slogdet`
      of the N×N matrix, same Gaussian-envelope convention as `FermiSets`. Real orbitals
      -> discrete (0/pi) phase from slogdet's sign, same as any real-orbital Slater in
      VMC (PauliNet/FermiNet-style) — expected, not a bug. Wired into `main.py` as
      `ansatz.model: fermi_slater_nn` (needs "fermi" substring for the
      `is_fermionic`/statistics dispatch). Sanity-checked in
      `tests/stage0_sanity_slater.py` (antisymmetry under all pair swaps, N-cycle parity,
      collision -> -inf-scale not NaN, at N=3 and N=6) — all pass. 20-iter CPU/GPU
      shakedown at N=3 (`qho_slater_2d_3N_bench`, reduced samples/chains): energy
      7.7 -> 5.19 in 20 iters (exact GS 5.0) — very fast, clean convergence, no NaN,
      full pipeline (checkpoint/plot) verified working end-to-end.
- [x] Note for the writeup regardless of results: Slater gets L_z symmetry FOR FREE by
      choosing orbitals with definite angular momentum (zero extra cost); FermiSets pays
      O(K) for the same guarantee via projection. State this asymmetry plainly — it
      stands even if this ansatz is never built (a valid textual point in the thesis).
- [x] **DONE 2026-07-19**: N=3 QHO 4.99997±0.00018 (0.0006%), N=6 QHO 13.9994±0.00035
      (0.004%), N=6 dot 19.165±0.015 (0.85% vs ED 19.0038). Slater avoids the
      trap-plateau entirely on both QHO systems (expected — no signature encoder to get
      lazy about) and beats FermiSets+K=6's best dot result so far (20.187±0.048, 6.2%
      err). Full writeup + framing (QHO is a non-issue by construction — the exact GS
      IS a single Slater determinant there; the dot is the only test that's actually
      informative about correlation) in RESEARCH_LOG.md 2026-07-19.
- [ ] **Open question raised by these results**: FermiSets has not yet beaten bare
      Slater on accuracy OR speed on any system tested at N=6. If FermiSets/L_z-trick is
      going to justify itself for the thesis, the interacting dot is the only place it
      still could — needs either (a) a longer/better-tuned FermiSets dot run to see if
      it can close the gap, or (b) a more strongly-correlated system where a single
      Slater determinant's representability ceiling actually bites. Decide with user
      before spending more GPU time here.
      **CLOSED 2026-07-21**: ran the bounded continuation (`tools/run_dot_until_converged.py`,
      capped at 6 batches). Killed by user after 3 batches (1500 extra iters) — same
      qualitative noisy-non-progress signature as the QHO: E went 20.187 -> 20.109 ->
      20.100 -> back up to ~20.18 (batch 3's own early readings), net ~0 progress.
      Falsifies the "interactions lift the degeneracy and help" hypothesis within the
      budget tested. Full writeup RESEARCH_LOG.md 2026-07-20/21. **No further GPU time on
      the dot** — FermiSets+L_z-projection does not beat SlaterNN on accuracy or speed on
      any system tested (N=3/6 QHO, N=6 dot). The thesis conclusion: projection is a real,
      provable fix for the wrong-sector pathology (clean win at N=3), but not sufficient
      by itself to make the true GS's non-factorizable sign structure easy to reach as N
      grows, interacting or not.

---

## P3 — deflation-penalty regularizer (DEMOTED — finish in-flight work only, no further investment)

Kept for the record; not a thesis pillar (see pivot note at top). The physics is
interesting and the result is real, but it doesn't meet the rigor bar this thesis is
built around (see reasoning at top of file) — treat as one future-work paragraph, not a
result to keep polishing.

- [x] **v1 (differential CR-defect / Laplacian) FAILED, diagnosed precisely**: a plateaued
      N=4 no-projection run (`holo_penalty`, first-order D) converged cleanly to E=10 (the
      trap energy) with D elevated — looked like progress. A 2nd-order Laplacian upgrade
      (`laplacian_defect_batch`, `tools/holomorphy_defect.py`) showed the plateaued state
      scored AS HIGH AS the actual analytic true GS on both D and L — proof that no local
      differential quantity can distinguish genuine GS-ward progress from remixing within
      the exactly-degenerate {holo, antiholo} subspace (any reflection-symmetric potential
      has this degeneracy). Both diagnostics kept in `tools/holomorphy_defect.py` for
      reference, just not usable alone as a training penalty.
- [x] **v2 (projector/deflation against the KNOWN {holo, antiholo} subspace)**:
      `src/deflation_penalty.py` — GLOBAL Choo-Carleo overlap estimator, y-samples via
      complex-Ginibre-ensemble eigenvalues (exact, no MCMC). Config keys
      `trainer.deflation_penalty{,_mu0,_decay,_lr,_n_ginibre}`.
- [x] **Result, seed 42, canonical GPU resources** (`outputs/2026-07-16/21-12-11`, 1500
      iters): **8.361 ± 0.029**, clean plateau 8.38-8.44 from step ~1150-1450 — vs.
      historical no-penalty plateau 10.03-10.08 (exact-but-degenerate GS = 8.0). First
      from-scratch escape with NEITHER L_z projection NOR a known exact target state.
- [ ] **Multi-seed replication — IN PROGRESS, let finish, do not extend further**:
      chained background job, seed=43 then seed=44, same config/1500 iters each. Console
      logs `outputs/n4_deflate_seed{43,44}_console.log`. When both land: one sentence in
      RESEARCH_LOG (typical range across 3 seeds), then STOP — no new seeds, no new
      systems, no re-calibration, unless the user explicitly asks to revisit this.

---

## Done (moved here only as a pointer; full writeups live in RESEARCH_LOG.md)

- K-vs-N margin analysis + `tools/lz_margin.py` — 2026-07-16.
- Holomorphy-defect diagnostic tool + 4-checkpoint validation — 2026-07-16.
- N=6 QHO (System A) scaling run: E=17.66, not yet converged to 14.0, no plateau — 2026-07-16.
- N=6 interacting dot (System B) scaling run: E=20.187±0.048 vs e_ref 19.0038 (~6.2%),
  not yet converged, no plateau, purely an iteration-budget question — 2026-07-16/17.
