"""
30-hour N=6 K=2 marathon orchestrator (user request 2026-07-23). Two arms:

  Arm A (budget)   : qho_fermisets_2d_6N_lz_k2       hidden=64,  n_samples=4096
  Arm B (capacity) : qho_fermisets_2d_6N_lz_k2_wide  hidden=128, n_samples=2048

Both use K=2 (cheapest projection that removes the N=6 holomorphic trap; margin=1).
Question: does the cheap fixed-K projection reach the exact GS E=14.0 given a big
budget (Arm A) and/or more capacity (Arm B), or is there a genuine wall? The 2x2 of
outcomes separates optimisation-budget from representability.

Robustness (the point of running unattended for 30h):
  * 300-iter resumable batches; each batch is a fresh main.py resumed from the newest
    checkpoint via ansatz.pretrained_path (LR/optimizer reset per batch keeps LR from
    decaying to zero and localises any crash to <=300 lost iters).
  * crash (non-zero exit / exception / OOM): retried from the newest good checkpoint.
  * blow-up (final validation E NaN or outside [13.5, 40]): that batch's checkpoints are
    discarded and we roll back to the last GOOD checkpoint, bumping diag_shift one notch
    (0.05 -> 0.1 -> 0.2 -> 0.4) for SR conditioning headroom.
  * fully resumable: on restart it scans existing batch dirs, continues from the newest
    good checkpoint and the cumulative iter count, so if THIS process dies it can just be
    relaunched with the same command.

Per-arm stop conditions: CONVERGED (rel err <= 1e-3 of 14.0), or ITER_CAP reached, or
TIME_BUDGET elapsed. Writes a machine-readable state file after every batch so the
monitor (tools/marathon_status.py) can report without re-deriving anything.
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = "/home/ilya/FermiNQS"
PY = os.path.join(REPO, ".venv", "bin", "python")
EXACT_GS = 14.0
BATCH_ITERS = 300
SHIFT_LADDER = [0.05, 0.1, 0.2, 0.4]
STATE_FILE = os.path.join(REPO, "outputs", "marathon", "campaign_state.json")

ARMS = [
    dict(name="armA_h64",  experiment="qho_fermisets_2d_6N_lz_k2",
         iter_cap=12000, time_budget_h=13.5, base_shift_idx=0),
    dict(name="armB_h128", experiment="qho_fermisets_2d_6N_lz_k2_wide",
         iter_cap=12000, time_budget_h=13.0, base_shift_idx=1),
]


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def read_final_energy(run_dir):
    """(max_iter, final validation-energy real) for a batch dir, or (None, None)."""
    p = os.path.join(run_dir, "optimization_results.log")
    if not os.path.exists(p):
        return None, None
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return None, None
    key = "Validation_Energy" if "Validation_Energy" in d else "Energy"
    try:
        iters = d[key]["iters"]
        mean = d[key]["Mean"]
        mean = mean["real"] if isinstance(mean, dict) else mean
        vals = [(it, m) for it, m in zip(iters, mean) if m is not None and np.isfinite(m)]
        if not vals:
            return (max(iters) if iters else None), None
        return vals[-1]
    except Exception:
        return None, None


def newest_checkpoint(*run_dirs):
    cks = []
    for rd in run_dirs:
        cks += glob.glob(os.path.join(rd, "checkpoints", "step_*.mpack"))
    if not cks:
        return None
    return max(cks, key=os.path.getmtime)


def batch_dirs(arm_base):
    if not os.path.isdir(arm_base):
        return []
    ds = [os.path.join(arm_base, d) for d in os.listdir(arm_base)
          if d.startswith("batch")]
    return sorted(ds, key=lambda d: int(d.rsplit("batch", 1)[1]))


def energy_ok(e):
    return e is not None and np.isfinite(e) and 13.5 <= e <= 40.0


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"arms": {}}


def run_arm(arm, state):
    name = arm["name"]
    arm_base = os.path.join(REPO, "outputs", "marathon", name)
    os.makedirs(arm_base, exist_ok=True)
    time_budget = arm["time_budget_h"] * 3600
    t0 = time.time()

    # ---- resume: reconstruct progress from existing good batches ----
    good_ckpt = None
    iters_done = 0
    shift_idx = arm["base_shift_idx"]
    existing = batch_dirs(arm_base)
    next_batch = 1
    for rd in existing:
        it, e = read_final_energy(rd)
        b = int(rd.rsplit("batch", 1)[1])
        next_batch = max(next_batch, b + 1)
        if energy_ok(e) and newest_checkpoint(rd):
            good_ckpt = newest_checkpoint(rd)
            iters_done += (it + 1 if it is not None else BATCH_ITERS)
    log(f"[{name}] resume: iters_done~{iters_done}, good_ckpt={good_ckpt}, "
        f"next_batch={next_batch}")

    arm_state = state["arms"].setdefault(name, {})
    arm_state.update(experiment=arm["experiment"], iter_cap=arm["iter_cap"],
                     time_budget_h=arm["time_budget_h"])
    fail_streak = 0

    while True:
        if iters_done >= arm["iter_cap"]:
            arm_state["status"] = "iter_cap"
            break
        if time.time() - t0 >= time_budget:
            arm_state["status"] = "time_budget"
            break

        run_dir = os.path.join(arm_base, f"batch{next_batch}")
        shift = SHIFT_LADDER[min(shift_idx, len(SHIFT_LADDER) - 1)]
        cmd = [PY, "main.py", f"+experiment={arm['experiment']}",
               f"trainer.vmc_iters={BATCH_ITERS}",
               f"trainer.diag_shift={shift}",
               f"hydra.run.dir={run_dir}"]
        if good_ckpt:
            cmd.append(f"ansatz.pretrained_path={good_ckpt}")
        console = run_dir + "_console.log"
        os.makedirs(run_dir, exist_ok=True)
        log(f"[{name}] batch {next_batch}: shift={shift} ckpt={'scratch' if not good_ckpt else os.path.basename(good_ckpt)}")

        rc = None
        try:
            with open(console, "w") as lf:
                rc = subprocess.run(cmd, cwd=REPO, stdout=lf,
                                    stderr=subprocess.STDOUT, timeout=4 * 3600).returncode
        except subprocess.TimeoutExpired:
            log(f"[{name}] batch {next_batch} TIMED OUT")
        except Exception as ex:
            log(f"[{name}] batch {next_batch} raised {ex}")

        it, e = read_final_energy(run_dir)
        newest = newest_checkpoint(run_dir)
        log(f"[{name}] batch {next_batch} done: rc={rc} maxiter={it} E={e}")

        if energy_ok(e) and newest:
            good_ckpt = newest
            iters_done += (it + 1 if it is not None else BATCH_ITERS)
            fail_streak = 0
            rel = abs(e - EXACT_GS) / EXACT_GS
            arm_state.update(last_E=float(e), iters_done=int(iters_done),
                             rel_err=float(rel), last_shift=shift,
                             last_ckpt=good_ckpt, updated=time.time())
            save_state(state)
            if rel <= 1e-3:
                arm_state["status"] = "converged"
                log(f"[{name}] CONVERGED E={e:.4f} rel={rel:.2e}")
                break
        else:
            # crash or blow-up: discard this batch, roll back, harden SR
            fail_streak += 1
            shift_idx = min(shift_idx + 1, len(SHIFT_LADDER) - 1)
            log(f"[{name}] bad batch (E={e}); fail_streak={fail_streak}, "
                f"raising shift to {SHIFT_LADDER[shift_idx]}, rolling back")
            arm_state.update(last_bad_E=(None if e is None else float(e)),
                             fail_streak=fail_streak, last_shift=SHIFT_LADDER[shift_idx],
                             updated=time.time())
            save_state(state)
            if fail_streak >= 6:
                arm_state["status"] = "abort_unstable"
                log(f"[{name}] ABORT: 6 consecutive bad batches even at max shift")
                break
        next_batch += 1

    arm_state["elapsed_h"] = (time.time() - t0) / 3600
    save_state(state)
    log(f"[{name}] FINISHED status={arm_state.get('status')} "
        f"E={arm_state.get('last_E')} iters~{arm_state.get('iters_done')}")


def main():
    state = load_state()
    state["pid"] = os.getpid()
    state["heartbeat"] = time.time()
    save_state(state)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for arm in ARMS:
        if only and arm["name"] != only:
            continue
        st = state["arms"].get(arm["name"], {}).get("status")
        if st in ("converged", "iter_cap", "time_budget", "abort_unstable"):
            log(f"[{arm['name']}] already finished ({st}), skipping")
            continue
        run_arm(arm, state)
    state["done"] = True
    state["heartbeat"] = time.time()
    save_state(state)
    log("CAMPAIGN COMPLETE")


if __name__ == "__main__":
    main()
