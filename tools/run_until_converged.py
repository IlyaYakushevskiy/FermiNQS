"""
QUEUE.md P0 follow-up (user request 2026-07-19): resume K=3 then K=6 (N=6 QHO,
L_z-margin ablation) in 500-iter batches, chained automatically from the latest
checkpoint each time, until one of:
  - CONVERGED: validation energy within 5% relative of the exact GS (14.0)
  - PLATEAU: the local slope (linear fit over the last 300 iters) stays below
    a near-zero threshold for two consecutive batches in a row
  - MAX_BATCHES: a safety cap (6 batches = 3000 more iters) is hit either way,
    so this can't run forever if neither condition is ever met cleanly.
Runs K=3 fully to a verdict before starting K=6 (user-specified order).
"""
import json
import os
import subprocess
import sys

import numpy as np

REPO = "/home/ilya/FermiNQS"
EXACT_GS = 14.0
MAX_BATCHES = 15  # bumped 2026-07-20: user wants longer runs before giving up on convergence
BATCH_ITERS = 500
SLOPE_FLAT_THRESHOLD = 0.0005  # energy units per iter


def batch_complete(run_dir):
    """A batch is complete if its log exists and recorded the final iteration."""
    log_path = os.path.join(run_dir, "optimization_results.log")
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path) as f:
            data = json.load(f)
        return max(data["Energy"]["iters"]) >= BATCH_ITERS - 1
    except Exception:
        return False


def run_batch(k_label, experiment, pretrained_path, batch_idx):
    run_dir = os.path.join(REPO, "outputs", "converge", f"k{k_label}_batch{batch_idx}")
    if batch_complete(run_dir):
        print(f"[k{k_label} batch {batch_idx}] already complete, skipping (resume)", flush=True)
        return run_dir
    os.makedirs(run_dir, exist_ok=True)
    console_log = run_dir + "_console.log"
    cmd = [
        os.path.join(REPO, ".venv", "bin", "python"), "main.py",
        f"+experiment={experiment}",
        f"trainer.vmc_iters={BATCH_ITERS}",
        f"ansatz.pretrained_path={pretrained_path}",
        f"hydra.run.dir={run_dir}",
    ]
    print(f"[k{k_label} batch {batch_idx}] launching: {' '.join(cmd)}", flush=True)
    with open(console_log, "w") as logf:
        subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT, check=True)
    return run_dir


def load_validation(run_dir):
    with open(os.path.join(run_dir, "optimization_results.log")) as f:
        data = json.load(f)
    ve = data["Validation_Energy"]
    return np.array(ve["iters"], dtype=float), np.array(ve["Mean"]["real"], dtype=float)


def latest_checkpoint(run_dir):
    ckdir = os.path.join(run_dir, "checkpoints")
    files = [f for f in os.listdir(ckdir) if f.startswith("step_")]
    files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(ckdir, files[-1])


def local_slope(iters, means, window=300):
    mask = iters >= (iters[-1] - window)
    if mask.sum() < 2:
        return 0.0
    A = np.vstack([iters[mask], np.ones(mask.sum())]).T
    s, _ = np.linalg.lstsq(A, means[mask], rcond=None)[0]
    return float(s)


def converge_K(k_label, experiment, start_ckpt):
    ckpt = start_ckpt
    prev_slope = None
    plateau_streak = 0
    history = []
    for b in range(1, MAX_BATCHES + 1):
        run_dir = run_batch(k_label, experiment, ckpt, b)
        iters, means = load_validation(run_dir)
        e_now = float(means[-1])
        s = local_slope(iters, means)
        history.append((b, e_now, s))
        print(f"[k{k_label} batch {b}] E={e_now:.4f}  slope(last 300)={s:.6f}", flush=True)

        rel_err = abs(e_now - EXACT_GS) / EXACT_GS
        if rel_err <= 0.05:
            print(f"[k{k_label}] CONVERGED at batch {b}: E={e_now:.4f} (rel err {rel_err:.3%})", flush=True)
            return "converged", run_dir, history

        if prev_slope is not None and abs(s) < SLOPE_FLAT_THRESHOLD and abs(prev_slope) < SLOPE_FLAT_THRESHOLD:
            plateau_streak += 1
        else:
            plateau_streak = 0
        prev_slope = s

        if plateau_streak >= 1:
            print(f"[k{k_label}] PLATEAU confirmed at batch {b}: E={e_now:.4f}", flush=True)
            return "plateau", run_dir, history

        ckpt = latest_checkpoint(run_dir)

    print(f"[k{k_label}] hit MAX_BATCHES ({MAX_BATCHES}) without a verdict", flush=True)
    return "max_batches", run_dir, history


if __name__ == "__main__":
    print("=== K=3 ===", flush=True)
    status3, dir3, hist3 = converge_K(
        "3", "qho_fermisets_2d_6N_lz_k3_cont",
        start_ckpt=f"{REPO}/outputs/2026-07-19/07-46-05/checkpoints/step_450.mpack",
    )
    print(f"K=3 VERDICT: {status3} ({dir3})", flush=True)
    print(f"K=3 history: {hist3}", flush=True)

    print("=== K=6 ===", flush=True)
    status6, dir6, hist6 = converge_K(
        "6", "qho_fermisets_2d_6N_lz",
        start_ckpt=f"{REPO}/outputs/2026-07-16/11-45-06/checkpoints/step_450.mpack",
    )
    print(f"K=6 VERDICT: {status6} ({dir6})", flush=True)
    print(f"K=6 history: {hist6}", flush=True)

    print("ALL DONE", flush=True)
