"""
QUEUE.md P2 follow-up (user request 2026-07-20): bounded continuation of the N=6
interacting-dot FermiSets+K=6 run, to test whether it converges more cleanly than the
N=6 QHO did. Motivation (RESEARCH_LOG.md 2026-07-20): the QHO's noisy plateau may be an
artifact of the non-interacting system's massive near-degeneracy; the dot's Gaussian
repulsion generically lifts that degeneracy, so it might not show the same plateau.

Deliberately BOUNDED (MAX_BATCHES=6, not 15 like the QHO run): this is a falsification
test, not an open-ended chase. Same checkpoint-resume-in-500-iter-batches protocol as
tools/run_until_converged.py, stopping on:
  - CONVERGED: within 5% relative of the ED reference (19.0038)
  - PLATEAU: local slope (last 300 iters) near-zero for two consecutive batches
  - MAX_BATCHES: 6 batches = 3000 more iters, then stop regardless and report

Resumes from the existing run's final checkpoint (outputs/2026-07-16/17-05-18,
E=20.187+-0.048 at step 800, marked "not converged, no plateau" when it was stopped).
"""
import json
import os
import subprocess

import numpy as np

REPO = "/home/ilya/FermiNQS"
EXACT_GS = 19.0038
MAX_BATCHES = 6
BATCH_ITERS = 500
SLOPE_FLAT_THRESHOLD = 0.0005

RUN_BASE = os.path.join(REPO, "outputs", "converge_dot")


def batch_complete(run_dir):
    log_path = os.path.join(run_dir, "optimization_results.log")
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path) as f:
            data = json.load(f)
        return max(data["Energy"]["iters"]) >= BATCH_ITERS - 1
    except Exception:
        return False


def run_batch(experiment, pretrained_path, batch_idx):
    run_dir = os.path.join(RUN_BASE, f"k6_batch{batch_idx}")
    if batch_complete(run_dir):
        print(f"[dot k6 batch {batch_idx}] already complete, skipping (resume)", flush=True)
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
    print(f"[dot k6 batch {batch_idx}] launching: {' '.join(cmd)}", flush=True)
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


def converge_dot(experiment, start_ckpt):
    ckpt = start_ckpt
    prev_slope = None
    plateau_streak = 0
    history = []
    for b in range(1, MAX_BATCHES + 1):
        run_dir = run_batch(experiment, ckpt, b)
        iters, means = load_validation(run_dir)
        e_now = float(means[-1])
        s = local_slope(iters, means)
        history.append((b, e_now, s))
        print(f"[dot k6 batch {b}] E={e_now:.4f}  slope(last 300)={s:.6f}", flush=True)

        rel_err = abs(e_now - EXACT_GS) / EXACT_GS
        if rel_err <= 0.05:
            print(f"[dot k6] CONVERGED at batch {b}: E={e_now:.4f} (rel err {rel_err:.3%})", flush=True)
            return "converged", run_dir, history

        if prev_slope is not None and abs(s) < SLOPE_FLAT_THRESHOLD and abs(prev_slope) < SLOPE_FLAT_THRESHOLD:
            plateau_streak += 1
        else:
            plateau_streak = 0
        prev_slope = s

        if plateau_streak >= 1:
            print(f"[dot k6] PLATEAU confirmed at batch {b}: E={e_now:.4f}", flush=True)
            return "plateau", run_dir, history

        ckpt = latest_checkpoint(run_dir)

    print(f"[dot k6] hit MAX_BATCHES ({MAX_BATCHES}) without a verdict", flush=True)
    return "max_batches", run_dir, history


if __name__ == "__main__":
    print("=== DOT K=6 ===", flush=True)
    status, run_dir, history = converge_dot(
        "dot_gauss_2d_6N_lz",
        start_ckpt=f"{REPO}/outputs/2026-07-16/17-05-18/checkpoints/step_800.mpack",
    )
    print(f"DOT K=6 VERDICT: {status} ({run_dir})", flush=True)
    print(f"DOT K=6 history: {history}", flush=True)
    print("ALL DONE", flush=True)
