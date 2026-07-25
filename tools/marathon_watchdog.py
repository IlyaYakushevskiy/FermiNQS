"""Self-healing watchdog for the N=6 K=2 marathon (user: 'keep the monitors running to
reset the batch'). The orchestrator (marathon_n6_k2.py) already retries per-batch crashes;
this second layer relaunches the ORCHESTRATOR process itself if it dies, so the whole
campaign survives 30h unattended. Checks every CHECK_S seconds:
  * if campaign_state.json says done -> exit.
  * if the recorded orchestrator pid is not alive and not done -> relaunch it (it resumes
    from the newest good checkpoint on its own) and record the event.
Runs at most MAX_H hours then exits. Idempotent: safe to run one at a time.
"""
import json
import os
import subprocess
import time

REPO = "/home/ilya/FermiNQS"
PY = os.path.join(REPO, ".venv", "bin", "python")
STATE = os.path.join(REPO, "outputs", "marathon", "campaign_state.json")
ORCH_LOG = os.path.join(REPO, "logs", "marathon_orchestrator.log")
WD_LOG = os.path.join(REPO, "logs", "marathon_watchdog.log")
CHECK_S = 300
MAX_H = 31


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(WD_LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def read_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return None


def relaunch():
    # append orchestrator output so we keep the full history
    with open(ORCH_LOG, "a") as lf:
        lf.write(f"\n=== watchdog relaunch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        p = subprocess.Popen([PY, "-u", os.path.join(REPO, "tools", "marathon_n6_k2.py")],
                             cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)
    log(f"relaunched orchestrator, new pid={p.pid}")


def main():
    t0 = time.time()
    log("watchdog started")
    misses = 0
    while time.time() - t0 < MAX_H * 3600:
        st = read_state()
        if st is None:
            misses += 1
            log(f"no state file yet (miss {misses})")
        elif st.get("done"):
            log("campaign done -> watchdog exiting")
            return
        else:
            pid = st.get("pid")
            if pid and not alive(pid):
                log(f"orchestrator pid={pid} DEAD and not done -> relaunching")
                relaunch()
            # else: healthy, stay quiet
        time.sleep(CHECK_S)
    log("watchdog hit MAX_H -> exiting")


if __name__ == "__main__":
    main()
