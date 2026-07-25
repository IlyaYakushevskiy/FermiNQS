"""Quick campaign status for the N=6 K=2 marathon monitor. Prints per-arm energy,
iters, rel-err, fail streak, and whether the orchestrator process is alive."""
import json
import os
import time

REPO = "/home/ilya/FermiNQS"
STATE = os.path.join(REPO, "outputs", "marathon", "campaign_state.json")


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def main():
    if not os.path.exists(STATE):
        print("no campaign_state.json yet")
        return
    with open(STATE) as f:
        s = json.load(f)
    pid = s.get("pid")
    hb = s.get("heartbeat")
    age = (time.time() - hb) / 60 if hb else None
    print(f"orchestrator pid={pid} alive={alive(pid) if pid else '?'} "
          f"heartbeat_age={age:.1f}min  done={s.get('done', False)}")
    for name, a in s.get("arms", {}).items():
        upd = a.get("updated")
        uage = f"{(time.time()-upd)/60:.0f}min ago" if upd else "n/a"
        print(f"  {name}: status={a.get('status','running')} "
              f"E={a.get('last_E')} iters~{a.get('iters_done')} "
              f"rel_err={a.get('rel_err')} shift={a.get('last_shift')} "
              f"fails={a.get('fail_streak',0)} (upd {uage})")


if __name__ == "__main__":
    main()
