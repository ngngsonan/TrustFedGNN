"""Run one re-run campaign with provenance guards, logging and a status file.

    python -m experiments.campaigns.run_campaign --list
    python -m experiments.campaigns.run_campaign --id C05 --smoke
    python -m experiments.campaigns.run_campaign --id C05
    python -m experiments.campaigns.run_campaign --id C01 --threads 2

Guards (a full run refuses to start unless all hold):
  1. Provenance is known and clean. Locally: `git rev-parse HEAD` and
     `git status --porcelain -- . ':!results'` must be empty. On Colab (no
     .git): /content/manifest_local.json from 00_pack.sh must exist with
     dirty=false; its commit is exported as FEDFAIR_GIT_COMMIT for the runners.
  2. If RERUN_EXPECTED_COMMIT is set, the commit must match it -- this is what
     keeps every campaign of the re-run on ONE commit.
  3. gpu-colab campaigns need CUDA (override: --allow-cpu, smoke runs exempt).
  4. An existing output produced by another commit is moved aside to
     <name>.stale-<UTC timestamp> (never deleted) before the run.

Writes:
  results/campaigns/logs/<ID>[_smoke].log       full stdout+stderr
  results/campaigns/status/<ID>[_smoke].json    state, commit, timing, exit codes, output sha256
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from experiments.campaigns.campaigns import CAMPAIGNS, BY_ID, OUT  # noqa: E402

COLAB_MANIFEST = "/content/manifest_local.json"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance() -> dict:
    """Return {commit, dirty, source}. Exports FEDFAIR_GIT_* on Colab."""
    if os.path.exists(COLAB_MANIFEST):
        with open(COLAB_MANIFEST) as f:
            man = json.load(f)
        commit, dirty = man.get("commit", "unknown"), bool(man.get("dirty", True))
        os.environ["FEDFAIR_GIT_COMMIT"] = commit
        os.environ["FEDFAIR_GIT_DIRTY"] = "1" if dirty else "0"
        return {"commit": commit, "dirty": dirty, "source": COLAB_MANIFEST,
                "tarball_sha256": man.get("tarball_sha256")}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--", ".", ":!results"],
                                         cwd=REPO, text=True).strip()
        return {"commit": commit, "dirty": bool(status), "source": "git", "dirty_files": status.splitlines()[:20]}
    except Exception as e:  # pragma: no cover
        return {"commit": "unknown", "dirty": True, "source": f"error: {e}"}


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def list_campaigns() -> None:
    print(f"{'ID':4s} {'Pri':3s} {'Where':10s} {'Findings':22s} {'Estimate':55s} Title")
    for c in CAMPAIGNS:
        print(f"{c.id:4s} {c.priority:3s} {c.where:10s} {','.join(c.findings):22s} {c.est_minutes[:55]:55s} {c.title}")


def move_stale(path: str, commit: str) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path) as f:
            d = json.load(f)
        m = d.get("manifest") or d.get("_manifest") or {}
        old = m.get("git_commit")
    except Exception:
        old = None
    if old == commit:
        print(f"[resume] {path} exists from this commit -- the runner may resume from it.", flush=True)
        return
    dst = f"{path}.stale-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    shutil.move(path, dst)
    print(f"[stale] {path} (commit {old}) moved to {dst}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--id")
    ap.add_argument("--smoke", action="store_true", help="run the cheap smoke variant")
    ap.add_argument("--threads", type=int, default=None, help="OMP/MKL threads per process")
    ap.add_argument("--allow-cpu", action="store_true", help="run a gpu-colab campaign on CPU (slow)")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke runs only")
    ap.add_argument("--append-args", nargs=argparse.REMAINDER, default=None,
                    help="recovery only: extra argv appended to a single-command campaign, "
                         "e.g. --append-args --rules fltrust cgsv  (logged in the status file)")
    a = ap.parse_args()

    if a.list or not a.id:
        list_campaigns()
        return 0
    c = BY_ID.get(a.id)
    if c is None:
        print(f"unknown campaign {a.id}; see --list", file=sys.stderr)
        return 2
    cmds, outputs = (c.smoke_cmds, c.smoke_outputs) if a.smoke else (c.cmds, c.outputs)
    if a.append_args:
        if len(cmds) != 1:
            print("--append-args only applies to single-command campaigns", file=sys.stderr)
            return 2
        cmds = (list(cmds[0]) + list(a.append_args),)
    if not cmds:
        print(f"{c.id} has no {'smoke' if a.smoke else 'full'} command (see campaigns.py notes).", file=sys.stderr)
        return 2

    os.chdir(REPO)
    prov = provenance()
    tag = f"{c.id}_smoke" if a.smoke else c.id
    log_dir, status_dir = os.path.join(OUT, "logs"), os.path.join(OUT, "status")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(status_dir, exist_ok=True)
    status_path = os.path.join(status_dir, f"{tag}.json")
    log_path = os.path.join(log_dir, f"{tag}.log")

    # ---- guards -----------------------------------------------------------
    problems = []
    if prov["commit"] == "unknown":
        problems.append("git commit unknown")
    if prov["dirty"] and not (a.smoke and a.allow_dirty):
        problems.append(f"working tree dirty ({prov.get('dirty_files') or prov['source']})")
    expected = os.environ.get("RERUN_EXPECTED_COMMIT")
    if expected and not prov["commit"].startswith(expected) and not a.smoke:
        problems.append(f"commit {prov['commit'][:10]} != RERUN_EXPECTED_COMMIT {expected}")
    if c.where == "gpu-colab" and not a.smoke and not a.allow_cpu and not cuda_available():
        problems.append("gpu-colab campaign but CUDA is not available")
    if problems:
        for p in problems:
            print(f"[ABORT] {p}", file=sys.stderr)
        with open(status_path, "w") as f:
            json.dump({"id": c.id, "smoke": a.smoke, "state": "refused", "problems": problems,
                       "provenance": prov, "time": now()}, f, indent=2)
        return 3

    if not a.smoke and not a.append_args:  # a recovery run must keep the partial artifact
        for p in outputs:
            move_stale(p, prov["commit"])

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    env.update(c.env)
    if c.env.get("FEDFAIR_DEVICE") == "cuda" and (a.smoke or a.allow_cpu) and not cuda_available():
        env["FEDFAIR_DEVICE"] = "cpu"
    if a.threads:
        env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = str(a.threads)
    env["PYTHONUNBUFFERED"] = "1"

    status = {"id": c.id, "title": c.title, "smoke": a.smoke, "state": "running", "started": now(),
              "provenance": prov, "platform": platform.platform(),
              "python": platform.python_version(), "cuda_available": cuda_available(),
              "threads": a.threads, "env_overrides": c.env, "cmds": [list(x) for x in cmds],
              "append_args": a.append_args,
              "exit_codes": [], "outputs": {}}

    def save_status():
        tmp = status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp, status_path)

    save_status()
    t0 = time.perf_counter()
    with open(log_path, "a") as log:
        log.write(f"\n===== {tag} START {now()} commit={prov['commit']} dirty={prov['dirty']} =====\n")
        for cmd in cmds:
            argv = [sys.executable if x == "python" else x for x in cmd]
            log.write(f"$ {' '.join(argv)}\n")
            log.flush()
            print(f"[run] {' '.join(cmd)}", flush=True)
            p = subprocess.Popen(argv, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
            # Flush per line: the log and the lane*.out files are the monitoring
            # surface, and block buffering made them look stalled for minutes.
            for line in p.stdout:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            p.wait()
            log.flush()
            status["exit_codes"].append(p.returncode)
            save_status()
            if p.returncode != 0:
                break
        log.write(f"===== {tag} END {now()} exit={status['exit_codes']} =====\n")

    status["wall_clock_min"] = round((time.perf_counter() - t0) / 60, 2)
    ok = all(rc == 0 for rc in status["exit_codes"]) and len(status["exit_codes"]) == len(cmds)
    for p in outputs:
        status["outputs"][p] = ({"sha256": sha256(p), "bytes": os.path.getsize(p)} if os.path.exists(p)
                                else None)
    missing = [p for p, v in status["outputs"].items() if v is None]
    status["state"] = "done" if ok and not missing else "failed"
    if missing:
        status["missing_outputs"] = missing
    status["finished"] = now()
    save_status()
    print(f"[{status['state'].upper()}] {tag} in {status['wall_clock_min']} min; status -> {status_path}",
          flush=True)
    return 0 if status["state"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
