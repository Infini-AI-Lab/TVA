#!/usr/bin/env python3
"""The verifier of a TVA sandbox task (PROTOCOL.md § Verification). It runs in a container of its own, made
from the same TVA image, and reads only the record the review service exported (/export).

The reward is 1 when the exported kernel, put into a fresh copy of the TVA tree, passes the gate run here,
and the certificate holds: every assigned reviewer accepts every item at the gate's digests, no blocking
finding is unresolved, and the ledger follows from its recorded files. The verifier also checks that the
kernel's fixed fields are the task's, that the ledger is the one the service wrote (its sha256, which
only the service's root process records), and that the reviewers are the task's. It writes
/logs/verifier/reward.json (numbers only) and /logs/verifier/details.json (the reasons).
"""
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import sys

ROOT = "/home/tva/fpv/tva"
EXPORT = os.environ.get("TVA_SANDBOX_EXPORT", "/export")
OUT = "/logs/verifier"
TASK = json.load(open("/tests/task.json"))
K = TASK["kernel"]
KD = os.path.join(ROOT, "kernels", K)
PATH = "/opt/elan/bin:/usr/local/bin:/usr/bin:/bin"

details = {"kernel": K, "checks": []}
reward = {"reward": 0, "gate_pass": 0, "certificate": 0}


def finish():
    os.makedirs(OUT, exist_ok=True)
    reward["reward"] = 1 if reward["gate_pass"] and reward["certificate"] and all(c["ok"] for c in details["checks"]) else 0
    with open(os.path.join(OUT, "details.json"), "w") as f:
        json.dump(details, f, indent=2)
    with open(os.path.join(OUT, "reward.json"), "w") as f:
        json.dump(reward, f)
    print(json.dumps(reward))
    sys.exit(0)


def check(name, ok, why=""):
    details["checks"].append({"check": name, "ok": bool(ok), **({"why": why} if why and not ok else {})})
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {why}" if why and not ok else ""))
    return ok


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else ""


def tvr(*args, timeout=None):
    pw = pwd.getpwnam("tva")
    env = {"HOME": pw.pw_dir, "PATH": PATH, "LANG": "C.UTF-8", "ELAN_HOME": "/opt/elan"}
    p = subprocess.run(["python3", "-I", "tools/tvr.py", *args], cwd=ROOT, env=env, user=pw.pw_uid, group=pw.pw_gid,
                       extra_groups=[], capture_output=True, text=True, timeout=timeout, errors="replace")
    return p.returncode, p.stdout + p.stderr


def main():
    state_p = os.path.join(EXPORT, "state.json")
    if not check("the review service exported its record", os.path.isfile(state_p) and os.path.isdir(os.path.join(EXPORT, "kernel")),
                 f"no {state_p} or {EXPORT}/kernel"):
        finish()
    S = json.load(open(state_p))
    end = S.get("status_at_end", S.get("status"))
    details["service"] = {k: S.get(k) for k in ("mode", "model", "status_at_end", "reviewed", "rejected", "costs",
                                                 "error", "tva_commit", "started", "exported")}
    reward.update(rounds_reviewed=S.get("reviewed", 0), rejected=S.get("rejected", 0),
                  reviewer_cost_usd=round(sum((S.get("costs") or {}).values()), 4),
                  service_error=1 if end == "error" else 0, escalated=1 if end == "escalated" else 0)
    if S.get("mode") != "claude" and not TASK.get("allow_scripted"):
        check("the reviewers are agents", False, f"the service ran {S.get('mode')} reviewers, which this task does not allow")
        finish()
    if S.get("mode") != "claude":
        reward["scripted"] = 1
    check("the image is the task's", S.get("tva_commit") == TASK["tva_commit"],
          f"the service ran TVA {S.get('tva_commit')}, the task pins {TASK['tva_commit']}")

    # the exported kernel replaces the image's copy (the artifact upload already dropped links and devices)
    shutil.rmtree(KD)
    shutil.copytree(os.path.join(EXPORT, "kernel"), KD, symlinks=False)
    pw = pwd.getpwnam("tva")
    for dirpath, dirs, files in os.walk(KD):
        os.lchown(dirpath, pw.pw_uid, pw.pw_gid)
        for n in files:
            os.lchown(os.path.join(dirpath, n), pw.pw_uid, pw.pw_gid)

    man = json.load(open(os.path.join(KD, "kernel.json")))
    src = man.get("source") or {}
    fixed = (man.get("protocol"), (man.get("lean") or {}).get("module"), (man.get("lean") or {}).get("namespace"),
             src.get("file"), src.get("sha256"), man.get("doc", "README.md"))
    want = ("tvr-2", TASK["module"], TASK["namespace"], TASK["source"], TASK["source_sha256"], "README.md")
    check("kernel.json keeps the task's fixed fields", fixed == want, f"{fixed} != {want}")
    check("the source is the task's", sha(os.path.join(KD, TASK["source"])) == TASK["source_sha256"],
          f"{TASK['source']} differs from the task's")

    ledger = os.path.join(KD, "review", "ledger.jsonl")
    check("the ledger is the one the service recorded", sha(ledger) == S.get("ledger_sha") == S.get("ledger_sha_now"),
          "its sha256 differs from the service's record")
    events = [json.loads(l) for l in open(ledger) if l.strip()] if os.path.isfile(ledger) else []
    assigned = sorted({e["actor"] for e in events if e.get("type") == "assign"})
    check("the reviewers are the task's", assigned == sorted(TASK["reviewers"]), f"assigned {assigned}")
    actors = {e.get("actor") for e in events} - set(TASK["reviewers"]) - {"prover", "tvr"}
    check("only the task's roles act in the ledger", not actors, f"also {sorted(actors)}")

    rc, out = tvr("check", K, timeout=TASK.get("gate_timeout_sec", 3600))
    last = out.strip().split("\n")[-1] if out.strip() else ""
    details["gate"] = out[-6000:]
    reward["gate_pass"] = 1 if rc == 0 and last.startswith("gate PASS") else 0
    check("the gate passes here", reward["gate_pass"], last)
    gate = json.load(open(os.path.join(KD, "gate.json"))) if os.path.isfile(os.path.join(KD, "gate.json")) else {}

    rc, out = tvr("status", K)
    details["status"] = out[-6000:]
    granted = rc == 0 and "**Certificate: GRANTED**" in out
    cert_p = os.path.join(KD, "review", "certificate.json")
    cert = json.load(open(cert_p)) if os.path.isfile(cert_p) else {}
    check("the certificate holds against the gate run here", granted, out.strip().split("\n")[-1] if out.strip() else "")
    check("the service's certificate names this gate and the task's reviewers",
          cert.get("gate") == gate.get("digest") and sorted(cert.get("reviewers", [])) == sorted(TASK["reviewers"]),
          f"certificate gate {str(cert.get('gate'))[:16]}, reviewers {cert.get('reviewers')}; gate {str(gate.get('digest'))[:16]}")
    reward["certificate"] = 1 if granted and cert.get("gate") == gate.get("digest") else 0
    reward["findings"] = sum(1 for e in events if e.get("type") == "raise")
    finish()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        details["error"] = traceback.format_exc()
        check("the verifier ran", False, repr(e))
        finish()
