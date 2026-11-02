#!/usr/bin/env python3
"""The review service of a TVA sandbox task: the reviewer side of the prover/reviewer protocol (PROTOCOL.md).

  server.py serve          take the prover's submissions over HTTP (port 8765) until stopped
  server.py export DIR     stop taking submissions and copy the record to DIR, for the verifier

It runs as root in the reviewer container and holds the authority copy of TVA at ~tva/fpv/tva: the
library as the image ships it, and the task's kernel as the prover's accepted submissions made it. The
`tvr` commands that change the record run as the user `tva`, who owns the kernel's folder; the library and
the tool are root's. Each reviewer agent runs as a user of its own (rv1, rv2, …) in a copy of the tree
made at the start of its round, whose `lean/.lake` is the authority's build, read-only to it. The service
records each step's ledger digest where only root can write, so the verifier can tell that the ledger it
receives is the one the service wrote.

Standard library only. Configuration: /srv/sandbox/task.json (written by make_task.py) and the
environment (TVA_REVIEWER_MODE, TVA_REVIEWER_MODEL, ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN,
ANTHROPIC_BASE_URL).
"""
import contextlib
import datetime
import fcntl
import hashlib
import http.server
import io
import ipaddress
import json
import os
import pwd
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.parse

SRV = "/srv/sandbox"
STATE = "/srv/state"
AUTH = "/home/tva/fpv/tva"
PORT = 8765
TASK = json.load(open(os.path.join(SRV, "task.json")))
K = TASK["kernel"]
KD = os.path.join(AUTH, "kernels", K)
REVIEWERS = TASK["reviewers"]  # ids, in order; reviewer i runs as user rv<i+1>
USERS = {r: f"rv{i + 1}" for i, r in enumerate(REVIEWERS)}
MODE = os.environ.get("TVA_REVIEWER_MODE", TASK.get("mode", "claude"))
MODEL = os.environ.get("TVA_REVIEWER_MODEL") or TASK.get("model", "claude-opus-5-5")
LIMITS = {"max_reviewed": TASK.get("max_reviewed", 6), "max_rejected": TASK.get("max_rejected", 40)}
GATE_TIMEOUT = TASK.get("gate_timeout_sec", 3600)
REVIEWER_TIMEOUT = TASK.get("reviewer_timeout_sec", 5400)
FILE_RETRIES = TASK.get("file_retries", 2)   # times a reviewer is sent back to fix a filing `tvr file` refuses
REFILES = TASK.get("refiles", 2)             # times a reviewer is sent back to finish an incomplete review
PATH = "/opt/elan/bin:/usr/local/bin:/usr/bin:/bin"

# What a submission may hold, by path in the kernel folder (PROTOCOL.md § Submissions).
MAX_FILES, MAX_FILE, MAX_TOTAL = 400, 2 * 2 ** 20, 16 * 2 ** 20
NOTE_EXT = (".json", ".md", ".txt", ".lean", ".py")
KERNEL_FIELDS = ("title", "ranges", "host", "unused")  # of kernel.json, the prover's; the rest is fixed

sys.path.insert(0, os.path.join(AUTH, "tools"))
import tvr  # noqa: E402  (root's copy of the tool; read-only queries only, the changes run as `tva`)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(p):
    if not os.path.exists(p):
        return ""
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ledger_path():
    return os.path.join(KD, "review", "ledger.jsonl")


# ---------------------------------------------------------------- state and locks

class Record:
    """The lock on the authority's record: held by every step that changes the kernel's folder or the
    ledger, and by `export`, across processes (flock) and threads."""
    _t = threading.Lock()

    def __enter__(self):
        self._t.acquire()
        os.makedirs(STATE, mode=0o700, exist_ok=True)
        self.f = open(os.path.join(STATE, "record.lock"), "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()
        self._t.release()


_S_LOCK = threading.RLock()
_DONE = threading.Condition(_S_LOCK)
_depth = [0]


@contextlib.contextmanager
def state_lock():
    """The lock on state.json: re-entrant in this process, and exclusive against `export` (flock)."""
    with _S_LOCK:
        f = None
        if _depth[0] == 0:
            f = open(os.path.join(STATE, "state.lock"), "w")
            fcntl.flock(f, fcntl.LOCK_EX)
        _depth[0] += 1
        try:
            yield
        finally:
            _depth[0] -= 1
            if f:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()


def load_state():
    p = os.path.join(STATE, "state.json")
    return json.load(open(p)) if os.path.exists(p) else None


def save_state(S):
    p = os.path.join(STATE, "state.json")
    with open(p + ".tmp", "w") as f:
        json.dump(S, f, indent=2, sort_keys=True)
    os.replace(p + ".tmp", p)


def update(fn):
    """Change the service state under its lock, and save it."""
    with state_lock():
        S = load_state()
        fn(S)
        save_state(S)
        return S


def note_ledger(S, step):
    S["ledger_sha"] = sha_file(ledger_path())
    S.setdefault("ledger_history", []).append({"step": step, "sha256": S["ledger_sha"], "time": now()})


# ---------------------------------------------------------------- running things

def as_user(user, cmd, cwd, timeout, env_extra=None, capture=True):
    """Run `cmd` as `user` with a clean environment; on timeout kill its whole process group.
    Returns (exit code or None on timeout, output)."""
    pw = pwd.getpwnam(user)
    env = {"HOME": pw.pw_dir, "USER": user, "LOGNAME": user, "PATH": PATH, "LANG": "C.UTF-8",
           "ELAN_HOME": "/opt/elan", **(env_extra or {})}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, user=pw.pw_uid, group=pw.pw_gid, extra_groups=[],
                         stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                         stderr=subprocess.STDOUT if capture else subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, start_new_session=True, text=True, errors="replace")
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out or ""
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        return None, (out or "") + f"\n[timed out after {timeout} s]"


def tvr_run(*args, timeout=600, user="tva", root=AUTH):
    rc, out = as_user(user, ["python3", "-I", "tools/tvr.py", *args], root, timeout)
    return rc == 0, out


def tail(text, n=60):
    lines = text.rstrip("\n").split("\n")
    return "\n".join((["…"] if len(lines) > n else []) + lines[-n:])


def chown_tree(path, user):
    pw = pwd.getpwnam(user)
    for dirpath, dirs, files in os.walk(path):
        os.lchown(dirpath, pw.pw_uid, pw.pw_gid)
        for n in dirs + files:
            os.lchown(os.path.join(dirpath, n), pw.pw_uid, pw.pw_gid)


def round_state():
    """The kernel's ledger, folded by the tool, with the current gate."""
    S = tvr.state(K)
    gate = tvr.load_json(os.path.join(KD, "gate.json"), {"items": {}, "ok": False, "digest": ""})
    return S, gate


def certified():
    S, gate = round_state()
    return tvr.cert_state(K, S, gate)[0]


# ---------------------------------------------------------------- submissions

class Reject(Exception):
    def __init__(self, reason, output=""):
        super().__init__(reason)
        self.reason, self.output = reason, output


def normal(name):
    name = name[2:] if name.startswith("./") else name
    if not name or name.startswith("/") or "\\" in name or any(c in ("", ".", "..") for c in name.split("/")):
        return None
    return name


def unpack(blob, rnd):
    """The files of a submission, {path in the kernel folder: bytes}, after the intake checks."""
    try:  # an uncompressed tar: its size is bounded by the request's
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:")
        members = tf.getmembers()
    except (tarfile.TarError, EOFError, OSError) as e:
        raise Reject(f"the submission is not a tar archive ({e})")
    files, total = {}, 0
    prover = f"review/round-{rnd}/prover/"
    for m in members:
        name = normal(m.name.rstrip("/"))
        if name is None:
            raise Reject(f"bad path in the submission: {m.name!r}")
        if m.isdir():
            continue
        if not m.isfile():
            raise Reject(f"{name}: only regular files may be submitted (no links or devices)")
        ok = (name in ("README.md", "model-map.json", "kernel.json") or
              (name.startswith("lean/") and name.endswith(".lean")) or
              (name.startswith(prover) and name.endswith(NOTE_EXT)))
        if not ok:
            raise Reject(f"{name}: not a file a submission may hold (lean/**.lean, README.md, model-map.json, "
                         f"kernel.json, {prover}*)")
        if m.size > MAX_FILE:
            raise Reject(f"{name}: larger than {MAX_FILE} bytes")
        total += m.size
        if len(files) >= MAX_FILES or total > MAX_TOTAL:
            raise Reject(f"the submission holds more than {MAX_FILES} files or {MAX_TOTAL} bytes")
        data = tf.extractfile(m).read()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise Reject(f"{name}: not UTF-8 text")
        files[name] = data
    for need in ("README.md", "model-map.json", "kernel.json"):
        if need not in files:
            raise Reject(f"the submission has no {need}")
    if not any(n.startswith("lean/") for n in files):
        raise Reject("the submission has no Lean file (lean/**.lean)")
    return files


def merged_manifest(data):
    """kernel.json: the task's fixed fields, with the prover's title and line classes."""
    cur = json.load(open(os.path.join(KD, "kernel.json")))
    try:
        new = json.loads(data)
    except ValueError as e:
        raise Reject(f"kernel.json is not JSON ({e})")
    if not isinstance(new, dict):
        raise Reject("kernel.json is not a JSON object")
    src = new.get("source") if isinstance(new.get("source"), dict) else {}
    out = json.loads(json.dumps(cur))
    if isinstance(new.get("title"), str) and 0 < len(new["title"]) <= 200:
        out["title"] = new["title"]
    # the line classes are the prover's (`tvr check` checks their shape)
    out["source"]["ranges"] = src.get("ranges", [])
    out["source"]["host"] = src.get("host", [])
    out["source"].pop("unused", None)
    if "unused" in src:
        out["source"]["unused"] = src["unused"]
    changed = [f for f in ("protocol", "lean", "doc") if new.get(f) != cur.get(f)] + \
              [f"source.{f}" for f in ("file", "sha256") if src.get(f) != cur["source"].get(f)]
    return out, changed


def submission_kind():
    S, _ = round_state()
    N = S["round"]
    filed = [r for r in REVIEWERS if S["filed"].get(r, {}).get(N)]
    open_ = sorted(i for i, f in S["findings"].items() if f["state"] == "OPEN")
    if N == 1 and not filed:
        return "initial", N
    if len(filed) == len(REVIEWERS) and open_:
        return "response", N
    raise Reject(f"round {N} takes no submission now ({tvr.phase(K, S)})")


def snapshot():
    snap = os.path.join(STATE, "snap")
    shutil.rmtree(snap, ignore_errors=True)
    shutil.copytree(KD, snap, symlinks=True)
    return snap


def restore(snap):
    shutil.rmtree(KD)
    shutil.copytree(snap, KD, symlinks=True)
    chown_tree(KD, "tva")
    os.chmod(KD, 0o700)


def apply(files, kind, rnd):
    man, changed = merged_manifest(files["kernel.json"])
    shutil.rmtree(os.path.join(KD, "lean"))
    os.makedirs(os.path.join(KD, "lean"))
    for name, data in files.items():
        if name == "kernel.json":
            continue
        p = os.path.join(KD, *name.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    tvr.dump_json(os.path.join(KD, "kernel.json"), man)
    chown_tree(KD, "tva")
    return changed


def process_submission(job, blob, log):
    kind, rnd = submission_kind()
    files = unpack(blob, rnd)
    has_response = any(n.startswith(f"review/round-{rnd}/prover/") for n in files)
    if kind == "initial" and has_response:
        raise Reject(f"round {rnd} has no finding to answer yet: submit without review/round-{rnd}/prover/")
    if kind == "response" and f"review/round-{rnd}/prover/response.json" not in files:
        raise Reject(f"round {rnd} has open findings: the submission must answer them in "
                     f"review/round-{rnd}/prover/response.json (python3 -I tools/tvr.py start {K} prover)")
    log(f"submission: {kind}, round {rnd}, {len(files)} file(s)")
    with Record():
        before = sha_file(ledger_path())
        snap = snapshot()
        try:
            changed = apply(files, kind, rnd)
            if changed:
                log("kernel.json: kept the task's " + ", ".join(changed) + " (the prover sets only the title and "
                    "source.ranges, source.host, source.unused)")
            set_step(job, "gate")
            t0 = time.time()
            ok, out = tvr_run("check", K, timeout=GATE_TIMEOUT)
            log(f"tvr check {K} ({time.time() - t0:.0f} s):\n" + tail(out))
            if sha_file(ledger_path()) != before:
                raise RuntimeError("the ledger changed while the gate ran")
            if not ok:
                raise Reject("the gate fails (tvr check)", tail(out))
            if kind == "response":
                set_step(job, "response")
                ok, out = tvr_run("lint", K, "prover")
                log(f"tvr lint {K} prover:\n" + tail(out))
                if not ok:
                    raise Reject("the response is not regular (tvr lint prover)", tail(out))
                ok, out = tvr_run("respond", K)
                log(f"tvr respond {K}:\n" + tail(out))
                if not ok:
                    raise Reject("tvr respond refused the response", tail(out))
                update(lambda S: note_ledger(S, f"respond round {rnd}"))
        except BaseException:
            restore(snap)
            raise
    return kind, rnd


# ---------------------------------------------------------------- reviewers

def workspace(r):
    return os.path.join(pwd.getpwnam(USERS[r]).pw_dir, "fpv", "tva")


def make_workspace(r):
    """A copy of the authority's tree for reviewer r at the start of its round: the round's record as it
    stands (no other reviewer of the round has filed yet), with the authority's build linked read-only."""
    ws = workspace(r)
    shutil.rmtree(ws, ignore_errors=True)
    shutil.copytree(AUTH, ws, symlinks=True,
                    ignore=lambda d, names: [".lake"] if os.path.normpath(d) == os.path.join(AUTH, "lean") else [])
    os.symlink(os.path.join(AUTH, "lean", ".lake"), os.path.join(ws, "lean", ".lake"))
    chown_tree(ws, USERS[r])
    os.chmod(os.path.dirname(os.path.dirname(ws)), 0o700)


def sync_kernel(r):
    """Give reviewer r the whole record of the kernel (after every reviewer of the round has filed)."""
    dst = os.path.join(workspace(r), "kernels", K)
    shutil.rmtree(dst)
    shutil.copytree(KD, dst, symlinks=True)
    chown_tree(dst, USERS[r])


def reviewer_prompt(r):
    ok, text = tvr_run("prompt", K, "reviewer", r, user=USERS[r], root=workspace(r))
    if not ok:
        raise RuntimeError(f"tvr prompt failed for {r}: {tail(text)}")
    note = open(os.path.join(SRV, "reviewer-note.md")).read()
    return text.rstrip("\n") + "\n\n" + note.replace("{{kernel}}", K).replace("{{reviewer}}", r)


PASS_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
            "API_TIMEOUT_MS", "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "MAX_THINKING_TOKENS")
ALIASES = ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
           "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "CLAUDE_CODE_SUBAGENT_MODEL")


def agent_env():
    """The reviewer agents' environment: the key and endpoint the host gave, and, for an endpoint other than
    Anthropic's, every model alias Claude Code uses (subagents, background calls) set to the one model."""
    env = {"DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_TELEMETRY": "1",
           "IS_SANDBOX": "1"}
    for v in PASS_ENV:
        if os.environ.get(v):
            env[v] = os.environ[v]
    if os.environ.get("ANTHROPIC_BASE_URL"):
        for v in ALIASES:
            env[v] = MODEL
        env.setdefault("API_TIMEOUT_MS", "600000")
    return env


def cli_result(out):
    """The Claude CLI's JSON result (`--output-format json`), after anything it printed before it."""
    for line in reversed(out.strip().split("\n")):
        with contextlib.suppress(ValueError):
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("type") == "result":
                return obj
    return {}


def preflight():
    """Check, before the prover starts, that the reviewers' agent runs with the key and the model given: a
    bad key would otherwise surface only when the first round fails."""
    u = USERS[REVIEWERS[0]]
    rc, out = as_user(u, ["claude", "-p", "Reply with the single word: ready", "--output-format", "json",
                          "--model", MODEL], pwd.getpwnam(u).pw_dir, 300, agent_env())
    res = cli_result(out)
    if rc != 0 or not res or res.get("is_error"):
        sys.exit(f"the reviewer agents cannot run (model {MODEL}): {res.get('result') or tail(out, 5)}")
    return res.get("total_cost_usd") or 0


def run_claude(r, prompt, tag):
    """One turn of reviewer r's agent (resuming its session when it has one); returns its result text."""
    S = load_state()
    sid = S["sessions"].get(r)
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", MODEL, "--dangerously-skip-permissions"]
    if sid:
        cmd += ["--resume", sid]
    t0 = time.time()
    # the agent starts in /, where the reviewer's harness is (/CLAUDE.md, /.claude/); its TVA copy is ~/fpv/tva
    rc, out = as_user(USERS[r], cmd, "/", REVIEWER_TIMEOUT, agent_env())
    os.makedirs(os.path.join(STATE, "agents"), exist_ok=True)
    with open(os.path.join(STATE, "agents", f"{r}-{tag}.out"), "w") as f:
        f.write(out)
    res = cli_result(out)
    cost = res.get("total_cost_usd") or 0
    # the session's tokens, by model (`usage` can hold only the last request's)
    use = {"input": 0, "cached": 0, "output": 0}
    for m in (res.get("modelUsage") or {}).values():
        use["input"] += (m.get("inputTokens") or 0) + (m.get("cacheCreationInputTokens") or 0)
        use["cached"] += m.get("cacheReadInputTokens") or 0
        use["output"] += m.get("outputTokens") or 0

    def keep(S):
        if res.get("session_id"):
            S["sessions"][r] = res["session_id"]
        S["costs"][r] = S["costs"].get(r, 0) + cost
        # the tokens are the endpoint's count; the cost is Claude Code's estimate at Anthropic's prices
        tk = S.setdefault("tokens", {}).setdefault(r, {"input": 0, "cached": 0, "output": 0})
        for key in tk:
            tk[key] += use[key]
        S["agent_runs"].append({"reviewer": r, "tag": tag, "exit": rc, "seconds": round(time.time() - t0),
                                "turns": res.get("num_turns"), "cost_usd": cost, "error": bool(res.get("is_error"))})
    update(keep)
    if rc is None:
        return "(the agent timed out)"
    return res.get("result") or tail(out, 20)


def run_scripted(r, tag):
    """The scripted reviewer (plumbing tests only): accept every item, raise nothing, release what it raised.
    `scripted-findings` exercises the rest of the protocol: in round 1 it raises a blocking finding on the
    spec and accepts the other items; in round 2 its first filing is unfinished (it accepts and releases
    nothing), so the service sends it back, and its next filing releases the finding and accepts all."""
    ws = workspace(r)
    ok, out = tvr_run("start", K, "reviewer", r, user=USERS[r], root=ws)
    made = re.findall(r"^wrote (\S+)$", out, re.M)
    fpath = next((m for m in made if m.endswith(".json")), None)
    rpath = next((m for m in made if m.endswith(".md")), None)
    if not (ok and fpath and rpath):
        return f"tvr start: {out}"
    S, gate = round_state()
    N = S["round"]
    findings, accept = [], "all"
    if MODE == "scripted-findings" and N == 1:
        findings = [{"kind": "SPEC_TOO_WEAK", "target": "item:spec", "summary": "A scripted finding (plumbing test).",
                     "detail": "The scripted reviewer raises this in round 1 to exercise the response path."}]
        accept = [f"item:{i}" for i in sorted(gate["items"]) if i != "spec"]
    elif MODE == "scripted-findings" and "refile" not in tag:
        accept = []
    rel = [] if accept == [] else [
        {"id": i, "note": "Settled (scripted reviewer)."} for i, f in sorted(S["findings"].items(), key=lambda x: int(x[0][1:]))
        if r in f["raisers"] and f["state"] in ("ANSWERED", "REFERRED") and r not in f["released"]]
    filing = {"reviewer": r, "kernel": K, "round": N, "findings": findings, "release": rel, "reraise": [],
              "accept": accept, "notes": "The sandbox's scripted reviewer: it reads nothing."}
    verdict = "accept" if accept == "all" and not findings else "reject" if not accept else "accept-partly"
    bullets = [f"- **{f['kind']}** `{f['target']}` — {f['summary']}\n" for f in findings] + \
              [f"- **release {x['id']}** — settled (scripted reviewer).\n" for x in rel]
    accepted = "All items.\n" if accept == "all" else ("".join(f"- `{a}`\n" for a in accept) or "None.\n")
    report = (f"# Review of {K}, round {N}, by {r}\n\n## Verdict\n\n{verdict}\n\n## Findings\n\n" +
              ("".join(bullets) or "None.\n") + f"\n## Accepted\n\n{accepted}\n## Checked\n\n- Nothing: this is "
              "the sandbox's scripted reviewer, for plumbing tests only.\n\n## Not checked\n\n- Everything: the "
              "scripted reviewer reads nothing.\n")
    pw = pwd.getpwnam(USERS[r])
    for path, text in ((fpath, json.dumps(filing, indent=2) + "\n"), (rpath, report)):
        p = os.path.join(ws, path)
        with open(p, "w") as f:
            f.write(text)
        os.chown(p, pw.pw_uid, pw.pw_gid)
    ok, out = tvr_run("lint", K, "reviewer", r, user=USERS[r], root=ws)
    return "regular" if ok else f"tvr lint: {out}"


def agent_turn(r, prompt, tag):
    return run_claude(r, prompt, tag) if MODE == "claude" else run_scripted(r, tag)


def file_review(r, N, log):
    """Copy reviewer r's round folder into the authority and record it with `tvr file`. Files recorded
    before stay as they were; on a refusal the files this added are removed again."""
    src = os.path.join(workspace(r), "kernels", K, "review", f"round-{N}", r)
    dst = os.path.join(KD, "review", f"round-{N}", r)
    if not os.path.isdir(src) or os.path.islink(src):
        return False, f"there is no folder review/round-{N}/{r}/ (run `python3 -I tools/tvr.py start {K} reviewer {r}`)"
    new, n, total = [], 0, 0
    for dirpath, dirs, files in os.walk(src):
        for name in dirs + files:
            p = os.path.join(dirpath, name)
            if os.path.islink(p):
                return False, f"{os.path.relpath(p, src)}: a symbolic link; a round folder holds only files"
        for name in files:
            p = os.path.join(dirpath, name)
            relp = os.path.relpath(p, src)
            n, total = n + 1, total + os.path.getsize(p)
            if n > MAX_FILES or total > MAX_TOTAL or os.path.getsize(p) > MAX_FILE:
                return False, f"the round folder is too large (at most {MAX_FILES} files, {MAX_FILE} bytes each)"
            q = os.path.join(dst, relp)
            if os.path.exists(q):
                if sha_file(q) != sha_file(p):
                    return False, f"{relp} was recorded by an earlier filing and is frozen; write the next filing instead"
            else:
                new.append((p, q))
    with Record():
        made_dirs = []
        for p, q in new:
            d = os.path.dirname(q)
            while not os.path.isdir(d):
                made_dirs.append(d)
                d = os.path.dirname(d)
            os.makedirs(os.path.dirname(q), exist_ok=True)
            shutil.copyfile(p, q)
        chown_tree(os.path.join(KD, "review"), "tva")
        ok, out = tvr_run("file", K, r)
        log(f"tvr file {K} {r}:\n" + tail(out, 30))
        if ok:
            update(lambda S: note_ledger(S, f"file {r} round {N}"))
        else:
            for p, q in new:
                os.remove(q)
            for d in sorted(set(made_dirs), key=len, reverse=True):
                with contextlib.suppress(OSError):
                    os.rmdir(d)
    return ok, out


def unfinished(r):
    """What reviewer r's record leaves open: items it neither accepts at the current digest nor holds by a
    blocking finding of its own, and blocking findings of its own the prover answered that it neither
    released nor re-raised."""
    S, gate = round_state()
    items = gate["items"]
    held = set()
    for f in S["findings"].values():
        if r in f["raisers"] and f["blocking"] and f["state"] in ("OPEN", "ESCALATED"):
            held |= tvr.affected(K, f, items)
    loose = sorted(i for i, d in items.items() if S["accepts"].get(r, {}).get(i) != d and i not in held)
    answered = sorted((i for i, f in S["findings"].items() if r in f["raisers"] and f["blocking"]
                       and f["state"] == "ANSWERED" and r not in f["released"]), key=lambda x: int(x[1:]))
    return loose, answered


def review_until_filed(r, N, prompt, tag, log):
    """Run reviewer r until `tvr file` records its filing, sending it back with the refusal at most
    FILE_RETRIES times."""
    for attempt in range(FILE_RETRIES + 1):
        msg = agent_turn(r, prompt, f"{tag}-{attempt}")
        log(f"{r}: agent finished ({tag}, attempt {attempt + 1})")
        log(f"{r}: final message ({tag}, attempt {attempt + 1}): {msg[:2000]}", private=True)
        ok, out = file_review(r, N, log)
        if ok:
            return True
        prompt = (f"The review service could not record your filing. `tvr file {K} {r}` said:\n\n{tail(out, 40)}\n\n"
                  f"Fix the files in your round folder (kernels/{K}/review/round-{N}/{r}/), run "
                  f"`python3 -I tools/tvr.py lint {K} reviewer {r}` until it prints `regular`, and stop: the service "
                  f"records them again. Do not run `tvr file` yourself.")
    return False


def run_round(job, N, log):
    """Round N: every reviewer that has not filed in it reviews, in parallel, from a copy of the record
    made before any of them files; then each one whose record leaves something open is sent back, with the
    whole record, at most REFILES times."""
    S, _ = round_state()
    need = [r for r in REVIEWERS if not S["filed"].get(r, {}).get(N)]
    for r in need:
        make_workspace(r)
    prompts = {r: reviewer_prompt(r) for r in need}
    set_step(job, f"review round {N}: " + ", ".join(need))
    results = {}

    def one(r):
        try:
            results[r] = review_until_filed(r, N, prompts[r], f"round{N}", log)
        except Exception:
            log(f"{r}: {traceback.format_exc()}")
            results[r] = False
    threads = [threading.Thread(target=one, args=(r,)) for r in need]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [r for r in need if not results.get(r)]
    if failed:
        return f"reviewer(s) {', '.join(failed)} could not file a regular review"
    for k in range(REFILES):
        left = {r: unfinished(r) for r in REVIEWERS}
        left = {r: v for r, v in left.items() if v[0] or v[1]}
        if not left:
            return None
        set_step(job, f"review round {N}: finishing " + ", ".join(left))
        results.clear()

        def again(r, loose=None, answered=None):
            try:
                sync_kernel(r)
                parts = (["items you neither accept at the current digest nor hold by a blocking finding of yours: "
                          + ", ".join(f"`{i}`" for i in loose)] if loose else []) + \
                        ([f"findings of yours the prover answered, neither released nor re-raised: {', '.join(answered)}"]
                         if answered else [])
                prompt = (f"The review service recorded your filing, but your review of round {N} is unfinished:\n- "
                          + "\n- ".join(parts) + "\n\nEvery item ends accepted at its current digest or held by a "
                          "blocking finding of yours, and every answered finding of yours released or re-raised. Your "
                          f"copy of kernels/{K}/ now holds the whole record (REVIEW.md, every filing of the round). Run "
                          f"`python3 -I tools/tvr.py start {K} reviewer {r}` for your next filing, fill it in, run "
                          f"`python3 -I tools/tvr.py lint {K} reviewer {r}` until it prints `regular`, and stop: the "
                          "service records it.")
                results[r] = review_until_filed(r, N, prompt, f"round{N}-refile{k + 1}", log)
            except Exception:
                log(f"{r}: {traceback.format_exc()}")
                results[r] = False
        threads = [threading.Thread(target=again, args=(r, *v)) for r, v in left.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        failed = [r for r in left if not results.get(r)]
        if failed:
            return f"reviewer(s) {', '.join(failed)} could not file a regular review"
    left = [r for r in REVIEWERS if any(unfinished(r))]
    return f"reviewer(s) {', '.join(left)} left the review unfinished" if left else None


# ---------------------------------------------------------------- jobs

def set_step(job, step):
    def f(S):
        for j in S["jobs"]:
            if j["id"] == job:
                j["step"] = step
    update(f)


def outcome():
    """The status after a round: certified, escalated, ready for the prover, or out of submissions."""
    S, gate = round_state()
    if tvr.cert_state(K, S, gate)[0]:
        return "certified"
    if any(f["state"] == "ESCALATED" for f in S["findings"].values()):
        return "escalated"
    st = load_state()
    if st["reviewed"] >= LIMITS["max_reviewed"]:
        return "exhausted"
    return "ready"


def run_job(job, blob):
    logp = os.path.join(STATE, "jobs", f"{job}.log")

    def log(msg, private=False):
        """The job's log, which the prover reads (`tva-review log`); `private` lines (a reviewer agent's
        final message, meant for the coordinator, not the prover) go to the export only."""
        with open(logp.replace(".log", ".private.log") if private else logp, "a") as f:
            f.write(f"[{now()}] {msg}\n")
    result, status = {}, "ready"
    try:
        kind, rnd = process_submission(job, blob, log)
        update(lambda S: S.__setitem__("reviewed", S["reviewed"] + 1))
        S, _ = round_state()
        N = S["round"]
        err = run_round(job, N, log)
        if err:
            log(err)
            result, status = {"accepted": True, "round": N, "error": err}, "error"
        else:
            status = outcome()
            S, gate = round_state()
            result = {"accepted": True, "round": N, "gate": gate["digest"][:16], "certificate": status == "certified",
                      "open": sorted((i for i, f in S["findings"].items() if f["state"] == "OPEN"), key=lambda x: int(x[1:]))}
        log(f"done: {status}")
    except Reject as e:
        log(f"rejected: {e.reason}" + (f"\n{e.output}" if e.output else ""))
        result = {"accepted": False, "reason": e.reason, "output": e.output}
        st = update(lambda S: S.__setitem__("rejected", S["rejected"] + 1))
        status = "exhausted" if st["rejected"] >= LIMITS["max_rejected"] else "ready"
    except Exception:
        log(traceback.format_exc())
        result, status = {"accepted": False, "error": "the review service failed"}, "error"

    def fin(S):
        for j in S["jobs"]:
            if j["id"] == job:
                j.update(state="done", finished=now(), result=result, step="done")
        if S["status"] != "stopped":
            S["status"] = status
        if status == "error":
            S["error"] = result.get("error", "")
    with state_lock():
        update(fin)
        _DONE.notify_all()


def submit(blob):
    with state_lock():
        S = load_state()
        if S["status"] != "ready":
            return 409, {"error": f"the service takes no submission now (status {S['status']})", "status": S["status"]}
        job = len(S["jobs"]) + 1
        os.makedirs(os.path.join(STATE, "submissions"), exist_ok=True)
        with open(os.path.join(STATE, "submissions", f"{job}.tar"), "wb") as f:
            f.write(blob)
        S["jobs"].append({"id": job, "state": "running", "step": "intake", "started": now(), "bytes": len(blob)})
        S["status"] = "busy"
        save_state(S)
    threading.Thread(target=run_job, args=(job, blob), daemon=True).start()
    return 202, {"job": job}


NEXT = {
    "ready": "your turn: {todo}",
    "busy": "a submission is under review ({step}): run `tva-review wait`",
    "certified": "the certificate is granted: nothing more to do",
    "escalated": "a finding was re-raised twice and escalated to a user, and no user decides in this task: no "
                 "certificate can follow",
    "exhausted": "the submission limit is reached: no more submissions",
    "stopped": "the task has ended",
    "error": "the review service failed ({error}): this is not your doing; stop",
}


def public_status():
    st = load_state()
    job = st["jobs"][-1] if st["jobs"] else None
    try:
        return record_status(st, job)
    except Exception:  # a step is writing the record right now (the gate's pages, the ledger)
        return {"kernel": K, "status": st["status"], "job": job, "reviewers": REVIEWERS,
                "submissions": {"reviewed": st["reviewed"], "rejected": st["rejected"], **LIMITS},
                "next": NEXT[st["status"]].format(todo="see `tva-review status` again", step=(job or {}).get("step", ""),
                                                  error=st.get("error", ""))}


def record_status(st, job):
    S, gate = round_state()
    N = S["round"]
    open_ = sorted((i for i, f in S["findings"].items() if f["state"] == "OPEN"), key=lambda x: int(x[1:]))
    if N == 1 and not any(S["filed"].get(r, {}).get(1) for r in REVIEWERS):
        todo = f"write the claim, make `python3 -I tools/tvr.py check {K}` pass, then `tva-review submit`"
    else:
        todo = (f"answer the open findings ({', '.join(open_)}): fix, re-run the gate, `python3 -I tools/tvr.py start {K} "
                f"prover`, fill in response.json and notes.md, `python3 -I tools/tvr.py lint {K} prover`, then "
                "`tva-review submit`")
    nxt = NEXT[st["status"]].format(todo=todo, step=(job or {}).get("step", ""), error=st.get("error", ""))
    return {"kernel": K, "status": st["status"], "round": N, "phase": tvr.phase(K, S),
            "certificate": tvr.cert_state(K, S, gate)[0], "gate": gate.get("digest", "")[:16],
            "open_findings": open_, "reviewers": REVIEWERS,
            "submissions": {"reviewed": st["reviewed"], "rejected": st["rejected"], **LIMITS},
            "job": job, "next": nxt}


def bundle():
    """The kernel's record for the prover: the generated pages and the review folder."""
    buf = io.BytesIO()
    with Record(), tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name in ("REVIEW.md", "claim.md", "gate.json", "review"):
            p = os.path.join(KD, name)
            if os.path.exists(p):
                tf.add(p, arcname=name, filter=lambda ti: ti if not ti.issym() and not ti.islnk() else None)
    return buf.getvalue()


# ---------------------------------------------------------------- HTTP

def local_addresses():
    out = {"127.0.0.1", "::1"}
    with contextlib.suppress(OSError):
        out |= set(socket.gethostbyname_ex(socket.gethostname())[2])
    with contextlib.suppress(OSError):
        for line in open("/proc/net/fib_trie"):
            m = re.search(r"\|-- (\d+\.\d+\.\d+\.\d+)", line)
            if m:
                out.add(m.group(1))
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "tva-review/1"
    LOCAL = set()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{now()}] {self.client_address[0]} {fmt % args}\n")

    def reply(self, code, obj=None, body=None, ctype="application/json"):
        data = body if body is not None else (json.dumps(obj, indent=2) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def allowed(self):
        # only the prover's container: a reviewer agent in this container reaches the service from here
        if self.path == "/health":
            return True
        ip = self.client_address[0]
        with contextlib.suppress(ValueError):
            if ipaddress.ip_address(ip).is_loopback or ip in self.LOCAL:
                self.reply(403, {"error": "submissions come from the prover's container only"})
                return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/health":
            return self.reply(200, {"ok": True})
        if u.path == "/status":
            return self.reply(200, public_status())
        if u.path == "/wait":
            try:
                timeout = max(0.0, min(600.0, float(q.get("timeout", ["540"])[0])))
            except ValueError:
                timeout = 540.0
            end = time.time() + timeout
            with _DONE:
                while load_state()["status"] == "busy" and time.time() < end:
                    _DONE.wait(max(0.1, min(30, end - time.time())))
            return self.reply(200, public_status())
        if u.path == "/bundle":
            return self.reply(200, body=bundle(), ctype="application/gzip")
        if u.path == "/log":
            S = load_state()
            job = q.get("job", [str(len(S["jobs"]))])[0]
            p = os.path.join(STATE, "jobs", f"{int(job) if job.isdigit() else 0}.log")
            if not os.path.exists(p):
                return self.reply(404, {"error": f"no log for job {job}"})
            return self.reply(200, body=open(p, "rb").read(), ctype="text/plain; charset=utf-8")
        self.reply(404, {"error": "GET /status, /wait?timeout=S, /bundle, /log?job=N; POST /submit"})

    def do_POST(self):
        if not self.allowed():
            return
        if urllib.parse.urlparse(self.path).path != "/submit":
            return self.reply(404, {"error": "POST /submit"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_TOTAL + 2 ** 20:
            return self.reply(413, {"error": f"a submission is a tar archive of at most {MAX_TOTAL} bytes"})
        code, obj = submit(self.rfile.read(n))
        self.reply(code, obj)


def init():
    os.makedirs(os.path.join(STATE, "jobs"), mode=0o700, exist_ok=True)
    os.chmod(STATE, 0o700)
    if MODE not in ("claude", "scripted", "scripted-findings"):
        sys.exit(f"TVA_REVIEWER_MODE must be claude, scripted or scripted-findings, not {MODE!r}")
    if MODE == "claude" and not any(os.environ.get(v) for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                                                                   "CLAUDE_CODE_OAUTH_TOKEN")):
        sys.exit("the reviewers need a key: set TVA_REVIEWER_ANTHROPIC_API_KEY or TVA_REVIEWER_ANTHROPIC_AUTH_TOKEN (or "
                 "TVA_REVIEWER_CLAUDE_CODE_OAUTH_TOKEN) in the environment that runs harbor")
    if load_state() is not None:
        S = load_state()
        if S["status"] == "busy":  # the container restarted mid-job: the job is lost
            S["status"], S["error"] = "error", "the review service restarted during a job"
            save_state(S)
        return
    check_cost = preflight() if MODE == "claude" else 0
    chown_tree(KD, "tva")
    os.chmod(KD, 0o700)  # the reviewers see the record only through their own copies
    for r in REVIEWERS:
        ok, out = tvr_run("assign", K, r)
        if not ok:
            sys.exit(f"tvr assign {K} {r} failed: {out}")
    S = {"kernel": K, "mode": MODE, "model": MODEL if MODE == "claude" else None, "reviewers": REVIEWERS,
         "endpoint": urllib.parse.urlparse(os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").netloc,
         "status": "ready", "reviewed": 0, "rejected": 0, "jobs": [], "sessions": {}, "costs": {"preflight": check_cost},
         "agent_runs": [], "limits": LIMITS, "started": now(), "tva_commit": open(os.path.join(AUTH, "TVA_COMMIT")).read().strip()}
    note_ledger(S, "assign")
    save_state(S)


def serve():
    init()
    Handler.LOCAL = local_addresses()
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    print(f"[{now()}] tva-review service for {K} on :{PORT} ({MODE} reviewers: {', '.join(REVIEWERS)})", flush=True)
    srv.serve_forever()


def export(dest):
    """Stop taking submissions, then copy the record (under the record lock, so between steps)."""
    with state_lock():
        S = load_state()
        if S is None:
            sys.exit("the service never started")
        prior = S["status"]
        if S["status"] in ("ready", "busy"):
            S["status"] = "stopped"
        S["status_at_end"] = prior
        save_state(S)
    with Record():
        S = load_state()
        S["exported"] = now()
        S["ledger_sha_now"] = sha_file(ledger_path())
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        shutil.copytree(KD, os.path.join(dest, "kernel"), symlinks=False)
        if os.path.isdir(os.path.join(STATE, "jobs")):
            shutil.copytree(os.path.join(STATE, "jobs"), os.path.join(dest, "jobs"))
        if os.path.isdir(os.path.join(STATE, "agents")):
            shutil.copytree(os.path.join(STATE, "agents"), os.path.join(dest, "agents"))
        for r, u in USERS.items():  # the reviewer agents' transcripts, for whoever studies the run
            t = os.path.join(pwd.getpwnam(u).pw_dir, ".claude", "projects")
            if os.path.isdir(t):
                shutil.copytree(t, os.path.join(dest, "agents", "transcripts", r), symlinks=False,
                                ignore_dangling_symlinks=True, dirs_exist_ok=True)
        with open(os.path.join(dest, "state.json"), "w") as f:
            json.dump(S, f, indent=2, sort_keys=True)
    for dirpath, dirs, files in os.walk(dest):
        os.chmod(dirpath, 0o755)
        for n in files:
            os.chmod(os.path.join(dirpath, n), 0o644)
    print(f"exported the record of {K} to {dest} (status {S['status_at_end']}, ledger {S['ledger_sha_now'][:16]})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["serve"]:
        serve()
    elif sys.argv[1:2] == ["export"] and len(sys.argv) == 3:
        export(sys.argv[2])
    else:
        sys.exit(__doc__)
