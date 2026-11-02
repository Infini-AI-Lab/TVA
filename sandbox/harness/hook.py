#!/usr/bin/env python3
"""The sandbox harness's Claude Code hooks (HARNESS.md), for the prover and the reviewers.

  hook.py session-start | pre-tool | post-tool | stop      (the hook's JSON on stdin, Claude Code's on stdout)

- session-start: where the agent is and what to do next (also after a compaction).
- pre-tool: writes only where the role writes; no command that records for the service, breaks the
  prebuilt toolchain, or writes into the library; no search of the whole filesystem.
- post-tool: after an edit of the kernel's Lean, build it and report the errors; after an edit of a JSON
  record, say whether it parses.
- stop: the prover does not stop while the service is reviewing or findings are open, or before a first
  submission; a reviewer does not stop before its filing is regular.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvakit as kit  # noqa: E402

HOME = os.path.expanduser("~")
STOP_STATE = os.path.join(kit.SCRATCH, ".stop-hook.json")
PROGRESS = os.path.join(kit.SCRATCH, "progress.md")


def out(obj):
    print(json.dumps(obj))
    sys.exit(0)


def captured(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            fn(*a)
        except SystemExit:
            pass
        except Exception as e:  # noqa: BLE001
            print(f"({type(e).__name__}: {e})")
    return buf.getvalue().strip()


def real(path, cwd):
    if not path:
        return ""
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(cwd or "/", path)
    d, b = os.path.split(os.path.normpath(path))
    return os.path.join(os.path.realpath(d), b)


def under(p, d):
    d = os.path.realpath(d)
    return p == d or p.startswith(d.rstrip("/") + "/")


# ---------------------------------------------------------------- session start

MANUAL = "/CLAUDE.md"


def session_start(inp):
    lines = []
    # Claude Code reads CLAUDE.md from the working directory and its parents but not from /, where the
    # harness lives; so the manual comes from here, at every start, resume and compaction
    if os.path.exists(MANUAL):
        lines.append(f"Contents of {MANUAL} (your operating manual; follow it):\n\n" + open(MANUAL).read().strip())
    if kit.ROLE == "reviewer":
        who = kit.reviewer_id()
        lines.append(f"You are reviewer `{who}` of the TVA kernel `{kit.K}` (CLAUDE.md is your operating manual). "
                     f"Your TVA copy is ~/fpv/tva; you write only in its kernels/{kit.K}/review/round-N/{who}/ (with "
                     "`tva-kit filing …`) and in ~/scratch.")
        lines.append(captured(kit.cmd_review_brief, ["--short"]))
    else:
        lines.append(f"You are the prover of the TVA kernel `{kit.K}` (CLAUDE.md is your operating manual). TVA is "
                     f"~/fpv/tva; you write only in ~/fpv/tva/kernels/{kit.K}/ and ~/scratch.")
        lines.append(captured(kit.cmd_brief, []))
    if os.path.exists(PROGRESS):
        tail = open(PROGRESS).read().strip().split("\n")[-40:]
        lines.append("Your progress notes (~/scratch/progress.md, last lines):\n" + "\n".join(tail))
    else:
        lines.append("Keep short progress notes in ~/scratch/progress.md (what is done, what is next): they are shown "
                     "to you again after a compaction.")
    out({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "\n\n".join(lines)}})


# ---------------------------------------------------------------- before a tool

EDIT_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
RECORDING = re.compile(r"\btvr(\.py)?\s+(respond|file|assign|new|freeze|decide|note|record|release)\b")
TOOLCHAIN = re.compile(r"\blake\s+(update|clean|new|init|upgrade)\b|\blake\s+exe\s+cache\b|\belan\s+"
                       r"(update|toolchain|default|override|self)\b|rm\s+-\S*r\S*\s+[^;&|]*\.lake\b")
def write_targets(cmd, cwd):
    """The paths a shell command writes, as far as plain reading tells: the destination of cp/mv/install/rsync,
    the files of sed -i, tee, rm, touch, mkdir, truncate, chmod, and the target of each redirection."""
    import shlex
    out = [m.group(1) for m in re.finditer(r"(?<![0-9&<])>>?\s*([^\s;&|<>]+)", cmd) if m.group(1) != "/dev/null"
           and not m.group(1).startswith("&")]
    for seg in re.split(r"&&|\|\||[;|\n]", cmd):
        try:
            tok = shlex.split(seg, comments=True)
        except ValueError:
            tok = seg.split()
        while tok and (re.match(r"^\w+=", tok[0]) or tok[0] in ("sudo", "env", "command", "nohup", "time")):
            tok = tok[1:]
        if not tok:
            continue
        args = [t for t in tok[1:] if not t.startswith("-")]
        name = os.path.basename(tok[0])
        if name in ("cp", "mv", "install", "rsync", "ln") and args:
            out.append(args[-1])
        elif name == "sed" and any(t.startswith("-i") or t == "--in-place" for t in tok):
            out += args[1:]
        elif name in ("tee", "rm", "rmdir", "unlink", "touch", "mkdir", "truncate", "chmod", "chown", "patch"):
            out += args
    return [real(t, cwd) for t in out if t and not t.startswith("$")]


def protected(p):
    """A path the role must not write: inside the TVA tree, outside what the role writes."""
    if not under(p, os.path.realpath(kit.ROOT)):
        return False
    return bool(reviewer_write_ok(p) if kit.ROLE == "reviewer" else prover_write_ok(p))


def deny(reason):
    out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                "permissionDecisionReason": reason}})


def prover_write_ok(p):
    kd = os.path.realpath(kit.KD)
    if under(p, os.path.join(kd, "lean")):
        return None if p.endswith(".lean") else "only .lean files go in lean/"
    if p in (os.path.join(kd, f) for f in ("README.md", "model-map.json", "kernel.json")):
        return None
    if re.match(re.escape(os.path.join(kd, "review")) + r"/round-\d+/prover/", p):
        return None
    if under(p, kit.SCRATCH) or under(p, "/tmp"):
        return None
    root = os.path.realpath(kit.ROOT)
    if under(p, os.path.join(kd, "source")):
        return "the kernel's source is fixed (its sha256 is pinned): change the model, never the source"
    if under(p, kd):
        return ("this file is generated by tvr or written by a reviewer; you write lean/, README.md, model-map.json, "
                "kernel.json and, when answering findings, review/round-N/prover/ (use `tva-kit respond …`)")
    if under(p, os.path.join(root, "kernels")):
        return "other kernels are read-only examples (copy what you need into your own kernel's files)"
    if under(p, root):
        return ("the library, its tools and documents are fixed: the review service uses its own copy, so a change "
                "here has no effect there. If the kernel needs something the library lacks, say so as a library "
                "request (`tva-kit respond request …` once you answer findings, or in README.md) and cover what you can")
    return f"write only in ~/fpv/tva/kernels/{kit.K}/ (your kernel's files) and ~/scratch (your notes and trials)"


def reviewer_write_ok(p):
    who = kit.reviewer_id()
    kd = os.path.realpath(kit.KD)
    if re.match(re.escape(os.path.join(kd, "review")) + r"/round-\d+/" + re.escape(who) + "/", p):
        return None
    if under(p, kit.SCRATCH) or under(p, "/tmp"):
        return None
    return (f"a reviewer writes only in kernels/{kit.K}/review/round-N/{who}/ (use `tva-kit filing …`; work files in "
            "its work/ folder) and in ~/scratch; you raise findings, you never change the kernel")


def pre_tool(inp):
    tool, ti, cwd = inp.get("tool_name", ""), inp.get("tool_input") or {}, inp.get("cwd") or os.getcwd()
    if tool in EDIT_TOOLS:
        p = real(ti.get("file_path") or ti.get("notebook_path") or "", cwd)
        why = reviewer_write_ok(p) if kit.ROLE == "reviewer" else prover_write_ok(p)
        if why:
            deny(f"{p}: {why}.")
    elif tool == "Bash":
        cmd = ti.get("command", "")
        if RECORDING.search(cmd):
            deny("the review service records the protocol's steps; you never run `tvr respond`, `tvr file`, `tvr assign` "
                 "and the like. " + ("Submit with `tva-review submit` (answer findings with `tva-kit respond …`)."
                                     if kit.ROLE != "reviewer" else "Finish with `tva-kit filing lint`, then your final message."))
        if TOOLCHAIN.search(cmd):
            deny("this would change or delete the prebuilt Lean toolchain or Mathlib (hours to rebuild, and the service "
                 "uses its own copy anyway). Build with `tva-kit build`.")
        if kit.ROLE == "reviewer" and re.search(r"\blake\s+build\b|\btvr(\.py)?\s+check\b|\btva-kit\s+(gate|build)\b", cmd):
            deny("the build folder is the service's and read-only to you, and the gate has passed: check Lean on your "
                 "work files with `cd ~/fpv/tva/lean && lake env lean <file>`.")
        bad = [p for p in write_targets(cmd, cwd) if protected(p)]
        if bad:
            deny(f"this command writes {bad[0]}, which is fixed (the library, its tools and documents, another kernel, "
                 "or a generated record). " + ("Write only in your kernel's folder and ~/scratch."
                                               if kit.ROLE != "reviewer" else "Write only in your round folder and ~/scratch."))
    elif tool in ("Grep", "Glob") and not ti.get("path") and os.path.realpath(cwd) == "/":
        deny("give a `path` (for example ~/fpv/tva/docs, ~/fpv/tva/lean/TVA or ~/fpv/tva/kernels): a search from / "
             "reads the whole filesystem, Mathlib's 8 GB included")
    sys.exit(0)


# ---------------------------------------------------------------- after a tool

def post_tool(inp):
    tool, ti, cwd = inp.get("tool_name", ""), inp.get("tool_input") or {}, inp.get("cwd") or os.getcwd()
    if tool not in EDIT_TOOLS:
        sys.exit(0)
    p = real(ti.get("file_path") or ti.get("notebook_path") or "", cwd)
    if p.endswith(".json") and under(p, os.path.realpath(kit.KD)):
        try:
            json.load(open(p))
        except ValueError as e:
            out({"decision": "block", "reason": f"{os.path.basename(p)} is no longer valid JSON ({e}): fix it now"})
        if kit.ROLE != "reviewer" and p.endswith("model-map.json"):
            msg = captured(kit.cmd_map_check, [])
            out({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "tva-kit map-check:\n" +
                                        kit.short(msg, 1500)}})
        sys.exit(0)
    if (kit.ROLE != "reviewer" and p.endswith(".lean") and under(p, os.path.realpath(os.path.join(kit.KD, "lean")))
            and os.environ.get("TVA_HARNESS_AUTOBUILD", "1") != "0"):
        mod = kit.module_of(p)
        t0 = time.time()
        rc, text = kit.build(mod, timeout=int(os.environ.get("TVA_HARNESS_AUTOBUILD_TIMEOUT", "840")))
        errs, warns, sorries, lines = kit.summarize_build(text, limit=6)
        took = f"{time.time() - t0:.0f} s"
        if rc != 0:
            out({"decision": "block", "reason": f"automatic `lake build {mod}` after your edit ({took}): {len(errs)} "
                 "error(s). Fix the first one first.\n" + "\n".join(lines)[:6000]})
        out({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext":
                                    f"automatic `lake build {mod}`: OK ({took}; {warns} warning(s), {sorries} `sorry`)"
                                    + (" — the gate refuses `sorry`: replace each one" if sorries else "")}})
    sys.exit(0)


# ---------------------------------------------------------------- stop

def stop_state(key, active):
    """How many times in a row the agent tried to stop with nothing changed (`key`)."""
    st = kit.load(STOP_STATE, {}) or {}
    n = st.get("count", 0) + 1 if active and st.get("key") == key else 1
    os.makedirs(kit.SCRATCH, exist_ok=True)
    json.dump({"key": key, "count": n, "time": time.time()}, open(STOP_STATE, "w"))
    return n


def files_key():
    h = hashlib.sha256()
    for root, dirs, files in os.walk(kit.KD):
        for f in sorted(files):
            p = os.path.join(root, f)
            with contextlib.suppress(OSError):
                h.update(f"{p}:{os.path.getmtime(p)}".encode())
    return h.hexdigest()[:16]


def block(reason):
    out({"decision": "block", "reason": reason})


def stop_prover(inp):
    st = kit.service_status()
    if not st:
        sys.exit(0)  # the service is unreachable: do not keep the agent forever
    s = st.get("status")
    if s in ("certified", "escalated", "exhausted", "error", "stopped"):
        sys.exit(0)
    if s == "busy":
        step = (st.get("job") or {}).get("step", "")
        block(f"Do not stop: your submission is under review ({step}). Run `tva-review wait` (it waits up to 9 "
              "minutes; run it again while it says busy), then act on the result.")
    n = stop_state(f"{s}:{st.get('round')}:{files_key()}", inp.get("stop_hook_active"))
    if n > 8:
        sys.exit(0)  # eight tries in a row with nothing changed: let it stop
    job = st.get("job") or {}
    res = job.get("result") or {}
    rejected = f"Your last submission was rejected: {res.get('reason')}. `tva-review log` has the details.\n" \
        if job.get("state") == "done" and res.get("accepted") is False else ""
    if st.get("open_findings"):
        block(rejected + f"Do not stop: findings {', '.join(st['open_findings'])} are open, and the task succeeds only "
              "with a granted certificate. Answer them: `tva-kit findings`; fix what they show; `tva-kit gate`; "
              "`tva-kit respond init` and `tva-kit respond set …` for each; `tva-kit respond lint`; `tva-review submit`.")
    stages = kit.stages()
    todo = [f"[ ] {n_} — {d}" for n_, done, d in stages if not done]
    block(rejected + "Do not stop: no submission of yours has been reviewed yet, and the task succeeds only with a "
          "granted certificate. What is left:\n" + "\n".join(todo[:7]) + "\nContinue with the first of these; when "
          "`tva-kit gate` passes, run `tva-review submit`. If a proof is too hard at full generality, narrow the "
          "conditions (state each narrowing in README.md) rather than stopping.")


def stop_reviewer(inp):
    who = kit.reviewer_id()
    try:
        _, N, fp, rp = kit.filing_paths()
    except Exception:  # noqa: BLE001
        sys.exit(0)
    if not os.path.exists(fp):
        reason = f"Do not stop yet: you have not written your filing. Run `tva-kit filing init`, then record your " \
                 "review with `tva-kit filing add/accept/release/reraise/checked`, and `tva-kit filing lint`."
    else:
        rc, text = kit.run_tvr("lint", kit.K, "reviewer", who, timeout=300)
        if rc == 0:
            sys.exit(0)
        reason = "Do not stop yet: your filing is not regular (`tvr lint` says):\n" + kit.short(text, 2500) + \
                 "\nFix it with `tva-kit filing …` (it writes filing.json and report.md consistently), then " \
                 "`tva-kit filing lint` until it says regular."
    n = stop_state(f"{reason[:500]}", inp.get("stop_hook_active"))
    if n > 6:
        sys.exit(0)  # the service sends a reviewer back with `tvr file`'s refusal anyway
    block(reason)


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        inp = {}
    if kit.tvr is None or not kit.K:
        sys.exit(0)
    if event == "session-start":
        session_start(inp)
    elif event == "pre-tool":
        pre_tool(inp)
    elif event == "post-tool":
        post_tool(inp)
    elif event == "stop":
        stop_reviewer(inp) if kit.ROLE == "reviewer" else stop_prover(inp)
    sys.exit(0)


if __name__ == "__main__":
    main()
