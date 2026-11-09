#!/usr/bin/env python3
"""The sandbox harness's Claude Code hooks (HARNESS.md), for the prover and the reviewers.

  hook.py session-start | subagent-start | pre-tool | post-tool | stop   (the hook's JSON on stdin, Claude Code's on stdout)

- session-start: where the agent is and what to do next (also after a compaction); the prover's clock starts.
- subagent-start: a subagent gets the role's rules and the Lean advice (the manual reaches only the main agent).
- pre-tool: the Write and Edit tools write only where the role writes (their path is exact); no command
  that records for the service or breaks the prebuilt toolchain; no search of the whole filesystem; no
  waiting loop that spins a core; none is resumed (SendMessage). The prover's subagents run in the background
  (the prover works on, and each report starts a new turn), and never run `tva-review`; a reviewer's subagents
  run in the foreground.
  A Bash command is not read for the files it writes: shell is too irregular to read reliably.
- post-tool: after an edit of a JSON record, say whether it parses; after the prover's Bash command, say
  which fixed files it changed (compared with a snapshot taken at the session's start). An edit of the
  kernel's Lean is not built: the prover builds when it chooses (`tva-kit build`, `lake build`), and the
  gate builds everything. A kernel module that Lake reports slow, in a Bash `lake build` (its output, or
  the file it went to), gets the advice to split it. Every hour, the prover's agents are told the time left.
- stop: the prover does not stop while the service is reviewing or findings are open, or before a first
  submission, except to wait for its background subagents (Claude Code then waits for them, and each report
  starts a new turn); a reviewer does not stop before its filing is regular.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvakit as kit  # noqa: E402

HOME = os.path.expanduser("~")
STOP_STATE = os.path.join(kit.SCRATCH, ".stop-hook.json")
PROGRESS = os.path.join(kit.SCRATCH, "progress.md")


TEXT_FIELDS = ("additionalContext", "permissionDecisionReason", "reason")


def with_clock(obj):
    """`obj` with the prover's clock at the end of each text the agent hears (once): a model's own sense of
    the time passed can be hours off (pilot 7 guessed 5 h at 15 min), and every note is a chance to correct it."""
    line = kit.clock_line() if kit.ROLE != "reviewer" else ""
    if not line:
        return obj
    for d in (obj, obj.get("hookSpecificOutput") or {}):
        for k in TEXT_FIELDS:
            if isinstance(d.get(k), str) and "used of" not in d[k]:
                d[k] += f"\n\nClock: {line}."
    return obj


def out(obj):
    print(json.dumps(with_clock(obj)))
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
    if os.path.exists(MANUAL) and os.environ.get("TVA_HARNESS_MANUAL", "1") != "0":  # dsh: AGENTS.md holds it
        lines.append(f"Contents of {MANUAL} (your operating manual; follow it):\n\n" + open(MANUAL).read().strip())
    if kit.ROLE == "reviewer":
        who = kit.reviewer_id()
        lines.append(f"You are reviewer `{who}` of the TVA kernel `{kit.K}` (CLAUDE.md is your operating manual). "
                     f"Your TVA copy is ~/fpv/tva; you write only in its kernels/{kit.K}/review/round-N/{who}/ (with "
                     "`tva-kit filing …`) and in ~/scratch.")
        lines.append(captured(kit.cmd_review_brief, ["--short"]))
    else:
        if not os.path.exists(SNAPSHOT):
            with contextlib.suppress(OSError):
                save(SNAPSHOT, fixed_files())
        if not os.path.exists(kit.CLOCK):
            with contextlib.suppress(OSError):
                save(kit.CLOCK, {"start": time.time()})
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


LEAN_ADVICE = (
    "Lean work here: look a name up before you use it (`tva-kit lookup NAME`). Nothing is built for you: "
    "build when you choose (`tva-kit build`); the build's whole output, every error with its goal, is "
    "kept in ~/scratch/build/<Module>.log: read the goals there instead of building again. Lean re-checks a whole "
    "file at every build, so keep files short: work on a hard lemma in a small file of its own (imported "
    "where it is used), so that a try takes seconds, not minutes, and make all the edits of one try before "
    "the build. Never edit a file another agent is editing.")


ATTEMPT_RULES = (
    "You are one proof attempt of several, started at once by the prover at one lemma, each with its own approach "
    "(best of N). Your prompt names the lemma and your attempt K: run `tva-kit try brief LEMMA K` first and follow it. "
    "Work only in your attempt's folder (~/scratch/tries/LEMMA/aK/): your Try.lean below its marker line, and scratch "
    "files there; never edit the kernel's files (the prover merges the winner), never run `lake build`, `tva-kit "
    "build` or `tva-kit gate` (an exclusive build would stop the other attempts' checks). Check with `tva-kit try "
    "check LEMMA K`. The first attempt that Lean accepts wins: when the harness says the attempts are over, end at "
    "once with your report. ")


def subagent_start(inp):
    if kit.ROLE != "reviewer" and inp.get("agent_type") == "proof-attempt":
        text = ATTEMPT_RULES + LEAN_ADVICE.replace("Nothing is built for you: build when you choose (`tva-kit build`); "
                                                   "the build's whole output, every error with its goal, is "
                                                   "kept in ~/scratch/build/<Module>.log: read the goals there instead "
                                                   "of building again. ",
                                                   "The check keeps Lean's whole output in your folder's check.log: read "
                                                   "the goals there instead of checking again. ")
        out({"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": text}})
    if kit.ROLE == "reviewer":
        text = (f"You are a subagent of reviewer `{kit.reviewer_id()}` of the TVA kernel `{kit.K}` (TVA at ~/fpv/tva). "
                "Read only: never edit a file, never run `lake build`, `tvr` or anything that writes outside ~/scratch. "
                "Answer in a few lines with path:line references.")
    else:
        text = (f"You are a subagent of the prover of the TVA kernel `{kit.K}` (TVA at ~/fpv/tva). Write only in "
                f"~/fpv/tva/kernels/{kit.K}/ (never its source/) and ~/scratch; the library, its documents and tools "
                "and the other kernels are fixed. Never run `tvr respond`/`tvr file`, `lake update`/`lake clean`, nor "
                "touch `.lake`. " + LEAN_ADVICE + " End with a short report: what builds, what you changed, what is left.")
        c = kit.clock()
        if c:
            seen = kit.load(CLOCK_NOTES, {}) or {}
            seen[inp.get("agent_id") or "?"] = sum(1 for m in clock_marks(c[1]) if m <= c[0])
            with contextlib.suppress(OSError):
                save(CLOCK_NOTES, seen)
            text += (f" The prover's {kit.clock_line()}: report back before you run out of it (`tva-kit clock` "
                     "prints the time; do not estimate it).")
    out({"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": text}})


# ---------------------------------------------------------------- before a tool

EDIT_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
RECORDING = re.compile(r"\btvr(\.py)?\s+(respond|file|assign|new|freeze|decide|note|record|release)\b")
TOOLCHAIN = re.compile(r"\blake\s+(update|clean|new|init|upgrade)\b|\blake\s+exe\s+cache\b|\belan\s+"
                       r"(update|toolchain|default|override|self)\b|rm\s+-\S*r\S*\s+[^;&|]*\.lake\b")
# `read -t N < /dev/zero` returns at once and is retried: a wait that keeps a core busy (Claude Code blocks `sleep`)
SPIN = re.compile(r"\bread\b[^;&|\n]*<\s*/dev/(zero|urandom|random)\b")
# a resumed subagent keeps its whole earlier context; a new one with the report starts small
RESUME = ("The harness does not resume subagents (SendMessage). To continue a subagent's work, start a new subagent "
          "with the Agent tool and give it the earlier report (what builds, what is left) and what to do next; the "
          "files are on disk.")
# the prover's subagents run in the background (dsh's way): the prover works on, and each report starts a new turn
BACKGROUND = ("Subagents run in the background here: this call returns at once, and the subagent's closing report "
              "comes to you as a message when it ends. Work on something else meanwhile (not on its file), or end "
              "your turn when only waiting is left: the next report starts a new turn.")


def without_heredocs(cmd):
    """The command without the bodies of its here-documents, which are text, not commands."""
    lines, out, i = cmd.split("\n"), [], 0
    while i < len(lines):
        out.append(lines[i])
        i += 1
        for m in HEREDOC.finditer(out[-1]):
            while i < len(lines) and lines[i].strip() != m.group(2):
                i += 1
            i += 1
    return "\n".join(out)


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


TRY_CMD = re.compile(r"\btva-kit\s+try\s+(brief|check)\s+(\S+)\s+(\d+)\b")
ATTEMPT_FORBIDDEN = re.compile(r"\blake\s+build\b|\btva-kit\s+(build|gate)\b|\btva-kit\s+try\s+(start|accept|stop)\b|"
                               r"\btva-review\b")


def attempt_of(inp, tool, ti, cwd):
    """[lemma, K]: the attempt a subagent works on, bound at its first `tva-kit try brief|check LEMMA K` or its
    first write in an attempt's folder (~/scratch/tries/LEMMA/aK/), or None."""
    aid = inp.get("agent_id")
    if not aid or inp.get("agent_type") not in (None, "proof-attempt"):
        return None  # only an attempt is bound (Claude Code names a subagent's type; without it, any subagent)
    m = kit.load(kit.TRY_AGENTS, {}) or {}
    if aid in m:
        return m[aid]
    got = None
    if tool == "Bash":
        x = TRY_CMD.search(without_heredocs(ti.get("command", "")))
        if x and kit.load(os.path.join(kit.try_dir(x.group(2)), "task.json")):
            got = [x.group(2), int(x.group(3))]
    elif tool in EDIT_TOOLS:
        x = re.match(re.escape(os.path.realpath(kit.TRIES)) + r"/([^/]+)/a(\d+)/",
                     real(ti.get("file_path") or ti.get("notebook_path") or "", cwd))
        t = kit.load(os.path.join(kit.TRIES, x.group(1), "task.json")) if x else None
        if t:
            got = [t["name"], int(x.group(2))]
    if got:
        m[aid] = got
        with contextlib.suppress(OSError):
            save(kit.TRY_AGENTS, m)
    return got


def pre_attempt(inp, tool, ti, cwd):
    """The rules of a proof attempt (`tva-kit try`): its own folder only, no exclusive build, and nothing more once
    the attempts are over (another attempt won, the prover accepted or stopped them, or the time is up)."""
    att = attempt_of(inp, tool, ti, cwd)
    if not att and inp.get("agent_type") != "proof-attempt":
        return
    cmd = without_heredocs(ti.get("command", "")) if tool == "Bash" else ""
    if att:
        t = kit.load(os.path.join(kit.try_dir(att[0]), "task.json")) or {}
        over = kit.try_over(t) if t else ""
        if over:
            w = kit.load(os.path.join(kit.try_dir(att[0]), "WINNER")) or {}
            last_check = not w and not t.get("done") and re.match(r"^\s*tva-kit\s+try\s+(check|status)\b", cmd)
            if not last_check or os.path.exists(os.path.join(kit.try_dir(att[0]), f"a{att[1]}", ".final")):
                deny(f"{over}" + (" (yours)" if w.get("attempt") == att[1] else "") + f": attempt {att[1]} is over. End "
                     "now, with no more tool calls: reply with your report (the check's last state, the helper lemmas "
                     "you stated, what you would try next).")
            with contextlib.suppress(OSError):
                open(os.path.join(kit.try_dir(att[0]), f"a{att[1]}", ".final"), "w").close()
    if tool in EDIT_TOOLS:
        p = real(ti.get("file_path") or ti.get("notebook_path") or "", cwd)
        mine = os.path.join(os.path.realpath(kit.try_dir(att[0])), f"a{att[1]}") if att else None
        if not (mine and under(p, mine)) and not under(p, "/tmp"):
            deny(f"{p}: a proof attempt writes only in its own folder" + (f" ({mine}/)" if mine else
                 " (~/scratch/tries/LEMMA/aK/: run `tva-kit try brief LEMMA K` first, as your prompt says)") +
                 "; the prover merges the winner into the kernel's files.")
    if ATTEMPT_FORBIDDEN.search(cmd):
        deny("a proof attempt checks its file with `tva-kit try check LEMMA K` only: a `lake build` (or `tva-kit build`/"
             "`gate`) holds the build folder alone and would stop every other attempt's check, and starting, "
             "accepting or stopping attempts, and submitting, are the prover's.")


def pre_tool(inp):
    tool, ti, cwd = inp.get("tool_name", ""), inp.get("tool_input") or {}, inp.get("cwd") or os.getcwd()
    if kit.ROLE != "reviewer" and inp.get("agent_id"):
        pre_attempt(inp, tool, ti, cwd)
    if tool in EDIT_TOOLS:
        p = real(ti.get("file_path") or ti.get("notebook_path") or "", cwd)
        why = reviewer_write_ok(p) if kit.ROLE == "reviewer" else prover_write_ok(p)
        if why:
            deny(f"{p}: {why}.")
    elif tool == "Bash":
        cmd = without_heredocs(ti.get("command", ""))
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
        if SPIN.search(cmd):
            deny("`read` from /dev/zero returns at once, so this loop keeps a CPU core busy that the Lean builds need. "
                 + ("Do not wait: a subagent's report comes to you by itself (end your turn if only waiting is left), "
                    "`tva-review wait` waits for the review service, and a long build is better split than waited on."
                    if kit.ROLE != "reviewer" else
                    "Do not wait: subagents run in the foreground here (their report comes back when they finish)."))
        if kit.ROLE != "reviewer" and inp.get("agent_id") and re.search(r"(^|[\s;&|(])tva-review\b", cmd):
            deny("`tva-review` is the prover's (submitting, waiting, the service's status): a subagent does its task "
                 "and ends with its report.")
    elif tool == "SendMessage":
        deny(RESUME)
    elif tool in ("Agent", "Task") and kit.ROLE != "reviewer":
        # the prover's subagents run in the background: it works on, and each report starts a new turn (the
        # Stop hook lets it end a turn while one runs, and Claude Code waits for them); set the field always
        out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                    "updatedInput": {**ti, "run_in_background": True},
                                    "additionalContext": BACKGROUND}})
    elif tool in ("Agent", "Task") and ti.get("run_in_background") is not False:
        # a reviewer waits for its subagents: in the background it would have nothing to do but poll; Claude Code
        # may run a subagent there when the field is absent (its test is `run_in_background !== false`), so set it
        out({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                    "updatedInput": {**ti, "run_in_background": False},
                                    "additionalContext": "The harness runs subagents in the foreground: this one's "
                                    "report is the result of this call. To run several at once, start them in one "
                                    "message, each on its own file."}})
    elif tool in ("Grep", "Glob") and not ti.get("path") and os.path.realpath(cwd) == "/":
        deny("give a `path` (for example ~/fpv/tva/docs, ~/fpv/tva/lean/TVA or ~/fpv/tva/kernels): a search from / "
             "reads the whole filesystem, Mathlib's 8 GB included")
    sys.exit(0)


# ---------------------------------------------------------------- after a tool

SNAPSHOT = os.path.join(kit.SCRATCH, ".fixed-files.json")
GENERATED = ("kernels/README.md", "docs/library-review/REPORTS.md")  # indexes that `tvr check` rewrites


def save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"))


def fixed_files():
    """The prover's fixed files, by path from TVA's root, with their size and modification time: the tree
    outside the kernel's folder (build folders aside, and the indexes `tvr` rewrites) and the kernel's source."""
    root, kd = os.path.realpath(kit.ROOT), os.path.realpath(kit.KD)
    files = {}
    for d, dirs, names in os.walk(root):
        if d == kd:
            dirs[:], names = [x for x in dirs if x == "source"], []
        dirs[:] = [x for x in dirs if x not in (".lake", "__pycache__", ".git")]
        for n in names:
            r = os.path.relpath(os.path.join(d, n), root)
            with contextlib.suppress(OSError):
                st = os.lstat(os.path.join(d, n))
                if r not in GENERATED:
                    files[r] = [st.st_size, st.st_mtime_ns]
    return files


def check_fixed():
    """Tell the prover which fixed files its last command changed, each change once."""
    before = kit.load(SNAPSHOT, None)
    now = fixed_files()
    if before is None:
        save(SNAPSHOT, now)
        return
    changed = sorted(r for r in set(before) | set(now) if before.get(r) != now.get(r))
    if not changed:
        return None
    save(SNAPSHOT, now)
    how = lambda r: "removed" if r not in now else "added" if r not in before else "changed"
    return {"decision": "block", "reason": "this command changed files that are fixed: " +
            ", ".join(f"{r} ({how(r)})" for r in changed[:6]) + (f" and {len(changed) - 6} more" if len(changed) > 6 else "") +
            ". The review service gates your submission against its own copy of TVA, so a change here has no effect "
            "there, and your local gate may pass where the service's fails. Undo it; write only in "
            f"kernels/{kit.K}/ (not its source/) and ~/scratch."}


def after_edit(p):
    """What the agent hears after it wrote the kernel file `p`: whether a JSON record parses (and the map
    check). A Lean file is not built: the prover builds when it chooses, and the gate builds everything."""
    if p.endswith(".json") and under(p, os.path.realpath(kit.KD)):
        try:
            json.load(open(p))
        except ValueError as e:
            return {"decision": "block", "reason": f"{os.path.basename(p)} is no longer valid JSON ({e}): fix it now"}
        if kit.ROLE != "reviewer" and p.endswith("model-map.json"):
            msg = captured(kit.cmd_map_check, [])
            return {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "tva-kit map-check:\n" +
                                           kit.short(msg, 1500)}}
    return None


LAKE_BUILD = re.compile(r"\blake\s+build\b")
REDIRECT = re.compile(r"(?:&>>?|\d?>>?|\btee(?:\s+-a)?)\s*([^\s;&|<>()]+)")  # the files a command sends output to


def after_lake(inp, cmd):
    """After a Bash command that ran `lake build`: the module times Lake reports, recorded, the warning when one
    puts the gate at risk, and the advice to split the slow modules, from the command's output or from a file
    it sent the output to (one written while it ran)."""
    resp = inp.get("tool_response")
    text = "\n".join(str(resp.get(k) or "") for k in ("stdout", "stderr")) if isinstance(resp, dict) else str(resp or "")
    start = time.time() - (inp.get("duration_ms") or 0) / 1000 - 2
    for f in REDIRECT.findall(cmd):
        p = real(f.strip("'\""), inp.get("cwd"))
        with contextlib.suppress(OSError):
            st = os.stat(p)
            if stat.S_ISREG(st.st_mode) and st.st_mtime >= start and st.st_size < 32 << 20:
                text += "\n" + open(p, errors="replace").read()
    times = kit.note_build_times(text)
    return "\n\n".join(x for x in (kit.gate_risk_advice(times), kit.split_advice(kit.slow_modules(text))) if x)


def add_note(res, note):
    """The PostToolUse answer `res` (or None) with `note` added to what the agent hears."""
    if not note:
        return res
    if res and res.get("decision") == "block":
        res["reason"] += "\n\n" + note
    elif res:
        res["hookSpecificOutput"]["additionalContext"] += "\n\n" + note
    else:
        res = {"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": note}}
    return res


DESIGN_NOTES = os.path.join(kit.SCRATCH, ".design-notes.json")  # {agent|file:declaration: proof length when told}


def design_note(inp):
    """A `sorry` inside a long proof (a lemma nobody stated), told to the agent that works on it right after
    its edit or build: once per declaration, and again each time the proof has grown by half."""
    who = inp.get("agent_id") or "main"
    seen = kit.load(DESIGN_NOTES, {}) or {}
    new = []
    for s in kit.sorry_sites():
        p, line, name, n, how = s
        key = f"{who}|{os.path.relpath(p, kit.KD)}:{name}"
        if how == "inside" and n > kit.LONG_PROOF and n >= 1.5 * seen.get(key, 0):
            new.append(s)
            seen[key] = n
    if not new:
        return ""
    with contextlib.suppress(OSError):
        save(DESIGN_NOTES, seen)
    where = "; ".join(f"{os.path.relpath(p, kit.KD)}:{line} in {name} ({n}-line proof)" for p, line, name, n, _ in new[:3])
    return (f"Design: a `sorry` inside a long proof ({where}). A `sorry` inside a long proof is "
            "a lemma nobody stated. State what is left as its own lemma, with one hypothesis per model line it needs "
            "(numeric steps on plain values, in a small file of their own), and have this proof call it (skill "
            "tva-design, §2-3): a hole in a long walk costs minutes per build and hides what is left.")


CLOCK_NOTES = os.path.join(kit.SCRATCH, ".clock-notes.json")  # {agent: marks already told}


def clock_marks(limit):
    """When the prover's agents are told the time (seconds from the start): every hour, and at three
    quarters and seven eighths of the limit."""
    return sorted(set(range(3600, limit, 3600)) | {int(limit * 0.75), int(limit * 0.875)})


def clock_note(inp):
    """The time used, once per mark for each agent (the prover or one of its subagents), with what it
    means while nothing is submitted."""
    c = kit.clock()
    if not c:
        return ""
    used, limit = c
    passed = sum(1 for m in clock_marks(limit) if m <= used)
    who = inp.get("agent_id") or "main"
    seen = kit.load(CLOCK_NOTES, {}) or {}
    if passed <= seen.get(who, 0):
        return ""
    seen[who] = passed
    with contextlib.suppress(OSError):
        save(CLOCK_NOTES, seen)
    note = f"Clock: {kit.clock_line()}."
    st = kit.service_status() or {}
    if st.get("status") == "busy" or (st.get("submissions") or {}).get("reviewed"):
        return note
    if not inp.get("agent_id") and not os.path.exists(kit.AUDIT) and "def Post" in kit.lean_text():
        note += (" The claim is written but not audited: start the claim-audit subagent (stage 7) before more proof "
                 "work. In an earlier run it found six statements that were false or could not be proved, before "
                 "any time went into them.")
    if inp.get("agent_id"):
        return note + (" Report back to the prover soon with what builds and what is left." if used >= limit / 2 else "")
    if used >= limit * 0.75:
        return note + (" Nothing is submitted yet, and the task succeeds only with a certificate: finish or narrow what "
                       "is left now (a smaller `Inst.ok`, each narrowing stated in README.md), then `tva-kit gate` and "
                       "`tva-review submit`. A round of review takes 20 to 60 minutes.")
    if used >= limit / 2:
        return note + (" Half the time is gone and nothing is submitted. A round of review takes 20 to 60 minutes and "
                       f"findings need another: aim to submit by {kit.hm(limit * 0.75)}. If a proof is not converging, "
                       "narrow the claim (each narrowing stated in README.md) so that the rest is provable. For a "
                       "stated lemma that resists, `tva-kit try start LEMMA` runs several subagents at it at once, "
                       "each with its own approach; the first proof Lean accepts wins.")
    return note


def post_tool(inp):
    tool, ti, cwd = inp.get("tool_name", ""), inp.get("tool_input") or {}, inp.get("cwd") or os.getcwd()
    res = None
    if tool == "Bash" and kit.ROLE != "reviewer":
        res = check_fixed()
        cmd = without_heredocs(ti.get("command", ""))
        if LAKE_BUILD.search(cmd):
            res = add_note(res, after_lake(inp, cmd))
    elif tool in EDIT_TOOLS:
        res = after_edit(real(ti.get("file_path") or ti.get("notebook_path") or "", cwd))
    elif tool in ("Agent", "Task") and "SendMessage" in json.dumps(inp.get("tool_response") or ""):
        # Claude Code ends a subagent's report with "use SendMessage … to continue this agent"
        res = add_note(res, RESUME)
    if kit.ROLE != "reviewer":
        if inp.get("agent_type") != "proof-attempt" and (
                (tool in EDIT_TOOLS and str(ti.get("file_path", "")).endswith(".lean")) or (
                tool == "Bash" and re.search(r"\blake\b", ti.get("command", "")))):
            res = add_note(res, design_note(inp))
        res = add_note(res, clock_note(inp))
    if res:
        out(res)
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
    if any(t.get("type") == "subagent" and t.get("status") == "running" for t in inp.get("background_tasks") or []):
        sys.exit(0)  # ending a turn to wait for subagents: Claude Code waits, and each report starts a new turn
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
    nxt, _ = kit.next_stage(stages)
    todo = [f"[ ] {n_} — {d}" for n_, done, d in sorted(stages, key=lambda x: x[0] != nxt) if not done]
    block(rejected + "Do not stop: no submission of yours has been reviewed yet, and the task succeeds only with a "
          "granted certificate. What is left:\n" + "\n".join(todo[:7]) + "\nContinue with the first of these; when "
          "`tva-kit gate` passes, run `tva-review submit`. If a proof is too hard at full generality, narrow the "
          "conditions (state each narrowing in README.md) rather than stopping." +
          (f"\n{kit.clock_line()}." if kit.clock() else ""))


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
    elif event == "subagent-start":
        subagent_start(inp)
    elif event == "pre-tool":
        pre_tool(inp)
    elif event == "post-tool":
        post_tool(inp)
    elif event == "stop":
        stop_reviewer(inp) if kit.ROLE == "reviewer" else stop_prover(inp)
    sys.exit(0)


if __name__ == "__main__":
    main()
