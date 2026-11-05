#!/usr/bin/env python3
"""The prover's hooks under DeepSeek Harness (dsh): its Claude Code hook bridge's events, translated into the
Claude Code events `hook.py` reads (HARNESS.md, "The dsh prover").

  dsh_hook.py session-start | subagent-start | pre-tool | post-tool | stop   (the bridge's JSON on stdin)

dsh's bridge (`@deepseek-ai/dsh-hooks-claude-code`) runs Claude Code command hooks, with three differences
this script makes up for:
- the tools are dsh's: `bash`, `write`, `edit`, `read`, `grep`, `glob`, `subagent` (with dsh's arguments);
  they are renamed to Claude Code's (`Bash`, `Write`, …, `Agent`), `workdir` folded into `cwd`;
- an event names no agent: every one carries its session's id, the main agent's or a subagent's. The main
  session is the first to start (this script records it); every other session is a subagent, given to
  `hook.py` as `agent_id`. A subagent's turn ending is no `Stop` of the prover (Claude Code sends it
  `SubagentStop`), so it is let through. The bridge's `stop_hook_active` is always false, so this script
  says whether the prover's last stop was blocked, as Claude Code does;
- a subagent has no type: the prover names one in the first line of the `subagent` prompt (`Role: NAME`);
  a call without a known role is refused with the list. The role's instructions (the Claude Code agent file,
  /.claude/agents/NAME.md, without its header) reach the child at its start, with `hook.py`'s rules.

The manual is dsh's instructions file ($DSH_HOME/AGENTS.md), which dsh restores after a compaction, so the
session-start hook leaves it out (TVA_HARNESS_MANUAL=0).
"""
import contextlib
import fcntl
import json
import os
import re
import subprocess
import sys

LIB = os.path.dirname(os.path.abspath(__file__))
HOOK = os.path.join(LIB, "hook.py")
AGENTS = "/.claude/agents"
STATE = os.path.join(os.path.expanduser("~"), "scratch", ".dsh-sessions.json")
TOOLS = {"bash": "Bash", "write": "Write", "edit": "Edit", "read": "Read", "grep": "Grep", "glob": "Glob",
         "subagent": "Agent"}
ROLE_LINE = re.compile(r"\A\s*Role:\s*([A-Za-z][\w-]*)")


def roles():
    with contextlib.suppress(OSError):
        return sorted(f[:-3] for f in os.listdir(AGENTS) if f.endswith(".md")) + ["general"]
    return ["general"]


def role_text(role):
    """The role's agent file without its front matter."""
    with contextlib.suppress(OSError):
        t = open(os.path.join(AGENTS, f"{role}.md")).read()
        return re.sub(r"\A---\n.*?\n---\n", "", t, flags=re.S).strip()
    return ""


@contextlib.contextmanager
def state():
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE + ".lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            s = json.load(open(STATE))
        except (OSError, ValueError):
            s = {}
        s.setdefault("main", None)
        s.setdefault("children", {})
        s.setdefault("pending", [])
        yield s
        tmp = STATE + ".tmp"
        json.dump(s, open(tmp, "w"), indent=1)
        os.replace(tmp, STATE)


def run_hook(event, inp):
    """`hook.py EVENT` on the translated input: its JSON answer (or None)."""
    env = {**os.environ, "TVA_HARNESS_MANUAL": "0"}
    p = subprocess.run([sys.executable, "-I", HOOK, event], input=json.dumps(inp), capture_output=True, text=True,
                       env=env)
    if p.stderr:
        sys.stderr.write(p.stderr)
    with contextlib.suppress(ValueError):
        return json.loads(p.stdout) if p.stdout.strip() else None
    return None


def out(obj):
    if obj:
        hs = obj.get("hookSpecificOutput")
        if isinstance(hs, dict):
            hs.pop("updatedInput", None)  # the bridge does not apply it
        print(json.dumps(obj))
    sys.exit(0)


def translate(inp, s):
    """The Claude Code shape of a dsh tool event, for `hook.py`."""
    sid = inp.get("session_id")
    tool = inp.get("tool_name", "")
    ti = dict(inp.get("tool_input") or {})
    cwd = inp.get("cwd") or os.getcwd()
    if tool == "bash" and ti.get("workdir"):
        cwd = os.path.join(cwd, ti["workdir"])
    if tool == "subagent":
        m = ROLE_LINE.match(ti.get("prompt") or "")
        ti = {"description": ti.get("description", ""), "prompt": ti.get("prompt", ""),
              "subagent_type": m.group(1) if m else "general", "run_in_background": False}
    res = {**inp, "tool_name": TOOLS.get(tool, tool), "tool_input": ti, "cwd": cwd}
    if sid and s["main"] and sid != s["main"]:
        res["agent_id"] = sid
        if s["children"].get(sid) == "proof-attempt":
            res["agent_type"] = "proof-attempt"
    return res


def main():
    event = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        inp = json.load(sys.stdin)
    except ValueError:
        inp = {}
    sid = inp.get("session_id")
    with state() as s:
        if event in ("session-start", "pre-tool", "post-tool", "stop") and not s["main"] and sid \
                and sid not in s["children"]:
            s["main"] = sid  # the first session seen that no subagent start named
        main_sid = s["main"]
        if event == "subagent-start" and sid:
            s["children"][sid] = s["pending"].pop(0) if s["pending"] else "general"
        role = s["children"].get(sid)
        if event == "pre-tool" and inp.get("tool_name") == "subagent" and (not main_sid or sid == main_sid):
            m = ROLE_LINE.match((inp.get("tool_input") or {}).get("prompt") or "")
            if m and m.group(1) in roles():
                s["pending"].append(m.group(1))
        ev = translate(inp, s) if event in ("pre-tool", "post-tool") else inp
    child = bool(sid and main_sid and sid != main_sid)

    if event == "session-start":
        if child:
            sys.exit(0)
        out(run_hook("session-start", inp))
    if event == "subagent-start":
        r = run_hook("subagent-start", {**inp, "agent_id": sid,
                                        **({"agent_type": "proof-attempt"} if role == "proof-attempt" else {})})
        extra = role_text(role) if role and role != "general" else ""
        text = "\n\n".join(x for x in ((((r or {}).get("hookSpecificOutput") or {}).get("additionalContext") or ""),
                                       f"Your instructions as the {role} subagent:\n\n{extra}" if extra else "") if x)
        out({"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": text}} if text else None)
    if event == "pre-tool":
        if inp.get("tool_name") == "subagent":
            m = ROLE_LINE.match((inp.get("tool_input") or {}).get("prompt") or "")
            if not m or m.group(1) not in roles():
                out({"hookSpecificOutput": {
                    "hookEventName": "PreToolUse", "permissionDecision": "deny",
                    "permissionDecisionReason":
                        "Start the prompt of every subagent with a line `Role: NAME`, NAME one of "
                        + ", ".join(roles()) + " (the harness then gives it that role's instructions; `general` "
                        "for none), then the whole task: the subagent does not see your conversation."}})
        if child and inp.get("tool_name") == "bash" and re.search(r"(^|[\s;&|(])tva-review\b",
                                                                   (inp.get("tool_input") or {}).get("command") or ""):
            out({"hookSpecificOutput": {
                "hookEventName": "PreToolUse", "permissionDecision": "deny",
                "permissionDecisionReason": "`tva-review` is the prover's (submitting, waiting, the service's "
                                            "status): a subagent does its task and ends with its report."}})
        out(run_hook("pre-tool", ev))
    if event == "post-tool":
        out(run_hook("post-tool", ev))
    if event == "stop":
        if child:
            sys.exit(0)
        # Claude Code marks a stop that follows a blocked one (`stop_hook_active`), which `hook.py` needs to count
        # identical refusals and let a stuck agent stop; the bridge always says false, so this script keeps it
        with state() as s:
            active = bool(s.get("stop_blocked"))
        r = run_hook("stop", {**inp, "stop_hook_active": active})
        with state() as s:
            s["stop_blocked"] = bool(r and r.get("decision") == "block")
        out(r)
    sys.exit(0)


if __name__ == "__main__":
    main()
