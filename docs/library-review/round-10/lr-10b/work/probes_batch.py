"""lr-10b, round 10: probes of the batch's tool fixes (L50, L52), on small inputs and on a scratch copy.

Usage: python3 -I docs/library-review/round-10/lr-10b/work/probes_batch.py SCRATCH
  SCRATCH  an empty folder outside TVA; its path is never printed.

The tree is never written. No GPU, no Triton, no Lean.
"""
import json
import os
import shutil
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

SCRATCH = os.path.realpath(sys.argv[1])
assert os.path.isdir(SCRATCH) and not os.listdir(SCRATCH), "SCRATCH must be an empty folder"


def show(label, value):
    print(f"- {label}: {str(value).replace(SCRATCH, '<scratch>').replace(TVA, '<tva>')}")


def p1():
    print("## P1 (L50): which functions `jit_functions` finds, and what the ranges then need\n")
    kernel = ("    pid = tl.program_id(0)\n"
              "    x = tl.load(p + pid)\n"
              "    tl.store(p + pid, helper(x))\n")
    sources = {
        "decorated helper (control)": (
            "import triton\nimport triton.language as tl\n\n@triton.jit\ndef helper(x):\n    return tl.atomic_add(x, 1)\n\n"
            "@triton.jit\ndef kernel(p):\n" + kernel),
        "helper made by a call, helper = triton.jit(_helper)": (
            "import triton\nimport triton.language as tl\n\ndef _helper(x):\n    return tl.atomic_add(x, 1)\n\n"
            "helper = triton.jit(_helper)\n\n@triton.jit\ndef kernel(p):\n" + kernel),
        "helper under an aliased decorator, from triton import jit as tjit": (
            "import triton\nimport triton.language as tl\nfrom triton import jit as tjit\n\n@tjit\ndef helper(x):\n"
            "    return tl.atomic_add(x, 1)\n\n@triton.jit\ndef kernel(p):\n" + kernel),
        "helper under a decorator bound to a name, deco = triton.jit": (
            "import triton\nimport triton.language as tl\ndeco = triton.jit\n\n@deco\ndef helper(x):\n"
            "    return tl.atomic_add(x, 1)\n\n@triton.jit\ndef kernel(p):\n" + kernel),
    }
    for name, src in sources.items():
        lines = src.split("\n")
        first = next(i for i, l in enumerate(lines, 1) if l.startswith("def kernel"))
        last = len(lines) - 1
        # the prover's ranges hold only the kernel; the map covers its code lines
        man = {"source": {"ranges": [[first - 1, last]], "unused": []}}
        mm = {"entries": [{"src": [first - 1, first], "drop": "decl"},
                          {"src": list(range(first + 1, last + 1)), "model": "x"}]}
        probs, missing = tvr.map_problems(man, mm, src, "x")
        show(f"{name}: jit functions found", tvr.jit_functions(src))
        show(f"{name}: map problems with ranges holding the kernel only", probs or "none (passes)")
    print()


def p2():
    print("## P2 (L52): a recorded file changed, and the ledger re-recorded by a hand-written line\n")
    root = os.path.join(SCRATCH, "p2")
    for d in ("docs",):
        shutil.copytree(os.path.join(TVA, d), os.path.join(root, d),
                        ignore=shutil.ignore_patterns(".lake", "__pycache__", ".git", ".venv"))
    lp = tvr.LIBRARY_LEDGER_PATH
    show("the library ledger of the copy, as shipped", tvr.ledger_problems(root, lp, lp) or "consistent")
    events = [json.loads(l) for l in open(os.path.join(root, lp), encoding="utf-8") if l.strip()]
    rec = next(e for e in events if e.get("dir") == "docs/library-review/round-7/lr-7a")
    rep = "docs/library-review/round-7/lr-7a/report.md"
    text = open(os.path.join(root, rep), encoding="utf-8").read()
    # an edit of a frozen record: L7's verdict turned around
    text = text.replace("- **L7** REOPEN", "- **L7** ACCEPT", 1)
    open(os.path.join(root, rep), "w", encoding="utf-8").write(text)
    show("after an edit of round 7's lr-7a report (REOPEN to ACCEPT)", tvr.ledger_problems(root, lp, lp))
    # route 1: append one line that records the folder again, with the new sha256
    with open(os.path.join(root, lp), "a", encoding="utf-8") as f:
        f.write(json.dumps({"seq": len(events) + 1, "time": "2026-10-28T00:00:00Z", "type": "record",
                            "actor": "coordinator", "round": 7, "who": "lr-7a", "dir": rec["dir"], "linted": False,
                            "files": {**rec["files"], rep: tvr.file_sha(os.path.join(root, rep))}},
                           sort_keys=True) + "\n")
    show("route 1, one appended line that records the folder again", tvr.ledger_problems(root, lp, lp) or "consistent")
    show("route 1, the release check's records step on the copy", [x for x in tvr.records_problems(root) if "round-7" in x]
         or "nothing about round 7")
    # route 2: restore the ledger, and edit the sha256 in the original line instead
    lines = open(os.path.join(root, lp), encoding="utf-8").read().splitlines()[:-1]
    i = next(n for n, l in enumerate(lines) if json.loads(l).get("dir") == rec["dir"])
    e = json.loads(lines[i])
    e["files"][rep] = tvr.file_sha(os.path.join(root, rep))
    lines[i] = json.dumps(e, sort_keys=True)
    open(os.path.join(root, lp), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    show("route 2, the original line's sha256 edited", tvr.ledger_problems(root, lp, lp) or "consistent")
    print()


def p3():
    print("## P3 (L7): which git, bwrap and find the release check runs\n")
    import subprocess
    fake = os.path.join(SCRATCH, "p3-bin")  # stands for a user-writable folder of the home directory on PATH (a conda env's bin)
    os.makedirs(fake)
    log = os.path.join(SCRATCH, "p3.log")
    for name in ("git", "bwrap", "find"):
        real = shutil.which(name, path="/usr/bin")
        p = os.path.join(fake, name)
        open(p, "w").write(f"#!/bin/sh\necho {name} \"$1\" >> {log}\nexec {real} \"$@\"\n")
        os.chmod(p, 0o755)
    show("the PATH folders before /usr/bin in this session, by kind",
         [("home" if d.startswith(os.path.expanduser("~")) else d) for d in os.environ["PATH"].split(os.pathsep)
          if d and os.path.realpath(d) != "/usr/bin"][:4])
    code = ("import sys, shutil; sys.dont_write_bytecode=True; sys.path.insert(0, 'tools'); import tvr; "
            "print('shutil.which(bwrap) is the fake one:', shutil.which('bwrap') == sys.argv[1] + '/bwrap'); "
            "tvr.shipped_files(); tvr.system_problems(['/usr/share/doc']); "
            "print('git run by _git:', tvr._git('--version').stdout.decode().strip())")
    env = dict(os.environ, PATH=fake + os.pathsep + os.environ["PATH"])
    r = subprocess.run([sys.executable, "-I", "-c", code, fake], cwd=TVA, capture_output=True, text=True, env=env)
    print(r.stdout.strip().replace(SCRATCH, "<scratch>"))
    show("the programs the fake folder's wrappers saw called (first argument)",
         sorted(set(open(log).read().split("\n")) - {""}) if os.path.exists(log) else "none")
    print()


def p4():
    print("## P4 (L40): the comment reader on Lean's other quote-like tokens\n")
    cases = {
        "a guillemet name holding a quote": 'def «a"b» : Nat := 1  #eval 7\ndef y := 2 -- "\n',
        "a character literal `'\"'`": "def c : Char := '\"'  #eval 7\n",
        "a character literal `'\\''`": "def c : Char := '\\''  #eval 7\n",
        "a primed name before a string": "def x' : String := \"a\"  #eval 7\n",
        "a raw string `r#\"a\"b\"#`": 'def s : String := r#"a"b"#  #eval 7\n',
        "a guillemet name, then `open … in`": 'def «a"b» : Nat := 1  open Nat in\ntheorem t : zero = 0 := rfl -- "\n',
    }
    for name, text in cases.items():
        code = tvr._uncommented(text)
        show(f"{name}: refused as a string with a brace", tvr.brace_string_problems(text) or "no")
        show(f"{name}: the command after it is read as code", "#eval" in code or "open Nat" in code)
    print()


if __name__ == "__main__":
    p1()
    p2()
    p3()
    p4()
