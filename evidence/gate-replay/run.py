"""TVA library evidence (library issue L13): does the gate catch a kernel whose Lean adds a theorem
without the kernel's check?

Run from the TVA root:  python3 -I evidence/gate-replay/run.py > evidence/gate-replay/out.txt

Each case is one Lean file, compiled to an `.olean` in a fresh temporary directory with the library's
Lean (`lake env lean -o`). Then:
- `elab`: whether the file elaborates (exit code of `lean`), as `lake build` would see it;
- `replay`: the gate's `replay` check, Lean's own `leanchecker` on the compiled module: it replays every
  declaration of the module into the environment of its imports, checking each with Lean's kernel;
- `hygiene`: the gate's `hygiene` list (`HYGIENE` in `tools/tvr.py`), applied to the file's text.

The cases are this folder's Lean files and the library reviewer's bundle of round 2
(`docs/library-review/round-2/lr-2a/work/lean/GateHygiene.lean`, a claim with `Post = False`). The
output prints no path.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
LEAN_DIR = os.path.join(ROOT, "lean")
LAKE = os.path.expanduser("~/.elan/bin/lake")

spec = importlib.util.spec_from_file_location("tvr", os.path.join(ROOT, "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvr)

CASES = [
    ("C1", "control: an ordinary module with a checked theorem", os.path.join(HERE, "Control.lean"), "Control"),
    ("A1", "run_cmd + addDeclCore (doCheck := false): `main : False := True.intro`",
     os.path.join(HERE, "AddUnchecked.lean"), "AddUnchecked"),
    ("A2", "the same, through a command the file defines with `elab` (no `run_cmd`)",
     os.path.join(HERE, "ElabRoute.lean"), "ElabRoute"),
    ("R1", "the library reviewer's bundle of round 2 (Post = False, main added unchecked)",
     os.path.join(ROOT, "docs", "library-review", "round-2", "lr-2a", "work", "lean", "GateHygiene.lean"), "GateHygiene"),
]


def lean_prefix():
    r = subprocess.run([LAKE, "env", "lean", "--print-prefix"], cwd=LEAN_DIR, capture_output=True, text=True)
    return r.stdout.strip()


def last_line(text, scrub):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    errs = [l for l in lines if "error" in l or "kernel" in l]
    line = (errs or lines or [""])[-1]
    for s in scrub:
        line = line.replace(s, "<tmp>")
    return re.sub(r"(/[^\s:'\"]+)+", "<path>", line)[:140]


def main():
    prefix = lean_prefix()
    checker = os.path.join(prefix, "bin", "leanchecker")
    ver = subprocess.run([LAKE, "env", "lean", "--version"], cwd=LEAN_DIR, capture_output=True, text=True).stdout
    m = re.search(r"version ([0-9.]+)", ver)
    print(f"Lean {m.group(1) if m else '?'}, with leanchecker from the same toolchain", flush=True)
    for cid, what, path, mod in CASES:
        tmp = tempfile.mkdtemp()
        try:
            src = os.path.join(tmp, mod + ".lean")
            shutil.copyfile(path, src)
            r = subprocess.run([LAKE, "env", "lean", f"--root={tmp}", "-o", os.path.join(tmp, mod + ".olean"), src],
                               cwd=LEAN_DIR, capture_output=True, text=True)
            elab = "elaborates" if r.returncode == 0 else "fails: " + last_line(r.stdout + r.stderr, [tmp])
            env = dict(os.environ, LEAN_SYSROOT=prefix,
                       PATH=os.path.join(prefix, "bin") + os.pathsep + os.environ.get("PATH", ""))
            if r.returncode == 0:
                c = subprocess.run([LAKE, "env", "bash", "-c", f'LEAN_PATH="{tmp}:$LEAN_PATH" exec "{checker}" {mod}'],
                                   cwd=LEAN_DIR, capture_output=True, text=True, env=env)
                rep = "accepts" if c.returncode == 0 else "REJECTS: " + last_line(c.stderr + c.stdout, [tmp])
            else:
                rep = "not run"
            text = open(path, encoding="utf-8").read()
            hits = sorted({what_ for what_, pat in tvr.HYGIENE if re.search(pat, text, re.M)})
            hyg = "flags " + "; ".join(hits) if hits else "passes"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"{cid} {what}\n    elab: {elab}\n    replay (leanchecker): {rep}\n    hygiene: {hyg}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
