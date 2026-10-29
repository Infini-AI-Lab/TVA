"""The gate and the audit after round 7: the manifest check (L38), the map's code lines (L39), the library's
hygiene and lake options (L40), shipped files in skipped folders (L41), stale gates after a library change
(L42), and rule pages for modules in subfolders (L43).

Part U imports the tree's own tools/tvr.py and calls its functions on small inputs (nothing is written).
Part C copies the TVA tree, with its git metadata, to a fresh folder under the system's temporary
directory (the Lean build is copied and Mathlib's packages are linked, as in evidence/gate-digests), changes
one thing at a time there, and runs the copy's `tvr`. The real tree is never written, except for out.txt.

Usage: TMPDIR=<a folder outside the home directory> python3 -I evidence/gate-round7/run.py > evidence/gate-round7/out.txt
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TVA, "tools"))
import tvr  # noqa: E402

spec = importlib.util.spec_from_file_location("gate_digests", os.path.join(TVA, "evidence", "gate-digests", "run.py"))
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)
TQ = '"' * 3
SIBLING = "fpv" + "/" + "sibling" + "/notes"  # a path into a sibling of TVA, built at run time


def round6_loop(src_lines, ranges):
    """The round-6 coverage loop (tools/tvr.py before round 7), for comparison."""
    missing, in_doc = [], False
    for a, b in ranges:
        for i in range(a, b + 1):
            t = src_lines[i - 1].strip() if i <= len(src_lines) else ""
            q = t.count(TQ)
            if in_doc or q:
                if q % 2 == 1:
                    in_doc = not in_doc
                continue
            if t and not t.startswith("#"):
                missing.append(i)
    return set(missing)


def part_u():
    print("# Part U: the tool's functions on small inputs\n")
    print("## U1 (L39): code lines, round 6 against round 7")
    cases = {
        "K1 a triple-quoted argument": ["@triton.jit", "def k(c, x_ptr):", "    x = tl.load(x_ptr)",
                                        f"    tl.device_print({TQ}x{TQ}, tl.atomic_add(c, 1))", "    tl.store(x_ptr, x)"],
        "K2 triple quotes in two comments": ["@triton.jit", "def k(c, x_ptr):", f"    x = tl.load(x_ptr)  # {TQ}",
                                             "    tl.atomic_add(c, 1)", f"    tl.store(x_ptr, x)  # {TQ}"],
        "K3 a docstring (control)": ["@triton.jit", "def k(c, x_ptr):", f"    {TQ}Adds one.",
                                     "    tl.atomic_add(c, 1) is how it would read", f"    {TQ}", "    tl.atomic_add(c, 1)"],
        "K4 an empty triple-quoted string after `;`": ["def k(x_ptr, c_ptr):", f"    x = tl.atomic_add(c_ptr, 2)  ; s = {TQ}{TQ}",
                                                       "    tl.store(x_ptr, x)"],
        "K5 a bare non-ASCII string, then code": ["def k(c):", "    'é' ; tl.atomic_add(c, 1)", "    'a bare string'",
                                                  "    x = f(", f"        {TQ}two", f"        lines{TQ},", "    )"],
    }
    for name, lines in cases.items():
        new, why = tvr.code_lines("\n".join(lines) + "\n")
        old = round6_loop(lines, [[1, len(lines)]])
        print(f"{name}:")
        for i, l in enumerate(lines, 1):
            print(f"  {i}: round 6 {'code' if i in old else '----'} | round 7 {'code' if i in new else '----'} | {l}")
    print(f"a source that does not parse: {tvr.code_lines('def k(:' + chr(10))}")
    print()

    print("## U2 (L38): the manifest check")
    man = tvr.manifest("vec_add_f32")
    print(f"the real manifest: {tvr.manifest_problems('vec_add_f32', man) or 'no problem'}")
    for what, change in (
            ("the namespace of lr-6a's injection", ("lean", "namespace", "Nat), True := fun _ => trivial\n#eval IO.println \"x\"\n#exit\n--")),
            ("a module with a line break", ("lean", "module", "VecAdd\n#exit")),
            ("a namespace with a space", ("lean", "namespace", "Kernels.Vec Add")),
            ("a source file outside the kernel folder", ("source", "file", os.path.join(*[os.pardir] * 3, "README.md")))):
        m = json.loads(json.dumps(man))
        m[change[0]][change[1]] = change[2]
        print(f"{what}: {tvr.manifest_problems('vec_add_f32', m)}")
    print()

    print("## U3 (L40): the library hygiene's view of Lean code (comments removed, literals kept)")
    snippets = {
        "a command after a string holding --": 'def s : String := "--"  #eval 1\n',
        "a command after a char literal of a quote": "def c : Char := '\"'  #eval 2\n",
        "a raw string holding a quote and --": 'def r : String := r#"a"--"#  #eval 3\n',
        "a guillemet name holding --": "def «a--b» : Nat := 1  #eval 4\n",
        "a nested comment": "/- outer /- inner -/ still a comment #eval 5 -/\ndef ok : Nat := 0\n",
        "a docstring naming syntax (prose)": "/-- the `syntax` of a call, and `macro` in prose -/\ndef ok2 : Nat := 0\n",
        "a line comment naming run_cmd": "-- run_cmd in a comment\ndef ok3 : Nat := 0\n",
    }
    for name, s in snippets.items():
        code = tvr._uncommented(s)
        hits = sorted({what for what, pat in tvr.LIBRARY_HYGIENE if re.search(pat, code, re.M)})
        print(f"{name}: {hits or 'nothing flagged'}")
    print(f"the library as it is: {tvr.library_hygiene_problems() or 'nothing flagged'}")
    print()


def copy_with_git(dst):
    gd.copy_tree(dst)
    shutil.copytree(os.path.join(TVA, ".git"), os.path.join(dst, ".git"), symlinks=True)


def run(root, *args):
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py"] + list(args), cwd=root, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def short(out, n=3):
    lines = [l for l in out.splitlines() if l.strip()]
    return " | ".join(lines[:n])[:400]


def part_c():
    print("# Part C: on a copy of the tree\n")
    tmp = tempfile.mkdtemp(prefix="tva-gate7-")
    root = os.path.join(tmp, "tva")
    try:
        copy_with_git(root)
        p = lambda *x: os.path.join(root, *x)
        k = "vec_add_f32"
        print("## C0: the copy as it is")
        print(f"tvr audit: exit {run(root, 'audit')[0]} ({short(run(root, 'audit')[1], 1)})")
        g = json.load(open(p("kernels", k, "gate.json")))
        print(f"the shipped gate of {k}: ok {g['ok']}, stale: "
              f"{subprocess.run([sys.executable, '-I', '-c', 'import sys; sys.path.insert(0, \"tools\"); import tvr; import json; g = json.load(open(tvr.kpath(\"vec_add_f32\", \"gate.json\"))); print(tvr.gate_staleness(\"vec_add_f32\", g))'], cwd=root, capture_output=True, text=True).stdout.strip()}")
        print()

        print("## C1 (L43): rule pages for modules in subfolders")
        os.makedirs(p("lean", "TVA", "Rules", "Num"))
        open(p("lean", "TVA", "Rules", "Num", "Exp2.lean"), "w").write("import TVA.Rules.Num\n")
        print(f"lean/TVA/Rules/Num/Exp2.lean added: tvr audit exit {run(root, 'audit')[0]} ({short(run(root, 'audit')[1], 2)})")
        probe = ("import sys; sys.path.insert(0, 'tools'); import tvr; "
                 "c = {'TVA.Rules.Num.Exp2.acc': {'module': 'TVA.Rules.Num.Exp2'}}; "
                 "print(sorted(x for x in tvr.library_files(c, c) if x.startswith('file:')))")
        print("  files in the digest of an item that reaches only TVA.Rules.Num.Exp2: " +
              subprocess.run([sys.executable, "-I", "-c", probe], cwd=root, capture_output=True, text=True).stdout.strip())
        os.makedirs(p("lean", "TVA", "Rules", "Extra"))
        open(p("lean", "TVA", "Rules", "Extra", "Y.lean"), "w").write("import TVA.Rules.Num\n")
        print(f"lean/TVA/Rules/Extra/Y.lean added (no page above it): tvr audit exit {run(root, 'audit')[0]} "
              f"({short(run(root, 'audit')[1], 2)})")
        shutil.rmtree(p("lean", "TVA", "Rules", "Num"))
        shutil.rmtree(p("lean", "TVA", "Rules", "Extra"))
        print()

        print("## C2 (L40): package-wide Lean options in lean/lakefile.toml")
        lake = p("lean", "lakefile.toml")
        base = open(lake).read()
        for opt in ("debug.skipKernelTC = true", "maxHeartbeats = 400000", "linter.unusedVariables = false",
                    "trace.Meta.synthInstance = true", "pp.all = true", 'autoImplicit = "yes"'):
            open(lake, "w").write(base.replace("autoImplicit = false\n", "autoImplicit = false\n" + opt + "\n", 1)
                                  if not opt.startswith("autoImplicit") else base.replace("autoImplicit = false", opt, 1))
            rc, out = run(root, "audit")
            print(f"[leanOptions] {opt}: tvr audit exit {rc} ({short(out, 1)})")
        open(lake, "w").write(base)
        print()

        print("## C3 (L41): a note in a nested .lake folder")
        os.makedirs(p("docs", ".lake"))
        open(p("docs", ".lake", "note.md"), "w").write(f"see ~/{SIBLING}\n")
        ls = subprocess.run(["git", "-C", root, "ls-files", "--others", "--exclude-standard", "docs"], capture_output=True, text=True).stdout.split()
        print(f"untracked docs/.lake/note.md: shipped (git ls-files): {'docs/.lake/note.md' in ls}; tvr audit exit {run(root, 'audit')[0]}")
        subprocess.run(["git", "-C", root, "add", "-f", "docs/.lake/note.md"], capture_output=True)
        rc, out = run(root, "audit")
        print(f"after `git add -f`: tvr audit exit {rc} ({short(out, 2)})")
        subprocess.run(["git", "-C", root, "rm", "-q", "--cached", "docs/.lake/note.md"], capture_output=True)
        shutil.rmtree(p("docs", ".lake"))
        open(p("docs", "note.md"), "w").write(f"see ~/{SIBLING}\n")
        rc, out = run(root, "audit")
        print(f"the same note at docs/note.md (control): tvr audit exit {rc} ({short(out, 1)})")
        os.remove(p("docs", "note.md"))
        print()

        print("## C4 (L38): lr-6a's namespace injection, end to end")
        mp = p("kernels", k, "kernel.json")
        saved_man = open(mp).read()
        m = json.loads(saved_man)
        m["lean"]["namespace"] = "Nat), True := fun _ => trivial\n#eval IO.println \"TVR_JSON {}\"\n#exit\n--"
        json.dump(m, open(mp, "w"), indent=2)
        reader = p("lean", ".lake", "tvr", k, "claim_reader.lean")
        if os.path.exists(reader):
            os.remove(reader)
        rc, out = run(root, "check", k)
        print(f"tvr check: exit {rc}: {short(out, 2)}")
        print(f"  claim reader written: {os.path.exists(reader)}")
        open(mp, "w").write(saved_man)
        print()

        print("## C5 (L42): a library change after a certificate")
        gd.update_kernel(root)
        for d in ("round-1", "round-2"):
            shutil.rmtree(p("kernels", k, "review", d), ignore_errors=True)
        for f in ("ledger.jsonl", "certificate.json"):
            if os.path.exists(p("kernels", k, "review", f)):
                os.remove(p("kernels", k, "review", f))
        rc, out = run(root, "check", k)
        print(f"the kernel updated in the copy, its review record cleared: tvr check exit {rc} ({out.splitlines()[-1]})")
        for l in out.splitlines():
            if l.startswith("FAIL"):
                print("  " + l[:200])
        run(root, "assign", k, "r-t")
        run(root, "start", k, "reviewer", "r-t")
        rd = p("kernels", k, "review", "round-1", "r-t")
        json.dump({"reviewer": "r-t", "kernel": k, "round": 1, "findings": [], "release": [], "reraise": [],
                   "accept": "all", "notes": ""}, open(os.path.join(rd, "filing.json"), "w"), indent=2)
        open(os.path.join(rd, "report.md"), "w").write(
            f"# Review of {k}, round 1, by r-t\n\n## Verdict\n\naccept\nA test filing on a scratch copy.\n\n"
            "## Findings\n\nNone.\n\n## Accepted\n\nAll items.\n\n## Checked\n\n- the gate after a library change\n\n"
            "## Not checked\n\nNothing.\n")
        rc, out = run(root, "file", k, "r-t")
        print(f"tvr file (accept all): exit {rc}; certificate.json present: {os.path.exists(p('kernels', k, 'review', 'certificate.json'))}")
        open(p("lean", "TVA", "Lang", "Ops.lean"), "a").write("\n-- a comment added after the certificate\n")
        rc, out = run(root, "index")
        review = open(p("kernels", k, "REVIEW.md")).read()
        index = open(p("kernels", "README.md")).read()
        print(f"a comment added to lean/TVA/Lang/Ops.lean, then tvr index: certificate.json present: "
              f"{os.path.exists(p('kernels', k, 'review', 'certificate.json'))}")
        print("  REVIEW.md: " + next(l for l in review.splitlines() if l.startswith("- Gate:"))[:220])
        print("  REVIEW.md: " + next(l for l in review.splitlines() if l.startswith("**Certificate"))[:220])
        print("  kernels/README.md: " + next(l for l in index.splitlines() if l.startswith(f"| [{k}]"))[:200])
        run(root, "start", k, "reviewer", "r-t")
        rd2 = p("kernels", k, "review", "round-1", "r-t")
        rc, out = run(root, "file", k, "r-t")
        print(f"a further filing: tvr file exit {rc}: {short(out, 1)}")
        rc, out = run(root, "check", k)
        print(f"tvr check again: exit {rc} ({out.splitlines()[-1]})")
        rc, out = run(root, "status", k)
        print("tvr status after it: " + " | ".join(l for l in out.splitlines() if l.startswith("| `") or l.startswith("**"))[:600])
        print()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class Masked:
    """Standard output with the probes' paths masked, so that out.txt holds no path out of TVA."""

    def __init__(self, s):
        self.s = s

    def write(self, t):
        return self.s.write(t.replace(SIBLING, "<a sibling of TVA>")
                            .replace(os.path.join(*[os.pardir] * 3, "README.md"), "<three folders up>/README.md"))

    def flush(self):
        self.s.flush()


def main():
    sys.stdout = Masked(sys.stdout)
    part_u()
    part_c()


if __name__ == "__main__":
    main()
