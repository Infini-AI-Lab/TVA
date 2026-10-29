"""lr-4b, round 4 (L7): more spellings of a path out of TVA, against the round-4 tool.

Part 1 calls the tool's own `foreign_refs` (tools/tvr.py) on each probe text. Part 2 writes some of
the probes into a scratch copy of the tree and runs the copy's `tools/tvr.py audit` end to end.

`otherproj` stands for a sibling folder of TVA (no project is named). Every path is joined at run
time, so this file holds no path out of TVA, and the output prints no probe text.

Run from the TVA root (or from a clean copy of it):
  python3 -I docs/library-review/round-4/lr-4b/work/l7_probes4.py SCRATCH
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT = os.getcwd()
spec = importlib.util.spec_from_file_location("tvr", os.path.join(ROOT, "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tvr)

S, BS, H = "/", "\\", "~"
F, O, T = "fpv", "otherproj", "tva"
UH = S + "home" + S + "someone"
Q, SQ = '"', "'"
WHERE = os.path.join(ROOT, "docs")

# (id, should be flagged, description, text)
PROBES = [
    # the fpv/<name> rule, with the separator spelled in other ways
    ("A1", True, "fpv, two slashes, sibling", F + S + S + O),
    ("A2", True, "fpv, dot, sibling", F + S + "." + S + O),
    ("A3", True, "fpv, tva, dot-dot, sibling", F + S + T + S + ".." + S + O),
    ("A4", True, "fpv, backslash, sibling", F + BS + O),
    ("A5", True, "fpv, %2f, sibling", F + "%2f" + O),
    ("A6", True, "fpv, &#47;, sibling", F + "&#47;" + O),
    ("A7", True, "fpv, JSON \\u002f, sibling", F + BS + "u002f" + O),
    ("A8", True, "fpv, \\x2f, sibling", F + BS + "x2f" + O),
    ("A9", True, "fpv, fullwidth solidus (NFKC), sibling", F + "／" + O),
    ("A10", True, "FPV then Otherproj, upper case", F.upper() + S + "Otherproj"),
    ("A11", True, "fpv, &#x2F;, sibling", F + "&#x2F;" + O),
    ("A12", True, "fpv, JSON \\/, sibling", F + BS + S + O),
    # spellings a reader resolves but that break the literal text fpv/<name>
    ("B1", True, "shell: home, then \"fpv\" quoted, then sibling", "cd " + H + S + Q + F + Q + S + O),
    ("B2", True, "shell: home/fpv/, then the sibling quoted", "cd " + H + S + F + S + Q + O + Q),
    ("B3", True, "shell: home/fpv'' (empty quotes)/sibling", "cd " + H + S + F + SQ + SQ + S + O),
    ("B4", True, "shell: home/fpv/ then backslash-newline then sibling", "cd " + H + S + F + S + BS + "\n" + O + S + "x"),
    ("B5", True, "shell: absolute home, fpv, then the sibling in single quotes, then x", "cat " + UH + S + F + S + SQ + O + SQ + S + "x"),
    ("B6", True, "HTML named entity &sol; as the separator (Markdown renders it as /)", F + "&sol;" + O),
    ("B7", True, "Python octal escape \\057 as the separator", "p = " + Q + F + BS + "057" + O + Q),
    ("B8", True, "Python \\N{SOLIDUS} as the separator", "p = " + Q + F + BS + "N{SOLIDUS}" + O + Q),
    ("B9", True, "shell brace expansion: home, fpv, then {sibling,x}", "ls " + H + S + F + S + "{" + O + ",x}"),
    ("B10", True, "shell glob home/fpv/[o]therproj", "ls " + H + S + F + S + "[o]therproj"),
    # home directory spellings the page promises, with other separators or nesting
    ("C1", True, "${HOME:-${X}} (nested expansion) then a sibling directly under home", "cd ${HOME:-${X}}" + S + O + S + "x"),
    ("C2", True, "%USERPROFILE% then a backslash path", "cd %USERPROFILE%" + BS + O + BS + "x"),
    ("C3", True, "PowerShell $HOME then a backslash path", "cd $HOME" + BS + O + BS + "x"),
    ("C4", True, "`echo ~` (backquotes) then one component", "cd `echo " + H + "`" + S + O),
    ("C5", True, "$(…) nested three deep then one component", "cd $(a $(b $(echo " + H + ")))" + S + O),
    ("C6", True, "Makefile $(HOME) then a sibling", "cd $(HOME)" + S + O + S + "x"),
    ("C7", True, "%7E (escaped tilde) then a sibling", "cd %7E" + S + O + S + "x"),
    ("C8", True, "file://localhost then a home path", "file://localhost" + UH + S + O + S + "x"),
    ("C9", True, "a Markdown link to home, fpv, sibling", "[x](" + UH + S + F + S + O + ")"),
    ("C10", True, "absolute: root, fpv, sibling", S + "root" + S + F + S + O),
    ("C11", True, "home/../someone/sibling", H + S + ".." + S + "someone" + S + O),
    ("C12", True, "home path with a doubled slash", UH.replace(S + "someone", S + S + "someone") + S + O),
    ("C13", True, "tilde-user then sibling", H + "someone" + S + O),
    ("C14", True, "home, fpv, tva, two dot-dots, sibling", H + S + F + S + T + S + ".." + S + ".." + S + O),
    # controls: TVA's own text, which must pass
    ("K1", False, "control: home/fpv/tva/docs", "see " + H + S + F + S + T + S + "docs"),
    ("K2", False, "control: fpv/<name> and fpv/… placeholders", "a path " + F + S + "<name>, " + F + S + "…"),
    ("K3", False, "control: a web URL whose path holds fpv then sibling", "https" + ":" + S + S + "example.org" + S + F + S + O),
    ("K4", False, "control: a relative path x/home/someone/sibling (no home)", "x" + UH + S + O),
    ("K5", False, "control: round-<N>/author and {kernel}/review", "round-<N>" + S + "author, {kernel}" + S + "review"),
]


def part1():
    wrong = 0
    for pid, want, desc, text in PROBES:
        got = bool(tvr.foreign_refs(text, WHERE))
        ok = got == want
        wrong += not ok
        print(f"{pid}: {'flagged' if got else 'passed '} ({'as the rule asks' if ok else 'NOT as the rule asks'}) - {desc}")
    print(f"part 1: {len(PROBES)} probes, {wrong} not as the rule asks")


def copy(scratch):
    dst = os.path.join(scratch, "tva")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(ROOT, dst, ignore=lambda d, n: [x for x in n if x in (".lake", ".venv", ".git", "__pycache__")],
                    symlinks=True)
    return dst


def audit(dst):
    r = subprocess.run([sys.executable, "-I", os.path.join(dst, "tools", "tvr.py"), "audit"], cwd=dst,
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def part2(scratch):
    """End to end: each case changes one thing in a fresh copy and runs the copy's audit."""
    dst = copy(scratch)
    code, out = audit(dst)
    print(f"part 2 baseline: exit {code}, {'independent' if code == 0 else 'NOT independent'}")

    def doc_case(text):
        def f(d):
            with open(os.path.join(d, "docs", "SEMANTICS.md"), "a") as fh:
                fh.write("\n" + text + "\n")
        return f

    def pinned_primitive(d):
        # a new file in primitives/, listed with its sha256 in the semantics manifest, holding a path
        # into a sibling folder of TVA
        p = os.path.join(d, "primitives", "triton-3.8.0-notes.txt")
        body = "see " + H + S + F + S + O + S + "results.txt\n"
        with open(p, "w") as fh:
            fh.write(body)
        mp = os.path.join(d, "primitives", "semantics-manifest.json")
        m = json.load(open(mp))
        m.setdefault("sources", []).append({"source": "triton-3.8.0-notes.txt",
                                           "sha256": hashlib.sha256(body.encode()).hexdigest()})
        json.dump(m, open(mp, "w"), indent=2)

    def kernel_source(d):
        # the kernel's pinned source gains a comment with a path into a sibling folder; kernel.json
        # pins the new sha256
        k = "vec_add_f32"
        mp = os.path.join(d, "kernels", k, "kernel.json")
        m = json.load(open(mp))
        src = os.path.join(d, "kernels", k, m["source"]["file"])
        with open(src, "a") as fh:
            fh.write("# from " + H + S + F + S + O + S + "vec_add.py\n")
        m["source"]["sha256"] = hashlib.sha256(open(src, "rb").read()).hexdigest()
        json.dump(m, open(mp, "w"), indent=2)

    cases = [
        ("E1", True, "docs: shell, home then \"fpv\" quoted then sibling (B1)", doc_case("cd " + H + S + Q + F + Q + S + O)),
        ("E2", True, "docs: shell, home/fpv/ then the sibling quoted (B2)", doc_case("cd " + H + S + F + S + Q + O + Q)),
        ("E3", True, "docs: &sol; as the separator (B6)", doc_case(H + "&sol;" + F + "&sol;" + O)),
        ("E4", True, "docs: Python octal escapes as the separator (B7)",
         doc_case("p = " + Q + H + BS + "057" + F + BS + "057" + O + Q)),
        ("E5", True, "primitives: a new pinned file with a path into a sibling", pinned_primitive),
        ("E6", True, "kernel source (pinned by kernel.json) with a path into a sibling", kernel_source),
        ("E7", True, "control-check: docs, plain home, fpv, sibling", doc_case(H + S + F + S + O)),
    ]
    wrong = 0
    for cid, want, desc, change in cases:
        dst = copy(scratch)
        change(dst)
        code, out = audit(dst)
        got = code != 0
        ok = got == want
        wrong += not ok
        print(f"{cid}: {'flagged' if got else 'passed '} ({'as the rule asks' if ok else 'NOT as the rule asks'}) - {desc}")
    print(f"part 2: {len(cases)} cases, {wrong} not as the rule asks")


if __name__ == "__main__":
    part1()
    if len(sys.argv) > 1:
        part2(sys.argv[1])
