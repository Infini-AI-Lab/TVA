"""TVA library evidence: does `tvr audit` catch each kind of reference to another project it promises to
catch (docs/REVIEW_PROTOCOL.md §2; library issue L7), and does it leave TVA's own text alone?

Usage, from the TVA root:
  python3 -I evidence/independence-audit/probes.py NAME SCRATCH > evidence/independence-audit/out.txt

  NAME     the name of another project, one whose sha256 is in tvr's FOREIGN set. It is read from the
           command line, so that this file does not contain it.
  SCRATCH  an empty scratch directory outside TVA.

For each probe a fresh copy of the TVA tree (without .venv, .lake, .git, __pycache__) is made in
SCRATCH, the probe changes one thing, and the copy's own `tools/tvr.py audit` runs. The output names
each probe, whether the audit flagged it, and whether that is what the rule asks. No probe text is
printed, so the output names no project and no path. Paths are joined at run time, so that this file
holds no path out of TVA either.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys

TVA = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAME, SCRATCH = sys.argv[1], sys.argv[2]
SEP = "/"
HOME = "~"
DOLLAR = "$"
UP = ".." + SEP
USER_HOME = SEP + "home" + SEP + "someone"
Q, SQ, BS, BQ = '"', "'", "\\", "\x60"  # a double and a single quote, a backslash, a backquote
LIB = "lean/TVA/Claim.lean"
DOC = "docs/SEMANTICS.md"
KERNEL = "vec_add_f32"
KLEAN = f"kernels/{KERNEL}/lean/VecAdd.lean"
LAKE = "lean/lakefile.toml"


def fresh():
    dst = os.path.join(SCRATCH, "tva")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(TVA, dst, symlinks=True,
                    ignore=shutil.ignore_patterns(".venv", ".lake", ".git", "__pycache__"))
    return dst


PROBLEMS = (("the name of another project", "name"), ("a path outside", "path"), ("a relative path", "relative path"),
            ("not UTF-8", "not text"), ("a symbolic link", "link"), ("header", "header"), ("an import", "header"))


def kind(line):
    """The kind of one problem the audit reports: `<file> imports <module>`, or `<file>: <problem>`."""
    if ": " not in line and " imports " in line:
        return "import"
    where, _, problem = line.partition(": ")
    for key, k in PROBLEMS:
        if key in problem:
            return k
    if where.startswith("lean/lakefile"):
        return "lake"
    if where.startswith("lean/lake-manifest"):
        return "manifest"
    return "other"


def audit(root):
    """The audit's exit code, and the kinds of problem it reports (never its text)."""
    r = subprocess.run([sys.executable, "-I", os.path.join(root, "tools", "tvr.py"), "audit"],
                       capture_output=True, text=True, cwd=root)
    kinds = sorted({kind(line) for line in r.stdout.splitlines() if line.strip()}) if r.returncode else []
    return r.returncode, kinds


def append(root, relpath, text):
    p = os.path.join(root, relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a") as f:
        f.write(text)


def prepend(root, relpath, text):
    p = os.path.join(root, relpath)
    old = open(p).read()
    with open(p, "w") as f:
        f.write(text + old)


def replace(root, relpath, old, new):
    p = os.path.join(root, relpath)
    s = open(p).read()
    assert old in s, (relpath, old)
    with open(p, "w") as f:
        f.write(s.replace(old, new, 1))


def write(root, relpath, data, mode="w"):
    p = os.path.join(root, relpath)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode) as f:
        f.write(data)


def manifest_edit(root, fn):
    p = os.path.join(root, "lean", "lake-manifest.json")
    m = json.load(open(p))
    fn(m)
    json.dump(m, open(p, "w"), indent=2)


def symlink_out(root):
    target = os.path.join(SCRATCH, "outside")
    os.makedirs(target, exist_ok=True)
    os.symlink(target, os.path.join(root, "docs", "linked"))


def other_package(m):
    m["packages"].append({"type": "path", "name": "other", "dir": UP + UP + "other", "inherited": False})


def mathlib_fork(m):
    for p in m["packages"]:
        if p["name"] == "mathlib":
            p["url"] = "https://github.com/someone/mathlib4"


def kernel_source_path(root):
    """The kernel's source gains a comment with a home path into a sibling folder; kernel.json pins the
    new sha256, as a prover would."""
    mp = os.path.join(root, "kernels", KERNEL, "kernel.json")
    m = json.load(open(mp))
    src = os.path.join(root, "kernels", KERNEL, m["source"]["file"])
    with open(src, "a") as f:
        f.write("# from " + HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "vec_add.py\n")
    m["source"]["sha256"] = hashlib.sha256(open(src, "rb").read()).hexdigest()
    json.dump(m, open(mp, "w"), indent=2)


def pinned_primitive(root):
    """A new file in primitives/, pinned by its sha256 in the semantics manifest, with a home path into
    a sibling folder."""
    body = "see " + HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "results.txt\n"
    write(root, "primitives/triton-3.8.0-notes.txt", body)
    mp = os.path.join(root, "primitives", "semantics-manifest.json")
    m = json.load(open(mp))
    m["sources"].append({"source": "triton-3.8.0-notes.txt", "sha256": hashlib.sha256(body.encode()).hexdigest()})
    json.dump(m, open(mp, "w"), indent=2)


NAME_SPLIT = "abc" + NAME[0].lower() + NAME[1:].upper() + "def"  # the name across a case boundary

PROBES = [
    # (id, what, whether the rule asks the audit to flag it, mutation)
    ("I1", "library Lean: `import Aesop`", True, lambda r: prepend(r, LIB, "import Aesop\n")),
    ("I2", "library Lean: `module` then `public import Aesop`", True,
     lambda r: prepend(r, LIB, "module\npublic import Aesop\n")),
    ("I3", "library Lean: `module` then `meta import Aesop`", True,
     lambda r: prepend(r, LIB, "module\nmeta import Aesop\n")),
    ("I4", "library Lean: `public meta import Aesop`", True,
     lambda r: prepend(r, LIB, "module\npublic meta import Aesop\n")),
    ("I5", "library Lean: `import all Aesop`", True, lambda r: prepend(r, LIB, "module\nimport all Aesop\n")),
    ("I6", "library Lean: `import «Aesop»` (escaped name)", True, lambda r: prepend(r, LIB, "import «Aesop»\n")),
    ("I7", "library Lean: a second import on the same line as the first", True,
     lambda r: replace(r, LIB, "import TVA.Lang.Mem", "import TVA.Lang.Mem import Aesop")),
    ("I8", "library Lean: an import after a block comment in the header", True,
     lambda r: prepend(r, LIB, "/- a comment -/\nimport Aesop\n")),
    ("I9", "library Lean: an import with a nested comment before the module name", True,
     lambda r: prepend(r, LIB, "import /- a /- nested -/ comment -/ Aesop\n")),
    ("I10", "library Lean: `prelude` then an import", True, lambda r: prepend(r, LIB, "prelude\nimport Aesop\n")),
    ("I11", "library Lean: a header the audit cannot read (an unterminated comment)", True,
     lambda r: prepend(r, LIB, "/- unterminated\n")),
    ("I12", "library Lean: an `import` line after the header", True,
     lambda r: append(r, LIB, "\nimport Aesop\n")),
    ("I13", "a Lean file under lean/ outside lean/TVA that imports another root", True,
     lambda r: write(r, "lean/Other.lean", "import Aesop\n")),
    ("I14", "kernel Lean: an import of another root", True, lambda r: prepend(r, KLEAN, "import Aesop\n")),
    ("I15", "control: kernel Lean importing its own module and TVA", False,
     lambda r: write(r, f"kernels/{KERNEL}/lean/VecAdd/Extra.lean", "import VecAdd\nimport TVA.Claim\n")),
    ("I16", "control: `import` inside a docstring after the header", False,
     lambda r: append(r, LIB, "\n/-- An example:\nimport is a keyword. -/\ndef lr2Probe : Nat := 0\n")),
    ("K1", "lakefile: [[require]] of another package, `name` first", True,
     lambda r: append(r, LAKE, '\n[[require]]\nname = "other"\npath = "' + UP + UP + 'other"\n')),
    ("K2", "lakefile: [[require]] of another package, `path` before `name`", True,
     lambda r: append(r, LAKE, '\n[[require]]\npath = "' + UP + UP + 'other"\nname = "other"\n')),
    ("K3", "lakefile: [[lean_lib]] whose srcDir is outside TVA", True,
     lambda r: append(r, LAKE, '\n[[lean_lib]]\nname = "Other"\nsrcDir = "' + UP + UP + 'other/lean"\n')),
    ("K4", "lakefile: a comment naming the other project", True, lambda r: append(r, LAKE, f"\n# see {NAME}\n")),
    ("K5", "lean/lakefile.lean present (lake would read it)", True,
     lambda r: write(r, "lean/lakefile.lean", "import Lake\nopen Lake DSL\npackage TVA\n")),
    ("K6", "lakefile: a top-level key that passes arguments to Lean", True,
     lambda r: replace(r, LAKE, 'version = "2.0.0"', 'version = "2.0.0"\nmoreLeanArgs = ["--load-dynlib=x.so"]')),
    ("K7", "lakefile: the mathlib requirement taken from a git fork", True,
     lambda r: replace(r, LAKE, 'scope = "leanprover-community"\n', 'git = "https://github.com/someone/mathlib4"\n')),
    ("K8", "lakefile: the TVA library with a second root", True,
     lambda r: replace(r, LAKE, '[[lean_lib]]\nname = "TVA"\n', '[[lean_lib]]\nname = "TVA"\nroots = ["TVA", "Other"]\n')),
    ("K9", "lakefile: a kernel library with a srcDir into another kernel's folder", True,
     lambda r: replace(r, LAKE, f'srcDir = "../kernels/{KERNEL}/lean"', 'srcDir = "../kernels/other/lean"')),
    ("K10", "lake-manifest.json: a direct path package", True, lambda r: manifest_edit(r, other_package)),
    ("K11", "lake-manifest.json: mathlib from another repository", True, lambda r: manifest_edit(r, mathlib_fork)),
    ("N1", "docs: the name as a word", True, lambda r: append(r, DOC, f"\nAs in {NAME}.\n")),
    ("N2", "docs: the name in upper case", True, lambda r: append(r, DOC, f"\nAs in {NAME.upper()}.\n")),
    ("N3", "docs: the name as the prefix of an identifier (upper case + `Lemmas`)", True,
     lambda r: append(r, DOC, f"\nAs in `{NAME.upper()}Lemmas.foo`.\n")),
    ("N4", "docs: the name joined by an underscore (name + `_v2`)", True, lambda r: append(r, DOC, f"\nAs in {NAME}_v2.\n")),
    ("N5", "Lean docstring: the name as a word", True, lambda r: append(r, LIB, f"\n/-! From {NAME}. -/\n")),
    ("N6", "docs: the name run into a lower-case word (name + `lemmas`)", True,
     lambda r: append(r, DOC, f"\nAs in {NAME.lower()}lemmas.\n")),
    ("N7", "docs: the name in camel case before a word (name + `Lemmas`)", True,
     lambda r: append(r, DOC, f"\nAs in {NAME.lower()}Lemmas.\n")),
    ("N8", "control: the name's letters across a case boundary inside a longer word", False,
     lambda r: append(r, DOC, f"\nAs in {NAME_SPLIT}.\n")),
    ("P1", "docs: a home path into a sibling directory of TVA", True,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "x.md.\n")),
    ("P2", "docs: a home path into an archive directory", True,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + "fpv" + SEP + "archive" + SEP + "otherproj" + SEP + "x.lean.\n")),
    ("P3", "docs: a home path to a directory whose name starts with `tva`", True,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + "fpv" + SEP + "tva-old" + SEP + "x.lean.\n")),
    ("P4", "docs: a relative path that climbs out of TVA", True,
     lambda r: append(r, DOC, "\nSee " + UP + UP + "otherproj/lean/X.lean.\n")),
    ("P5", "docs: an absolute path outside /home", True,
     lambda r: append(r, DOC, "\nSee " + SEP + "scratch" + SEP + "other" + SEP + "x.lean.\n")),
    ("P6", "docs: a home path into TVA that climbs out with `..`", True,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + "fpv" + SEP + "tva" + SEP + UP + "otherproj" + SEP + "x.\n")),
    ("P7", "docs: an absolute /home path to a directory whose name starts with `tva`", True,
     lambda r: append(r, DOC, "\nSee " + USER_HOME + SEP + "fpv" + SEP + "tva2" + SEP + "x.\n")),
    ("P8", "docs: a Markdown link that climbs out of TVA", True,
     lambda r: append(r, DOC, "\nSee [x](" + UP + UP + "otherproj/x.md).\n")),
    ("P9", "control: paths inside TVA, the toolchain, the system, and a URL", False,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + "fpv" + SEP + "tva" + SEP + "lean, " + USER_HOME + SEP + "fpv" +
                      SEP + "tva" + SEP + "docs, " + HOME + SEP + ".elan" + SEP + "bin, " + SEP + "usr" + SEP + "bin" + SEP +
                      "python3, " + UP + "evidence" + SEP + "x, and https://github.com/triton-lang/triton/blob/v3.8.0/x.\n")),
    # round 3 (L7): home paths in the spellings a shell or a browser resolves, and paths glued to punctuation
    ("P10", "docs: `$HOME` then a sibling directory of TVA", True,
     lambda r: append(r, DOC, "\nSee " + DOLLAR + "HOME" + SEP + "fpv" + SEP + "otherproj" + SEP + "x.md.\n")),
    ("P11", "docs: `${HOME}` then a sibling directory of TVA", True,
     lambda r: append(r, DOC, "\nSee " + DOLLAR + "{HOME}" + SEP + "fpv" + SEP + "otherproj" + SEP + "x.md.\n")),
    ("P12", "docs: a `file:` URL with three slashes into a sibling directory", True,
     lambda r: append(r, DOC, "\nSee file:" + SEP + SEP + USER_HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "x.md.\n")),
    ("P13", "docs: the same `file:` URL in angle brackets", True,
     lambda r: append(r, DOC, "\nSee <file:" + SEP + SEP + USER_HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "x.md>.\n")),
    ("P14", "docs: `~<user>` then a sibling directory", True,
     lambda r: append(r, DOC, "\nSee " + HOME + "someone" + SEP + "fpv" + SEP + "otherproj" + SEP + "x.md.\n")),
    ("P15", "docs: an absolute home path glued to a semicolon", True,
     lambda r: append(r, DOC, "\nRun cd x;" + USER_HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "run.sh.\n")),
    ("P16", "docs: `%USERPROFILE%` then a sibling directory", True,
     lambda r: append(r, DOC, "\nSee %USERPROFILE%" + SEP + "fpv" + SEP + "otherproj" + SEP + "x.md.\n")),
    ("P17", "docs: a relative path after `=` that climbs out of TVA", True,
     lambda r: append(r, DOC, "\nSet P=" + UP + UP + "otherproj" + SEP + "x.\n")),
    ("P18", "evidence JSON: a home path with escaped slashes", True,
     lambda r: write(r, "evidence/hints-dry-run/x.json", '{"p": "' + USER_HOME.replace(SEP, "\\" + SEP) + "\\" + SEP + "fpv"
                     + "\\" + SEP + "otherproj" + '"}\n')),
    ("P19", "control: the same spellings inside TVA or above it, placeholders, and a web URL", False,
     lambda r: append(r, DOC, "\nSee " + DOLLAR + "HOME" + SEP + "fpv" + SEP + "tva" + SEP + "docs, file:" + SEP + SEP +
                      USER_HOME + SEP + "fpv" + SEP + "tva" + SEP + "x, " + HOME + SEP + "fpv, round-<N>" + SEP +
                      "author" + SEP + "notes.md, {kernel}" + SEP + "review" + SEP + "x, https:" + SEP + SEP +
                      "example.org" + USER_HOME + SEP + "fpv" + SEP + "otherproj.\n")),
    # round 4 (L7): spellings that passed round 3's scanner, and the `fpv/<name>` rule
    ("P20", "docs: `${HOME:-}` (parameter expansion with a default) then a sibling", True,
     lambda r: append(r, DOC, "\nRun cd " + DOLLAR + "{HOME:-}" + SEP + "fpv" + SEP + "otherproj.\n")),
    ("P21", "docs: `${HOME%/}` (suffix removed) then a sibling", True,
     lambda r: append(r, DOC, "\nRun cd " + DOLLAR + "{HOME%" + SEP + "}" + SEP + "fpv" + SEP + "otherproj.\n")),
    ("P22", "docs: `$(echo ~)` (command substitution) then a sibling", True,
     lambda r: append(r, DOC, "\nRun cd " + DOLLAR + "(echo " + HOME + ")" + SEP + "fpv" + SEP + "otherproj.\n")),
    ("P23", "evidence JSON: a home path with `\\u002F` escapes", True,
     lambda r: write(r, "evidence/hints-dry-run/y.json", '{"p": "' + HOME + "\\u002F" + "fpv" + "\\u002F" + "otherproj" + '"}\n')),
    ("P24", "docs: a shell redirect into a sibling, no space after `>`", True,
     lambda r: append(r, DOC, "\nRun python3 run.py >" + USER_HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "out.txt\n")),
    ("P25", "docs: `2>~` into a sibling", True,
     lambda r: append(r, DOC, "\nRun python3 run.py 2>" + HOME + SEP + "fpv" + SEP + "otherproj" + SEP + "err.txt\n")),
    ("P26", "docs: a home path not under `fpv` right after a closing bracket", True,
     lambda r: append(r, DOC, "\nSee (the notes)" + USER_HOME + SEP + "otherdir" + SEP + "x.md\n")),
    ("P27", "docs: `~/…` not under `fpv` right after `]`", True,
     lambda r: append(r, DOC, "\nSee x]" + HOME + SEP + "otherdir" + SEP + "x.md\n")),
    ("P28", "docs: a home path right after `-`", True,
     lambda r: append(r, DOC, "\nSee a-" + USER_HOME + SEP + "otherdir" + SEP + "x.md\n")),
    ("P29", "docs: `$(dirname $0)` then `..` out of TVA", True,
     lambda r: append(r, DOC, "\nRun cd " + DOLLAR + "(dirname " + DOLLAR + "0)" + (SEP + "..") * 6 + SEP + "otherdir\n")),
    ("P30", "evidence script: an f-string home path into a sibling", True,
     lambda r: write(r, "evidence/hints-dry-run/z.py", "p = f\"{os.environ['HOME']}" + SEP + "fpv" + SEP + "otherproj" + SEP
                     + "x\"\n")),
    ("P31", "docs: URL-escaped slashes (`%2F`)", True,
     lambda r: append(r, DOC, "\nSee " + HOME + "%2F" + "fpv" + "%2F" + "otherproj" + "%2F" + "x\n")),
    ("P32", "docs: `fpv/<name>` with no home prefix", True,
     lambda r: append(r, DOC, "\nSee fpv" + SEP + "otherproj" + SEP + "x.md\n")),
    ("P33", "docs: `fpv/` then a folder whose name starts with `tva`", True,
     lambda r: append(r, DOC, "\nSee fpv" + SEP + "tva-old" + SEP + "x.md\n")),
    ("P34", "control: TVA, placeholders and redirects that stay in TVA", False,
     lambda r: append(r, DOC, "\nSee fpv" + SEP + "tva" + SEP + "docs, fpv" + SEP + "<dir>" + SEP + "x, " + HOME + SEP + "fpv, fpv"
                      + SEP + "…, run.py >out.txt, echo x >" + HOME + SEP + "fpv" + SEP + "tva" + SEP + "x, round-<N>" + SEP
                      + "author" + SEP + "notes.md, kernels" + SEP + "{kernel}" + SEP + "x, " + DOLLAR + "(date -Is), "
                      + DOLLAR + "{HOME}" + SEP + "fpv" + SEP + "tva" + SEP + "docs, 100%, a-b" + SEP + "c.\n")),
    # round 5 (library issue L7): shell quoting, more escapes, home spellings with nesting and
    # backslashes, and the files pinned by sha256 (a kernel's source, a new file in primitives/)
    ("P35", "docs: shell, home, then `fpv` in double quotes, then a sibling", True,
     lambda r: append(r, DOC, "\nRun cd " + HOME + SEP + Q + "fpv" + Q + SEP + "otherproj" + SEP + "x\n")),
    ("P36", "docs: shell, an absolute home path with the sibling in single quotes", True,
     lambda r: append(r, DOC, "\nRun cat " + USER_HOME + SEP + "fpv" + SEP + SQ + "otherproj" + SQ + SEP + "x\n")),
    ("P37", "docs: shell, empty quotes after `fpv`", True,
     lambda r: append(r, DOC, "\nRun cd " + HOME + SEP + "fpv" + SQ + SQ + SEP + "otherproj\n")),
    ("P38", "docs: the named HTML entity `&sol;` as the separator", True,
     lambda r: append(r, DOC, "\nSee " + HOME + "&sol;" + "fpv" + "&sol;" + "otherproj.\n")),
    ("P39", "evidence script: Python octal escapes `\\057` as the separator", True,
     lambda r: write(r, "evidence/hints-dry-run/z.py", "p = " + Q + HOME + BS + "057" + "fpv" + BS + "057" + "otherproj" + Q + "\n")),
    ("P40", "evidence script: Python `\\N{SOLIDUS}` as the separator", True,
     lambda r: write(r, "evidence/hints-dry-run/z.py", "p = " + Q + HOME + BS + "N{SOLIDUS}" + "fpv" + BS + "N{SOLIDUS}"
                     + "otherproj" + Q + "\n")),
    ("P41", "docs: shell brace expansion after `fpv/`", True,
     lambda r: append(r, DOC, "\nRun ls " + HOME + SEP + "fpv" + SEP + "{" + "otherproj,x}" + SEP + "y\n")),
    ("P42", "docs: `${HOME:-${X}}` (a nested expansion) then a folder under home", True,
     lambda r: append(r, DOC, "\nRun cd " + DOLLAR + "{HOME:-" + DOLLAR + "{X}}" + SEP + "otherdir" + SEP + "x\n")),
    ("P43", "docs: `%USERPROFILE%` then a backslash path", True,
     lambda r: append(r, DOC, "\nRun cd %USERPROFILE%" + BS + "otherdir" + BS + "x\n")),
    ("P44", "docs: PowerShell `$HOME` then a backslash path", True,
     lambda r: append(r, DOC, "\nRun cd " + DOLLAR + "HOME" + BS + "otherdir" + BS + "x\n")),
    ("P45", "docs: a backquote command substitution of the home directory", True,
     lambda r: append(r, DOC, "\nRun cd " + BQ + "echo " + HOME + BQ + SEP + "otherdir\n")),
    ("P46", "docs: `$(…)` nested three deep, then a folder", True,
     lambda r: append(r, DOC, "\nRun cd " + DOLLAR + "(a " + DOLLAR + "(b " + DOLLAR + "(echo " + HOME + ")))" + SEP
                      + "otherdir\n")),
    ("P47", "docs: a line continuation (backslash, line break) inside a path into a sibling", True,
     lambda r: append(r, DOC, "\nRun cd " + HOME + SEP + "fpv" + SEP + BS + "\n" + "otherproj" + SEP + "x\n")),
    ("P48", "kernel source (pinned by kernel.json): a comment with a home path into a sibling", True, kernel_source_path),
    ("P49", "primitives: a new file, pinned in the semantics manifest, with a home path into a sibling", True,
     pinned_primitive),
    ("P50", "control: the round-5 spellings inside TVA, and placeholders", False,
     lambda r: append(r, DOC, "\nSee " + HOME + SEP + Q + "fpv" + Q + SEP + "tva" + SEP + "docs, fpv" + SEP + Q + "<dir>" + Q
                      + ", fpv" + SEP + "{tva,<dir>}" + SEP + "x, " + HOME + "&" + "sol;fpv&" + "sol;tva, " + DOLLAR + "{HOME:-"
                      + DOLLAR + "{X}}" + SEP + "fpv" + SEP + "tva, " + BQ + "max" + BQ + SEP + BQ + "min" + BQ + ".\n")),
    ("P51", "lexical limit (protocol §2): a glob, `cd` then a relative path, a shell variable", False,
     lambda r: append(r, DOC, "\nRun ls " + HOME + SEP + "fpv" + SEP + "*" + SEP + "x; cd " + HOME + SEP + "fpv && cat "
                      + "otherproj" + SEP + "x; D=" + HOME + SEP + "fpv; cat " + DOLLAR + "D" + SEP + "otherproj.\n")),
    ("N9", "docs: the name with its first letter as a JSON `\\u` escape", True,
     lambda r: append(r, DOC, "\nAs in " + "\\u%04x" % ord(NAME[0]) + NAME[1:] + ".\n")),
    ("F1", "evidence: the name inside a `.log` file", True, lambda r: append(r, "evidence/hints-dry-run/run.log", f"{NAME}\n")),
    ("F2", "evidence: the name inside a `.csv` file", True, lambda r: append(r, "evidence/hints-dry-run/data.csv", f"{NAME},1\n")),
    ("F3", "evidence: the name inside a `.sh` script", True, lambda r: append(r, "evidence/hints-dry-run/run.sh", f"# {NAME}\n")),
    ("F4", "kernels/README.md: the name", True, lambda r: append(r, "kernels/README.md", f"\n{NAME}\n")),
    ("F5", "lean/lakefile.toml read as text: a home path out of TVA in a comment", True,
     lambda r: append(r, LAKE, "\n# " + HOME + SEP + "fpv" + SEP + "otherproj\n")),
    ("F6", "evidence: a file that is not UTF-8 text", True,
     lambda r: write(r, "evidence/hints-dry-run/data.bin", b"\xff\xfe\x00binary", "wb")),
    ("F7", "docs: a symbolic link to a directory outside TVA", True, symlink_out),
    ("F8", "a kernel reviewer's work file with an unusual extension naming the project", True,
     lambda r: write(r, f"kernels/{KERNEL}/review/round-1/r-vecadd/work/notes.org", f"{NAME}\n")),
    ("F9", "tools: the name in a new tool file", True, lambda r: write(r, "tools/helper.sh", f"# {NAME}\n")),
]


def main():
    assert hashlib.sha256(NAME.lower().encode()).hexdigest() in open(os.path.join(TVA, "tools", "tvr.py")).read(), \
        "NAME must be one of tvr's hashed names"
    os.makedirs(SCRATCH, exist_ok=True)
    code, kinds = audit(fresh())
    print(f"baseline (the tree as it is): exit {code}, {'independent' if code == 0 else 'flagged ' + ', '.join(kinds)}")
    wrong = 0
    for pid, what, should, mutate in PROBES:
        root = fresh()
        mutate(root)
        code, kinds = audit(root)
        caught = code != 0
        ok = caught == should
        wrong += not ok
        print(f"{pid}: {'flagged [' + ', '.join(kinds) + ']' if caught else 'passed'} "
              f"({'as the rule asks' if ok else 'WRONG'}) - {what}")
    print(f"{len(PROBES)} probes, {wrong} wrong")


if __name__ == "__main__":
    main()
