#!/usr/bin/env python3
"""Write the declaration index of the TVA library, docs/index/.

    python3 -I tools/make_index.py           # regenerate docs/index/
    python3 -I tools/make_index.py --check   # exit 1 if docs/index/ is not what the library gives now

It elaborates tools/Index.lean against the built library (lean/, so run `lake build` and
`lake build TVA.Proof` first), which dumps every declaration of a `TVA.*` module: name, kind,
module, line, signature, docstring. From that it writes:

* docs/index/<area>.md: one file per part of the library, each declaration with its signature,
  its docstring and a link to its source line, in source order;
* docs/index/NAMES.tsv: one line per declaration (name, kind, file:line, first line of its
  docstring), for `grep`.

Declarations Lean generates (equation lemmas, recursors, `noConfusion`, unnamed instances,
structure constructors without a docstring) are left out.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEAN = os.path.join(ROOT, "lean")
OUT = os.path.join(ROOT, "docs", "index")
LOCK = "/var/tmp/tva-lake.lock"

# (file, title, what it holds, module prefixes); the first matching prefix wins.
AREAS = [
    ("Claim.md", "Claim", "what a claim is and what `claim.Holds` means", ["TVA.Claim"]),
    ("Base.md", "Base", "bytes, integer widths, float formats and rounding, dtypes", ["TVA.Base"]),
    ("Core.md", "Core", "tensors, memory, the platform, the program monad", ["TVA.Tensor", "TVA.Memory", "TVA.Platform", "TVA.Program", "TVA.Core"]),
    ("Triton.md", "Triton", "the meaning of each `tl.*` operation", ["TVA.Triton"]),
    ("Launch.md", "Launch", "a launch of a grid of programs, its outcomes, `WellBehaved`, `Correct`", ["TVA.Launch"]),
    ("Device.md", "Device", "the device facts `Device P`, the reference platform, tolerance formulas", ["TVA.Device"]),
    ("Proof.Rules.md", "Proof.Rules", "the rule of each `tl.*` operation for `Safe`", ["TVA.Proof.Rules"]),
    ("Proof.Shared.md", "Proof.Shared", "programs that meet only through atomics: `SafeA`, `correct_of_safeA`", ["TVA.Proof.Shared"]),
    ("Proof.Pattern.md", "Proof.Pattern", "1-D blocks, 2-D tiles, block pointers, descriptors", ["TVA.Proof.Pattern"]),
    ("Proof.Int.md", "Proof.Int", "integer widths, casts, `//` and `%`, lanes", ["TVA.Proof.Int"]),
    ("Proof.Float.md", "Proof.Float", "rounding, errors of operations, trees and sums", ["TVA.Proof.Float"]),
    ("Proof.Dot.md", "Proof.Dot", "`acc += tl.dot(a, b)` loops and chunked accumulation", ["TVA.Proof.Dot"]),
    ("Proof.Math.md", "Proof.Scan, Random, Softmax, Recurrence", "scans, sorts, histograms, `tl.rand`, online softmax, decayed recurrences",
     ["TVA.Proof.Scan", "TVA.Proof.Random", "TVA.Proof.Softmax", "TVA.Proof.Recurrence"]),
    ("Proof.Basic.md", "Proof.Basic", "`Safe`, memory and footprint lemmas, composition, `correct_of_safe`", ["TVA.Proof"]),
]

SKIP_PARTS = {"below", "ibelow"}
SKIP_LAST = {"noConfusionType", "noConfusion", "ctorIdx", "ofNat_ctorIdx", "ext_iff", "mk.injEq"}


def dump():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "index.json")
        env = dict(os.environ, TVA_INDEX_OUT=out)
        cmd = ["flock", "-s", LOCK, "lake", "env", "lean", os.path.join(ROOT, "tools", "Index.lean")]
        r = subprocess.run(cmd, cwd=LEAN, env=env, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            sys.exit(f"tools/Index.lean failed:\n{r.stdout}{r.stderr}")
        return json.load(open(out))


def keep(x):
    if x["line"] == 0:
        return False
    parts = x["name"].split(".")
    if SKIP_PARTS & set(parts) or parts[-1] in SKIP_LAST:
        return False
    if not x["doc"] and (any(p.startswith("inst") for p in parts) or (x["kind"] == "constructor" and parts[-1] == "mk")):
        return False
    return True


def path_of(module):
    return "lean/" + module.replace(".", "/") + ".lean"


def area_of(module):
    for a in AREAS:
        if any(module == p or module.startswith(p + ".") for p in a[3]):
            return a
    sys.exit(f"no area for module {module}")


def first_line(doc):
    """The docstring's first sentence, on one line (cut at 240 characters)."""
    text = " ".join(doc.split("\n\n")[0].split())
    for i, c in enumerate(text):
        if c == "." and (i + 1 == len(text) or text[i + 1] == " ") and not text[max(0, i - 3):i].endswith(("e.g", "i.e")):
            text = text[:i + 1]
            break
    return text if len(text) <= 240 else text[:237] + "..."


def render(decls):
    files = {}
    by_area = {}
    for x in decls:
        by_area.setdefault(area_of(x["module"])[0], []).append(x)
    total = 0
    for fname, title, what, _ in AREAS:
        xs = sorted(by_area.get(fname, []), key=lambda x: (x["module"], x["line"], x["name"]))
        total += len(xs)
        out = [f"# {title}", "",
               f"Every declaration of this part of the library ({what}), in source order, with its signature",
               "and docstring. Generated by `tools/make_index.py` from the built library; do not edit.",
               f"{len(xs)} declarations. Back to [the index](../INDEX.md).", ""]
        mod = None
        for x in xs:
            if x["module"] != mod:
                mod = x["module"]
                out += [f"## `{mod}`", "", f"[{path_of(mod)}](../../{path_of(mod)})", ""]
            loc = f"{path_of(mod)}:{x['line']}"
            out += [f"### `{x['name']}`", "",
                    f"{x['kind']} · [{loc}](../../{path_of(mod)}#L{x['line']})", "",
                    "```lean", x["sig"].rstrip(), "```", ""]
            if x["doc"].strip():
                out += [x["doc"].strip(), ""]
        files[fname] = "\n".join(out).rstrip() + "\n"
    tsv = ["# name\tkind\tfile:line\tsummary (first line of the docstring). Generated by tools/make_index.py."]
    for x in sorted(decls, key=lambda x: x["name"]):
        s = first_line(x["doc"]).replace("\t", " ")
        tsv.append(f"{x['name']}\t{x['kind']}\t{path_of(x['module'])}:{x['line']}\t{s}")
    files["NAMES.tsv"] = "\n".join(tsv) + "\n"
    return files, total


def main():
    check = "--check" in sys.argv[1:]
    decls = [x for x in dump() if keep(x)]
    files, total = render(decls)
    if check:
        stale = [f for f, t in files.items()
                 if not os.path.exists(os.path.join(OUT, f)) or open(os.path.join(OUT, f)).read() != t]
        extra = [f for f in (os.listdir(OUT) if os.path.isdir(OUT) else []) if f not in files]
        if stale or extra:
            sys.exit(f"docs/index/ is stale ({', '.join(stale + extra)}): run python3 -I tools/make_index.py")
        print(f"docs/index/ is current ({total} declarations)")
        return
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f not in files:
            os.remove(os.path.join(OUT, f))
    for f, t in files.items():
        with open(os.path.join(OUT, f), "w") as h:
            h.write(t)
    print(f"wrote docs/index/: {len(files)} files, {total} declarations")


if __name__ == "__main__":
    main()
