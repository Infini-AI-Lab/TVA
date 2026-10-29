"""lr-7a, library review round 7: probes of the gate's functions (tools/tvr.py), called on small inputs.
Nothing is built and nothing in the tree is written.

  R   rerun of evidence/gate-round7 Part U (U1-U3), through that script's own function
  M1  the map check's coverage when `source.ranges` leaves lines out (the tool's `code_lines` and the
      `missing` expression of `check`, copied verbatim)
  M2  claim.md's source column (`src_text.splitlines()`) against the line numbers the gate checks
      (`code_lines`, Python's tokenizer), for a source with a form feed or a line separator in a comment
  M3  more code-line cases (L39): a string statement followed by code after `;`, a bare f-string, a
      string statement continued with a backslash, a docstring-like string as the last statement
  H1  the library hygiene's comment stripper (`_uncommented`) on Lean interpolated strings
  N1  `manifest_problems` on names that are plain but special in Lean (`_root_`, keywords)
  G1  `gate_mismatch` (the release check's comparison) on a certificate no ledger supports
Usage: python3 -I work/gate_probes.py
"""
import importlib.util
import io
import os
import re
import sys

sys.dont_write_bytecode = True  # import the tree's modules without writing caches into it

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tvr = load("tvr", os.path.join(TVA, "tools", "tvr.py"))
sys.path.insert(0, os.path.join(TVA, "tools"))  # evidence/gate-round7 imports tvr by name
ev = load("gate_round7", os.path.join(TVA, "evidence", "gate-round7", "run.py"))
TQ = '"' * 3

print("# R: evidence/gate-round7 Part U, rerun")
ev.part_u()

print("# M1: map coverage when the ranges leave lines out")
src = open(os.path.join(TVA, "kernels", "vec_add_f32", "source", "01-vector-add.py")).read()
code, _ = tvr.code_lines(src)
lines = src.splitlines()
covered = set(range(29, 52)) | {53, 54}          # a map that leaves out line 52 (`output = x + y`)
for ranges in ([[29, 54]], [[29, 51], [53, 54]], []):
    missing = [i for a, b in ranges for i in range(a, b + 1) if i in code and i not in covered]
    print(f"ranges {ranges}: uncovered code lines {missing} -> map {'FAIL' if missing else 'PASS'}")
print(f"  line 52 is code: {52 in code}: {lines[51].strip()!r}")
print()

print("# M2: claim.md's source column against the gate's line numbers")
for name, sep in (("form feed", "\x0c"), ("line separator U+2028", " "), ("next line U+0085", "\x85")):
    s = ("@triton.jit\n"
         "def k(x_ptr, c_ptr):\n"
         f"    # a comment with a{sep}in it\n"
         "    x = tl.load(x_ptr)\n"
         "    tl.atomic_add(c_ptr, 1)\n"
         "    tl.store(x_ptr, x)\n")
    code, why = tvr.code_lines(s)
    shown = s.splitlines()  # what render_claim uses for the map table
    real = io.StringIO(s).readlines()
    print(f"{name}: the source parses: {why is None}; code lines {sorted(code)}")
    for n in (4, 5, 6):
        print(f"  line {n}: Python reads {real[n - 1].strip()!r}; claim.md shows {shown[n - 1].strip()!r}")
print()

print("# M3: more code-line cases (L39)")
cases = {
    "a string statement, then code after `;`": ["def k(c):", f"    {TQ}doc{TQ}; tl.atomic_add(c, 1)"],
    "a bare f-string statement": ["def k(c):", "    f'{tl.atomic_add(c, 1)}'"],
    "a string statement continued by a backslash": ["def k(c):", "    'a' \\", "    'b'", "    tl.atomic_add(c, 1)"],
    "a bytes statement": ["def k(c):", "    b'x'", "    tl.atomic_add(c, 1)"],
}
for name, ls in cases.items():
    code, why = tvr.code_lines("\n".join(ls) + "\n")
    print(f"{name}: " + "; ".join(f"{i}: {'code' if i in code else '----'} {l.strip()!r}" for i, l in enumerate(ls, 1)))
print()

print("# H1: the library hygiene's comment stripper on Lean interpolated strings")
pats = tvr.LIBRARY_HYGIENE
for name, text in (
        ("a char literal of a quote inside an interpolation, then `--` inside the string",
         "def s : String := s!\"{'\"'} -- \" #eval 7\n"),
        ("a string literal inside an interpolation (control)", "def s : String := s!\"{\"a\"} -- x\" #eval 8\n"),
):
    view = tvr._uncommented(text)
    hits = sorted({what for what, pat in pats for _ in re.finditer(pat, view, re.M)})
    print(f"{name}:\n  text {text.strip()!r}\n  stripped {view.strip()!r}\n  flagged: {hits or 'nothing'}")
print()

print("# N1: manifest_problems on plain names that are special in Lean")
man = tvr.manifest("vec_add_f32")
for ns in ("_root_", "end", "Kernels.VecAdd.end", "TVA", "_"):
    m = {**man, "lean": {**man["lean"], "namespace": ns}}
    print(f"namespace {ns!r}: {tvr.manifest_problems('vec_add_f32', m) or 'accepted'}")
print()

print("# G1: the release check's comparison on a certificate no ledger supports")
new = {"ok": True, "digest": "d" * 64, "items": {"model": "a" * 64, "spec": "b" * 64}}
forged = {"kernel": "k", "gate": new["digest"], "items": new["items"], "reviewers": [], "time": "2026-10-28T00:00:00Z"}
print(f"shipped gate.json = recomputed, certificate.json naming it with no reviewer: gate_mismatch -> "
      f"{tvr.gate_mismatch(dict(new), forged, new) or '[] (the gate step passes)'}")
