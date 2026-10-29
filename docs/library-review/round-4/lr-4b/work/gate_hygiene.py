"""lr-4b, library review round 4 (L20): which texts tools/tvr.py's HYGIENE list flags, read the way
`hygiene_problems` reads a kernel's Lean file (the whole text, re.M). Read-only.
Run from the TVA root:  python3 -I docs/library-review/round-4/lr-4b/work/gate_hygiene.py
"""
import importlib.util, os, re
spec = importlib.util.spec_from_file_location("tvr", os.path.join(os.getcwd(), "tools", "tvr.py"))
tvr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tvr)
PROBES = [
    ("local notation", 'local notation "tl.add" => TVA.tl.addc', True),
    ("notation3", 'notation3 "tl.add" => TVA.tl.addc', True),
    ("macro_rules", "macro_rules | `(tl.add) => `(TVA.tl.addc)", True),
    ("scoped instance", "scoped instance : HAdd F32 F32 F32 := ⟨FloatV.sub⟩", True),
    ("attribute [instance]", "attribute [instance] foo", True),
    ("attribute [instance 2000]", "attribute [instance 2000] foo", True),
    ("@[instance] def", "@[instance] def foo : HAdd Nat Nat Nat := ⟨Nat.sub⟩", True),
    ("attribute [-instance]", "attribute [-instance] instHDiv", True),
    ("attribute [- instance] spaced", "attribute [ - instHDiv]", True),
    ("export", "export TVA.tl (addc)", True),
    ("open renaming", "open TVA.tl renaming addc → add", True),
    ("set_option hygiene", "set_option hygiene false in", True),
    ("set_option on two lines", "set_option\n  hygiene false", True),
    ("set_option pp", "set_option pp.notation false", False),
    ("instance (priority := high) on Nat", "instance (priority := high) : HDiv Nat Nat Nat := ⟨fun a b => a * b⟩", None),
    ("plain instance on Nat", "instance : HDiv Nat Nat Nat := ⟨fun a b => a * b⟩", None),
    ("open on two lines", "open TVA\n  Block", None),
    ("open ... in", "open TVA.tl in\ndef add := 1", None),
    ("def Launch.mk", "def Launch.mk : Nat := 1", None),
    ("class_abbrev", "class abbrev Foo := Bar", None),
    ("deriving instance", "deriving instance Repr for Inst", None),
    ("variable binder tl", "variable (tl : Ops)", None),
    ("@[reducible] def", "@[reducible] def tl_add := 1", None),
    ("attribute [simp]", "attribute [simp] foo", None),
    ("@[coe] def", "@[coe] def toF : I32 → F32 := sorry", True),
]
for what, text, expect in PROBES:
    hits = [w for w, pat in tvr.HYGIENE if re.search(pat, text, re.M)]
    got = bool(hits)
    tag = "" if expect is None else ("as expected" if got == expect else "UNEXPECTED")
    print(f"{what:38} {'flagged' if got else 'passes '} {tag} {hits[:2]}")
