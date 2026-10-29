"""lr-7a, library review round 7: probes of tools/tvr.py on scratch trees (nothing in the tree is written).

  S1  `gate_staleness` (L42) on a scratch kernel whose passing gate has a round-7 snapshot: the kernel's
      source file changes (content and mtime), then its Lean file
  O1  `lake_violations` (L40) on `leanOptions` spellings: dotted, nested tables, quoted keys, values
  P1  `rule_page` and `fact_page_problems` (L43) on a scratch lean/TVA/Rules tree
  F1  `shipped_problems` (L41) on paths inside folders the text scan skips
Usage: python3 -I work/tool_probes.py SCRATCH   (an empty folder outside TVA; never printed)
"""
import importlib.util
import json
import os
import shutil
import sys

sys.dont_write_bytecode = True  # import the tree's modules without writing caches into it
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, *[os.pardir] * 5))
scratch = os.path.abspath(sys.argv[1])
assert not os.listdir(scratch), "SCRATCH must be empty"


def fresh():
    spec = importlib.util.spec_from_file_location("tvr", os.path.join(TVA, "tools", "tvr.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)


# S1 -----------------------------------------------------------------------------------------------
tvr = fresh()
kroot = os.path.join(scratch, "s1", "kernels")
tvr.KROOT = kroot
k = "probe"
put(tvr.kpath(k, "kernel.json"), json.dumps({"lean": {"module": "Probe", "namespace": "Kernels.Probe"},
                                             "source": {"file": "source/k.py", "sha256": "", "ranges": [[1, 2]]}}))
put(tvr.kpath(k, "lean", "Probe.lean"), "-- the kernel's Lean\n")
put(tvr.kpath(k, "model-map.json"), '{"entries": []}\n')
put(tvr.kpath(k, "source", "k.py"), "def k(c):\n    tl.atomic_add(c, 1)\n")
put(tvr.kpath(k, "README.md"), "## Specification\n")
time.sleep(2)
digest = "ab" * 32
files = {"file:lean/TVA/Lang/Ops.lean": tvr._file_digest("lean/TVA/Lang/Ops.lean"), "packages": tvr.packages_digest()}
put(tvr.kpath(k, "review", "snapshots", digest[:16] + ".json"),
    json.dumps({"items": {"model": {"digest": "x", "cone": {"Kernels.Probe.model": "1-2", **files}}}}))
gate = {"ok": True, "time": tvr.now(), "digest": digest, "items": {"model": "x"}}
print("# S1 (L42): gate_staleness on a passing round-7 gate")
print(f"just after the gate: {tvr.gate_staleness(k, gate) or 'not stale'}")
time.sleep(2)
put(tvr.kpath(k, "source", "k.py"), "def k(c):\n    tl.atomic_add(c, 2)\n")
put(tvr.kpath(k, "README.md"), "## Specification\nchanged\n")
print(f"the source file and the write-up changed after the gate: {tvr.gate_staleness(k, gate) or 'not stale'}")
put(tvr.kpath(k, "lean", "Probe.lean"), "-- the kernel's Lean, changed\n")
print(f"then the kernel's Lean file changed: {tvr.gate_staleness(k, gate) or 'not stale'}")
old = os.path.getmtime(tvr.kpath(k, "kernel.json"))
os.utime(tvr.kpath(k, "lean", "Probe.lean"), (old, old))
print(f"the same Lean change with its old mtime put back: {tvr.gate_staleness(k, gate) or 'not stale'}")
print()

# O1 -----------------------------------------------------------------------------------------------
print("# O1 (L40): leanOptions in lean/lakefile.toml")
tvr = fresh()
root = os.path.join(scratch, "o1")
os.makedirs(os.path.join(root, "lean"))
shutil.copy(os.path.join(TVA, "lean", "lake-manifest.json"), os.path.join(root, "lean"))
base = open(os.path.join(TVA, "lean", "lakefile.toml")).read().split("# One library per kernel folder")[0]
base = base.replace(', "vec_add_f32"', "")
tvr.ROOT = root
for opts in ("autoImplicit = false", "debug.skipKernelTC = true", '"debug.skipKernelTC" = true',
             "[leanOptions.debug]\nskipKernelTC = true", "maxHeartbeats = 0", "pp.all = true",
             "linter.all = \"on\"", "maxRecDepth = 1.5", "backward.synthInstance.canonInstances = false"):
    text = base.replace("[leanOptions]\nautoImplicit = false", "[leanOptions]\n" + opts if not opts.startswith("[")
                        else "[leanOptions]\nautoImplicit = false\n" + opts)
    put(os.path.join(root, "lean", "lakefile.toml"), text)
    bad = [b for b in tvr.lake_violations() if "leanOptions" in b]
    print(f"{opts.replace(chr(10), ' / ')!r}: {'refused' if bad else 'accepted'}")
print()

# P1 -----------------------------------------------------------------------------------------------
print("# P1 (L43): rule pages for modules of lean/TVA/Rules")
tvr = fresh()
root = os.path.join(scratch, "p1")
for f in ("Num.lean", "Compiler.lean", "Num/Exp2.lean", "Num/Deep/Log2.lean"):
    put(os.path.join(root, "lean", "TVA", "Rules", f), "-- module\n")
for f in ("num.md", "compiler-faithful.md"):
    put(os.path.join(root, "docs", "rules", f), "page\n")
for f in ("SEMANTICS.md", "exclusions.md"):
    put(os.path.join(root, "docs", f), "page\n")
tvr.ROOT = root
tvr.rel = lambda p, base=root: os.path.relpath(p, base)  # `rel`'s default base is bound to the tree at import
print(f"modules: {sorted(tvr.rules_modules())}")
for m in sorted(tvr.rules_modules()) + ["TVA.Rules", "TVA.Lemmas.Num"]:
    print(f"  rule_page({m}) = {tvr.rule_page(m)}")
print(f"fact_page_problems: {tvr.fact_page_problems() or 'none'}")
put(os.path.join(root, "lean", "TVA", "Rules", "Extra", "Y.lean"), "-- module\n")
put(os.path.join(root, "docs", "rules", "extra.md"), "page\n")
print(f"with Rules/Extra/Y.lean and an unclaimed docs/rules/extra.md: {tvr.fact_page_problems()}")
print()

# F1 -----------------------------------------------------------------------------------------------
print("# F1 (L41): shipped_problems")
for f in ("docs/.lake/note.md", "kernels/k/lean/.lake/x.md", "a/__pycache__/m.cpython-312.pyc", "x/.venv/y", "sub/repo/",
          "docs/lake/note.md", "docs/.lakefile"):
    print(f"{f}: {tvr.shipped_problems([f]) or 'shipped, scanned'}")
