"""Library review round 3 (lr-3b), library issue L12: which edits re-open a kernel's `map` and
`model`. Uses evidence/gate-digests/run.py's own helpers (copy the tree to a temporary folder, bring
`kernels/vec_add_f32` up to date with round 3 in the copy, run the copy's `tvr check`), then edits
one thing at a time in the copy and compares the item digests with the base case. The real tree is
not written.

Cases:
  X1  the last paragraph of TRANSCRIPTION.md section 3 ("No construct of exclusions.md is used...
      no aggregate class"), which the page's introduction calls general text (every model): delete
      its last clause about aggregate classes;
  X2  the same paragraph edited, and the vec_add map made to name [atomic] (a control: then the
      digests do change);
  X3  SEMANTICS.md, the statement [call] defers to for which definition a call is: the "Constants"
      bullet (what a zero literal compiles to);
  X4  SEMANTICS.md: the bf16 conversion that `tl.maximum` needs, which only SEMANTICS.md asks the
      model to write.

Run from the TVA root (TMPDIR in a scratch directory):
  python3 -I docs/library-review/round-3/lr-3b/work/l12_digests.py
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

ROOT = os.getcwd()
spec = importlib.util.spec_from_file_location("gd", os.path.join(ROOT, "evidence", "gate-digests", "run.py"))
gd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gd)


def show(name, g, base):
    print(f"## {name}")
    print(f"gate ok: {g['ok']}")
    for item in ("map", "model", "spec", "conditions"):
        same = g["items"].get(item) == base["items"].get(item)
        print(f"  item {item}: {'unchanged' if same else 'CHANGED'}")
    print()


def main():
    tmp = tempfile.mkdtemp(prefix="lr3b-l12-")
    root = os.path.join(tmp, "tva")
    try:
        gd.copy_tree(root)
        gd.update_kernel(root)
        base = gd.gate(root)
        print(f"## base: the kernel, updated for round 3 in the copy\ngate ok: {base['ok']}\n")
        trans = os.path.join(root, "docs", "TRANSCRIPTION.md")
        sem = os.path.join(root, "docs", "SEMANTICS.md")
        mp = os.path.join(root, "kernels", "vec_add_f32", "model-map.json")
        saved = {p: open(p).read() for p in (trans, sem, mp)}

        def restore():
            for p, s in saved.items():
                open(p, "w").write(s)

        para = ", and no aggregate class (a class made with\nTriton's `_aggregate`)."
        gd.edit(trans, para, ".")
        show("X1: the last paragraph of section 3 (general text, by the page's introduction) loses its "
             "aggregate clause", gd.gate(root), base)
        m = json.loads(saved[mp])
        for e in m["entries"]:
            if "model" in e:
                e["rules"] = e["rules"] + ["atomic"]
                break
        restore()
        json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
        b2 = gd.gate(root)
        gd.edit(trans, para, ".")
        show("X2 (control): the same edit, with the map naming [atomic] (digests against the map naming [atomic])",
             gd.gate(root), b2)
        restore()
        gd.edit(sem, "A literal equal to zero, `-0.0` included, compiles to `+0.0`",
                "A literal equal to zero compiles to `+0.0`, and `-0.0` to `-0.0`")
        show("X3: SEMANTICS.md 'Constants' (which value a literal is) changes", gd.gate(root), base)
        restore()
        gd.edit(sem, "Triton converts bf16\n  operands to fp32 first, and the model writes that conversion.",
                "bf16 operands stay\n  bf16.")
        show("X4: SEMANTICS.md 'tl.maximum' (the bf16 conversion the model writes) changes", gd.gate(root), base)
        restore()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
