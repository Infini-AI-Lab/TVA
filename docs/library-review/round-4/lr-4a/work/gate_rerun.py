"""The gate on a copy of the tree: rule digests (library issue L12) and the hygiene and replay
checks (L13), end to end with `tvr check`; round 4 adds the general text and library docstrings
(L12, L27) and names that mean another operation (L20).

The script copies this TVA tree to a temporary folder (the Lean build and packages are shared or
copied, nothing else is), brings `kernels/vec_add_f32` up to date with the round-3 library in
the copy only, and runs `python3 -I tools/tvr.py check vec_add_f32` there. It then changes one
thing at a time in the copy and runs the gate again. The real tree is never written.

lr-4a's copy of evidence/gate-digests/run.py: cases A and I-P rerun, and new cases Q (a local
binder named `tl` whose field `add` is `addc`) and S (an unindented line continuing a rule).
Usage (from the TVA root): python3 -I docs/library-review/round-4/lr-4a/work/gate_rerun.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))  # lr-4a: run from the work folder


def copy_tree(dst):
    def ignore(d, names):
        return [n for n in names if n in (".lake", ".venv", "__pycache__")]
    shutil.copytree(TVA, dst, ignore=ignore, symlinks=True)
    os.makedirs(os.path.join(dst, "lean", ".lake"))
    os.symlink(os.path.join(TVA, "lean", ".lake", "packages"), os.path.join(dst, "lean", ".lake", "packages"))
    shutil.copytree(os.path.join(TVA, "lean", ".lake", "build"), os.path.join(dst, "lean", ".lake", "build"))


def edit(path, old, new):
    s = open(path).read()
    assert old in s, (path, old[:60])
    open(path, "w").write(s.replace(old, new))


# The kernel's own update for round 3, made in the copy only: `Launch.valid_grid1` (L17 added a
# conjunct), the rules each map entry applies (L12), and disjoint read and write footprints
# (L15: the in-place case, an output that is an input, is a conflict within the program).
def update_kernel(root):
    k = os.path.join(root, "kernels", "vec_add_f32")
    lean = os.path.join(k, "lean", "VecAdd.lean")
    edit(lean, "  refine ⟨?_, by simp [launch, Launch.grid1], by simp [launch, Launch.grid1], rfl, ?_⟩",
         "  refine Launch.valid_grid1 ?_ ?_")
    edit(lean, "    simp only [launch, Launch.grid1, List.mem_singleton] at hs", "    simp only [List.mem_singleton] at hs")
    edit(lean, "  output_x : Disjoint d.output.bytes d.x.bytes ∨ d.output.base = d.x.base",
         "  output_x : Disjoint d.output.bytes d.x.bytes")
    edit(lean, "  output_y : Disjoint d.output.bytes d.y.bytes ∨ d.output.base = d.y.base",
         "  output_y : Disjoint d.output.bytes d.y.bytes")
    edit(lean, "hd.x_len hd.output_x hij'", "hd.x_len (Or.inl hd.output_x) hij'")
    edit(lean, "hd.y_len hd.output_y hij'", "hd.y_len (Or.inl hd.output_y) hij'")
    edit(lean, "(fun d init i m => Q d init i m) ?_ ?_ ?_\n", "(fun d init i m => Q d init i m) ?_ ?_ ?_ ?_\n")
    edit(lean, "    · exact d.output.tile_disjoint hij'\n  · -- the programs' results give `Post`",
         "    · exact d.output.tile_disjoint hij'\n"
         "  · -- no program reads what it writes\n"
         "    intro d hd init _ i\n"
         "    simp only [R, W]\n"
         "    exact Disjoint.union_left (Buf.tile_disjoint_of d.x d.output hd.output_x.symm hd.x_len hd.output_len)\n"
         "      (Buf.tile_disjoint_of d.y d.output hd.output_y.symm hd.y_len hd.output_len)\n"
         "  · -- the programs' results give `Post`")
    mp = os.path.join(k, "model-map.json")
    m = json.load(open(mp))
    rules = {"30-36": ["call", "index", "pointer"], "39": ["call", "index"], "44": ["index", "implicit"],
             "45": ["index", "broadcast", "call"], "47": ["broadcast", "call"], "50": ["call", "mask", "pointer"],
             "51": ["call", "mask", "pointer"], "52": ["call", "fusion", "float-prog"],
             "54": ["call", "mask", "pointer"]}
    for e in m["entries"]:
        if "model" in e:
            e["rules"] = rules[str(e["src"][0])]
    json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)


def gate(root):
    r = subprocess.run([sys.executable, "-I", "tools/tvr.py", "check", "vec_add_f32"], cwd=root,
                       capture_output=True, text=True)
    g = json.load(open(os.path.join(root, "kernels", "vec_add_f32", "gate.json")))
    return g


def show(name, g, base=None):
    print(f"## {name}")
    print(f"gate ok: {g['ok']}")
    for c in g["checks"]:
        if not c["ok"] or c["name"] in ("hygiene", "build", "replay", "map"):
            print(f"  {c['name']}: {'pass' if c['ok'] else 'FAIL'}: {c['detail'][:160]}")
    if base is not None:
        for item in ("map", "model", "spec", "conditions", "assumption:faithful"):
            same = g["items"].get(item) == base["items"].get(item)
            print(f"  item {item}: {'unchanged' if same else 'CHANGED'}")
    print()


ATTACK = """
open Lean Elab Command in
run_cmd do
  let decl := Declaration.thmDecl
    { name := `Kernels.VecAdd.extraFalse, levelParams := [], type := mkConst ``False,
      value := mkConst ``True.intro }
  match (← getEnv).addDeclCore 0 0 decl none (doCheck := false) with
  | .ok env => setEnv env
  | .error _ => throwError "addDeclCore failed"
"""


def main():
    tmp = tempfile.mkdtemp(prefix="tva-gate-")
    root = os.path.join(tmp, "tva")
    try:
        copy_tree(root)
        update_kernel(root)
        base = gate(root)
        show("A: the kernel, updated for round 3 in the copy", base)
        trans = os.path.join(root, "docs", "TRANSCRIPTION.md")
        excl = os.path.join(root, "docs", "exclusions.md")
        sem = os.path.join(root, "docs", "SEMANTICS.md")
        ops = os.path.join(root, "lean", "TVA", "Lang", "Ops.lean")
        lean = os.path.join(root, "kernels", "vec_add_f32", "lean", "VecAdd.lean")
        tvr = os.path.join(root, "tools", "tvr.py")
        saved = {p: open(p).read() for p in (trans, excl, sem, ops, lean, tvr)}

        def restore():
            for p, s in saved.items():
                open(p, "w").write(s)

        # round 4 (L12): the closing paragraph of section 3 is general text, not part of [atomic]
        edit(trans, "No construct of [exclusions.md](exclusions.md) is used.",
             "No construct of [exclusions.md](exclusions.md) is used (changed).")
        show("I: the closing paragraph of TRANSCRIPTION.md section 3 (general text) changes; the map does not "
             "name [atomic]", gate(root), base)
        restore()
        # round 4 (L27): SEMANTICS.md is general text
        edit(sem, "Triton converts bf16\n  operands to fp32 first, and the model writes that conversion.",
             "bf16 operands stay\n  bf16.")
        show("J: a sentence of SEMANTICS.md changes (what tl.maximum does with bf16)", gate(root), base)
        restore()
        # round 4 (L27): the docstring of a library definition the model reaches
        edit(ops, "/-- `x + y` on floats, IEEE (round to nearest even).",
             "/-- `x + y` on floats, IEEE (round to nearest even; changed).")
        show("K: the docstring of TVA.tl.add, which the model uses, changes", gate(root), base)
        restore()
        edit(ops, "/-- `x + y` on floats, IEEE (round to nearest even).",
             "/--   `x + y`   on floats,\n  IEEE (round to nearest even).")
        show("K2: the same docstring, whitespace only", gate(root), base)
        restore()
        # round 4 (L20): model text that means another operation, by four routes
        end = "end Kernels.VecAdd"
        edit(lean, end, "def tl.add {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) :\n"
             "    Prog (Tensor s (FloatV f)) := TVA.tl.addc x y\n\n" + end)
        show("L: the kernel declares tl.add in its own namespace (it means tl.addc)", gate(root), base)
        restore()
        edit(lean, end, "local notation \"tl.add\" => TVA.tl.addc\n\n" + end)
        show("M: the kernel declares a local notation tl.add", gate(root), base)
        restore()
        edit(lean, end, "instance : HAdd F32 F32 F32 := ⟨FloatV.sub⟩\n\n" + end)
        show("N: the kernel declares an instance on a library type (+ on F32 is subtraction)", gate(root), base)
        restore()
        edit(lean, end, "theorem lane_tile : True := trivial\n\n" + end)
        show("O: the kernel declares lane_tile, a name of the library namespace Block that it opens", gate(root), base)
        restore()
        # lr-4a Q: a local binder `tl` of a kernel structure whose field `add` is `addc`
        edit(lean, "/-! ## The model -/",
             "/-! ## The model -/\n\n/-- operations -/\nstructure Ops where\n"
             "  add : ∀ {s : Shape} {f : TVA.Math.Float.Format}, Tensor s (FloatV f) → Tensor s (FloatV f) →\n"
             "    Prog (Tensor s (FloatV f))\n\n/-- the operations -/\ndef ops : Ops := ⟨fun x y => TVA.tl.addc x y⟩\n")
        edit(lean, "    let output ← tl.add x y", "    let tl : Ops := ops\n    let output ← tl.add x y")
        show("Q (lr-4a): a local binder tl : Ops (field add := addc) before the model line `let output ← tl.add x y`",
             gate(root), base)
        restore()
        # lr-4a S: an unindented line that continues a rule's paragraph
        edit(trans, "- **[mask]** ", "- **[mask]** ")
        s0 = open(trans).read()
        i = s0.index("- **[mask]** ")
        j = s0.index("\n", i)
        open(trans, "w").write(s0[:j] + "\nAn unindented continuation line." + s0[j:])
        show("S (lr-4a): an unindented line directly continues the first line of [mask]", gate(root), base)
        restore()
        show("P: everything restored", gate(root), base)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
