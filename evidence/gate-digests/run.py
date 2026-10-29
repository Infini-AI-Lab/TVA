"""The gate on a copy of the tree: what re-opens a kernel item, what the map check checks, and the
hygiene, replay and namespace checks, end to end with `tvr check`.

Round 6 (library issue L36): digests hold the whole library files an item's cone reaches (the Lean
modules, SEMANTICS.md and exclusions.md, the rule pages), TRANSCRIPTION.md is guidance in no digest,
and the map check checks coverage and model text only. The cases of rounds 3-5 (L12, L13, L20, L27,
L32) are rerun with the round-6 tool; some now give a different result on purpose (B, C, E, K2).

The script copies this TVA tree to a temporary folder (the Lean build is copied and Mathlib's packages
are linked; nothing else is shared), brings `kernels/vec_add_f32` up to date with the library in the
copy only (`Launch.valid_grid1`, disjoint read and write footprints; its map is left as it is, with no
`rules`), and runs `python3 -I tools/tvr.py check vec_add_f32` there. It then changes one thing at a
time in the copy and runs the gate again. The real tree is never written.

Usage: python3 -I evidence/gate-digests/run.py > evidence/gate-digests/out.txt
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TVA = os.path.dirname(os.path.dirname(HERE))
ITEMS = ("map", "model", "spec", "conditions", "assumption:faithful")


def copy_tree(dst):
    def ignore(d, names):
        return [n for n in names if n in (".lake", ".venv", "__pycache__", ".git")]
    shutil.copytree(TVA, dst, ignore=ignore, symlinks=True)
    os.makedirs(os.path.join(dst, "lean", ".lake"))
    os.symlink(os.path.join(TVA, "lean", ".lake", "packages"), os.path.join(dst, "lean", ".lake", "packages"))
    shutil.copytree(os.path.join(TVA, "lean", ".lake", "build"), os.path.join(dst, "lean", ".lake", "build"))


def edit(path, old, new):
    s = open(path).read()
    assert old in s, (path, old[:60])
    open(path, "w").write(s.replace(old, new, 1))


# The kernel's own update, made in the copy only: `Launch.valid_grid1` (L17 added a conjunct) and
# disjoint read and write footprints (L15: the in-place case, an output that is an input, is a
# conflict within the program). Its map is not touched: since round 6 it needs no `rules`.
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


def gate(root):
    subprocess.run([sys.executable, "-I", "tools/tvr.py", "check", "vec_add_f32"], cwd=root,
                   capture_output=True, text=True)
    return json.load(open(os.path.join(root, "kernels", "vec_add_f32", "gate.json")))


def show(name, g, base=None):
    print(f"## {name}")
    print(f"gate ok: {g['ok']}")
    for c in g["checks"]:
        if not c["ok"] or c["name"] in ("hygiene", "build", "replay", "namespace", "map"):
            print(f"  {c['name']}: {'pass' if c['ok'] else 'FAIL'}: {c['detail'][:170]}")
    if base is not None:
        for item in ITEMS:
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
        show("A: the kernel, updated in the copy (its map names no rules)", base)
        snap = json.load(open(os.path.join(root, "kernels", "vec_add_f32", "review", "snapshots",
                                           base["digest"][:16] + ".json")))["items"]
        for item in ITEMS:
            files = sorted(n[5:] for n in snap.get(item, {}).get("cone", {}) if n.startswith("file:"))
            print(f"  library files in the {item} digest: {', '.join(files) or 'none'}")
        print()
        p = lambda *x: os.path.join(root, *x)
        trans, excl, sem = p("docs", "TRANSCRIPTION.md"), p("docs", "exclusions.md"), p("docs", "SEMANTICS.md")
        ops, softmax = p("lean", "TVA", "Lang", "Ops.lean"), p("lean", "TVA", "Lemmas", "Softmax.lean")
        claim, launch = p("lean", "TVA", "Claim.lean"), p("lean", "TVA", "Sem", "Launch.lean")
        cf, num = p("docs", "rules", "compiler-faithful.md"), p("docs", "rules", "num.md")
        lean = p("kernels", "vec_add_f32", "lean", "VecAdd.lean")
        mp = p("kernels", "vec_add_f32", "model-map.json")
        tvr = p("tools", "tvr.py")
        saved = {x: open(x).read() for x in (trans, excl, sem, ops, softmax, claim, launch, cf, num, lean, mp, tvr)}

        def restore():
            for x, s in saved.items():
                open(x, "w").write(s)

        # TRANSCRIPTION.md is guidance, in no digest (L36)
        edit(trans, "- **An IEEE `+` that a product feeds**", "- **An IEEE `+` that a product feeds** (changed)")
        show("B: a pitfall of TRANSCRIPTION.md changes (guidance)", gate(root), base)
        restore()
        open(trans, "w").write("# Transcription guidance\n\n(replaced)\n")
        show("C: TRANSCRIPTION.md replaced by one line", gate(root), base)
        restore()
        # the fact pages
        edit(excl, "These Triton features are hard", "These (changed) Triton features are hard")
        show("D: a sentence of exclusions.md (a fact page) changes", gate(root), base)
        restore()
        edit(sem, "Triton converts bf16\n  operands to fp32 first, and the model writes that conversion.",
             "bf16 operands stay\n  bf16.")
        show("J: a sentence of SEMANTICS.md (a fact page) changes", gate(root), base)
        restore()
        # the map check: coverage and model text only (L36)
        m = json.loads(saved[mp])
        m["entries"][1]["rules"] = ["no-such-rule"]
        json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
        show("E: a map entry names `rules` that nothing defines (free labels, unchecked)", gate(root), base)
        restore()
        m = json.loads(saved[mp])
        m["entries"][0]["drop"] = "the decorator, covered by the model's definition"
        json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
        show("E2: a drop reason in free text", gate(root), base)
        restore()
        m = json.loads(saved[mp])
        del m["entries"][0]["drop"]
        json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
        show("E3: an entry with neither model text nor a drop reason", gate(root), base)
        restore()
        m = json.loads(saved[mp])
        m["entries"] = [e for e in m["entries"] if e["src"] != [52]]
        json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
        show("E4: the entry of source line 52 removed", gate(root), base)
        restore()
        m = json.loads(saved[mp])
        m["entries"].append({"model": "let pid := tl.program_id ctx 0"})
        json.dump(m, open(mp, "w"), indent=2, ensure_ascii=False)
        show("E5: an entry without `src`", gate(root), base)
        restore()
        # meta code (L13)
        open(lean, "a").write(ATTACK)
        show("F: the kernel's Lean adds a theorem through run_cmd and addDeclCore (doCheck := false)", gate(root), base)
        edit(tvr, "    bad = hygiene_problems(k) + library_hygiene_problems()\n", "    bad = []\n")
        show("G: the same, with the copy's tool not running the hygiene list", gate(root), base)
        restore()
        show("H: everything restored", gate(root), base)
        # whole library files (L36, subsuming L27 and L32)
        edit(ops, "/-- `x + y` on floats, IEEE (round to nearest even).",
             "/-- `x + y` on floats, IEEE (round to nearest even; changed).")
        show("K: the docstring of TVA.tl.add, which the model uses, changes", gate(root), base)
        restore()
        edit(ops, "/-- `x + y` on floats, IEEE (round to nearest even).",
             "/--   `x + y`   on floats,\n  IEEE (round to nearest even).")
        show("K2: the same docstring, whitespace only (the file is hashed whole)", gate(root), base)
        restore()
        open(ops, "a").write("\n-- a comment at the end of the file\n")
        show("K3: a comment at the end of lean/TVA/Lang/Ops.lean", gate(root), base)
        restore()
        open(softmax, "a").write("\n-- a comment at the end of the file\n")
        show("K4: a comment at the end of lean/TVA/Lemmas/Softmax.lean, a module no item reaches", gate(root), base)
        restore()
        edit(lean, "  · -- each program is safe\n", "  · -- each program is safe (a comment of the kernel's proof)\n")
        show("K5: a comment in the kernel's proof", gate(root), base)
        restore()
        edit(claim, "* `init` is the global memory when the launch starts, and the outcome is the launch's own",
             "* (changed) `init` is the global memory when the launch starts, and the outcome is the launch's own")
        show("Y: the module docstring of TVA.Claim changes", gate(root), base)
        restore()
        edit(launch, "/-- The shapes of the source's block values on this instance",
             "/-- (changed) The shapes of the source's block values on this instance")
        show("Z: the docstring of the field TVA.Launch.blocks changes (no term projects it)", gate(root), base)
        restore()
        edit(cf, "## What it says", "## What it says (changed)")
        show("AA: the rule page docs/rules/compiler-faithful.md changes", gate(root), base)
        restore()
        edit(num, "## Facts the library states", "## Facts the library states (changed)")
        show("AB: the rule page docs/rules/num.md changes (the kernel cites no TVA.Rules.Num fact)", gate(root), base)
        restore()
        # names that mean another operation (L20): the backstops kept in round 6
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
        ops_struct = ("structure Ops where\n  add : {s : Shape} → {f : TVA.Math.Float.Format} → Tensor s (FloatV f) → "
                      "Tensor s (FloatV f) →\n    Prog (Tensor s (FloatV f))\n\n")
        edit(lean, end, "def Launch.mk (grid : Fin 3 → Nat) (fusion debug : Bool) (numCtas : Nat) (pdl : Bool)\n"
             "    (blocks : List Shape) : TVA.Launch :=\n  { grid := grid, fusion := fusion, debug := !debug, "
             "numCtas := numCtas, pdl := pdl, blocks := blocks }\n\n" + end)
        show("Q: the kernel declares def Launch.mk (model text Launch.mk … would set debug := true)", gate(root), base)
        restore()
        edit(lean, end, "instance instHDivNatK : HDiv Nat Nat Nat := ⟨fun a b => a / b + 1⟩\n\n" + end)
        show("R: the kernel declares an instance on core types (/ on Nat, as in the grid formula)", gate(root), base)
        restore()
        edit(lean, end, "def tℓ.add {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) :\n"
             "    Prog (Tensor s (FloatV f)) := TVA.tl.addc x y\n\n" + end)
        show("S: the kernel declares tℓ.add (U+2113), a look-alike of tl.add", gate(root), base)
        restore()
        edit(lean, "/-! ## The model -/", "/-! ## The model -/\n\n" + ops_struct +
             "/-- the operations -/\ndef ops : Ops := ⟨fun x y => TVA.tl.add x y⟩\n")
        src = open(lean).read()
        src = src.replace("    let output ← tl.add x y", "    let tl : Ops := ops\n    let output ← tl.add x y", 1)
        src = src.replace("    tl.store (output_ptr + offsets) output (some mask)",
                          "    TVA.tl.store (output_ptr + offsets) output (some mask)", 1)
        open(lean, "w").write(src)
        show("T: a local binder tl : Ops before the model line let output ← tl.add x y", gate(root), base)
        restore()
        edit(lean, end, ops_struct + "def lookBinder (x y : Tensor [4] F32) (tℓ : Ops := ⟨TVA.tl.addc⟩) : "
             "Prog (Tensor [4] F32) :=\n  tℓ.add x y\n\n" + end)
        show("U: a binder tℓ (a look-alike of tl) with a default value", gate(root), base)
        restore()
        edit(lean, "open Block\n", "open TVA\n  Block\n")
        edit(lean, end, "theorem lane_tile : True := trivial\n\n" + end)
        show("V: open TVA with Block on the next line, and theorem lane_tile", gate(root), base)
        restore()
        edit(lean, end, "def min (a b : Nat) : Nat := a + b\n\n" + end)
        show("W: the kernel declares min, which at the root is the alias of Min.min", gate(root), base)
        restore()
        edit(lean, end, "set_option pp.notation false in\ntheorem ppProbe : True := trivial\n\n" + end)
        show("X: control: set_option pp.notation (a printing option) passes hygiene", gate(root), base)
        restore()
        show("AD: everything restored", gate(root), base)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
