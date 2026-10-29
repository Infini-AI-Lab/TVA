"""lr-4a, L20: case Q of gate_rerun.py, made to build. A local binder `tl : Ops` (a kernel structure
with a field `add`) comes before the model line `let output ← tl.add x y` in the model only (the
proof's copies of the line are left alone); the next model line is written `TVA.tl.store …`, which
still contains the map's model text `tl.store …`. To keep the kernel's proof building, the field
is `TVA.tl.add` itself here (lean/LocalTl.lean shows, by `rfl`, that the same binder with
`add := addc` makes the line mean `addc`). Does the gate (`hygiene`, `namespace`, `map`) stop a
binder named `tl`?

Usage (from the TVA root): python3 -I docs/library-review/round-4/lr-4a/work/gate_q.py
"""
import importlib.util
import os
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gate_rerun", os.path.join(HERE, "gate_rerun.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def main():
    tmp = tempfile.mkdtemp(prefix="tva-gate-")
    root = os.path.join(tmp, "tva")
    try:
        g.copy_tree(root)
        g.update_kernel(root)
        base = g.gate(root)
        g.show("A: the kernel, updated for round 3 in the copy", base)
        lean = os.path.join(root, "kernels", "vec_add_f32", "lean", "VecAdd.lean")
        g.edit(lean, "/-! ## The model -/",
               "/-! ## The model -/\n\n/-- operations -/\nstructure Ops where\n"
               "  add : ∀ {s : Shape} {f : TVA.Math.Float.Format}, Tensor s (FloatV f) → Tensor s (FloatV f) →\n"
               "    Prog (Tensor s (FloatV f))\n\n/-- the operations -/\ndef ops : Ops := ⟨fun x y => TVA.tl.add x y⟩\n")
        src = open(lean).read()
        src = src.replace("    let output ← tl.add x y", "    let tl : Ops := ops\n    let output ← tl.add x y", 1)
        src = src.replace("    tl.store (output_ptr + offsets) output (some mask)",
                          "    TVA.tl.store (output_ptr + offsets) output (some mask)", 1)
        open(lean, "w").write(src)
        r = g.gate(root)
        g.show("Q2 (lr-4a): `let tl : Ops := ops` before the model line `let output ← tl.add x y`", r, base)
        print("model line in the Lean:", [l.strip() for l in open(lean) if "tl.add x y" in l])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
