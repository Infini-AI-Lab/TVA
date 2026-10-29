import TVA

/-!
lr-4a, L20: model text `tl.add x y` that elaborates to something other than `TVA.tl.add`, with no
constant named `tl`, no notation, no instance, no `open … renaming` and no `export`. The name
`tl` is a local binder (a `let` in the body) of a kernel structure whose field `add` is `addc`.
Lean reads `tl.add x y` as the field access `Ops.add tl x y`.

Check: `cd lean && ~/.elan/bin/lake env lean ../docs/library-review/round-4/lr-4a/work/lean/LocalTl.lean`
-/

namespace Kernels.Probe

open TVA

structure Ops where
  add : ∀ {s : Shape} {f : TVA.Math.Float.Format}, Tensor s (FloatV f) → Tensor s (FloatV f) →
    Prog (Tensor s (FloatV f))

def ops : Ops := ⟨fun x y => TVA.tl.addc x y⟩

def body (x y : Tensor [4] (FloatV .binary32)) : Prog (Tensor [4] (FloatV .binary32)) := do
  let tl : Ops := ops
  let output ← tl.add x y
  return output

/-- The model line `let output ← tl.add x y` means `addc`, not the IEEE `tl.add`. -/
example (x y : Tensor [4] (FloatV .binary32)) : body x y = (do let o ← TVA.tl.addc x y; return o) := rfl

end Kernels.Probe
