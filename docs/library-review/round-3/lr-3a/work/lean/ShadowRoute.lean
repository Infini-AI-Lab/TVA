import TVA

/-! lr-3a, round 3 (gate `namespace` check): a declaration `tl.add` in the kernel's own namespace
(root `Kernels`, not `TVA`) makes the model text `tl.add x y` mean another operation. No meta code,
nothing under `TVA`. -/

open TVA

namespace Kernels.ShadowRoute

def tl.add {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  TVA.tl.addc x y

/-- What a model line `let z ← tl.add x y` means in this namespace. -/
def line {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  tl.add x y

theorem line_is_addc {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) :
    line x y = TVA.tl.addc x y := rfl

end Kernels.ShadowRoute
