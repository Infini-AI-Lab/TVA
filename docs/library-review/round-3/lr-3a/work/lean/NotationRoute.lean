import TVA

/-! lr-3a, round 3 (L13, hygiene): can a kernel file make model text mean another operation without
meta code that the hygiene list names? `local notation` is not on the list. Here the text
`tl.add x y` elaborates to the fusable `TVA.tl.addc x y`, while the reviewer reads IEEE `tl.add`. -/

open TVA

namespace Kernels.NotationRoute

local notation "tl.add" => TVA.tl.addc

/-- What a model line `let z ← tl.add x y` would mean in this file. -/
def line {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) : Prog (Tensor s (FloatV f)) :=
  tl.add x y

theorem line_is_addc {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) :
    line x y = TVA.tl.addc x y := rfl

#print line

end Kernels.NotationRoute
