import TVA

/-! lr-4b, library review round 4, L20: names a kernel can give another meaning that the gate's
`namespace` and `hygiene` checks do not flag (see `gate_probes.py`, `gate_probes.out`). Each
`example` shows what the model text elaborates to inside a kernel namespace with `open TVA`. -/

namespace Kernels.Probe

open TVA

-- R1: a definition named `Launch.mk` (the claim reader exempts the last component `mk`)
def Launch.mk (grid : Fin 3 → Nat) (fusion debug : Bool) (numCtas : Nat) (pdl : Bool)
    (blocks : List Shape) : TVA.Launch :=
  { grid := grid, fusion := fusion, debug := !debug, numCtas := numCtas, pdl := pdl, blocks := blocks }

def launchText : TVA.Launch := Launch.mk ![4, 1, 1] true false 1 false [[1024]]
-- the model text says `debug := false`, the value has `debug = true`
example : launchText.debug = true := rfl

-- R2: an instance of a core class on core types (not flagged: neither a library class nor a library type)
instance instHDivNatK : HDiv Nat Nat Nat := ⟨fun a b => a / b + 1⟩
def gridText (n B : Nat) : Nat := (n + B - 1) / B
-- the grid formula `(n + B - 1) / B` now means one more program
example : gridText 8 4 = 3 := rfl

-- R3: a look-alike of `tl` (`tℓ`, U+2113, a letter-like symbol Lean accepts in identifiers)
def tℓ.add {s : Shape} {f : TVA.Math.Float.Format} (x y : Tensor s (FloatV f)) :
    Prog (Tensor s (FloatV f)) := TVA.tl.addc x y
def lookText (x y : Tensor [4] F32) : Prog (Tensor [4] F32) := tℓ.add x y
example (x y : Tensor [4] F32) : lookText x y = TVA.tl.addc x y := rfl

-- R4: a local binder named `tl` (a parameter with a default value): `tl.add` is a field of it
structure Ops where
  add : {s : Shape} → {f : TVA.Math.Float.Format} → Tensor s (FloatV f) → Tensor s (FloatV f) →
    Prog (Tensor s (FloatV f))
def binderText (x y : Tensor [4] F32) (tl : Ops := ⟨TVA.tl.addc⟩) : Prog (Tensor [4] F32) :=
  tl.add x y
example (x y : Tensor [4] F32) : binderText x y = TVA.tl.addc x y := rfl

-- R5: under a two-line `open`, a kernel name shadows the library name the model writes
section
open TVA
  Block
def offs (p B : Nat) : Tensor [B] I32 := Tensor.full 0
def offsText : Tensor [4] I32 := offs 1 4
example : offsText = Kernels.Probe.offs 1 4 := rfl
end

-- R6: a core name in the kernel's namespace (not a library name)
def min (a b : Nat) : Nat := a + b
example : min 2 3 = 5 := rfl

end Kernels.Probe
