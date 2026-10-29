import TVA.Lang.Promote

/-! lr-20a, round 20: `FloatV.litF8` (through `Operand.constIn?`) on probes.py's P1 values.
Check with `cd lean && lake env lean ../docs/library-review/round-20/lr-20a/work/Fp8Probe.lean`;
output in Fp8Probe.out. Each line: the pattern `constIn?` gives, or `none` (rejected on the host). -/

open TVA

def sh (o : Option Nat) : String := match o with | some n => toString n | none => "none"

/-- `1 + 2^-30` (the double `0x3ff0000000400000`), a double whose `float32` constant is `1.0`. -/
def onePlus : Operand := .float (FloatV.ofPat 4607182418804211712)
/-- `-1e-50` (the double `0xb58dee7a4ad4b81f`), whose `float32` constant is `-0.0`. -/
def tinyNeg : Operand := .float (FloatV.ofPat 13082374701590034463)
/-- `1e39`, whose `float32` constant is `+inf`. -/
def big : Operand := .float (FloatV.ofRat (10 ^ 39))

#eval IO.println s!"1+2^-30 e5m2 {sh ((onePlus.constIn? .f8e5).map (·.pat))}"
#eval IO.println s!"1+2^-30 e4m3 {sh ((onePlus.constIn? .f8e4nv).map (·.pat))}"
#eval IO.println s!"-1e-50 e5m2 {sh ((tinyNeg.constIn? .f8e5).map (·.pat))}"
#eval IO.println s!"-1e-50 e4m3 {sh ((tinyNeg.constIn? .f8e4nv).map (·.pat))}"
#eval IO.println s!"1e39 e5m2 {sh ((big.constIn? .f8e5).map (·.pat))}"
#eval IO.println s!"3 e5m2 {sh (((Operand.int 3).constIn? .f8e5).map (·.pat))}"
#eval IO.println s!"448 e4m3 {sh (((Operand.int 448).constIn? .f8e4nv).map (·.pat))}"
#eval IO.println s!"True e5m2 {sh (((Operand.bool true).constIn? .f8e5).map (·.pat))}"
