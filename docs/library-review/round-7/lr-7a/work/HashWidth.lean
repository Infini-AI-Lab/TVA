-- lr-7a: Expr.hash is 32 bits; a structural collision between two different values is found in ~2^15 tries.
import Lean
open Lean

/-- a model-like value: `fun x : Nat => x + n` -/
def e (n : Nat) : Expr :=
  mkLambda `x .default (mkConst ``Nat)
    (mkAppN (mkConst ``HAdd.hAdd [0, 0, 0]) #[mkConst ``Nat, mkConst ``Nat, mkConst ``Nat,
      mkAppN (mkConst ``instHAdd [0]) #[mkConst ``Nat, mkConst ``instAddNat], mkBVar 0,
      mkAppN (mkConst ``OfNat.ofNat [0]) #[mkConst ``Nat, mkRawNatLit n, mkApp (mkConst ``instOfNatNat) (mkRawNatLit n)]])

/-- the value hash the claim reader computes (`constJson`'s `vh`) -/
def vh (v : Expr) : UInt64 := mixHash v.hash (hash (toString v))

def main : IO Unit := do
  IO.println s!"Expr.hash of e 1: {(e 1).hash} (< 2^32: {decide ((e 1).hash < 4294967296)})"
  -- a 32-bit structural collision by birthday search
  let mut seen : Std.HashMap UInt64 Nat := {}
  let mut found := none
  for n in [0:300000] do
    let h := (e n).hash
    if let some m := seen[h]? then
      found := some (m, n); break
    seen := seen.insert h n
  match found with
  | some (m, n) =>
    IO.println s!"Expr.hash collision: e {m} and e {n}, hash {(e m).hash} = {(e n).hash}; values equal: {e m == e n}"
    IO.println s!"  their vh: {vh (e m)} and {vh (e n)}"
  | none => IO.println "no structural collision in 300000"
  let t0 ← IO.monoMsNow
  let mut acc : UInt64 := 0
  for n in [0:20000] do
    acc := acc ^^^ vh (e n)
  let t1 ← IO.monoMsNow
  IO.println s!"vh of 20000 values (interpreted): {t1 - t0} ms ({acc % 7})"

#eval main
