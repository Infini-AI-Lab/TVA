import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `vec_add_f32` run by `execLaunch` on a small instance. Not a test
against the device (phase 4): it checks that `exec` runs a real claim's model end to end, and
that the result is what `Post` says. -/

open TVA TVA.Validation Claims.VecAdd

/-- A platform that relates nothing: `vec_add_f32` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- `n = 10` elements, `BLOCK_SIZE = 4`: three programs, the last one masked. -/
def n : Nat := 10
def d : Inst := mk n 2

def xs (j : Nat) : F32 := FloatV.ofRat (j : Rat)
def ys (j : Nat) : F32 := FloatV.ofRat ((100 * j : Nat) : Rat)

/-- The buffers of `d`, written into an empty memory. -/
def init : CMem :=
  let c := CMem.ofElems ∅ .f32 ((List.range n).map fun j : Nat => ((d.x.ptr j).addr, xs j))
  let c := CMem.ofElems c .f32 ((List.range n).map fun j : Nat => ((d.y.ptr j).addr, ys j))
  CMem.ofElems c .f32 ((List.range n).map fun j => ((d.output.ptr j).addr, FloatV.ofRat 7))

def run : LRes (launch d).size × List Record :=
  execLaunch ((launch d).env P0) Oracle.empty fill0 1000 (model d) (launch d) init

/-- The expected output (`Post`: `x + y`, exact on these values) against the final memory. -/
def agree (c : CConfig (launch d).size) : Option String :=
  let bad := (List.range n).filter fun j : Nat =>
    ((CMem.toMem c.mem).read .f32 (d.output.ptr j).addr).map (·.toRat) != some ((101 * j : Nat) : Rat)
  if bad.isEmpty then none else some s!"output elements {bad} differ from x + y"

/-- The verdict: certified, since `vec_add_f32` has no platform operation. (Against `x + y`, not
yet against the device: phase 4.) -/
def verdict : Verdict := Verdict.ofLaunch agree run

#eval IO.println (verdict.line "vec_add_f32")
#guard verdict == .certified

/-- The output's elements, as rationals, if the run finished with every answer proved. -/
def outs : Option (List Rat) := match run with
  | (.done c, l) => if l.all (·.proved) then
      (List.range n).mapM fun j => ((CMem.toMem c.mem).read .f32 (d.output.ptr j).addr).map (·.toRat)
    else none
  | _ => none

-- The run answered some requests (the masks' guards, the additions), numbered in order.
#guard run.2.length > 0 && run.2.map (·.occ) == List.range run.2.length

#eval outs
#guard outs == some ((List.range n).map fun j => ((101 * j : Nat) : Rat))
