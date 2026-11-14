import Claim
import Exec

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

/-- An oracle with no hints and no platform answers; disabled lanes read zero bits. -/
def O0 (E : Env) : Oracle E := ⟨fun _ => none, fun _ => none, fun d _ _ => d.ofBits 0⟩

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

def run : Option (Res Unit) := execLaunch ((launch d).env P0) (O0 _) 1000 (model d) (launch d) init

/-- The output's elements, as rationals, if the run finished. -/
def outs : Option (List Rat) := match run with
  | some (.done () m) =>
    (List.range n).mapM fun j => ((CMem.toMem m).read .f32 (d.output.ptr j).addr).map (·.toRat)
  | _ => none

#eval outs
#guard outs == some ((List.range n).map fun j => ((101 * j : Nat) : Rat))
