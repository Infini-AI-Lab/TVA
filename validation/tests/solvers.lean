import Solvers
import Exec

/-! Smoke tests of Triton's solvers (`validation/Solvers.lean`) on small concrete sites: each
outcome (`ok` with the expected value, `empty`, `assumed`, and each gap) where a family can give
it. -/

open TVA TVA.Validation

/-- A platform that relates nothing. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

def E0 : Env := { P := P0 }

/-- The oracle that answers nothing. -/
def H0 : Hint := fun _ => none

/-- An oracle whose tree is a single leaf (wrong for more than one leaf). -/
def Hleaf : Hint := fun q => match q with
  | .tree _ => some (.leaf 0)
  | _ => none

/-- An oracle whose lane order is empty. -/
def Hnoorder : Hint := fun q => match q with
  | .order _ => some []
  | _ => none

/-- An oracle answering every device question with zero bits, and the left-leaning tree. -/
def Hdev : Hint := fun q => match q with
  | .convert _ d' _ => some (d'.ofBits 0)
  | .atomicAdd d _ _ => some (d.ofBits 0)
  | .tree n => some (Tree.seq n)
  | _ => none

def tag (s : Site) (H : Hint := H0) : String := (solveSite s H).tag

def f32 (q : Rat) : F32 := FloatV.ofRat q
def i32 (z : Int) : I32 := IntV.ofInt z

/-- `[1, 2, 3, 4]` in `float32`. -/
def x4 : Tensor [4] F32 := Tensor.of1 fun i => f32 (i.1 + 1)

-- guards and hints
#guard tag (.guard (1 < 2)) == "ok"
#guard tag (.guard (2 < 1)) == "empty"
#guard tag (.hint Nat 3 (3 ∣ 6)) == "ok"
#guard tag (.hint Nat 3 (4 ∣ 6)) == "empty"

-- partial lanewise computations: `//` by zero has no result
def num : Tensor [3] I32 := Tensor.of1 fun i => i32 (10 * i.1)
def den (z : Int) : Tensor [3] I32 := Tensor.of1 fun i => i32 (z + i.1)
#guard tag (.lanes [3] I32 fun i => (num.get i).div? ((den 1).get i)) == "ok"
#guard tag (.lanes [3] I32 fun i => (num.get i).div? ((den (-1)).get i)) == "empty"
#guard ((solveSite (.lanes [3] I32 fun i => (num.get i).div? ((den 1).get i)) H0).val?.map
  fun r : Tensor [3] I32 => r.toList.map IntV.toInt) == some [0, 5, 6]

-- float sums: the sequential tree, the balanced one under a tighter bound, none under a bound
-- no tree meets, the oracle's tree checked
def fsumVal (depth : Nat) (H : Hint := H0) : Option Rat :=
  (solveSite (.fsum .binary32 false depth 4 x4) H).val?.map fun r : F32 => r.toRat
#guard fsumVal 3 == some 10
#guard fsumVal 2 == some 10
#guard tag (.fsum .binary32 false 2 4 x4) == "ok"
#guard (tag (.fsum .binary32 false 1 4 x4)).startsWith "none: TVA.Validation.Gap.searchFailed"
#guard (tag (.fsum .binary32 false 3 4 x4) Hleaf).startsWith "none: TVA.Validation.Gap.badAnswer"
#guard tag (.fsum .binary32 false 3 0 (Tensor.of1 fun i => nomatch i)) == "empty"
#guard tag (.fsum1 .binary32 false 3 2 4 (Tensor.of2 fun _ j => x4.get1 j)) == "ok"
#guard ((solveSite (.fmax .binary32 true 4 x4) H0).val?.map fun r : F32 => r.toRat) == some 4

-- index reductions: the first maximum, with `tie_break_left`
def y4 : Tensor [4] I32 := Tensor.of1 fun i => i32 ([3, 7, 7, 1].getD i.1 0)
#guard ((solveSite (.arg I32 true true 4 y4) H0).val?.map fun r : I32 × I32 =>
  (r.1.toInt, r.2.toInt)) == some (7, 1)
#guard ((solveSite (.arg I32 true false 4 y4) H0).val?.map fun r : I32 × I32 =>
  (r.1.toInt, r.2.toInt)) == some (7, 2)

-- histograms: an enabled element out of range has no result
def h4 (z : Int) : Tensor [4] I32 := Tensor.of1 fun i => i32 ([0, 1, 1, z].getD i.1 0)
#guard ((solveSite (.histogram 4 3 true 32 (h4 2) none) H0).val?.map fun r : Tensor [3] I32 =>
  r.toList.map IntV.toInt) == some [1, 2, 1]
#guard tag (.histogram 4 3 true 32 (h4 3) none) == "empty"
#guard tag (.histogram 4 3 true 32 (h4 3) (some (Tensor.of1 fun i => decide (i.1 < 3)))) == "ok"

-- padding: a NaN of an integer dtype does not exist
#guard tag (.pad [2] .f32 .nan (fun _ => false) (Tensor.full (f32 0))) == "ok"
#guard tag (.pad [2] .i32 .nan (fun _ => false) (Tensor.full (i32 0))) == "empty"
#guard tag (.pad [2] .i32 .nan (fun _ => true) (Tensor.full (i32 0))) == "ok"

-- lane orders: the oracle's checked, index order by default
def lanes2 : List (Idx [2]) := Idx.all [2]
#guard tag (.order [2] lanes2) == "ok"
#guard (tag (.order [2] lanes2) Hnoorder).startsWith "none: TVA.Validation.Gap.badAnswer"

-- conversions: computed, except into fp8, the oracle's (assumed)
#guard tag (.convert P0 [4] .i32 .f32 (Tensor.of1 fun i => i32 i.1)) == "ok"
#guard tag (.convert P0 [4] .f32 .i32 (Tensor.full (f32 (1 / 2)))) == "ok"
#guard tag (.convert P0 [4] .f32 .i8 (Tensor.full (f32 1000))) == "empty"
#guard tag (.convert P0 [4] .i32 .f8e5 (Tensor.of1 fun i => i32 i.1)) == "empty"
#guard (tag (.convert P0 [4] .f32 .f8e5 x4)).startsWith "none: TVA.Validation.Gap.noOracle"
#guard tag (.convert P0 [4] .f32 .f8e5 x4) Hdev == "assumed"

-- platform operations: the oracle's only
#guard (tag (.plat P0 (.div .binary32 x4 x4))).startsWith "none: TVA.Validation.Gap.noOracle"

-- sums of broadcast products: the products and the sequential tree
def m2 : Tensor [2, 2] DType.f32.Val := Tensor.of2 fun i j => f32 (i.1 + j.1)
#guard tag (.sumBcastMul P0 false .f32 2 2 2 m2 m2) == "ok"
#guard tag (.sumBcastMul P0 false .f32 2 2 2 m2 m2) Hdev == "ok"
#guard (tag (.other Nat fun _ => True)).startsWith "none: TVA.Validation.Gap.unsupported"

-- updates
def utag {d : DType} (u : Upd d) (old : d.Val) (H : Hint := H0) : String := (solveUpd u old H).tag
#guard ((solveUpd (Upd.add (d := .i32) E0 (i32 5)) (i32 7) H0).val?.map IntV.toInt) == some 12
#guard (utag (Upd.add (d := .f32) E0 (f32 1)) (f32 2)).startsWith
  "none: TVA.Validation.Gap.noOracle"
#guard utag (Upd.add (d := .f32) E0 (f32 1)) (f32 2) Hdev == "assumed"
#guard utag (Upd.add (d := .i1) E0 true) false == "empty"
#guard ((solveUpd (Upd.cas (d := .i32) (i32 7) (i32 9)) (i32 7) H0).val?.map IntV.toInt) == some 9
#guard ((solveUpd (Upd.max (d := .i32) false (i32 5)) (i32 7) H0).val?.map IntV.toInt) == some 5
#guard ((solveUpd (Upd.max (d := .f32) true (f32 5)) (f32 7) H0).val?.map FloatV.toRat) == some 7
#guard utag (Upd.red (d := .i32) E0 .xor (i32 5)) (i32 3) == "ok"
#guard utag (Upd.bit (d := .f32) IntV.and (f32 5)) (f32 3) == "empty"

/-! `exec` on tiny programs: what it reports, and its log. -/

def fill0 : Fill := fun d _ _ => d.ofBits 0
def run1 {α : Type} (p : Prog α) (O : Oracle := Oracle.empty) : Res α × List Record :=
  exec E0 O fill0 100 0 ∅ p
def kind {α : Type} : Res α → String
  | .done _ _ => "done" | .ub _ => "ub" | .stuck o _ => s!"stuck {o}" | .outOfFuel => "fuel"

-- two guards, then one that fails: undefined behaviour at request 2, after two records
def p3 : Prog Unit :=
  .pick (.guard (1 < 2)) fun _ => .pick (.guard (2 < 3)) fun _ =>
  .pick (.guard (3 < 3)) fun _ => .ret ()
#guard kind (run1 p3).1 == "ub" && (run1 p3).2.map (·.occ) == [0, 1]
-- an unsupported site: stuck at its occurrence
#guard kind (run1 (.pick (.guard True) fun _ => .pick (.other Nat fun _ => True) fun _ => .ret ())).1
  == "stuck 1"
-- a float `atomic_add`: the oracle's value, recorded as assumed; without one, stuck
def pAdd : Prog F32 := .rmw .f32 0 (Upd.add (d := .f32) E0 (f32 1)) .relaxed .gpu fun o => .ret o
def mem1 : CMem := CMem.write ∅ .f32 0 (f32 2)
#guard kind (exec E0 Oracle.empty fill0 10 0 mem1 pAdd).1 == "stuck 0"
#guard (exec E0 (fun _ => Hdev) fill0 10 0 mem1 pAdd).2.map (·.proved) == [false]
-- an atomic on an unmapped address is undefined behaviour
#guard kind (run1 pAdd).1 == "ub"
-- out of fuel
#guard kind (exec E0 Oracle.empty fill0 1 0 ∅ p3).1 == "fuel"
