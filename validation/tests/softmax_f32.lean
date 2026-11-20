import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `softmax_f32` run by `execLaunch` on two small covered instances
(`n_cols = 128`, the smallest the claim covers, `BLOCK_SIZE = 128`). Not a test against the
device (phase 4): it checks that `exec` runs the model end to end, and that the final memory is
what `Post` says, both halves.

* `strided`: `4` rows on `3` programs (program `0` takes rows `0` and `3`: the strided row loop).
  Row `0` is constant (`5/2`), row `1` holds `-3` in the even columns and `-∞` in the odd ones,
  row `2` holds one finite value (`7`, column `5`) and `-∞` elsewhere, row `3` is constant (`-1/4`).
* `nonfinite`: `2` rows on `2` programs. Row `0` is constant (`3/8`); row `1` is not `RowOk` (a NaN
  in column `0`, `1` elsewhere), for which `Post` asks only that every output holds a `float32`.

On a `RowOk` row whose finite inputs are all equal, the exact softmax is `1/k` at the `k` finite
inputs and `0` at the `-∞` ones, whatever `e` is: the test computes `σ` that way, from the inputs,
and `softmaxGap = log(σ_max/σ_j)` is `0` at a finite input and `log(σ_max/0) = log 0 = 0` (Lean's
conventions) at a `-∞` one. Each output `j` must then be finite and within
`elemTol · 4 · σ_j + floorTol` of `sc · σ_j` for one factor `sc` of the row with
`|sc − 1| ≤ sumTol`: the test computes the set of such `sc` exactly (an interval per positive
`σ_j`, `|c| ≤ floorTol` where `σ_j = 0`) and checks it is not empty, which is `Post`'s third
conjunct on these inputs, neither looser nor stricter.

The platform operations are `tl.exp` (`Platform.math .exp`) and the division (`Platform.div`),
one request each per row: the verdicts are `conditional` with `2` answers per row. The oracle
(`expOracle`) answers `tl.exp` only on lanes `±0` (with `1`), `-∞` (with `+0`) and NaN (a NaN),
and the division with the IEEE quotient. On `RowOk` rows every `x − max` is `0` or `-∞`, so the
exponentials are exactly `1` or `0`, the sum is the integer `k`, and the quotients `1/k`, `0/k`
(`k ∈ {1, 64, 128}`) are exact: values any correct device gives. On the NaN row the answers are
NaNs (any NaN; `Post` constrains that row only to hold a `float32`). The reductions (`tl.max`,
`tl.sum`) use the solver's default trees: on these inputs every tree gives the same value. -/

open TVA TVA.Validation Claims.Softmax

/-- A platform whose `tl.exp` is exact on `±0` and `-∞` and whose division is the IEEE quotient.
(`exec` does not consult it: the answers come from `expOracle`; it says what those answers
assume.) -/
def P0 : Platform where
  math fn _ x r := fn = .exp ∧
    ((x.isZero = true ∧ r = FloatV.ofRat 1) ∨ (x = FloatV.negInf ∧ r = FloatV.ofRat 0) ∨
      (x.isNaN = true ∧ r.isNaN = true))
  div _ x y r := r = FloatV.divRn x y
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- `e^x` on the lanes where it is exact: `1` at `±0`, `+0` at `-∞`, a NaN at a NaN; none
elsewhere. -/
def expExact {f : Format} (x : FloatV f) : Option (FloatV f) :=
  if x.isZero then some (FloatV.ofRat 1)
  else if x.isNaN then some FloatV.nan
  else if x.isInf && x.sign then some (FloatV.ofRat 0)
  else none

/-- The oracle: `tl.exp` lane by lane where `expExact` has a value (no answer if some lane has
none), the IEEE quotient for every division; nothing else (trees, orders: defaults). -/
def expOracle : Oracle := fun _ _ q => match q with
  | .plat (@PlatQ.math .exp _ s x) =>
    if (Idx.all s).all fun i => (expExact (x.get i)).isSome then
      some (x.map fun a => (expExact a).getD a)
    else none
  | .plat (.div _ x y) => some (Tensor.map2 FloatV.divRn x y)
  | _ => none

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- An input element: a finite rational, `-∞`, or a NaN. -/
inductive Cell where
  | fin (q : Rat)
  | ninf
  | nan
deriving BEq

def Cell.val : Cell → F32
  | .fin q => FloatV.ofRat q
  | .ninf => FloatV.negInf
  | .nan => FloatV.nan

/-- A sentinel in the output elements, and just outside both buffers. -/
def sentinel : F32 := FloatV.ofRat (-7)

/-- Sentinel elements on each side of `y`. -/
def pad : Nat := 4

/-- The width: the smallest covered (`n_cols` a positive multiple of `128`), `BLOCK_SIZE = 2^7`. -/
def C : Nat := 128

/-- One case: `R` rows on `np` programs, the input cells. -/
structure Case where
  R : Nat
  np : Nat
  cell : Nat → Nat → Cell

def Case.inst (c : Case) : Inst := Claims.Softmax.mk c.R C 7 4 c.np

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .f32 ((List.range (c.R * C)).map fun k : Nat =>
    ((d.x.ptr k).addr, (c.cell (k / C) (k % C)).val))
  CMem.ofElems m .f32 ((List.range (c.R * C + 2 * pad)).map fun k : Nat =>
    ((d.y.ptr ((k : Int) - pad)).addr, sentinel))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) expOracle fill0 1000 (model c.inst) (launch c.inst) c.init

/-- The row is `RowOk` (every cell finite or `-∞`, one finite) and its finite cells are all equal:
the class of rows whose exact softmax the test computes. `none` for a row outside `RowOk`. -/
def Case.rowClass (c : Case) (r : Nat) : Option (Option Rat) :=
  let cs := (List.range C).map (c.cell r)
  if cs.any (· == .nan) then none
  else match cs.find? (fun x => x != .ninf) with
    | some (.fin q) => if cs.all (fun x => x == .ninf || x == .fin q) then some (some q) else some none
    | _ => none

/-- The exact softmax `σ_j` of a `RowOk` row whose finite inputs are equal: `1/k` at the `k`
finite inputs, `0` at `-∞`. -/
def Case.sigma (c : Case) (r j : Nat) : Rat :=
  let k := ((List.range C).filter fun j' => c.cell r j' != .ninf).length
  if c.cell r j == .ninf then 0 else 1 / (k : Rat)

/-- `Post`'s third conjunct for row `r` on the outputs `cs` (all finite): some `sc` with
`|sc − 1| ≤ sumTol` such that `|c_j − sc·σ_j| ≤ elemTol·4·σ_j + floorTol` for every `j`. The
feasible `sc` form an interval; `none` when it is empty. -/
def scInterval (sig : Nat → Rat) (cs : List Rat) : Option (Rat × Rat) :=
  let init : Option (Rat × Rat) := some (1 - sumTol, 1 + sumTol)
  (cs.zipIdx).foldl (fun acc (v, j) => acc.bind fun (lo, hi) =>
    let s := sig j
    let tol := elemTol * 4 * s + floorTol
    if s == 0 then (if |v| ≤ tol then some (lo, hi) else none)
    else
      let lo' := max lo ((v - tol) / s)
      let hi' := min hi ((v + tol) / s)
      if lo' ≤ hi' then some (lo', hi') else none) init

/-- The final memory against `Post`, with `sig` the expected softmax (`c.sigma` for the test):
for each row, every output element holds a `float32`; on a `RowOk` row each is finite and a
common factor `sc` exists (`scInterval`); every byte outside the output elements (`x`, the
sentinels around `y`) is unchanged, and no byte is added. -/
def Case.agreeE (c : Case) (sig : Nat → Nat → Rat) (cf : CConfig (launch c.inst).size) :
    Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let badRows := (List.range c.R).filter fun r : Nat =>
    let outs := (List.range C).map fun j : Nat => m.read .f32 (d.y.ptr (outIdx d r j)).addr
    if outs.any (·.isNone) then true
    else match c.rowClass r with
      | none => false
      | some none => true  -- a `RowOk` row this test cannot compute: fail loudly
      | some (some _) =>
        let vs := outs.filterMap id
        !(vs.all (·.isFinite)) || (scInterval (sig r) (vs.map (·.toRat))).isNone
  let lo := d.y.base
  let hi := d.y.base + (c.R * C * 4 : Nat)
  let changed := c.init.toList.filter fun (a, b) => !(lo ≤ a && a < hi) && cf.mem[a]? != some b
  if !badRows.isEmpty then some s!"rows {badRows} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the output elements changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c.sigma

/-- Four rows on three programs: two constant rows, half `-∞`, one finite. -/
def strided : Case :=
  { R := 4, np := 3, cell := fun r j =>
      match r with
      | 0 => .fin (5 / 2)
      | 1 => if j % 2 = 0 then .fin (-3) else .ninf
      | 2 => if j = 5 then .fin 7 else .ninf
      | _ => .fin (-1 / 4) }

/-- Two rows on two programs: a constant row, and a row with a NaN (not `RowOk`). -/
def nonfinite : Case :=
  { R := 2, np := 2, cell := fun r j =>
      match r with
      | 0 => .fin (3 / 8)
      | _ => if j = 0 then .nan else .fin 1 }

theorem strided_ok : strided.inst.ok :=
  mk_ok (by norm_num) (by norm_num [C]) (by norm_num [strided]) (by norm_num [strided])
    (by norm_num [strided]) (by norm_num [strided, C]) (by norm_num [strided, C]) (by norm_num [C])
theorem nonfinite_ok : nonfinite.inst.ok :=
  mk_ok (by norm_num) (by norm_num [C]) (by norm_num [nonfinite]) (by norm_num [nonfinite])
    (by norm_num [nonfinite]) (by norm_num [nonfinite, C]) (by norm_num [nonfinite, C])
    (by norm_num [C])

def cases : List (String × Case) := [("strided", strided), ("nonfinite", nonfinite)]

-- `Pre`: both buffers aligned and inside `[0, 2^64)`, every element present in `init`.
#guard cases.all fun (_, c) =>
  let d := c.inst
  let m := CMem.toMem c.init
  d.x.base % 4 == 0 && d.y.base % 4 == 0 && d.y.base + (d.y.len * 4 : Nat) ≤ 2 ^ 64 &&
  (List.range d.x.len).all (fun k : Nat => (m.read .f32 (d.x.ptr k).addr).isSome) &&
  (List.range d.y.len).all (fun k : Nat => (m.read .f32 (d.y.ptr k).addr).isSome)
-- The cells are encoded exactly; the rows are the classes described (`RowOk` with equal finite
-- values, except `nonfinite`'s row 1).
#guard cases.all fun (_, c) => (List.range c.R).all fun r : Nat => (List.range C).all fun j : Nat =>
  match c.cell r j with
  | .fin q => (Cell.val (.fin q)).toRat == q && (Cell.val (.fin q)).isFinite
  | .ninf => Cell.val .ninf == FloatV.negInf
  | .nan => (Cell.val .nan).isNaN
#guard [0, 1, 2, 3].all (fun r => (strided.rowClass r).isSome && (strided.rowClass r).get!.isSome) &&
  (nonfinite.rowClass 0).isSome && (nonfinite.rowClass 1).isNone
-- No expected output equals the sentinel's value (`σ ∈ {0, 1/128, 1/64, 1}`).
#guard (List.range 4).all fun r : Nat => (List.range C).all fun j : Nat => strided.sigma r j != -7

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch c.agree c.run

def verdictStrided : Verdict := strided.verdict
def verdictNonfinite : Verdict := nonfinite.verdict

#eval IO.println (verdictStrided.line "softmax_f32/strided")
#eval IO.println (verdictNonfinite.line "softmax_f32/nonfinite")
-- Agreement resting on two answers per row (its `tl.exp` and its division).
#guard verdictStrided == .conditional 8
#guard verdictNonfinite == .conditional 4

-- Not vacuous: a wrong expectation is caught (row 1's softmax as if it had no `-∞`; row 2's
-- single `1` moved to column 6).
#guard
  let r := strided.run
  (Verdict.ofLaunch (strided.agreeE fun r j => if r = 1 then 1 / 128 else strided.sigma r j) r).isDiscrepancy &&
  (Verdict.ofLaunch (strided.agreeE fun r j =>
    if r = 2 then (if j = 6 then 1 else 0) else strided.sigma r j) r).isDiscrepancy
