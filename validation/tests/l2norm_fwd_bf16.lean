import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `l2norm_fwd_bf16` run by `execLaunch` on four small covered
instances. Not a test against the device (phase 4): it checks that `exec` runs the model end to
end, and that the final memory is what `Post` says, both halves: for each row `t < T`, `rstd[t]`
and every `y[t, j]` within `Post`'s tolerances of the exact `1 / sqrt(Σ_j x[t, j]² + eps)` and
`x[t, j]·rstdRef`; and every other byte unchanged (`x`, sentinels in `y` and `rstd` past their
written elements, which the masked rows and columns must not write, and no byte added).

The expectations are computed here from the inputs, as rationals. `rstdRef` is irrational in
general, so the checks are decided without it: `|rc − r| ≤ tol·r` with `r = 1/√V` is, for
`rc > 0`, exactly `((1 − tol)/rc)² ≤ V ≤ ((1 + tol)/rc)²`, decided in the rationals; the check of
`y[t, j]`, `|v − a·r| ≤ yTol·|a|·r + yFloor`, is concave in `r`, so it holds on the whole interval
`[1/hi, 1/lo]` (`lo ≤ √V ≤ hi`, rational bounds `2^-200` apart, equal when `V` is a square) when it
holds at both ends: never looser than `Post`. The tolerances are `Post`'s own constants
(`rstdTol`, `yTol`, `yFloor`, `yTol32`, `yFloor32`, which `Post` names).

**Platform operations.** Each program runs one `tl.sqrt` and one `/` (`1 / sqrt`), both platform
operations (`Platform.math .sqrt`, `Platform.div`); the oracle answers them with the IEEE
correctly rounded results (`FloatV.sqrtRn`, `FloatV.divRn`), so each verdict is `conditional`
with two assumed answers per program.
* `exact_bf16`, `exact_f32` (`eps = 2^-124`): every row's `Σ x²` is `4^k` (or `0`), all products
  and partial sums are exact (so the fusable sum and `+ eps` give the same value along any tree,
  fused or not), `+ eps` rounds back to `Σ x²` (or is `2^-124` on a zero row), so `tl.sqrt` sees a
  `float32` square (`2^k`, or `2^-62`) and `1 / sqrt` a power of two: the answers are the exact
  values, the ones a correct device gives (the device facts allow `2^-22` around them).
* `host_one`, `host_f32` (the host's `eps = 1e-6`): `Σ x² + eps` is not a square; the answers are
  the correctly rounded square root and quotient, which `Device.SqrtF32` and `Device.DivFull`
  allow (a device may give others within `2^-22`); the outputs are then checked against `Post`'s
  tolerance on these values. -/

open TVA TVA.Validation Claims.L2NormFwd

/-- The platform the oracle's answers represent: IEEE square root and division. (`exec` does not
consult it; it says what the answers assume.) -/
def P0 : Platform where
  math fn _ x r := fn = .sqrt ∧ r = FloatV.sqrtRn x
  div _ x y r := r = FloatV.divRn x y
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- The oracle: `tl.sqrt` and `/` lane by lane, correctly rounded; nothing else. -/
def sqrtDivOracle : Oracle := fun _ _ q => match q with
  | .plat (.math .sqrt _ x) => some (x.map FloatV.sqrtRn)
  | .plat (.div _ x y) => some (Tensor.map2 FloatV.divRn x y)
  | _ => none

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## Square roots, bounded in the rationals -/

/-- Rational bounds `lo ≤ √q ≤ hi`, `2^-200 / den` apart (equal when `q·den²·4^200` is a square):
`√(n/m) = √(n·m·4^200) / (m·2^200)`. -/
def sqrtBounds (q : Rat) : Rat × Rat :=
  let n := q.num.toNat
  let m := q.den
  let N := n * m * 4 ^ 200
  let s := Nat.sqrt N
  let sc : Rat := (m * 2 ^ 200 : Nat)
  (s / sc, if s * s == N then s / sc else (s + 1 : Nat) / sc)

/-- `rc` is within `tol·r` of `r = 1/√V`, decided exactly (`rc > 0` and
`((1 − tol)/rc)² ≤ V ≤ ((1 + tol)/rc)²`). -/
def rstdWithin (tol V rc : Rat) : Bool :=
  0 < rc && ((1 - tol) / rc) ^ 2 ≤ V && V ≤ ((1 + tol) / rc) ^ 2

/-- `|v − a·r| ≤ rel·|a|·r + flr` for every `r` in `[1/hi, 1/lo]` (`lo ≤ √V ≤ hi`): checked at both
ends, the condition being concave in `r`. -/
def yWithin (rel flr V a v : Rat) : Bool :=
  let (lo, hi) := sqrtBounds V
  0 < lo && [1 / hi, 1 / lo].all fun r => |v - a * r| ≤ rel * |a| * r + flr

#guard sqrtBounds 4 == (2, 2) && sqrtBounds (1 / 4) == (1 / 2, 1 / 2)
#guard let (lo, hi) := sqrtBounds 2; lo * lo ≤ 2 && 2 ≤ hi * hi && hi - lo < 1 / 2 ^ 190

/-! ## The cases -/

/-- The double `2^-124` (`eps` of the exact cases: `≥ 2^-125`, and its square root `2^-62` is a
`float32`). -/
def eps124 : F64 := FloatV.ofPat 0x3830000000000000

/-- One case: the shape, `y`'s dtype, `eps`, and the inputs `x[t, j]` as rationals (each exact in
`bfloat16`, checked below). -/
structure Case where
  T : Nat
  D : Nat
  BD : Nat
  f32 : Bool
  eps : F64
  xv : Nat → Nat → Rat

/-- The host's instance (`mk`: `BT = 16`, its addresses) with the case's `eps`. -/
def Case.inst (c : Case) : Inst := { Claims.L2NormFwd.mk c.T c.D c.BD c.f32 with eps := c.eps }

/-- Sentinels: in every element of `y` and `rstd` (and `pad` elements on each side) before the
launch. -/
def sentY : Rat := -99
def sentR : Rat := -77
def pad : Nat := 128

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .bf16 ((List.range (c.T * c.D)).map fun k : Nat =>
    ((d.x.ptr k).addr, (FloatV.ofRat (c.xv (k / c.D) (k % c.D)) : BF16)))
  let ks := (List.range (c.T * c.D + 2 * pad)).map fun e : Nat => (e : Int) - pad
  let m := match d.y with
    | .inl b => CMem.ofElems m .bf16 (ks.map fun k => ((b.ptr k).addr, (FloatV.ofRat sentY : BF16)))
    | .inr b => CMem.ofElems m .f32 (ks.map fun k => ((b.ptr k).addr, (FloatV.ofRat sentY : F32)))
  CMem.ofElems m .f32 ((List.range (c.T + 2 * pad)).map fun e : Nat =>
    ((d.rstd.ptr ((e : Int) - pad)).addr, (FloatV.ofRat sentR : F32)))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) sqrtDivOracle fill0 1000 (model c.inst) (launch c.inst) c.init

/-- Element `k` of `y` in memory `m`: finite, and its value. -/
def Case.readY (c : Case) (m : Mem) (k : Nat) : Option (Bool × Rat) :=
  match c.inst.y with
  | .inl b => (m.read .bf16 (b.ptr k).addr).map fun v => (v.isFinite, v.toRat)
  | .inr b => (m.read .f32 (b.ptr k).addr).map fun v => (v.isFinite, v.toRat)

/-- `y`'s tolerance, relative and absolute, by its dtype (`Post`'s `YOk`). -/
def Case.yTols (c : Case) : Rat × Rat := if c.f32 then (yTol32, yFloor32) else (yTol, yFloor)

/-- `V = Σ_j x[t, j]² + eps` of row `t`, exact, from the inputs `xs`. -/
def Case.vOf (c : Case) (xs : Nat → Nat → Rat) (t : Nat) : Rat :=
  ((List.range c.D).map fun j => xs t j ^ 2).sum + c.eps.toRat

/-- The bytes of `y`'s and `rstd`'s written elements (`T·D` of `y`, `T` of `rstd`). -/
def Case.written (c : Case) (a : Int) : Bool :=
  let d := c.inst
  let yb : Int := match d.y with | .inl b => b.base | .inr b => b.base
  let ysz : Nat := if c.f32 then 4 else 2
  (yb ≤ a && a < yb + (c.T * c.D * ysz : Nat)) ||
    (d.rstd.base ≤ a && a < d.rstd.base + (c.T * 4 : Nat))

/-- The final memory against `Post`, the expectations computed from the inputs `xs` (`xs = xv`
for the test): each `rstd[t]` and `y[t, j]` within `Post`'s tolerance; every byte outside the
written elements unchanged; no byte added. -/
def Case.agreeX (c : Case) (xs : Nat → Nat → Rat) (cf : CConfig (launch c.inst).size) :
    Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let (rel, flr) := c.yTols
  let badR := (List.range c.T).filter fun t : Nat =>
    match m.read .f32 (d.rstd.ptr t).addr with
    | some rc => !(rc.isFinite && rstdWithin rstdTol (c.vOf xs t) rc.toRat)
    | none => true
  let badY := (List.range (c.T * c.D)).filter fun k : Nat =>
    match c.readY m k with
    | some (fin, v) => !(fin && yWithin rel flr (c.vOf xs (k / c.D)) (xs (k / c.D) (k % c.D)) v)
    | none => true
  let changed := c.init.toList.filter fun (a, b) => !c.written a && cf.mem[a]? != some b
  if !badR.isEmpty then some s!"rstd elements {badR.take 8} differ from Post"
  else if !badY.isEmpty then some s!"y elements {badY.take 8} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the written elements changed"
  else if cf.mem.size != c.init.size then some "the launch wrote outside the initial memory"
  else none

def Case.agree (c : Case) := c.agreeX c.xv

/-- Row patterns whose `Σ x²` is `4^k` or `0`: `256`, `0`, `16`, `64`, `1`, `1/4`, `2^80`, `2^-78`
(the last rounds to `2^-78` with `eps = 2^-124`, but `rstdRef` keeps the `eps`). -/
def rows5 (t j : Nat) : Rat :=
  match t % 8 with
  | 0 => [12, 4, -4, 4, 8].getD j 0
  | 1 => 0
  | 2 => [2, -2, 2, -2, 0].getD j 0
  | 3 => [6, -4, 2, 2, -2].getD j 0
  | 4 => [1 / 2, 1 / 2, -1 / 2, 1 / 2, 0].getD j 0
  | 5 => [1 / 4, -1 / 4, 1 / 4, 1 / 4, 0].getD j 0
  | 6 => [2 ^ 40, 0, 0, 0, 0].getD j 0
  | _ => [1 / 2 ^ 40, 1 / 2 ^ 40, -1 / 2 ^ 40, 1 / 2 ^ 40, 0].getD j 0

/-- `T = 19`, `D = 5`, `BD = 8`, a `bfloat16` `y`, `eps = 2^-124`: two programs of `BT = 16` rows,
the second with 13 masked rows (their loads `0`, their stores disabled: the sentinels after `y`
and `rstd` must stay), three masked columns in every row; a zero row (`rstd = 2^62`, `y = 0`). -/
def exactBf16 : Case := { T := 19, D := 5, BD := 8, f32 := false, eps := eps124, xv := rows5 }

/-- `T = 5`, `D = 4`, `BD = 4`, a `float32` `y` (`output_dtype = torch.float32`), `eps = 2^-124`:
one program, no masked column; `Σ x²` = `4`, `16`, `1`, `0`, `64`. -/
def exactF32 : Case :=
  { T := 5, D := 4, BD := 4, f32 := true, eps := eps124,
    xv := fun t j => match t with
      | 0 => [1, 1, 1, 1].getD j 0
      | 1 => [-2, 2, 2, 2].getD j 0
      | 2 => [1 / 2, 1 / 2, 1 / 2, -1 / 2].getD j 0
      | 3 => 0
      | _ => [-4, 4, 4, 4].getD j 0 }

/-- The host's test case `(1, 1, 1, 1)` (`setup_1_1_1_1`: `T = D = BD = 1`, `eps = 1e-6`, a
`bfloat16` `y`), with `x = 3`: `9 + eps` is not a square. -/
def hostOne : Case := { T := 1, D := 1, BD := 1, f32 := false, eps := eps1e6, xv := fun _ _ => 3 }

/-- The host's launch with a `float32` `y`, `T = 3`, `D = 3`, `BD = 4`, `eps = 1e-6`: rows
`[1, 2, 3]`, `[-1/2, 1/4, 7]` (sums `14`, `49.3125`, exact in `float32`) and a zero row
(`rstd ≈ 1000`, `y = 0`). -/
def hostF32 : Case :=
  { T := 3, D := 3, BD := 4, f32 := true, eps := eps1e6,
    xv := fun t j => match t with
      | 0 => [1, 2, 3].getD j 0
      | 1 => [-1 / 2, 1 / 4, 7].getD j 0
      | _ => 0 }

/-! ## Covered -/

/-- `Inst.ok` of a case, from `mk_ok` (every field but `eps`'s) and `eps`'s three facts. -/
theorem Case.ok_of (c : Case) (h : (Claims.L2NormFwd.mk c.T c.D c.BD c.f32).ok)
    (h1 : c.eps.isFinite = true) (h2 : (2 : ℚ) ^ (-125 : ℤ) ≤ c.eps.toRat)
    (h3 : c.eps.toRat ≤ (2 : ℚ) ^ (126 : ℤ)) : c.inst.ok where
  d_pos := h.d_pos
  d_le := h.d_le
  bd_pow2 := h.bd_pow2
  d_bd := h.d_bd
  bd_le := h.bd_le
  bt_pow2 := h.bt_pow2
  block_le := h.block_le
  grid_lt := h.grid_lt
  x_len := h.x_len
  y_len := h.y_len
  r_len := h.r_len
  y_x := h.y_x
  r_x := h.r_x
  r_y := h.r_y
  eps_fin := h1
  eps_lo := h2
  eps_hi := h3

theorem eps124_fin : eps124.isFinite = true := by decide
theorem eps124_lo : (2 : ℚ) ^ (-125 : ℤ) ≤ eps124.toRat := by decide +kernel
theorem eps124_hi : eps124.toRat ≤ (2 : ℚ) ^ (126 : ℤ) := by decide +kernel

theorem exactBf16_ok : exactBf16.inst.ok := by
  unfold exactBf16
  exact Case.ok_of _ (mk_ok 19 5 8 3 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)) eps124_fin eps124_lo eps124_hi
theorem exactF32_ok : exactF32.inst.ok := by
  unfold exactF32
  exact Case.ok_of _ (mk_ok 5 4 4 2 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) true) eps124_fin eps124_lo eps124_hi
theorem hostOne_ok : hostOne.inst.ok := by
  unfold hostOne
  exact Case.ok_of _ setup_1_1_1_1_ok eps1e6_fin eps1e6_lo eps1e6_hi
theorem hostF32_ok : hostF32.inst.ok := by
  unfold hostF32
  exact Case.ok_of _ (mk_ok 3 3 4 2 (by norm_num) (by norm_num) rfl (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) true) eps1e6_fin eps1e6_lo eps1e6_hi

-- The host cases are the host's own instances (`mk`'s `eps` is `eps1e6`).
example : hostOne.inst = setup_1_1_1_1 := rfl
example : hostF32.inst = Claims.L2NormFwd.mk 3 3 4 true := rfl

def cases : List (String × Case) :=
  [("exact_bf16", exactBf16), ("exact_f32", exactF32), ("host_one", hostOne), ("host_f32", hostF32)]

-- `eps124` is `2^-124`, and the kernel's `float32` `eps` is the same value.
#guard eps124.toRat == 1 / 2 ^ 124 && (FloatV.hostF32 eps124).toRat == 1 / 2 ^ 124

-- `Pre`: the inputs are encoded exactly (finite `bfloat16`s), every row has `Σ x² + eps ≤ 2^126`;
-- the three buffers are aligned and inside `[0, 2^64)`, and every byte of them is present.
#guard cases.all fun (_, c) =>
  let d := c.inst
  let m := CMem.toMem c.init
  (List.range (c.T * c.D)).all (fun k : Nat =>
    let v : BF16 := FloatV.ofRat (c.xv (k / c.D) (k % c.D))
    v.isFinite && v.toRat == c.xv (k / c.D) (k % c.D) &&
      (m.read .bf16 (d.x.ptr k).addr).map DType.bf16.toBits == some (DType.bf16.toBits v)) &&
  (List.range c.T).all (fun t => c.vOf c.xv t ≤ 2 ^ 126) &&
  d.x.base % 2 == 0 && d.rstd.base % 4 == 0 && d.rstd.base + d.rstd.len * 4 ≤ 2 ^ 64 &&
  (List.range d.x.len).all (fun k : Nat => (m.read .bf16 (d.x.ptr k).addr).isSome) &&
  (List.range d.rstd.len).all (fun k : Nat => (m.read .f32 (d.rstd.ptr k).addr).isSome) &&
  (List.range (yLen d)).all (fun k : Nat => (c.readY m k).isSome) &&
  (match d.y with | .inl b => b.base % 2 == 0 | .inr b => b.base % 4 == 0)

-- The sentinels are outside every tolerance (an unwritten element would be caught).
#guard cases.all fun (_, c) =>
  let (rel, flr) := c.yTols
  (List.range c.T).all fun t => !rstdWithin rstdTol (c.vOf c.xv t) sentR &&
    (List.range c.D).all fun j => !yWithin rel flr (c.vOf c.xv t) (c.xv t j) sentY

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch c.agree c.run

/-- A wrong expectation: row 0's first input taken as `x + 1`. -/
def Case.wrongX (c : Case) : Nat → Nat → Rat := fun t j => if t = 0 ∧ j = 0 then c.xv t j + 1 else c.xv t j

/-- Each case's verdict, and the verdict of the wrong expectation on the same run. -/
def results : List (String × Case × Verdict × Verdict) := cases.map fun (n, c) =>
  let r := c.run
  (n, c, Verdict.ofLaunch c.agree r, Verdict.ofLaunch (c.agreeX c.wrongX) r)

#eval results.forM fun (n, _, v, _) => IO.println (v.line s!"l2norm_fwd_bf16/{n}")
-- Agreement resting on two assumed answers (`tl.sqrt`, `/`) per program; and not vacuous: the
-- wrong expectation is a discrepancy.
#guard results.all fun (_, c, v, w) =>
  v == .conditional (2 * ((c.T + 15) / 16)) && w.isDiscrepancy

-- In the exact cases the outputs are the forced exact values: `rstd = 1/√(Σ x²)` (`2^62` on a
-- zero row, `2^39` on the `2^-78` row, whose `+ eps` rounds away) and `y = x·rstd`.
#guard [exactBf16, exactF32].all fun c => match c.run with
  | (.done cf, _) =>
    let m := CMem.toMem cf.mem
    let d := c.inst
    (List.range c.T).all fun t : Nat =>
      let s := ((List.range c.D).map fun j => c.xv t j ^ 2).sum
      let r : Rat := if s = 0 then 2 ^ 62 else 1 / (sqrtBounds s).1
      (sqrtBounds s).1 == (sqrtBounds s).2 &&
      (m.read .f32 (d.rstd.ptr t).addr).map (·.toRat) == some r &&
      (List.range c.D).all fun j => (c.readY m (t * c.D + j)).map (·.2) == some (c.xv t j * r)
  | _ => false
