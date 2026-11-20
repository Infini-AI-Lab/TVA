import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `gdn_gating_bf16` run by `execLaunch` on two small covered instances
(`mk`: contiguous `a`, `b`; and split, the halves of one `[batch, 2·num_heads]` projection). Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says, all three parts: every `g[i·H + h]` a finite `float32` within
`gTol` of `gRef = -e^A·softplus(a + dt_bias)`; every `beta_output[i·H + h]` a finite `float32`
within `bTol` of `σ(b)`; every other byte unchanged (the inputs, sentinels after both outputs, which
the masked head lanes must not write, and no byte added).

**How `Post` is decided.** `gRef`, `gTol`, `σ` and `bTol` involve `e^·` and `ln`, irrational in
general. The test bounds them with rationals computed here (`expBounds`: Taylor series with its
remainder, on `q / 2^k`, then `k` squarings, rounded outwards at `2^-400`; `lnBounds`: `k·ln 2 +
2·atanh((m − 1)/(m + 1))` with `m ∈ [1, 2)`, rounded outwards, plus the series' remainder), and
checks `Post` on the whole box: `g`'s condition `|c + E·sp| ≤ gTol(E, sp)` holds for every
`E ∈ [Elo, Ehi]`, `sp ∈ [splo, sphi]` when `max(|c + Elo·splo|, |c + Ehi·sphi|) ≤ gTol(Elo, splo)`
(`|c + P|` is convex in `P = E·sp`, `gTol` increasing in both); `beta_output`'s condition is concave
in `σ`, so it holds on `[1/(1 + ehi), 1/(1 + elo)]` when it holds at both ends. Never looser than
`Post`; the tolerances are `Post`'s own formulas, written out here with these bounds.

**Platform operations.** Each program (one token, one block of 8 heads) runs five: `tl.exp` of
`x`, `tl.log` of `1 + exp(x)`, `tl.exp` of `A_log`, `tl.exp` of `-b`, and the `/` of `sigmoid`. The
oracle answers each lane with the correctly rounded value (`exp` and `log` the rounding of the
midpoint of the bounds above, which are far tighter than one `float32` ulp; `±∞`, `0` and NaN at the
special operands; `/` the IEEE quotient `divRn`), so the verdicts are `conditional` with `5` per
program.
* `exact`: every output is the same for every answer the device facts allow (`Device.ExpF32`,
  `LogF32`, `DivFull`, `ExpF32NegInf`): `A_log = 0` (`exp(0) = 1`); `x = a + dt_bias` either above
  `20` (`softplus = x`; the `exp`/`log` lanes are computed and discarded by `tl.where`, `exp(x)`
  overflowing to `+∞` for `x ≥ 89`), or `≤ -32` (`exp(x) < 2^-46`, so `1 + exp(x)` rounds to `1`
  and `log(1) = 0`: `g = -0`), or `-∞` (the `float32` sum of two `-1.5·2^127` overflows, `exp(-∞) =
  0`); `b = 0` (`σ = 1/2`) or `b ≥ 17` (`1 + exp(-b)` rounds to `1`: `σ = 1`). So `g` is exactly
  `-x` or `-0` and `beta_output` exactly `1/2` or `1`.
* `general`: realistic values (`A_log = ln A` with `A ∈ (0, 16)`, `x` around the threshold, `x = 20`
  exactly, `b` down to `-88`, whose `σ ≈ 6·10^-39` is a `float32` and `bfloat16` subnormal); the
  answers are the correctly rounded values, one of many the device facts allow, and the outputs are
  checked against `Post`'s tolerances on them. -/

open TVA TVA.Validation Claims.GdnGating

/-! ## Rational bounds on `exp` and `ln` -/

/-- The rounding precision of the bounds: `2^-400` absolute. -/
def prec : Nat := 400
def rdn (q : Rat) : Rat := ((q * 2 ^ prec).floor : Rat) / 2 ^ prec
def rup (q : Rat) : Rat := ((q * 2 ^ prec).ceil : Rat) / 2 ^ prec

/-- `Σ_{i<n} r^i/i!` and `r^n/n!` (exact). -/
def taylor (r : Rat) (n : Nat) : Rat × Rat :=
  (List.range n).foldl (fun (s, t) i => (s + t, t * r / (i + 1 : Nat))) (0, 1)

/-- `lo ≤ e^q ≤ hi`. For `|q| ≤ 2^(k-4)`, `r = q/2^k` has `|r| ≤ 1/16` and the rest of the series
after 30 terms is at most `2·|r|^30/30!`; then `e^q = (e^r)^(2^k)`, each square rounded outwards.
Below `-200`, `(0, 2^-280)` (`e^-200 < 2^-288`). -/
def expBounds (q : Rat) : Rat × Rat :=
  if q < -200 then (0, 1 / 2 ^ 280) else
  let k := Nat.log2 (q.num.natAbs / q.den + 1) + 5
  let r := q / 2 ^ k
  let (s, t) := taylor r 30
  let lo := rdn (s - 2 * |t|)
  let hi := rup (s + 2 * |t|)
  (List.range k).foldl (fun (lo, hi) _ => (rdn (lo * lo), rup (hi * hi))) (lo, hi)

/-- `2·Σ_{i<80} z^(2i+1)/(2i+1)`, rounded down (`lower`) or up (adding the rest, at most
`3·z^161` for `0 ≤ z ≤ 1/3`): bounds on `ln((1 + z)/(1 − z))`, for `z` rounded the same way. -/
def atanh2 (z : Rat) (lower : Bool) : Rat :=
  let rd := if lower then rdn else rup
  let z2 := rd (z * z)
  let (s, pw) := (List.range 80).foldl
    (fun (s, pw) i => (rd (s + rd (2 * pw / (2 * i + 1 : Nat))), rd (pw * z2))) (0, rd z)
  if lower then s else rup (s + 3 * pw)

def ln2lo : Rat := atanh2 (1 / 3) true
def ln2hi : Rat := atanh2 (1 / 3) false

/-- `⌊log₂ y⌋` for `y > 0`. -/
def flog2 (y : Rat) : Int :=
  let k0 : Int := (Nat.log2 y.num.natAbs : Int) - Nat.log2 y.den
  let m := y / (2 : Rat) ^ k0
  if m < 1 then k0 - 1 else if 2 ≤ m then k0 + 1 else k0

/-- `ln y ≥ lnLo y` (`y > 0`): `y = 2^k·m`, `m ∈ [1, 2)`, `ln m = 2·atanh((m − 1)/(m + 1))`. -/
def lnLo (y : Rat) : Rat :=
  let k := flog2 y
  let m := y / (2 : Rat) ^ k
  k * (if 0 ≤ k then ln2lo else ln2hi) + atanh2 (rdn ((m - 1) / (m + 1))) true

/-- `ln y ≤ lnHi y` (`y > 0`). -/
def lnHi (y : Rat) : Rat :=
  let k := flog2 y
  let m := y / (2 : Rat) ^ k
  k * (if 0 ≤ k then ln2hi else ln2lo) + atanh2 (rup ((m - 1) / (m + 1))) false

-- Sanity: `e^1`, `e^-88`, `ln 2`, `ln 10` against known decimals.
#guard let (lo, hi) := expBounds 1; 2.718281828459045 < lo && hi < 2.718281828459046
#guard let (lo, hi) := expBounds (-88); 6.0546018954e-39 < lo && hi < 6.0546018955e-39 && hi - lo < lo / 2 ^ 150
#guard 0.6931471805599453 < ln2lo && ln2hi < 0.6931471805599454 && ln2hi - ln2lo < 1 / 2 ^ 250
#guard 2.302585092994045 < lnLo 10 && lnHi 10 < 2.302585092994046 && lnLo 1 == 0 && lnHi 1 == 0

/-! ## The oracle -/

/-- The correctly rounded `e^x` (to nearest; the midpoint of bounds `2^-150`-relative tight), with
`exp(-∞) = +0`, `exp(+∞) = +∞`, `exp(x) = +∞` for `x ≥ 89` and `+0` below `-200`. -/
def expRn {f : Format} (x : FloatV f) : FloatV f :=
  if x.isNaN then x
  else if x.isInf then (if FloatV.lt x (FloatV.ofRat 0) then FloatV.ofRat 0 else x)
  else if 89 ≤ x.toRat then FloatV.posInf
  else let (lo, hi) := expBounds x.toRat; FloatV.ofRat ((lo + hi) / 2)

/-- The correctly rounded `ln y`, with `ln(+∞) = +∞`, `ln(±0) = -∞`, NaN below `0`. -/
def logRn {f : Format} (y : FloatV f) : FloatV f :=
  if y.isNaN then y
  else if y.isZero then FloatV.negInf
  else if FloatV.lt y (FloatV.ofRat 0) then FloatV.nan
  else if y.isInf then y
  else FloatV.ofRat ((lnLo y.toRat + lnHi y.toRat) / 2)

/-- The platform the oracle's answers represent: correctly rounded `exp`, `log` and division.
(`exec` does not consult it; it says what the answers assume.) -/
def P0 : Platform where
  math fn _ x r := (fn = .exp ∧ r = expRn x) ∨ (fn = .log ∧ r = logRn x)
  div _ x y r := r = FloatV.divRn x y
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- The oracle: `tl.exp`, `tl.log` and `/` lane by lane, correctly rounded; nothing else. -/
def gdnOracle : Oracle := fun _ _ q => match q with
  | .plat (.math .exp _ x) => some (x.map expRn)
  | .plat (.math .log _ x) => some (x.map logRn)
  | .plat (.div _ x y) => some (Tensor.map2 FloatV.divRn x y)
  | _ => none

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

#guard (expRn (FloatV.ofRat 0 : F32)).toRat == 1 && (logRn (FloatV.ofRat 1 : F32)).toRat == 0 &&
  (expRn (FloatV.negInf : F32)).toRat == 0 && (expRn (FloatV.ofRat 100 : F32)).isInf &&
  (expRn (FloatV.ofRat (-32) : F32)).toRat < 1 / 2 ^ 46

/-! ## `Post`, decided on the bounds -/

/-- `g`'s condition at `A`, `x` (exact) for the final `c` (finite branch only: a `-∞` here would
be reported). `sp = x` above `20`, else `ln(1 + e^x)`, bounded. -/
def gOk (A x : Rat) (c : F32) : Bool :=
  let (elo, ehi) := expBounds A
  let (splo, sphi) : Rat × Rat :=
    if x ≤ 20 then let (xlo, xhi) := expBounds x; (lnLo (1 + xlo), lnHi (1 + xhi)) else (x, x)
  let tolLo := (1 / 2 ^ 23 : Rat) * (64 + min |A| 128) * (elo + 1 / 2 ^ 121) * (splo + 1 / 2 ^ 6)
  c.isFinite && 0 < elo && 0 ≤ splo &&
    max |c.toRat + elo * splo| |c.toRat + ehi * sphi| ≤ tolLo

/-- `beta_output`'s condition at `b` for the final `c`: `|c − σ| ≤ 2^-7·σ + 2^-133` at both ends
of `σ`'s bounds. -/
def bOk (b : Rat) (c : F32) : Bool :=
  let (elo, ehi) := expBounds (-b)
  c.isFinite && [1 / (1 + ehi), 1 / (1 + elo)].all fun s =>
    |c.toRat - s| ≤ (1 / 2 ^ 7 : Rat) * s + 1 / 2 ^ 133

/-! ## The cases -/

/-- One case: `batch`, `num_heads`, the layout, and the inputs as rationals (`A_log[h]` exact in
`float32`, the others in `bfloat16`; checked below). -/
structure Case where
  batch : Nat
  H : Nat
  split : Bool
  Alog : Nat → Rat
  bias : Nat → Rat
  av : Nat → Nat → Rat
  bv : Nat → Nat → Rat

def Case.inst (c : Case) : Inst := Claims.GdnGating.mk c.batch c.H c.split

/-- Sentinels: in every element of `g` and `beta_output` and `pad` elements after each (and, before
the rows are written, in all of `a`'s and `b`'s storage). -/
def sentG : Rat := -99
def sentB : Rat := -77
def sentIn : Rat := 55
def pad : Nat := 8

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .f32 ((List.range c.H).map fun h : Nat =>
    ((d.A_log.ptr h).addr, (FloatV.ofRat (c.Alog h) : F32)))
  let m := CMem.ofElems m .bf16 ((List.range c.H).map fun h : Nat =>
    ((d.dt_bias.ptr h).addr, (FloatV.ofRat (c.bias h) : BF16)))
  -- `a`'s and `b`'s whole storage: the sentinel first, then the rows
  let m := CMem.ofElems m .bf16 ((List.range d.b.len).map fun k : Nat =>
    ((d.b.ptr k).addr, (FloatV.ofRat sentIn : BF16)))
  let m := CMem.ofElems m .bf16 ((List.range d.a.len).map fun k : Nat =>
    ((d.a.ptr k).addr, (FloatV.ofRat sentIn : BF16)))
  let rows := (List.range c.batch).flatMap fun i => (List.range c.H).map fun h => (i, h)
  let m := CMem.ofElems m .bf16 (rows.map fun (i, h) =>
    ((d.b.ptr (i * d.stride_b + h : Nat)).addr, (FloatV.ofRat (c.bv i h) : BF16)))
  let m := CMem.ofElems m .bf16 (rows.map fun (i, h) =>
    ((d.a.ptr (i * d.stride_a + h : Nat)).addr, (FloatV.ofRat (c.av i h) : BF16)))
  let m := CMem.ofElems m .f32 ((List.range (c.batch * c.H + pad)).map fun k : Nat =>
    ((d.g.ptr k).addr, (FloatV.ofRat sentG : F32)))
  CMem.ofElems m .f32 ((List.range (c.batch * c.H + pad)).map fun k : Nat =>
    ((d.bout.ptr k).addr, (FloatV.ofRat sentB : F32)))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) gdnOracle fill0 1000 (model c.inst) (launch c.inst) c.init

/-- The bytes of the written elements of `g` and `beta_output` (`batch·H` each). -/
def Case.written (c : Case) (a : Int) : Bool :=
  let d := c.inst
  let n : Nat := c.batch * c.H * 4
  (d.g.base ≤ a && a < d.g.base + n) || (d.bout.base ≤ a && a < d.bout.base + n)

/-- The final memory against `Post`, with the expectations computed from inputs `e` (`e = c` for
the test). -/
def Case.agreeE (c e : Case) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let ks := List.range (c.batch * c.H)
  let badG := ks.filter fun k : Nat =>
    match m.read .f32 (d.g.ptr k).addr with
    | some v => !gOk (e.Alog (k % c.H)) (e.av (k / c.H) (k % c.H) + e.bias (k % c.H)) v
    | none => true
  let badB := ks.filter fun k : Nat =>
    match m.read .f32 (d.bout.ptr k).addr with
    | some v => !bOk (e.bv (k / c.H) (k % c.H)) v
    | none => true
  let changed := c.init.toList.filter fun (a, b) => !c.written a && cf.mem[a]? != some b
  if !badG.isEmpty then some s!"g elements {badG.take 8} differ from Post"
  else if !badB.isEmpty then some s!"beta_output elements {badB.take 8} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the written elements changed"
  else if cf.mem.size != c.init.size then some "the launch wrote outside the initial memory"
  else none

def Case.agree (c : Case) := c.agreeE c

/-- `-1.5·2^127`, a `bfloat16`: two of them sum to `-3·2^127`, which rounds to `-∞` in `float32`. -/
def big : Rat := 3 * 2 ^ 126

/-- `batch = 3`, `num_heads = 5` (one head block, three masked lanes), contiguous; `A_log = 0`.
`x = a + dt_bias` by head: `21, -32, 64`; `-∞, 2^125, -∞`; `-39, 25, 100`; `22, -38, -108`;
`33, -47.5, 20.5`. `b` in `{0, 17, 20, 32, 100}`. Every output is exact and forced. -/
def exact : Case :=
  { batch := 3, H := 5, split := false,
    Alog := fun _ => 0,
    bias := fun h => [0, -big, 1, -8, 1 / 2].getD h 0,
    av := fun i h => ([[21, -big, -40, 30, 65 / 2], [-32, 7 * 2 ^ 125, 24, -30, -48],
      [64, -big, 99, -100, 20]].getD i []).getD h 0,
    bv := fun i h => ([[0, 17, 20, 32, 100], [100, 0, 32, 17, 20], [20, 32, 0, 100, 17]].getD i []).getD h 0 }

/-- `batch = 2`, `num_heads = 10` (two head blocks, the second with six masked lanes), split (row
stride `20`, `a` the second half); realistic `A_log` (`ln A`, `A` from `0.05` to `15.6`),
`x` from `-26` to `42`, with `x = 20` twice (the threshold: the `log` branch) and `19.5`, `b` from `-88`
to `10`. -/
def general : Case :=
  { batch := 2, H := 10, split := true,
    Alog := fun h => [1 / 2, -5 / 4, 11 / 4, 1, 0, -3, 1 / 8, 2, 3 / 2, -1 / 2].getD h 0,
    bias := fun h => [1 / 2, -1, 0, 2, -1 / 4, 3 / 2, 0, -2, 3 / 4, 3].getD h 0,
    av := fun i h => ([[-7 / 2, 1 / 4, 1, 11 / 2, -15, 37 / 2, 20, -24, 0, 39],
      [-1 / 2, 2, -6, 12, 1 / 8, -1, 39 / 2, 40, -18, -4]].getD i []).getD h 0,
    bv := fun i h => ([[-5 / 2, 3 / 4, 3, -8, 0, 5 / 4, -1 / 8, 6, -30, 10],
      [4, -1, 1 / 2, -60, 2, -3, 15 / 2, -3 / 4, 1, -88]].getD i []).getD h 0 }

theorem exact_ok : exact.inst.ok := mk_ok 3 5 false (by norm_num) (by norm_num) (by norm_num) (by norm_num)
theorem general_ok : general.inst.ok := mk_ok 2 10 true (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def cases : List (String × Case) := [("exact", exact), ("general", general)]

-- `Pre`: the six buffers are aligned and inside `[0, 2^64)` (`mk`'s addresses), every byte of
-- them present; the inputs are encoded exactly (finite); `A_log ≤ 88` and `b ≥ -88` (so `ExpRoom`
-- holds: the claim's `ExpRoom` holds for every `t ≤ 88.72282`); `a + dt_bias < 2^128 − 2^103`.
#guard cases.all fun (_, c) =>
  let d := c.inst
  let m := CMem.toMem c.init
  let rows := (List.range c.batch).flatMap fun i => (List.range c.H).map fun h => (i, h)
  let f32ok (b : Buf .f32) := b.base % 4 == 0 && 0 ≤ b.base && b.base + b.len * 4 ≤ 2 ^ 64 &&
    (List.range b.len).all fun k : Nat => (m.read .f32 (b.ptr k).addr).isSome
  let bf16ok (b : Buf .bf16) := b.base % 2 == 0 && 0 ≤ b.base && b.base + b.len * 2 ≤ 2 ^ 64 &&
    (List.range b.len).all fun k : Nat => (m.read .bf16 (b.ptr k).addr).isSome
  f32ok d.A_log && f32ok d.g && f32ok d.bout && bf16ok d.a && bf16ok d.b && bf16ok d.dt_bias &&
  (List.range c.H).all (fun h => let v : F32 := FloatV.ofRat (c.Alog h)
    v.isFinite && v.toRat == c.Alog h && c.Alog h ≤ 88 &&
    let w : BF16 := FloatV.ofRat (c.bias h); w.isFinite && w.toRat == c.bias h) &&
  rows.all fun (i, h) =>
    let a : BF16 := FloatV.ofRat (c.av i h)
    let b : BF16 := FloatV.ofRat (c.bv i h)
    a.isFinite && a.toRat == c.av i h && b.isFinite && b.toRat == c.bv i h && -88 ≤ c.bv i h &&
      c.av i h + c.bias h < 2 ^ 128 - 2 ^ 103 &&
      (m.read .bf16 (d.a.ptr (i * d.stride_a + h : Nat)).addr).map (·.toRat) == some (c.av i h) &&
      (m.read .bf16 (d.b.ptr (i * d.stride_b + h : Nat)).addr).map (·.toRat) == some (c.bv i h)

-- The edges the cases claim: in `exact`, two sums round to `-∞` in `float32` and one is `2^125`;
-- in `general`, one `x` is exactly `20`.
#guard (FloatV.add (FloatV.ofRat (-big) : F32) (FloatV.ofRat (-big))).isInf &&
  (exact.av 1 1 + exact.bias 1 == 2 ^ 125) && (general.av 0 6 + general.bias 6 == 20)

-- The sentinels are outside every tolerance (an unwritten element would be caught).
#guard cases.all fun (_, c) => (List.range c.batch).all fun i => (List.range c.H).all fun h =>
  !gOk (c.Alog h) (c.av i h + c.bias h) (FloatV.ofRat sentG) && !bOk (c.bv i h) (FloatV.ofRat sentB)

/-- A wrong expectation: `b[0, 0]` taken as `b + 1`, and `A_log[1]` as `A_log + 1/4`. -/
def Case.wrong (c : Case) : Case :=
  { c with bv := fun i h => if i = 0 ∧ h = 0 then c.bv i h + 1 else c.bv i h,
           Alog := fun h => if h = 1 then c.Alog h + 1 / 4 else c.Alog h }

/-- Each case's verdict, and the wrong expectation's verdict on the same run. -/
def results : List (String × Case × Verdict × Verdict) := cases.map fun (n, c) =>
  let r := c.run
  (n, c, Verdict.ofLaunch c.agree r, Verdict.ofLaunch (c.agreeE c.wrong) r)

#eval results.forM fun (n, _, v, _) => IO.println (v.line s!"gdn_gating_bf16/{n}")
-- Agreement resting on five assumed answers per program (`batch · cdiv(H, 8)` programs); and not
-- vacuous: the wrong expectation is a discrepancy.
#guard results.all fun (_, c, v, w) => v == .conditional (5 * c.batch * ((c.H + 7) / 8)) && w.isDiscrepancy

-- In `exact`, the outputs are the forced exact values: `g = -x` above the threshold, `-0` (or `+0`)
-- otherwise, and `beta_output` `1/2` at `b = 0`, `1` elsewhere.
#guard match exact.run with
  | (.done cf, _) =>
    let m := CMem.toMem cf.mem
    let d := exact.inst
    (List.range (exact.batch * exact.H)).all fun k : Nat =>
      let x := exact.av (k / exact.H) (k % exact.H) + exact.bias (k % exact.H)
      (m.read .f32 (d.g.ptr k).addr).map (·.toRat) == some (if x > 20 then -x else 0) &&
      (m.read .f32 (d.bout.ptr k).addr).map (·.toRat) ==
        some (if exact.bv (k / exact.H) (k % exact.H) = 0 then 1 / 2 else 1)
  | _ => false
