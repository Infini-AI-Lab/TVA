import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `moe_router_bf16` run by `execLaunch` on two small covered instances.
Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and that
the final memory is what `Post` says on these inputs, both halves: each token's `topk_ids` and
`topk_weights` (`TokenPost`), and every other byte unchanged (`x`, the gate, sentinels on both sides
of each output buffer).

**The inputs.** `x[t, h]` (`bfloat16`) and `W[e, h]` (`float32`) are small integers. Lanes
`h < 8` carry the logits: `W[e, h] = [e = h]` and `x[t, h]` the token's chosen logit of expert
`h`; the lanes `h ≥ 8` come in pairs `(2i, 2i + 1)` with equal `x` and opposite `W`, so they cancel
exactly (and make `S_e = Σ_h |x·W|` nonzero). Every product and every partial sum is an integer of
magnitude below `2^10`: the `float32` logit is exactly `μ_e`, whatever the tree of the fused row sum
(`tl.sum1c`; the solver takes its default tree within `Device.SumDepth`'s depth).

**The platform answers assumed** (the oracle `O`): six per program, `/ 30`, `exp`, `top / bottom`,
`exp` of the differences, `1 / den`, `exp` of `top2_v − top1_v`. A division is answered with the
IEEE quotient (`FloatV.divRn`), an `exp` with Lean's `Float.exp` (double) at the operand, rounded
to `float32`: values within `Device.DivFull` and `Device.ExpF32` (correctly rounded, or one double
rounding away from it). On `zeros` every logit is `0`, and every answer is exact and the only one
any correct device gives (`0 / 30 = 0`, `exp 0 = 1`, `0 / 2 = 0`, `1 / 8`); on `mixed` the
answers are the correctly rounded values (up to that double rounding), not forced.

**The check of `TokenPost`.** `Post` asks for reals `C e` within `capEps` of `capRef μ_e`; the
test takes `C e = capRef μ_e` itself (distance `0`). Since `capRef` is strictly increasing, `i0`
and `i1` are then the first index of the largest `μ` and the first index of the largest `μ` among
the others, computed here from the integers `μ`. The weights must be finite and within
`2^-14·softRef + 2^-130` of `softRef C i0 i0`, `softRef C i0 i1`: on `zeros` that is `1/8` exactly
(checked as rationals, equality); on `mixed`, `softRef` is evaluated in double (`Float.tanh`,
`Float.exp`, relative error below `10^-14`), and the test asks for `|a − ref| + 10^-12·ref ≤
2^-14·ref`, a stricter check than `Post`'s (the floor `2^-130` dropped, a margin for the double
evaluation). Agreement with this witness `C` is sufficient for `TokenPost`; a failure would only
show that this witness does not work.

The conditions: `Inst.ok` by `mk_ok`; `Pre`'s decidable parts guarded below (four buffers aligned
and every element present, inputs finite, `S_e ≤ 2^120`, `μ_e + capTol ≤ 1320`: here
`|μ_e| ≤ 30` and `S_e < 2^12`, so `capTol < 1` and `μ_e + capTol < 1320`). -/

open TVA TVA.Validation Claims.MoeRouter

/-- A platform that relates nothing: the platform answers come from the oracle, never from `P0`. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle -/

/-- A float as a Lean `Float` (exactly: through `float64`). -/
def toFloat {f : Format} (x : FloatV f) : Float :=
  Float.ofBits (UInt64.ofNat (FloatV.cvt x : F64).pat)

/-- A Lean `Float` rounded to the format `f` (to nearest even). -/
def ofFloat {f : Format} (y : Float) : FloatV f :=
  FloatV.cvt (FloatV.ofPat y.toBits.toNat : F64)

/-- `exp` answered in double then rounded; `/` answered with the IEEE quotient; nothing else. -/
def O : Oracle := fun _ _ q => match q with
  | .plat (.math .exp _ x) => some (x.map fun a => ofFloat (Float.exp (toFloat a)))
  | .plat (.div _ x y) => some (Tensor.map2 FloatV.divRn x y)
  | _ => none

/-! ## The cases -/

/-- A sentinel in both outputs before the launch, and around them. -/
def sentinel : Int := -7

/-- Sentinel elements on each side of each output buffer. -/
def pad : Nat := 4

/-- One case: the claim's `bs`, `hidden_dim`, `BLOCK_SIZE`, and the chosen logits of each token. -/
structure Case where
  bs : Nat
  hidden : Nat
  B : Nat
  /-- `mu t e`: token `t`'s logit of expert `e`, a small integer. -/
  mu : Nat → Nat → Int
  /-- All logits `0`: the answers and the weights are exact. -/
  exact : Bool

def Case.inst (c : Case) : Inst := Claims.MoeRouter.mk c.bs c.hidden c.B

/-- `x[t, h]`: the logit lanes `h < 8`, then pairs of equal values `1 … 3`. -/
def Case.xv (c : Case) (t h : Nat) : Int :=
  if h < 8 then c.mu t h else ((h / 2 + t) % 3 : Nat) + 1

/-- `W[e, h]`: the identity on the logit lanes, then pairs of opposite values in `-1 … 2`. -/
def wv (e h : Nat) : Int :=
  if h < 8 then (if e = h then 1 else 0)
  else (if h % 2 = 0 then 1 else -1) * (((h / 2 + e) % 4 : Nat) - (1 : Int))

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .bf16 ((List.range (c.bs * c.hidden)).map fun j : Nat =>
    ((d.x.ptr j).addr, (FloatV.ofRat (c.xv (j / c.hidden) (j % c.hidden) : Rat) : BF16)))
  let m := CMem.ofElems m .f32 ((List.range (8 * c.hidden)).map fun j : Nat =>
    ((d.w.ptr j).addr, (FloatV.ofRat (wv (j / c.hidden) (j % c.hidden) : Rat) : F32)))
  let m := CMem.ofElems m .f32 ((List.range (2 * c.bs + 2 * pad)).map fun j : Nat =>
    ((d.outw.ptr ((j : Int) - pad)).addr, (FloatV.ofRat (sentinel : Rat) : F32)))
  CMem.ofElems m .i32 ((List.range (2 * c.bs + 2 * pad)).map fun j : Nat =>
    ((d.outi.ptr ((j : Int) - pad)).addr, (IntV.ofInt sentinel : I32)))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) O fill0 1000 (model c.inst) (launch c.inst) c.init

/-- `μ_e`, computed from the inputs (as integers). -/
def Case.muOf (c : Case) (t e : Nat) : Int :=
  ((List.range c.hidden).map fun h => c.xv t h * wv e h).sum

/-- `S_e = Σ_h |x·W|`. -/
def Case.absOf (c : Case) (t e : Nat) : Int :=
  ((List.range c.hidden).map fun h => (c.xv t h * wv e h).natAbs).sum

/-- The first index of the largest `μ` among the experts `e ∉ skip`. -/
def firstMax (mu : Nat → Int) (skip : List Nat) : Nat :=
  (List.range 8).foldl (fun b e =>
    if skip.contains e then b else if skip.contains b || mu e > mu b then e else b)
    (if skip.contains 0 then 1 else 0)

/-- `capRef μ = 30·tanh(μ/30)` in double. -/
def capF (mu : Int) : Float := 30 * Float.tanh (Float.ofInt mu / 30)

/-- `softRef C i0 j` in double, for `C e = capRef μ_e`. -/
def softF (mu : Nat → Int) (i0 j : Nat) : Float :=
  let z := ((List.range 8).map fun k => Float.exp (capF (mu k) - capF (mu i0))).foldl (· + ·) 0
  Float.exp (capF (mu j) - capF (mu i0)) / z

/-- A weight against `Post`'s bound, stricter: `|a − ref| + 10^-12·ref ≤ 2^-14·ref`. -/
def weightOk (a : F32) (ref : Float) : Bool :=
  a.isFinite && (toFloat a - ref).abs + 1e-12 * ref ≤ ref / 16384

/-- Token `t`'s outputs against `TokenPost` for the expected logits `mu` (`= c.muOf` for the
test). -/
def Case.tokenOk (c : Case) (mu : Nat → Int) (m : Mem) (t : Nat) : Bool :=
  let d := c.inst
  let i0 := firstMax mu []
  let i1 := firstMax mu [i0]
  let idsOk := m.read .i32 (d.outi.ptr (2 * t : Nat)).addr == some (IntV.ofNat i0) &&
    m.read .i32 (d.outi.ptr (2 * t + 1 : Nat)).addr == some (IntV.ofNat i1)
  let wOk := match m.read .f32 (d.outw.ptr (2 * t : Nat)).addr,
      m.read .f32 (d.outw.ptr (2 * t + 1 : Nat)).addr with
    | some a0, some a1 =>
      if c.exact then a0.isFinite && a1.isFinite && a0.toRat == 1 / 8 && a1.toRat == 1 / 8
      else weightOk a0 (softF mu i0 i0) && weightOk a1 (softF mu i0 i1)
    | _, _ => false
  idsOk && wOk

/-- The final memory against `Post`, with the expected logits `mu t`: every token's outputs, then
every byte outside the `2·bs` elements of each output unchanged, and no byte added. -/
def Case.agreeE (c : Case) (mu : Nat → Nat → Int) (cf : CConfig (launch c.inst).size) :
    Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let bad := (List.range c.bs).filter fun t => !c.tokenOk (mu t) m t
  let inOut (a : Int) : Bool :=
    (d.outw.base ≤ a && a < d.outw.base + (2 * c.bs * 4 : Nat)) ||
    (d.outi.base ≤ a && a < d.outi.base + (2 * c.bs * 4 : Nat))
  let changed := c.init.toList.filter fun (a, b) => !inOut a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"tokens {bad}: outputs differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the outputs changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c.muOf

/-- One token, `hidden_dim = BLOCK_SIZE = 128` (no masked lane), every logit `0`: the eight capped
logits tie, `topk_ids = [0, 1]` (`tie_break_left`), both weights exactly `1/8`. -/
def zeros : Case := { bs := 1, hidden := 128, B := 128, mu := fun _ _ => 0, exact := true }

/-- Two tokens, `hidden_dim = 128` in a block of `256` (lanes `128 … 255` masked). Token 0's two
largest logits tie (`12` at experts 2 and 4: `i0 = 2`, `i1 = 4`); token 1's are distinct (`25` at
5, `18` at 3, neither at an end). -/
def mixed : Case :=
  { bs := 2, hidden := 128, B := 256, exact := false,
    mu := fun t e => if t = 0 then [3, -5, 12, 0, 12, 7, -20, 1].getD e 0
      else [-2, 9, 4, 18, -7, 25, 0, 11].getD e 0 }

theorem zeros_ok : zeros.inst.ok := mk_ok 1 128 128 7 rfl (by decide) (by decide)
theorem mixed_ok : mixed.inst.ok := mk_ok 2 128 256 8 rfl (by decide) (by decide)

def cases : List (String × Case) := [("zeros", zeros), ("mixed", mixed)]

-- `Pre`, decidable parts: the four buffers aligned (and below `2^64`), every element of each
-- present in `init`; inputs finite; the logits are the chosen ones, `|μ_e| ≤ 30`, `S_e < 2^12`.
#guard cases.all fun (_, c) =>
  let d := c.inst
  let m := CMem.toMem c.init
  d.x.base % 2 == 0 && d.w.base % 4 == 0 && d.outw.base % 4 == 0 && d.outi.base % 4 == 0 &&
  (List.range d.x.len).all (fun j : Nat => (m.read .bf16 (d.x.ptr j).addr).any (·.isFinite)) &&
  (List.range d.w.len).all (fun j : Nat => (m.read .f32 (d.w.ptr j).addr).any (·.isFinite)) &&
  (List.range d.outw.len).all (fun j : Nat => (m.read .f32 (d.outw.ptr j).addr).isSome) &&
  (List.range d.outi.len).all (fun j : Nat => (m.read .i32 (d.outi.ptr j).addr).isSome) &&
  (List.range c.bs).all fun t => (List.range 8).all fun e =>
    c.muOf t e == c.mu t e && (c.mu t e).natAbs ≤ 30 && c.absOf t e < 2 ^ 12
-- The inputs are encoded exactly.
#guard cases.all fun (_, c) => (List.range c.bs).all fun t => (List.range c.hidden).all fun h =>
  (FloatV.ofRat (c.xv t h : Rat) : BF16).toRat == (c.xv t h : Rat)
-- The expected top-2 of `mixed`: ties broken to the left; the distinct order.
#guard firstMax (mixed.mu 0) [] == 2 && firstMax (mixed.mu 0) [2] == 4 &&
  firstMax (mixed.mu 1) [] == 5 && firstMax (mixed.mu 1) [5] == 3 &&
  firstMax (zeros.mu 0) [] == 0 && firstMax (zeros.mu 0) [0] == 1

def verdictZeros : Verdict := Verdict.ofLaunch zeros.agree zeros.run
def verdictMixed : Verdict := Verdict.ofLaunch mixed.agree mixed.run

#eval IO.println (verdictZeros.line "moe_router_bf16/zeros")
#eval IO.println (verdictMixed.line "moe_router_bf16/mixed")
-- Six assumed answers per program (token): the three divisions and the three `exp`s.
#guard verdictZeros == .conditional 6
#guard verdictMixed == .conditional 12

-- A wrong expectation is caught: `agree` is not vacuous (token 0 of `mixed` with expert 4's logit
-- raised to `13`, so `i0 = 4`, `i1 = 2`; `zeros` against `mixed`'s first token's logits, with the
-- weights checked by tolerance).
#guard
  let r := mixed.run
  (Verdict.ofLaunch (mixed.agreeE fun t e => if t = 0 ∧ e = 4 then 13 else mixed.mu t e) r).isDiscrepancy
#guard (Verdict.ofLaunch ({ zeros with exact := false }.agreeE fun _ => mixed.mu 0) zeros.run).isDiscrepancy
