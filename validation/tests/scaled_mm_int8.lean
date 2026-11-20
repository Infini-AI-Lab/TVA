import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `scaled_mm_int8` run by `execLaunch` on small covered instances. Not
a test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says on these inputs.

Three instances, built with the claim's `hostInst`: the default tile `(32, 32, 32)` with
per-tensor scales, no bias, a `bfloat16` output and a contiguous `B` (two programs, a ragged
column tile); the heuristic's `(64, 64, 256)` tile with per-row and per-column scales, a `float16`
bias and output and the transposed `B`; the heuristic's `(64, 128, 128)` tile over two row tiles
(`M = 65`) with scales that are not powers of two, a `bfloat16` bias and output, and `int8`
operands over the whole range `[-128, 127]`.

**The `tl.dot` values are assumed, not proved.** The `int8 × int8 → int32` `tl.dot` is a platform
operation (`PlatQ.dot`): only the oracle can supply its value. The oracle answers each with the
exact `acc + a·b`, computed from the request's own operands, when it fits in `int32` (it answers
nothing otherwise); every partial sum here is at most `128·128·128 = 2^21` in magnitude, so this is
the value `Device.DotI8` forces (the dot is exact when `|acc| + Σ|a·b| ≤ 2^31 − 1`). The
verdicts are `conditional n`, `n` the number of dots (programs × `K / BLOCK_SIZE_K`). Every other
operation (`int32 → float32`, the two scale products, the bias addition, the rounding to the
output dtype) is exact or IEEE and computed by the solvers.

The check is `Post`'s, element by element, computed in the test from the inputs: with
`S = Σ_k A[i, k]·B[k, j]` (exact), `sa`, `sb` the row's and column's `float32` scales and `β` the
bias (`0` without one), `R = sa·sb·S + β`, the element must be the output-dtype rounding of some
finite `float32` `v` with `|v − R| ≤ tol` (searched among the 129 `float32` values nearest
`R`), and, when `|R| + tol ≤ outMax`, finite and within `outTol` of `R`. `tol`, `outTol` and
`outMax` are the claim's (`Post` names them). On the first two instances every intermediate value is
exact (dyadic scales, small integers), and the test also asks `c = R` exactly. `Room32` (the
hypothesis of `OutSpec`) holds: every scale is at most `4`, every bias at most `2^10`, every `|S|`
at most `2^21`. Every byte outside the matrix `C` (inputs, scales, bias, sentinels around and in
`C`'s storage) is unchanged. -/

open TVA TVA.Validation

namespace ScaledMmTest

open Claims.ScaledMmInt8

/-- A platform that relates nothing: the dots are answered by the oracle, never by `P0`. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle: exact integer dots -/

/-- An element as an integer, when it is an integer (or a float with an integer value). -/
def intOf (d : DType) (x : d.Val) : Option Int :=
  match d.val? x with
  | some q => if q.den == 1 then some q.num else none
  | none => none

/-- An integer as an accumulator element, when the accumulator is `int32` and holds it. -/
def accOf : (d : DType) → Int → Option d.Val
  | .i32, z => if -2 ^ 31 ≤ z ∧ z < 2 ^ 31 then some (IntV.ofInt z) else none
  | _, _ => none

/-- The exact `acc + a·b` of a `tl.dot` request, from its own operands: `none` unless every
operand is an integer and every result fits in the `int32` accumulator. -/
def exactDot : (q : PlatQ) → Option q.Out
  | @PlatQ.dot cfg M N K a b acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => a.get2 i k).toArray.mapM (intOf cfg.a)
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => b.get2 k j).toArray.mapM (intOf cfg.b)
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM (intOf cfg.acc)
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      accOf cfg.acc ((List.range K).foldl (fun s k => s + A[i * K + k]! * B[k * N + j]!) C[e]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- The oracle: the exact dot at every platform request; nothing else. -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactDot q
  | _ => none

/-! ## The cases -/

/-- A value of the output dtype (`float16` or `bfloat16`), rounded to nearest even. -/
def odv : (od : DType) → Rat → od.Val
  | .f16, q => (FloatV.ofRat q : F16)
  | .bf16, q => (FloatV.ofRat q : BF16)
  | d, _ => d.ofBits 0

/-- `c` is the rounding of the `float32` `v` to the output dtype, up to NaN payload (`OutIs`). -/
def cvtIs : (od : DType) → F32 → od.Val → Bool
  | .f16, v, c => let r : F16 := FloatV.cvt v; r == c || (r.isNaN && c.isNaN)
  | .bf16, v, c => let r : BF16 := FloatV.cvt v; r == c || (r.isNaN && c.isNaN)
  | _, _, _ => false

/-- The sentinel in `C`'s storage and around it. -/
def sentinel : Rat := -1 / 1024

/-- Sentinel elements on each side of `C`'s storage. -/
def pad : Nat := 4

/-- One case: an instance from `hostInst`, the values of `A` and `B`, the scales (element `e` of
each scale buffer), the bias (element `j`), and whether every intermediate value is exact. -/
structure Case where
  d : Inst
  av : Nat → Nat → Int
  bv : Nat → Nat → Int
  saE : Nat → F32
  sbE : Nat → F32
  biasE : Nat → Rat
  exact : Bool

def i8v (z : Int) : I8 := IntV.ofInt z

/-- The inputs: `A`, `B` at the elements their strides give, the scales, the bias, and `C`'s whole
storage and the elements around it holding the sentinel. -/
def Case.init (c : Case) : CMem :=
  let d := c.d
  let m := CMem.ofElems ∅ .i8 ((List.range d.M).flatMap fun i : Nat => (List.range d.K).map fun k : Nat =>
    ((d.a.ptr (i * d.stride_am + k * d.stride_ak : Nat)).addr, i8v (c.av i k)))
  let m := CMem.ofElems m .i8 ((List.range d.K).flatMap fun k : Nat => (List.range d.N).map fun j : Nat =>
    ((d.b.ptr (k * d.stride_bk + j * d.stride_bn : Nat)).addr, i8v (c.bv k j)))
  let m := CMem.ofElems m .f32 ((List.range d.sa.len).map fun e : Nat => ((d.sa.ptr e).addr, c.saE e))
  let m := CMem.ofElems m .f32 ((List.range d.sb.len).map fun e : Nat => ((d.sb.ptr e).addr, c.sbE e))
  let m := if d.hasBias then
      CMem.ofElems m d.od ((List.range d.bias.len).map fun e : Nat => ((d.bias.ptr e).addr, odv d.od (c.biasE e)))
    else m
  CMem.ofElems m d.od ((List.range (d.c.len + 2 * pad)).map fun e : Nat =>
    ((d.c.ptr ((e : Int) - pad)).addr, odv d.od sentinel))

def Case.run (c : Case) : LRes (launch c.d).size × List Record :=
  execLaunch ((launch c.d).env P0) O fill0 100000 (model c.d) (launch c.d) c.init

/-- `S`, `sa`, `sb`, `β` of element `(i, j)`, from the inputs. -/
def Case.S (c : Case) (i j : Nat) : Rat := (((List.range c.d.K).map fun k => c.av i k * c.bv k j).sum : Int)
def Case.sa (c : Case) (i : Nat) : Rat := (c.saE (if c.d.saVec then i else 0)).toRat
def Case.sb (c : Case) (j : Nat) : Rat := (c.sbE (if c.d.sbVec then j else 0)).toRat
def Case.beta (c : Case) (j : Nat) : Rat :=
  if c.d.hasBias then (odv c.d.od (c.biasE j) |> c.d.od.val?).getD 0 else 0

/-- `R = sa·sb·S + β`. -/
def Case.R (c : Case) (i j : Nat) : Rat := c.sa i * c.sb j * c.S i j + c.beta j

/-- `OutSpec`'s conclusion for element `(i, j)` holding `x`, on these inputs; with `exact`, also
`x = R`. -/
def Case.elemOk (c : Case) (R : Nat → Nat → Rat) (i j : Nat) (x : c.d.od.Val) : Bool :=
  let r := R i j
  let S := c.S i j
  let t := tol (c.sa i) (c.sb j) S (c.beta j)
  let v0 : F32 := FloatV.ofRat r
  -- clause 1: some finite `float32` within `tol` of `R` rounds to `x`
  let c1 := (List.range 129).any fun δ : Nat =>
    let p : Int := (v0.pat : Int) + δ - 64
    if p < 0 ∨ 2 ^ 32 ≤ p then false else
      let v : F32 := FloatV.ofPat p.toNat
      v.isFinite && |v.toRat - r| ≤ t && cvtIs c.d.od v x
  -- clause 2: in range, finite and within `outTol`
  let c2 := if |r| + t ≤ outMax c.d.od then
      match c.d.od.val? x with
      | some q => |q - r| ≤ outTol c.d.od (c.sa i) (c.sb j) S (c.beta j)
      | none => false
    else true
  let c3 := !c.exact || c.d.od.val? x == some r
  c1 && c2 && c3

/-- The final memory against `Post`: every element of the matrix `C` as `elemOk` says (for the
expectation `R`); every byte outside it unchanged, and no byte added. -/
def Case.agree (c : Case) (R : Nat → Nat → Rat) (cf : CConfig (launch c.d).size) : Option String :=
  let d := c.d
  let m := CMem.toMem cf.mem
  let cs := (List.range d.M).flatMap fun i : Nat => (List.range d.N).map fun j : Nat =>
    (i, j, (d.c.ptr (i * d.stride_cm + j * d.stride_cn : Nat)).addr)
  let badOut := cs.filter fun (i, j, a) => match m.read d.od a with
    | some x => !c.elemOk R i j x
    | none => true
  let cBytes : Std.HashSet Int := cs.foldl (fun s (_, _, a) => (s.insert a).insert (a + 1)) {}
  let init := c.init
  let badElse := init.toList.filter fun (a, b) => !cBytes.contains a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"C elements {(badOut.take 8).map fun (i, j, _) => (i, j)} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside the matrix C changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) (R : Nat → Nat → Rat) : Verdict := Verdict.ofLaunch (c.agree R) c.run

def f32q (q : Rat) : F32 := FloatV.ofRat q

/-- `M = 3`, `N = 40`, `K = 128`, the default tile `(32, 32, 32)`: two programs (the second
column tile has `8` valid columns), four dots each; rows `3..31` masked. Per-tensor scales `1/2`
and `1/4`, no bias, `bfloat16` output, contiguous `B`. `|S| ≤ 256`: `R = S/8` is a `bfloat16`. -/
def plain : Case :=
  { d := hostInst 3 40 128 .bf16 false false false false false
    av := fun i k => ((2 * i + 3 * k) % 3 : Nat) - (1 : Int)
    bv := fun k j => ((7 * k + 3 * j + 1) % 5 : Nat) - (2 : Int)
    saE := fun _ => f32q (1 / 2), sbE := fun _ => f32q (1 / 4), biasE := fun _ => 0, exact := true }
theorem plain_ok : plain.d.ok := hostInst_ok (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by decide) (by decide)

/-- `M = 2`, `N = 16`, `K = 256`, the heuristic's tile `(64, 64, 256)`: one program, one dot.
Per-row scales `1`, `2`, per-column scales `1`, `1/2`, `1`, …, a `float16` bias `j − 8` and output,
`B` the transposed view (strides `(1, K)`). `R` has at most `11` significant bits. -/
def perRow : Case :=
  { d := hostInst 2 16 256 .f16 true true true true true
    av := fun i k => ((2 * i + 3 * k) % 3 : Nat) - (1 : Int)
    bv := fun k j => ((7 * k + 3 * j + 1) % 5 : Nat) - (2 : Int)
    saE := fun e => f32q (if e % 2 = 0 then 1 else 2), sbE := fun e => f32q (if e % 2 = 0 then 1 else 1 / 2)
    biasE := fun e => (e : Rat) - 8, exact := true }
theorem perRow_ok : perRow.d.ok := hostInst_ok (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num) (by decide) (by decide)

/-- `M = 65`, `N = 128`, `K = 128`, the heuristic's tile `(64, 128, 128)`: two programs (row
tiles; the second has one valid row), one dot each. `int8` values over `[-128, 127]`; per-row
scales `(1 + i % 3)/10` and the per-tensor scale `1/3` (rounded to `float32`), a `bfloat16` bias
`j/7` (rounded), `bfloat16` output: the results are rounded, checked against `Post`'s
tolerances. -/
def rounded : Case :=
  { d := hostInst 65 128 128 .bf16 true true false false true
    av := fun i k => ((37 * i + 11 * k) % 256 : Nat) - (128 : Int)
    bv := fun k j => ((13 * k + 29 * j + 5) % 256 : Nat) - (128 : Int)
    saE := fun e => f32q ((1 + (e % 3 : Nat)) / 10), sbE := fun _ => f32q (1 / 3)
    biasE := fun e => (e : Rat) / 7, exact := false }
theorem rounded_ok : rounded.d.ok := hostInst_ok (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by decide) (by decide)

end ScaledMmTest

open ScaledMmTest Claims.ScaledMmInt8

-- The tiles: `(32, 32, 32)`, `(64, 64, 256)`, `(64, 128, 128)`; grids of 2, 1, 2 programs.
#guard [plain, perRow, rounded].map (fun c => (c.d.BM, c.d.BN, c.d.BK, (launch c.d).size)) ==
  [(32, 32, 32, 2), (64, 64, 256, 1), (64, 128, 128, 2)]

-- `Pre`: every buffer the kernel accesses is aligned, inside `[0, 2^64)`, and every element of it
-- is present in `init`. And the inputs are the intended values (`int8` in range, scales finite).
#guard [plain, perRow, rounded].all fun c =>
  let d := c.d
  let m := CMem.toMem c.init
  let ok {dt : DType} (b : Buf dt) : Bool :=
    0 ≤ b.base && b.base + b.len * dt.bytes ≤ 2 ^ 64 && b.base % dt.bytes == 0 &&
    (List.range b.len).all fun e : Nat => (m.read dt (b.ptr e).addr).isSome
  ok d.a && ok d.b && ok d.sa && ok d.sb && ok d.c && (!d.hasBias || ok d.bias) &&
  ((List.range d.M).all fun i : Nat => (List.range d.K).all fun k : Nat => -128 ≤ c.av i k && c.av i k ≤ 127) &&
  ((List.range d.K).all fun k : Nat => (List.range d.N).all fun j : Nat => -128 ≤ c.bv k j && c.bv k j ≤ 127) &&
  ((List.range d.sa.len).all fun e => (c.saE e).isFinite) && ((List.range d.sb.len).all fun e => (c.sbE e).isFinite)

-- `Room32`'s sufficient bounds: scales at most `4`, biases at most `2^10`, `|S| ≤ 2^21`; and no
-- expectation is the sentinel.
#guard [plain, perRow, rounded].all fun c => (List.range c.d.M).all fun i : Nat => (List.range c.d.N).all fun j : Nat =>
  |c.sa i| ≤ 4 && |c.sb j| ≤ 4 && |c.beta j| ≤ 2 ^ 10 && |c.S i j| ≤ 2 ^ 21 && c.R i j != sentinel

-- The rounded case does round: some `R` is not a `bfloat16`.
#guard (List.range rounded.d.M).any fun i : Nat => (List.range rounded.d.N).any fun j : Nat =>
  ((FloatV.ofRat (rounded.R i j) : BF16).toRat != rounded.R i j)

def verdictPlain : Verdict := plain.verdict plain.R
def verdictPerRow : Verdict := perRow.verdict perRow.R
def verdictRounded : Verdict := rounded.verdict rounded.R

#eval IO.println (verdictPlain.line "scaled_mm_int8/plain")
#eval IO.println (verdictPerRow.line "scaled_mm_int8/per_row")
#eval IO.println (verdictRounded.line "scaled_mm_int8/rounded")
-- Agreement resting on one assumed answer per dot (programs × `K / BLOCK_SIZE_K`).
#guard let v := verdictPlain; v == .conditional 8 && v.agrees && v.assumed > 0
#guard let v := verdictPerRow; v == .conditional 1 && v.agrees && v.assumed > 0
#guard let v := verdictRounded; v == .conditional 2 && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (one element of `plain` off by `1/8`, the
-- smallest step of its exact results; the scales of `rounded` swapped row by row; `perRow` without its
-- bias).
#guard
  (Verdict.ofLaunch (plain.agree fun i j => if i = 2 ∧ j = 39 then plain.R i j + 1 / 8 else plain.R i j)
    plain.run).isDiscrepancy &&
  (Verdict.ofLaunch (perRow.agree fun i j => perRow.R i j - perRow.beta j) perRow.run).isDiscrepancy
#guard (Verdict.ofLaunch (rounded.agree fun i j => rounded.R ((i + 1) % rounded.d.M) j) rounded.run).isDiscrepancy
