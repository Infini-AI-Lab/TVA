import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `matmul_e5m2` run by `execLaunch` on small covered instances. Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says on these inputs.

The block sizes are the claim's (`128 × 256` tiles, `k` steps of `64`), and `Inst.ok` asks
`128 ∣ N`, `128 ∣ K`: the smallest covered instances already run one `128 × 256` tile per program.

**Two kinds of platform answers are assumed, not proved**, each answered by the oracle below:

* `tl.dot` of `float8e5` operands into `float32` (`PlatQ.dot`): the oracle answers each dot with
  the exact `acc + a·b`, computed from the request's own operands (in integers, scaled by
  `2^10`), and only when that value is a `float32` (it answers nothing otherwise). The inputs are `a[i, k] ∈ {-2, …, 2}` and
  `b[k, j] ∈ {-1, -1/2, 0, 1/2, 1}` (all exact in `float8e5`), `K ≤ 256`: every partial sum is a
  multiple of `1/2` of magnitude at most `512`, exact in `float32` (and the result, in `float16`).
  The exact value is the only one any correct device gives, whatever its order, chunking and
  rounding: `Post`'s tolerance is about inexact sums, and these have none.
* the conversion of `other=0.0` (the `float32` `+0.0`) to `float8e5` (a conversion into fp8,
  `TQuery.convert`), once per masked load: the oracle answers each lane with the `float8e5` value
  equal to its operand, when there is one (here `+0`, exactly representable, the only correct
  result). The loads' masks are all-true on these instances (`K` a multiple of `64`), so the
  `other` values are never read anyway.

The verdicts are therefore `conditional n`, `n = 3` per program and `k` step (one dot, two
conversions). Each output element is then exactly `Σ_k a[i, k]·b[k, j]`, which `Post` allows (the
`float32` value `v = S` at distance `0 ≤ tol`, its `float16` rounding exact), and every other byte
is unchanged (`a`, `b`, the padding of `c`'s storage, sentinels on both sides of `c`). -/

open TVA TVA.Validation Claims.MatmulE5m2

/-- A platform that relates nothing: the dots and fp8 conversions are answered by the oracle,
never by `P0`. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle: exact dots, exact conversions -/

/-- A rational as an accumulator element, when the accumulator is `float32` and holds it exactly. -/
def accOfQ : (d : DType) → Rat → Option d.Val
  | .f32, z => let v : F32 := FloatV.ofRat z; if v.toRat == z then some v else none
  | _, _ => none

/-- `q` as an integer multiple of `2^-s`, when it is one. -/
def scaled (s : Nat) (q : Rat) : Option Int :=
  let z := q * (2 ^ s : Nat)
  if z.den == 1 then some z.num else none

/-- The exact `acc + a·b` of a `tl.dot` request, from its own operands: `none` unless every
operand is a finite multiple of `2^-10` (the accumulator of `2^-20`: the sums are then computed
in integers, exactly) and every result an exact `float32`. -/
def exactDotQ : (q : PlatQ) → Option q.Out
  | @PlatQ.dot cfg M N K a b acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => a.get2 i k).toArray.mapM
      fun x => cfg.a.val? x >>= scaled 10
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => b.get2 k j).toArray.mapM
      fun x => cfg.b.val? x >>= scaled 10
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM
      fun x => cfg.acc.val? x >>= scaled 20
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      let z := (List.range K).foldl (fun s k => s + A[i * K + k]! * B[k * N + j]!) C[e]!
      accOfQ cfg.acc ((z : Rat) / (2 ^ 20 : Nat))
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- The device's conversion of `x` into `float8e5`, when `x` is a finite value `float8e5` holds
exactly (the only correct result); nothing otherwise. -/
def exactConv (d : DType) : (d' : DType) → d.Val → Option d'.Val
  | .f8e5, x => match d.val? x with
    | some q => let v : F8E5 := FloatV.ofRat q; if v.toRat == q then some v else none
    | none => none
  | _, _ => none

/-- The oracle: the exact dot at every platform request, the exact conversion into `float8e5`;
nothing else (trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactDotQ q
  | .convert d d' x => exactConv d d' x
  | _ => none

/-! ## The cases -/

/-- A sentinel in `c`'s storage before the launch, and in the elements just outside it: those
outside the matrix `c` must keep it; every matrix element must be overwritten. -/
def sentinel : Int := -999

/-- Sentinel elements on each side of `c`'s storage. -/
def pad : Nat := 4

/-- `a[i, k] ∈ {-2, …, 2}`. -/
def a0 (i k : Nat) : Int := ((3 * i + 5 * k + 1) % 5 : Nat) - (2 : Int)
/-- Twice `b[k, j]`: `b[k, j] = b2 k j / 2 ∈ {-1, -1/2, 0, 1/2, 1}`. -/
def b2 (k j : Nat) : Int := ((7 * k + 3 * j + 1) % 5 : Nat) - (2 : Int)

def e5v (q : Rat) : F8E5 := FloatV.ofRat q
def f16v (z : Int) : F16 := FloatV.ofRat (z : Rat)

/-- A filler in the elements of `a`'s and `b`'s storage outside the matrices (the padding of
`a`'s rows in `strided`): `Pre` asks every byte present, and they must be unchanged. -/
def filler : F8E5 := FloatV.ofRat 3

/-- The inputs of an instance, at the elements its strides give (the rest of `a`'s and `b`'s
storage holding the filler), and `c`'s whole storage and the sentinels around it holding the
sentinel. -/
def initOf (d : Inst) : CMem :=
  let m := CMem.ofElems ∅ .f8e5 ((List.range d.a.len).map fun e : Nat => ((d.a.ptr e).addr, filler))
  let m := CMem.ofElems m .f8e5 ((List.range d.b.len).map fun e : Nat => ((d.b.ptr e).addr, filler))
  let m := CMem.ofElems m .f8e5 ((List.range d.M).flatMap fun i : Nat => (List.range d.K).map fun k : Nat =>
    ((d.a.ptr (i * d.stride_am + k * d.stride_ak : Nat)).addr, e5v (a0 i k)))
  let m := CMem.ofElems m .f8e5 ((List.range d.K).flatMap fun k : Nat => (List.range d.N).map fun j : Nat =>
    ((d.b.ptr (k * d.stride_bk + j * d.stride_bn : Nat)).addr, e5v ((b2 k j : Rat) / 2)))
  CMem.ofElems m .f16 ((List.range (d.c.len + 2 * pad)).map fun e : Nat =>
    ((d.c.ptr ((e : Int) - pad)).addr, f16v sentinel))

def runOf (d : Inst) : LRes (launch d).size × List Record :=
  execLaunch ((launch d).env P0) O fill0 10000 (model d) (launch d) (initOf d)

/-- What `Post` says element `(i, j)` of `c` holds on these inputs: `Σ_k a[i, k]·b[k, j]`, exact. -/
def expectOf (d : Inst) (i j : Nat) : Rat :=
  (((List.range d.K).foldl (fun s k => s + a0 i k * b2 k j) 0 : Int) : Rat) / 2

/-- The final memory against `Post`: every element of the matrix `c` is the expectation; every
byte outside it (`a`, `b`, `c`'s padding, the sentinels) keeps its initial value, and no byte is
added. -/
def agreeOf (d : Inst) (init : CMem) (expect : Nat → Nat → Rat) (cf : CConfig (launch d).size) :
    Option String :=
  let m := CMem.toMem cf.mem
  let cElems := (List.range d.M).flatMap fun i : Nat => (List.range d.N).map fun j : Nat =>
    (i, j, (d.c.ptr (i * d.stride_cm + j * d.stride_cn : Nat)).addr)
  let badOut := cElems.filter fun (i, j, a) => (m.read .f16 a).map (·.toRat) != some (expect i j)
  let cBytes : Std.HashSet Int := cElems.foldl (fun s (_, _, a) => (s.insert a).insert (a + 1)) {}
  let badElse := init.toList.filter fun (a, b) => !cBytes.contains a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"c elements {(badOut.take 8).map fun (i, j, _) => (i, j)} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside the matrix c changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def verdictOf (d : Inst) (expect : Nat → Nat → Rat) : Verdict :=
  Verdict.ofLaunch (agreeOf d (initOf d) expect) (runOf d)

/-- `M = 3`, `N = 128`, `K = 128`, the host's layout (`a` contiguous, `b` the transpose of a
contiguous matrix, strides `(1, K)`): one program, two `k` steps; rows `3..127` of the tile masked
(their loads wrap to rows `% M`, their stores are disabled) and columns `128..255` masked (half a
column tile, as `128 ∣ N` allows). -/
def small : Inst := hostInst 3 128 128
theorem small_ok : small.ok := hostInst_ok (by norm_num)

/-- `M = 2`, `N = 384`, `K = 128`, other layouts: `b` row-major (`stride_bk = N`,
`stride_bn = 1`), `a`'s rows padded to `136` elements, `c`'s rows padded to `400` elements (the
padding must keep the sentinel). Two programs (two column tiles, the second half masked), two `k`
steps each. -/
def strided : Inst :=
  { M := 2, N := 384, K := 128, stride_am := 136, stride_ak := 1, stride_bk := 384, stride_bn := 1,
    stride_cm := 400, stride_cn := 1,
    a := ⟨2 ^ 32, 2 * 136⟩, b := ⟨2 ^ 34, 128 * 384⟩, c := ⟨2 ^ 36, 2 * 400⟩ }

theorem strided_ok : strided.ok := by
  have hcS : strided.c.elems (cIdx strided) ⊆ strided.c.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, j, hi, hj, rfl⟩; simp only [strided] at *; omega)
  have haS : strided.a.elems (aIdx strided) ⊆ strided.a.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, k, hi, hk, rfl⟩; simp only [strided] at *; omega)
  have hbS : strided.b.elems (bIdx strided) ⊆ strided.b.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨k, j, hk, hj, rfl⟩; simp only [strided] at *; omega)
  constructor
  all_goals try first
    | (norm_num [strided]; done)
    | (intro i j i' j' hi hj hi' hj' he; simp only [strided] at *; omega)
    | (intro i k hi hk; simp only [strided] at *; omega)
  · refine Disjoint.mono hcS haS (bytes_disjoint ?_)
    simp only [strided, DType.bytes]; omega
  · refine Disjoint.mono hcS hbS (bytes_disjoint ?_)
    simp only [strided, DType.bytes]; omega

/-- `M = 130`, `N = 128`, `K = 256`, the host's layout: two programs (two row tiles, through the
`GROUP_SIZE_M` grouping), four `k` steps each; the second tile has two valid rows. -/
def tall : Inst := hostInst 130 128 256
theorem tall_ok : tall.ok := hostInst_ok (by norm_num)

-- `Pre`: every byte of the three buffers is present in `initOf` (the buffers are aligned and
-- inside `[0, 2^64)` by their addresses: `float8e5` needs no alignment, `c`'s base is even).
#guard [small, strided, tall].all fun d =>
  let m := CMem.toMem (initOf d)
  d.c.base % 2 == 0 &&
  ((List.range d.a.len).all fun e : Nat => (m.read .f8e5 (d.a.ptr e).addr).isSome) &&
  ((List.range d.b.len).all fun e : Nat => (m.read .f8e5 (d.b.ptr e).addr).isSome) &&
  ((List.range d.c.len).all fun e : Nat => (m.read .f16 (d.c.ptr e).addr).isSome)

-- The inputs are encoded exactly. Every expectation is a multiple of `1/2` of magnitude at most
-- `2·K ≤ 512` (`|a| ≤ 2`, `|b| ≤ 1`): each such value is an exact `float16`, and none is the
-- sentinel.
#guard (List.range 5).all fun z : Nat =>
  (FloatV.ofRat ((z : Rat) - 2) : F8E5).toRat == (z : Rat) - 2 &&
  (FloatV.ofRat (((z : Rat) - 2) / 2) : F8E5).toRat == ((z : Rat) - 2) / 2
#guard [small, strided, tall].all (·.K ≤ 256) && 2 * 256 < 999 &&
  (List.range (8 * 256 + 1)).all fun z : Nat =>
    let q : Rat := (((z : Int) - 4 * 256 : Int) : Rat) / 2
    (FloatV.ofRat q : F16).toRat == q

def verdictSmall : Verdict := verdictOf small (expectOf small)
def verdictStrided : Verdict := verdictOf strided (expectOf strided)
def verdictTall : Verdict := verdictOf tall (expectOf tall)

-- (Each command below runs its launches again, so each verdict is printed once and guarded once.)
#eval IO.println (verdictSmall.line "matmul_e5m2/small")
#eval IO.println (verdictStrided.line "matmul_e5m2/strided")
#eval IO.println (verdictTall.line "matmul_e5m2/tall")
-- Agreement resting on three assumed answers per program and `k` step (one dot, two
-- conversions of `other`), and at least one.
#guard let v := verdictSmall; v == .conditional 6 && v.agrees && v.assumed > 0
#guard let v := verdictStrided; v == .conditional 12 && v.agrees && v.assumed > 0
#guard let v := verdictTall; v == .conditional 24 && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (one element off by one half; `c` with
-- rows and columns swapped).
#guard
  let r := runOf small
  (Verdict.ofLaunch (agreeOf small (initOf small) fun i j =>
    if i = 2 ∧ j = 127 then expectOf small i j + 1 / 2 else expectOf small i j) r).isDiscrepancy &&
  (Verdict.ofLaunch (agreeOf small (initOf small) fun i j => expectOf small (j % 3) i) r).isDiscrepancy
