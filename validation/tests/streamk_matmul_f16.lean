import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `streamk_matmul_f16` run by `execLaunch` on small covered instances.
Not a test against the device (phase 4): it checks that `exec` runs the model end to end, through
both of its paths (a tile one program owns entirely is stored; a tile split between programs is
accumulated into `c` with float `tl.atomic_add`), and that the final memory is what `Post` says on
these inputs.

The block sizes are the claim's (`128 × 128` tiles, `k` steps of `64`), and `Inst.ok` asks
`128 ∣ N`, `128 ∣ K`: the smallest covered instances already run one `128 × 128` tile and two
`k` steps.

**The `tl.dot` values and the atomic additions are assumed, not proved.** Both are platform
operations: `tl.dot` (`PlatQ.dot`) and a float `atomic_add` (`TQuery.atomicAdd`, the platform's
`atomicAdd`) can only be answered by the oracle. The oracle below answers each dot with the exact
`acc + a·b` computed from the request's own operands, and each atomic addition with the exact
`old + v`, each only when that value is a `float32` (it answers nothing otherwise). The inputs are
small integers (`|a|, |b| ≤ 2`, `K ≤ 384`), so every partial sum of a dot, every partial product
a program adds, and every sum of them in `c` is an integer of magnitude at most `1536`, exact in
`float32`: the exact value is the only one any correct device gives, whatever its order of
accumulation and of the atomic additions (`Device.DotF16` and `Device.AtomicAddF32` bound the
error by a multiple of the operands' magnitudes, `0` here only through exactness: the result is
the exact one because it is representable and both facts are within one rounding of it). The
verdicts are therefore `conditional n`, `n` the number of dots (one per iteration of the
`K` loops, `total_iters`) plus the number of atomic lane updates (one per enabled lane of each
`atomic_add`, `M·128` for a split tile of `M < 128` rows, per program owning part of it).

Each output element is then exactly `S = Σ_k a[i, k]·b[k, j]` (which `Post` allows: distance
`0 ≤ tol`), and every other byte is unchanged (`a`, `b`, the padding of `c`'s storage, sentinels
on both sides of `c`); `Pre` asks every element of the matrix `c` to start at zero, which
`initOf` writes. -/

open TVA TVA.Validation

namespace StreamKTest

open Claims.StreamKMatmul

/-- A platform that relates nothing: dots and atomics are answered by the oracle, never by `P0`. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle: exact dots, exact atomic additions -/

/-- An element as an integer, when it is a finite float (or an integer) with an integer value. -/
def intOf (d : DType) (x : d.Val) : Option Int :=
  match d.val? x with
  | some q => if q.den == 1 then some q.num else none
  | none => none

/-- An integer as a `float32`, when it holds it exactly. -/
def f32Of (z : Int) : Option F32 :=
  let v : F32 := FloatV.ofRat (z : Rat)
  if v.toRat == (z : Rat) then some v else none

/-- An integer as an accumulator element, when the accumulator is `float32` and holds it exactly. -/
def accOf : (d : DType) → Int → Option d.Val
  | .f32, z => f32Of z
  | _, _ => none

/-- The exact `acc + a·b` of a `tl.dot` request, from its own operands: `none` unless every operand
is an integer and every result an exact `float32`. -/
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

/-- The exact `old + v` of a `float32` atomic addition of integers, when it is a `float32`. -/
def exactAtomic (old v : F32) : Option F32 := do
  let o ← intOf .f32 old
  let w ← intOf .f32 v
  f32Of (o + w)

/-- The oracle: the exact dot at every platform request, the exact sum at every float atomic
addition; nothing else (trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactDot q
  | .atomicAdd .f32 old v => exactAtomic old v
  | _ => none

/-! ## The cases -/

/-- A sentinel in `c`'s storage outside the matrix, and in the elements just outside the storage:
they must keep it. Every element of the matrix starts at `0` (`Pre`). -/
def sentinel : Int := -999

/-- Sentinel elements on each side of `c`'s storage. -/
def pad : Nat := 4

def a0 (i k : Nat) : Int := ((3 * i + 5 * k) % 5 : Nat) - (2 : Int)
def b0 (k j : Nat) : Int := ((7 * k + 3 * j + 1) % 5 : Nat) - (2 : Int)

def f16v (z : Int) : F16 := FloatV.ofRat (z : Rat)
def f32v (z : Int) : F32 := FloatV.ofRat (z : Rat)

/-- The elements of the matrix `c`: `(i, j, address)`. -/
def cElems (d : Inst) : List (Nat × Nat × Int) :=
  (List.range d.M).flatMap fun i : Nat => (List.range d.N).map fun j : Nat =>
    (i, j, (d.c.ptr (i * d.stride_cm + j * d.stride_cn : Nat)).addr)

/-- The inputs of an instance at the elements its strides give; `c`'s whole storage and the
elements around it holding the sentinel, then the matrix `c` zero. -/
def initOf (d : Inst) : CMem :=
  let m := CMem.ofElems ∅ .f16 ((List.range d.M).flatMap fun i : Nat => (List.range d.K).map fun k : Nat =>
    ((d.a.ptr (i * d.stride_am + k * d.stride_ak : Nat)).addr, f16v (a0 i k)))
  let m := CMem.ofElems m .f16 ((List.range d.K).flatMap fun k : Nat => (List.range d.N).map fun j : Nat =>
    ((d.b.ptr (k * d.stride_bk + j * d.stride_bn : Nat)).addr, f16v (b0 k j)))
  let m := CMem.ofElems m .f32 ((List.range (d.c.len + 2 * pad)).map fun e : Nat =>
    ((d.c.ptr ((e : Int) - pad)).addr, f32v sentinel))
  CMem.ofElems m .f32 ((cElems d).map fun (_, _, a) => (a, f32v 0))

def runOf (d : Inst) : LRes (launch d).size × List Record :=
  execLaunch ((launch d).env P0) O fill0 100000 (model d) (launch d) (initOf d)

/-- What `Post` says element `(i, j)` of `c` holds on these inputs: `Σ_k a[i, k]·b[k, j]`, exact
(at distance `0`, within `tol`). -/
def expectOf (d : Inst) (i j : Nat) : Rat :=
  (((List.range d.K).map fun k => a0 i k * b0 k j).sum : Int)

/-- The final memory against `Post`: every element of the matrix `c` is the expectation; every
byte outside it (`a`, `b`, `c`'s padding, the sentinels) keeps its initial value, and no byte is
added. -/
def agreeOf (d : Inst) (init : CMem) (expect : Nat → Nat → Rat) (cf : CConfig (launch d).size) :
    Option String :=
  let m := CMem.toMem cf.mem
  let cs := cElems d
  let badOut := cs.filter fun (i, j, a) =>
    (m.read .f32 a).map (fun v : F32 => (v.isFinite, v.toRat)) != some (true, expect i j)
  let cBytes : Std.HashSet Int :=
    cs.foldl (fun s (_, _, a) => (((s.insert a).insert (a + 1)).insert (a + 2)).insert (a + 3)) {}
  let badElse := init.toList.filter fun (a, b) => !cBytes.contains a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"c elements {(badOut.take 8).map fun (i, j, _) => (i, j)} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside the matrix c changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def verdictOf (d : Inst) (expect : Nat → Nat → Rat) : Verdict :=
  Verdict.ofLaunch (agreeOf d (initOf d) expect) (runOf d)

/-- Hypotheses of `mk_ok` on small sizes, discharged by `norm_num`. -/
macro "mk_ok_small" : term =>
  `(mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))

/-- `M = 3`, `N = 128`, `K = 384`, 4 programs (the host's contiguous layout): one tile of six
`k` steps, `iters_per_program = 2`: programs 0–2 own steps `[0, 2)`, `[2, 4)`, `[4, 6)` and add
their partial products with `atomic_add` (three atomic additions per element, into the zero `c`);
program 3's range `[6, 8)` is past `total_iters = 6`: it does nothing. Rows `3..127` masked. -/
def split : Inst := mk 3 128 384 4
theorem split_ok : split.ok := mk_ok_small

/-- `M = 2`, `N = 384`, `K = 128`, 2 programs: three tiles of two steps, `iters_per_program = 3`:
program 0 owns tile 0 entirely (stored) and step 0 of tile 1 (atomic); program 1 owns step 1 of
tile 1 (atomic) and tile 2 entirely (stored). Both paths, and a range across tiles. -/
def mixed : Inst := mk 2 384 128 2
theorem mixed_ok : mixed.ok := mk_ok_small

/-- `M = 2`, `N = 128`, `K = 128`, 3 programs, other layouts: `b` column-major (`stride_bk = 1`,
`stride_bn = 128`), `c`'s rows padded to `136` elements (the padding must keep the sentinel).
`iters_per_program = 1`: programs 0 and 1 each add one step with `atomic_add`, program 2 does
nothing. -/
def strided : Inst :=
  { M := 2, N := 128, K := 128, stride_am := 128, stride_ak := 1, stride_bk := 1, stride_bn := 128,
    stride_cm := 136, stride_cn := 1, num_programs := 3, iters_per_program := 1, total_iters := 2,
    a := ⟨2 ^ 32, 2 * 128⟩, b := ⟨2 ^ 34, 128 * 128⟩, c := ⟨2 ^ 36, 2 * 136⟩ }

theorem strided_ok : strided.ok := by
  have hcS : ∀ j ∈ cIdx strided, j < strided.c.len := by
    rintro _ ⟨i, j, hi, hj, rfl⟩; simp only [strided] at *; omega
  have haS : ∀ j ∈ aIdx strided, j < strided.a.len := by
    rintro _ ⟨i, k, hi, hk, rfl⟩; simp only [strided] at *; omega
  have hbS : ∀ j ∈ bIdx strided, j < strided.b.len := by
    rintro _ ⟨k, j, hk, hj, rfl⟩; simp only [strided] at *; omega
  have hdis : ∀ {d₁ d₂ : DType} (x : Buf d₁) (y : Buf d₂),
      x.base + x.len * d₁.bytes ≤ y.base ∨ y.base + y.len * d₂.bytes ≤ x.base →
      Disjoint x.bytes y.bytes := by
    intro _ _ x y h
    rw [Set.disjoint_left]
    intro z hx hy
    simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
    omega
  refine ⟨by norm_num [strided], by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by norm_num [strided], by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by norm_num [strided], by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by norm_num [strided], by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by norm_num [strided], by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by simp [strided, tilesM, tilesN, steps], by norm_num [strided], by norm_num [strided],
    ?_, ?_, ?_, ?_, ?_⟩
  · intro i k hi hk; simp only [strided] at *; omega
  · intro k j hk hj; simp only [strided] at *; omega
  · intro i j hi hj; simp only [strided] at *; omega
  · exact Buf.elems_disjoint_of _ _ (hdis _ _ (by simp only [strided, DType.bytes]; omega)) hcS haS
  · exact Buf.elems_disjoint_of _ _ (hdis _ _ (by simp only [strided, DType.bytes]; omega)) hcS hbS

end StreamKTest

open StreamKTest Claims.StreamKMatmul

-- The instances' host-chosen counts: `total_iters`, `iters_per_program` as the cases describe.
#guard (split.total_iters, split.iters_per_program, mixed.total_iters, mixed.iters_per_program) == (6, 2, 6, 3)

-- `Pre`: the buffers are aligned and inside `[0, 2^64)`, every byte of them is present in
-- `initOf`, and every element of the matrix `c` holds a finite zero.
#guard [split, mixed, strided].all fun d =>
  let m := CMem.toMem (initOf d)
  d.a.base % 2 == 0 && d.b.base % 2 == 0 && d.c.base % 4 == 0 && 0 ≤ d.a.base &&
  d.c.base + d.c.len * 4 ≤ 2 ^ 64 && d.b.base + d.b.len * 2 ≤ 2 ^ 64 && d.a.base + d.a.len * 2 ≤ 2 ^ 64 &&
  ((List.range d.a.len).all fun e : Nat => (m.read .f16 (d.a.ptr e).addr).isSome) &&
  ((List.range d.b.len).all fun e : Nat => (m.read .f16 (d.b.ptr e).addr).isSome) &&
  ((List.range d.c.len).all fun e : Nat => (m.read .f32 (d.c.ptr e).addr).isSome) &&
  ((cElems d).all fun (_, _, a) => (m.read .f32 a).map (fun v : F32 => (v.isFinite, v.toRat)) == some (true, 0))

-- The inputs are encoded exactly, and no expectation equals the sentinel or `0`'s complement
-- trivially (some expectation is nonzero).
#guard (List.range 5).all fun z : Nat => (f16v (z - 2 : Int)).toRat == ((z - 2 : Int) : Rat)
#guard [split, mixed, strided].all fun d => (List.range d.M).all fun i : Nat => (List.range d.N).all fun j : Nat =>
  expectOf d i j != (sentinel : Rat)
#guard [split, mixed, strided].all fun d => (List.range d.M).any fun i : Nat => (List.range d.N).any fun j : Nat =>
  expectOf d i j != 0

def verdictSplit : Verdict := verdictOf split (expectOf split)
def verdictMixed : Verdict := verdictOf mixed (expectOf mixed)
def verdictStrided : Verdict := verdictOf strided (expectOf strided)

#eval IO.println (verdictSplit.line "streamk_matmul_f16/split")
#eval IO.println (verdictMixed.line "streamk_matmul_f16/mixed")
#eval IO.println (verdictStrided.line "streamk_matmul_f16/strided")
-- Agreement resting on one assumed answer per dot (`total_iters`) and per atomic lane update
-- (`M · 128` per `atomic_add` of a split tile): `split` 6 dots + 3 atomics × 384 lanes, `mixed`
-- 6 dots + 2 × 256, `strided` 2 dots + 2 × 256.
#guard let v := verdictSplit; v == .conditional (6 + 3 * 384) && v.agrees && v.assumed > 0
#guard let v := verdictMixed; v == .conditional (6 + 2 * 256) && v.agrees && v.assumed > 0
#guard let v := verdictStrided; v == .conditional (2 + 2 * 256) && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (one element off by one; one 64-chunk of
-- `K` left out, the value a lost atomic addition would leave).
#guard
  let r := runOf strided
  (Verdict.ofLaunch (agreeOf strided (initOf strided) fun i j =>
    if i = 1 ∧ j = 127 then expectOf strided i j + 1 else expectOf strided i j) r).isDiscrepancy &&
  (Verdict.ofLaunch (agreeOf strided (initOf strided) fun i j =>
    (((List.range 64).map fun k => a0 i k * b0 k j).sum : Int)) r).isDiscrepancy
