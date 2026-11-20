import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `matmul_f16` run by `execLaunch` on small covered instances. Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says on these inputs.

The block sizes are the claim's (`128 × 256` tiles, `k` steps of `64`), and `Inst.ok` asks
`128 ∣ N`, `128 ∣ K`: the smallest covered instances already run one `128 × 256` tile per program.

**The `tl.dot` values are assumed, not proved.** `tl.dot` is a platform operation (`PlatQ.dot`):
only the oracle can supply its value. The oracle below answers each dot with the exact
`acc + a·b`, computed from the request's own operands, and only when that value is a `float32`
(it answers nothing otherwise). The inputs are small integers (`|a|, |b| ≤ 2`, `K ≤ 256`), so every
partial sum is an integer of magnitude at most `1024`: the exact value is the only one any correct
device gives, whatever its order and rounding, and the `float16` rounding of the result is exact.
The verdicts are therefore `conditional n`, `n` the number of dots answered (one per program and
`k` step). Each output element is then exactly `Σ_k a[i, k]·b[k, j]` (which `Post` allows, at
distance `0`), and every other byte is unchanged (`a`, `b`, the padding of `c`'s storage,
sentinels on both sides of `c`). -/

open TVA TVA.Validation Claims.Matmul

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

/-! ## The oracle: exact dots -/

/-- An element as an integer, when it is a finite float (or an integer) with an integer value. -/
def intOf (d : DType) (x : d.Val) : Option Int :=
  match d.val? x with
  | some q => if q.den == 1 then some q.num else none
  | none => none

/-- An integer as an accumulator element, when the accumulator is `float32` and holds it exactly. -/
def accOf : (d : DType) → Int → Option d.Val
  | .f32, z => let v : F32 := FloatV.ofRat (z : Rat); if v.toRat == (z : Rat) then some v else none
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

/-- The oracle: the exact dot at every platform request; nothing else (trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactDot q
  | _ => none

/-! ## The cases -/

/-- A sentinel in `c`'s storage before the launch, and in the elements just outside it: those
outside the matrix `c` must keep it; every matrix element must be overwritten. -/
def sentinel : Int := -999

/-- Sentinel elements on each side of `c`'s storage. -/
def pad : Nat := 4

def a0 (i k : Nat) : Int := ((3 * i + 5 * k) % 5 : Nat) - (2 : Int)
def b0 (k j : Nat) : Int := ((7 * k + 3 * j + 1) % 5 : Nat) - (2 : Int)

def f16v (z : Int) : F16 := FloatV.ofRat (z : Rat)

/-- The inputs of an instance, at the elements its strides give, and `c`'s whole storage and the
sentinels around it holding the sentinel. -/
def initOf (d : Inst) : CMem :=
  let m := CMem.ofElems ∅ .f16 ((List.range d.M).flatMap fun i : Nat => (List.range d.K).map fun k : Nat =>
    ((d.a.ptr (i * d.stride_am + k * d.stride_ak : Nat)).addr, f16v (a0 i k)))
  let m := CMem.ofElems m .f16 ((List.range d.K).flatMap fun k : Nat => (List.range d.N).map fun j : Nat =>
    ((d.b.ptr (k * d.stride_bk + j * d.stride_bn : Nat)).addr, f16v (b0 k j)))
  CMem.ofElems m .f16 ((List.range (d.c.len + 2 * pad)).map fun e : Nat =>
    ((d.c.ptr ((e : Int) - pad)).addr, f16v sentinel))

def runOf (d : Inst) : LRes (launch d).size × List Record :=
  execLaunch ((launch d).env P0) O fill0 10000 (model d) (launch d) (initOf d)

/-- What `Post` says element `(i, j)` of `c` holds on these inputs: `Σ_k a[i, k]·b[k, j]`, exact. -/
def expectOf (d : Inst) (i j : Nat) : Rat :=
  (((List.range d.K).map fun k => a0 i k * b0 k j).sum : Int)

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

/-- `M = 3`, `N = 128`, `K = 128`, the host's contiguous layout: one program, two `k` steps (two
dots); rows `3..127` of the tile masked (their loads wrap to rows `% M`, their stores are
disabled) and columns `128..255` masked (half a column tile, as `128 ∣ N` allows). -/
def small : Inst := hostInst 3 128 128
theorem small_ok : small.ok := hostInst_ok (by norm_num)

/-- `M = 2`, `N = 384`, `K = 128`, other layouts: `b` column-major (`stride_bk = 1`,
`stride_bn = K`), `c`'s rows padded to `400` elements (the padding must keep the sentinel). Two
programs (two column tiles, the second half masked), two dots each. -/
def strided : Inst :=
  { M := 2, N := 384, K := 128, stride_am := 128, stride_ak := 1, stride_bk := 1, stride_bn := 128,
    stride_cm := 400, stride_cn := 1,
    a := ⟨2 ^ 32, 2 * 128⟩, b := ⟨2 ^ 34, 128 * 384⟩, c := ⟨2 ^ 36, 2 * 400⟩ }

theorem strided_ok : strided.ok := by
  have hcS : strided.c.elems (cIdx strided) ⊆ strided.c.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, j, hi, hj, rfl⟩; simp only [strided] at *; omega)
  have haS : strided.a.elems (aIdx strided) ⊆ strided.a.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, k, hi, hk, rfl⟩; simp only [strided] at *; omega)
  have hbS : strided.b.elems (bIdx strided) ⊆ strided.b.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨k, j, hk, hj, rfl⟩; simp only [strided] at *; omega)
  refine ⟨by norm_num [strided], by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by norm_num [strided], by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by norm_num [strided], by norm_num [strided], by norm_num [strided], by norm_num [strided],
    by norm_num [strided], by norm_num [strided], ?_, ?_, ?_, ?_, ?_, ?_, by norm_num [strided],
    by norm_num [strided], by norm_num [strided]⟩
  · intro i j i' j' hi hj hi' hj' he; simp only [strided] at *; omega
  · intro i k hi hk; simp only [strided] at *; omega
  · intro k j hk hj; simp only [strided] at *; omega
  · intro i j hi hj; simp only [strided] at *; omega
  · refine Disjoint.mono hcS haS (bytes_disjoint ?_)
    simp only [strided, DType.bytes]; omega
  · refine Disjoint.mono hcS hbS (bytes_disjoint ?_)
    simp only [strided, DType.bytes]; omega

/-- `M = 130`, `N = 128`, `K = 256`, contiguous: two programs (two row tiles, through the
`GROUP_SIZE_M` grouping), four `k` steps each (four dots); the second tile has two valid rows. -/
def tall : Inst := hostInst 130 128 256
theorem tall_ok : tall.ok := hostInst_ok (by norm_num)

-- `Pre`: every byte of the three buffers is present in `initOf` (the buffers are aligned and
-- inside `[0, 2^64)` by their addresses), here checked element by element.
#guard [small, strided].all fun d =>
  let m := CMem.toMem (initOf d)
  ((List.range d.a.len).all fun e : Nat => (m.read .f16 (d.a.ptr e).addr).isSome) &&
  ((List.range d.b.len).all fun e : Nat => (m.read .f16 (d.b.ptr e).addr).isSome) &&
  ((List.range d.c.len).all fun e : Nat => (m.read .f16 (d.c.ptr e).addr).isSome)

-- The inputs are encoded exactly, and no expectation equals the sentinel.
#guard (List.range 5).all fun z : Nat => (f16v (z - 2 : Int)).toRat == ((z - 2 : Int) : Rat)
#guard [small, strided].all fun d => (List.range d.M).all fun i : Nat => (List.range d.N).all fun j : Nat =>
  expectOf d i j != (sentinel : Rat)

def verdictSmall : Verdict := verdictOf small (expectOf small)
def verdictStrided : Verdict := verdictOf strided (expectOf strided)
def verdictTall : Verdict := verdictOf tall (expectOf tall)

-- (Each command below runs its launches again: about 7 s for `small`, 14 s for `strided`, 31 s
-- for `tall`. So each verdict is printed once and guarded once.)
#eval IO.println (verdictSmall.line "matmul_f16/small")
#eval IO.println (verdictStrided.line "matmul_f16/strided")
#eval IO.println (verdictTall.line "matmul_f16/tall")
-- Agreement resting on one assumed answer per dot (programs × `k` steps), and at least one.
#guard let v := verdictSmall; v == .conditional 2 && v.agrees && v.assumed > 0
#guard let v := verdictStrided; v == .conditional 4 && v.agrees && v.assumed > 0
#guard let v := verdictTall; v == .conditional 8 && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (one element off by one; `c` with rows
-- and columns swapped).
#guard
  let r := runOf small
  (Verdict.ofLaunch (agreeOf small (initOf small) fun i j =>
    if i = 2 ∧ j = 127 then expectOf small i j + 1 else expectOf small i j) r).isDiscrepancy &&
  (Verdict.ofLaunch (agreeOf small (initOf small) fun i j => expectOf small (j % 3) i) r).isDiscrepancy
