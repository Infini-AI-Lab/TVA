import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `persistent_matmul_f16` run by `execLaunch` on small covered
instances. Not a test against the device (phase 4): it checks that `exec` runs the model end to
end, and that the final memory is what `Post` says on these inputs.

The block sizes are the claim's (`128 × 256` tiles, `k` steps of `64`), and `Inst.ok` asks whole
tiles (`128 ∣ M`, `256 ∣ N`) and `128 ∣ K`: the smallest covered instance is one full tile with two
`k` steps. The cases cover the persistent schedule: three tiles over two programs (`NUM_SMS = 2`:
program `0` loops over tiles `0` and `2`, program `1` takes tile `1`), and one program for one
tile (`NUM_SMS = 132`, the H200's, above the tile count, with padded and row-major layouts).
(Every tile is a full `128 × 256` tile and costs about 18 s in the interpreter: two cases, four
tiles.)

**The `tl.dot` values are assumed, not proved.** `tl.dot` is a platform operation (`PlatQ.dot`):
only the oracle can supply its value. The oracle below answers each dot with the exact
`acc + a·b`, computed from the request's own operands (in integers, scaled by `2^10`), and only
when that value is a `float32` (it answers nothing otherwise). The inputs are `a[i, k] ∈ {-2, …, 2}` and
`b[k, j] ∈ {-1, -1/2, 0, 1/2, 1}` (exact in `float16`), `K = 128`: every partial sum is a multiple
of `1/2` of magnitude at most `256`, exact in `float32` and in `float16`. The exact value is the
only one any correct device gives, whatever its order and rounding. The conversions of
`other=0.0` into `float16` are standard conversions, proved. The verdicts are therefore
`conditional n`, `n` the number of dots (one per tile and `k` step). Each output element is then
exactly `Σ_k a[i, k]·b[k, j]` (which `Post` allows: the `float32` value `v = S` at distance
`0 ≤ tol`, its `float16` rounding exact), and every other byte is unchanged (`a`, `b` with the
padding of their storage, the padding of `c`'s storage, sentinels on both sides of `c`). -/

open TVA TVA.Validation Claims.PersistentMatmul

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

/-- The oracle: the exact dot at every platform request; nothing else (trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactDotQ q
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

def f16q (q : Rat) : F16 := FloatV.ofRat q

/-- A filler in the elements of `a`'s and `b`'s storage outside the matrices (row padding): `Pre`
asks every byte present, and they must be unchanged. -/
def filler : F16 := FloatV.ofRat 3

/-- The inputs of an instance, at the elements its strides give (the rest of `a`'s and `b`'s
storage holding the filler), and `c`'s whole storage and the sentinels around it holding the
sentinel. -/
def initOf (d : Inst) : CMem :=
  let m := CMem.ofElems ∅ .f16 ((List.range d.a.len).map fun e : Nat => ((d.a.ptr e).addr, filler))
  let m := CMem.ofElems m .f16 ((List.range d.b.len).map fun e : Nat => ((d.b.ptr e).addr, filler))
  let m := CMem.ofElems m .f16 ((List.range d.M).flatMap fun i : Nat => (List.range d.K).map fun k : Nat =>
    ((d.a.ptr (i * d.stride_am + k * d.stride_ak : Nat)).addr, f16q (a0 i k)))
  let m := CMem.ofElems m .f16 ((List.range d.K).flatMap fun k : Nat => (List.range d.N).map fun j : Nat =>
    ((d.b.ptr (k * d.stride_bk + j * d.stride_bn : Nat)).addr, f16q ((b2 k j : Rat) / 2)))
  CMem.ofElems m .f16 ((List.range (d.c.len + 2 * pad)).map fun e : Nat =>
    ((d.c.ptr ((e : Int) - pad)).addr, f16q sentinel))

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

/-- `M = 384`, `N = 256`, `K = 128`, `NUM_SMS = 2`, the host's layout (`a` contiguous, `b` the
transpose of a contiguous matrix, strides `(1, K)`): three tiles over two programs; program `0`
loops over tiles `0` and `2`, program `1` takes tile `1` (six dots). -/
def uneven : Inst := hostInst 384 256 128 2
theorem uneven_ok : uneven.ok := hostInst_ok (by norm_num)

/-- `M = 128`, `N = 256`, `K = 128`, `NUM_SMS = 132` (the H200's, above the tile count), other
layouts: `a`'s rows padded to `136` elements, `b` row-major (`stride_bk = N`, `stride_bn = 1`),
`c`'s rows padded to `264` elements (the padding must keep the sentinel). One tile, so
`min(132, 1) = 1` program, two `k` steps (two dots). -/
def strided : Inst :=
  { M := 128, N := 256, K := 128, NUM_SMS := 132, stride_am := 136, stride_ak := 1,
    stride_bk := 256, stride_bn := 1, stride_cm := 264, stride_cn := 1,
    a := ⟨2 ^ 32, 128 * 136⟩, b := ⟨2 ^ 34, 128 * 256⟩, c := ⟨2 ^ 36, 128 * 264⟩ }

theorem strided_ok : strided.ok := by
  have hcS : strided.c.elems (cIdx strided) ⊆ strided.c.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, j, hi, hj, rfl⟩; simp only [strided] at *; omega)
  have haS : strided.a.elems (aIdx strided) ⊆ strided.a.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨i, k, hi, hk, rfl⟩; simp only [strided] at *; omega)
  have hbS : strided.b.elems (bIdx strided) ⊆ strided.b.bytes :=
    Buf.elems_sub_bytes _ (by rintro _ ⟨k, j, hk, hj, rfl⟩; simp only [strided] at *; omega)
  constructor
  all_goals try first
    | (norm_num [strided, tiles]; done)
    | (intro i j i' j' hi hj hi' hj' he; simp only [strided] at *; omega)
    | (intro i k hi hk; simp only [strided] at *; omega)
  · refine Disjoint.mono hcS haS (bytes_disjoint ?_)
    simp only [strided, DType.bytes]; omega
  · refine Disjoint.mono hcS hbS (bytes_disjoint ?_)
    simp only [strided, DType.bytes]; omega

-- `Pre`: every byte of the three buffers is present in `initOf`, and each base is a multiple of
-- `2` (the buffers are inside `[0, 2^64)` by their addresses).
#guard [uneven, strided].all fun d =>
  let m := CMem.toMem (initOf d)
  d.a.base % 2 == 0 && d.b.base % 2 == 0 && d.c.base % 2 == 0 &&
  ((List.range d.a.len).all fun e : Nat => (m.read .f16 (d.a.ptr e).addr).isSome) &&
  ((List.range d.b.len).all fun e : Nat => (m.read .f16 (d.b.ptr e).addr).isSome) &&
  ((List.range d.c.len).all fun e : Nat => (m.read .f16 (d.c.ptr e).addr).isSome)

-- The grids: two programs, one program.
#guard (launch uneven).size == 2 && (launch strided).size == 1

-- The inputs are encoded exactly. Every expectation is a multiple of `1/2` of magnitude at most
-- `2·K ≤ 256` (`|a| ≤ 2`, `|b| ≤ 1`): each such value is an exact `float16`, and none is the
-- sentinel.
#guard (List.range 5).all fun z : Nat =>
  (FloatV.ofRat ((z : Rat) - 2) : F16).toRat == (z : Rat) - 2 &&
  (FloatV.ofRat (((z : Rat) - 2) / 2) : F16).toRat == ((z : Rat) - 2) / 2
#guard [uneven, strided].all (·.K ≤ 128) && 2 * 128 < 999 &&
  (List.range (8 * 128 + 1)).all fun z : Nat =>
    let q : Rat := (((z : Int) - 4 * 128 : Int) : Rat) / 2
    (FloatV.ofRat q : F16).toRat == q

def verdictUneven : Verdict := verdictOf uneven (expectOf uneven)
def verdictStrided : Verdict := verdictOf strided (expectOf strided)

-- (Each command below runs its launches again, about 18 s per tile: each verdict is printed once
-- and guarded once, and the non-vacuity guard shares the run of `strided`'s guard.)
#eval IO.println (verdictUneven.line "persistent_matmul_f16/uneven")
#eval IO.println (verdictStrided.line "persistent_matmul_f16/strided")
-- Agreement resting on one assumed answer per dot (tiles × `k` steps), and at least one.
#guard let v := verdictUneven; v == .conditional 6 && v.agrees && v.assumed > 0
-- The same for `strided`; and a wrong expectation is caught on the same run: `agree` is not
-- vacuous (one element off by one half; `c` with rows and columns swapped).
#guard
  let r := runOf strided
  let v := Verdict.ofLaunch (agreeOf strided (initOf strided) (expectOf strided)) r
  v == .conditional 2 && v.agrees && v.assumed > 0 &&
  (Verdict.ofLaunch (agreeOf strided (initOf strided) fun i j =>
    if i = 127 ∧ j = 255 then expectOf strided i j + 1 / 2 else expectOf strided i j) r).isDiscrepancy &&
  (Verdict.ofLaunch (agreeOf strided (initOf strided) fun i j => expectOf strided (j % 128) i) r).isDiscrepancy
