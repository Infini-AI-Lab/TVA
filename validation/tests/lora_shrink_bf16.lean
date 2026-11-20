import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `lora_shrink_bf16` run by `execLaunch` on two small covered instances,
one per host configuration. Not a test against the device (phase 4): it checks that `exec` runs the
model end to end, and that the final memory is what `Post` says on these inputs, both halves: every
element of the output's `[S, M, R]` tensor (`+0.0` for a token with no adapter; within `tol` of
`scaling · S` otherwise), and every byte outside it unchanged (the inputs, the weights, the address
table, the metadata, sentinels on both sides of the output).

**The platform answers assumed** (the oracle `O`):
* each `tl.dot_add` of `mm_k` (line 152, 166 or 177; `Site.dotAdd`, asking `PlatQ.dot`): the exact
  `acc + a·b` from the request's own operands, answered only when it is a `float32` (as
  `matmul_f16`'s oracle). The inputs are small integers (`|x|, |w| ≤ 2`, `K ≤ 300`; masked lanes
  load `+0`), so every partial sum is an integer of magnitude at most `1200`: the exact value is the
  only one any correct device gives, whatever its order and rounding;
* each lane of the relaxed `tl.atomic_add` (line 305; a float `atomic_add`, asking
  `TQuery.atomicAdd`): the exact `old + v` (`FloatV.add`, exact here). The values are the parts'
  integer sums times `scaling` (`1/2` or `1`) and their partial sums, multiples of `1/2` of
  magnitude below `2^11`: every IEEE sum is exact, so again the only value a correct device gives,
  in any order of the additions (`exec` runs the programs in grid order).

So the verdicts are `conditional n`, `n` the number of dots plus the number of atomic lanes.

**The check of `Post`.** For a token on adapter `l ≥ 0`, the inputs are finite, `dotMag` and
`vMag + atomTol` are far below their bounds (all magnitudes below `2^12`), so `Post` asks for a
finite `v` with `|v − scaling·S| ≤ tol`; `tol > 0` (its floor `2^-100`), and the test asks for
`v = scaling·S` exactly (`S` the exact dot product, computed here from the inputs as an integer,
`scaling` the host's double as a rational), which is stricter. For a token with no adapter it asks
for the `+0.0` bits, as `Post`. -/

open TVA TVA.Validation Claims.LoraShrink

/-- A platform that relates nothing: the dots and the atomics are answered by the oracle. -/
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

/-- The exact sum of a `float32` atomic addition, when the IEEE sum is exact. -/
def exactAdd (old v : F32) : Option F32 :=
  let r := FloatV.add old v
  if old.isFinite && v.isFinite && r.toRat == old.toRat + v.toRat then some r else none

/-- The oracle: the exact dot at every platform request, the exact sum at every `float32` atomic
addition; nothing else (trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactDot q
  | .atomicAdd .f32 old v => exactAdd old v
  | _ => none

/-! ## The cases -/

/-- A sentinel in the elements just outside the output (whose own elements start at `+0.0`). -/
def sentinel : Int := -99

/-- Sentinel elements on each side of the output. -/
def pad : Nat := 4

/-- `inputs[t, k]` and slice `s`'s weights `[l, j, k]`: small integers in `-2 … 2`. -/
def x0 (t k : Nat) : Int := ((3 * t + 5 * k + k / 7) % 5 : Nat) - (2 : Int)
def w0 (s l j k : Nat) : Int := ((7 * k + 3 * j + 2 * l + 4 * s + k / 11 + 1) % 5 : Nat) - (2 : Int)

def initOf (d : Inst) : CMem :=
  let m := CMem.ofElems ∅ .bf16 ((List.range d.inp.len).map fun e : Nat =>
    ((d.inp.ptr e).addr, (FloatV.ofRat (x0 (e / d.K) (e % d.K) : Rat) : BF16)))
  let m := (List.range d.S).foldl (fun m s =>
    CMem.ofElems m .bf16 ((List.range (d.lora s).len).map fun e : Nat =>
      (((d.lora s).ptr e).addr,
        (FloatV.ofRat (w0 s (e / (d.R * d.K)) (e / d.K % d.R) (e % d.K) : Rat) : BF16)))) m
  let m := if 1 < d.S then CMem.ofElems m .u64 ((List.range d.S).map fun s : Nat =>
      ((d.tab.ptr s).addr, (IntV.ofInt (d.lora s).base : U64))) else m
  let m := CMem.ofElems m .f32 ((List.range (d.out.len + 2 * pad)).map fun e : Nat =>
    ((d.out.ptr ((e : Int) - pad)).addr,
      if pad ≤ e ∧ e < pad + d.out.len then (FloatV.ofPat 0 : F32) else FloatV.ofRat (sentinel : Rat)))
  let m := CMem.ofElems m .i32 ((List.range d.tIdxBuf.len).map fun p : Nat =>
    ((d.tIdxBuf.ptr p).addr, (IntV.ofNat (d.tIdx p) : I32)))
  let m := CMem.ofElems m .i32 ((List.range d.cntBuf.len).map fun g : Nat =>
    ((d.cntBuf.ptr g).addr, (IntV.ofNat (cntAt d g) : I32)))
  let m := CMem.ofElems m .i32 ((List.range d.startBuf.len).map fun g : Nat =>
    ((d.startBuf.ptr g).addr, (IntV.ofNat (startAt d g) : I32)))
  CMem.ofElems m .i32 ((List.range d.idBuf.len).map fun g : Nat =>
    ((d.idBuf.ptr g).addr, (IntV.ofInt (idAt d g) : I32)))

def runOf (d : Inst) : LRes (launch d).size × List Record :=
  execLaunch ((launch d).env P0) O fill0 100000 (model d) (launch d) (initOf d)

/-- What `Post` says element `(s, t, j)` holds, with the scale `sc` and the slice `sl s` of the
weights (`d.scaling.toRat` and `s` for the test): `none` for the `+0.0` bits (no adapter), else
`sc · S`, `S = Σ_k x[t, k]·A_{s, l}[j, k]`. -/
def expectOf (d : Inst) (sc : Rat) (sl : Nat → Nat) (s t j : Nat) : Option Rat :=
  if d.adapter t < 0 then none
  else some (sc * (((List.range d.K).map fun k => x0 t k * w0 (sl s) (d.adapter t).toNat j k).sum : Int))

/-- The final memory against `Post`: every element of the output's tensor (the `+0.0` bits, or a
finite value equal to the expectation), then every byte outside the output (which is the tensor)
unchanged, and no byte added. -/
def agreeOf (d : Inst) (init : CMem) (sc : Rat) (sl : Nat → Nat) (cf : CConfig (launch d).size) :
    Option String :=
  let m := CMem.toMem cf.mem
  let elems := (List.range d.S).flatMap fun s => (List.range d.M).flatMap fun t =>
    (List.range d.R).map fun j => (s, t, j)
  let bad := elems.filter fun (s, t, j) =>
    match m.read .f32 (d.out.ptr (outAt d s t j)).addr, expectOf d sc sl s t j with
    | some v, none => DType.toBits .f32 v != 0
    | some v, some q => !(v.isFinite && v.toRat == q)
    | none, _ => true
  let lo := d.out.base
  let hi := d.out.base + (d.out.len * 4 : Nat)
  let changed := init.toList.filter fun (a, b) => !(lo ≤ a && a < hi) && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"output elements {bad.take 8} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the output changed"
  else if cf.mem.size != init.size then some "bytes were added"
  else none

/-- Decode configuration (`M < 128`: `BLOCK_K = 256`, `SPLIT_K = 64`), `EVEN_K = False`: 3 tokens,
hidden `300`, rank 16, 2 adapters, **two slices** (the `uint64` address table and `intToPtr`),
`scaling = 0.5`; token 0 has no adapter, tokens 1 and 2 are on adapter 1; 3 groups in the grid
(`-1`, `1`, and one past the distinct ids). Part 0 takes `k < 256` unmasked (line 158), part 1
`256 ≤ k < 300` masked (line 169), parts 2 … 63 skip (line 155) but still add their zero tile. -/
def decode : Inst := Claims.LoraShrink.twoInst 3 300 16 2 2 3 1 1 scHalf

theorem decode_ok : decode.ok :=
  twoInst_ok (by cov_shape scHalf_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

/-- Prefill configuration (`M ≥ 128`: `BLOCK_K = 32`, `SPLIT_K = 8`), `EVEN_K = True` (`256 = 32·8`):
128 tokens, rank 32 (two column tiles), one adapter, one slice, `scaling = 1.0`; the first 5 tokens
have no adapter, the other 123 are on adapter 0, so the group's last row tile has 27 rows (its
`offset_m` wraps, line 387, and its store mask cuts, line 293). -/
def prefill : Inst := Claims.LoraShrink.twoInst 128 256 32 1 1 2 5 0 scOne

theorem prefill_ok : prefill.ok :=
  twoInst_ok (by cov_shape scOne_fin) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num)

-- The configurations: `BLOCK_K`, `SPLIT_K`, `EVEN_K`; the grid's axis 0; the scales exact.
#guard bk decode == 256 && sk decode == 64 && !evenK decode && grid0 decode == 64 &&
  bk prefill == 32 && sk prefill == 8 && evenK prefill && grid0 prefill == 64 &&
  decode.scaling.toRat == 1 / 2 && (scaling32 decode).toRat == 1 / 2 &&
  prefill.scaling.toRat == 1 && (scaling32 prefill).toRat == 1

-- `Pre`, decidable parts: the buffers aligned (and below `2^64`) and every element present in
-- `init` (each slice's weights; with two slices the table, holding each slice's base); the
-- metadata as `lora_kernel_meta` writes it (`Holds` over each buffer); the output's tensor `+0.0`.
#guard [decode, prefill].all fun d =>
  let m := CMem.toMem (initOf d)
  let present {dt : DType} (b : Buf dt) : Bool :=
    b.base % dt.bytes == 0 && (List.range b.len).all fun j : Nat => (m.read dt (b.ptr j).addr).isSome
  present d.inp && (List.range d.S).all (fun s => present (d.lora s)) && present d.out &&
  present d.tIdxBuf && present d.cntBuf && present d.startBuf && present d.idBuf &&
  (!(1 < d.S) || (present d.tab && (List.range d.S).all fun s =>
    m.read .u64 (d.tab.ptr s).addr == some (IntV.ofInt (d.lora s).base))) &&
  (List.range d.cntBuf.len).all (fun g => m.read .i32 (d.cntBuf.ptr g).addr == some (IntV.ofNat (cntAt d g))) &&
  (List.range d.startBuf.len).all (fun g => m.read .i32 (d.startBuf.ptr g).addr == some (IntV.ofNat (startAt d g))) &&
  (List.range d.idBuf.len).all (fun g => m.read .i32 (d.idBuf.ptr g).addr == some (IntV.ofInt (idAt d g))) &&
  (List.range d.M).all (fun p => m.read .i32 (d.tIdxBuf.ptr p).addr == some (IntV.ofNat (d.tIdx p))) &&
  (List.range d.out.len).all fun e : Nat => m.read .f32 (d.out.ptr e).addr == some (FloatV.ofPat 0)
-- The inputs are encoded exactly; both kinds of token occur; no expectation is `+0.0`'s ambiguity
-- (a token on an adapter has a nonzero expectation somewhere).
#guard (List.range 5).all fun z : Nat => (FloatV.ofRat ((z : Int) - 2 : Rat) : BF16).toRat == ((z : Int) - 2 : Rat)
#guard [decode, prefill].all fun d => d.adapter 0 < 0 && 0 ≤ d.adapter (d.M - 1) &&
  (List.range d.R).any fun j => expectOf d d.scaling.toRat id 0 (d.M - 1) j != some 0

def verdictDecode : Verdict := Verdict.ofLaunch (agreeOf decode (initOf decode) decode.scaling.toRat id) (runOf decode)
def verdictPrefill : Verdict := Verdict.ofLaunch (agreeOf prefill (initOf prefill) prefill.scaling.toRat id) (runOf prefill)

#eval IO.println (verdictDecode.line "lora_shrink_bf16/decode")
#eval IO.println (verdictPrefill.line "lora_shrink_bf16/prefill")
-- Agreement resting on the assumed dots and atomic lanes. `decode`: per slice, 2 dots (parts 0
-- and 1) and 64 programs × 2 rows × 16 columns of atomic lanes: `2 · (2 + 2048)`. `prefill`: 64
-- dots (8 parts × 4 row tiles × 2 column tiles) and `8 · 2 · 123 · 16` atomic lanes (each row of
-- the group's 123 in one tile per column tile and part).
#guard let v := verdictDecode; v == .conditional 4100 && v.agrees && v.assumed > 0
#guard let v := verdictPrefill; v == .conditional (64 + 8 * 2 * 123 * 16) && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (`decode` with the two slices' weights
-- swapped; `prefill` with `scaling = 1/2`).
#guard (Verdict.ofLaunch (agreeOf decode (initOf decode) decode.scaling.toRat (fun s => 1 - s)) (runOf decode)).isDiscrepancy
#guard (Verdict.ofLaunch (agreeOf prefill (initOf prefill) (1 / 2) id) (runOf prefill)).isDiscrepancy
