import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `attn_bwd_f16` run by `execLaunch` on small covered instances. Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says on these inputs, both halves: (a) every element of `dv`, `dk`,
`dq` is as `OutSpec` says (the heads are covered: `HeadV.Ok`, see below); (b) every other byte is
unchanged (the six inputs, sentinels on both sides of each output), and no byte is added.

The claim's block sizes are fixed (`BLOCK_N1 = BLOCK_M2 = 128`, `128 ∣ N_CTX`): the smallest
covered launch is `N_CTX = 128`, one program per head; `HEAD_DIM = 16`, the smallest covered.

**The inputs** make every intermediate value exact. `K` and `Q` are in `{-1, 0, 1}` in columns
`0, 1`; in the other columns they are in `[-2, 2]`, paired so that columns `2i` and `2i + 1`
cancel in `K·Q` (`K` equal on both, `Q` opposite). So every score `s = K[n]·Q[m]` is an integer in
`[-2, 2]`; `L[m] = M[m]` is the integer `max_n s(n, m)`, so every exponent `s − L` is an integer
in `[-4, 0]` and `P = 2^(s − L)` a power of two at least `1/16`. `V`, `dO` are integers in
`[-2, 2]` and `Δ` in `[-8, 8]`, so `dP − Δ` is an integer of magnitude below `128` and
`dS = P·(dP − Δ)` is exact in `float16` (checked below), as is `P`. Every sum the kernel
accumulates (`dV`, `Σ dS·Q`, `Σ dS·K`) is then a multiple of `1/16` of magnitude below `2^15`,
exact in `float32` whatever the order. `sm_scale` is the test's `0.5`.

**The platform answers are assumed** (the verdicts are `conditional n`, `n` the number of
platform requests: five per step of the `dk`/`dv` loop, `tl.dot(k, qT)`, `exp2`, `dv + tl.dot`,
`tl.dot(v, doᵀ)`, `dk + tl.dot`; four per step of the `dq` loop, `tl.dot(q, kT)`, `exp2`,
`tl.dot(do, vT)`, `dq + tl.dot`), each the value any correct device gives on these operands:
- every `tl.dot` (with or without an accumulator): the exact value, answered only when it is a
  `float32`; every partial sum is exact too (the magnitudes' sum is below `2^15`, a multiple of
  `1/16`), so any order and any correct accumulation gives it;
- `tl.math.exp2` at an integer `z` (`-4 ≤ z ≤ 0`, masked lanes included): `2^z`, exact.

**The check.** Each output's exact value `E` (`dVE`, `dKE`, `dQE`: `P`, `dS` and the sums computed
here as rationals from the inputs) is a `float32` sum times `1`, `sm_scale` or `LN2`; the witness of
`OutSpec` is `v = float32(E)` (exact for `dV` and `dK`, one rounding for `dQ`), and the check is
`OutSpec` with `T` replaced by a lower bound `Tlow` of it: `0` for `dV`, `2^-24·|E| + 2^-150`
for `dK` and `dQ` (`dKTol` and `dQTol` are that plus nonnegative terms: `ds16Err`, `accB`,
`lift16`, `lm` are nonnegative, `pBr ≥ 2^xE` since `eX ≥ 0`). Then `|v − E| ≤ Tlow ≤ T`,
`c` must be `float16(v)` (`NaNEq`), and when `|E| + Tlow ≤ 65504` (implied by `|E| + T ≤ 65504`)
`c` is finite and within `Tlow + (|E| + Tlow)·2^-11 + 2^-25`, which is at most `Post`'s
`T + (|E| + T)·2^-11 + 2^-25`: the check implies `Post`, never looser. -/

open TVA TVA.Validation Claims.AttnBwd

/-- A platform that relates nothing: the platform answers come from the oracle. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits (no load of this kernel is masked). -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle -/

/-- Rationals as integers over a common denominator. -/
def scaleInts (xs : Array Rat) : Nat × Array Int :=
  let den := xs.foldl (fun m q => Nat.lcm m q.den) 1
  (den, xs.map fun q => (q * (den : Rat)).num)

/-- A rational as an element of a float accumulator dtype, when it is exactly representable. -/
def accOf : (d : DType) → Rat → Option d.Val
  | .f32, q => let v : F32 := FloatV.ofRat q; if v.toRat == q then some v else none
  | _, _ => none

/-- The exact `acc + a·b` of a `tl.dot` request from its own operands (finite floats), answered
only when every element of the result is an exact `float32`. -/
def exactDot : (q : PlatQ) → Option q.Out
  | @PlatQ.dot cfg M N K a b acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => a.get2 i k).toArray.mapM (cfg.a.val? ·)
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => b.get2 k j).toArray.mapM (cfg.b.val? ·)
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM (cfg.acc.val? ·)
    let (da, Ai) := scaleInts A
    let (db, Bi) := scaleInts B
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      let s : Int := (List.range K).foldl (fun s k => s + Ai[i * K + k]! * Bi[k * N + j]!) 0
      accOf cfg.acc ((s : Rat) / ((da * db : Nat) : Rat) + C[e]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- `2^z` at a finite integer `z` in `[-149, 100]`, exact; nothing otherwise. -/
def exp2Lane {f : Format} (x : FloatV f) : Option (FloatV f) :=
  if x.isFinite && x.toRat.den == 1 && -149 ≤ x.toRat.num && x.toRat.num ≤ 100 then
    some (FloatV.ofRat ((2 : Rat) ^ x.toRat.num))
  else none

/-- The position of an index in row-major order (as `Tensor.toList`). -/
def flatIdx : (s : Shape) → Idx s → Nat
  | [], _ => 0
  | _ :: s, (i, j) => i.1 * s.foldr (· * ·) 1 + flatIdx s j

/-- A tensor read from an array in row-major order: the oracle's answers are stored, not
recomputed from their operands at every read (which keeps the interpreter's time down). -/
def ofArr {s : Shape} {f : Format} (a : Array (FloatV f)) : Tensor s (FloatV f) :=
  ⟨fun i => a.getD (flatIdx s i) (FloatV.ofPat 0)⟩

/-- Lane by lane, when every lane has an answer. -/
def lanewise {s : Shape} {f : Format} (g : FloatV f → Option (FloatV f)) (x : Tensor s (FloatV f)) :
    Option (Tensor s (FloatV f)) :=
  (x.toList.mapM g).map fun l => ofArr l.toArray

/-- The oracle: exact dots and exact `exp2`; nothing else (trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat (.math .exp2 _ x) => lanewise exp2Lane x
  | .plat q => exactDot q
  | _ => none

/-! ## The inputs (head `bh`, row `r`, column `c`) -/

def kI (bh r c : Nat) : Int :=
  if c < 2 then ((2 * r + 3 * c + bh) % 3 : Nat) - (1 : Int)
  else ((r + 3 * (c / 2) + 2 * bh) % 5 : Nat) - (2 : Int)
def qI (bh r c : Nat) : Int :=
  if c < 2 then ((r + 2 * c + (r / 5) + bh) % 3 : Nat) - (1 : Int)
  else (if c % 2 = 0 then 1 else -1) * ((((3 * r + c / 2 + bh) % 5 : Nat) : Int) - 2)
def vI (bh r c : Nat) : Int := ((3 * r + 5 * c + bh) % 5 : Nat) - (2 : Int)
def doI (bh r c : Nat) : Int := ((7 * r + 2 * c + 3 * bh + r / 3) % 5 : Nat) - (2 : Int)
def dlI (bh r : Nat) : Int := ((5 * r + 7 * bh) % 17 : Nat) - (8 : Int)

/-- The score `Σ_c K[n, c]·Q[m, c]` and `L[m] = max_n` of it. -/
def sI (d : Inst) (bh n m : Nat) : Int := ((List.range d.HEAD_DIM).map fun c => kI bh n c * qI bh m c).sum
def lseI (d : Inst) (bh m : Nat) : Int := ((List.range d.N_CTX).map fun n => sI d bh n m).foldl max (-1000)
/-- `dP[n, m] = Σ_c V[n, c]·dO[m, c]`. -/
def dpI (d : Inst) (bh n m : Nat) : Int := ((List.range d.HEAD_DIM).map fun c => vI bh n c * doI bh m c).sum

def f16v (z : Int) : F16 := FloatV.ofRat (z : Rat)
def f32v (z : Int) : F32 := FloatV.ofRat (z : Rat)

/-- Sentinels in every output element before the launch, and in `pad` elements on each side of
each output. -/
def sent : F16 := FloatV.ofRat (-999)
def pad : Nat := 8

def initOf (d : Inst) : CMem :=
  let all := (List.range (G d * d.N_CTX)).flatMap fun r : Nat => (List.range d.HEAD_DIM).map fun c : Nat =>
    (r / d.N_CTX, r % d.N_CTX, c)
  let m := CMem.ofElems ∅ .f16 (all.map fun (b, r, c) => ((d.Q.ptr (eIdx d b r c)).addr, f16v (qI b r c)))
  let m := CMem.ofElems m .f16 (all.map fun (b, r, c) => ((d.K.ptr (eIdx d b r c)).addr, f16v (kI b r c)))
  let m := CMem.ofElems m .f16 (all.map fun (b, r, c) => ((d.V.ptr (eIdx d b r c)).addr, f16v (vI b r c)))
  let m := CMem.ofElems m .f16 (all.map fun (b, r, c) => ((d.DO.ptr (eIdx d b r c)).addr, f16v (doI b r c)))
  let rowsL := (List.range (G d * d.N_CTX)).map fun r : Nat => (r / d.N_CTX, r % d.N_CTX)
  let m := CMem.ofElems m .f32 (rowsL.map fun (b, r) => ((d.M.ptr (rIdx d b r)).addr, f32v (lseI d b r)))
  let m := CMem.ofElems m .f32 (rowsL.map fun (b, r) => ((d.D.ptr (rIdx d b r)).addr, f32v (dlI b r)))
  let n := G d * d.N_CTX * d.HEAD_DIM + 2 * pad
  let sents (B : Buf .f16) := (List.range n).map fun j : Nat => ((B.ptr ((j : Int) - pad)).addr, sent)
  CMem.ofElems (CMem.ofElems (CMem.ofElems m .f16 (sents d.DQ)) .f16 (sents d.DK)) .f16 (sents d.DV)

def runOf (d : Inst) : LRes (launch d).size × List Record :=
  execLaunch ((launch d).env P0) O fill0 1000000 (model d) (launch d) (initOf d)

/-! ## The expectation -/

/-- The exact values of one head: `dV`, `Σ dS·Q`, `Σ dS·K` (row-major, `N_CTX × HEAD_DIM`), with
the mask `allowed` and the `Δ` shift `dlShift` (the test's: `Allowed`, `0`). -/
structure HeadE where
  dV : Array Rat
  dKS : Array Rat
  dQS : Array Rat

/-- `L[m]` of every row of head `bh` (computed once: `lseI` is a loop over the keys). -/
def lseArr (d : Inst) (bh : Nat) : Array Int := (Array.range d.N_CTX).map (lseI d bh)

def pR (d : Inst) (allowed : Nat → Nat → Bool) (Ls : Array Int) (bh n m : Nat) : Rat :=
  if allowed n m then (2 : Rat) ^ (sI d bh n m - Ls[m]!) else 0

def headE (d : Inst) (allowed : Nat → Nat → Bool) (dlShift : Rat) (bh : Nat) : HeadE :=
  let N := d.N_CTX
  let D := d.HEAD_DIM
  let Ls := lseArr d bh
  let P := (Array.range (N * N)).map fun e => pR d allowed Ls bh (e / N) (e % N)
  let dS := (Array.range (N * N)).map fun e =>
    P[e]! * ((dpI d bh (e / N) (e % N) : Rat) - (dlI bh (e % N) + dlShift))
  let dV := (Array.range (N * D)).map fun e => ((List.range N).map fun m =>
    P[(e / D) * N + m]! * doI bh m (e % D)).sum
  let dKS := (Array.range (N * D)).map fun e => ((List.range N).map fun m =>
    dS[(e / D) * N + m]! * qI bh m (e % D)).sum
  let dQS := (Array.range (N * D)).map fun e => ((List.range N).map fun n =>
    dS[n * N + e / D]! * kI bh n (e % D)).sum
  { dV, dKS, dQS }

/-- The check of one output element `c` against `OutSpec` with the exact value `E` and the lower
bound `Tlow` of the tolerance (module doc): the witness `v = float32(E)`. -/
def outOk (c : F16) (E Tlow : Rat) : Bool :=
  let v : F32 := FloatV.ofRat E
  let cv : F16 := FloatV.cvt v
  v.isFinite && |v.toRat - E| ≤ Tlow &&
  (cv.pat == c.pat || (cv.isNaN && c.isNaN)) &&
  (!(|E| + Tlow ≤ 65504) ||
    (c.isFinite && |c.toRat - E| ≤ Tlow + (|E| + Tlow) / 2 ^ 11 + 1 / 2 ^ 25))

/-- `2^-24·|E| + 2^-150`. -/
def tRound (E : Rat) : Rat := |E| / 2 ^ 24 + 1 / 2 ^ 150

/-- The final memory against `Post` with the expectation `H` and the scales `sk` (`sm_scale`'s
`float32`) and `sq` (`LN2`): every output element as `outOk` says; every byte outside the three
outputs unchanged, none added. -/
def agreeOf (d : Inst) (init : CMem) (H : Nat → HeadE) (sk sq : Rat) (cf : CConfig (launch d).size) :
    Option String :=
  let m := CMem.toMem cf.mem
  let N := d.N_CTX
  let D := d.HEAD_DIM
  let bad := (List.range (G d)).flatMap fun b =>
    let h := H b
    ((List.range (N * D)).filter fun e : Nat =>
      let r := e / D
      let j := e % D
      let rd (B : Buf .f16) := m.read .f16 (B.ptr (eIdx d b r j)).addr
      let ok (B : Buf .f16) (E Tlow : Rat) := match rd B with
        | some c => outOk c E Tlow
        | none => false
      let eK := sk * h.dKS[e]!
      let eQ := sq * h.dQS[e]!
      !(ok d.DV h.dV[e]! 0 && ok d.DK eK (tRound eK) && ok d.DQ eQ (tRound eQ))).map fun e => (b, e / D, e % D)
  let outA (a : Int) : Bool :=
    let n := 2 * (G d * N * D : Nat)
    (d.DQ.base ≤ a && a < d.DQ.base + n) || (d.DK.base ≤ a && a < d.DK.base + n) ||
    (d.DV.base ≤ a && a < d.DV.base + n)
  let badElse := init.toList.filter fun (a, b) => !outA a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"output elements (head, row, col) {bad.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside dq, dk, dv changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def allowedOf (d : Inst) (n m : Nat) : Bool := !d.CAUSAL || n ≤ m

/-! ## The cases -/

/-- Not causal, `Z = 1`, `H = 2`, `N_CTX = 128`, `HEAD_DIM = 16`: two programs (the second at
`bhid = 1`: `adj = stride_h`, `off_chz = N_CTX`), each four steps of `32` query rows for `dk`,
`dv` and four of `32` key rows for `dq`. -/
def full : Inst := canon 1 2 128 16 false smTest
/-- Causal, `Z = 2`, `H = 1`: two programs (the second at `bhid = 1`: `adj = stride_z`), each eight
masked steps of `16` rows for `dk`, `dv` and eight for `dq`, then the empty unmasked loops. -/
def causal : Inst := canon 2 1 128 16 true smTest

theorem full_ok : full.ok :=
  canon_ok 1 2 128 16 false smTest (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) smTest_ok
theorem causal_ok : causal.ok :=
  canon_ok 2 1 128 16 true smTest (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) smTest_ok

-- `Pre`: the nine buffers aligned, every byte present in `initOf`.
#guard [full, causal].all fun d =>
  let m := CMem.toMem (initOf d)
  let all4 (B : Buf .f16) := B.base % 2 == 0 && (List.range B.len).all fun e : Nat => (m.read .f16 (B.ptr e).addr).isSome
  let all1 (B : Buf .f32) := B.base % 4 == 0 && (List.range B.len).all fun e : Nat => (m.read .f32 (B.ptr e).addr).isSome
  all4 d.Q && all4 d.K && all4 d.V && all4 d.DO && all4 d.DQ && all4 d.DK && all4 d.DV &&
  all1 d.M && all1 d.D

-- The inputs are encoded exactly (and finite); `sm_scale`'s `float32` is `1/2`.
#guard [full, causal].all fun d => (List.range (G d)).all fun b => (List.range d.N_CTX).all fun r =>
  (f32v (lseI d b r)).toRat == lseI d b r && (f32v (dlI b r)).toRat == dlI b r &&
  (List.range d.HEAD_DIM).all fun c =>
    (f16v (qI b r c)).toRat == qI b r c && (f16v (kI b r c)).toRat == kI b r c &&
    (f16v (vI b r c)).toRat == vI b r c && (f16v (doI b r c)).toRat == doI b r c
#guard (smF full).toRat == 1 / 2 && (smF causal).toRat == 1 / 2

/-- `eSub` of `Post` over the rationals (`Device.relDot16`, `Device.absDot16`, `u32 = 2^-24`,
`eta32 = 2^-150`), with the magnitudes of integer operands (no subnormals: `lm x = |x|`). -/
def eSubQ (d : Inst) (s a x : Rat) : Rat :=
  Device.relDot16 d.HEAD_DIM * a + Device.absDot16 d.HEAD_DIM +
    3 / 2 ^ 24 * (|s| + Device.relDot16 d.HEAD_DIM * a + Device.absDot16 d.HEAD_DIM + |x|) + 4 / 2 ^ 150

-- `HeadV.Ok`: the inputs finite (above); `|L|, |Δ| ≤ 8 ≤ 2^126`; and for every pair, the exponent
-- `xE = s − L ≤ 0` and its error `eX ≤ 1`, so `xE + eX ≤ 15` and `2^(xE + eX) ≤ 2`, whence
-- `pBr ≤ (1 + 2^-22)·2 + 2^-126 ≤ 3`; and `3·yB = 3·(|dP − Δ| + eY) ≤ 2^15`. Also: the scores
-- and exponents are as the module doc says, and `P` and `dS` are exact `float16` values.
#guard [full, causal].all fun d => (List.range (G d)).all fun b =>
  let Ls := lseArr d b
  (List.range d.N_CTX).all fun n => (List.range d.N_CTX).all fun m =>
    let s := sI d b n m
    let L := Ls[m]!
    let aQK : Rat := ((List.range d.HEAD_DIM).map fun c => ((kI b n c).natAbs * (qI b m c).natAbs : Nat)).sum
    let aVD : Rat := ((List.range d.HEAD_DIM).map fun c => ((vI b n c).natAbs * (doI b m c).natAbs : Nat)).sum
    let y : Rat := dpI d b n m - dlI b m
    let p : Rat := (2 : Rat) ^ (s - L)
    s.natAbs ≤ 2 && -4 ≤ s - L && s - L ≤ 0 && eSubQ d s aQK L ≤ 1 &&
    3 * (|y| + eSubQ d (dpI d b n m) aVD (dlI b m)) ≤ 2 ^ 15 && |y| < 128 &&
    (FloatV.ofRat p : F16).toRat == p && (FloatV.ofRat (p * y) : F16).toRat == p * y

-- The expectations are not trivial: every output has nonzero elements, and the causal mask
-- changes them.
#guard [full, causal].all fun d => (List.range (G d)).all fun b =>
  let h := headE d (allowedOf d) 0 b
  h.dV.any (· != 0) && h.dKS.any (· != 0) && h.dQS.any (· != 0)

/-- The verdict of a run `r` of `d` against the expectation with the mask `allowed`, the `Δ`
shift `dl` and the factor `sq` of `dq` (the test's: `Allowed`, `0`, `LN2`). -/
def verdictWith (d : Inst) (r : LRes (launch d).size × List Record) (allowed : Nat → Nat → Bool)
    (dl sq : Rat) : Verdict :=
  Verdict.ofLaunch (agreeOf d (initOf d) (headE d allowed dl) (smF d).toRat sq) r

/-- One case, on one run: print its verdict, and fail (an error, which fails the test as a failed
`#guard` does) unless the verdict is `conditional n` and every wrong expectation (a mask, a `Δ`
shift, a factor of `dq`) is a discrepancy on the same run. A run takes about 20 s in the
interpreter, and a `#guard` after the `#eval` would run it again: so the guards on a run are
checked in the command that prints its verdict. -/
def check (name : String) (d : Inst) (n : Nat) (bad : List ((Nat → Nat → Bool) × Rat × Rat)) :
    IO Unit := do
  let r := runOf d
  let v := verdictWith d r (allowedOf d) 0 ln2F.toRat
  IO.println (v.line s!"attn_bwd_f16/{name}")
  unless v == .conditional n do throw (IO.userError s!"{name}: not conditional {n}")
  for ((al, dl, sq), k) in bad.zipIdx do
    unless (verdictWith d r al dl sq).isDiscrepancy do
      throw (IO.userError s!"{name}: wrong expectation {k} not caught")

-- Agreement resting on one assumed answer per platform request (2 programs × (4 steps × 5 +
-- 4 steps × 4) not causal; 2 × (8 × 5 + 8 × 4) causal). A wrong expectation is caught: `Δ` off
-- by one, `dq` without the factor `LN2` (not causal); the causal run against the unmasked
-- expectation.
#eval check "full" full 72 [(allowedOf full, 1, ln2F.toRat), (allowedOf full, 0, 1)]
#eval check "causal" causal 144 [(fun _ _ => true, 0, ln2F.toRat)]
