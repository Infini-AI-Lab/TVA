import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `recompute_wu_bf16` run by `execLaunch` on three small covered
instances (the claim's own `canon`): a padded batch with a `float32` `beta` and grouped value
heads (`HV = 2·H`); a packed batch with a `bfloat16` `beta`, `int32` metadata, an empty sequence and
`chunk_indices` listed out of order; a packed batch with `int64` metadata whose second sequence
spans two 64-token chunks, with `K = 65` (two column tiles of `w`, the second masked past one
column). Not a test against the device (phase 4): it checks that `exec` runs the model end to
end, and that the final memory is what `Post` says, both halves: every element of `u` and `w`
(each a finite `bfloat16`, at distance `0` from `uRef`, `wRef`, which `Post` allows), and every
other byte unchanged (the inputs, the metadata, sentinels on both sides of `u` and `w`), with no
byte added.

The expectations are computed here from the inputs as rationals:
`u[s0+t, h, c] = Σ_j A[s0+t, h, j]·v[f_j, h, c]·beta[f_j, h]` and
`w[s0+t, h, c] = Σ_j A[s0+t, h, j]·k[f_j, h/(HV/H), c]·beta[f_j, h]·2^g[f_j, h]`, `j < 64` over the
columns whose token `f_j = s0 + t/64·64 + j` lies in the sequence (`Post`'s `uRef`, `wRef`). The
inputs are small: `k, v ∈ [-2, 2]` integers, `A ∈ {-1, 0, 1}`, `beta ∈ {1/2, 1, 2}`, `g ∈ {-1, 0, 1}`
integers. So every product the kernel rounds (`v·beta`, `k·beta`, `·2^g`, in `float32` or
`bfloat16`) is exact, and every operand `Post`'s `vbV` names equals the exact product (checked
below), every partial sum of a dot is a multiple of `1/4` far inside `float32`, and every result is
a `bfloat16` (checked below): the outputs are exact.

**Platform answers are assumed, not proved.** Two platform operations run: `tl.math.exp2` on the
gate (line 121), once per program, and the two `tl.dot`s (lines 116, 132), once per program and
column tile. The oracle answers `exp2` lane by lane with `2^g` exactly (`g` an integer in
`[-1, 1]`, or `0` on the masked lanes: a power of two, what the device's `ex2.approx` gives on an
integer; `Device.Exp2F32` alone would allow `2^g·(1 ± 2^-22)`, so this is an assumption), and
each dot with the exact `Σ A·b` from its own operands (only when every operand is finite and the
sum is a `float32`; every partial sum is then exact whatever the order, so the exact value is the
only one a correct device gives). The verdicts are `conditional n`, `n` the number of answers. -/

open TVA TVA.Validation Claims.RecomputeWU

/-- A platform that relates nothing: the platform values come from the oracle. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle: exact `exp2` of integers, exact dots -/

/-- An accumulator element holding `q`, when the accumulator is `float32` and holds `q` exactly. -/
def accOfR : (d : DType) → Rat → Option d.Val
  | .f32, q => let v : F32 := FloatV.ofRat q; if v.toRat == q then some v else none
  | _, _ => none

/-- `2^n` for an integer `n`. -/
def pow2 (n : Int) : Rat := if 0 ≤ n then ((2 ^ n.toNat : Nat) : Rat) else 1 / ((2 ^ (-n).toNat : Nat) : Rat)

/-- The exact answer to a platform request: `2^x` for `exp2` on integer lanes in `[-60, 60]`; the
exact `acc + a·b` of a `tl.dot` whose operands are all finite and whose result is a `float32`;
nothing otherwise. -/
def exactPlat : (q : PlatQ) → Option q.Out
  | @PlatQ.math .exp2 _ _ x =>
    if x.toList.all (fun v => v.isFinite && v.toRat.den == 1 && decide (-60 ≤ v.toRat.num) &&
        decide (v.toRat.num ≤ 60)) then
      some (x.map fun v => FloatV.ofRat (pow2 v.toRat.num))
    else none
  | @PlatQ.dot cfg M N K a b acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => a.get2 i k).toArray.mapM (cfg.a.val? ·)
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => b.get2 k j).toArray.mapM (cfg.b.val? ·)
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM (cfg.acc.val? ·)
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      accOfR cfg.acc ((List.range K).foldl (fun s k =>
        let x := A[i * K + k]!
        if x == 0 then s else s + x * B[k * N + j]!) C[e]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- The oracle: `exactPlat` at every platform request; nothing else. -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactPlat q
  | _ => none

/-! ## The inputs -/

/-- `k[f, h, c]`, `v[f, h, c]`: integers in `[-2, 2]`. -/
def kval (f h c : Nat) : Int := ((7 * f + 5 * h + 3 * c + 1) % 5 : Nat) - (2 : Int)
def vval (f h c : Nat) : Int := ((3 * f + 7 * h + 2 * c + 2) % 5 : Nat) - (2 : Int)
/-- `A[f, h, j]`: `-1`, `0` or `1`, about one entry in four nonzero. -/
def aval (f h j : Nat) : Int :=
  if (5 * f + 3 * h + 7 * j) % 4 = 1 then (if (f + 2 * j + h) % 3 = 0 then -1 else 1) else 0
/-- `beta[f, h]` in `{1/2, 1, 2}`. -/
def betaval (f h : Nat) : Rat := [1 / 2, 1, 2].getD ((f + h) % 3) 1
/-- `g[f, h]` in `{-1, 0, 1}`. -/
def gval (f h : Nat) : Int := ((2 * f + h) % 3 : Nat) - (1 : Int)

def bf (q : Rat) : BF16 := FloatV.ofRat q
def f32 (q : Rat) : F32 := FloatV.ofRat q

/-- A sentinel in every element of `u` and `w` before the launch, and in `pad` elements on each
side of each. -/
def sentinel : Rat := -99
def pad : Nat := 4

/-- An index element (`int32` or `int64`). -/
def ival (d : DType) (z : Int) : d.Val := d.ofBits (z % (2 ^ (8 * d.bytes) : Int)).toNat

/-- One case: the claim's instance, and for a packed batch `cu_seqlens` and `chunk_indices`. -/
structure Case where
  inst : Inst
  cu : List Nat
  ci : List (Nat × Nat)

/-- The sequences `(s0, len)`, as `Post` reads them. -/
def Case.seqs (c : Case) : List (Nat × Nat) :=
  if c.inst.isVarlen then (List.range c.inst.N).map fun n => (c.cu.getD n 0, c.cu.getD (n + 1) 0 - c.cu.getD n 0)
  else (List.range c.inst.B).map fun b => (b * c.inst.T, c.inst.T)

/-- The sequence holding flat token `f`. -/
def Case.seqOf (c : Case) (f : Nat) : Nat × Nat :=
  (c.seqs.find? fun (s0, len) => s0 ≤ f && f < s0 + len).getD (0, 0)

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let nt := nTok d
  let triples (F H C : Nat) := (List.range F).flatMap fun f => (List.range H).flatMap fun h =>
    (List.range C).map fun x => (f, h, x)
  let m := CMem.ofElems ∅ .bf16 ((triples nt d.H d.K).map fun (f, h, x) =>
    ((d.k.ptr (kIdx d f h x : Nat)).addr, bf (kval f h x)))
  let m := CMem.ofElems m .bf16 ((triples nt d.HV d.V).map fun (f, h, x) =>
    ((d.v.ptr (vIdx d f h x : Nat)).addr, bf (vval f h x)))
  let m := CMem.ofElems m .bf16 ((triples nt d.HV 64).map fun (f, h, j) =>
    ((d.A.ptr (aIdx d f h j : Nat)).addr, bf (aval f h j)))
  let pairs := (List.range nt).flatMap fun f => (List.range d.HV).map fun h => (f, h)
  let m := if d.bbf then
      CMem.ofElems m .bf16 (pairs.map fun (f, h) => ((d.betab.ptr (hIdx d f h : Nat)).addr, bf (betaval f h)))
    else
      CMem.ofElems m .f32 (pairs.map fun (f, h) => ((d.beta.ptr (hIdx d f h : Nat)).addr, f32 (betaval f h)))
  let m := CMem.ofElems m .f32 (pairs.map fun (f, h) => ((d.g.ptr (hIdx d f h : Nat)).addr, f32 (gval f h)))
  let m := CMem.ofElems m .bf16 ((List.range (nt * d.HV * d.V + 2 * pad)).map fun e : Nat =>
    ((d.u.ptr ((e : Int) - pad)).addr, bf sentinel))
  let m := CMem.ofElems m .bf16 ((List.range (nt * d.HV * d.K + 2 * pad)).map fun e : Nat =>
    ((d.w.ptr ((e : Int) - pad)).addr, bf sentinel))
  if d.isVarlen then
    let m := CMem.ofElems m d.idx ((List.range c.cu.length).map fun n : Nat =>
      ((d.cu.ptr n).addr, ival d.idx (c.cu.getD n 0)))
    CMem.ofElems m d.idx ((List.range c.ci.length).flatMap fun i : Nat =>
      [((d.ci.ptr (2 * i : Nat)).addr, ival d.idx (c.ci.getD i (0, 0)).1),
       ((d.ci.ptr (2 * i + 1 : Nat)).addr, ival d.idx (c.ci.getD i (0, 0)).2)])
  else m

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) O fill0 10000 (model c.inst) (launch c.inst) c.init

/-- The columns `j < 64` of flat token `f`'s row whose token is in its sequence, with that token. -/
def Case.cols (c : Case) (f : Nat) : List (Nat × Nat) :=
  let (s0, len) := c.seqOf f
  let t := f - s0
  ((List.range 64).filter fun j => t / 64 * 64 + j < len).map fun j => (j, s0 + t / 64 * 64 + j)

/-- `v·beta`, the operand `Post`'s `vbV` names (exact here). -/
def vb (f h x : Nat) : Rat := vval f h x * betaval f h

/-- What `Post` says `u[f, h, c]` holds on these inputs: `uRef`, exact. -/
def Case.uExp (c : Case) (f h x : Nat) : Rat :=
  ((c.cols f).map fun (j, fj) => (aval f h j : Rat) * vb fj h x).sum

/-- What `Post` says `w[f, h, c]` holds on these inputs: `wRef`, exact. -/
def Case.wExp (c : Case) (f h x : Nat) : Rat :=
  let kh := h / (c.inst.HV / c.inst.H)
  ((c.cols f).map fun (j, fj) => (aval f h j : Rat) * kval fj kh x * betaval fj h * pow2 (gval fj h)).sum

/-- The final memory against expectations `uE`, `wE`: every element of `u` and `w` a finite
`bfloat16` of that value; every byte outside them unchanged; no byte added. -/
def Case.agree (c : Case) (uE wE : Nat → Nat → Nat → Rat) (cf : CConfig (launch c.inst).size) :
    Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let nU := nTok d * d.HV * d.V
  let nW := nTok d * d.HV * d.K
  let badU := (List.range nU).filter fun e : Nat =>
    ((m.read .bf16 (d.u.ptr e).addr).bind (DType.bf16.val? ·)) !=
      some (uE (e / (d.HV * d.V)) (e / d.V % d.HV) (e % d.V))
  let badW := (List.range nW).filter fun e : Nat =>
    ((m.read .bf16 (d.w.ptr e).addr).bind (DType.bf16.val? ·)) !=
      some (wE (e / (d.HV * d.K)) (e / d.K % d.HV) (e % d.K))
  let inOut (a : Int) : Bool :=
    (d.u.base ≤ a && a < d.u.base + 2 * (nU : Int)) || (d.w.base ≤ a && a < d.w.base + 2 * (nW : Int))
  let badElse := c.init.toList.filter fun (a, b) => !inOut a && cf.mem[a]? != some b
  if !badU.isEmpty then some s!"u elements {badU.take 8} differ from Post"
  else if !badW.isEmpty then some s!"w elements {badW.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside u and w changed"
  else if cf.mem.size != c.init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch (c.agree c.uExp c.wExp) c.run

/-! ## The cases -/

/-- Padded, `float32` `beta`: `B = 2`, `T = 5`, `H = 1`, `HV = 2` (both value heads read key head
`0`), `K = 3`, `V = 2`. Grid `(1, 4)`: one chunk per sequence, rows `5..63` masked. -/
def paddedI : Inst := canon .i64 false 2 5 1 2 3 2 2 1 false
theorem paddedI_ok : paddedI.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)
def padded : Case := { inst := paddedI, cu := [], ci := [] }

/-- Packed, `bfloat16` `beta`, `int32` metadata: `T = 7`, `cu_seqlens = [0, 3, 3, 7]` (sequence `1`
empty), `chunk_indices = [(2, 0), (0, 0)]` (the pairs in reverse order), `H = HV = 2`, `K = 2`,
`V = 3`. Grid `(2, 2)`. -/
def packedI : Inst := canon .i32 true 1 7 2 2 2 3 3 2 true
theorem packedI_ok : packedI.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)
def packed : Case := { inst := packedI, cu := [0, 3, 3, 7], ci := [(2, 0), (0, 0)] }

/-- Packed, `float32` `beta`, `int64` metadata: `T = 68`, `cu_seqlens = [0, 2, 68]` (sequence `1`
spans two chunks: `64` tokens and `2`), `chunk_indices = [(0, 0), (1, 0), (1, 1)]`, `H = HV = 1`,
`K = 65` (two column tiles of `w`), `V = 1`. Grid `(3, 1)`. -/
def twoChunksI : Inst := canon .i64 false 1 68 1 1 65 1 2 3 true
theorem twoChunksI_ok : twoChunksI.ok := by
  constructor <;> first
    | decide
    | (apply bytes_disj; decide)
def twoChunks : Case := { inst := twoChunksI, cu := [0, 2, 68], ci := [(0, 0), (1, 0), (1, 1)] }

def cases : List Case := [padded, packed, twoChunks]

/-! ## Coverage (`Pre`'s decidable parts) and exactness -/

-- Every buffer is aligned and inside `[0, 2^64)`, and every element of it is present in `init`
-- (`Buf.Valid`; with `u` and `w`, which hold the sentinel).
#guard cases.all fun c =>
  let d := c.inst
  let m := CMem.toMem c.init
  let okB {dt : DType} (b : Buf dt) : Bool :=
    decide (0 ≤ b.base) && decide (b.base + b.len * dt.bytes ≤ 2 ^ 64) && decide (b.base % dt.bytes = 0) &&
    (List.range b.len).all fun e : Nat => (m.read dt (b.ptr e).addr).isSome
  okB d.k && okB d.v && okB d.A && okB d.g && okB d.u && okB d.w &&
    (d.bbf || okB d.beta) && (!d.bbf || okB d.betab) && (!d.isVarlen || (okB d.cu && okB d.ci))

-- The metadata are the instance's (`N`, `NT`, `T`) and as fla builds them (`SeqOk`): `cu_seqlens`
-- non-decreasing from `0` to `T`; `chunk_indices` lists each (sequence, chunk) pair once.
#guard cases.all fun c =>
  let d := c.inst
  !d.isVarlen || (
    c.cu.length == d.N + 1 && c.ci.length == d.NT && c.cu.head? == some 0 && c.cu.getLast? == some d.T &&
    (List.range d.N).all (fun n => c.cu.getD n 0 ≤ c.cu.getD (n + 1) 0) &&
    c.ci.all (fun (n, j) => n < d.N && j * 64 < c.cu.getD (n + 1) 0 - c.cu.getD n 0) &&
    (List.range d.N).all (fun n => (List.range 64).all fun j =>
      !(j * 64 < c.cu.getD (n + 1) 0 - c.cu.getD n 0) || c.ci.count (n, j) == 1) &&
    c.ci.Nodup)

-- The values are in `Pre`'s ranges and encoded exactly: `k, v` within `2^32`, `A` within `2^64`,
-- `beta` within `2^8`, `g ≤ 8`.
#guard (List.range 5).all fun z : Nat => (bf (z - 2 : Int)).toRat == ((z - 2 : Int) : Rat)
#guard [1 / 2, 1, 2].all fun q : Rat => (bf q).toRat == q && (f32 q).toRat == q
#guard (List.range 3).all fun z : Nat => (f32 (z - 1 : Int)).toRat == ((z - 1 : Int) : Rat)

-- Every operand `Post`'s `vbV` names is the exact product (`v·beta` a `bfloat16`), as is every
-- `w` operand `k·beta·2^g` (so lines 129–132 round nothing).
#guard cases.all fun c => let d := c.inst
  (List.range (nTok d)).all fun f => (List.range d.HV).all fun h =>
    ((List.range d.V).all fun x => (bf (vb f h x)).toRat == vb f h x) &&
    ((List.range d.K).all fun x =>
      let p := (kval f (h / (d.HV / d.H)) x : Rat) * betaval f h * pow2 (gval f h)
      (bf p).toRat == p && (bf ((kval f (h / (d.HV / d.H)) x : Rat) * betaval f h)).toRat ==
        (kval f (h / (d.HV / d.H)) x : Rat) * betaval f h)

-- Every expectation is a `bfloat16` (the stores round nothing) and differs from the sentinel.
#guard cases.all fun c => let d := c.inst
  (List.range (nTok d)).all fun f => (List.range d.HV).all fun h =>
    ((List.range d.V).all fun x => (bf (c.uExp f h x)).toRat == c.uExp f h x && c.uExp f h x != sentinel) &&
    ((List.range d.K).all fun x => (bf (c.wExp f h x)).toRat == c.wExp f h x && c.wExp f h x != sentinel)

-- The expectations are not trivial: some `u` and `w` elements are nonzero, some use a column other
-- than the token's own, and in `twoChunks` the second chunk's tokens read `A`'s columns `0, 1`.
#guard cases.all fun c => (List.range (nTok c.inst)).any fun f =>
  ((List.range c.inst.HV).any fun h => (List.range c.inst.V).any fun x => c.uExp f h x != 0) &&
  ((List.range c.inst.HV).any fun h => (List.range c.inst.K).any fun x => c.wExp f h x != 0 && c.wExp f h x != c.uExp f h 0)
#guard (twoChunks.cols 67).map Prod.snd == [66, 67] && (twoChunks.cols 1).map Prod.snd == [0, 1] &&
  (twoChunks.cols 5).length == 64

def verdictPadded : Verdict := padded.verdict
def verdictPacked : Verdict := packed.verdict
def verdictTwo : Verdict := twoChunks.verdict

#eval IO.println (verdictPadded.line "recompute_wu_bf16/padded")
#eval IO.println (verdictPacked.line "recompute_wu_bf16/packed")
#eval IO.println (verdictTwo.line "recompute_wu_bf16/two_chunks")
-- Agreement resting on one `exp2` and `cdiv(V, 64) + cdiv(K, 64)` dots per program.
#guard let v := verdictPadded; v == .conditional (4 * 3) && v.agrees && v.assumed > 0
#guard let v := verdictPacked; v == .conditional (4 * 3) && v.agrees && v.assumed > 0
#guard let v := verdictTwo; v == .conditional (3 * 4) && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (`w` without the gate `2^g`; `u` reading
-- the key head's `k` instead of `v`).
#guard
  let r := packed.run
  (Verdict.ofLaunch (packed.agree packed.uExp fun f h x =>
    let kh := h / (packed.inst.HV / packed.inst.H)
    ((packed.cols f).map fun (j, fj) => (aval f h j : Rat) * kval fj kh x * betaval fj h).sum) r).isDiscrepancy &&
  (Verdict.ofLaunch (packed.agree (fun f h x => packed.uExp f h x + if f = 6 ∧ h = 1 ∧ x = 2 then 1 else 0)
    packed.wExp) r).isDiscrepancy
