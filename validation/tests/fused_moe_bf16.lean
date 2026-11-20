import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `fused_moe_bf16` run by `execLaunch` on two small covered instances.
Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and that
the final memory is what `Post` says on these inputs, both halves: every element of the matrix `C`
(unchanged when its flat index is not routed; `+0.0` when its block's expert is `-1`; otherwise the
`bfloat16` rounding of the accumulator, times the routing weight with `MUL_ROUTED_WEIGHT`), and
every byte outside the matrix unchanged (`A`, `B`, the weights, the expert assignment, sentinels on
both sides of `C`).

The tile is the smallest of `cfgs`, `(16, 32, 64, 1)`: programs of `16 × 32` outputs, `k` steps of
`64`.

**The `tl.dot` values are assumed, not proved.** Each trip's `tl.dot_add` (line 373) is a platform
site (`Site.dotAdd`, asking `PlatQ.dot`): only the oracle can supply its value. The oracle answers
each with the exact `acc + a·b` from the request's own operands, and only when that value is a
`float32` (as `matmul_f16`'s). The inputs are small integers (`|a|, |b| ≤ 2`, `K ≤ 70`; the masked
lanes load `+0`), so every partial sum is an integer of magnitude at most `280`: the exact value is
the only one any correct device gives, whatever its order and rounding. The verdicts are therefore
`conditional n`, `n` the number of dots answered (one per program with a local expert and `k`
step).

**The check of `Post`.** For a routed element of a block with expert `e ≥ 0`, `Post` asks for a
finite `float32` `u` within `tol` of the exact dot product `S` with the output the `bfloat16`
rounding of `u` (of `fl(u·w)` with `MUL_ROUTED_WEIGHT`), up to NaN payload. The test takes
`u = S` (exact in `float32`; distance `0 ≤ tol`), computes `S` from the inputs as an integer, the
product and the rounding with `FloatV.mul` and `FloatV.cvt` (the functions `Post` names), and asks
for the same bits and a finite output (stricter than `NaNEq`, and than the finiteness clause). -/

open TVA TVA.Validation Claims.FusedMoe

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

/-- A sentinel in `C`'s storage before the launch, and in the elements just outside it. -/
def sentinel : Int := -99

/-- Sentinel elements on each side of `C`'s storage. -/
def pad : Nat := 4

/-- `A[t, k]` and `B[e, c, k]`: small integers in `-2 … 2`. -/
def a0 (t k : Nat) : Int := ((3 * t + 5 * k + k / 7) % 5 : Nat) - (2 : Int)
def b0 (e c k : Nat) : Int := ((7 * k + 3 * c + 2 * e + 1) % 5 : Nat) - (2 : Int)

/-- One case: the instance, and the memory the expert assignment and the weights hold. -/
structure Case where
  d : Inst
  /-- `sorted_token_ids`' storage (`d.s.len` entries). -/
  sids : List Int
  /-- `expert_ids`' storage (`d.eid.len` entries). -/
  eids : List Int
  /-- `num_tokens_post_padded`. -/
  ntp : Nat
  /-- `topk_weights`' storage (`d.w.len` entries). -/
  wts : List Rat

def Case.init (c : Case) : CMem :=
  let d := c.d
  let m := CMem.ofElems ∅ .bf16 ((List.range d.a.len).map fun j : Nat =>
    ((d.a.ptr j).addr, (FloatV.ofRat (a0 (j / d.K) (j % d.K) : Rat) : BF16)))
  let m := CMem.ofElems m .bf16 ((List.range d.b.len).map fun j : Nat =>
    ((d.b.ptr j).addr, (FloatV.ofRat (b0 (j / (d.N * d.K)) (j / d.K % d.N) (j % d.K) : Rat) : BF16)))
  let m := CMem.ofElems m .bf16 ((List.range (d.c.len + 2 * pad)).map fun j : Nat =>
    ((d.c.ptr ((j : Int) - pad)).addr, (FloatV.ofRat (sentinel : Rat) : BF16)))
  let m := CMem.ofElems m .f32 ((List.range d.w.len).map fun j : Nat =>
    ((d.w.ptr j).addr, (FloatV.ofRat (c.wts.getD j 0) : F32)))
  let m := CMem.ofElems m .i32 ((List.range d.s.len).map fun j : Nat =>
    ((d.s.ptr j).addr, (IntV.ofInt (c.sids.getD j 0) : I32)))
  let m := CMem.ofElems m .i32 ((List.range d.eid.len).map fun j : Nat =>
    ((d.eid.ptr j).addr, (IntV.ofInt (c.eids.getD j 0) : I32)))
  CMem.ofElems m .i32 [((d.ntpb.ptr 0).addr, (IntV.ofNat c.ntp : I32))]

def Case.run (c : Case) : LRes (launch c.d).size × List Record :=
  execLaunch ((launch c.d).env P0) O fill0 10000 (model c.d) (launch c.d) c.init

/-- The flat index at position `pos` (`entryOf`, on this memory): with `naive`, the block's index
in its first lane and `M·top_k` in the others; otherwise entry `pos` of `sorted_token_ids`. -/
def Case.entry (c : Case) (pos : Nat) : Int :=
  if c.d.naive then (if pos % c.d.BM = 0 then ((pos / c.d.BM : Nat) : Int) else (nv c.d : Int))
  else c.sids.getD pos (nv c.d)

/-- What `Post` says element `(f, col)` of `C` holds, as `bfloat16` bits: `none` for "unchanged"
(no position below `ntp` holds `f`); `0` (`+0.0`) for expert `-1`; otherwise the rounding of the
exact dot product `S` (of `fl(S·w[f])` with `MUL_ROUTED_WEIGHT`). `eids` is the expected expert
assignment (`c.eids` for the test). -/
def Case.expectE (c : Case) (eids : List Int) (f col : Nat) : Option Nat :=
  let d := c.d
  match (List.range c.ntp).find? (fun pos => c.entry pos == (f : Int)) with
  | none => none
  | some pos =>
    let e := eids.getD (pos / d.BM) 0
    if e = -1 then some 0
    else
      let S : Int := ((List.range d.K).map fun k => a0 (f / d.top_k) k * b0 e.toNat col k).sum
      let u : F32 := FloatV.ofRat (S : Rat)
      let r : F32 := if d.rw then FloatV.mul u (FloatV.ofRat (c.wts.getD f 0)) else u
      some (DType.toBits .bf16 (FloatV.cvt r : BF16))

/-- The final memory against `Post`: every element of the matrix `C` as `expectE` says (and
finite), then every byte outside `C`'s storage (which is the matrix) unchanged, and no byte added. -/
def Case.agreeE (c : Case) (eids : List Int) (cf : CConfig (launch c.d).size) : Option String :=
  let d := c.d
  let m := CMem.toMem cf.mem
  let m0 := CMem.toMem c.init
  let bad := ((List.range (nv d)).flatMap fun f => (List.range d.N).map fun col => (f, col)).filter
    fun (f, col) =>
      let a := (d.c.ptr ((f * d.N + col : Nat) : Int)).addr
      match c.expectE eids f col with
      | none => (m.read .bf16 a).map (DType.toBits .bf16) != (m0.read .bf16 a).map (DType.toBits .bf16)
      | some bits => match m.read .bf16 a with
        | some v => !(v.isFinite && DType.toBits .bf16 v == bits)
        | none => true
  let lo := d.c.base
  let hi := d.c.base + (d.c.len * 2 : Nat)
  let changed := c.init.toList.filter fun (a, b) => !(lo ≤ a && a < hi) && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"C elements {bad.take 8} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside C changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c.eids

/-- The `moe_align_block_size` assignment, as a first GEMM builds it: 3 tokens, `top_k = 2` (flat
indices `0 … 5`), 2 local experts, `N = 40` (two column tiles, the second masked from column 8),
`K = 70` (two `k` steps, the second masked from `k = 6`), no `MUL_ROUTED_WEIGHT`. `EM = 64`, four
blocks of 16: block 0 is expert 1's (`0, 3, 5`, then the padding `6`), block 1 expert `-1`'s (`1`:
its row is zeroed), block 2 expert 0's (`2`); `ntp = 48`, so block 3 is past the padded total: its
entries hold `4` and its expert `0`, which no program may act on. Flat index `4` is not routed: its
row keeps the sentinel. -/
def sortedInst : Inst := Claims.FusedMoe.hostInst 3 2 2 40 70 64 64 4 false false (16, 32, 64, 1) 2

theorem sortedInst_ok : sortedInst.ok := hostInst_ok (by norm_num [cfgs])

def pad6 (xs : List Int) : List Int := xs ++ List.replicate (16 - xs.length) 6

def sorted : Case :=
  { d := sortedInst, ntp := 48, eids := [1, -1, 0, 0], wts := [1, 1, 1, 1, 1, 1],
    sids := pad6 [0, 3, 5] ++ pad6 [1] ++ pad6 [2] ++ List.replicate 16 4 }

/-- The naive assignment with `MUL_ROUTED_WEIGHT`, as a second GEMM builds it for one token with
`top_k_ids = 3` (`A` has `3` rows, the kernel's `top_k = 1`): no `sorted_token_ids` (its buffer is
empty and never read), `EM = ntp = 3·16`, block `p` holds flat index `p` in its first lane;
experts `2, 0, 2` of 4; `N = 33` (the second column tile has one valid column), `K = 64` (one
unmasked `k` step); weights `1/2`, `1/4`, `-3/2`. -/
def naiveInst : Inst := Claims.FusedMoe.hostInst 3 1 4 33 64 48 0 3 true true (16, 32, 64, 1) 3

theorem naiveInst_ok : naiveInst.ok := hostInst_ok (by norm_num [cfgs])

def naive : Case :=
  { d := naiveInst, ntp := 48, eids := [2, 0, 2], wts := [1 / 2, 1 / 4, -3 / 2], sids := [] }

def cases : List (String × Case) := [("sorted", sorted), ("naive_rw", naive)]

-- `Pre`, decidable parts: the seven buffers aligned (and below `2^64`), every element of each
-- present in `init`; `ntp` a multiple of `BLOCK_SIZE_M`, at most `EM`; the entries below `ntp`
-- nonnegative, the valid ones distinct; `expert_ids` covering the blocks below `ntp`, each `-1` or
-- in `[0, E)`; with `MUL_ROUTED_WEIGHT` the weights finite (and encoded exactly).
#guard cases.all fun (_, c) =>
  let d := c.d
  let m := CMem.toMem c.init
  let present {dt : DType} (b : Buf dt) : Bool :=
    b.base % dt.bytes == 0 && (List.range b.len).all fun j : Nat => (m.read dt (b.ptr j).addr).isSome
  present d.a && present d.b && present d.c && present d.w && present d.s && present d.eid &&
    present d.ntpb &&
  c.ntp % d.BM == 0 && c.ntp ≤ d.EM && (d.naive || c.sids.length == d.s.len) &&
  c.eids.length == d.eid.len && c.wts.length == d.w.len &&
  (List.range c.ntp).all (fun pos => 0 ≤ c.entry pos) &&
  (List.range c.ntp).all (fun p => (List.range c.ntp).all fun p' =>
    !(c.entry p < nv d && c.entry p' == c.entry p) || p == p') &&
  (List.range c.ntp).all (fun pos => pos / d.BM < d.eid.len &&
    let e := c.eids.getD (pos / d.BM) 0; e == -1 || (0 ≤ e && e < d.E)) &&
  (List.range d.w.len).all fun j => (FloatV.ofRat (c.wts.getD j 0) : F32).toRat == c.wts.getD j 0
-- The inputs are encoded exactly.
#guard (List.range 5).all fun z : Nat => (FloatV.ofRat ((z : Int) - 2 : Rat) : BF16).toRat == ((z : Int) - 2 : Rat)
-- Every kind of element occurs: unrouted (sorted's flat 4), zeroed (flat 1), computed; and no
-- computed element's expectation is the sentinel's bits.
#guard sorted.expectE sorted.eids 4 0 == none && sorted.expectE sorted.eids 1 39 == some 0 &&
  cases.all fun (_, c) => (List.range (nv c.d)).all fun f => (List.range c.d.N).all fun col =>
    c.expectE c.eids f col != some (DType.toBits .bf16 (FloatV.ofRat (sentinel : Rat) : BF16))

def verdictSorted : Verdict := Verdict.ofLaunch sorted.agree sorted.run
def verdictNaive : Verdict := Verdict.ofLaunch naive.agree naive.run

#eval IO.println (verdictSorted.line "fused_moe_bf16/sorted")
#eval IO.println (verdictNaive.line "fused_moe_bf16/naive_rw")
-- Agreement resting on one assumed answer per dot: `sorted`, two blocks with a local expert × two
-- column tiles × two `k` steps; `naive_rw`, three blocks × two column tiles × one step.
#guard let v := verdictSorted; v == .conditional 8 && v.agrees && v.assumed > 0
#guard let v := verdictNaive; v == .conditional 6 && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (`sorted` with blocks 0 and 2's experts
-- swapped; `naive_rw` with block 1's expert taken as `-1`).
#guard (Verdict.ofLaunch (sorted.agreeE [0, -1, 1, 0]) sorted.run).isDiscrepancy
#guard (Verdict.ofLaunch (naive.agreeE [2, -1, 2]) naive.run).isDiscrepancy
