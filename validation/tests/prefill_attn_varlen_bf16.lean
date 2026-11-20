import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `prefill_attn_varlen_bf16` run by `execLaunch` on two small covered
instances of the host's layout (`Claims.PrefillAttnVarlen.canon`, covered by `canon_ok`). Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says, both halves.

The block sizes are the claim's (`BLOCK_M = BLOCK_N = 128`): every program runs full
`128 × 128` tiles, whatever the sequence lengths.

* `causal`: `is_causal = True`, one sequence of `256` tokens from token `2` of `260` (the
  unowned tokens' outputs must keep their sentinels), `Lk = 12` in `BLOCK_DMODEL = 16` (lanes
  `12..15` masked), `max_input_len = 256`: two query blocks, the second running two key trips
  (the online softmax across trips: rescaling, a key group split across the two trips).
* `full`: `is_causal = False`, a sequence of `64` tokens from token `4` of `72` (query rows
  `64..127` of the block past its end, keys past its end masked) and an empty sequence (its
  programs run no trip and store nothing), 2 query heads on 1 key/value head,
  `Lk = BLOCK_DMODEL = 16`.

Each key trip costs about 75 s in `exec` (the claim's tiles are `128 × 128`), so the instances
are the smallest that cover these paths: 5 trips in all.

`sm_scale = 0.25` (`1/sqrt(16)`).

**The inputs make every platform answer exact.** Key `j` of a sequence is `64` in coordinate
`⌊log₂(j + 1)⌋` (the keys `2^k − 1 … 2^(k+1) − 2` share coordinate `k`: groups of `2^k` keys),
`0` in the other coordinates below `8`, and constant above; query `p` (head `h`) is `16` in the
coordinate of one group of keys: every other row whose last attended key `L` cuts its group
into a part of `1`, `2`, `4`, … keys picks that group (a key past `L`, masked, has the same
coordinate: attending it would change the result), the others a group they attend entirely
(`(p + h) mod C`, `C` the number of complete groups); `0` in the others below `8`, small integers
above. Unowned tokens' keys are `64` in every coordinate below `8`: attending one would change
the result too. So each row's scaled scores
take two values `256` apart (`hi` on a group of `2^k` keys, `lo` elsewhere), the query rows past a
sequence's end (`q = 0`) score `0` on the `64` keys they see, and every argument of
`tl.exp` is `0`, `-∞` or `-256`. The oracle answers `exp(0) = 1`, `exp(-∞) = +0`, `exp(x) = +0`
for `x ≤ -200` (`e^-200 < 2^-288`: far below half the smallest `float32` subnormal, the correctly
rounded value and the only one a relative error bound allows); `/` with the IEEE quotient (here
`1/2^k`, `0/2^k`, `2^k/2^k`: exact); each `tl.dot` with its exact value (`acc + Σ a·b` of the
request's own operands, answered only when it is a `float32`: products of small integers and
powers of two). The online softmax is then exact (`l_i` counts the keys at the maximum, a power of
two at every step; `p` rounds to `bfloat16` exactly), and so is the stored mean. The verdicts are
`conditional`: per key trip two `tl.dot`, three `tl.exp`, two `/`.

**The check** (`agree`), from the inputs as the test writes them: for every `(b, p, h, e)`
(`p < len_b`, `e < Lk`), the exact scores `σ·Σ_e q·k` (rationals) over the keys the row attends
(`j ≤ p` causal, `j < len_b` not); `c` the mean of the values at the maximum score; the output
must be finite and equal `c` exactly. `Post` compares with `attnRef`, whose other weights are
`e^-Δ` (`Δ ≥ 256`) rather than `0`: `|c − attnRef| ≤ 2^-⌊Δ⌋·(Σ_low |v| + n_low·|c|)/n_top`, which
the test checks is at most `2^-80`, below `Post`'s absolute term `2^-80·(1 + Σ|v|)`. Every other
byte is unchanged (the inputs, the unowned tokens' outputs, sentinels on both sides of `o`), and
no byte is added. `SeqOk` and `ValsOk` (finite inputs, `|v| ≤ 2^120`, `scoreMag ≤ 2^12` on every
pair of a sequence) are checked on the inputs. -/

open TVA TVA.Validation

/-- A platform that relates nothing: the platform answers come from the oracle. -/
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

/-- The position of an index in `Idx.all s` (row-major). -/
def flatIdx : (s : Shape) → Idx s → Nat
  | [], _ => 0
  | _ :: s, (i, j) => i.1 * s.foldr (· * ·) 1 + flatIdx s j

/-- A tensor with its elements computed once, into an array (the oracle's answers are read many
times: without this each read recomputes the operands). -/
def materialize {α : Type} [Inhabited α] {s : Shape} (x : Tensor s α) : Tensor s α :=
  let xs := (Idx.all s).toArray.map x.get
  ⟨fun i => xs[flatIdx s i]!⟩

/-- A lanewise answer: every lane answered, or none. -/
def lanewise {f : Format} {s : Shape} (g : FloatV f → Option (FloatV f)) (x : Tensor s (FloatV f)) :
    Option (Tensor s (FloatV f)) :=
  let xs := (Idx.all s).toArray.map fun i => g (x.get i)
  if xs.all (·.isSome) then
    let ys := xs.map (·.getD default)
    some ⟨fun i => ys[flatIdx s i]!⟩
  else none

/-- `exp` where its value is exact or forced. -/
def expAns {f : Format} (x : FloatV f) : Option (FloatV f) :=
  if x.isNaN then some FloatV.nan
  else if x.isInf then (if x.sign then some (FloatV.ofPat 0) else none)
  else if x.toRat == 0 then some (FloatV.ofRat 1)
  else if x.toRat ≤ -200 then some (FloatV.ofPat 0)
  else none

/-- A rational as an accumulator element, when the accumulator is `float32` and holds it exactly. -/
def accOf : (d : DType) → Rat → Option d.Val
  | .f32, q => let v : F32 := FloatV.ofRat q; if v.toRat == q then some v else none
  | _, _ => none

/-- The exact `acc + a·b` of a `tl.dot` request, from its own operands (finite), skipping the zero
elements of `a`: `none` unless every result is an exact `float32`. -/
def exactDot : (q : PlatQ) → Option q.Out
  | @PlatQ.dot cfg M N K a b acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => a.get2 i k).toArray.mapM cfg.a.val?
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => b.get2 k j).toArray.mapM cfg.b.val?
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM cfg.acc.val?
    let R ← (Array.range M).flatMapM fun i => do
      let nz := (List.range K).filterMap fun k => let x := A[i * K + k]!; if x == 0 then none else some (k, x)
      (Array.range N).mapM fun j =>
        accOf cfg.acc (nz.foldl (fun s (k, x) => s + x * B[k * N + j]!) C[i * N + j]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

def O : Oracle := fun _ _ q => match q with
  | .plat (.math .exp _ x) => lanewise expAns x
  | .plat (.div _ x y) => some (materialize (Tensor.map2 FloatV.divRn x y))
  | .plat q => exactDot q
  | _ => none

/-! ## The cases -/

structure Case where
  causal : Bool
  total : Nat
  H : Nat
  HKV : Nat
  Lk : Nat
  BD : Nat
  maxLen : Nat
  /-- `(b_start_loc[b], b_seq_len[b])`. -/
  seqs : List (Nat × Nat)
  /-- For the non-vacuity guard only: expectations computed with this mode instead. -/
  expCausal : Option Bool := none

/-- `sm_scale`: the double `0.25`. -/
def smBits : Nat := 0x3FD0000000000000
def sigma : Rat := 1 / 4

def Case.inst (c : Case) : Claims.PrefillAttnVarlen.Inst :=
  Claims.PrefillAttnVarlen.canon c.seqs.length c.total c.H c.HKV c.Lk c.BD c.maxLen
    (FloatV.ofPat smBits) c.causal

abbrev Case.L (c : Case) := Claims.PrefillAttnVarlen.launch c.inst

def Case.batch (c : Case) : Nat := c.seqs.length
def Case.start (c : Case) (b : Nat) : Nat := (c.seqs.getD b (0, 0)).1
def Case.len (c : Case) (b : Nat) : Nat := (c.seqs.getD b (0, 0)).2

/-- The sequence and local position of token `t`, if a sequence owns it. -/
def Case.locate (c : Case) (t : Nat) : Option (Nat × Nat) :=
  ((List.range c.batch).find? fun b => c.start b ≤ t ∧ t < c.start b + c.len b).map
    fun b => (b, t - c.start b)

/-- The last key row `p` of sequence `b` attends. -/
def Case.last (c : Case) (b p : Nat) : Nat := if c.causal then p else c.len b - 1

/-- The key group of local key `j`: `⌊log₂(j + 1)⌋`. -/
def grp (j : Nat) : Nat := Nat.log2 (j + 1)

/-- The groups complete within the keys `0 … L`: `k` with `2^(k+1) − 2 ≤ L`. -/
def complete (L : Nat) : Nat := ((List.range 8).filter fun k => 2 ^ (k + 1) - 2 ≤ L).length

def isPow2 (n : Nat) : Bool := n > 0 && n &&& (n - 1) == 0

/-- Row `p`'s (head `h`) group: the group of its last key `L` when only part of it is attended
and that part has a power-of-two size (`2^(k+1) − 2 > L`, `L − 2^k + 2` keys; every other row of
such kind), else a complete group, `(p + h) mod C`. -/
def Case.rowGroup (c : Case) (b h p : Nat) : Nat :=
  let L := c.last b p
  let k := grp L
  if k < 8 && 2 ^ (k + 1) - 2 > L && isPow2 (L + 2 - 2 ^ k) && (p + h) % 2 == 0 then k
  else (p + h) % complete L

/-- `q[t, h, e]`: `16` in the coordinate of the row's group, small integers above `8`;
an unowned token `1`. -/
def Case.qv (c : Case) (t h e : Nat) : Int :=
  match c.locate t with
  | some (b, p) =>
    if e < 8 then (if e = c.rowGroup b h p then 16 else 0)
    else ((p + 2 * h + e) % 5 : Nat) - (2 : Int)
  | none => 1

/-- `k[t, g, e]`: `64` in its group's coordinate, `1 + g` above `8`; an unowned token `64` in
every coordinate below `8` (it would outscore every key if a row attended it). -/
def Case.kv (c : Case) (t g e : Nat) : Int :=
  match c.locate t with
  | some (_, j) => if e < 8 then (if e = grp j then 64 else 0) else 1 + g
  | none => if e < 8 then 64 else 1

/-- `v[t, g, e]`: integers in `[-2, 2]`; an unowned token `3`. -/
def Case.vv (c : Case) (t g e : Nat) : Int :=
  match c.locate t with
  | some (b, j) => ((3 * j + 5 * e + 7 * g + b) % 5 : Nat) - (2 : Int)
  | none => 3

def bfv (z : Int) : BF16 := FloatV.ofRat (z : Rat)
def i32v (z : Int) : I32 := IntV.ofInt z

/-- A sentinel in `o` and in `edge` elements on each side of it. -/
def sentinel : BF16 := FloatV.ofRat (-99)
def edge : Nat := 4

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let tens (b : Buf .bf16) (heads : Nat) (x : Nat → Nat → Nat → Int) : List (Int × BF16) :=
    (List.range (c.total * heads * c.Lk)).map fun k : Nat =>
      ((b.ptr k).addr, bfv (x (k / (heads * c.Lk)) (k / c.Lk % heads) (k % c.Lk)))
  let m := CMem.ofElems ∅ .bf16 (tens d.q c.H c.qv)
  let m := CMem.ofElems m .bf16 (tens d.k c.HKV c.kv)
  let m := CMem.ofElems m .bf16 (tens d.v c.HKV c.vv)
  let m := CMem.ofElems m .i32 ((List.range c.batch).map fun b : Nat => ((d.b_start_loc.ptr b).addr, i32v (c.start b)))
  let m := CMem.ofElems m .i32 ((List.range c.batch).map fun b : Nat => ((d.b_seq_len.ptr b).addr, i32v (c.len b)))
  CMem.ofElems m .bf16 ((List.range (c.total * c.H * c.Lk + 2 * edge)).map fun e : Nat =>
    ((d.o.ptr ((e : Int) - edge)).addr, sentinel))

def Case.run (c : Case) : LRes c.L.size × List Record :=
  execLaunch (c.L.env P0) O fill0 100000 (Claims.PrefillAttnVarlen.model c.inst) c.L c.init

/-! ## The reference, from the inputs -/

def Case.kvh (c : Case) (h : Nat) : Nat := h / (c.H / c.HKV)

/-- The exact scaled score of row `p` against key `j` (sequence `b`, head `h`). -/
def Case.score (c : Case) (b h p j : Nat) : Rat :=
  sigma * (((List.range c.Lk).map fun e =>
    c.qv (c.start b + p) h e * c.kv (c.start b + j) (c.kvh h) e).sum : Int)

/-- `sm_scale · Σ liftMag q · liftMag k` on these (integer) inputs: `σ·Σ|q·k|`. -/
def Case.scoreMag (c : Case) (b h p j : Nat) : Rat :=
  sigma * (((List.range c.Lk).map fun e =>
    |c.qv (c.start b + p) h e * c.kv (c.start b + j) (c.kvh h) e|).sum : Int)

/-- Row `(b, h, p)`'s expected outputs, `e < Lk`: the mean `c` of the values at the maximum score,
and the bound `r` on its distance to `attnRef` (`none` if the score gap is below `200`). -/
def Case.expectRow (c : Case) (b h p : Nat) : Option (Array (Rat × Rat)) :=
  let lastE := if c.expCausal.getD c.causal then p else c.len b - 1
  let js := List.range (lastE + 1)
  let sc := js.map fun j => (j, c.score b h p j)
  let mx := (sc.map (·.2)).foldl max ((sc.headD (0, 0)).2)
  let top := (sc.filter fun (_, s) => s == mx).map (·.1)
  let low := (sc.filter fun (_, s) => s != mx).map (·.1)
  let gap := (sc.filter fun (_, s) => s != mx).foldl (fun g (_, s) => min g (mx - s)) (256 : Rat)
  let n : Rat := top.length
  if gap < 200 then none
  else some ((Array.range c.Lk).map fun e =>
    let v (j : Nat) : Rat := c.vv (c.start b + j) (c.kvh h) e
    let ctr := (top.map v).sum / n
    (ctr, (2 : Rat) ^ (-(gap.floor) : Int) * ((low.map fun j => |v j|).sum + (low.length : Rat) * |ctr|) / n))

/-- The final memory of `c`'s run against `e`'s expectations (`e = c` for the test; the same
memory layout). -/
def Case.agreeE (c e : Case) (cf : CConfig c.L.size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let rows := (List.range c.batch).flatMap fun b : Nat => (List.range (c.len b)).flatMap fun p : Nat =>
    (List.range c.H).map fun h : Nat => (b, p, h)
  let bad := rows.filter fun (b, p, h) =>
    match e.expectRow b h p with
    | none => true
    | some ex => (List.range c.Lk).any fun i =>
      match m.read .bf16 (d.o.ptr (((c.start b + p) * c.H + h) * c.Lk + i : Nat)).addr, ex[i]? with
      | some w, some (ctr, r) => !(w.isFinite && w.toRat == ctr && r ≤ 1 / 2 ^ 80)
      | _, _ => true
  -- the bytes of the owned tokens' outputs
  let owned (a : Int) : Bool :=
    let k := (a - d.o.base) / 2
    d.o.base ≤ a && a < d.o.base + 2 * (c.total * c.H * c.Lk : Nat) &&
      (c.locate (k / (c.H * c.Lk)).toNat).isSome
  let changed := c.init.toList.filter fun (a, b) => !owned a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"rows (b, p, h) {bad.take 8} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the owned outputs changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c

/-- The key trips of program `(b, h, m)`, and the platform answers of the run: per trip two
`tl.dot`, three `tl.exp`, two `/`. -/
def Case.trips (c : Case) (b m : Nat) : Nat :=
  if 128 * m < c.len b then
    let endN := if c.causal then min ((m + 1) * 128) (c.len b) else c.len b
    (endN + 127) / 128
  else 0

def Case.answers (c : Case) : Nat :=
  7 * c.H * ((List.range c.batch).map fun b =>
    ((List.range ((c.maxLen + 127) / 128)).map (c.trips b)).sum).sum

/-- `Pre` on the inputs: the six buffers aligned (every byte is written by `init`); `SeqOk`
(`0 ≤ len ≤ max_input_len`, `start + len ≤ total`, non-empty sequences disjoint); `ValsOk`
(finite `bfloat16` inputs: small integers, exact; `|v| ≤ 2^120`; `scoreMag ≤ 2^12` on every pair of
a sequence); and the expectations exist. -/
def Case.pre (c : Case) : Bool :=
  let d := c.inst
  d.q.base % 2 == 0 && d.k.base % 2 == 0 && d.v.base % 2 == 0 && d.o.base % 2 == 0 &&
  d.b_start_loc.base % 4 == 0 && d.b_seq_len.base % 4 == 0 &&
  (List.range c.batch).all (fun b => c.len b ≤ c.maxLen && c.start b + c.len b ≤ c.total) &&
  (List.range c.batch).all (fun b => (List.range c.batch).all fun b' =>
    b == b' || c.len b == 0 || c.len b' == 0 ||
      c.start b + c.len b ≤ c.start b' || c.start b' + c.len b' ≤ c.start b) &&
  (List.range c.total).all (fun t => (List.range c.Lk).all fun e =>
    (List.range c.H).all (fun h => (bfv (c.qv t h e)).toRat == c.qv t h e) &&
    (List.range c.HKV).all fun g => ((bfv (c.kv t g e)).toRat == c.kv t g e &&
      (bfv (c.vv t g e)).toRat == c.vv t g e && decide (|c.vv t g e| ≤ 2 ^ 120))) &&
  (List.range c.batch).all fun b => (List.range c.H).all fun h =>
    (List.range (c.len b)).all fun p => (List.range (c.len b)).all (fun j => c.scoreMag b h p j ≤ 2 ^ 12) &&
      (c.expectRow b h p).isSome

/-! ## The instances -/

/-- One causal sequence of `256` tokens from token `2` of `260` (tokens `0, 1, 258, 259`
unowned), 1 head, `Lk = 12` in `BLOCK_DMODEL = 16`: query block `0` runs one key trip, block `1`
two. -/
def causal : Case :=
  { causal := true, total := 260, H := 1, HKV := 1, Lk := 12, BD := 16, maxLen := 256,
    seqs := [(2, 256)] }
theorem causal_ok : causal.inst.ok :=
  Claims.PrefillAttnVarlen.canon_ok 1 260 1 1 12 16 256 4 _ true (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩

/-- Not causal: a sequence of `64` tokens from token `4` of `72` and an empty one, 2 query heads
on 1 key/value head, `Lk = BLOCK_DMODEL = 16`: the empty sequence's programs run no trip and store
nothing. -/
def full : Case :=
  { causal := false, total := 72, H := 2, HKV := 1, Lk := 16, BD := 16, maxLen := 64,
    seqs := [(4, 64), (0, 0)] }
theorem full_ok : full.inst.ok :=
  Claims.PrefillAttnVarlen.canon_ok 2 72 2 1 16 16 64 4 _ false (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩

#guard (FloatV.ofPat smBits : F64).toRat == sigma
#guard full.pre && causal.pre

/-- Run a case once: print its verdict line, and fail unless the verdict is `conditional` with the
expected count, and every wrong expectation `es` is caught (a discrepancy) on the same run.
(Each run takes about 75 s per key trip: one command per case, so each runs once.) -/
def check (name : String) (c : Case) (es : List Case) : IO Unit := do
  let r := c.run
  let v := Verdict.ofLaunch c.agree r
  IO.println (v.line s!"prefill_attn_varlen_bf16/{name}")
  unless v == .conditional c.answers && v.assumed > 0 do
    throw (IO.userError s!"{name}: expected conditional {c.answers}")
  unless es.all fun e => (Verdict.ofLaunch (c.agreeE e) r).isDiscrepancy do
    throw (IO.userError s!"{name}: a wrong expectation agrees")

-- Not vacuous: a wrong expectation is caught on the same run (the inputs read in the other mode).
#eval check "full" full [{ full with expCausal := some true }]
#eval check "causal" causal [{ causal with expCausal := some false }]
