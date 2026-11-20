import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `paged_attention_f16` run by `execLaunch` on two small covered
instances of the host's layout (`Claims.PagedAttention.canon`, covered by `canon_ok`). Not a test
against the device (phase 4): it checks that `exec` runs the model end to end, and that the final
memory is what `Post` says, both halves.

* `gqa`: 3 sequences of `5, 4, 1` tokens, 2 query heads on 1 key/value head
  (`QUERIES_PER_KV = 2`), `HEAD_DIM = 4`, pages of 2 tokens (`PAGE_SIZE ≥ 2`: the `tl.mul_add`
  path), `scale = 0.5`. The block tables read pages out of order, share a page between two
  sequences, and hold out-of-range entries where the kernel does not read; the last page of the
  first sequence is half masked.
* `page1`: pages of one token (`PAGE_SIZE = 1`: the `tl.addc` path), 2 heads on 2 key/value heads,
  `HEAD_DIM = 2`, `scale = 0.25`; one sequence reads the same page twice, and one sequence is empty
  (outside `SeqVals`: `Post` only says a `float16` value is stored; the kernel computes `0/0`).

**The inputs make every platform answer exact.** Each query is `16` in one coordinate (its
head's) and small integers elsewhere; each key is `32` (or `64`) or `0` in that coordinate and
constant elsewhere. So within a sequence and head the scaled scores take two values `256` apart
(`hi` for the keys marked in the head's coordinate, `lo` for the others), and every argument of
`tl.exp` is `0`, `-∞` (masked lanes, the first page's `alpha`) or `≤ -256`; the oracle answers
`exp(0) = 1`, `exp(-∞) = +0`, `exp(x) = +0` for `x ≤ -200` (`e^-200 < 2^-288`, far below half the
smallest `float32` subnormal: the correctly rounded value, and the only one a relative error bound
allows). Every online-softmax step is then exact: `l_i` counts the keys at the maximum (`1`, `2`
or `4`), `acc` sums their values (small integers), and the final `/` (the oracle's IEEE quotient)
and the `float16` rounding are exact. The verdicts are `conditional`: per program, two `exp` per
page and one `/`.

**The check** (`agree`), computed from the inputs as the test writes them: for a sequence with
tokens, the exact scores `σ·q·k` (rationals) over its tokens read through the block table; `c` the
mean of the values at the maximum score; the output must be finite and equal `c` exactly. `Post`
compares with `attnRef`, whose other weights are `e^-Δ` (`Δ ≥ 256` the score gap) rather than `0`:
`|c − attnRef| ≤ 2^-⌊Δ⌋·(Σ_low |v| + n_low·|c|) / n_top`, which the test checks is at most `2^-24`,
`Post`'s absolute term (so `Post` holds for every `X`). An empty sequence's outputs: some `float16`
value. Every other byte is unchanged (inputs, block-table entries never read, sentinels on both
sides of `out`), and no byte is added. `SeqOk` and `SeqVals` (`|v| ≤ 2^15`,
`σ·Σ|q·k| ≤ 2^10`) are checked on the inputs. -/

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

/-- A lanewise answer: every lane answered, or none. -/
def lanewise {f : Format} {s : Shape} (g : FloatV f → Option (FloatV f)) (x : Tensor s (FloatV f)) :
    Option (Tensor s (FloatV f)) :=
  let xs := (Idx.all s).toArray.map fun i => g (x.get i)
  if xs.all (·.isSome) then
    let ys := xs.map (·.getD default)
    let pos : Std.HashMap (List Nat) Nat :=
      ((Idx.all s).zipIdx.map fun (i, k) => (idxList s i, k)).foldl (fun m (a, k) => m.insert a k) {}
    some ⟨fun i => ys[pos.getD (idxList s i) 0]!⟩
  else none
where
  idxList : (s : Shape) → Idx s → List Nat
    | [], _ => []
    | _ :: s, (i, j) => i.1 :: idxList s j

/-- `exp` where its value is exact or forced. -/
def expAns {f : Format} (x : FloatV f) : Option (FloatV f) :=
  if x.isNaN then some FloatV.nan
  else if x.isInf then (if x.sign then some (FloatV.ofPat 0) else none)
  else if x.toRat == 0 then some (FloatV.ofRat 1)
  else if x.toRat ≤ -200 then some (FloatV.ofPat 0)
  else none

def O : Oracle := fun _ _ q => match q with
  | .plat (.math .exp _ x) => lanewise expAns x
  | .plat (.div _ x y) => some (Tensor.map2 FloatV.divRn x y)
  | _ => none

/-! ## The cases -/

structure Case where
  S : Nat
  H : Nat
  KV : Nat
  D : Nat
  P : Nat
  NP : Nat
  MP : Nat
  /-- `scale`: the double's pattern and its value. -/
  scBits : Nat
  sigma : Rat
  /-- `q[s, h, e]`, `k_cache[p, r, g, e]`, `v_cache[p, r, g, e]`. -/
  qv : Nat → Nat → Nat → Int
  kv : Nat → Nat → Nat → Nat → Int
  vv : Nat → Nat → Nat → Nat → Int
  /-- `block_tables[s, j]` and `seq_lens[s]`. -/
  bt : Nat → Nat → Int
  lens : Nat → Nat

def Case.inst (c : Case) : Claims.PagedAttention.Inst :=
  Claims.PagedAttention.canon c.S c.H c.KV c.D c.P c.NP c.MP (FloatV.ofPat c.scBits)

abbrev Case.L (c : Case) := Claims.PagedAttention.launch c.inst

def f16v (z : Int) : F16 := FloatV.ofRat (z : Rat)
def i32v (z : Int) : I32 := IntV.ofInt z

/-- A sentinel in `out` and in `edge` elements on each side of it. -/
def sentinel : F16 := FloatV.ofRat (-999)
def edge : Nat := 4

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .f16 ((List.range (c.S * c.H * c.D)).map fun k : Nat =>
    ((d.q.ptr k).addr, f16v (c.qv (k / (c.H * c.D)) (k / c.D % c.H) (k % c.D))))
  let cache (b : Buf .f16) (x : Nat → Nat → Nat → Nat → Int) : List (Int × F16) :=
    (List.range (c.NP * c.P * c.KV * c.D)).map fun k : Nat =>
      ((b.ptr k).addr, f16v (x (k / (c.P * c.KV * c.D)) (k / (c.KV * c.D) % c.P) (k / c.D % c.KV) (k % c.D)))
  let m := CMem.ofElems m .f16 (cache d.k_cache c.kv)
  let m := CMem.ofElems m .f16 (cache d.v_cache c.vv)
  let m := CMem.ofElems m .i32 ((List.range (c.S * c.MP)).map fun k : Nat =>
    ((d.block_tables.ptr k).addr, i32v (c.bt (k / c.MP) (k % c.MP))))
  let m := CMem.ofElems m .i32 ((List.range c.S).map fun s : Nat => ((d.seq_lens.ptr s).addr, i32v (c.lens s)))
  CMem.ofElems m .f16 ((List.range (c.S * c.H * c.D + 2 * edge)).map fun e : Nat =>
    ((d.out.ptr ((e : Int) - edge)).addr, sentinel))

def Case.run (c : Case) : LRes c.L.size × List Record :=
  execLaunch (c.L.env P0) O fill0 10000 (Claims.PagedAttention.model c.inst) c.L c.init

/-! ## The reference, from the inputs -/

/-- Token `t` of sequence `s`: its page and row. -/
def Case.tok (c : Case) (s t : Nat) : Nat × Nat := ((c.bt s (t / c.P)).toNat, t % c.P)

/-- The exact scaled score of token `t` for `(s, h)`, with key/value head `h / (H / KV)`. -/
def Case.score (c : Case) (s h t : Nat) : Rat :=
  let (p, r) := c.tok s t
  c.sigma * (((List.range c.D).map fun e => c.qv s h e * c.kv p r (h / (c.H / c.KV)) e).sum : Int)

def Case.scoreAbs (c : Case) (s h t : Nat) : Rat :=
  let (p, r) := c.tok s t
  c.sigma * (((List.range c.D).map fun e => |c.qv s h e * c.kv p r (h / (c.H / c.KV)) e|).sum : Int)

def Case.v (c : Case) (s h t e : Nat) : Rat :=
  let (p, r) := c.tok s t
  c.vv p r (h / (c.H / c.KV)) e

/-- For a sequence with tokens: the expected output `c` (the mean of the values at the maximum
score) and the bound `r` on its distance to `attnRef` (`none` if the gap to the other scores is
below `200`). -/
def Case.expect (c : Case) (s h e : Nat) : Option (Rat × Rat) :=
  let ts := List.range (c.lens s)
  let sc := ts.map (c.score s h)
  let mx := sc.foldl max (sc.headD 0)
  let top := ts.filter fun t => c.score s h t == mx
  let low := ts.filter fun t => c.score s h t != mx
  let n : Rat := top.length
  let ctr := ((top.map fun t => c.v s h t e).sum) / n
  let gap := (low.map fun t => mx - c.score s h t).foldl min 256
  if gap < 200 then none
  else some (ctr, (2 : Rat) ^ (-(gap.floor) : Int) *
    (((low.map fun t => |c.v s h t e|).sum) + (low.length : Rat) * |ctr|) / n)

/-- The final memory of `c`'s run against `e`'s expectations (`e = c` for the test). -/
def Case.agreeE (c e : Case) (cf : CConfig c.L.size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let elems := (List.range c.S).flatMap fun s : Nat => (List.range c.H).flatMap fun h : Nat =>
    (List.range c.D).map fun i : Nat => (s, h, i)
  let bad := elems.filter fun (s, h, i) =>
    match m.read .f16 (d.out.ptr ((s * c.H + h) * c.D + i : Nat)).addr with
    | none => true
    | some w =>
      if e.lens s = 0 then false
      else match e.expect s h i with
        | some (ctr, r) => !(w.isFinite && w.toRat == ctr && r ≤ 1 / 2 ^ 24)
        | none => true
  let inOut (a : Int) : Bool := d.out.base ≤ a && a < d.out.base + 2 * (c.S * c.H * c.D : Nat)
  let changed := c.init.toList.filter fun (a, b) => !inOut a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"out elements {bad.take 8} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside out changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c

/-- The platform answers of the run: per program, two `exp` per page visited, and one `/`. -/
def Case.answers (c : Case) : Nat :=
  c.H * ((List.range c.S).map fun s => 2 * ((c.lens s + c.P - 1) / c.P) + 1).sum

/-- `Pre` and `SeqVals` on the inputs: the buffers aligned (every byte is written by `init`); each
length in `[0, MP·P]` and each block-table entry read a page below `NP`; finite `float16` inputs
(small integers), `|v| ≤ 2^15`, `σ·Σ|q·k| ≤ 2^10`; the expectations exist. -/
def Case.pre (c : Case) : Bool :=
  let d := c.inst
  d.q.base % 2 == 0 && d.k_cache.base % 2 == 0 && d.v_cache.base % 2 == 0 && d.out.base % 2 == 0 &&
  d.block_tables.base % 4 == 0 && d.seq_lens.base % 4 == 0 &&
  (List.range c.S).all fun s =>
    c.lens s ≤ c.MP * c.P &&
    (List.range ((c.lens s + c.P - 1) / c.P)).all (fun j => 0 ≤ c.bt s j && c.bt s j < c.NP) &&
    (List.range c.H).all fun h =>
      (List.range (c.lens s)).all (fun t => c.scoreAbs s h t ≤ 2 ^ 10 &&
        (List.range c.D).all fun e => |c.v s h t e| ≤ 2 ^ 15 ∧ (f16v (c.v s h t e).num).toRat == c.v s h t e) &&
      (c.lens s == 0 || (List.range c.D).all fun e => (c.expect s h e).isSome)

/-! ## The instances -/

/-- The keys marked (score `hi`) for query head `0` and head `1` (the coordinates `0`, `1`), by
`(page, row)`. -/
def top0 (p r : Nat) : Bool := (p, r) ∈ [(6, 1), (3, 0)]
def top1 (p r : Nat) : Bool := (p, r) ∈ [(6, 0), (1, 0), (1, 1), (3, 0), (0, 1)]

def gqa : Case :=
  { S := 3, H := 2, KV := 1, D := 4, P := 2, NP := 8, MP := 3,
    scBits := 0x3FE0000000000000, sigma := 1 / 2,
    qv := fun s h e => match e with
      | 0 => if h = 0 then 16 else 0
      | 1 => if h = 1 then 16 else 0
      | 2 => [1, -2, 3].getD s 0
      | _ => [2, 1, -1].getD s 0,
    kv := fun p r _ e => match e with
      | 0 => if top0 p r then 32 else 0
      | 1 => if top1 p r then 32 else 0
      | 2 => 1
      | _ => 2,
    vv := fun p r _ e => ((p * 5 + r * 3 + e * 7) % 7 : Nat) - (3 : Int),
    bt := fun s j => [[6, 1, 3], [0, 3, 1000], [5, -5, 99999]].getD s [] |>.getD j 0,
    lens := fun s => [5, 4, 1].getD s 0 }
theorem gqa_ok : gqa.inst.ok :=
  Claims.PagedAttention.canon_ok 3 2 1 4 2 8 3 _ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) ⟨2, by norm_num, by norm_num⟩ ⟨1, by norm_num, by norm_num⟩
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩ (by norm_num)

def page1 : Case :=
  { S := 3, H := 2, KV := 2, D := 2, P := 1, NP := 4, MP := 3,
    scBits := 0x3FD0000000000000, sigma := 1 / 4,
    qv := fun s h e => if e = 0 then 16 else [[3, -1], [2, 5], [7, 7]].getD s [] |>.getD h 0,
    kv := fun p _ g e => if e = 0 then (if (g = 0 ∧ (p = 0 ∨ p = 3)) ∨ (g = 1 ∧ p = 2) then 64 else 0) else 1,
    vv := fun p _ g e => ((p * 3 + g * 2 + e) % 5 : Nat) - (2 : Int),
    bt := fun s j => [[2, 0, 3], [1, 1, 7], [12345, 0, 0]].getD s [] |>.getD j 0,
    lens := fun s => [3, 2, 0].getD s 0 }
theorem page1_ok : page1.inst.ok :=
  Claims.PagedAttention.canon_ok 3 2 2 2 1 4 3 _ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) ⟨1, by norm_num, by norm_num⟩ ⟨0, by norm_num, by norm_num⟩
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩ (by norm_num)

-- The scales are the doubles `0.5`, `0.25`; `Pre`, `SeqVals`, the expectations exist.
#guard (FloatV.ofPat gqa.scBits : F64).toRat == gqa.sigma && (FloatV.ofPat page1.scBits : F64).toRat == page1.sigma
#guard gqa.pre && page1.pre
-- The top counts are `1`, `2` or `4`; the sequences mix pages without and with marked keys.
#guard [gqa, page1].all fun c => (List.range c.S).all fun s => (List.range c.H).all fun h =>
  let ts := List.range (c.lens s)
  let mx := (ts.map (c.score s h)).foldl max ((ts.map (c.score s h)).headD 0)
  (ts.filter fun t => c.score s h t == mx).length ∈ [0, 1, 2, 4]

def verdictGqa : Verdict := Verdict.ofLaunch gqa.agree gqa.run
def verdictPage1 : Verdict := Verdict.ofLaunch page1.agree page1.run

#eval IO.println (verdictGqa.line "paged_attention_f16/gqa")
#eval IO.println (verdictPage1.line "paged_attention_f16/page1")
#guard verdictGqa == .conditional gqa.answers
#guard verdictPage1 == .conditional page1.answers

-- Not vacuous: wrong expectations are caught (the first sequence's pages read in another order;
-- the key/value heads swapped).
#guard
  let r := gqa.run
  (Verdict.ofLaunch (gqa.agreeE { gqa with bt := fun s j => if s = 0 then [3, 1, 6].getD j 0 else gqa.bt s j }) r).isDiscrepancy
#guard
  (Verdict.ofLaunch (page1.agreeE { page1 with vv := fun p r g e => page1.vv p r (1 - g) e }) page1.run).isDiscrepancy
