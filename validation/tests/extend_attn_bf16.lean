import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `extend_attn_bf16` run by `execLaunch` on two small covered
instances of the host's layout (`Claims.ExtendAttn.canon`, covered by `canon_ok`). Not a test
against the device (phase 4): it checks that `exec` runs the model end to end, and that the final
memory is what `Post` says, both halves.

The block sizes are the claim's H200 choice (`BLOCK_M = 128` for head dims up to `128`,
`BLOCK_N = BLOCK_N_PREFIX = 64`): every program runs full `128 × 64` tiles.

* `extend`: an extend batch (`int64` `qo_indptr`, causal, no window), 1 head, `D = 12` in `BLOCK_DMODEL = 16` (lanes `12..15` masked), `sm_scale = 0.25`.
  Request `0`: a prefix of `70` cached tokens (two prefix trips, the second masked past `6`), read
  through `kv_indices` in a scrambled slot order, and `5` new tokens; request `1`: no prefix and
  `130` new tokens (two query blocks; the second block's new-token loop runs three trips, the
  first block's two). `qo_indptr = [2, 7, 137]` of `140` new tokens and `kv_indptr = [1, 71, 71]`
  of `72` indices: the unowned new tokens' outputs must keep their sentinels; request `0`'s second
  block returns at once (`ext ≤ BLOCK_M`).
* `verify`: a target verification (`int32` `qo_indptr`, the tree mask `custom_mask` with
  `mask_indptr`), with a sliding window of `8` and `window_kv_offsets = [2, 0]`, 2 query heads on
  1 key/value head, `D = 16`.
  Request `0`: a prefix of `20` and `6` draft tokens in a tree (`0 ← 1, 2`; `1 ← 3, 4`; `2 ← 5`);
  request `1`: no prefix, `4` draft tokens in a chain. The window cuts the prefix (`pre + i ≤ j + 8`).

Each trip costs about 22 s in `exec` (`128 × 64` tiles): 8 trips for `extend`, 6 for `verify`.

**The inputs make every platform answer exact.** Each key (cached or new) is `64` in one
coordinate below `8` (its position's "colour": in `extend` its group `⌊log₂(j + 1)⌋`, in
`verify` a mixing function of the position) and `0` in the others, constant above; each query row
is `16` in one colour, `0` in the others, small integers above. The row's colour is chosen (by
the test, from the positions the row attends) so that the attended keys of that colour are `1`,
`2`, `4`, … in number, preferring a colour that unattended keys also have (so a mask that let one
through would change the result). So each row's scaled scores take two values `256` apart, and
every argument of `tl.exp` is `0`, `-∞`, `-256`, or `-1e20` minus a score (a tile with every key
of a row masked: `row_max = -1e20`); the oracle answers `exp(0) = 1`, `exp(-∞) = +0`,
`exp(x) = +0` for `x ≤ -200` (`e^-200 < 2^-288`, far below half the smallest `float32` subnormal:
the correctly rounded value and the only one a relative error bound allows); `/` with the IEEE
quotient (a sum of small integers by a power of two, and `0/1` on the rows past a request's end:
exact); each `tl.dot` with its exact value (`acc + Σ a·b` of the request's own operands, answered
only when it is a `float32`). The verdicts are `conditional`: per trip run (not skipped) two
`tl.dot` and two `tl.exp`, and one `/` per program that does not return at once.

**The check** (`agree`), from the inputs as the test writes them: for every new token `i` of
request `b`, head `h`, component `e < D`, the positions the row attends (`attends`, written here
from the claim's description: every prefix position or, with the tree mask, each ancestor and
itself, else each earlier new token and itself; with a window `W`, only `pre + i ≤ j + W`), the
exact scores `σ·Σ q·k` (rationals); `c` the mean of the values at the maximum score; the output
must be finite and equal `c` exactly. `Post` compares with `attnRef`, whose other weights are
`e^-Δ` (`Δ ≥ 256`) rather than `0`: `|c − attnRef| ≤ 2^-⌊Δ⌋·(Σ_low |v| + n_low·|c|)/n_top`,
which the test checks is at most `2^-80`, below `Post`'s absolute term. Every other byte is
unchanged (inputs, the unused `kv_indices` entries, the mask's unread columns, the unowned
tokens' outputs, sentinels on both sides of `o_extend`), and no byte is added. `ReqOk`, `MaskOk`
and `ValsOk` are checked on the inputs. -/

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

/-- A request: its prefix's cache slots, its new tokens, and (with the tree mask) each new
token's parent and its window offset. -/
structure Req where
  slots : List Nat
  ext : Nat
  parent : Nat → Option Nat := fun _ => none
  wOff : Nat := 0

def Req.pre (r : Req) : Nat := r.slots.length

structure Case where
  cm : Bool
  sw : Nat
  H : Nat
  HKV : Nat
  D : Nat
  BD : Nat
  /-- Cache slots, new tokens, `kv_indices` entries. -/
  S : Nat
  T : Nat
  NI : Nat
  /-- `qo_indptr[0]`, `kv_indptr[0]`, `mask_indptr[0]`. -/
  qo0 : Nat
  kv0 : Nat
  mi0 : Nat := 0
  reqs : List Req
  /-- Positions coloured by their group `⌊log₂(j + 1)⌋` (else by a mixing function). -/
  grouped : Bool := false
  /-- For the non-vacuity guard only: expectations computed with this window instead. -/
  expSw : Option Nat := none

def smBits : Nat := 0x3FD0000000000000
def sigma : Rat := 1 / 4

def Case.batch (c : Case) : Nat := c.reqs.length
def Case.req (c : Case) (b : Nat) : Req := c.reqs.getD b { slots := [], ext := 0 }
def Case.maxExt (c : Case) : Nat := (c.reqs.map (·.ext)).foldl max 0
/-- `qo_indptr[b]`, `kv_indptr[b]`, `mask_indptr[b]`. -/
def Case.qs (c : Case) (b : Nat) : Nat := c.qo0 + ((c.reqs.take b).map (·.ext)).sum
def Case.ks (c : Case) (b : Nat) : Nat := c.kv0 + ((c.reqs.take b).map (·.pre)).sum
def Case.rowLen (c : Case) (b : Nat) : Nat := (c.req b).pre + (c.req b).ext + (c.req b).wOff
def Case.ms (c : Case) (b : Nat) : Nat :=
  c.mi0 + ((List.range b).map fun b' => (c.req b').ext * c.rowLen b').sum
def Case.NM (c : Case) : Nat := c.ms c.batch + 3

def Case.inst (c : Case) : Claims.ExtendAttn.Inst :=
  Claims.ExtendAttn.canon c.cm c.sw c.batch c.T c.S c.NI c.NM c.H c.HKV c.D c.BD 128 c.maxExt
    (FloatV.ofPat smBits)

abbrev Case.L (c : Case) := Claims.ExtendAttn.launch c.inst

/-- New token `j` of request `b` is an ancestor of (or is) new token `i` in its tree. -/
def Req.anc (r : Req) (i j : Nat) : Bool :=
  (List.range (i + 1)).foldl (fun (acc : Bool × Option Nat) _ =>
    match acc.2 with
    | some k => (acc.1 || k == j, r.parent k)
    | none => acc) (false, some i) |>.1

/-- The mask bit of new token `i` for new token `j` (the tree: ancestors and itself). -/
def Case.bit (c : Case) (b i j : Nat) : Bool := (c.req b).anc i j

/-- Row `i` of request `b` attends position `j` (`j < pre` a cached token, else new token
`j − pre`). -/
def Case.attendsW (c : Case) (sw b i j : Nat) : Bool :=
  let pre := (c.req b).pre
  (j < pre || (if c.cm then c.bit b i (j - pre) else j ≤ pre + i)) && (sw == 0 || pre + i ≤ j + sw)

def Case.attends (c : Case) (b i j : Nat) : Bool := c.attendsW c.sw b i j

/-- The colour (a coordinate below `8`) of position `j` of request `b`: its group
`⌊log₂(j + 1)⌋` (positions `2^k − 1 … 2^(k+1) − 2`; below `8` for `j < 255`), or a mixing
function. -/
def Case.colour (c : Case) (b j : Nat) : Nat :=
  if c.grouped then Nat.log2 (j + 1) % 8 else (j * 5 + b * 3 + j / 8) % 8

def isPow2 (n : Nat) : Bool := n > 0 && n &&& (n - 1) == 0

/-- Row `(b, h, i)`'s colour: one whose attended positions are a power of two in number,
preferring one that an unattended position also has; `none` if there is none. -/
def Case.rowColour (c : Case) (b h i : Nat) : Option Nat :=
  let n := (c.req b).pre + (c.req b).ext
  let cnt (k : Nat) := ((List.range n).filter fun j => c.colour b j == k && c.attends b i j).length
  let other (k : Nat) := (List.range n).any fun j => c.colour b j == k && !c.attends b i j
  let cands := (List.range 8).map (fun k => (k + i + h) % 8) |>.filter fun k => isPow2 (cnt k)
  match cands.find? other with
  | some k => some k
  | none => cands.head?

/-- The request and local index of new token `t`. -/
def Case.locate (c : Case) (t : Nat) : Option (Nat × Nat) :=
  ((List.range c.batch).find? fun b => c.qs b ≤ t ∧ t < c.qs b + (c.req b).ext).map fun b => (b, t - c.qs b)

/-- The request and position of cache slot `s` (first request that uses it). -/
def Case.slotOwner (c : Case) (s : Nat) : Option (Nat × Nat) :=
  ((List.range c.batch).find? fun b => (c.req b).slots.contains s).map fun b =>
    (b, (c.req b).slots.idxOf s)

/-- A key at position `j` of request `b`: `64` in its colour, `1 + g` above `8`. -/
def Case.keyAt (c : Case) (b j g e : Nat) : Int := if e < 8 then (if e = c.colour b j then 64 else 0) else 1 + g
/-- A value at position `j` of request `b`: integers in `[-2, 2]`. -/
def valAt (b j g e : Nat) : Int := ((3 * j + 5 * e + 7 * g + b) % 5 : Nat) - (2 : Int)

def Case.qv (c : Case) (t h e : Nat) : Int :=
  match c.locate t with
  | some (b, i) =>
    if e < 8 then (if some e = c.rowColour b h i then 16 else 0)
    else ((i + 2 * h + e) % 5 : Nat) - (2 : Int)
  | none => 1
def Case.kv (c : Case) (t g e : Nat) : Int :=
  match c.locate t with
  | some (b, i) => c.keyAt b ((c.req b).pre + i) g e
  | none => 64
def Case.vv (c : Case) (t g e : Nat) : Int :=
  match c.locate t with
  | some (b, i) => valAt b ((c.req b).pre + i) g e
  | none => 3
def Case.kbv (c : Case) (s g e : Nat) : Int :=
  match c.slotOwner s with
  | some (b, j) => c.keyAt b j g e
  | none => 64
def Case.vbv (c : Case) (s g e : Nat) : Int :=
  match c.slotOwner s with
  | some (b, j) => valAt b j g e
  | none => 3

def bfv (z : Int) : BF16 := FloatV.ofRat (z : Rat)

/-- A sentinel in `o_extend` and in `edge` elements on each side of it. -/
def sentinel : BF16 := FloatV.ofRat (-99)
def edge : Nat := 4

/-- `kv_indices`: each request's slots from `kv_indptr[b]`; the unused entries `-1`. -/
def Case.kvIndex (c : Case) (n : Nat) : Int :=
  match (List.range c.batch).find? fun b => c.ks b ≤ n ∧ n < c.ks b + (c.req b).pre with
  | some b => ((c.req b).slots.getD (n - c.ks b) 0 : Nat)
  | none => -1

/-- `custom_mask`: request `b`'s block, row `i`, column `x`: the window's offset columns `false`,
the prefix columns `true` (never read), the new tokens' the tree; elsewhere `false`. -/
def Case.maskAt (c : Case) (n : Nat) : Bool :=
  match (List.range c.batch).find? fun b => c.ms b ≤ n ∧ n < c.ms b + (c.req b).ext * c.rowLen b with
  | some b =>
    let r := c.req b
    let i := (n - c.ms b) / c.rowLen b
    let x := (n - c.ms b) % c.rowLen b
    if x < r.wOff then false else if x < r.wOff + r.pre then true else c.bit b i (x - r.wOff - r.pre)
  | none => false

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let tens (buf : Buf .bf16) (rows heads : Nat) (x : Nat → Nat → Nat → Int) : List (Int × BF16) :=
    (List.range (rows * heads * c.D)).map fun k : Nat =>
      ((buf.ptr k).addr, bfv (x (k / (heads * c.D)) (k / c.D % heads) (k % c.D)))
  let qrow := (List.range c.T).map fun t => (List.range c.H).map fun h => (List.range c.D).map fun e => c.qv t h e
  let m := CMem.ofElems ∅ .bf16 (tens d.q c.T c.H fun t h e => (qrow.getD t []).getD h [] |>.getD e 0)
  let m := CMem.ofElems m .bf16 (tens d.k c.T c.HKV c.kv)
  let m := CMem.ofElems m .bf16 (tens d.v c.T c.HKV c.vv)
  let m := CMem.ofElems m .bf16 (tens d.kb c.S c.HKV c.kbv)
  let m := CMem.ofElems m .bf16 (tens d.vb c.S c.HKV c.vbv)
  let m := CMem.ofElems m (Claims.ExtendAttn.qoD c.cm) ((List.range (c.batch + 1)).map fun b : Nat =>
    ((d.qo.ptr b).addr, (Claims.ExtendAttn.qoD c.cm).ofBits (c.qs b)))
  let m := CMem.ofElems m .i32 ((List.range (c.batch + 1)).map fun b : Nat =>
    ((d.kvp.ptr b).addr, (IntV.ofNat (c.ks b) : I32)))
  let m := CMem.ofElems m .i64 ((List.range c.NI).map fun n : Nat =>
    ((d.kvi.ptr n).addr, (IntV.ofInt (c.kvIndex n) : I64)))
  let m := if c.cm then
      let m := CMem.ofElems m .i1 ((List.range c.NM).map fun n : Nat => ((d.cmask.ptr n).addr, c.maskAt n))
      let m := CMem.ofElems m .i64 ((List.range (c.batch + 1)).map fun b : Nat =>
        ((d.mip.ptr b).addr, (IntV.ofNat (c.ms b) : I64)))
      if 0 < c.sw then CMem.ofElems m .i64 ((List.range c.batch).map fun b : Nat =>
        ((d.wko.ptr b).addr, (IntV.ofNat (c.req b).wOff : I64))) else m
    else m
  CMem.ofElems m .bf16 ((List.range (c.T * c.H * c.D + 2 * edge)).map fun e : Nat =>
    ((d.o.ptr ((e : Int) - edge)).addr, sentinel))

def Case.run (c : Case) : LRes c.L.size × List Record :=
  execLaunch (c.L.env P0) O fill0 100000 (Claims.ExtendAttn.model c.inst) c.L c.init

/-! ## The reference, from the inputs -/

def Case.kvh (c : Case) (h : Nat) : Nat := h / (c.H / c.HKV)

/-- Component `e` of the key / value at position `j` of request `b` (head `h`): through the slots
for a cached position, else the new token's. -/
def Case.keyK (c : Case) (b h j e : Nat) : Int :=
  let r := c.req b
  if j < r.pre then c.kbv (r.slots.getD j 0) (c.kvh h) e else c.kv (c.qs b + (j - r.pre)) (c.kvh h) e
def Case.keyV (c : Case) (b h j e : Nat) : Int :=
  let r := c.req b
  if j < r.pre then c.vbv (r.slots.getD j 0) (c.kvh h) e else c.vv (c.qs b + (j - r.pre)) (c.kvh h) e

/-- Query row `(b, h, i)`. -/
def Case.qrow (c : Case) (b h i : Nat) : Array Int := (Array.range c.D).map (c.qv (c.qs b + i) h)

def Case.score (c : Case) (qr : Array Int) (b h j : Nat) : Rat :=
  sigma * (((List.range c.D).map fun e => qr[e]! * c.keyK b h j e).sum : Int)

def Case.scoreMag (c : Case) (qr : Array Int) (b h j : Nat) : Rat :=
  sigma * (((List.range c.D).map fun e => |qr[e]! * c.keyK b h j e|).sum : Int)

/-- Row `(b, h, i)`'s expected outputs: the mean `c` of the values at the maximum score over the
attended positions, and the bound `r` on its distance to `attnRef` (`none` if the gap is below
`200` or the row attends nothing). -/
def Case.expectRow (c : Case) (b h i : Nat) : Option (Array (Rat × Rat)) :=
  let r := c.req b
  let js := (List.range (r.pre + r.ext)).filter (c.attendsW (c.expSw.getD c.sw) b i)
  let qr := c.qrow b h i
  let sc := js.map fun j => (j, c.score qr b h j)
  let mx := (sc.map (·.2)).foldl max ((sc.headD (0, 0)).2)
  let top := (sc.filter fun (_, s) => s == mx).map (·.1)
  let low := (sc.filter fun (_, s) => s != mx).map (·.1)
  let gap := (sc.filter fun (_, s) => s != mx).foldl (fun g (_, s) => min g (mx - s)) (256 : Rat)
  let n : Rat := top.length
  if gap < 200 || js.isEmpty then none
  else some ((Array.range c.D).map fun e =>
    let v (j : Nat) : Rat := c.keyV b h j e
    let ctr := (top.map v).sum / n
    (ctr, (2 : Rat) ^ (-(gap.floor) : Int) * ((low.map fun j => |v j|).sum + (low.length : Rat) * |ctr|) / n))

/-- The final memory of `c`'s run against `e`'s expectations (`e = c` for the test). -/
def Case.agreeE (c e : Case) (cf : CConfig c.L.size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let rows := (List.range c.batch).flatMap fun b : Nat => (List.range (c.req b).ext).flatMap fun i : Nat =>
    (List.range c.H).map fun h : Nat => (b, i, h)
  let bad := rows.filter fun (b, i, h) =>
    match e.expectRow b h i with
    | none => true
    | some ex => (List.range c.D).any fun k =>
      match m.read .bf16 (d.o.ptr (((c.qs b + i) * c.H + h) * c.D + k : Nat)).addr, ex[k]? with
      | some w, some (ctr, r) => !(w.isFinite && w.toRat == ctr && r ≤ 1 / 2 ^ 80)
      | _, _ => true
  let owned (a : Int) : Bool :=
    let k := (a - d.o.base) / 2
    d.o.base ≤ a && a < d.o.base + 2 * (c.T * c.H * c.D : Nat) &&
      (c.locate (k / (c.H * c.D)).toNat).isSome
  let changed := c.init.toList.filter fun (a, b) => !owned a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"rows (b, i, h) {bad.take 8} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the owned outputs changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c

/-! ## `Pre` on the inputs -/

/-- `ReqOk`, `MaskOk`, `ValsOk` and the buffers' alignment (every byte is written by `init`); every
stored row has a colour (its top count a power of two) and an expectation. -/
def Case.pre (c : Case) : Bool :=
  let d := c.inst
  d.q.base % 2 == 0 && d.k.base % 2 == 0 && d.v.base % 2 == 0 && d.o.base % 2 == 0 &&
  d.kb.base % 2 == 0 && d.vb.base % 2 == 0 && d.qo.base % (Claims.ExtendAttn.qoD c.cm).bytes == 0 &&
  d.kvp.base % 4 == 0 && d.kvi.base % 8 == 0 && d.mip.base % 8 == 0 && d.wko.base % 8 == 0 &&
  -- `ReqOk`
  c.qs c.batch ≤ c.T && c.ks c.batch ≤ c.NI &&
  (List.range c.batch).all (fun b =>
    (c.req b).ext ≤ c.maxExt && (c.req b).pre + (c.req b).ext ≤ 2 ^ 18 &&
    (c.req b).slots.all (· < c.S)) &&
  -- `MaskOk`: the blocks fit; each new token sees itself and no later one
  (!c.cm || (c.ms c.batch ≤ c.NM && (List.range c.batch).all fun b =>
    (c.req b).wOff ≤ 2 ^ 40 && (List.range (c.req b).ext).all fun i =>
      c.bit b i i && (List.range (c.req b).ext).all fun j => j ≤ i || !c.bit b i j)) &&
  -- `ValsOk` (small integers, exact in `bfloat16`; `scoreMag ≤ 2^12` on the attended pairs)
  (List.range c.batch).all fun b => (List.range c.H).all fun h => (List.range (c.req b).ext).all fun i =>
    (let qr := c.qrow b h i
     (List.range ((c.req b).pre + (c.req b).ext)).all fun j =>
      !c.attends b i j || decide (c.scoreMag qr b h j ≤ 2 ^ 12)) &&
    (c.rowColour b h i).isSome && (c.expectRow b h i).isSome

/-- The platform answers: per program that does not return at once (`ext > 128·m`), one `/`, and
per trip run two `tl.dot` and two `tl.exp`. The trips: prefix tiles of `64`, then new-token tiles
from `extend_start` to `min(ext, 128·(m + 1))`; with a window or the tree mask, a tile is skipped
when no row of the block attends any of its keys. -/
def Case.answers (c : Case) : Nat :=
  c.H * ((List.range c.batch).map fun b =>
    let r := c.req b
    ((List.range ((c.maxExt + 127) / 128)).map fun m =>
      if r.ext ≤ 128 * m then 0 else
        let rowsM := (List.range r.ext).filter fun i => 128 * m ≤ i ∧ i < 128 * (m + 1)
        let pTiles := (List.range ((r.pre + 63) / 64)).filter fun t =>
          c.sw == 0 || rowsM.any fun i => (List.range 64).any fun n =>
            64 * t + n < r.pre && c.attends b i (64 * t + n)
        let start := if c.sw > 0 then (128 * m - min (128 * m) c.sw) / 64 * 64 else 0
        let endE := min r.ext (128 * (m + 1))
        let eTiles := (List.range ((endE + 63) / 64)).filter fun t => start ≤ 64 * t ∧
          (!(c.cm || c.sw > 0) || rowsM.any fun i => (List.range 64).any fun n =>
            64 * t + n < endE && c.attends b i (r.pre + 64 * t + n))
        1 + 4 * (pTiles.length + eTiles.length)).sum).sum

/-! ## The instances -/

def extend : Case :=
  { cm := false, sw := 0, H := 1, HKV := 1, D := 12, BD := 16, S := 80, T := 140, NI := 72,
    qo0 := 2, kv0 := 1, grouped := true,
    reqs := [{ slots := (List.range 70).map fun j => (j * 7 + 3) % 80, ext := 5 },
             { slots := [], ext := 130 }] }
theorem extend_ok : extend.inst.ok :=
  Claims.ExtendAttn.canon_ok false 0 2 140 80 72 _ 1 1 12 16 128 130 4 _
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by decide)
    ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩ (by norm_num) (by norm_num)

def verify : Case :=
  { cm := true, sw := 8, H := 2, HKV := 1, D := 16, BD := 16, S := 24, T := 12, NI := 22,
    qo0 := 1, kv0 := 2, mi0 := 5,
    reqs := [{ slots := (List.range 20).map fun j => (j * 5 + 1) % 24, ext := 6,
               parent := fun i => [none, some 0, some 0, some 1, some 1, some 2].getD i none,
               wOff := 2 },
             { slots := [], ext := 4, parent := fun i => if i = 0 then none else some (i - 1) }] }
theorem verify_ok : verify.inst.ok :=
  Claims.ExtendAttn.canon_ok true 8 2 12 24 22 _ 2 1 16 16 128 6 4 _
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (fun _ => ⟨by norm_num, by norm_num⟩) (by decide)
    ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩ (by norm_num) (by norm_num)

#guard (FloatV.ofPat smBits : F64).toRat == sigma
#guard verify.pre && extend.pre

/-- Run a case once: print its verdict line, and fail unless the verdict is `conditional` with the
expected count, and every wrong expectation `es` is caught (a discrepancy) on the same run.
(A trip takes about 22 s in `exec`: one command per case, so each runs once.) -/
def check (name : String) (c : Case) (es : List Case) : IO Unit := do
  let r := c.run
  let v := Verdict.ofLaunch c.agree r
  IO.println (v.line s!"extend_attn_bf16/{name}")
  unless v == .conditional c.answers && v.assumed > 0 do
    throw (IO.userError s!"{name}: expected conditional {c.answers}")
  unless es.all fun e => (Verdict.ofLaunch (c.agreeE e) r).isDiscrepancy do
    throw (IO.userError s!"{name}: a wrong expectation agrees")

-- Not vacuous: wrong expectations are caught on the same runs (`verify`'s inputs read without
-- the window; `extend`'s with a window of `16`).
#eval check "verify" verify [{ verify with expSw := some 0 }]
#eval check "extend" extend [{ extend with expSw := some 16 }]
