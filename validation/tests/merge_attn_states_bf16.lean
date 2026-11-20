import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `merge_attn_states_bf16` run by `execLaunch` on two small covered
instances of the host's layout (`Claims.MergeAttnStates.canon`, covered by `canon_ok`). Not a test
against the device (phase 4): it checks that `exec` runs the model end to end, and that the final
memory is what `Post` says, both halves.

* `withLse`: `return_lse = True`, 4 tokens, 3 heads, `head_size = 3` padded to `4` (lane `3`
  masked), heads padded by one element in the inputs, transposed lse views (`lseT`),
  `prefill_tokens_with_context = 3`: tokens `0..2` merged, token `3` copied. The nine merged
  `(t, h)` cover every path of the merge: two equal finite lse values; one side empty (`-∞`, or
  `+∞` read as `-∞`) on either side; both sides empty (`-∞, -∞` and `+∞, +∞`, with NaN in their
  output rows, which `Pre` allows and the kernel must zero); two finite lse values `250` apart on
  either side. The copied token holds NaN lse values and a NaN and a `-0` in its rows (nothing is
  required of them; they must be copied bit for bit).
* `noLse`: `return_lse = False` (`OUTPUT_LSE = False`), 3 tokens, 2 heads, `head_size = 2`
  (unpadded), `[H, T]` lse, `prefill_tokens_with_context = 2`; the unused `output_lse` storage
  holds sentinels that must stay.

**Platform answers** (`tl.exp`, `tl.log`, `/` on `float32` are platform sites; the verdicts are
`conditional`, one answer per site per merged program: two `exp`, two `/`, and one `log` with
`output_lse`). The oracle (`O`) answers lane by lane only these values, which are the only ones a
correct device gives (any other request is left unanswered):
* `exp(0) = 1`, `exp(-∞) = +0`, `exp(NaN) = NaN` (both-empty rows: `-∞ - (-∞)`), and
  `exp(x) = +0` for `x ≤ -200` (`e^-200 < 2^-288`, far below half the smallest `float32`
  subnormal: the correctly rounded value, and the one any relative error bound allows);
* `log(1) = +0`, `log(NaN) = NaN`, and `log(2)`, the correctly rounded `float32` of `ln 2`
  (checked: both ends of a rational interval around `ln 2` round to it). `ln 2` is irrational, so
  this one is not exact; the merged lse of the equal-lse rows is checked against `Post`'s own
  tolerance `2^-16·(1 + |refLse|)` over the whole interval;
* `/`: the IEEE quotient (`1/2`, `1/1`, `0/1`, `NaN/NaN` here: exact).

**The check** (`agree`), from the inputs as the test writes them (rationals, not the claim's
definitions): for a merged `(t, h)` with both sides empty, the lse is exactly `-∞` and each output
element exactly `+0`; otherwise the reference `refLse = m + log(Σ e^(x−m))` and
`refOut = Σ e^(x−m)·v / Σ e^(x−m)` are enclosed in a rational centre `c` and radius `r` (a weight
`e^-250 ≤ 2^-250` is the only inexact one besides `ln 2`), and the output is finite with
`|w − c| + r ≤` `Post`'s tolerance at the smallest `|ref|` in the enclosure: a sufficient condition
for `Post`. Where the reference is exact (`r = 0` for the output: every row but the `250`-apart
ones), the output must moreover equal `c` exactly. A copied `(t, h)`: the suffix's lse and row
bit for bit. Every other byte is unchanged (inputs, the inputs' head padding, sentinels around
`output`, the unused `output_lse`), and no byte is added. -/

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

/-- Rational bounds on `ln 2 = 0.693147180559945309…`. -/
def ln2lo : Rat := 6931471805599453 / 10 ^ 16
def ln2hi : Rat := 6931471805599454 / 10 ^ 16

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

/-- `log` where its value is exact, or (at `2`) correctly rounded. -/
def logAns {f : Format} (x : FloatV f) : Option (FloatV f) :=
  if x.isNaN then some FloatV.nan
  else if x.isFinite && x.toRat == 1 then some (FloatV.ofPat 0)
  else if x.isFinite && x.toRat == 2 &&
      (FloatV.ofRat ln2lo : FloatV f).pat == (FloatV.ofRat ln2hi : FloatV f).pat then
    some (FloatV.ofRat ln2lo)
  else none

def O : Oracle := fun _ _ q => match q with
  | .plat (.math .exp _ x) => lanewise expAns x
  | .plat (.math .log _ x) => lanewise logAns x
  | .plat (.div _ x y) => some (Tensor.map2 FloatV.divRn x y)
  | _ => none

-- `ln 2` lies in `[ln2lo, ln2hi]` (`0.69314718055994530941…`), and both ends round to the same
-- `float32` (`0x3F317218`): the answer is the correctly rounded `log(2)`.
#guard (FloatV.ofRat ln2lo : F32).pat == 0x3F317218 && (FloatV.ofRat ln2hi : F32).pat == 0x3F317218

/-! ## The cases -/

/-- An lse value as the test writes it. -/
inductive LV where
  | num (z : Int)
  | pinf
  | ninf
  | nan
  deriving BEq, Inhabited

def LV.f32 : LV → F32
  | .num z => FloatV.ofRat (z : Rat)
  | .pinf => FloatV.posInf
  | .ninf => FloatV.negInf
  | .nan => FloatV.nan

/-- `+∞` read as `-∞` (an empty side), as the source reads it. -/
def LV.corr : LV → LV
  | .pinf => .ninf
  | x => x

/-- An output element as the test writes it: a small integer, `-0` or NaN. -/
inductive EV where
  | num (z : Int)
  | negZero
  | nan
  deriving BEq, Inhabited

def EV.bf16 : EV → BF16
  | .num z => FloatV.ofRat (z : Rat)
  | .negZero => FloatV.ofPat 0x8000
  | .nan => FloatV.nan

def EV.rat : EV → Rat
  | .num z => z
  | _ => 0

structure Case where
  T : Nat
  H : Nat
  hs : Nat
  padded : Nat
  ptwc : Nat
  pad : Nat
  lseT : Bool
  outLse : Bool
  /-- `prefix_lse[h, t]`, `suffix_lse[h, t]`. -/
  plse : Nat → Nat → LV
  slse : Nat → Nat → LV
  /-- `prefix_output[t, h, j]`, `suffix_output[t, h, j]` (`j < head_size`). -/
  pv : Nat → Nat → Nat → EV
  sv : Nat → Nat → Nat → EV

def Case.inst (c : Case) : Claims.MergeAttnStates.Inst :=
  Claims.MergeAttnStates.canon c.T c.H c.hs c.padded c.ptwc c.pad c.lseT c.outLse

/-- Sentinels: in `output`, `output_lse`, the inputs' head padding, and `pad` elements on each side
of `output` and `output_lse`. -/
def sentB : BF16 := FloatV.ofRat (-77)
def sentF : F32 := FloatV.ofRat (-55)
def edge : Nat := 4

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let hp := c.hs + c.pad
  -- the inputs' storage: every element, the head padding a sentinel
  let rows (b : Buf .bf16) (v : Nat → Nat → Nat → EV) : List (Int × BF16) :=
    (List.range (c.T * c.H * hp)).map fun k : Nat =>
      let t := k / (c.H * hp)
      let h := k / hp % c.H
      let j := k % hp
      ((b.ptr k).addr, if j < c.hs then (v t h j).bf16 else sentB)
  let lses (b : Buf .f32) (v : Nat → Nat → LV) : List (Int × F32) :=
    (List.range c.H).flatMap fun h : Nat => (List.range c.T).map fun t : Nat =>
      ((b.ptr (if c.lseT then t * c.H + h else h * c.T + t : Nat)).addr, (v h t).f32)
  let m := CMem.ofElems ∅ .bf16 (rows d.prefix_output c.pv)
  let m := CMem.ofElems m .bf16 (rows d.suffix_output c.sv)
  let m := CMem.ofElems m .f32 (lses d.prefix_lse c.plse)
  let m := CMem.ofElems m .f32 (lses d.suffix_lse c.slse)
  let m := CMem.ofElems m .bf16 ((List.range (c.T * c.H * c.hs + 2 * edge)).map fun e : Nat =>
    ((d.output.ptr ((e : Int) - edge)).addr, sentB))
  CMem.ofElems m .f32 ((List.range (c.H * c.T + 2 * edge)).map fun e : Nat =>
    ((d.output_lse.ptr ((e : Int) - edge)).addr, sentF))

def Case.run (c : Case) : LRes (Claims.MergeAttnStates.launch c.inst).size × List Record :=
  execLaunch ((Claims.MergeAttnStates.launch c.inst).env P0) O fill0 1000
    (Claims.MergeAttnStates.model c.inst) (Claims.MergeAttnStates.launch c.inst) c.init

/-! ## The reference, from the inputs -/

/-- A side's weight `e^(x − m)` enclosed: `(centre, radius)`; `none` outside the test's
categories. -/
def weight (x : LV) (m : Int) : Option (Rat × Rat) :=
  match x with
  | .ninf => some (0, 0)
  | .num z => if z = m then some (1, 0) else if z - m ≤ -200 then some (2 ^ (-201 : Int), 2 ^ (-201 : Int)) else none
  | _ => none

/-- The reference of a merged `(t, h)` that is not both-empty, enclosed: the lse
(`(centre, radius)`) and, per element, the output (`(centre, radius)`). -/
def refOf (p s : LV) (pv sv : Rat) : Option ((Rat × Rat) × (Rat × Rat)) := do
  let m : Int ← match p, s with
    | .num a, .num b => some (max a b)
    | .num a, .ninf => some a
    | .ninf, .num b => some b
    | _, _ => none
  let (wp, rp) ← weight p m
  let (ws, rs) ← weight s m
  -- the exact sum of the weights lies in `[lo, hi]`
  let lo := wp - rp + (ws - rs)
  let hi := wp + rp + (ws + rs)
  -- `log` of the sum: `0` at `1` (`log(1 + ε) ∈ [0, ε]`), `ln 2` at `2`
  let (lc, lr) ← if lo = 1 then some ((hi - 1) / 2, (hi - 1) / 2)
    else if lo = 2 ∧ hi = 2 then some ((ln2lo + ln2hi) / 2, (ln2hi - ln2lo) / 2)
    else none
  -- the output `(pv·wp + sv·ws) / S`: exact when both weights are; otherwise within
  -- `|pv − sv|·ε / 1` of the side of weight `1`
  let out ← if rp = 0 ∧ rs = 0 then some ((pv * wp + sv * ws) / (wp + ws), 0)
    else if rs ≠ 0 ∧ rp = 0 ∧ wp = 1 then some (pv, |sv - pv| * 2 * rs)
    else if rp ≠ 0 ∧ rs = 0 ∧ ws = 1 then some (sv, |pv - sv| * 2 * rp)
    else none
  some ((m + lc, lr), out)

def lseTol : Rat := 1 / 2 ^ 16
def bf16Round : Rat := 1 / 2 ^ 8
def outTol : Rat := 1 / 2 ^ 13
def absTol : Rat := 1 / 2 ^ 133

/-- `Post`'s tolerance holds at every value of an enclosure `(c, r)`. -/
def withinLse (x : F32) (c r : Rat) : Bool :=
  x.isFinite && |x.toRat - c| + r ≤ lseTol * (1 + max 0 (|c| - r))

def withinOut (w : BF16) (c r : Rat) (pv sv : Rat) : Bool :=
  w.isFinite && |w.toRat - c| + r ≤ bf16Round * max 0 (|c| - r) + outTol * max |pv| |sv| + absTol

/-- The final memory of `c`'s run against the expectations of `e` (`e = c` for the test; another
`e` for the non-vacuity guard): `e`'s inputs give the expected values, `c`'s initial memory the
unchanged bytes. -/
def Case.agreeE (c e : Case) (cf : CConfig (Claims.MergeAttnStates.launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let rdF (b : Buf .f32) (i : Nat) : Option F32 := m.read .f32 (b.ptr i).addr
  let rdB (b : Buf .bf16) (i : Nat) : Option BF16 := m.read .bf16 (b.ptr i).addr
  let oIdx (t h j : Nat) : Nat := (t * c.H + h) * c.hs + j
  let olIdx (t h : Nat) : Nat := h * c.T + t
  let pairs := (List.range c.T).flatMap fun t : Nat => (List.range c.H).map fun h : Nat => (t, h)
  let bad := pairs.filter fun (t, h) =>
    let p := (e.plse h t).corr
    let s := (e.slse h t).corr
    if t < c.ptwc then
      if p == .ninf && s == .ninf then
        (c.outLse && (rdF d.output_lse (olIdx t h)).map (·.pat) != some (FloatV.negInf : F32).pat) ||
        (List.range c.hs).any fun j => (rdB d.output (oIdx t h j)).map (·.pat) != some 0
      else
        (List.range c.hs).any (fun j =>
          let pv := (e.pv t h j).rat
          let sv := (e.sv t h j).rat
          match refOf p s pv sv, rdB d.output (oIdx t h j) with
          | some (_, (oc, or_)), some w => !withinOut w oc or_ pv sv || (or_ == 0 && w.toRat != oc)
          | _, _ => true) ||
        (c.outLse && match refOf p s 0 0, rdF d.output_lse (olIdx t h) with
          | some ((lc, lr), _), some x => !withinLse x lc lr
          | _, _ => true)
    else
      (c.outLse && (rdF d.output_lse (olIdx t h)).map (·.pat) != some (e.slse h t).f32.pat) ||
      (List.range c.hs).any fun j => (rdB d.output (oIdx t h j)).map (·.pat) != some (e.sv t h j).bf16.pat
  -- the bytes the launch may write: `output`'s `T·H·hs` elements, `output_lse`'s `H·T` with it
  let inOut (a : Int) : Bool :=
    (d.output.base ≤ a && a < d.output.base + 2 * (c.T * c.H * c.hs : Nat)) ||
    (c.outLse && d.output_lse.base ≤ a && a < d.output_lse.base + 4 * (c.H * c.T : Nat))
  let changed := c.init.toList.filter fun (a, b) => !inOut a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"(t, h) {bad} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the outputs changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c

/-! ## The instances -/

/-- `withLse`'s lse pairs `(prefix, suffix)` by `(t, h)`; token `3` is copied. -/
def lseA (t h : Nat) : LV × LV :=
  match t, h with
  | 0, 0 => (.num 0, .num 0)
  | 0, 1 => (.num 5, .ninf)
  | 0, 2 => (.ninf, .num (-3))
  | 1, 0 => (.pinf, .num (-2))
  | 1, 1 => (.ninf, .ninf)
  | 1, 2 => (.pinf, .pinf)
  | 2, 0 => (.num 100, .num (-150))
  | 2, 1 => (.num (-150), .num 100)
  | 2, 2 => (.num 7, .num 7)
  | _, _ => (.nan, .nan)

/-- Small integers, varying with every index. -/
def pvA (t h j : Nat) : EV :=
  if t = 1 ∧ h = 1 ∧ j = 0 then .nan  -- a both-empty row
  else if t = 3 ∧ h = 1 ∧ j = 1 then .nan  -- a copied row
  else .num (((t * 7 + h * 3 + j * 5) % 11 : Nat) - (5 : Int))
def svA (t h j : Nat) : EV :=
  if t = 1 ∧ h = 2 ∧ j = 2 then .nan
  else if t = 3 ∧ h = 2 ∧ j = 0 then .negZero
  else if t = 3 ∧ h = 0 ∧ j = 2 then .nan
  else .num (((t * 5 + h * 7 + j * 3 + 2) % 13 : Nat) - (6 : Int))

def withLse : Case :=
  { T := 4, H := 3, hs := 3, padded := 4, ptwc := 3, pad := 1, lseT := true, outLse := true,
    plse := fun h t => (lseA t h).1, slse := fun h t => (lseA t h).2, pv := pvA, sv := svA }
theorem withLse_ok : withLse.inst.ok :=
  Claims.MergeAttnStates.canon_ok 4 3 3 4 3 1 2 true true (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

def lseB (t h : Nat) : LV × LV :=
  match t, h with
  | 0, 0 => (.num (-4), .num (-4))
  | 0, 1 => (.num 2, .pinf)
  | 1, 0 => (.ninf, .num 1)
  | 1, 1 => (.num (-210), .num 0)
  | _, _ => (.num 3, .ninf)

def noLse : Case :=
  { T := 3, H := 2, hs := 2, padded := 2, ptwc := 2, pad := 0, lseT := false, outLse := false,
    plse := fun h t => (lseB t h).1, slse := fun h t => (lseB t h).2,
    pv := fun t h j => .num ((t + 2 * h + 3 * j : Nat) - (3 : Int)),
    sv := fun t h j => .num ((3 * t + h + 5 * j : Nat) % 7 - (2 : Int)) }
theorem noLse_ok : noLse.inst.ok :=
  Claims.MergeAttnStates.canon_ok 3 2 2 2 2 0 1 false false (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-! ## `Pre` on the initial memories -/

/-- Every buffer is aligned and every byte of it present (`output_lse` too: it is written either
way); each merged `(t, h)` has lse values that are not NaN and, unless both sides are empty, finite
rows; every enclosure the check needs exists (the test's categories). -/
def Case.pre (c : Case) : Bool :=
  let d := c.inst
  let m := CMem.toMem c.init
  let validB (b : Buf .bf16) := b.base % 2 == 0 && (List.range b.len).all fun i : Nat => (m.read .bf16 (b.ptr i).addr).isSome
  let validF (b : Buf .f32) := b.base % 4 == 0 && (List.range b.len).all fun i : Nat => (m.read .f32 (b.ptr i).addr).isSome
  validB d.output && validF d.output_lse && validB d.prefix_output && validB d.suffix_output &&
  validF d.prefix_lse && validF d.suffix_lse &&
  (List.range c.ptwc).all fun t => (List.range c.H).all fun h =>
    let p := c.plse h t
    let s := c.slse h t
    p != .nan && s != .nan &&
    (p.corr == .ninf && s.corr == .ninf ||
      (List.range c.hs).all fun j =>
        c.pv t h j != .nan && c.sv t h j != .nan && (refOf p.corr s.corr (c.pv t h j).rat (c.sv t h j).rat).isSome)

#guard withLse.pre && noLse.pre
-- Every merged row's expectation differs from the sentinel, so the stores are seen.
#guard [withLse, noLse].all fun c => (List.range c.ptwc).all fun t => (List.range c.H).all fun h =>
  (List.range c.hs).all fun j => match refOf (c.plse h t).corr (c.slse h t).corr (c.pv t h j).rat (c.sv t h j).rat with
    | some (_, (oc, _)) => oc != sentB.toRat
    | none => true

def verdictWith : Verdict := Verdict.ofLaunch withLse.agree withLse.run
def verdictNo : Verdict := Verdict.ofLaunch noLse.agree noLse.run

#eval IO.println (verdictWith.line "merge_attn_states_bf16/with_lse")
#eval IO.println (verdictNo.line "merge_attn_states_bf16/no_lse")
-- One answer per platform site of a merged program: `2·exp + 2·/ (+ log)`.
#guard verdictWith == .conditional (3 * 3 * 5)
#guard verdictNo == .conditional (2 * 2 * 4)

-- Not vacuous: wrong expectations are caught (the prefix and suffix rows swapped; the copied
-- token's suffix row replaced by its prefix row).
#guard
  let r := withLse.run
  (Verdict.ofLaunch (withLse.agreeE { withLse with pv := svA, sv := pvA }) r).isDiscrepancy &&
  (Verdict.ofLaunch (withLse.agreeE { withLse with sv := fun t h j => if t = 3 then pvA t h j else svA t h j }) r).isDiscrepancy
