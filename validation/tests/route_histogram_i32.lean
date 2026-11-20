import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `route_histogram_i32` run by `execLaunch` on four small instances,
each launching several programs that add into the same counters: the `BINS == 0` path (one
`tl.atomic_add` per pair) without and with groups, and the `tl.histogram` path (one
`tl.atomic_add` per nonzero bin) with groups of more than one per slot and without groups. Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says: each counter is its initial value plus the number of pairs in
its bucket (counted here from the inputs, with the `int32` wrap), and every other element (the
inputs, and two guard elements around `counts`) is unchanged. Integer atomics need no platform
answer, so the verdict is `certified` with no oracle. -/

open TVA TVA.Validation Claims.RouteHistogram

/-- A platform that relates nothing: `route_histogram_i32` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- A sentinel in the guard elements just outside `counts`: they must keep it. -/
def sentinel : Int := -7

/-- The `int32` wrap of an integer. -/
def wrap32 (z : Int) : Int := (z + 2 ^ 31) % 2 ^ 32 - 2 ^ 31

/-- One case: `ntok` tokens of `w` pairs, `gps` groups per slot, `ml` adapters, with groups or
not, `BLOCK = 2^eb`, `BINS = 2^ek` on the `tl.histogram` path (`hist`), and the contents. -/
structure Case where
  ntok : Nat
  w : Nat
  gps : Nat
  ml : Nat
  hg : Bool
  eb : Nat
  hist : Bool
  ek : Nat
  /-- `token_slots`, one per token. -/
  ts : List Int
  /-- `group_ids`, one per pair (empty without groups). -/
  gi : List Int
  /-- The initial `counts`, one per bucket. -/
  cnt : List Int

def Case.base (c : Case) : Inst :=
  Claims.RouteHistogram.mk c.ntok c.w c.gps c.ml c.hg c.eb c.hist c.ek
/-- The instance: `mk`'s, with the case's contents (`Pre`'s `Holds`). -/
def Case.inst (c : Case) : Inst :=
  { c.base with
    tsVal := fun j => IntV.ofInt (c.ts.getD j 0)
    giVal := fun j => IntV.ofInt (c.gi.getD j 0)
    cntVal := fun j => IntV.ofInt (c.cnt.getD j 0) }
def Case.n (c : Case) : Nat := c.ntok * c.w
def Case.nb (c : Case) : Nat := c.gps * c.ml + 1

def writeI32 (m : CMem) (b : Buf .i32) (xs : List Int) (off : Int := 0) : CMem :=
  CMem.ofElems m .i32 ((List.range xs.length).map fun j : Nat =>
    ((b.ptr ((j : Int) + off)).addr, (IntV.ofInt (xs.getD j 0) : I32)))

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := writeI32 ∅ d.tokenSlots c.ts
  -- without groups the kernel must not read `group_ids` (a load of an absent byte is `ub`)
  let m := if c.hg then writeI32 m d.groupIds c.gi else m
  -- `counts`, between two guard elements
  writeI32 m d.counts ([sentinel] ++ c.cnt ++ [sentinel]) (-1)

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) Oracle.empty fill0 4000 (model c.inst) (launch c.inst) c.init

/-- The bucket of pair `p`, from the inputs: its token's slot `s` and its group `g`; valid when
`0 ≤ s < MAX_LORAS` and, with groups, `g ≥ 0` and (more than one group per slot) `g <
GROUPS_PER_SLOT`; then `s * GROUPS_PER_SLOT + g` with groups and more than one group per slot,
`s` otherwise; the last bucket when invalid. -/
def Case.bucket (c : Case) (p : Nat) : Int :=
  let s := c.ts.getD (p / c.w) 0
  let g := c.gi.getD p 0
  let ok := 0 ≤ s && s < c.ml && (!c.hg || (0 ≤ g && (c.gps == 1 || g < c.gps)))
  if ok then (if c.hg && c.gps != 1 then s * c.gps + g else s) else c.nb - 1

/-- What `Post` says element `b` of `counts` holds: its initial value plus the number of pairs in
bucket `b`, wrapped to `int32`. -/
def Case.expect (c : Case) (b : Nat) : Int :=
  wrap32 (c.cnt.getD b 0 + ((List.range c.n).filter fun p => c.bucket p == (b : Int)).length)

/-- The final memory against an expectation `exp` for `counts` and `guard` for the two guard
elements: every counter, the guards, and the inputs unchanged. -/
def Case.agreeWith (c : Case) (exp : Nat → Int) (guard : Int) (cf : CConfig (launch c.inst).size) :
    Option String :=
  let d := c.inst
  let m0 := CMem.toMem c.init
  let m := CMem.toMem cf.mem
  let rd (j : Int) := (m.read .i32 (d.counts.ptr j).addr).map (·.toInt)
  let badOut := (List.range c.nb).filter fun b : Nat => rd b != some (exp b)
  let guardsOk := rd (-1) == some guard && rd c.nb == some guard
  let same (b : Buf .i32) (n : Nat) : Bool :=
    (List.range n).all fun j : Nat =>
      (m.read .i32 (b.ptr j).addr).map (·.toInt) == (m0.read .i32 (b.ptr j).addr).map (·.toInt)
  let inputsSame := same d.tokenSlots c.ntok && (!c.hg || same d.groupIds c.n)
  if !badOut.isEmpty then some s!"counts elements {badOut} differ from Post"
  else if !guardsOk then some "an element outside counts changed"
  else if !inputsSame then some "an input changed"
  else none

def Case.agree (c : Case) := c.agreeWith c.expect sentinel

/-- `BINS == 0`, no groups: 10 tokens, 3 adapters (`NUM_BUCKETS = 4`), `BLOCK = 4` (3 programs,
the last masked); invalid slots `-1`, `3`, `5` go to the last bucket; nonzero initial counts. -/
def atomicNoGroups : Case :=
  { ntok := 10, w := 1, gps := 1, ml := 3, hg := false, eb := 2, hist := false, ek := 0,
    ts := [0, 2, 1, 0, -1, 3, 2, 0, 5, 1], gi := [], cnt := [5, 0, 100, 2] }

/-- `tl.histogram` path, groups, 3 groups per slot, 2 adapters (`NUM_BUCKETS = 7`, `BINS = 8`): 5
tokens of 2 pairs, `BLOCK = 4` (3 programs; the masked lanes go to bin 7, not added); an invalid
slot (`-1`, and `2 = MAX_LORAS`), a negative group, a group `≥ GROUPS_PER_SLOT`. -/
def histGroups : Case :=
  { ntok := 5, w := 2, gps := 3, ml := 2, hg := true, eb := 2, hist := true, ek := 3,
    ts := [1, 0, -1, 2, 1], gi := [0, 2, 1, 1, 0, 2, 2, 0, -1, 3], cnt := [0, 0, 0, 0, 0, 0, 0] }

/-- `BINS == 0`, groups, one group per slot (the bucket is the slot; any group `≥ 0` is valid),
3 adapters (`NUM_BUCKETS = 4`): 3 tokens of 3 pairs, `BLOCK = 8` (2 programs). -/
def atomicGps1 : Case :=
  { ntok := 3, w := 3, gps := 1, ml := 3, hg := true, eb := 3, hist := false, ek := 0,
    ts := [2, 0, 1], gi := [0, 5, -1, 0, 0, 0, 7, -2, 1], cnt := [0, 0, 0, 0] }

/-- `tl.histogram` path, no groups, 1 adapter (`NUM_BUCKETS = 2`, `BINS = 4`), `BLOCK = 2` (3
programs); the last counter starts at `2^31 - 1`, so the sum wraps. -/
def histWrap : Case :=
  { ntok := 5, w := 1, gps := 1, ml := 1, hg := false, eb := 1, hist := true, ek := 2,
    ts := [0, 0, 7, 0, -3], gi := [], cnt := [1, 2147483647] }

def verdictAtomicNoGroups : Verdict := Verdict.ofLaunch atomicNoGroups.agree atomicNoGroups.run
def verdictHistGroups : Verdict := Verdict.ofLaunch histGroups.agree histGroups.run
def verdictAtomicGps1 : Verdict := Verdict.ofLaunch atomicGps1.agree atomicGps1.run
def verdictHistWrap : Verdict := Verdict.ofLaunch histWrap.agree histWrap.run

#eval IO.println (verdictAtomicNoGroups.line "route_histogram_i32/atomic_nogroups")
#eval IO.println (verdictHistGroups.line "route_histogram_i32/hist_groups")
#eval IO.println (verdictAtomicGps1.line "route_histogram_i32/atomic_groups_gps1")
#eval IO.println (verdictHistWrap.line "route_histogram_i32/hist_wrap")
#guard verdictAtomicNoGroups == .certified
#guard verdictHistGroups == .certified
#guard verdictAtomicGps1 == .certified
#guard verdictHistWrap == .certified

-- The expectations, written out: several pairs (from several programs) per counter.
#guard (List.range 4).map atomicNoGroups.expect == [8, 2, 102, 5]
#guard (List.range 7).map histGroups.expect == [0, 2, 0, 1, 0, 1, 6]
#guard (List.range 4).map atomicGps1.expect == [3, 2, 2, 2]
#guard (List.range 2).map histWrap.expect == [4, -2147483647]
-- The launches have several programs.
#guard [atomicNoGroups, histGroups, atomicGps1, histWrap].map (fun c => (launch c.inst).size) ==
  [3, 3, 2, 3]

-- The instances are covered: `Inst.ok` (from `mk_ok`; it does not mention the contents). `Pre`
-- covers every `int32` content.
theorem atomicNoGroups_ok : atomicNoGroups.inst.ok := by
  have h := mk_ok (ntok := 10) (w := 1) (gps := 1) (ml := 3) (hg := false) (eb := 2) (hist := false)
    (ek := 0) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
  cases h; constructor <;> assumption
theorem histGroups_ok : histGroups.inst.ok := by
  have h := mk_ok (ntok := 5) (w := 2) (gps := 3) (ml := 2) (hg := true) (eb := 2) (hist := true)
    (ek := 3) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
  cases h; constructor <;> assumption
theorem atomicGps1_ok : atomicGps1.inst.ok := by
  have h := mk_ok (ntok := 3) (w := 3) (gps := 1) (ml := 3) (hg := true) (eb := 3) (hist := false)
    (ek := 0) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
  cases h; constructor <;> assumption
theorem histWrap_ok : histWrap.inst.ok := by
  have h := mk_ok (ntok := 5) (w := 1) (gps := 1) (ml := 1) (hg := false) (eb := 1) (hist := true)
    (ek := 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)
  cases h; constructor <;> assumption
-- The contents have the instance's lengths.
#guard [atomicNoGroups, histGroups, atomicGps1, histWrap].all fun c =>
  c.ts.length == c.ntok && (!c.hg || c.gi.length == c.n) && c.cnt.length == c.nb

-- A wrong expectation is caught: `agree` is not vacuous, in both halves.
#guard (Verdict.ofLaunch (histGroups.agreeWith (fun b => histGroups.expect b + if b = 6 then 1 else 0)
  sentinel) histGroups.run).isDiscrepancy
#guard (Verdict.ofLaunch (atomicNoGroups.agreeWith (fun b => if b = 0 then 3 else atomicNoGroups.expect b)
  sentinel) atomicNoGroups.run).isDiscrepancy
#guard (Verdict.ofLaunch (atomicGps1.agreeWith atomicGps1.expect (sentinel + 1)) atomicGps1.run).isDiscrepancy
