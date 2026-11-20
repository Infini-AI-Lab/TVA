import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `count_expert_tokens_i32` run by `execLaunch` on three small
instances (`int64` ids without a map; `int32` ids with a map; `int64` ids with a map and a block
larger than `numel`). Not a test against the device (phase 4): it checks that `exec` runs the
model end to end, and that the final memory is what `Post` says: element `e` of
`expert_num_tokens` is the number of positions routed to local expert `e` (counted here from the
inputs), and every other element (the inputs, and two guard elements around the output) is
unchanged. Every program reads every entry of `topk_ids`, so each entry is seen by all
`num_local_experts` programs and counted by at most one. -/

open TVA TVA.Validation Claims.CountExpertTokens

/-- A platform that relates nothing: `count_expert_tokens_i32` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- A sentinel: the output's initial elements (the wrapper's buffer is not zeroed: any initial
value is covered, every element is stored), and the guard elements just outside the output,
which must keep it. -/
def sentinel : Int := -7

/-- An integer element of an `int32`/`int64` dtype (two's complement). -/
def ival (d : DType) (z : Int) : d.Val := d.ofBits (z % (2 ^ (8 * d.bytes) : Int)).toNat

/-- One case: the ids' dtype, the map (if any), the local experts, `BLOCK_SIZE = 2^e`, the ids. -/
structure Case where
  dt : DType
  /-- `expert_map`'s entries (its length is `numGlobal`), or none. -/
  emap : Option (List Int)
  numLocal : Nat
  /-- `numGlobal` without a map (unused by the kernel then). -/
  numGlobal0 : Nat
  e : Nat
  ids : List Int

def Case.numel (c : Case) : Nat := c.ids.length
def Case.numGlobal (c : Case) : Nat := match c.emap with
  | some m => m.length
  | none => c.numGlobal0
def Case.base (c : Case) : Inst :=
  Claims.CountExpertTokens.mk c.dt c.emap.isSome c.numLocal c.numGlobal c.numel c.e
/-- The instance: `mk`'s, with the case's contents (`Pre`'s `Holds`). -/
def Case.inst (c : Case) : Inst :=
  { c.base with
    idsVal := fun j => ival c.base.dtype (c.ids.getD j 0)
    mapVal := fun j => IntV.ofInt ((c.emap.getD []).getD j 0) }

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ d.dtype ((List.range c.numel).map fun j : Nat =>
    ((d.ids.ptr j).addr, ival d.dtype (c.ids.getD j 0)))
  let m := match c.emap with
    | some mp => CMem.ofElems m .i32 ((List.range mp.length).map fun j : Nat =>
        ((d.emap.ptr j).addr, (IntV.ofInt (mp.getD j 0) : I32)))
    | none => m  -- no map: the kernel must not read one (a load of an absent byte is `ub`)
  -- the output, and one guard element on each side of it
  CMem.ofElems m .i32 (((List.range (c.numLocal + 2)).map fun j : Nat =>
    ((d.out.ptr ((j : Int) - 1)).addr, (IntV.ofInt sentinel : I32))))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) Oracle.empty fill0 2000 (model c.inst) (launch c.inst) c.init

/-- The local expert position `j` goes to, from the inputs: the entry itself without a map; with
one, the map's entry for a non-negative id and `-1` for a negative one. -/
def Case.route (c : Case) (j : Nat) : Int :=
  let a := c.ids.getD j 0
  match c.emap with
  | none => a
  | some mp => if 0 ≤ a then mp.getD a.toNat 0 else -1

/-- What `Post` says element `e` of `expert_num_tokens` holds: the number of positions routed to
`e`. -/
def Case.expect (c : Case) (e : Nat) : Int :=
  ((List.range c.numel).filter fun j => c.route j == (e : Int)).length

/-- The final memory against an expectation `exp` for the output and `guard` for the two guard
elements: every output element, the guards, and the inputs unchanged. -/
def Case.agreeWith (c : Case) (exp : Nat → Int) (guard : Int) (cf : CConfig (launch c.inst).size) :
    Option String :=
  let d := c.inst
  let m0 := CMem.toMem c.init
  let m := CMem.toMem cf.mem
  let rd (j : Int) := (m.read .i32 (d.out.ptr j).addr).map (·.toInt)
  let badOut := (List.range c.numLocal).filter fun e : Nat => rd e != some (exp e)
  let guardsOk := rd (-1) == some guard && rd c.numLocal == some guard
  let same {dt : DType} (b : Buf dt) (n : Nat) : Bool :=
    (List.range n).all fun j : Nat =>
      (m.read dt (b.ptr j).addr).map dt.toBits == (m0.read dt (b.ptr j).addr).map dt.toBits
  let inputsSame := same d.ids c.numel && (c.emap.isNone || same d.emap c.numGlobal)
  if !badOut.isEmpty then some s!"expert_num_tokens elements {badOut} differ from Post"
  else if !guardsOk then some "an element outside the output changed"
  else if !inputsSame then some "an input changed"
  else none

def Case.agree (c : Case) := c.agreeWith c.expect sentinel

/-- `int64` ids, no map, 4 local experts, `BLOCK_SIZE = 4` over 13 entries (4 trips, the last
masked): repeated ids (several positions per expert), a negative id, an id `≥ numLocal`, and ids
`2^32 + 1`, `2^32 + 2` whose low 32 bits are local experts (the comparison is in `int64`). -/
def noMap64 : Case :=
  { dt := .i64, emap := none, numLocal := 4, numGlobal0 := 4, e := 2,
    ids := [0, 2, 2, 1, -1, 5, 0, 2, 4294967297, 3, 2, 4294967298, 0] }

/-- `int32` ids, a map of 6 global experts onto 3 local ones (`-1` for a remote expert, `7`
outside the local range), `BLOCK_SIZE = 8` over 11 entries (2 trips): negative ids skip the
gather. -/
def map32 : Case :=
  { dt := .i32, emap := some [2, -1, 0, 7, 2, 1], numLocal := 3, numGlobal0 := 0, e := 3,
    ids := [0, 1, 2, 3, 4, 5, -1, 4, 0, 2, -5] }

/-- `int64` ids with a map, `BLOCK_SIZE = 16` over 5 entries (one trip, mostly masked), 2 local
experts. -/
def map64 : Case :=
  { dt := .i64, emap := some [1, 1, -1, 0], numLocal := 2, numGlobal0 := 0, e := 4,
    ids := [1, 0, -3, 3, 2] }

def verdictNoMap64 : Verdict := Verdict.ofLaunch noMap64.agree noMap64.run
def verdictMap32 : Verdict := Verdict.ofLaunch map32.agree map32.run
def verdictMap64 : Verdict := Verdict.ofLaunch map64.agree map64.run

#eval IO.println (verdictNoMap64.line "count_expert_tokens_i32/nomap_i64")
#eval IO.println (verdictMap32.line "count_expert_tokens_i32/map_i32")
#eval IO.println (verdictMap64.line "count_expert_tokens_i32/map_i64")
#guard verdictNoMap64 == .certified
#guard verdictMap32 == .certified
#guard verdictMap64 == .certified

-- The expectations, written out: several positions per expert.
#guard (List.range 4).map noMap64.expect == [3, 1, 4, 1]
#guard (List.range 3).map map32.expect == [2, 1, 4]
#guard (List.range 2).map map64.expect == [1, 2]

-- The instances are covered: `Inst.ok` (from `mk_ok`; it does not mention the contents) …
theorem noMap64_ok : noMap64.inst.ok := by
  have h := mk_ok (dt := .i64) (hm := false) (nl := 4) (ng := 4) (numel := 13) (e := 2)
    (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
  cases h; constructor <;> assumption
theorem map32_ok : map32.inst.ok := by
  have h := mk_ok (dt := .i32) (hm := true) (nl := 3) (ng := 6) (numel := 11) (e := 3)
    (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
  cases h; constructor <;> assumption
theorem map64_ok : map64.inst.ok := by
  have h := mk_ok (dt := .i64) (hm := true) (nl := 2) (ng := 4) (numel := 5) (e := 4)
    (Or.inr rfl) (by norm_num) (by norm_num) (by norm_num) (by decide) (by norm_num)
  cases h; constructor <;> assumption
-- … and `Pre`'s condition on the contents: with a map, every id is below `numGlobal`.
#guard [map32, map64].all fun c => c.ids.all (· < (c.numGlobal : Int))

-- A wrong expectation is caught: `agree` is not vacuous, in both halves.
#guard (Verdict.ofLaunch (map32.agreeWith (fun e => map32.expect e + if e = 1 then 1 else 0) sentinel)
  map32.run).isDiscrepancy
#guard (Verdict.ofLaunch (noMap64.agreeWith noMap64.expect (sentinel + 1)) noMap64.run).isDiscrepancy
