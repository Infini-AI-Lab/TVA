import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `rejection_greedy_i32` run by `execLaunch` on small instances, an
all-greedy batch (`is_greedy = None`) and two mixed batches (an `is_greedy` mask). Not a test
against the device (phase 4): it checks that `exec` runs the model end to end, and that the final
memory is what `Post` says, both halves: each output element (the target's `int32`-truncated
argmax up to and including the first mismatch, the bonus token after a full match, the host's `-1`
elsewhere and in every row of a request that is not greedy), and every other byte of the memory
unchanged (the inputs, and sentinels past the output's `B * (S + 1)` elements). The expectation is
computed here from the case's own data (`Case.expect`), not from the claim's `expect`. -/

open TVA TVA.Validation Claims.RejectionGreedy

/-- A platform that relates nothing: `rejection_greedy_i32` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits (the kernel has no masked load). -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- The host's initial value of every output element. -/
def initOut : Int := -1

/-- A sentinel in the elements just past the output: they must keep it. -/
def sentinel : Int := -7
def pastEnd : Nat := 3

/-- One case: `max_spec_len`, each request's `(draft, target)` pairs (the target an `int64`),
the bonus tokens, and the mask's bytes (`none`: `is_greedy = None`). -/
structure Case where
  S : Nat
  reqs : List (List (Int × Int))
  bonus : List Int
  mask : Option (List Nat)

def Case.B (c : Case) : Nat := c.reqs.length
def Case.N (c : Case) : Nat := (c.reqs.map List.length).sum
def Case.inst (c : Case) : Inst := Claims.RejectionGreedy.mk c.B c.S c.N c.mask.isSome
/-- `cu_num_draft_tokens`: the inclusive running sums of the draft counts. -/
def Case.cu (c : Case) : List Nat := (List.range c.B).map fun r => ((c.reqs.take (r + 1)).map List.length).sum
def Case.flat (c : Case) : List (Int × Int) := c.reqs.flatten

def write32 (m : CMem) (b : Buf .i32) (xs : List Int) : CMem :=
  CMem.ofElems m .i32 ((List.range xs.length).map fun j : Nat => ((b.ptr j).addr, (IntV.ofInt (xs.getD j 0) : I32)))

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := write32 ∅ d.cu (c.cu.map (↑))
  let m := write32 m d.draft (c.flat.map (·.1))
  let m := CMem.ofElems m .i64 ((List.range c.N).map fun j : Nat =>
    ((d.target.ptr j).addr, (IntV.ofInt (c.flat.getD j (0, 0)).2 : I64)))
  let m := write32 m d.bonus c.bonus
  -- the mask's raw bytes (any non-zero byte is `true`)
  let m := match d.greedy, c.mask with
    | some g, some bs => CMem.ofElems m .u8 ((List.range bs.length).map fun j : Nat =>
        ((g.ptr j).addr, DType.ofBits .u8 (bs.getD j 0)))
    | _, _ => m
  let m := write32 m d.out (List.replicate (c.B * (c.S + 1)) initOut)
  CMem.ofElems m .i32 ((List.range pastEnd).map fun k : Nat =>
    ((d.out.ptr (c.B * (c.S + 1) + k)).addr, (IntV.ofInt sentinel : I32)))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) Oracle.empty fill0 1000 (model c.inst) (launch c.inst) c.init

/-- The low 32 bits of an `int64`, read as a signed `int32`. -/
def trunc32 (t : Int) : Int := (t + 2 ^ 31) % 2 ^ 32 - 2 ^ 31

/-- Request `r`'s row, computed from the case: for a greedy request, walk the drafts writing the
truncated target at each position, stop after the first mismatch, append the bonus after a full
match; `-1` everywhere else. -/
def Case.row (c : Case) (r : Nat) : List Int :=
  let greedy := match c.mask with | none => true | some bs => bs.getD r 0 != 0
  let pairs := c.reqs.getD r []
  let init := List.replicate (c.S + 1) initOut
  if !greedy then init
  else
    let (row, rejected) := (List.range pairs.length).foldl (fun (row, rej) p =>
      if rej then (row, rej)
      else
        let (dr, tg) := pairs.getD p (0, 0)
        (row.set p (trunc32 tg), dr != trunc32 tg)) (init, false)
    if rejected then row else row.set pairs.length (c.bonus.getD r 0)

def Case.expect (c : Case) : List Int := ((List.range c.B).map c.row).flatten

/-- The final memory against the case `e`'s expectation (`e = c` for the test): every output
element, then every other byte of the initial memory unchanged and no byte added. -/
def Case.agreeE (c e : Case) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let total := c.B * (c.S + 1)
  let badOut := (List.range total).filter fun j : Nat =>
    (m.read .i32 (d.out.ptr j).addr).map (·.toInt) != some (e.expect.getD j 0)
  let lo := d.out.base
  let hi := d.out.base + (total * 4 : Nat)
  let changed := c.init.toList.filter fun (a, b) => !(lo ≤ a && a < hi) && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"output elements {badOut} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the output changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c

/-- All greedy (`is_greedy = None`), `S = 3`: request 0 matches every draft (the bonus at column
3), with `int64` targets whose truncation matters (`2^32 + 4` truncates to `4`, `-2^33 - 3` to
`-3`); request 1 mismatches at position 1 (columns 0 and 1 written); request 2 has no draft (the
bonus at column 0); request 3 mismatches at position 0 (its second draft, which would match, is
not reached). -/
def allGreedy : Case :=
  { S := 3,
    reqs := [[(4, 2 ^ 32 + 4), (-3, -2 ^ 33 - 3), (7, 7)], [(1, 1), (2, 9), (5, 5)], [],
      [(8, 6), (9, 9)]],
    bonus := [100, 101, 102, 103], mask := none }

/-- A mixed batch, `S = 2`: mask bytes `[1, 0, 2, 0, 1]` (byte `2` reads as `true`); the
requests that are not greedy have drafts that would match (their rows must stay `-1`); the
last greedy one matches both drafts (the bonus at column `S`). -/
def mixed : Case :=
  { S := 2,
    reqs := [[(10, 10), (11, 12)], [(20, 20)], [(30, 30)], [], [(40, 40), (41, 41)]],
    bonus := [200, 201, 202, 203, 204], mask := some [1, 0, 2, 0, 1] }

/-- A mixed batch where every request is greedy by its mask and one has a large negative target
truncating to a value the draft does not match (`-2^31 - 1` truncates to `2^31 - 1`). -/
def maskedAllOn : Case :=
  { S := 1, reqs := [[(-1, -2 ^ 31 - 1)], [(2 ^ 31 - 1, -2 ^ 31 - 1)]], bonus := [300, 301],
    mask := some [1, 255] }

def verdictAll : Verdict := Verdict.ofLaunch allGreedy.agree allGreedy.run
def verdictMixed : Verdict := Verdict.ofLaunch mixed.agree mixed.run
def verdictOn : Verdict := Verdict.ofLaunch maskedAllOn.agree maskedAllOn.run

#eval IO.println (verdictAll.line "rejection_greedy_i32/all_greedy")
#eval IO.println (verdictMixed.line "rejection_greedy_i32/mixed")
#eval IO.println (verdictOn.line "rejection_greedy_i32/mask_all_on")
#guard verdictAll == .certified
#guard verdictMixed == .certified
#guard verdictOn == .certified

-- The instances are covered: `Inst.ok` holds by `mk_ok` (`1 ≤ B ≤ 8192`, `B * (S + 1) ≤ 2^31`),
-- and `Pre`'s decidable parts: `cu` nondecreasing from a nonnegative entry, its last entry the
-- number of drafts (`= draft.len = target.len`, `≤ 2^22`), every count at most `S`.
example : allGreedy.inst.ok := mk_ok _ _ _ _ (by decide) (by decide) (by decide) (by decide)
example : mixed.inst.ok := mk_ok _ _ _ _ (by decide) (by decide) (by decide) (by decide)
example : maskedAllOn.inst.ok := mk_ok _ _ _ _ (by decide) (by decide) (by decide) (by decide)
def Case.preOk (c : Case) : Bool :=
  let d := c.inst
  -- `Buf.Valid`'s alignment: `mk` places `target` at `4 * (B * (S + 2) + N)` bytes (plus
  -- multiples of `8`) past an aligned address, so it is `8`-aligned only when that count is even
  (d.out.base % 4 == 0 && d.cu.base % 4 == 0 && d.draft.base % 4 == 0 && d.target.base % 8 == 0 &&
    d.bonus.base % 4 == 0) &&
  c.cu.getLast? == some c.N && c.N ≤ 2 ^ 22 && c.reqs.all (·.length ≤ c.S) &&
    c.bonus.length == c.B && c.mask.all (·.length == c.B) && 1 ≤ c.B
#guard allGreedy.preOk && mixed.preOk && maskedAllOn.preOk
#guard allGreedy.cu == [3, 6, 6, 8] && mixed.cu == [2, 3, 4, 4, 6]

-- The expectations exercise each branch: a full match with bonus, mismatches, an empty request,
-- rows left at `-1` for requests that are not greedy, truncation of `int64` targets.
#guard allGreedy.expect ==
  [4, -3, 7, 100,  1, 9, -1, -1,  102, -1, -1, -1,  6, -1, -1, -1]
#guard mixed.expect == [10, 12, -1,  -1, -1, -1,  30, 202, -1,  -1, -1, -1,  40, 41, 204]
#guard maskedAllOn.expect == [2 ^ 31 - 1, -1,  2 ^ 31 - 1, 301]

-- Not vacuous: a wrong expectation is caught. With the second draft of request 1 matching
-- (expectation: written on to the bonus), against the real run; with the mask ignored.
def wrongAll : Case := { allGreedy with reqs := [[(4, 2 ^ 32 + 4), (-3, -2 ^ 33 - 3), (7, 7)],
  [(1, 1), (9, 9), (5, 5)], [], [(8, 6), (9, 9)]] }
def wrongMixed : Case := { mixed with mask := none }
#guard (Verdict.ofLaunch (allGreedy.agreeE wrongAll) allGreedy.run).isDiscrepancy
#guard (Verdict.ofLaunch (mixed.agreeE wrongMixed) mixed.run).isDiscrepancy
