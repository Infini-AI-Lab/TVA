import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `ssd_state_passing_f32` run by `execLaunch` on four small covered
instances (the claim's own `mk`), both models (`HAS_INITSTATES` true and false), packed batches
whose requests start inside a chunk (offset `> 0`: line 215's subtraction), at a chunk's start,
and run across a chunk without restarting; one instance with `dim = 1025` (two programs along
`dim`, the second with `1023` masked lanes). Not a test against the device (phase 4): it checks
that `exec` runs the model end to end, and that the final memory is what `Post` says, both halves:
`out[0, 0]` (the initial state of request `0`, bit for bit, or `+0`), every `out[0, c]` for
`1 ≤ c < nchunks` and `final_states` against the recurrence `sRef`, and every other byte
unchanged (the inputs, the metadata, sentinels on both sides of `out` and `final_states`), with no
byte added.

The recurrence is computed here from the case's own request lengths and values (rationals): the
request running at the end of chunk `c`, whether chunk `c` starts it and at which offset, its
decay `dA[c, chunk − 1]` (less `dA[c, off − 1]` when it starts at `off > 0`), and
`s(c+1) = exp(d)·x + states[c]`, `x` the request's initial state (or `0`) on a chunk that starts it,
`s(c)` otherwise (`Post`'s `sRef`). `chunk_offsets` is built here by the host's loop (and checked
equal to `Pre`'s `hostOff`), `seq_idx` from the lengths.

**The `tl.exp` values are assumed, not proved.** `tl.exp(dA_cs)` (line 223) is a platform
operation, run once per program and chunk. The oracle answers it with the correctly rounded
`float32` of `exp(x)` (`x ∈ [-1, 0]`; computed from two partial sums of the alternating Taylor
series that bracket `exp(x)` and round to the same `float32`; it answers nothing otherwise). Two
kinds of case:
- **exact** (`initExact`, `noinitExact`): every `dA_cumsum` entry is `0`, so every exponent is `0`
  and the answer is `1` (the value `tl.exp(0)` gives on the device; `Device.ExpF32` would allow
  `1 ± 2^-22`, so this is an assumption on top of the device facts). Every value is then a small
  multiple of `1/2`, the `tl.mul_add`s are exact, and each output must equal `sRef` exactly
  (distance `0`, which `Post` allows);
- **decay** (`initDecay`, `noinitDecay`): `dA_cumsum` entries in `[-1, 0]`, multiples of `1/16`, so
  line 215's subtraction is exact and the exponents are in `[-1, 0]`; the answers are the correctly
  rounded `exp` (within `Device.ExpF32`'s bound: one value a correct device may give). The outputs
  are checked against `Post`'s own tolerance: `|v − sRef c| ≤ tol c`, with `sRef c` known only to
  an interval (the Taylor brackets of each `exp`), and `tol c` replaced by a lower bound (`errE`'s
  `e^(d+δ)` by `e^d`, each `exp` by its lower bracket; every coefficient of `tol` is nonnegative):
  the check `max(|v − lo|, |v − hi|) ≤ tolLo` implies `Post`'s clause.

The inputs are covered by `Vals` (the hypothesis of `Post`'s numeric clauses): every entry finite,
`dA_cumsum ≤ 0` (so `dRef ≤ 0 ≤ 64`), and inputs at most `3` in magnitude over at most four chunks
with decays at most `1`, so every update is far below `2^126`. The verdicts are `conditional n`, `n` the number of `tl.exp` answers. -/

open TVA TVA.Validation Claims.SsdStatePassing

/-- A platform that relates nothing: the `exp` values come from the oracle. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## `exp` on `[-1, 0]`, bracketed -/

/-- The partial sums `S_n`, `S_(n+1)` of `Σ q^i / i!`; for `q ∈ [-1, 0]` the terms alternate and
shrink, so `exp q` lies between them. Returned as `(lo, hi)`. -/
def expBr (q : Rat) (n : Nat := 24) : Rat × Rat :=
  let (s, t) := (List.range n).foldl (fun (s, t) i => (s + t * q / (i + 1 : Nat), t * q / (i + 1 : Nat))) ((1 : Rat), (1 : Rat))
  let s' := s + t * q / (n + 1 : Nat)
  (min s s', max s s')

/-- The correctly rounded `exp q` in format `f`, when `q ∈ [-1, 0]` and both brackets round to the
same value. -/
def expRn {f : TVA.Format} (q : Rat) : Option (FloatV f) :=
  if -1 ≤ q ∧ q ≤ 0 then
    let (lo, hi) := expBr q
    let a : FloatV f := FloatV.ofRat lo
    if a.pat == (FloatV.ofRat hi : FloatV f).pat then some a else none
  else none

/-- The oracle: the correctly rounded `exp` at every `tl.exp` request (every lane finite and in
`[-1, 0]`); nothing else. -/
def expPlat : (q : PlatQ) → Option q.Out
  | @PlatQ.math .exp f _ x =>
    if x.toList.all (fun v => v.isFinite && (expRn (f := f) v.toRat).isSome) then
      some (x.map fun v => (expRn v.toRat).getD v)
    else none
  | _ => none

def O : Oracle := fun _ _ q => match q with
  | .plat q => expPlat q
  | _ => none

/-! ## The cases -/

def f32v (q : Rat) : F32 := FloatV.ofRat q

/-- A sentinel in every element of `out` and `final_states` before the launch, and in `pad`
elements on each side of each. -/
def sentinel : Rat := -77
def pad : Nat := 4

/-- One case: the host's arguments of `mk`, whether `dA_cumsum` is zero (`exact`), and the inputs:
`states[c, h, m]`, `initial_states[r, h, m]`, `dA_cumsum[h, c, p]`. -/
structure Case where
  hasInit : Bool
  lens : List Nat
  H : Nat
  D : Nat
  C : Nat
  exact : Bool

def Case.inst (c : Case) : Inst := Claims.SsdStatePassing.mk c.hasInit c.lens c.H c.D c.C
def Case.L (c : Case) : Nat := c.lens.sum
def Case.NC (c : Case) : Nat := (c.L + c.C - 1) / c.C
/-- The first token of request `r`. -/
def Case.start (c : Case) (r : Nat) : Nat := (c.lens.take r).sum
/-- The request of token `t`. -/
def Case.req (c : Case) (t : Nat) : Nat := ((List.range c.lens.length).filter fun r => c.start r ≤ t).length - 1

def stv (ch h m : Nat) : Rat := (((3 * ch + 5 * h + 7 * m + 1) % 9 : Nat) - (4 : Int) : Int) / 2
def inv (r h m : Nat) : Rat := (((2 * r + 3 * h + 5 * m + 2) % 7 : Nat) - (3 : Int) : Int)
def Case.dav (c : Case) (h ch p : Nat) : Rat :=
  if c.exact then 0 else -(((p + 1) * (h + ch % 3 + 1) : Nat) : Rat) / 16

/-- `chunk_offsets` by the host's loop (lines 345–366): `N = nchunks + #{later starts not on a chunk
boundary}` zeros; for each later request start `s`, `p += (s % C > 0)`, `offs[s / C + p] = s % C`. -/
def Case.offs (c : Case) : List Nat :=
  let later := (List.range c.lens.length).drop 1
  let n := c.NC + (later.filter fun r => c.start r % c.C > 0).length
  let (_, o) := later.foldl (fun (p, o) r =>
    let s := c.start r
    let p := p + (if s % c.C > 0 then 1 else 0)
    (p, o.set (s / c.C + p) (s % c.C))) (0, List.replicate n 0)
  if c.hasInit then o else []

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .f32 ((List.range c.NC).flatMap fun ch => (List.range c.H).flatMap fun h =>
    (List.range c.D).map fun x => ((d.states.ptr (stIdx d ch h x : Nat)).addr, f32v (stv ch h x)))
  let m := CMem.ofElems m .f32 ((List.range c.H).flatMap fun h => (List.range c.NC).flatMap fun ch =>
    (List.range c.C).map fun p => ((d.dA.ptr (dAIdx d h ch p : Nat)).addr, f32v (c.dav h ch p)))
  let m := if c.hasInit then CMem.ofElems m .f32 ((List.range c.lens.length).flatMap fun r =>
      (List.range c.H).flatMap fun h => (List.range c.D).map fun x =>
        ((d.initst.ptr (inIdx d r h x : Nat)).addr, f32v (inv r h x))) else m
  let m := CMem.ofElems m .i32 ((List.range c.L).map fun t : Nat =>
    ((d.seqIdx.ptr t).addr, (IntV.ofNat (c.req t) : I32)))
  let m := CMem.ofElems m .i32 ((List.range c.offs.length).map fun k : Nat =>
    ((d.chunkOffs.ptr k).addr, (IntV.ofNat (c.offs.getD k 0) : I32)))
  let m := CMem.ofElems m .f32 ((List.range (c.NC * c.H * c.D + 2 * pad)).map fun e : Nat =>
    ((d.out.ptr ((e : Int) - pad)).addr, f32v sentinel))
  CMem.ofElems m .f32 ((List.range (c.H * c.D + 2 * pad)).map fun e : Nat =>
    ((d.finalst.ptr ((e : Int) - pad)).addr, f32v sentinel))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) O fill0 10000 (model c.inst) (launch c.inst) c.init

/-! ## The recurrence, in intervals -/

/-- A rational interval `[lo, hi]`. -/
abbrev Iv := Rat × Rat
def Iv.pt (q : Rat) : Iv := (q, q)
def Iv.mul (a b : Iv) : Iv :=
  let ps := [a.1 * b.1, a.1 * b.2, a.2 * b.1, a.2 * b.2]
  (ps.foldl min ps.head!, ps.foldl max ps.head!)
def Iv.add (a : Iv) (q : Rat) : Iv := (a.1 + q, a.2 + q)
/-- The least `|x|` over the interval. -/
def Iv.absLo (a : Iv) : Rat := if 0 ≤ a.1 then a.1 else if a.2 ≤ 0 then -a.2 else 0

/-- Chunk `c`'s request at its end, whether `c` starts it, and its decay exponent. -/
def Case.chunk (c : Case) (h ch : Nat) : Nat × Bool × Rat :=
  let e := min ((ch + 1) * c.C) c.L - 1
  let r := c.req e
  let s := c.start r
  let reset := decide (ch * c.C ≤ s)
  let off := s - ch * c.C
  let dd := c.dav h ch (c.C - 1) - (if reset && decide (0 < off) then c.dav h ch (off - 1) else 0)
  (r, reset, dd)

/-- The state request `r` starts from. -/
def Case.startV (c : Case) (r h m : Nat) : Rat := if c.hasInit then inv r h m else 0

/-- `2^-k`. -/
def p2 (k : Nat) : Rat := 1 / ((2 ^ k : Nat) : Rat)

/-- The state entering chunk `ch` (an interval holding `sRef ch`) and a lower bound of `tol ch`. -/
def Case.spec (c : Case) (h m : Nat) : Nat → Iv × Rat
  | 0 => (Iv.pt (c.startV 0 h m), 0)
  | ch + 1 =>
    let (s, t) := c.spec h m ch
    let (r, reset, dd) := c.chunk h ch
    let x : Iv := if reset then Iv.pt (c.startV r h m) else s
    let tc : Rat := if reset then 0 else t
    let e := expBr dd
    let new := stv ch h m
    let rel := p2 22 + p2 23 * |dd|
    let errE := rel * e.1 + p2 140
    let eHat := e.1 + errE
    let rho := p2 23 + p2 47
    ((Iv.mul e x).add new,
      eHat * tc + errE * x.absLo + rho * (eHat * (x.absLo + tc) + |new|) + 3 * p2 150)

/-- `v` is what `Post` allows for the state entering chunk `ch`: exact cases, `sRef` itself (the
interval is a point); decay cases, finite and within the tolerance's lower bound of every point of
the interval. -/
def Case.okVal (c : Case) (h m ch : Nat) (v : F32) : Bool :=
  let (s, t) := c.spec h m ch
  v.isFinite && (if c.exact then s.1 == s.2 && v.toRat == s.1
    else max |v.toRat - s.1| |v.toRat - s.2| ≤ t)

/-- The final memory against `Post` (with `ok` deciding the states entering chunks `1 …
nchunks`): `out[0, 0]`, `out[0, c]`, `final_states`; every other byte unchanged; no byte added. -/
def Case.agree (c : Case) (ok : Nat → Nat → Nat → F32 → Bool) (cf : CConfig (launch c.inst).size) :
    Option String :=
  let d := c.inst
  let mm := CMem.toMem cf.mem
  let hm := (List.range c.H).flatMap fun h => (List.range c.D).map fun x => (h, x)
  let rd (b : Buf .f32) (i : Nat) : Option F32 := mm.read .f32 (b.ptr i).addr
  let bad0 := hm.filter fun (h, x) =>
    (rd d.out (stIdx d 0 h x)).map DType.f32.toBits !=
      some (if c.hasInit then DType.f32.toBits (f32v (inv 0 h x)) else 0)
  let badC := ((List.range c.NC).drop 1).flatMap fun ch => (hm.filter fun (h, x) =>
    !((rd d.out (stIdx d ch h x)).map (ok h x ch)).getD false).map fun (h, x) => (ch, h, x)
  let badF := hm.filter fun (h, x) => !((rd d.finalst (finIdx d h x)).map (ok h x c.NC)).getD false
  let inOut (a : Int) : Bool :=
    (d.out.base ≤ a && a < d.out.base + 4 * (c.NC * c.H * c.D : Nat)) ||
    (d.finalst.base ≤ a && a < d.finalst.base + 4 * (c.H * c.D : Nat))
  let badElse := c.init.toList.filter fun (a, b) => !inOut a && cf.mem[a]? != some b
  if !bad0.isEmpty then some s!"out[0, 0] elements {bad0.take 8} differ from Post"
  else if !badC.isEmpty then some s!"out[0, c] elements {badC.take 8} differ from Post"
  else if !badF.isEmpty then some s!"final_states elements {badF.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside out and final_states changed"
  else if cf.mem.size != c.init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch (c.agree c.okVal) c.run

/-- With initial states, `dA = 0`: requests of `3, 6, 2` tokens, chunks of `4` (chunk `0` ends in
request `1`, which starts at offset `3`; chunk `1` continues it; chunk `2` ends in request `2`,
starting at offset `1`), `nheads = 2`, `dim = 3`. -/
def initExact : Case := { hasInit := true, lens := [3, 6, 2], H := 2, D := 3, C := 4, exact := true }
theorem initExact_ok : initExact.inst.ok :=
  mk_ok _ _ _ _ _ 2 rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- Without initial states, `dA = 0`: requests `3, 6, 2`, chunks of `4`, `nheads = 1`, `dim = 1025`
(two programs along `dim`). -/
def noinitExact : Case := { hasInit := false, lens := [3, 6, 2], H := 1, D := 1025, C := 4, exact := true }
theorem noinitExact_ok : noinitExact.inst.ok :=
  mk_ok _ _ _ _ _ 2 rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- With initial states, decaying: requests `3, 5, 4, 2`, chunks of `4` (offsets `3`, then two
requests starting on a chunk's first token), `nheads = 2`, `dim = 2`. -/
def initDecay : Case := { hasInit := true, lens := [3, 5, 4, 2], H := 2, D := 2, C := 4, exact := false }
theorem initDecay_ok : initDecay.inst.ok :=
  mk_ok _ _ _ _ _ 2 rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

/-- Without initial states, decaying: requests `5, 7`, chunks of `4` (chunk `0` inside request
`0`, chunk `1` starting request `1` at offset `1`, chunk `2` continuing it), `nheads = 2`,
`dim = 3`. -/
def noinitDecay : Case := { hasInit := false, lens := [5, 7], H := 2, D := 3, C := 4, exact := false }
theorem noinitDecay_ok : noinitDecay.inst.ok :=
  mk_ok _ _ _ _ _ 2 rfl (by decide) (by decide) (by decide) (by decide) (by decide) (by decide)

def cases : List Case := [initExact, noinitExact, initDecay, noinitDecay]

/-! ## Coverage (`Pre`, `Vals`) and the expectations -/

-- `Pre`: the seven buffers are aligned and inside `[0, 2^64)`, every element present in `init`
-- (`initial_states` and `chunk_offsets` empty without initial states); `MetaOk`: `seq_idx` is the
-- claim's `reqOf`, and `chunk_offsets` (built here by the host's loop) is the claim's `hostOff`
-- and as long as `nmeta`.
#guard cases.all fun c =>
  let d := c.inst
  let m := CMem.toMem c.init
  let okB {dt : DType} (b : Buf dt) : Bool :=
    decide (0 ≤ b.base) && decide (b.base + b.len * dt.bytes ≤ 2 ^ 64) && decide (b.base % dt.bytes = 0) &&
    (List.range b.len).all fun e : Nat => (m.read dt (b.ptr e).addr).isSome
  okB d.states && okB d.dA && okB d.initst && okB d.seqIdx && okB d.chunkOffs && okB d.out && okB d.finalst &&
    (List.range c.L).all (fun t => c.req t == reqOf d t &&
      (m.read .i32 (d.seqIdx.ptr t).addr).map (·.toInt) == some (reqOf d t : Int)) &&
    c.offs.length == d.nmeta &&
    (List.range d.nmeta).all fun k => c.offs.getD k 0 == hostOff d k &&
      (m.read .i32 (d.chunkOffs.ptr k).addr).map (·.toInt) == some (hostOff d k : Int)

-- The sizes are the instance's, and `chunk_offsets` is nontrivial with initial states.
#guard cases.all fun c => c.inst.nchunks == c.NC && c.inst.seqlen == c.L
#guard initExact.offs == [0, 3, 0, 0, 1] && initDecay.offs == [0, 3, 0, 0, 0]

-- `Vals`, and exact encodings: every input value is a `float32` read back exactly; `dA ≤ 0`
-- (in `[-1, 0]`); every exponent in `[-1, 0]`; inputs at most `3` in magnitude.
#guard cases.all fun c =>
  ((List.range c.NC).all fun ch => (List.range c.H).all fun h =>
    ((List.range c.D).all fun x => (f32v (stv ch h x)).toRat == stv ch h x && |stv ch h x| ≤ 3) &&
    ((List.range c.C).all fun p => (f32v (c.dav h ch p)).toRat == c.dav h ch p && -1 ≤ c.dav h ch p && c.dav h ch p ≤ 0) &&
    (let dd := (c.chunk h ch).2.2; -1 ≤ dd && dd ≤ 0)) &&
  ((List.range c.lens.length).all fun r => (List.range c.H).all fun h => (List.range c.D).all fun x =>
    (f32v (inv r h x)).toRat == inv r h x && |inv r h x| ≤ 3)

-- The bracket is tight, and `exp(0) = 1` exactly.
#guard expBr 0 == (1, 1) && (let (lo, hi) := expBr (-1); hi - lo < 1 / 10 ^ 20 && lo < 3679 / 10000 && 3678 / 10000 < lo)

-- Every case's expectations exist, are not the sentinel, and are nontrivial: the decay cases have
-- exponents other than `0` (some from line 215's subtraction), and the exact cases carry a state
-- across a chunk (a non-reset chunk) and restart one (a reset chunk).
#guard cases.all fun c => (List.range c.H).all fun h => (List.range c.D).all fun x =>
  ((List.range (c.NC + 1)).all fun ch => let (s, _) := c.spec h x ch; decide (s.2 < sentinel) || decide (sentinel < s.1))
#guard (List.range 3).any (fun ch => (initDecay.chunk 0 ch).2.2 != 0 && (initDecay.chunk 0 ch).2.1) &&
  (initDecay.chunk 1 0).2.2 == initDecay.dav 1 0 3 - initDecay.dav 1 0 2 &&
  (List.range 3).any (fun ch => !(initExact.chunk 0 ch).2.1) && (initExact.chunk 0 2).2.1 &&
  (noinitDecay.chunk 0 1).2.1 && (noinitDecay.chunk 0 1).2.2 != 0 && !(noinitDecay.chunk 0 2).2.1

def verdictInitExact : Verdict := initExact.verdict
def verdictNoinitExact : Verdict := noinitExact.verdict
def verdictInitDecay : Verdict := initDecay.verdict
def verdictNoinitDecay : Verdict := noinitDecay.verdict

#eval IO.println (verdictInitExact.line "ssd_state_passing_f32/init_exact")
#eval IO.println (verdictNoinitExact.line "ssd_state_passing_f32/noinit_exact")
#eval IO.println (verdictInitDecay.line "ssd_state_passing_f32/init_decay")
#eval IO.println (verdictNoinitDecay.line "ssd_state_passing_f32/noinit_decay")
-- Agreement resting on one `tl.exp` answer per program and chunk (`programs = cdiv(dim, 1024) ·
-- nheads`).
#guard let v := verdictInitExact; v == .conditional (2 * 3) && v.agrees && v.assumed > 0
#guard let v := verdictNoinitExact; v == .conditional (2 * 3) && v.agrees && v.assumed > 0
#guard let v := verdictInitDecay; v == .conditional (2 * 4) && v.agrees && v.assumed > 0
#guard let v := verdictNoinitDecay; v == .conditional (2 * 3) && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (the decay of line 215 not subtracted;
-- the state not restarted where a request starts).
#guard
  let r := initDecay.run
  (Verdict.ofLaunch (initDecay.agree fun h m ch v =>
    let c' : Case := { initDecay with lens := [3, 5, 4, 2] }
    let wrong : Nat → Iv × Rat := fun ch =>
      -- the recurrence with `dRef` the chunk's total `dA[c, chunk − 1]` on every chunk
      (List.range ch).foldl (fun (s, _) k =>
        let (r, reset, _) := c'.chunk h k
        let x : Iv := if reset then Iv.pt (c'.startV r h m) else s
        ((Iv.mul (expBr (c'.dav h k 3)) x).add (stv k h m), 0)) (Iv.pt (c'.startV 0 h m), 0)
    let (s, _) := wrong ch
    v.isFinite && max |v.toRat - s.1| |v.toRat - s.2| ≤ 1 / 2 ^ 20) r).isDiscrepancy
#guard
  let r := initExact.run
  (Verdict.ofLaunch (initExact.agree fun h m ch v =>
    -- the state carried across every chunk (never restarted)
    let s := (List.range ch).foldl (fun s k => s + stv k h m) (inv 0 h m)
    v.toRat == s) r).isDiscrepancy
