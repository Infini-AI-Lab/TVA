import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `chunk_cumsum_f32` run by `execLaunch` on two small covered
instances: a fixed-length batch (`B = 2` rows of `T = 6` tokens, `H = 2` heads, chunks of
`BT = 4`, the last chunk of each row ragged, forward, scale `2`), and a variable-length one
(`int64` tables, three sequences of lengths `3, 0, 6` packed in `T = 10` tokens, the last token in
no sequence, `chunk_indices` rows in a permuted order, reverse, scale `1/2`). Not a test against the
device (phase 4): it checks that `exec` runs the model end to end, and that the final memory is
what `Post` says on these inputs: every written element `o[p, h]` is exactly `σ · Σ s[q, h]` over
`p`'s chunk-local prefix (forward) or suffix (reverse), chunks counted from the sequence's start
(computed here from `cu_seqlens` or the rows and `s` alone); every other byte (the inputs, the
tables, the output element of the uncovered token, sentinels on both sides of `o`) unchanged, and
no byte added.

The gates are small integers and the scales powers of two, so every `float32` partial sum and the
scaled result are exact whatever the scan's tree: `Post` allows the exact value (distance `0`).
`chunk_cumsum_f32` uses no platform operation, so the expected verdict is `certified`.

**`exec` cannot run this kernel yet.** `tl.cumsum` is `tl.associative_scan`, whose site is
`Site.other` (`TVA/Triton/Scan.lean:70`): the executor has no solver for it (`COVERAGE.md`: "no
solver"; `PICKS.md`: `other` (scans, …) unsupported). Each run therefore stops in program `0` at the
scan, after the program's loads, with the verdict `unsupported`; the guards below record that
(and that nothing before the scan is a predicted undefined behaviour or a discrepancy). When a scan
solver exists, the guards on the verdicts become `== .certified`, and the non-vacuity guard at the
end becomes meaningful. -/

open TVA TVA.Validation

namespace CumsumTest

/-- A platform that relates nothing: `chunk_cumsum_f32` uses no platform operation. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- A sentinel in every output element before the launch, and in the elements just outside `o`. -/
def sentinel : Rat := -1001 / 2

/-- Sentinel elements on each side of `o`. -/
def pad : Nat := 4

/-- An integer element of a table dtype (`int64` or `int32`). -/
def ival (d : DType) (z : Int) : d.Val := d.ofBits (z % (2 ^ (8 * d.bytes) : Int)).toNat

/-- One case: the claim's `mk` arguments, the scale's value (written independently of its
pattern, checked below), the gates as integers, and on variable length the sequences' lengths and
the `(sequence, chunk)` rows of `chunk_indices` in their order. -/
structure Case where
  vl : Bool
  B : Nat
  T : Nat
  H : Nat
  e : Nat
  rev : Bool
  hs : Bool
  sbits : Nat
  sigma : Rat
  dt : DType
  /-- Gate `s[j]` (flat element `j = p·H + h`). -/
  x : Nat → Int
  lens : List Nat
  rows : List (Nat × Nat)

def Case.BT (c : Case) : Nat := 2 ^ c.e
def Case.N (c : Case) : Nat := c.lens.length
def Case.NT (c : Case) : Nat := c.rows.length
/-- `cu_seqlens[i]`, the running sums of the lengths. -/
def Case.cu (c : Case) (i : Nat) : Nat := (c.lens.take i).sum
/-- `chunk_indices` flattened (`[NT, 2]`). -/
def Case.ci (c : Case) (i : Nat) : Nat :=
  let r := c.rows.getD (i / 2) (0, 0)
  if i % 2 = 0 then r.1 else r.2

def f32v (q : Rat) : F32 := FloatV.ofRat q

/-- The claim's instance, with the case's contents (as `Pre` requires the buffers to hold). -/
def Case.inst (c : Case) : Claims.ChunkCumsum.Inst :=
  { Claims.ChunkCumsum.mk c.vl c.B c.T c.H c.e c.rev c.hs c.sbits c.dt c.N c.NT with
    sv := fun j => f32v (c.x j)
    cuVal := fun i => ival c.dt (c.cu i)
    ciVal := fun i => ival c.dt (c.ci i) }

/-- `Inst.ok` does not read the contents: the case's instance is covered when `mk`'s is. -/
theorem Case.ok_of (c : Case)
    (h : (Claims.ChunkCumsum.mk c.vl c.B c.T c.H c.e c.rev c.hs c.sbits c.dt c.N c.NT).ok) :
    c.inst.ok := by
  cases h; constructor <;> assumption

def Case.n (c : Case) : Nat := c.B * c.T * c.H

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .f32 ((List.range c.n).map fun j : Nat => ((d.s.ptr j).addr, f32v (c.x j)))
  let m := CMem.ofElems m .f32 ((List.range (c.n + 2 * pad)).map fun j : Nat =>
    ((d.o.ptr ((j : Int) - pad)).addr, f32v sentinel))
  if c.vl then
    let m := CMem.ofElems m c.dt ((List.range (c.N + 1)).map fun i : Nat =>
      ((d.cu.ptr i).addr, ival c.dt (c.cu i)))
    CMem.ofElems m c.dt ((List.range (2 * c.NT)).map fun i : Nat =>
      ((d.ci.ptr i).addr, ival c.dt (c.ci i)))
  else m

def Case.run (c : Case) : LRes (Claims.ChunkCumsum.launch c.inst).size × List Record :=
  execLaunch ((Claims.ChunkCumsum.launch c.inst).env P0) Oracle.empty fill0 100000
    (Claims.ChunkCumsum.model c.inst) (Claims.ChunkCumsum.launch c.inst) c.init

/-- The sequences, as `[lo, hi)` ranges of the flat time axis: the rows `[b·T, b·T + T)`, or
`[cu[n], cu[n+1])`. -/
def Case.seqs (c : Case) : List (Nat × Nat) :=
  if c.vl then (List.range c.N).map fun n => (c.cu n, c.cu (n + 1))
  else (List.range c.B).map fun b => (b * c.T, b * c.T + c.T)

/-- What `Post` says `o[p, h]` holds, for `p` in the sequence `[lo, hi)`: `σ` times the sum of
`s[q, h]` over `q` from the chunk's start to `p` (forward), or from `p` to the chunk's end
(reverse), chunks of `BT` from `lo`. -/
def Case.expect (c : Case) (lo hi p h : Nat) : Rat :=
  let cs := lo + (p - lo) / c.BT * c.BT
  let qs := if c.rev then List.range' p (min (cs + c.BT) hi - p) else List.range' cs (p + 1 - cs)
  c.sigma * ((qs.map fun q => c.x (q * c.H + h)).sum : Int)

/-- The written elements, with their expectations. -/
def Case.outs (c : Case) : List (Nat × Rat) :=
  c.seqs.flatMap fun (lo, hi) => (List.range' lo (hi - lo)).flatMap fun p =>
    (List.range c.H).map fun h => (p * c.H + h, c.expect lo hi p h)

/-- The final memory against `Post`: every written element is its expectation; every byte outside
the written elements keeps its initial value, and no byte is added. -/
def Case.agree (c : Case) (outs : List (Nat × Rat))
    (cf : CConfig (Claims.ChunkCumsum.launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let badOut := outs.filter fun (j, v) => (m.read .f32 (d.o.ptr j).addr).map (·.toRat) != some v
  let wBytes : Std.HashSet Int := outs.foldl (fun s (j, _) =>
    (List.range 4).foldl (fun s i => s.insert ((d.o.ptr j).addr + i)) s) {}
  let init := c.init
  let badElse := init.toList.filter fun (a, b) => !wBytes.contains a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"o elements {(badOut.take 8).map (·.1)} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside the written elements changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch (c.agree c.outs) c.run

/-- Fixed length: `B = 2`, `T = 6`, `H = 2`, `BT = 4` (chunks `[0, 4)`, `[4, 6)` of each row, the
second ragged), forward, scale `2` (the double `0x4000000000000000`); a grid of `2·2·2 = 8`
programs. Signed gates in `[-4, 4]`. -/
def fixedFwd : Case :=
  { vl := false, B := 2, T := 6, H := 2, e := 2, rev := false, hs := true,
    sbits := 0x4000000000000000, sigma := 2, dt := .i32,
    x := fun j => (((5 * j + 3) % 9 : Nat) : Int) - 4, lens := [], rows := [] }

/-- Variable length: `B = 1`, `T = 10`, `H = 2`, `BT = 4`, sequences of lengths `3, 0, 6`
(`cu_seqlens = [0, 3, 3, 9]`: token `9` in no sequence, its output elements keep the sentinel),
`int64` tables; `chunk_indices` the four `(sequence, chunk)` pairs `(0, 0)`, `(2, 0)`, `(2, 1)`
in the order `(2, 1), (0, 0), (2, 0)`; reverse, scale `1/2` (the double `0x3fe0000000000000`). -/
def varlenRev : Case :=
  { vl := true, B := 1, T := 10, H := 2, e := 2, rev := true, hs := true,
    sbits := 0x3fe0000000000000, sigma := 1 / 2, dt := .i64,
    x := fun j => (((7 * j + 2) % 11 : Nat) : Int) - 5, lens := [3, 0, 6],
    rows := [(2, 1), (0, 0), (2, 0)] }

macro "covered" : tactic => `(tactic| exact Case.ok_of _ (Claims.ChunkCumsum.mk_ok
  (by norm_num [Case.N, Case.NT]) (by norm_num [Case.N, Case.NT]) (by norm_num [Case.N, Case.NT])
  (by norm_num [Case.N, Case.NT]) (by norm_num [Case.N, Case.NT]) (by decide) (by decide)
  (by norm_num [Case.N, Case.NT]) (by decide) (by norm_num [Case.N, Case.NT])
  (by norm_num [Case.N, Case.NT]) (by norm_num [Case.N, Case.NT]) (by norm_num [Case.N, Case.NT])
  (by norm_num [Case.N, Case.NT]) (by norm_num [Case.N, Case.NT])))

theorem fixedFwd_ok : fixedFwd.inst.ok := by
  unfold fixedFwd; covered
theorem varlenRev_ok : varlenRev.inst.ok := by
  unfold varlenRev; covered

/-! ## `Pre`, checked on `init` -/

/-- A buffer is valid in `init`: aligned, inside `[0, 2^64)`, every byte present. -/
def validIn {d : DType} (b : Buf d) (m : CMem) : Bool :=
  0 ≤ b.base && b.base + b.len * d.bytes ≤ 2 ^ 64 && b.base % d.bytes == 0 &&
    (List.range (b.len * d.bytes)).all fun i : Nat => m.contains (b.base + i)

/-- A buffer holds `v` in `init`. -/
def holdsIn {d : DType} (b : Buf d) (m : CMem) (v : Nat → d.Val) : Bool :=
  (List.range b.len).all fun j : Nat =>
    ((CMem.toMem m).read d (b.ptr j).addr).map d.toBits == some (d.toBits (v j))

/-- `TablesOk`'s parts, decided on the case's tables (through the claim's `cuAt`, `seqAt`,
`chkAt`, which `Pre` names). -/
def tablesOk (c : Case) : Bool :=
  let d := c.inst
  open Claims.ChunkCumsum in
  0 ≤ cuAt d 0 && (List.range c.N).all (fun i => cuAt d i ≤ cuAt d (i + 1)) && cuAt d c.N ≤ c.T &&
  (List.range c.NT).all (fun r => 0 ≤ seqAt d r && seqAt d r < c.N && 0 ≤ chkAt d r &&
    chkAt d r * c.BT < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat) &&
  (List.range c.NT).all (fun r => (List.range c.NT).all fun r' =>
    r == r' || (seqAt d r, chkAt d r) != (seqAt d r', chkAt d r')) &&
  (List.range c.N).all (fun n => (List.range (c.T + 1)).all fun ch =>
    !((ch * c.BT : Int) < cuAt d (n + 1) - cuAt d n) ||
      (List.range c.NT).any fun r => seqAt d r == n && chkAt d r == ch)

def Case.pre (c : Case) : Bool :=
  let d := c.inst
  let m := c.init
  validIn d.s m && validIn d.o m && holdsIn d.s m d.sv &&
    (!c.vl || (validIn d.cu m && validIn d.ci m && holdsIn d.cu m d.cuVal &&
      holdsIn d.ci m d.ciVal && tablesOk c))

#guard fixedFwd.pre && varlenRev.pre

-- The gates are encoded exactly; the scales' patterns are the values the expectations use (the
-- double, and its `float32` value the kernel receives).
#guard [fixedFwd, varlenRev].all fun c => (List.range c.n).all fun j : Nat =>
  (f32v (c.x j)).toRat == (c.x j : Rat)
#guard [fixedFwd, varlenRev].all fun c =>
  c.inst.scale.toRat == c.sigma && (Claims.ChunkCumsum.sigma32 c.inst).toRat == c.sigma

-- Every written element is covered once, the uncovered token's are not written, and no
-- expectation equals the sentinel (an unwritten element would be caught).
#guard fixedFwd.outs.length == 24 && varlenRev.outs.length == 18
#guard (varlenRev.outs.map (·.1)).all (· < 18)
#guard [fixedFwd, varlenRev].all fun c => c.outs.all fun (_, v) => v != sentinel

-- A hand-computed value: `varlenRev`, sequence 2 (`[3, 9)`), position 3, head 0, reverse: the
-- chunk `[3, 7)`, so `σ·(s[3,0] + s[4,0] + s[5,0] + s[6,0])`.
#guard varlenRev.expect 3 9 3 0 ==
  (1 / 2 : Rat) * (varlenRev.x 6 + varlenRev.x 8 + varlenRev.x 10 + varlenRev.x 12 : Int)

def verdictFixed : Verdict := fixedFwd.verdict
def verdictVarlen : Verdict := varlenRev.verdict

#eval IO.println (verdictFixed.line "chunk_cumsum_f32/fixed_fwd")
#eval IO.println (verdictVarlen.line "chunk_cumsum_f32/varlen_rev")

/-- The verdict until scans have a solver: `unsupported`, stopped at the scan (`Site.other`). -/
def stoppedAtScan : Verdict → Bool
  | .unsupported w => (w.splitOn "Site.other").length > 1
  | _ => false

-- No discrepancy, and the reason the run stops is the scan, nothing earlier. (With a scan
-- solver: `verdictFixed == .certified`, `verdictVarlen == .certified`.)
#guard stoppedAtScan verdictFixed || verdictFixed == .certified
#guard stoppedAtScan verdictVarlen || verdictVarlen == .certified

-- A wrong expectation is caught, once the runs finish: the scale forgotten. (While the runs stop
-- at the scan, both verdicts are `unsupported`, and this guard only checks that.)
#guard
  let v := Verdict.ofLaunch (fixedFwd.agree (fixedFwd.outs.map fun (j, q) => (j, q / 2))) fixedFwd.run
  v.isDiscrepancy || stoppedAtScan v

end CumsumTest
