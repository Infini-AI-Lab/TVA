import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `ssd_chunk_state_bf16` run by `execLaunch` on small covered
instances. Not a test against the device (phase 4): it checks that `exec` runs the model end to
end, and that the final memory is what `Post` says on these inputs, both halves: every element
`states[0, c, h, p, n]` is the exact chunk state, and every other byte is unchanged.

The instances (`Claims.SsdChunkState.mk`): `packed`, three requests whose boundaries fall inside
chunks of `8` (chunks `0` and `1` mix two requests: only the tokens of the request running at the
chunk's end count), a ragged last chunk, `nheads = 2` sharing one group, and SGLang's row layout
(`x` and `B` views of one `bfloat16` buffer of `conv_dim`-element rows, whose other elements hold a
filler value the kernel must not read); `long`, contiguous tensors, chunks of `128` (two `tl.dot`
steps of `64` tokens), a request boundary at token `100`, `ngroups = nheads = 2`; `edge`,
`headdim = 65`, `dstate = 129` (two row tiles and two column tiles per chunk, most lanes masked),
chunks of `4`.

The inputs: `x`, `B` small integers in `[-2, 2]`, `dt` in `{1/2, 1, 2}`, `dA_cumsum` all `0`
(covered: `Vals` asks `dA_cumsum ≤ 0`). So every decay is `exp(0 − 0) = 1`, every scaled `B`
(`B·dt`) is exact in `float32` and in `bfloat16`, and every `tl.dot` partial sum is a dyadic
rational of magnitude at most `2^11`, exact in `float32` in any order: each element of `states` is
then exactly `sRef = Σ_{t ∈ toks c} x[t, h, p]·B[t, h / ratio, n]·dt[h, c, t − c·chunk]` (the
request filter computed here from the request lengths), which `Post` allows at distance `0`.

**The platform answers are assumed, not proved.** `tl.exp` (`PlatQ.math .exp`, line 200) and
`tl.dot(x, b) + acc` (`dot_add`, line 204) are platform operations. The oracle answers `tl.exp`
with `1` when every lane's input is `±0` (and nothing otherwise: here every lane is `0`, the
loaded and the `other=0.0` values alike), and each dot with the exact `acc + x·b`, computed from
the request's operands, when that is a `float32` (nothing otherwise). These are the only results
a correct device gives: `exp(0) = 1` is exact, and the exact dot is representable. The verdicts
are `conditional n`, `n` = two answers (one `exp`, one dot) per loop step of each program. -/

open TVA TVA.Validation

namespace SsdChunkStateTest

open Claims.SsdChunkState (Inst launch model)

/-- A platform that relates nothing: the platform results come from the oracle, never from `P0`. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle: `exp(0) = 1`, exact dots -/

/-- An accumulator element holding the rational `q` exactly (`float32` only). -/
def accOf : (d : DType) → Rat → Option d.Val
  | .f32, q => let v : F32 := FloatV.ofRat q; if v.toRat == q then some v else none
  | _, _ => none

/-- The exact `acc + a·b` of a `tl.dot` request, from its own operands: `none` unless every
operand is finite and every result an exact `float32`. -/
def exactDot : (q : PlatQ) → Option q.Out
  | @PlatQ.dot cfg M N K a b acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => a.get2 i k).toArray.mapM cfg.a.val?
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => b.get2 k j).toArray.mapM cfg.b.val?
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM cfg.acc.val?
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      accOf cfg.acc ((List.range K).foldl (fun s k =>
        let x := A[i * K + k]!
        if x == 0 then s else s + x * B[k * N + j]!) C[e]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- `tl.exp` of a block of zeros: `1` in every lane; `none` if some lane is not `±0`. -/
def expZero : (q : PlatQ) → Option q.Out
  | @PlatQ.math .exp f s x =>
    if (Idx.all s).all (fun i => (x.get i).isFinite && (x.get i).toRat == 0) then
      some (x.map fun _ => (FloatV.ofRat 1 : FloatV f))
    else none
  | _ => none

/-- The oracle: `expZero` or `exactDot` at every platform request; nothing else. -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => (expZero q).orElse fun _ => exactDot q
  | _ => none

/-! ## The cases -/

/-- A sentinel in every element of `states` before the launch, and in the elements just outside
it: those outside must keep it; every inside one must be overwritten. -/
def sentinel : Rat := -999

/-- A filler in the unused elements of `x`'s and `B`'s storage (the rest of each row). -/
def filler : Rat := 3

/-- Sentinel elements on each side of `states`. -/
def pad : Nat := 4

/-- One case: the arguments of the claim's `mk`. -/
structure Case where
  lens : List Nat
  H : Nat
  P : Nat
  N : Nat
  G : Nat
  C : Nat
  rows : Bool

def Case.inst (c : Case) : Inst := Claims.SsdChunkState.mk c.lens c.H c.P c.N c.G c.C c.rows
def Case.L (c : Case) : Nat := c.lens.sum
def Case.nc (c : Case) : Nat := (c.L + c.C - 1) / c.C

/-- The request of token `t`, from the lengths (computed here, independently of the claim's
`reqOf`): the first `r` whose end `lens[0] + … + lens[r]` is above `t`. -/
def Case.req (c : Case) (t : Nat) : Nat :=
  let rec go (ls : List Nat) (r start : Nat) : Nat := match ls with
    | [] => r
    | l :: rest => if t < start + l then r else go rest (r + 1) (start + l)
  go c.lens 0 0

/-- The inputs: `x[t, h, p]`, `B[t, g, n]` in `[-2, 2]`; `dt[h, c, i]` in `{1/2, 1, 2}`. -/
def xv (t h p : Nat) : Int := ((3 * t + 5 * h + 7 * p) % 5 : Nat) - (2 : Int)
def bv (t g n : Nat) : Int := ((2 * t + 3 * g + 5 * n + 1) % 5 : Nat) - (2 : Int)
def dtv (h c i : Nat) : Rat := (2 ^ ((h + c + i) % 3) : Nat) / 2

/-- Element addresses, from the layout of `mk` (rows of `conv_dim = H·P + 2·G·N` elements, or
contiguous). -/
def Case.xT (c : Case) : Nat := if c.rows then c.H * c.P + 2 * c.G * c.N else c.H * c.P
def Case.bT (c : Case) : Nat := if c.rows then c.H * c.P + 2 * c.G * c.N else c.G * c.N
def Case.xA (c : Case) (t h p : Nat) : Int := (c.inst.x.ptr (t * c.xT + h * c.P + p : Nat)).addr
def Case.bA (c : Case) (t g n : Nat) : Int := (c.inst.b.ptr (t * c.bT + g * c.N + n : Nat)).addr
def Case.dA (c : Case) (h ch i : Nat) : Nat := h * (c.nc * c.C) + ch * c.C + i
def Case.stE (c : Case) (ch h p n : Nat) : Nat := ((ch * c.H + h) * c.P + p) * c.N + n

def bfv (q : Rat) : BF16 := FloatV.ofRat q
def f32v (q : Rat) : F32 := FloatV.ofRat q

def Case.init (c : Case) : CMem :=
  let d := c.inst
  -- every element of `x`'s and `B`'s storage: the filler, then the inputs over it
  let m := CMem.ofElems ∅ .bf16 ((List.range d.x.len).map fun e : Nat => ((d.x.ptr e).addr, bfv filler))
  let m := CMem.ofElems m .bf16 ((List.range d.b.len).map fun e : Nat => ((d.b.ptr e).addr, bfv filler))
  let m := CMem.ofElems m .bf16 ((List.range c.L).flatMap fun t => (List.range c.H).flatMap fun h =>
    (List.range c.P).map fun p => (c.xA t h p, bfv (xv t h p)))
  let m := CMem.ofElems m .bf16 ((List.range c.L).flatMap fun t => (List.range c.G).flatMap fun g =>
    (List.range c.N).map fun n => (c.bA t g n, bfv (bv t g n)))
  let m := CMem.ofElems m .f32 ((List.range c.H).flatMap fun h => (List.range c.nc).flatMap fun ch =>
    (List.range c.C).map fun i => ((d.dt.ptr (c.dA h ch i : Nat)).addr, f32v (dtv h ch i)))
  let m := CMem.ofElems m .f32 ((List.range d.dA.len).map fun e : Nat => ((d.dA.ptr e).addr, f32v 0))
  let m := CMem.ofElems m .i32 ((List.range c.L).map fun t : Nat =>
    ((d.seqIdx.ptr t).addr, (IntV.ofNat (c.req t) : I32)))
  CMem.ofElems m .f32 ((List.range (d.states.len + 2 * pad)).map fun e : Nat =>
    ((d.states.ptr ((e : Int) - pad)).addr, f32v sentinel))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) O fill0 100000 (model c.inst) (launch c.inst) c.init

/-- The tokens chunk `ch` sums: its tokens of the request running at its last token. -/
def Case.toks (c : Case) (ch : Nat) : List Nat :=
  let lo := ch * c.C
  let hi := min ((ch + 1) * c.C) c.L
  ((List.range (hi - lo)).map (lo + ·)).filter fun t => c.req t == c.req (hi - 1)

/-- What `Post` says `states[0, ch, h, p, n]` holds on these inputs: the exact chunk state
`Σ_{t ∈ toks} x·B·exp(0)·dt` (the group of head `h` is `h / (H / G)`). -/
def Case.expect (c : Case) (ch h p n : Nat) : Rat :=
  ((c.toks ch).map fun t => (xv t h p * bv t (h / (c.H / c.G)) n : Int) * dtv h ch (t - ch * c.C)).sum

/-- The elements of `states`: `(chunk, head, p, n)`. -/
def Case.outs (c : Case) : List (Nat × Nat × Nat × Nat) :=
  (List.range c.nc).flatMap fun ch => (List.range c.H).flatMap fun h =>
    (List.range c.P).flatMap fun p => (List.range c.N).map fun n => (ch, h, p, n)

/-- `states`' bytes: its `nchunks·H·P·N` elements (the whole buffer). -/
def Case.inOut (c : Case) (a : Int) : Bool :=
  c.inst.states.base ≤ a && a < c.inst.states.base + 4 * (c.nc * c.H * c.P * c.N : Nat)

/-- The final memory against `Post`: every element of `states` is a finite `float32` equal to
the expectation; every byte outside `states` (the inputs, the filler, the sentinels) keeps its
initial value, and no byte is added. -/
def Case.agree (c : Case) (expect : Nat → Nat → Nat → Nat → Rat)
    (cf : CConfig (launch c.inst).size) : Option String :=
  let m := CMem.toMem cf.mem
  let init := c.init
  let badOut := c.outs.filter fun (ch, h, p, n) =>
    match m.read .f32 (c.inst.states.ptr (c.stE ch h p n : Nat)).addr with
    | some v => !(v.isFinite && v.toRat == expect ch h p n)
    | none => true
  let badElse := init.toList.filter fun (a, b) => !c.inOut a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"states elements {badOut.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside states changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch (c.agree c.expect) c.run

/-- The number of loop steps (`cdiv(chunk_size_limit, 64)`) summed over the programs: two
platform answers each. -/
def Case.steps (c : Case) : Nat :=
  let gx := (c.P + 63) / 64 * ((c.N + 127) / 128)
  gx * c.H * ((List.range c.nc).map fun ch => (min ((ch + 1) * c.C) c.L - ch * c.C + 63) / 64).sum

/-- Three requests `[5, 6, 10]` (tokens `0–4`, `5–10`, `11–20`), chunks of `8`: chunk `0` sums
tokens `5–7`, chunk `1` tokens `11–15`, chunk `2` (ragged, `16–20`) all its tokens. `nheads = 2`,
`ngroups = 1`, `headdim = 3`, `dstate = 4`; SGLang's row layout. Six programs, one step each. -/
def packed : Case := { lens := [5, 6, 10], H := 2, P := 3, N := 4, G := 1, C := 8, rows := true }

/-- Two requests `[100, 60]`, chunks of `128`: chunk `0` sums tokens `100–127` (two dot steps),
chunk `1` tokens `128–159`. `nheads = ngroups = 2`, `headdim = 2`, `dstate = 3`; contiguous. -/
def long : Case := { lens := [100, 60], H := 2, P := 2, N := 3, G := 2, C := 128, rows := false }

/-- Two requests `[3, 2]`, chunks of `4`, one head, `headdim = 65`, `dstate = 129`: a `2 × 2` grid
of tiles per chunk, the second row tile with one valid row and the second column tile with one
valid column; row layout. -/
def edge : Case := { lens := [3, 2], H := 1, P := 65, N := 129, G := 1, C := 4, rows := true }

-- The instances are covered: `Inst.ok`, every condition decided, as the claim's `setup_*_ok`.
theorem packed_ok : packed.inst.ok := by
  constructor <;> first | decide | exact ⟨3, by decide⟩ |
    exact ⟨Claims.SsdChunkState.bytes_disj _ _ (by decide), Claims.SsdChunkState.bytes_disj _ _ (by decide),
      Claims.SsdChunkState.bytes_disj _ _ (by decide), Claims.SsdChunkState.bytes_disj _ _ (by decide),
      Claims.SsdChunkState.bytes_disj _ _ (by decide)⟩

theorem long_ok : long.inst.ok := by
  constructor <;> first | decide | exact ⟨7, by decide⟩ |
    exact ⟨Claims.SsdChunkState.bytes_disj _ _ (by decide), Claims.SsdChunkState.bytes_disj _ _ (by decide),
      Claims.SsdChunkState.bytes_disj _ _ (by decide), Claims.SsdChunkState.bytes_disj _ _ (by decide),
      Claims.SsdChunkState.bytes_disj _ _ (by decide)⟩

theorem edge_ok : edge.inst.ok := by
  constructor <;> first | decide | exact ⟨2, by decide⟩ |
    exact ⟨Claims.SsdChunkState.bytes_disj _ _ (by decide), Claims.SsdChunkState.bytes_disj _ _ (by decide),
      Claims.SsdChunkState.bytes_disj _ _ (by decide), Claims.SsdChunkState.bytes_disj _ _ (by decide),
      Claims.SsdChunkState.bytes_disj _ _ (by decide)⟩

/-- A buffer is valid in `m` (`Buf.Valid`): aligned, inside `[0, 2^64)`, every byte present. -/
def validB {dt : DType} (m : Mem) (b : Buf dt) : Bool :=
  decide (0 ≤ b.base) && decide (b.base + ((b.len * dt.bytes : Nat) : Int) ≤ 2 ^ 64) &&
    b.base % (dt.bytes : Int) == 0 &&
    (List.range (b.len * dt.bytes)).all fun k : Nat => (m (b.base + k)).isSome

-- `Pre`: the six buffers are valid, and `seq_idx` is the host's (`MetaOk`, with the claim's
-- `reqOf`, which `Pre` names; it agrees with the test's own `req`).
#guard [packed, long, edge].all fun c =>
  let d := c.inst
  let m := CMem.toMem c.init
  validB m d.x && validB m d.b && validB m d.dt && validB m d.dA && validB m d.seqIdx &&
    validB m d.states &&
    (List.range c.L).all fun t : Nat =>
      Claims.SsdChunkState.reqOf d t == c.req t &&
      (m.read .i32 (d.seqIdx.ptr t).addr).map DType.i32.toBits ==
        some (DType.i32.toBits (IntV.ofNat (Claims.SsdChunkState.reqOf d t) : I32))

-- The inputs are encoded exactly; the layouts are the claim's (`xIdx`, `bIdx`, `dIdx`,
-- `stIdx`); `dA_cumsum` is all `0` (`Vals`: at most `0`).
#guard [packed, long, edge].all fun c =>
  let d := c.inst
  let m := CMem.toMem c.init
  ((List.range c.L).all fun t : Nat => (List.range c.H).all fun h : Nat => (List.range c.P).all fun p : Nat =>
    c.xA t h p == (d.x.ptr (Claims.SsdChunkState.xIdx d t h p : Nat)).addr &&
    (m.read .bf16 (c.xA t h p)).map (·.toRat) == some (xv t h p : Rat)) &&
  ((List.range c.L).all fun t : Nat => (List.range c.G).all fun g : Nat => (List.range c.N).all fun n : Nat =>
    c.bA t g n == (d.b.ptr (Claims.SsdChunkState.bIdx d t g n : Nat)).addr &&
    (m.read .bf16 (c.bA t g n)).map (·.toRat) == some (bv t g n : Rat)) &&
  ((List.range c.H).all fun h : Nat => (List.range c.nc).all fun ch : Nat => (List.range c.C).all fun i : Nat =>
    c.dA h ch i == Claims.SsdChunkState.dIdx d h ch i &&
    (m.read .f32 (d.dt.ptr (c.dA h ch i : Nat)).addr).map (·.toRat) == some (dtv h ch i) &&
    (m.read .f32 (d.dA.ptr (c.dA h ch i : Nat)).addr).map (·.toRat) == some 0) &&
  c.outs.all (fun (ch, h, p, n) => c.stE ch h p n == Claims.SsdChunkState.stIdx d ch h p n) &&
  c.nc == Claims.SsdChunkState.nc d && c.outs.length == d.states.len

-- The request filter is exercised: chunks summing only part of their tokens.
#guard packed.toks 0 == [5, 6, 7] && packed.toks 1 == [11, 12, 13, 14, 15] &&
  packed.toks 2 == [16, 17, 18, 19, 20] && long.toks 0 == (List.range 28).map (100 + ·) &&
  edge.toks 0 == [3] && edge.toks 1 == [4]

-- No expectation equals the sentinel; every expectation is a `float32` (exact).
#guard [packed, long, edge].all fun c => c.outs.all fun (ch, h, p, n) =>
  c.expect ch h p n != sentinel && (f32v (c.expect ch h p n)).toRat == c.expect ch h p n

def verdictPacked : Verdict := packed.verdict
def verdictLong : Verdict := long.verdict
def verdictEdge : Verdict := edge.verdict

#eval IO.println (verdictPacked.line "ssd_chunk_state_bf16/packed")
#eval IO.println (verdictLong.line "ssd_chunk_state_bf16/long")
#eval IO.println (verdictEdge.line "ssd_chunk_state_bf16/edge")
-- Agreement resting on two assumed answers (`exp`, dot) per loop step, and at least one.
#guard let v := verdictPacked; v == .conditional (2 * packed.steps) && v.assumed > 0
#guard let v := verdictLong; v == .conditional (2 * long.steps) && v.assumed > 0
#guard let v := verdictEdge; v == .conditional (2 * edge.steps) && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (the request filter dropped: every token
-- of the chunk summed; one element off by `1/2`).
#guard
  let r := packed.run
  let all (ch h p n : Nat) : Rat :=
    ((List.range (min ((ch + 1) * 8) 21 - ch * 8)).map fun i =>
      let t := ch * 8 + i
      (xv t h p * bv t 0 n : Int) * dtv h ch i).sum
  (Verdict.ofLaunch (packed.agree all) r).isDiscrepancy &&
  (Verdict.ofLaunch (packed.agree fun ch h p n =>
    if ch = 1 ∧ h = 1 ∧ p = 2 ∧ n = 3 then packed.expect ch h p n + 1 / 2
    else packed.expect ch h p n) r).isDiscrepancy

end SsdChunkStateTest
