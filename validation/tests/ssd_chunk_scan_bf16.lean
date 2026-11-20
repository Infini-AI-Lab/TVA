import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `ssd_chunk_scan_bf16` run by `execLaunch` on small covered instances.
Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and that
the final memory is what `Post` says on these inputs, both halves: every `out[0, t, h, p]` is the
exact chunked-scan output `yRef`, and every other byte is unchanged.

The instances (`Claims.SsdChunkScan.mk`): `packedInit`, with initial states (`HAS_INITSTATES`),
three requests `[5, 6, 10]` in chunks of `8` (two requests start inside a chunk: five logical
chunks, the host's `chunk_indices = [0, 0, 1, 1, 2]`, `chunk_offsets = [0, 5, 0, 3, 0]`; the
entering state of a logical chunk is `states[c]` or the request's initial state), `nheads = 2`
sharing one group, `cb` nonzero across requests (the kernel must not sum those pairs);
`packedNoInit`, the same without initial states (`cb` zero across requests, as `CbZero` asks;
the state term of a row whose logical chunk starts its request is zero); `longInit`, with initial
states, two requests `[70, 60]` in chunks of `128` (two row tiles per logical chunk, up to four
loop steps of `32` tokens, a logical chunk starting at offset `70`), `dstate = 17`
(`BLOCK_SIZE_DSTATE = 32`).

The inputs: `x`, `C` in `[-1, 1]`, `cb` in `[-2, 2]`, `states` and `initial_states` small
integers, `D` in `{1, -2}`, `dt` in `{1/2, 1, 2}`, `dA_cumsum` all `0` (covered: `Vals` asks
`dA_cumsum ≤ 0`; the other parts of `Vals` hold: every value is finite and tiny). So every decay
is `exp(0) = 1`, every product `cb·dt` is exact in `float32` and in `bfloat16`, the states are
exact in `bfloat16`, every `tl.dot` partial sum is a dyadic rational far inside `float32`, and the
output `yRef = Σ_n C·S + Σ_{s ∈ toks t} cb·dt_s·x_s + D·x_t` (computed here from the request
lengths) is a `bfloat16` (checked below): each output is then exactly `yRef`, which `Post` allows
at distance `0`.

**The platform answers are assumed, not proved.** `tl.exp` (`PlatQ.math .exp`: lines 337–344 on
`[64]`, line 408 on `[64, 32]`), `tl.dot` (line 361) and `tl.dot(cb, x) + acc` (`dot_add`, line
420) are platform operations. The oracle answers `tl.exp` with `1` when every lane's input is `±0`
(and nothing otherwise: here every lane is `0`), and each dot with the exact `acc + a·b`, computed
from the request's operands, when that is a `float32` (nothing otherwise). These are the only
results a correct device gives: `exp(0) = 1` is exact, and the exact dot is representable. The
verdicts are `conditional n`, `n` = two answers per program (`exp` of `scale_m`, the state's dot)
and two per loop step (`exp`, `dot_add`). (`acc + x_residual·D`, line 442, is `mul_add`: `exec`
computes the unfused IEEE value, which is exact here, as the fused one is.) -/

open TVA TVA.Validation

namespace SsdChunkScanTest

open Claims.SsdChunkScan (Inst launch model)

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

/-- A sentinel in every element of `out` before the launch, and in the elements just outside
it: those outside must keep it; every inside one must be overwritten. -/
def sentinel : Rat := -99

/-- A filler in the unused elements of `x`'s and `C`'s storage (the rest of each row). -/
def filler : Rat := 5

/-- Sentinel elements on each side of `out`. -/
def pad : Nat := 4

/-- One case: the arguments of the claim's `mk`. -/
structure Case where
  hasInit : Bool
  lens : List Nat
  H : Nat
  P : Nat
  N : Nat
  G : Nat
  C : Nat
  BSD : Nat

def Case.inst (c : Case) : Inst := Claims.SsdChunkScan.mk c.hasInit c.lens c.H c.P c.N c.G c.C c.BSD
def Case.L (c : Case) : Nat := c.lens.sum
def Case.nc (c : Case) : Nat := (c.L + c.C - 1) / c.C
def Case.nreq (c : Case) : Nat := c.lens.length
/-- `conv_dim`: the token stride of `x` and `C` in `mk`'s layout. -/
def Case.conv (c : Case) : Nat := c.H * c.P + 2 * c.G * c.N

/-- The first token of request `r`. -/
def Case.start (c : Case) (r : Nat) : Nat := (c.lens.take r).sum

/-- The request of token `t`, from the lengths (computed here, independently of the claim's
`reqOf`). -/
def Case.req (c : Case) (t : Nat) : Nat :=
  let rec go (ls : List Nat) (r start : Nat) : Nat := match ls with
    | [] => r
    | l :: rest => if t < start + l then r else go rest (r + 1) (start + l)
  go c.lens 0 0

/-- The inputs. -/
def xv (t h p : Nat) : Int := ((2 * t + 3 * h + 5 * p) % 3 : Nat) - (1 : Int)
def cv (t g n : Nat) : Int := ((t + 2 * g + 3 * n + 1) % 3 : Nat) - (1 : Int)
def stv (ch h p n : Nat) : Int := ((ch + 2 * h + 3 * p + 5 * n) % 5 : Nat) - (2 : Int)
def isv (r h p n : Nat) : Int := ((3 * r + h + 2 * p + n + 1) % 7 : Nat) - (3 : Int)
def Dv (h : Nat) : Int := if h % 2 = 0 then 1 else -2
def dtv (h ch i : Nat) : Rat := (2 ^ ((h + ch + i) % 3) : Nat) / 2
/-- `cb[ch, g, i, j]`: in `[-2, 2]`; without initial states, `0` when tokens `ch·C + i` and
`ch·C + j` (both packed tokens) belong to different requests. -/
def Case.cbv (c : Case) (ch g i j : Nat) : Int :=
  let t := ch * c.C + i
  let s := ch * c.C + j
  if !c.hasInit && t < c.L && s < c.L && c.req t != c.req s then 0
  else ((ch + 3 * g + 2 * i + 5 * j + 1) % 5 : Nat) - (2 : Int)

/-- Element indices, from `mk`'s layout. -/
def Case.xE (c : Case) (t h p : Nat) : Nat := t * c.conv + h * c.P + p
def Case.cE (c : Case) (t g n : Nat) : Nat := t * c.conv + g * c.N + n
def Case.cbE (c : Case) (ch g i j : Nat) : Nat := ((ch * c.G + g) * c.C + i) * c.C + j
def Case.dE (c : Case) (h ch i : Nat) : Nat := (h * c.nc + ch) * c.C + i
def Case.stE (c : Case) (ch h p n : Nat) : Nat := ((ch * c.H + h) * c.P + p) * c.N + n
def Case.isE (c : Case) (r h p n : Nat) : Nat := ((r * c.H + h) * c.P + p) * c.N + n
def Case.outE (c : Case) (t h p : Nat) : Nat := (t * c.H + h) * c.P + p

def bfv (q : Rat) : BF16 := FloatV.ofRat q
def f32v (q : Rat) : F32 := FloatV.ofRat q

/-- `chunk_indices` and `chunk_offsets` with initial states, written by hand from the host's
loop (each case's docstring); empty without. -/
structure Meta where
  idx : List Nat
  offs : List Nat

def Case.init (c : Case) (mt : Meta) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .bf16 ((List.range d.x.len).map fun e : Nat => ((d.x.ptr e).addr, bfv filler))
  let m := CMem.ofElems m .bf16 ((List.range d.C.len).map fun e : Nat => ((d.C.ptr e).addr, bfv filler))
  let m := CMem.ofElems m .bf16 ((List.range c.L).flatMap fun t => (List.range c.H).flatMap fun h =>
    (List.range c.P).map fun p => ((d.x.ptr (c.xE t h p : Nat)).addr, bfv (xv t h p)))
  let m := CMem.ofElems m .bf16 ((List.range c.L).flatMap fun t => (List.range c.G).flatMap fun g =>
    (List.range c.N).map fun n => ((d.C.ptr (c.cE t g n : Nat)).addr, bfv (cv t g n)))
  let m := CMem.ofElems m .f32 ((List.range c.nc).flatMap fun ch => (List.range c.G).flatMap fun g =>
    (List.range c.C).flatMap fun i => (List.range c.C).map fun j =>
      ((d.cb.ptr (c.cbE ch g i j : Nat)).addr, f32v (c.cbv ch g i j)))
  let m := CMem.ofElems m .f32 ((List.range c.H).flatMap fun h => (List.range c.nc).flatMap fun ch =>
    (List.range c.C).map fun i => ((d.dt.ptr (c.dE h ch i : Nat)).addr, f32v (dtv h ch i)))
  let m := CMem.ofElems m .f32 ((List.range d.dA.len).map fun e : Nat => ((d.dA.ptr e).addr, f32v 0))
  let m := CMem.ofElems m .i32 ((List.range c.L).map fun t : Nat =>
    ((d.seqIdx.ptr t).addr, (IntV.ofNat (c.req t) : I32)))
  let m := CMem.ofElems m .f32 ((List.range c.nc).flatMap fun ch => (List.range c.H).flatMap fun h =>
    (List.range c.P).flatMap fun p => (List.range c.N).map fun n =>
      ((d.states.ptr (c.stE ch h p n : Nat)).addr, f32v (stv ch h p n)))
  let m := CMem.ofElems m .bf16 ((List.range c.H).map fun h : Nat => ((d.D.ptr h).addr, bfv (Dv h)))
  let m := if c.hasInit then CMem.ofElems m .f32 ((List.range c.nreq).flatMap fun r =>
      (List.range c.H).flatMap fun h => (List.range c.P).flatMap fun p => (List.range c.N).map fun n =>
        ((d.initst.ptr (c.isE r h p n : Nat)).addr, f32v (isv r h p n))) else m
  let m := CMem.ofElems m .i32 ((List.range mt.idx.length).map fun k : Nat =>
    ((d.chunkIdx.ptr k).addr, (IntV.ofNat (mt.idx.getD k 0) : I32)))
  let m := CMem.ofElems m .i32 ((List.range mt.offs.length).map fun k : Nat =>
    ((d.chunkOffs.ptr k).addr, (IntV.ofNat (mt.offs.getD k 0) : I32)))
  CMem.ofElems m .bf16 ((List.range (d.out.len + 2 * pad)).map fun e : Nat =>
    ((d.out.ptr ((e : Int) - pad)).addr, bfv sentinel))

def Case.run (c : Case) (mt : Meta) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) O fill0 100000 (model c.inst) (launch c.inst) (c.init mt)

/-! ### The expectation, from the request lengths -/

/-- The physical chunk of `t`, and the first token of its logical chunk: the later of the
chunk's first token and the request's. -/
def Case.chk (c : Case) (t : Nat) : Nat := t / c.C
def Case.lTok (c : Case) (t : Nat) : Nat := max (c.chk t * c.C) (c.start (c.req t))

/-- `t`'s logical chunk starts a request other than the first: it enters from the request's
initial state (zero without initial states), not from `states[chunk]`. -/
def Case.fromInit (c : Case) (t : Nat) : Bool := c.req t ≥ 1 && c.chk t * c.C ≤ c.start (c.req t)

/-- The entering state `S[p, n]` of `t`'s logical chunk. -/
def Case.S (c : Case) (t h p n : Nat) : Int :=
  if c.fromInit t then (if c.hasInit then isv (c.req t) h p n else 0) else stv (c.chk t) h p n

/-- What `Post` says `out[0, t, h, p]` holds on these inputs: `yRef` with every decay
`exp(0) = 1`: `Σ_n C[t, g, n]·S[p, n] + Σ_{lTok ≤ s ≤ t} cb[ch, g, t − ch·C, s − ch·C]·dt_s·x_s +
D·x_t`, `g = h / (H / G)`. -/
def Case.expect (c : Case) (t h p : Nat) : Rat :=
  let g := h / (c.H / c.G)
  let ch := c.chk t
  let sig : Int := ((List.range c.N).map fun n => cv t g n * c.S t h p n).sum
  let intra : Rat := ((List.range (t + 1 - c.lTok t)).map fun k =>
    let s := c.lTok t + k
    (c.cbv ch g (t - ch * c.C) (s - ch * c.C) * xv s h p : Int) * dtv h ch (s - ch * c.C)).sum
  (sig : Rat) + intra + (Dv h * xv t h p : Int)

/-- The elements of `out`: `(t, h, p)`. -/
def Case.outs (c : Case) : List (Nat × Nat × Nat) :=
  (List.range c.L).flatMap fun t => (List.range c.H).flatMap fun h => (List.range c.P).map fun p => (t, h, p)

/-- `out`'s bytes: its `seqlen·H·P` elements (the whole buffer). -/
def Case.inOut (c : Case) (a : Int) : Bool :=
  c.inst.out.base ≤ a && a < c.inst.out.base + 2 * (c.L * c.H * c.P : Nat)

/-- The final memory against `Post`: every element of `out` is a finite `bfloat16` equal to the
expectation; every byte outside `out` (the inputs, the filler, the sentinels) keeps its initial
value, and no byte is added. -/
def Case.agree (c : Case) (mt : Meta) (expect : Nat → Nat → Nat → Rat)
    (cf : CConfig (launch c.inst).size) : Option String :=
  let m := CMem.toMem cf.mem
  let init := c.init mt
  let badOut := c.outs.filter fun (t, h, p) =>
    match m.read .bf16 (c.inst.out.ptr (c.outE t h p : Nat)).addr with
    | some v => !(v.isFinite && v.toRat == expect t h p)
    | none => true
  let badElse := init.toList.filter fun (a, b) => !c.inOut a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"out elements {badOut.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside out changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) (mt : Meta) : Verdict := Verdict.ofLaunch (c.agree mt c.expect) (c.run mt)

/-- Three requests `[5, 6, 10]` (tokens `0–4`, `5–10`, `11–20`), chunks of `8`, with initial
states. The host's loop: request `1` (`5–10`) inserts one logical chunk (`chunk_indices[1:3] -= 1`,
`chunk_offsets[1] = 5`), request `2` (`11–20`) another (`chunk_indices[3:5] -= 2`,
`chunk_offsets[3] = 3`): logical chunks `(0, +0)`, `(0, +5)`, `(1, +0)`, `(1, +3)`, `(2, +0)`.
`nheads = 2`, `ngroups = 1`, `headdim = 3`, `dstate = 4`, `BLOCK_SIZE_DSTATE = 16`. Ten programs,
one loop step each. -/
def packedInit : Case :=
  { hasInit := true, lens := [5, 6, 10], H := 2, P := 3, N := 4, G := 1, C := 8, BSD := 16 }
def packedInitMeta : Meta := { idx := [0, 0, 1, 1, 2], offs := [0, 5, 0, 3, 0] }

/-- The same without initial states: the grid's `y` is `nchunks = 3`; no metadata. Six programs,
one loop step each. -/
def packedNoInit : Case := { packedInit with hasInit := false }
def noMeta : Meta := { idx := [], offs := [] }

/-- Two requests `[70, 60]` (tokens `0–69`, `70–129`), chunks of `128`, with initial states:
request `1` inserts one logical chunk (`chunk_indices = [0, 0, 1]`, `chunk_offsets = [0, 70, 0]`).
`nheads = ngroups = 1`, `headdim = 2`, `dstate = 17`, `BLOCK_SIZE_DSTATE = 32`. Two row tiles per
logical chunk: six programs, `2 + 3 + 2 + 4 + 1 + 1` loop steps (`K_MAX = min((pid_m + 1)·64,
chunk_size_limit)` with the limits `70`, `128`, `2`). -/
def longInit : Case :=
  { hasInit := true, lens := [70, 60], H := 1, P := 2, N := 17, G := 1, C := 128, BSD := 32 }
def longInitMeta : Meta := { idx := [0, 0, 1], offs := [0, 70, 0] }

-- The instances are covered (`Inst.ok`, by the claim's `mk_ok`, every condition decided).
theorem packedInit_ok : packedInit.inst.ok :=
  Claims.SsdChunkScan.mk_ok _ _ _ _ _ _ _ _ 4 3 rfl rfl (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem packedNoInit_ok : packedNoInit.inst.ok :=
  Claims.SsdChunkScan.mk_ok _ _ _ _ _ _ _ _ 4 3 rfl rfl (by decide) (by decide) (by decide)
    (by decide) (by decide)
theorem longInit_ok : longInit.inst.ok :=
  Claims.SsdChunkScan.mk_ok _ _ _ _ _ _ _ _ 5 7 rfl rfl (by decide) (by decide) (by decide)
    (by decide) (by decide)

/-- A buffer is valid in `m` (`Buf.Valid`): aligned, inside `[0, 2^64)`, every byte present. -/
def validB {dt : DType} (m : Mem) (b : Buf dt) : Bool :=
  decide (0 ≤ b.base) && decide (b.base + ((b.len * dt.bytes : Nat) : Int) ≤ 2 ^ 64) &&
    b.base % (dt.bytes : Int) == 0 &&
    (List.range (b.len * dt.bytes)).all fun k : Nat => (m (b.base + k)).isSome

def cases : List (Case × Meta) :=
  [(packedInit, packedInitMeta), (packedNoInit, noMeta), (longInit, longInitMeta)]

-- `Pre`: the twelve buffers are valid; the metadata are the host's (`MetaOk`, with the claim's
-- `reqOf`, `hostIdx`, `hostOff`, which `Pre` names; they agree with the test's own `req` and
-- hand-written metadata, and the grid's `y` is their length); without initial states `cb` is a
-- finite zero across requests (`CbZero`).
#guard cases.all fun (c, mt) =>
  let d := c.inst
  let m := CMem.toMem (c.init mt)
  validB m d.cb && validB m d.x && validB m d.dt && validB m d.dA && validB m d.seqIdx &&
  validB m d.C && validB m d.states && validB m d.D && validB m d.initst && validB m d.chunkIdx &&
  validB m d.chunkOffs && validB m d.out &&
  ((List.range c.L).all fun t : Nat =>
    Claims.SsdChunkScan.reqOf d t == c.req t &&
    (m.read .i32 (d.seqIdx.ptr t).addr).map DType.i32.toBits ==
      some (DType.i32.toBits (IntV.ofNat (Claims.SsdChunkScan.reqOf d t) : I32))) &&
  (!c.hasInit || (d.nmeta == mt.idx.length && d.nmeta == mt.offs.length &&
    (List.range d.nmeta).all fun k : Nat =>
      Claims.SsdChunkScan.hostIdx d k == mt.idx.getD k 0 &&
      Claims.SsdChunkScan.hostOff d k == mt.offs.getD k 0 &&
      (m.read .i32 (d.chunkIdx.ptr k).addr).map DType.i32.toBits ==
        some (DType.i32.toBits (IntV.ofNat (Claims.SsdChunkScan.hostIdx d k) : I32)) &&
      (m.read .i32 (d.chunkOffs.ptr k).addr).map DType.i32.toBits ==
        some (DType.i32.toBits (IntV.ofNat (Claims.SsdChunkScan.hostOff d k) : I32)))) &&
  (c.hasInit || (d.nmeta == c.nc &&
    (List.range c.L).all fun t : Nat => (List.range (t + 1 - t / c.C * c.C)).all fun k : Nat =>
      let s := t / c.C * c.C + k
      Claims.SsdChunkScan.reqOf d s == Claims.SsdChunkScan.reqOf d t ||
      (List.range c.G).all fun g : Nat =>
        match m.read .f32 (d.cb.ptr (Claims.SsdChunkScan.cbIdx d (t / c.C) g (t % c.C) k : Nat)).addr with
        | some v => v.isFinite && v.toRat == 0
        | none => false))

-- The inputs are encoded exactly at the claim's element indices (`xIdx`, `cIdx`, `cbIdx`,
-- `dtIdx`, `stIdx`, `inIdx`, `outIdx`); `dA_cumsum` is all `0`; the states and the initial states
-- are exact in `bfloat16` (line 360 rounds them).
#guard cases.all fun (c, mt) =>
  let d := c.inst
  let m := CMem.toMem (c.init mt)
  ((List.range c.L).all fun t : Nat => (List.range c.H).all fun h : Nat => (List.range c.P).all fun p : Nat =>
    c.xE t h p == Claims.SsdChunkScan.xIdx d t h p && c.outE t h p == Claims.SsdChunkScan.outIdx d t h p &&
    (m.read .bf16 (d.x.ptr (c.xE t h p : Nat)).addr).map (·.toRat) == some (xv t h p : Rat)) &&
  ((List.range c.L).all fun t : Nat => (List.range c.G).all fun g : Nat => (List.range c.N).all fun n : Nat =>
    c.cE t g n == Claims.SsdChunkScan.cIdx d t g n &&
    (m.read .bf16 (d.C.ptr (c.cE t g n : Nat)).addr).map (·.toRat) == some (cv t g n : Rat)) &&
  ((List.range c.nc).all fun ch : Nat => (List.range c.G).all fun g : Nat =>
    (List.range c.C).all fun i : Nat => (List.range c.C).all fun j : Nat =>
      c.cbE ch g i j == Claims.SsdChunkScan.cbIdx d ch g i j &&
      (m.read .f32 (d.cb.ptr (c.cbE ch g i j : Nat)).addr).map (·.toRat) == some (c.cbv ch g i j : Rat)) &&
  ((List.range c.H).all fun h : Nat => (List.range c.nc).all fun ch : Nat => (List.range c.C).all fun i : Nat =>
    c.dE h ch i == Claims.SsdChunkScan.dtIdx d h ch i &&
    (m.read .f32 (d.dt.ptr (c.dE h ch i : Nat)).addr).map (·.toRat) == some (dtv h ch i) &&
    (m.read .f32 (d.dA.ptr (c.dE h ch i : Nat)).addr).map (·.toRat) == some 0) &&
  ((List.range c.nc).all fun ch : Nat => (List.range c.H).all fun h : Nat =>
    (List.range c.P).all fun p : Nat => (List.range c.N).all fun n : Nat =>
      c.stE ch h p n == Claims.SsdChunkScan.stIdx d ch h p n &&
      (bfv (stv ch h p n)).toRat == (stv ch h p n : Rat)) &&
  ((List.range c.nreq).all fun r : Nat => (List.range c.H).all fun h : Nat =>
    (List.range c.P).all fun p : Nat => (List.range c.N).all fun n : Nat =>
      c.isE r h p n == Claims.SsdChunkScan.inIdx d r h p n &&
      (bfv (isv r h p n)).toRat == (isv r h p n : Rat)) &&
  (List.range c.H).all (fun h : Nat => (m.read .bf16 (d.D.ptr h).addr).map (·.toRat) == some (Dv h : Rat)) &&
  c.outs.length == d.out.len

-- The cases exercise what they claim to: rows entering from an initial state, from
-- `states[chunk]`, and (without initial states) from the zero state; partial logical chunks.
#guard packedInit.fromInit 5 && packedInit.fromInit 11 && !packedInit.fromInit 8 &&
  !packedInit.fromInit 16 && packedInit.lTok 13 == 11 && packedInit.lTok 9 == 8 &&
  longInit.fromInit 70 && longInit.fromInit 127 && !longInit.fromInit 128 && longInit.lTok 129 == 128

-- No expectation equals the sentinel; every expectation is a `bfloat16` (exact).
#guard cases.all fun (c, _) => c.outs.all fun (t, h, p) =>
  c.expect t h p != sentinel && (bfv (c.expect t h p)).toRat == c.expect t h p

def verdictPackedInit : Verdict := packedInit.verdict packedInitMeta
def verdictPackedNoInit : Verdict := packedNoInit.verdict noMeta
def verdictLongInit : Verdict := longInit.verdict longInitMeta

#eval IO.println (verdictPackedInit.line "ssd_chunk_scan_bf16/packed_init")
#eval IO.println (verdictPackedNoInit.line "ssd_chunk_scan_bf16/packed_noinit")
#eval IO.println (verdictLongInit.line "ssd_chunk_scan_bf16/long_init")
-- Agreement resting on two assumed answers per program and two per loop step (`packedInit`:
-- `10·2 + 10·2`; `packedNoInit`: `6·2 + 6·2`; `longInit`: `6·2 + 13·2`).
#guard let v := verdictPackedInit; v == .conditional 40 && v.assumed > 0
#guard let v := verdictPackedNoInit; v == .conditional 24 && v.assumed > 0
#guard let v := verdictLongInit; v == .conditional 38 && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (the cross-request pairs of `cb` summed
-- over the whole physical chunk; the initial states ignored; one element off by `1`).
#guard
  let r := packedInit.run packedInitMeta
  let c := packedInit
  let wholeChunk (t h p : Nat) : Rat :=
    let ch := c.chk t
    let sig : Int := ((List.range c.N).map fun n => cv t 0 n * c.S t h p n).sum
    sig + ((List.range (t + 1 - ch * c.C)).map fun k =>
      ((c.cbv ch 0 (t - ch * c.C) k * xv (ch * c.C + k) h p : Int) : Rat) * dtv h ch k).sum +
      (Dv h * xv t h p : Int)
  let noInit (t h p : Nat) : Rat := if c.fromInit t then c.expect t h p -
      ((List.range c.N).map fun n => cv t 0 n * c.S t h p n).sum else c.expect t h p
  (Verdict.ofLaunch (c.agree packedInitMeta wholeChunk) r).isDiscrepancy &&
  (Verdict.ofLaunch (c.agree packedInitMeta noInit) r).isDiscrepancy &&
  (Verdict.ofLaunch (c.agree packedInitMeta fun t h p =>
    if t = 20 ∧ h = 1 ∧ p = 2 then c.expect t h p + 1 else c.expect t h p) r).isDiscrepancy

end SsdChunkScanTest
