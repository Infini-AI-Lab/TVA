import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `chunk_delta_h_bf16` run by `execLaunch` on three small covered
instances (the claim's `mkF`, `mkV`, `noG`, the host's layouts):
- `fixedG`: `mkF 2 70 1 2 8 40 true true false`: fixed length, `B = 2` rows of `T = 70` tokens (two
  chunks, the second of six tokens), `H = 1` key head and `HV = 2` value heads (grouped), `K = 8`,
  `V = 40` (two value blocks, the second masked past `40`), the gate, an initial state, the final
  state stored, the `[K, V]` layout: eight programs of two chunks;
- `varlenVf`: `mkV [3, 0, 66] 2 2 80 20 .i64 false true true`: variable length, `int64`
  `cu_seqlens`, sequences of lengths `3`, `0` (no chunk: its final state is the zero initial
  state) and `66` (two chunks), `H = HV = 2`, `K = 80` (two state blocks, the second masked past
  `80`), `V = 20`, the gate, no initial state, the final state stored, the `[V, K]` layout
  (`state_v_first`): six programs;
- `noGate`: `noG (mkF 1 64 2 4 16 32 true false false)`: no gate (the delta rule), one row of
  exactly one chunk, `H = 2`, `HV = 4`, `K = 16`, `V = 32`, an initial state, no final state: four
  programs.

Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and that
the final memory is what `Post` says on these inputs. The test runs the recurrence itself, in exact
rationals, for each sequence `n`, value head `hv` and column `j`: from `Y 0` (the initial state, or
`0`), for each chunk `t` the stored state `H t = cvt(Y t)` (`bfloat16`, nearest even: `FloatV.cvt`,
which `Post` names), the new values `v_i = u_i − w_i·H t` of its tokens, their gated values
`vb_i = bf16(2^(g_last − g_i)·v_i)` and `Y (t + 1) = 2^g_last·Y t + Σ_i k_i·vb_i` (the kernel's
arithmetic, each `float32` operation exact on these inputs, which is checked: every `Y t` is a
`float32`). These `Y` are the witness for `Post`'s states, and the test checks, against `Post`:
- every `v_new` element is `bf16(v_i)`, within `tolV` of `vrefS = u − w·h[t]` (`v_i` itself, `h[t]`
  being the stored state): a rounding to nearest moves at most `2^-9·|v_i| ≤ 2^-8·vAbs`;
- every stored state element `h[cbase n + t, hv, c, j]` is `cvt(Y t c)` (`Post`'s exact clause);
- `Y 0` is the initial state; each `Y (t + 1)` is within `errStep t` of `Step t (Y t)`: the
  difference is `kᵀ·δ`, `δ_i = vb_i − 2^(g_last − g_i)·v_i`, and with `‖kᵀ‖₂ ≥ ‖kᵀ·δ‖/‖δ‖` and
  `G_i ≥ 2^(g_last − g_i)·|v_i|` it is enough that `‖δ‖² ≤ 2^-16·Σ_i (2^(g_last − g_i)·v_i)²`, which
  is checked (most `δ_i` are `0`: the gated values are exact `bfloat16`s);
- `ht[n, hv, :, j]` is `Y (nch n)` exactly (when stored);
- every other byte (the inputs, the tables, `chunk_offsets`, sentinels on both sides of each output)
  is unchanged, and no byte is added.
`Vals` holds of these inputs (small integers; gates in `[-5, 0]`, nonincreasing within each chunk,
so `g_last ≤ 0` and `g_last − g ≤ 0`; `padB` is tiny) and `StOk` of the stored states (finite,
norms far below `2^40`), so `Post`'s numeric clauses apply.

**The platform answers are assumed, not proved** (the verdicts are `conditional n`):
- each `tl.dot` (`bfloat16` operands, `float32` accumulator) is answered with the exact
  `acc + a·b`, from the request's own operands, and only when that value is a `float32`. The inputs
  are small integers and the gates powers of two with small exponents, so every product and partial
  sum is a `float32`: the exact value is the one any correct device gives, whatever its order;
- each `tl.exp2` (`float32`, the per-token `[64]` block and the scalar `g_last`) is answered lane by
  lane with the exact power of two, and only when every lane is an integer in `[-126, 127]`. The
  arguments are integer gates and differences of them; the exact power is the correctly rounded
  result (the reference platform's), which `Device.Exp2F32` allows (relative error `2^-22`) but
  does not force: it is what `exp2f` returns at an integer argument. -/

open TVA TVA.Validation

namespace DeltaHTest

open Claims.ChunkDeltaH (Inst)

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

/-! ## The oracle: exact dots, exact powers of two -/

/-- A rational as an accumulator element, when the accumulator is `float32` and holds it exactly. -/
def accOf : (d : DType) → Rat → Option d.Val
  | .f32, z => let v : F32 := FloatV.ofRat z; if v.toRat == z then some v else none
  | _, _ => none

/-- The exact `acc + a·b` of a `tl.dot` request, from its own operands: `none` unless every
operand is finite and every result an exact `float32`. -/
def exactDot : (q : PlatQ) → Option q.Out
  | @PlatQ.dot cfg M N K a b acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => a.get2 i k).toArray.mapM (cfg.a.val? ·)
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => b.get2 k j).toArray.mapM (cfg.b.val? ·)
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM (cfg.acc.val? ·)
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      accOf cfg.acc ((List.range K).foldl (fun s k =>
        let x := A[i * K + k]!
        if x == 0 then s else s + x * B[k * N + j]!) C[e]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- `2^n` as a rational. -/
def pow2 (n : Int) : Rat := if 0 ≤ n then (2 ^ n.toNat : Nat) else 1 / (2 ^ (-n).toNat : Nat)

/-- `exp2` of a `float32` lane: the exact power of two at an integer in `[-126, 127]`. -/
def exp2Lane (x : F32) : Option F32 :=
  if x.isFinite && x.toRat.den == 1 && -126 ≤ x.toRat.num && x.toRat.num ≤ 127 then
    some (FloatV.ofRat (pow2 x.toRat.num))
  else none

/-- The exact `exp2` of a scalar or rank-1 `float32` request (the kernel's shapes). -/
def exactExp2 : (q : PlatQ) → Option q.Out
  | @PlatQ.math .exp2 .binary32 [] x => do
    let r ← exp2Lane (x.get ())
    some (Tensor.scalar r)
  | @PlatQ.math .exp2 .binary32 [n] x => do
    let r ← (List.finRange n).toArray.mapM fun i => exp2Lane (x.get1 i)
    some (Tensor.of1 fun i => r[i.1]!)
  | _ => none

/-- The oracle: exact dots and powers of two; nothing else (trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => (exactDot q).orElse fun _ => exactExp2 q
  | _ => none

/-! ## The cases -/

/-- A sentinel in every output element before the launch, and in the elements just outside each
output. -/
def sentinel : Rat := -999

/-- Sentinel elements on each side of each output. -/
def pad : Nat := 4

def bfv (z : Int) : BF16 := FloatV.ofRat (z : Rat)
def f32v (z : Int) : F32 := FloatV.ofRat (z : Rat)

/-- One case: the claim's instance, the sequences' lengths (as the test reads the batch: `B` rows of
`T` tokens, or the packed sequences), and the inputs as integers by flat element. -/
structure Case where
  d : Inst
  lens : List Nat
  k : Nat → Int
  w : Nat → Int
  u : Nat → Int
  g : Nat → Int
  h0 : Nat → Int

/-- Sequence `n`: its batch row, its first token in the row, its length. -/
def Case.row (c : Case) (n : Nat) : Nat := if c.d.varlen then 0 else n
def Case.start (c : Case) (n : Nat) : Nat := if c.d.varlen then (c.lens.take n).sum else 0
def Case.len (c : Case) (n : Nat) : Nat := c.lens.getD n 0
/-- Its chunks, and its first chunk in `h` (the chunks of the sequences before it; on fixed length
`n·cdiv(T, 64)`, the same). -/
def Case.nch (c : Case) (n : Nat) : Nat := (c.len n + 63) / 64
def Case.cb (c : Case) (n : Nat) : Nat := ((List.range n).map c.nch).sum

def Case.init (c : Case) : CMem :=
  let d := c.d
  let bf (m : CMem) (b : Buf .bf16) (f : Nat → Int) : CMem :=
    CMem.ofElems m .bf16 ((List.range b.len).map fun j : Nat => ((b.ptr j).addr, bfv (f j)))
  let ff (m : CMem) (b : Buf .f32) (f : Nat → Int) : CMem :=
    CMem.ofElems m .f32 ((List.range b.len).map fun j : Nat => ((b.ptr j).addr, f32v (f j)))
  let sb (m : CMem) (b : Buf .bf16) : CMem :=
    CMem.ofElems m .bf16 ((List.range (b.len + 2 * pad)).map fun j : Nat =>
      ((b.ptr ((j : Int) - pad)).addr, (FloatV.ofRat sentinel : BF16)))
  let m := bf ∅ d.k c.k
  let m := bf m d.w c.w
  let m := bf m d.u c.u
  let m := if d.useG then ff m d.g c.g else m
  let m := if d.useH0 then ff m d.h0 c.h0 else m
  let m := sb m d.h
  let m := sb m d.vn
  let m := if d.storeHt then CMem.ofElems m .f32 ((List.range (d.ht.len + 2 * pad)).map fun j : Nat =>
      ((d.ht.ptr ((j : Int) - pad)).addr, (FloatV.ofRat sentinel : F32))) else m
  if d.varlen then
    let m := CMem.ofElems m d.cuD ((List.range (d.N + 1)).map fun i : Nat =>
      ((d.cu.ptr i).addr, d.cuD.ofBits (c.lens.take i).sum))
    CMem.ofElems m .i64 ((List.range (d.N + 1)).map fun i : Nat =>
      ((d.co.ptr i).addr, (IntV.ofNat (c.cb i) : I64)))
  else m

def Case.run (c : Case) : LRes (Claims.ChunkDeltaH.launch c.d).size × List Record :=
  execLaunch ((Claims.ChunkDeltaH.launch c.d).env P0) O fill0 1000000
    (Claims.ChunkDeltaH.model c.d) (Claims.ChunkDeltaH.launch c.d) c.init

/-! ## The recurrence, run by the test -/

/-- `bfloat16` rounding (nearest even) of a value. -/
def rnd (x : Rat) : Rat := (FloatV.ofRat x : BF16).toRat
/-- The stored state's rounding, `cvt` of the `float32` state (`Post`'s `FloatV.cvt`). -/
def cvtS (y : Rat) : Rat := (FloatV.cvt (FloatV.ofRat y : F32) : BF16).toRat
/-- `y` is a `float32` value. -/
def isF32 (y : Rat) : Bool := (FloatV.ofRat y : F32).toRat == y

/-- What the test expects: the elements of `h`, `v_new`, `ht` with their values, and whether the
side conditions hold (every state a `float32`; each step's `δ` small enough; each new value's
rounding within `2^-8` of it). -/
structure Expect where
  hs : List (Nat × Rat) := []
  vs : List (Nat × Rat) := []
  hts : List (Nat × Rat) := []
  ok : Bool := true
  /-- The number of gated values whose `bfloat16` rounding is inexact (`δ_i ≠ 0`). -/
  inexact : Nat := 0

/-- The flat indices of the layouts (`[B, T, ·, ·]` contiguous; `h[c₀, hv, c, j]`, or
`h[c₀, hv, j, c]` with `state_v_first`, the same for `h0`, `ht` with `n` for `c₀`). -/
def Case.kI (c : Case) (n s hk x : Nat) : Nat := ((c.row n * c.d.T + c.start n + s) * c.d.H + hk) * c.d.K + x
def Case.wI (c : Case) (n s hv x : Nat) : Nat := ((c.row n * c.d.T + c.start n + s) * c.d.HV + hv) * c.d.K + x
def Case.uI (c : Case) (n s hv j : Nat) : Nat := ((c.row n * c.d.T + c.start n + s) * c.d.HV + hv) * c.d.V + j
def Case.gI (c : Case) (n s hv : Nat) : Nat := (c.row n * c.d.T + c.start n + s) * c.d.HV + hv
def Case.sI (c : Case) (r hv x j : Nat) : Nat :=
  if c.d.vf then ((r * c.d.HV + hv) * c.d.V + j) * c.d.K + x else ((r * c.d.HV + hv) * c.d.K + x) * c.d.V + j

/-- The recurrence of sequence `n`, value head `hv`, column `j`. -/
def Case.seqExpect (c : Case) (n hv j : Nat) : Expect := Id.run do
  let K := c.d.K
  let hk := hv / (c.d.HV / c.d.H)
  let mut Y : Array Rat := (Array.range K).map fun x => if c.d.useH0 then (c.h0 (c.sI n hv x j) : Rat) else 0
  let mut e : Expect := {}
  for t in List.range (c.nch n) do
    let Ht := Y.map cvtS
    e := { e with hs := e.hs ++ (List.range K).map fun x => (c.sI (c.cb n + t) hv x j, Ht[x]!) }
    let cl := min 64 (c.len n - 64 * t)
    let gl : Int := if c.d.useG then c.g (c.gI n (64 * t + cl - 1) hv) else 0
    let mut Ynew := Y.map (· * pow2 gl)
    let mut dd : Rat := 0
    let mut gg : Rat := 0
    for i in List.range cl do
      let s := 64 * t + i
      let v : Rat := c.u (c.uI n s hv j) - ((List.range K).map fun x => c.w (c.wI n s hv x) * Ht[x]!).sum
      e := { e with vs := e.vs ++ [(c.uI n s hv j, rnd v)], ok := e.ok && |rnd v - v| ≤ |v| / 256 }
      let gv := v * (if c.d.useG then pow2 (gl - c.g (c.gI n s hv)) else 1)
      let vb := rnd gv
      dd := dd + (vb - gv) * (vb - gv)
      gg := gg + gv * gv
      if vb != gv then e := { e with inexact := e.inexact + 1 }
      Ynew := (Array.range K).map fun x => Ynew[x]! + c.k (c.kI n s hk x) * vb
    e := { e with ok := e.ok && dd ≤ gg / 65536 && Ynew.all isF32 }
    Y := Ynew
  if c.d.storeHt then
    e := { e with hts := (List.range K).map fun x => (c.sI n hv x j, Y[x]!) }
  return e

def Case.expect (c : Case) : Expect :=
  ((List.range c.d.N).flatMap fun n => (List.range c.d.HV).flatMap fun hv =>
    (List.range c.d.V).map fun j => c.seqExpect n hv j).foldl
    (fun a e => Expect.mk (a.hs ++ e.hs) (a.vs ++ e.vs) (a.hts ++ e.hts) (a.ok && e.ok)
      (a.inexact + e.inexact)) {}

/-- The final memory against `Post`: every element of `h`, `v_new`, `ht` is its expectation;
every byte outside them keeps its initial value, and no byte is added. -/
def Case.agree (c : Case) (e : Expect) (cf : CConfig (Claims.ChunkDeltaH.launch c.d).size) :
    Option String :=
  let d := c.d
  let m := CMem.toMem cf.mem
  let badH := e.hs.filter fun (i, x) => (m.read .bf16 (d.h.ptr i).addr).map (·.toRat) != some x
  let badV := e.vs.filter fun (i, x) => (m.read .bf16 (d.vn.ptr i).addr).map (·.toRat) != some x
  let badT := e.hts.filter fun (i, x) => (m.read .f32 (d.ht.ptr i).addr).map (·.toRat) != some x
  let addB {dt : DType} (b : Buf dt) (s : Std.HashSet Int) (l : List (Nat × Rat)) : Std.HashSet Int :=
    l.foldl (fun s (i, _) => (List.range dt.bytes).foldl (fun s o => s.insert ((b.ptr i).addr + o)) s) s
  let wBytes := addB d.ht (addB d.vn (addB d.h {} e.hs) e.vs) e.hts
  let init := c.init
  let badElse := init.toList.filter fun (a, b) => !wBytes.contains a && cf.mem[a]? != some b
  if !badH.isEmpty then some s!"h elements {(badH.take 8).map (·.1)} differ from Post"
  else if !badV.isEmpty then some s!"v_new elements {(badV.take 8).map (·.1)} differ from Post"
  else if !badT.isEmpty then some s!"ht elements {(badT.take 8).map (·.1)} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside h, v_new, ht changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

/-! ## The instances -/

/-- A small signed integer from an index, scrambled (no short period along any axis): in
`[-2, 2]`, or in `[-1, 1]`. -/
def mix (a b j : Nat) : Int := ((((a * j + b) * 2654435761 / 65536) % 5 : Nat) : Int) - 2
def mix3 (a b j : Nat) : Int := ((((a * j + b) * 2654435761 / 65536) % 3 : Nat) : Int) - 1

def fixedG : Case :=
  { d := Claims.ChunkDeltaH.mkF 2 70 1 2 8 40 true true false, lens := [70, 70],
    k := mix3 7 2, w := mix3 3 1, u := mix 2 3, h0 := mix 5 4,
    -- `g[b, t, hv] = -⌊(t mod 64)·(hv + 1)/24⌋`: within each chunk nonincreasing, in `[-5, 0]`
    g := fun i => -((((i / 2) % 70 % 64) * (i % 2 + 1)) / 24 : Nat) }

def varlenVf : Case :=
  { d := Claims.ChunkDeltaH.mkV [3, 0, 66] 2 2 80 20 .i64 false true true, lens := [3, 0, 66],
    k := mix3 7 2, w := mix3 3 1, u := mix 2 3, h0 := fun _ => 0,
    -- `g[tok, hv]` from the token's position `p` in its sequence (sequences at `0` and `3`):
    -- `-⌊(p mod 64)·(hv + 2)/40⌋`, in `[-4, 0]`
    g := fun i => -((((if i / 2 < 3 then i / 2 else i / 2 - 3) % 64) * (i % 2 + 2)) / 40 : Nat) }

def noGate : Case :=
  { d := Claims.ChunkDeltaH.noG (Claims.ChunkDeltaH.mkF 1 64 2 4 16 32 true false false),
    lens := [64], k := mix3 7 2, w := mix3 3 1, u := mix 2 3, h0 := mix 5 4, g := fun _ => 0 }

open Claims.ChunkDeltaH in
theorem fixedG_ok : fixedG.d.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
open Claims.ChunkDeltaH in
theorem varlenVf_ok : varlenVf.d.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)
open Claims.ChunkDeltaH in
theorem noGate_ok : noGate.d.ok :=
  ok_of _ (by decide) (by decide) (by decide) ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩ (by decide)

/-! ## `Pre`, and the inputs, checked -/

def validIn {d : DType} (b : Buf d) (m : CMem) : Bool :=
  0 ≤ b.base && b.base + b.len * d.bytes ≤ 2 ^ 64 && b.base % d.bytes == 0 &&
    (List.range (b.len * d.bytes)).all fun i : Nat => m.contains (b.base + i)

/-- `Pre`: the buffers valid (`g`, `h0`, `ht` when used); on variable length `cu_seqlens` holding
the instance's entries and `chunk_offsets[n] = cbase n` (`Pre` names the claim's `cbase`; it is
also the test's own count). -/
def Case.pre (c : Case) : Bool :=
  let d := c.d
  let m := c.init
  let mm := CMem.toMem m
  validIn d.k m && validIn d.w m && validIn d.u m && (!d.useG || validIn d.g m) && validIn d.h m &&
    validIn d.vn m && (!d.useH0 || validIn d.h0 m) && (!d.storeHt || validIn d.ht m) &&
    (!d.varlen || (validIn d.cu m && validIn d.co m &&
      (List.range d.cu.len).all (fun i : Nat =>
        (mm.read d.cuD (d.cu.ptr i).addr).map d.cuD.toBits == some (d.cuD.toBits (d.cuVal i))) &&
      (List.range (d.N + 1)).all fun i : Nat =>
        mm.read .i64 (d.co.ptr i).addr == some (IntV.ofNat (Claims.ChunkDeltaH.cbase d i)) &&
          Claims.ChunkDeltaH.cbase d i == c.cb i))

#guard [fixedG, varlenVf, noGate].all Case.pre

-- The test's reading of the batch is the instance's: `N` sequences whose lengths sum to the
-- tokens, and `h` holding their chunks.
#guard [fixedG, varlenVf, noGate].all fun c =>
  c.lens.length == c.d.N && (c.d.varlen || c.lens.all (· == c.d.T)) &&
  (!c.d.varlen || c.lens.sum == c.d.T) && c.cb c.d.N * c.d.HV * c.d.K * c.d.V == c.d.h.len

-- The inputs are signed, encoded exactly; `Vals`' bounds hold (`|·| ≤ 2^16`, gates `≤ 0`,
-- nonincreasing within each chunk: `g_last − g ≤ 0`).
#guard [fixedG, varlenVf, noGate].all fun c => (List.range 12000).all fun i : Nat =>
  (bfv (c.k i)).toRat == (c.k i : Rat) && (bfv (c.w i)).toRat == (c.w i : Rat) &&
  (bfv (c.u i)).toRat == (c.u i : Rat) && (f32v (c.g i)).toRat == (c.g i : Rat) &&
  (f32v (c.h0 i)).toRat == (c.h0 i : Rat) && c.g i ≤ 0
#guard (List.range 140).any (fun i => fixedG.k i < 0) && (List.range 140).any (fun i => fixedG.u i < 0)
#guard [fixedG, varlenVf].all fun c => (List.range c.d.N).all fun n => (List.range c.d.HV).all fun hv =>
  (List.range (c.len n)).all fun s => s % 64 == 63 || s + 1 == c.len n ||
    c.g (c.gI n (s + 1) hv) ≤ c.g (c.gI n s hv)
#guard (List.range 140).any fun i => fixedG.g i ≤ -4

/-- The expectations, computed once per command. -/
def expFixed : Expect := fixedG.expect
def expVarlen : Expect := varlenVf.expect
def expNoGate : Expect := noGate.expect

-- Every element of `h`, `v_new`, `ht` is expected once; the side conditions hold; the states and
-- new values are not trivial (some stored states rounded, some fractions from the gates); no
-- expectation equals the sentinel.
#guard [(fixedG, expFixed), (varlenVf, expVarlen), (noGate, expNoGate)].all fun (c, e) =>
  e.ok && (e.hs.map (·.1)).mergeSort == List.range c.d.h.len &&
  (e.vs.map (·.1)).mergeSort == List.range c.d.vn.len &&
  (e.hts.map (·.1)).mergeSort == (if c.d.storeHt then List.range c.d.ht.len else []) &&
  (e.hs ++ e.vs ++ e.hts).all (·.2 != sentinel)
#guard expFixed.hts.any (fun (_, y) => cvtS y != y) && expVarlen.hts.any (fun (_, y) => y.den > 1)
-- Some gated values are inexact `bfloat16`s (`δ_i ≠ 0`: `fixedG` 93, `varlenVf` 4), so the step
-- bound is exercised, not only met with `0`.
#guard expFixed.inexact > 0 && expVarlen.inexact > 0

/-! ## The runs -/

def verdictFixed : Verdict := Verdict.ofLaunch (fixedG.agree expFixed) fixedG.run
def verdictVarlen : Verdict := Verdict.ofLaunch (varlenVf.agree expVarlen) varlenVf.run
def verdictNoGate : Verdict := Verdict.ofLaunch (noGate.agree expNoGate) noGate.run

#eval IO.println (verdictFixed.line "chunk_delta_h_bf16/fixed_g")
#eval IO.println (verdictVarlen.line "chunk_delta_h_bf16/varlen_vf")
#eval IO.println (verdictNoGate.line "chunk_delta_h_bf16/no_gate")

-- Agreement resting on assumed answers: per chunk, one `w` dot and one `k` dot per state block,
-- and with the gate two `exp2`s (`fixedG`: 8 programs × 2 chunks × (1 + 1 + 2); `varlenVf`:
-- 2 heads × 3 chunks × (2 + 2 + 2); `noGate`: 4 programs × 1 chunk × (1 + 1)).
#guard verdictFixed == .conditional 64
#guard verdictVarlen == .conditional 36
#guard verdictNoGate == .conditional 8

-- A wrong expectation is caught: `agree` is not vacuous (the gate dropped; the initial state
-- dropped; one stored state element off).
#guard
  let r := noGate.run
  (Verdict.ofLaunch (noGate.agree { noGate with d := { noGate.d with useH0 := false } }.expect) r).isDiscrepancy &&
  (Verdict.ofLaunch (noGate.agree { expNoGate with
    hs := expNoGate.hs.map fun (i, x) => if i = 5 then (i, x + 1) else (i, x) }) r).isDiscrepancy
#guard (Verdict.ofLaunch (fixedG.agree { fixedG with d := { fixedG.d with useG := false } }.expect)
  fixedG.run).isDiscrepancy

end DeltaHTest
