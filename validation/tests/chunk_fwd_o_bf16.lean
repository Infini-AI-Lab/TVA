import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `chunk_fwd_o_bf16` run by `execLaunch` on three small covered
instances, covering the three gate kinds, both state layouts, fixed and variable length, grouped
value heads, ragged chunks, several `k` steps and value blocks, and the three configurations
`BK = BV ∈ {32, 64, 128}`:
- `gated`: fixed length, `B = 2` rows of `T = 5` tokens, `H = 1` query/key head and `HV = 2` value
  heads (grouped), `K = 4`, `V = 3`, the gates `g` (nonincreasing nonpositive integers), the state
  `[K, V]`, `BK = BV = 32`, scale `1/8`: four programs;
- `gammaVarlen`: variable length, `int32` tables, sequences of lengths `3, 0, 4` in `T = 8` tokens
  (the last token in no sequence), `chunk_indices` in a permuted order, `H = 1`, `HV = 2`, `K = 5`,
  `V = 4`, the decays `g_gamma = [-1, 1]`, the state `[V, K]` (`state_v_first`),
  `BK = BV = 64`, scale `1/16`: four programs;
- `multi`: fixed length, one row of `T = 66` tokens (two chunks, the second of two tokens),
  `H = HV = 2`, `K = 40` (two `k` steps of `32`, the second masked past `40`), `V = 33` (two value
  blocks, the second masked past `33`), no gate, the state `[K, V]`, `BK = BV = 32`, scale `1/8`:
  eight programs.
(`BK = BV = 128` is run on `gated`'s shapes as `gated128`.)

Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and that
the final memory is what `Post` says on these inputs. Each written element `o[t, j, e]` is compared
with the `bfloat16` rounding (nearest even) of the exact value `ref` (`Post`'s
`scale·(gate·Σ_x q·S + Σ_{s ≤ t} (q[t]·k[s])·wgt·v[s])`), computed here from the inputs as
rationals and from the chunks as `cu_seqlens` (or the rows) and `chunk_indices` name them; that
value is within `Post`'s tolerance (`tol ≥ 2^-8·|ref|`, and a rounding to nearest moves at most
`2^-9·|ref|`, checked below). Every other byte (the inputs, the tables, the uncovered token's
outputs, sentinels on both sides of `o`) is unchanged, and no byte is added.

**The platform answers are assumed, not proved** (the verdicts are `conditional n`):
- each `tl.dot` (`bfloat16` operands, `float32` accumulator) is answered with the exact
  `acc + a·b`, from the request's own operands, and only when that value is a `float32`. The
  inputs are small integers and the gates powers of two with small exponents, so every product and
  partial sum is a `float32` (a multiple of `2^-6` below `2^15`): the exact value is
  the one any correct device gives, whatever its order;
- each `tl.exp2` (`float32`) is answered lane by lane with the exact power of two, and only when
  every lane is an integer in `[-126, 127]`. The arguments are integers here (the gates, their
  differences, `γ·(i + 1)` and `γ·(i − i')` with `γ = ±1`); the exact power is the correctly rounded
  result (the reference platform's), which `Device.Exp2F32` allows (relative error `2^-22`) but does
  not force: it is what `exp2f` returns at an integer argument.
With these answers every `float32` operation of the model is exact, so the value before the
store's rounding is `ref` exactly; the `bfloat16` conversion of the scores (line 180) is exact too
(the scores are integers below `256` times powers of two). -/

open TVA TVA.Validation

namespace FwdOTest

open Claims.ChunkFwdO (Inst)

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

/-- The exact `exp2` of a rank-1 or rank-2 `float32` request (the kernel's shapes). -/
def exactExp2 : (q : PlatQ) → Option q.Out
  | @PlatQ.math .exp2 .binary32 [n] x => do
    let r ← (List.finRange n).toArray.mapM fun i => exp2Lane (x.get1 i)
    some (Tensor.of1 fun i => r[i.1]!)
  | @PlatQ.math .exp2 .binary32 [m, n] x => do
    let r ← (Array.range (m * n)).mapM fun e =>
      if h : e / n < m ∧ e % n < n then exp2Lane (x.get2 ⟨e / n, h.1⟩ ⟨e % n, h.2⟩) else none
    some (Tensor.of2 fun i j => r[i.1 * n + j.1]!)
  | _ => none

/-- The oracle: exact dots and powers of two; nothing else (trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => (exactDot q).orElse fun _ => exactExp2 q
  | _ => none

/-! ## The cases -/

/-- A sentinel in every output element before the launch, and in the elements just outside `o`. -/
def sentinel : Rat := -999

/-- Sentinel elements on each side of `o`. -/
def pad : Nat := 4

def ival (d : DType) (z : Int) : d.Val := d.ofBits (z % (2 ^ (8 * d.bytes) : Int)).toNat
def bfv (z : Int) : BF16 := FloatV.ofRat (z : Rat)
def f32v (z : Int) : F32 := FloatV.ofRat (z : Rat)

/-- One case: the claim's `mk` arguments, the scale's value (checked against its pattern), the
sequences' lengths and the rows of `chunk_indices` (variable length), and the contents as
integers by flat element (`g` at `t·HV + j`, `g_gamma` at `j`). -/
structure Case where
  vl : Bool
  useG : Bool
  useGm : Bool
  svf : Bool
  B : Nat
  T : Nat
  H : Nat
  HV : Nat
  K : Nat
  V : Nat
  sbits : Nat
  sigma : Rat
  dt : DType
  N : Nat
  NT : Nat
  bk : Nat
  lens : List Nat
  rows : List (Nat × Nat)
  q : Nat → Int
  k : Nat → Int
  v : Nat → Int
  h : Nat → Int
  g : Nat → Int
  gam : Nat → Int

def Case.base (c : Case) : Inst :=
  Claims.ChunkFwdO.mk c.vl c.useG c.useGm c.svf c.B c.T c.H c.HV c.K c.V c.sbits c.dt c.N c.NT c.bk

def Case.cu (c : Case) (i : Nat) : Nat := (c.lens.take i).sum
def Case.ci (c : Case) (i : Nat) : Nat :=
  let r := c.rows.getD (i / 2) (0, 0)
  if i % 2 = 0 then r.1 else r.2

/-- The claim's instance, with the case's contents (as `Pre` requires the buffers to hold). -/
def Case.inst (c : Case) : Inst :=
  { c.base with
    qv := fun i => bfv (c.q i), kv := fun i => bfv (c.k i), vv := fun i => bfv (c.v i),
    hv := fun i => bfv (c.h i), gv := fun i => f32v (c.g i), ggv := fun i => f32v (c.gam i),
    cuVal := fun i => ival c.dt (c.cu i), ciVal := fun i => ival c.dt (c.ci i) }

/-- `Inst.ok` does not read the contents: the case's instance is covered when `mk`'s is. -/
theorem Case.ok_of (c : Case) (h : c.base.ok) : c.inst.ok := by
  cases h; constructor <;> assumption

/-- The number of chunk states `h` holds. -/
def Case.nch (c : Case) : Nat := if c.vl then c.NT else c.B * ((c.T + 63) / 64)

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let bf (m : CMem) (b : Buf .bf16) (f : Nat → Int) : CMem :=
    CMem.ofElems m .bf16 ((List.range b.len).map fun j : Nat => ((b.ptr j).addr, bfv (f j)))
  let m := bf ∅ d.q c.q
  let m := bf m d.k c.k
  let m := bf m d.v c.v
  let m := bf m d.h c.h
  let m := CMem.ofElems m .bf16 ((List.range (d.o.len + 2 * pad)).map fun j : Nat =>
    ((d.o.ptr ((j : Int) - pad)).addr, (FloatV.ofRat sentinel : BF16)))
  let m := if c.useG then CMem.ofElems m .f32 ((List.range d.g.len).map fun j : Nat =>
    ((d.g.ptr j).addr, f32v (c.g j))) else m
  let m := if c.useGm then CMem.ofElems m .f32 ((List.range d.gg.len).map fun j : Nat =>
    ((d.gg.ptr j).addr, f32v (c.gam j))) else m
  if c.vl then
    let m := CMem.ofElems m c.dt ((List.range (c.N + 1)).map fun i : Nat =>
      ((d.cu.ptr i).addr, ival c.dt (c.cu i)))
    CMem.ofElems m c.dt ((List.range (2 * c.NT)).map fun i : Nat =>
      ((d.ci.ptr i).addr, ival c.dt (c.ci i)))
  else m

def Case.run (c : Case) : LRes (Claims.ChunkFwdO.launch c.inst).size × List Record :=
  execLaunch ((Claims.ChunkFwdO.launch c.inst).env P0) O fill0 1000000
    (Claims.ChunkFwdO.model c.inst) (Claims.ChunkFwdO.launch c.inst) c.init

/-! ## What `Post` says, computed from the inputs -/

/-- The chunks: `(r, lo, hi)`, chunk `r`'s state in `h` and its tokens `[lo, hi)` of the flat token
axis. Fixed length: chunk `c` of row `b` is `r = b·NT + c`, tokens from `b·T + 64c`; variable
length: row `r` of `chunk_indices` names chunk `c` of sequence `n`, tokens from `cu[n] + 64c`, up
to the sequence's end. -/
def Case.chunks (c : Case) : List (Nat × Nat × Nat) :=
  if c.vl then (List.range c.rows.length).map fun r =>
    let (n, ch) := c.rows.getD r (0, 0)
    let lo := c.cu n + 64 * ch
    (r, lo, min (lo + 64) (c.cu (n + 1)))
  else
    let nt := (c.T + 63) / 64
    (List.range c.B).flatMap fun b => (List.range nt).map fun ch =>
      let lo := b * c.T + 64 * ch
      (b * nt + ch, lo, min (lo + 64) (b * c.T + c.T))

/-- The written elements with their exact values `ref`: for each chunk `(r, lo, hi)`, token
`t ∈ [lo, hi)`, value head `j` (query/key head `j / (HV / H)`) and column `e`. -/
def Case.refs (c : Case) : List (Nat × Rat) :=
  c.chunks.flatMap fun (r, lo, hi) =>
    (List.range' lo (hi - lo)).flatMap fun t => (List.range c.HV).flatMap fun j =>
      let hq := j / (c.HV / c.H)
      let qx (x : Nat) : Int := c.q ((t * c.H + hq) * c.K + x)
      -- the scores `q[t]·k[s]`, for `s ∈ [lo, t]`
      let A : Array Int := (List.range' lo (t + 1 - lo)).toArray.map fun s =>
        ((List.range c.K).map fun x => qx x * c.k ((s * c.H + hq) * c.K + x)).sum
      let gate : Rat :=
        if c.useG then pow2 (c.g (t * c.HV + j))
        else if c.useGm then pow2 (c.gam j * ((t - lo : Nat) + 1)) else 1
      let wgt (s : Nat) : Rat :=
        if c.useG then pow2 (c.g (t * c.HV + j) - c.g (s * c.HV + j))
        else if c.useGm then pow2 (c.gam j * ((t : Int) - s)) else 1
      (List.range c.V).map fun e =>
        let S (x : Nat) : Int :=
          c.h (if c.svf then ((r * c.HV + j) * c.V + e) * c.K + x else ((r * c.HV + j) * c.K + x) * c.V + e)
        let QS : Int := ((List.range c.K).map fun x => qx x * S x).sum
        let intra : Rat := ((List.range' lo (t + 1 - lo)).map fun s =>
          (A[s - lo]! : Rat) * wgt s * c.v ((s * c.HV + j) * c.V + e)).sum
        ((t * c.HV + j) * c.V + e, c.sigma * (gate * QS + intra))

/-- The value `o[t, j, e]` must hold: `ref`, rounded to `bfloat16` (nearest even). -/
def rnd (x : Rat) : Rat := (FloatV.ofRat x : BF16).toRat

/-- The final memory against `Post`: every written element is its expectation; every byte outside
the written elements keeps its initial value, and no byte is added. -/
def Case.agree (c : Case) (outs : List (Nat × Rat))
    (cf : CConfig (Claims.ChunkFwdO.launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let badOut := outs.filter fun (j, x) => (m.read .bf16 (d.o.ptr j).addr).map (·.toRat) != some x
  let wBytes : Std.HashSet Int := outs.foldl (fun s (j, _) =>
    (s.insert (d.o.ptr j).addr).insert ((d.o.ptr j).addr + 1)) {}
  let init := c.init
  let badElse := init.toList.filter fun (a, b) => !wBytes.contains a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"o elements {(badOut.take 8).map (·.1)} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside the written elements changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def Case.outs (c : Case) : List (Nat × Rat) := c.refs.map fun (j, x) => (j, rnd x)

/-! ## The instances -/

/-- A small signed integer from an index (`[-2, 2]`), scrambled (no short period along any axis). -/
def mix (a b j : Nat) : Int := ((((a * j + b) * 2654435761 / 65536) % 5 : Nat) : Int) - 2

/-- `1/8`, `1/16`: the doubles `0x3fc0000000000000`, `0x3fb0000000000000` (`sc64`, `sc256`). -/
def gated : Case :=
  { vl := false, useG := true, useGm := false, svf := false, B := 2, T := 5, H := 1, HV := 2,
    K := 4, V := 3, sbits := Claims.ChunkFwdO.sc64, sigma := 1 / 8, dt := .i32, N := 0, NT := 0,
    bk := 32, lens := [], rows := [],
    q := mix 3 1, k := mix 7 2, v := mix 2 3, h := mix 3 4,
    -- `g[b·T + t, j] = -⌊(t·(j+1) + b)/2⌋`: nonincreasing in `t`, nonpositive
    g := fun i => -(((i / 2 % 5) * (i % 2 + 1) + i / 10) / 2 : Nat), gam := fun _ => 0 }

/-- `gated` with `BK = BV = 128`. -/
def gated128 : Case := { gated with bk := 128 }

def gammaVarlen : Case :=
  { vl := true, useG := false, useGm := true, svf := true, B := 1, T := 8, H := 1, HV := 2,
    K := 5, V := 4, sbits := Claims.ChunkFwdO.sc256, sigma := 1 / 16, dt := .i32, N := 3, NT := 2,
    bk := 64, lens := [3, 0, 4], rows := [(2, 0), (0, 0)],
    q := mix 2 1, k := mix 3 3, v := mix 4 1, h := mix 7 2, g := fun _ => 0,
    gam := fun j => if j = 0 then -1 else 1 }

def multi : Case :=
  { vl := false, useG := false, useGm := false, svf := false, B := 1, T := 66, H := 2, HV := 2,
    K := 40, V := 33, sbits := Claims.ChunkFwdO.sc64, sigma := 1 / 8, dt := .i32, N := 0, NT := 0,
    bk := 32, lens := [], rows := [],
    q := mix 3 1, k := mix 7 2, v := mix 2 3, h := mix 3 4,
    g := fun _ => 0, gam := fun _ => 0 }

theorem gated_ok : gated.inst.ok := Case.ok_of _ <| by
  dsimp only [Case.base, gated]
  exact Claims.ChunkFwdO.mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num [Claims.ChunkFwdO.BT]) (by simp)
    (by decide) (by decide) (by norm_num) (by decide) (by norm_num) (by norm_num) (by norm_num)
    (by decide) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) Claims.ChunkFwdO.sc64_ok

theorem gated128_ok : gated128.inst.ok := Case.ok_of _ <| by
  dsimp only [Case.base, gated128, gated]
  exact Claims.ChunkFwdO.mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num [Claims.ChunkFwdO.BT]) (by simp)
    (by decide) (by decide) (by norm_num) (by decide) (by norm_num) (by norm_num) (by norm_num)
    (by decide) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) Claims.ChunkFwdO.sc64_ok

theorem gammaVarlen_ok : gammaVarlen.inst.ok := Case.ok_of _ <| by
  dsimp only [Case.base, gammaVarlen]
  exact Claims.ChunkFwdO.mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num [Claims.ChunkFwdO.BT]) (by simp)
    (by decide) (by decide) (by norm_num) (by decide) (by norm_num) (by norm_num) (by norm_num)
    (by decide) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) Claims.ChunkFwdO.sc256_ok

theorem multi_ok : multi.inst.ok := Case.ok_of _ <| by
  dsimp only [Case.base, multi]
  exact Claims.ChunkFwdO.mk_ok (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num [Claims.ChunkFwdO.BT]) (by simp)
    (by decide) (by decide) (by norm_num) (by decide) (by norm_num) (by norm_num) (by norm_num)
    (by decide) (Or.inl rfl) (by norm_num) (by norm_num) (by norm_num) Claims.ChunkFwdO.sc64_ok

/-! ## `Pre`, and the inputs, checked -/

def validIn {d : DType} (b : Buf d) (m : CMem) : Bool :=
  0 ≤ b.base && b.base + b.len * d.bytes ≤ 2 ^ 64 && b.base % d.bytes == 0 &&
    (List.range (b.len * d.bytes)).all fun i : Nat => m.contains (b.base + i)

def holdsIn {d : DType} (b : Buf d) (m : CMem) (v : Nat → d.Val) : Bool :=
  (List.range b.len).all fun j : Nat =>
    ((CMem.toMem m).read d (b.ptr j).addr).map d.toBits == some (d.toBits (v j))

/-- `TablesOk`'s parts, decided (through the claim's `cuAt`, `seqAt`, `chkAt`, which `Pre`
names). -/
def tablesOk (c : Case) : Bool :=
  let d := c.inst
  open Claims.ChunkFwdO in
  0 ≤ cuAt d 0 && (List.range c.N).all (fun i => cuAt d i ≤ cuAt d (i + 1)) && cuAt d c.N ≤ c.T &&
  (List.range c.NT).all (fun r => 0 ≤ seqAt d r && seqAt d r < c.N && 0 ≤ chkAt d r &&
    chkAt d r * 64 < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat) &&
  (List.range c.NT).all (fun r => (List.range c.NT).all fun r' =>
    r == r' || (seqAt d r, chkAt d r) != (seqAt d r', chkAt d r')) &&
  (List.range c.N).all (fun n => (List.range (c.T + 1)).all fun ch =>
    !((ch * 64 : Int) < cuAt d (n + 1) - cuAt d n) ||
      (List.range c.NT).any fun r => seqAt d r == n && chkAt d r == ch)

def Case.pre (c : Case) : Bool :=
  let d := c.inst
  let m := c.init
  validIn d.q m && validIn d.k m && validIn d.v m && validIn d.h m && validIn d.o m &&
    holdsIn d.q m d.qv && holdsIn d.k m d.kv && holdsIn d.v m d.vv && holdsIn d.h m d.hv &&
    (!c.useG || (validIn d.g m && holdsIn d.g m d.gv)) &&
    (!c.useGm || (validIn d.gg m && holdsIn d.gg m d.ggv)) &&
    (!c.vl || (validIn d.cu m && validIn d.ci m && holdsIn d.cu m d.cuVal &&
      holdsIn d.ci m d.ciVal && tablesOk c))

#guard [gated, gammaVarlen, multi].all Case.pre

-- The case's bookkeeping: `N`, `NT` are the tables' sizes; `h` holds one state per chunk.
#guard gammaVarlen.N == gammaVarlen.lens.length && gammaVarlen.NT == gammaVarlen.rows.length
#guard [gated, gammaVarlen, multi].all fun c => c.chunks.length == c.nch

-- The scales' patterns are the values the expectations use (the double, and the `float32` the
-- kernel receives).
#guard [gated, gammaVarlen, multi].all fun c =>
  c.inst.scale.toRat == c.sigma && (FloatV.hostF32 c.inst.scale).toRat == c.sigma

-- The gates: `g` nonincreasing along each row, nonpositive (`InRange`: `g[t] − g[s] ≤ 64`,
-- `g[t] ≤ 64`); the decays `γ ∈ {-1, 1}` (`|γ| ≤ 64`, `γ ≤ 1`).
#guard (List.range 2).all fun b => (List.range 4).all fun t => (List.range 2).all fun j =>
  gated.g (((b * 5 + t + 1) * 2) + j) ≤ gated.g ((b * 5 + t) * 2 + j) && gated.g ((b * 5 + t) * 2 + j) ≤ 0
#guard (List.range 10).any fun i => gated.g i < -1

-- The inputs are signed, and every element is encoded exactly.
#guard (List.range 40).any (fun j => multi.q j < 0) && (List.range 40).any (fun j => gated.v j < 0)
#guard [gated, gammaVarlen, multi].all fun c => (List.range 6000).all fun j : Nat =>
  (bfv (c.q j)).toRat == (c.q j : Rat) && (bfv (c.k j)).toRat == (c.k j : Rat) &&
  (bfv (c.v j)).toRat == (c.v j : Rat) && (bfv (c.h j)).toRat == (c.h j : Rat) &&
  (f32v (c.g j)).toRat == (c.g j : Rat) && (f32v (c.gam j)).toRat == (c.gam j : Rat)

-- Every output element of a covered token is written once (all of `o` on fixed length; all but
-- the uncovered token's on variable length), the expectations are within `Post`'s `2^-8·|ref|`
-- of `ref`, and none equals the sentinel.
#guard [gated, multi].all fun c => (c.refs.map (·.1)).mergeSort == List.range (c.B * c.T * c.HV * c.V)
#guard (gammaVarlen.refs.map (·.1)).mergeSort == List.range (7 * 2 * 4)
#guard [gated, gammaVarlen, multi].all fun c => c.refs.all fun (_, x) =>
  |rnd x - x| ≤ |x| / 256 && rnd x != sentinel

-- The expectations are not trivial: some are fractions (the gates), some rounded (`multi`).
#guard gated.refs.any (fun (_, x) => x.den > 8) && gammaVarlen.refs.any (fun (_, x) => x.den > 16)
#guard multi.refs.any fun (_, x) => rnd x != x

/-! ## The runs -/

def verdictGated : Verdict := Verdict.ofLaunch (gated.agree gated.outs) gated.run
def verdictGated128 : Verdict := Verdict.ofLaunch (gated128.agree gated128.outs) gated128.run
def verdictGamma : Verdict := Verdict.ofLaunch (gammaVarlen.agree gammaVarlen.outs) gammaVarlen.run
def verdictMulti : Verdict := Verdict.ofLaunch (multi.agree multi.outs) multi.run

#eval IO.println (verdictGated.line "chunk_fwd_o_bf16/gated")
#eval IO.println (verdictGated128.line "chunk_fwd_o_bf16/gated128")
#eval IO.println (verdictGamma.line "chunk_fwd_o_bf16/gamma_varlen")
#eval IO.println (verdictMulti.line "chunk_fwd_o_bf16/multi")

-- Agreement resting on assumed answers: per program, two dots per `k` step, one final dot, and
-- with a gate two `exp2`s (`gated`: 4 programs × (2 + 1 + 2); `gammaVarlen`: 4 × (2 + 1 + 2);
-- `multi`: 8 × (2·2 + 1)).
#guard verdictGated == .conditional 20
#guard verdictGated128 == .conditional 20
#guard verdictGamma == .conditional 20
#guard verdictMulti == .conditional 40

-- A wrong expectation is caught: `agree` is not vacuous (the gates dropped; one element off by
-- one unit in the last place of `bfloat16`; the state's layout transposed).
#guard
  let r := gated.run
  (Verdict.ofLaunch (gated.agree ({ gated with useG := false }.refs.map fun (j, x) => (j, rnd x))) r).isDiscrepancy &&
  (Verdict.ofLaunch (gated.agree (gated.outs.map fun (j, x) =>
    if j = 7 then (j, x + |x| / 128 + 1 / 1024) else (j, x))) r).isDiscrepancy
#guard (Verdict.ofLaunch (gammaVarlen.agree ({ gammaVarlen with svf := false }.refs.map
  fun (j, x) => (j, rnd x))) gammaVarlen.run).isDiscrepancy

end FwdOTest
