import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `chunk_bwd_dv_local_bf16` run by `execLaunch` on three small covered
instances (`canon`, the host's layout, with the host's `BK`, `BV`):
- `gated`: fixed length, `B = 2` rows of `T = 20` tokens, `chunk_size = 16` (two chunks per row,
  the second of four tokens), `H = 1` query/key head and `HV = 2` value heads (grouped), `K = 5`,
  `V = 6` (`BK = BV = 16`), the gate `g` (nonincreasing nonpositive integers), scale `1/4`: eight
  programs;
- `varlen`: variable length, `int32` tables, sequences of lengths `5, 0, 33` in `T = 40` tokens (the
  last two in no sequence), `chunk_size = 32` (the long sequence's second chunk has one token),
  `chunk_indices` in a permuted order, `H = 2`, `HV = 4`, `K = 130` (`BK = 128`: two `k` steps, the
  second masked past `130`), `V = 20` (`BV = 32`), no gate, scale `1/8`: twelve programs;
- `wideV`: fixed length, one row of `T = 70` tokens, `chunk_size = 64` (two chunks), `H = HV = 1`,
  `K = 16`, `V = 130` (`BV = 128`: two value blocks, the second masked past `130`), the gate, scale
  `1/4`: two programs.

Not a test against the device (phase 4): it checks that `exec` runs the model end to end, and that
the final memory is what `Post` says on these inputs. Each written element `dv[p, hv, e]` is
compared with the `bfloat16` rounding (nearest even) of the exact `ref` of `Post`
(`Σ_{j = p}^{end(p) − 1} σ·(Σ_c k[p, h, c]·q[j, h, c])·2^{g[j] − g[p]}·do[j, hv, e]`, chunks of
`chunk_size` counted from each sequence's start), computed here from the inputs as rationals; that
value is within `Post`'s tolerance (`tol ≥ 2^-7·Σ_j |A·do| ≥ 2^-7·|ref|`, and a rounding to nearest
moves at most `2^-9·|ref|`, checked below). Every other byte (the inputs, the tables, the uncovered
tokens' outputs, sentinels on both sides of `dv`) is unchanged, and no byte is added.

**The platform answers are assumed, not proved** (the verdicts are `conditional n`):
- each `tl.dot` (`bfloat16` operands, `float32` accumulator) is answered with the exact
  `acc + a·b`, from the request's own operands, and only when that value is a `float32`. The inputs
  are small integers and the gates powers of two with small exponents, so every product and partial
  sum is a `float32`: the exact value is the one any correct device gives, whatever its order;
- each `tl.exp2` (`float32`) is answered lane by lane with the exact power of two, and only when
  every lane is an integer in `[-126, 127]`. The arguments are integer gate differences; the exact
  power is the correctly rounded result (the reference platform's), which `Device.Exp2F32` allows
  (relative error `2^-22`) but does not force: it is what `exp2f` returns at an integer argument.
With these answers every `float32` operation is exact; the `bfloat16` rounding of the entries of
`A` (line 140) is exact too (each is an integer below `256` times a power of two, checked below), so
the value before the store's rounding is `ref` exactly. -/

open TVA TVA.Validation

namespace BwdDvTest

open Claims.ChunkBwdDvLocal (Inst)

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

/-- The exact `exp2` of a rank-2 `float32` request (the kernel's shape, `[BT, BT]`). -/
def exactExp2 : (q : PlatQ) → Option q.Out
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

/-- A sentinel in every output element before the launch, and in the elements just outside `dv`. -/
def sentinel : Rat := -999

/-- Sentinel elements on each side of `dv`. -/
def pad : Nat := 4

def ival (d : DType) (z : Int) : d.Val := d.ofBits (z % (2 ^ (8 * d.bytes) : Int)).toNat
def bfv (z : Int) : BF16 := FloatV.ofRat (z : Rat)
def f32v (z : Int) : F32 := FloatV.ofRat (z : Rat)

/-- One case: the claim's `canon` arguments (with the scale's double), the scale's value (checked
against it), the sequences' lengths and the rows of `chunk_indices` (variable length), and the
contents as integers by flat element (`g` at `t·HV + hv`). -/
structure Case where
  vl : Bool
  G : Bool
  B : Nat
  T : Nat
  H : Nat
  HV : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
  sc : F64
  sigma : Rat
  dt : DType
  N : Nat
  NT : Nat
  lens : List Nat
  rows : List (Nat × Nat)
  q : Nat → Int
  k : Nat → Int
  dO : Nat → Int
  g : Nat → Int

def Case.cu (c : Case) (i : Nat) : Nat := (c.lens.take i).sum
def Case.ci (c : Case) (i : Nat) : Nat :=
  let r := c.rows.getD (i / 2) (0, 0)
  if i % 2 = 0 then r.1 else r.2

/-- The claim's instance (the tables' contents are part of it; the other contents are `Pre`'s, in
memory). -/
def Case.inst (c : Case) : Inst :=
  Claims.ChunkBwdDvLocal.canon c.vl c.G c.B c.T c.H c.HV c.K c.V c.BT c.BK c.BV c.sc c.dt c.N c.NT
    (fun i => ival c.dt (c.cu i)) (fun i => ival c.dt (c.ci i))

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let bf (m : CMem) (b : Buf .bf16) (f : Nat → Int) : CMem :=
    CMem.ofElems m .bf16 ((List.range b.len).map fun j : Nat => ((b.ptr j).addr, bfv (f j)))
  let m := bf ∅ d.q c.q
  let m := bf m d.k c.k
  let m := bf m d.dO c.dO
  let m := CMem.ofElems m .bf16 ((List.range (d.dv.len + 2 * pad)).map fun j : Nat =>
    ((d.dv.ptr ((j : Int) - pad)).addr, (FloatV.ofRat sentinel : BF16)))
  let m := if c.G then CMem.ofElems m .f32 ((List.range d.g.len).map fun j : Nat =>
    ((d.g.ptr j).addr, f32v (c.g j))) else m
  if c.vl then
    let m := CMem.ofElems m c.dt ((List.range (c.N + 1)).map fun i : Nat =>
      ((d.cu.ptr i).addr, ival c.dt (c.cu i)))
    CMem.ofElems m c.dt ((List.range (2 * c.NT)).map fun i : Nat =>
      ((d.ci.ptr i).addr, ival c.dt (c.ci i)))
  else m

def Case.run (c : Case) : LRes (Claims.ChunkBwdDvLocal.launch c.inst).size × List Record :=
  execLaunch ((Claims.ChunkBwdDvLocal.launch c.inst).env P0) O fill0 1000000
    (Claims.ChunkBwdDvLocal.model c.inst) (Claims.ChunkBwdDvLocal.launch c.inst) c.init

/-! ## What `Post` says, computed from the inputs -/

/-- The sequences, as `[lo, hi)` ranges of the flat token axis: the rows `[b·T, b·T + T)`, or
`[cu[n], cu[n+1])`. -/
def Case.seqs (c : Case) : List (Nat × Nat) :=
  if c.vl then (List.range c.N).map fun n => (c.cu n, c.cu (n + 1))
  else (List.range c.B).map fun b => (b * c.T, b * c.T + c.T)

/-- The entry `A[p, j] = σ·(Σ_c k[p, h, c]·q[j, h, c])·2^{g[j] − g[p]}` (`h = hv / (HV / H)`). -/
def Case.A (c : Case) (p j hv : Nat) : Rat :=
  let h := hv / (c.HV / c.H)
  let kq : Int := ((List.range c.K).map fun x => c.k ((p * c.H + h) * c.K + x) * c.q ((j * c.H + h) * c.K + x)).sum
  c.sigma * kq * (if c.G then pow2 (c.g (j * c.HV + hv) - c.g (p * c.HV + hv)) else 1)

/-- The written elements with `ref` and `Σ_j |A·do|`: for each sequence `[lo, hi)`, token `p`,
value head `hv`, column `e`, the sum over `j` from `p` to the end of `p`'s chunk. -/
def Case.refs (c : Case) : List (Nat × Rat × Rat) :=
  c.seqs.flatMap fun (lo, hi) => (List.range' lo (hi - lo)).flatMap fun p =>
    let cs := lo + (p - lo) / c.BT * c.BT
    let js := List.range' p (min (cs + c.BT) hi - p)
    (List.range c.HV).flatMap fun hv =>
      let As := js.map fun j => (j, c.A p j hv)
      (List.range c.V).map fun e =>
        let terms := As.map fun (j, a) => a * c.dO ((j * c.HV + hv) * c.V + e)
        ((p * c.HV + hv) * c.V + e, terms.sum, (terms.map fun t => |t|).sum)

/-- The value `dv[p, hv, e]` must hold: `ref`, rounded to `bfloat16` (nearest even). -/
def rnd (x : Rat) : Rat := (FloatV.ofRat x : BF16).toRat

def Case.outs (c : Case) : List (Nat × Rat) := c.refs.map fun (j, x, _) => (j, rnd x)

/-- The final memory against `Post`: every written element is its expectation; every byte outside
the written elements keeps its initial value, and no byte is added. -/
def Case.agree (c : Case) (outs : List (Nat × Rat))
    (cf : CConfig (Claims.ChunkBwdDvLocal.launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let badOut := outs.filter fun (j, x) => (m.read .bf16 (d.dv.ptr j).addr).map (·.toRat) != some x
  let wBytes : Std.HashSet Int := outs.foldl (fun s (j, _) =>
    (s.insert (d.dv.ptr j).addr).insert ((d.dv.ptr j).addr + 1)) {}
  let init := c.init
  let badElse := init.toList.filter fun (a, b) => !wBytes.contains a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"dv elements {(badOut.take 8).map (·.1)} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside the written elements changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

/-! ## The instances -/

/-- A small signed integer from an index (`[-2, 2]`), scrambled (no short period along any axis). -/
def mix (a b j : Nat) : Int := ((((a * j + b) * 2654435761 / 65536) % 5 : Nat) : Int) - 2
/-- The same in `[-1, 1]`. -/
def mix3 (a b j : Nat) : Int := ((((a * j + b) * 2654435761 / 65536) % 3 : Nat) : Int) - 1

def gated : Case :=
  { vl := false, G := true, B := 2, T := 20, H := 1, HV := 2, K := 5, V := 6, BT := 16, BK := 16,
    BV := 16, sc := Claims.ChunkBwdDvLocal.s16, sigma := 1 / 4, dt := .i64, N := 0, NT := 0,
    lens := [], rows := [], q := mix 3 1, k := mix 7 2, dO := mix 2 3,
    -- `g[b·T + t, hv] = -⌊(t·(hv+1) + 3b)/5⌋`: nonincreasing in `t`, nonpositive
    g := fun i => -((((i / 2) % 20) * (i % 2 + 1) + 3 * (i / 40)) / 5 : Nat) }

def varlen : Case :=
  { vl := true, G := false, B := 1, T := 40, H := 2, HV := 4, K := 130, V := 20, BT := 32,
    BK := 128, BV := 32, sc := Claims.ChunkBwdDvLocal.s64, sigma := 1 / 8, dt := .i32, N := 3,
    NT := 3, lens := [5, 0, 33], rows := [(2, 1), (0, 0), (2, 0)],
    q := mix3 3 1, k := mix3 7 2, dO := mix 2 3, g := fun _ => 0 }

def wideV : Case :=
  { vl := false, G := true, B := 1, T := 70, H := 1, HV := 1, K := 16, V := 130, BT := 64,
    BK := 16, BV := 128, sc := Claims.ChunkBwdDvLocal.s16, sigma := 1 / 4, dt := .i64, N := 0,
    NT := 0, lens := [], rows := [], q := mix 3 1, k := mix 7 2, dO := mix 2 3,
    -- `g[t] = -⌊t/16⌋`: nonincreasing, at most `4` apart within a chunk of `64`
    g := fun i => -(i / 16 : Nat) }

open Claims.ChunkBwdDvLocal in
theorem gated_ok : gated.inst.ok := by
  dsimp only [Case.inst, gated]
  exact canon_ok (by norm_num) ⟨4, pow4⟩ (by norm_num) (by norm_num) ⟨4, pow4⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s16_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

open Claims.ChunkBwdDvLocal in
theorem varlen_ok : varlen.inst.ok := by
  dsimp only [Case.inst, varlen]
  exact canon_ok (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num) ⟨5, pow5⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s64_ok (by norm_num) (by norm_num) (Or.inl rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

open Claims.ChunkBwdDvLocal in
theorem wideV_ok : wideV.inst.ok := by
  dsimp only [Case.inst, wideV]
  exact canon_ok (by norm_num) ⟨4, pow4⟩ (by norm_num) (by norm_num) ⟨7, pow7⟩ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by simp) (by norm_num) (by norm_num [nChunks, canon]) (by norm_num)
    (by norm_num [nChunks, canon]) s16_ok (by norm_num) (by norm_num) (Or.inr rfl)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)

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
  open Claims.ChunkBwdDvLocal in
  0 ≤ cuAt d 0 && (List.range c.N).all (fun i => cuAt d i ≤ cuAt d (i + 1)) && cuAt d c.N ≤ c.T &&
  (List.range c.NT).all (fun r => 0 ≤ seqAt d r && seqAt d r < c.N && 0 ≤ chkAt d r &&
    chkAt d r * c.BT < cuAt d ((seqAt d r).toNat + 1) - cuAt d (seqAt d r).toNat) &&
  (List.range c.NT).all (fun r => (List.range c.NT).all fun r' =>
    r == r' || (seqAt d r, chkAt d r) != (seqAt d r', chkAt d r')) &&
  (List.range c.N).all (fun n => (List.range (c.T + 1)).all fun ch =>
    !((ch * c.BT : Int) < cuAt d (n + 1) - cuAt d n) ||
      (List.range c.NT).any fun r => seqAt d r == n && chkAt d r == ch)

/-- `Pre`'s buffer and table parts. `ValsOk` holds of these inputs: every value is a finite small
integer (`|do| ≤ 2`), the gate differences within a chunk are at most `0`, and the magnitudes are
far below the room bounds (`kqMag ≤ 130`, `σ ≤ 1/4`, `wUp ≤ 2`, at most `64` terms). -/
def Case.pre (c : Case) : Bool :=
  let d := c.inst
  let m := c.init
  validIn d.q m && validIn d.k m && validIn d.dO m && validIn d.dv m &&
    (!c.G || validIn d.g m) &&
    (!c.vl || (validIn d.cu m && validIn d.ci m && holdsIn d.cu m d.cuVal &&
      holdsIn d.ci m d.ciVal && tablesOk c))

#guard [gated, varlen, wideV].all Case.pre

-- The case's bookkeeping: `N`, `NT` are the tables' sizes; the host's `BK`, `BV`
-- (`min(max(next_power_of_2(n), 16), 128)`).
#guard varlen.N == varlen.lens.length && varlen.NT == varlen.rows.length
#guard [gated, varlen, wideV].all fun c =>
  c.BK == min (max (Nat.nextPowerOfTwo c.K) 16) 128 && c.BV == min (max (Nat.nextPowerOfTwo c.V) 16) 128

-- The scales' patterns are the values the expectations use (the double, and the `float32` the
-- kernel receives).
#guard [gated, varlen, wideV].all fun c =>
  c.sc.toRat == c.sigma && (FloatV.hostF32 c.sc).toRat == c.sigma

-- The gates: nonincreasing along each row, nonpositive (`ValsOk`: `g[j] − g[p] ≤ 64`, `p ≤ j`).
#guard (List.range 2).all fun b => (List.range 19).all fun t => (List.range 2).all fun hv =>
  gated.g ((b * 20 + t + 1) * 2 + hv) ≤ gated.g ((b * 20 + t) * 2 + hv) && gated.g ((b * 20 + t) * 2 + hv) ≤ 0
#guard (List.range 69).all fun t => wideV.g (t + 1) ≤ wideV.g t

-- The inputs are signed, and every element is encoded exactly.
#guard (List.range 40).any (fun j => varlen.q j < 0) && (List.range 40).any (fun j => gated.dO j < 0)
#guard [gated, varlen, wideV].all fun c => (List.range 11000).all fun j : Nat =>
  (bfv (c.q j)).toRat == (c.q j : Rat) && (bfv (c.k j)).toRat == (c.k j : Rat) &&
  (bfv (c.dO j)).toRat == (c.dO j : Rat) && (f32v (c.g j)).toRat == (c.g j : Rat)

-- The entries of `A` are exact `bfloat16`s (so line 140's rounding is exact).
#guard [gated, varlen, wideV].all fun c => c.seqs.all fun (lo, hi) =>
  (List.range' lo (hi - lo)).all fun p => (List.range c.HV).all fun hv =>
    (List.range' p (min (lo + (p - lo) / c.BT * c.BT + c.BT) hi - p)).all fun j =>
      rnd (c.A p j hv) == c.A p j hv

-- Every output element of a covered token is written once (all of `dv` on fixed length; all but
-- the two uncovered tokens' on variable length), the expectations are within `Post`'s
-- `2^-7·Σ_j |A·do|` of `ref`, and none equals the sentinel.
#guard [gated, wideV].all fun c => (c.refs.map (·.1)).mergeSort == List.range (c.B * c.T * c.HV * c.V)
#guard (varlen.refs.map (·.1)).mergeSort == List.range (38 * 4 * 20)
#guard [gated, varlen, wideV].all fun c => c.refs.all fun (_, x, a) =>
  |rnd x - x| ≤ a / 128 && rnd x != sentinel

-- The expectations are not trivial: fractions (the gates), and some rounded by the store.
#guard gated.refs.any (fun (_, x, _) => x.den > 4) && wideV.refs.any (fun (_, x, _) => x.den > 4)
-- (`varlen`'s values, at most `37/4` in magnitude, are all exact `bfloat16`s.)
#guard [gated, wideV].all fun c => c.refs.any fun (_, x, _) => rnd x != x

/-! ## The runs -/

def verdictGated : Verdict := Verdict.ofLaunch (gated.agree gated.outs) gated.run
def verdictVarlen : Verdict := Verdict.ofLaunch (varlen.agree varlen.outs) varlen.run
def verdictWideV : Verdict := Verdict.ofLaunch (wideV.agree wideV.outs) wideV.run

#eval IO.println (verdictGated.line "chunk_bwd_dv_local_bf16/gated")
#eval IO.println (verdictVarlen.line "chunk_bwd_dv_local_bf16/varlen")
#eval IO.println (verdictWideV.line "chunk_bwd_dv_local_bf16/wide_v")

-- Agreement resting on assumed answers: per program, one dot per `k` step, one `exp2` with a
-- gate, one dot per value block (`gated`: 8 × (1 + 1 + 1); `varlen`: 12 × (2 + 1); `wideV`:
-- 2 × (1 + 1 + 2)).
#guard verdictGated == .conditional 24
#guard verdictVarlen == .conditional 36
#guard verdictWideV == .conditional 8

-- A wrong expectation is caught: `agree` is not vacuous (the gate dropped; `k` and `q` swapped in
-- the scores, i.e. `A` transposed; one element off by one unit in the last place).
#guard
  let r := gated.run
  (Verdict.ofLaunch (gated.agree { gated with G := false }.outs) r).isDiscrepancy &&
  (Verdict.ofLaunch (gated.agree { gated with q := gated.k, k := gated.q }.outs) r).isDiscrepancy &&
  (Verdict.ofLaunch (gated.agree (gated.outs.map fun (j, x) =>
    if j = 7 then (j, x + |x| / 64 + 1 / 1024) else (j, x))) r).isDiscrepancy

end BwdDvTest
