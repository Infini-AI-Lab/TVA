import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `ssd_bmm_chunk_bf16` run by `execLaunch` on three small covered
instances (the claim's own `mk`): contiguous `C`, `B` with two groups and a request boundary
inside a chunk, and a last chunk with one token; the mixer's strided views (`seqlen` stride `48`
for `ngroups·dstate = 40`, the eight padding columns of each row holding NaN, which the `k` mask
must keep out) with `dstate = 40` (two `tl.dot` steps, the second masked past `8` lanes), chunks of
`2`, and non-finite values (a NaN in a row of `C`, a `+∞` in a row of `B`); chunks of `128`
(two programs per chunk and group: `pid_m = 0, 1` against `BLOCK_SIZE_M = 64`). Not a test
against the device (phase 4): it checks that `exec` runs the model end to end, and that the final
memory is what `Post` says, both halves:
- every element `(c, g, m, n)` of `out` whose positions are tokens of one request (`sameReq`,
  computed here from the case's own request list) and whose two rows are finite holds a finite
  `float32` equal to the exact `Σ_k C[tm, g, k]·B[tn, g, k]` (distance `0`, which `Post` allows);
  one whose rows are not finite holds some `float32` (all `Post` asks); every other element
  (different requests, or a position past `seqlen`) holds `+0.0` exactly (bits `0`);
- every other byte is unchanged (`C`, `B`, `seq_idx`, sentinels on both sides of `out`), and no byte
  is added.

The inputs are integers in `[-3, 3]` (`dstate ≤ 40`), so every partial sum is an integer far inside
`float32`.

**The `tl.dot` values are assumed, not proved.** Line 161's `acc += tl.dot(a, b)` is a `dot_add`
site, answered by the oracle (`PlatQ.dot` with `acc` as accumulator): on each lane whose operands
(row, column, accumulator) are finite, the exact `acc + Σ a·b`, the only value a correct device
gives (every partial sum is an exact integer, whatever the order, fused or not); on a lane with a
non-finite operand, a NaN. Those lanes are the pairs whose `C` or `B` row is not finite: `Post`
asks nothing of their value when the pair is in one request, and `tl.where` replaces it with `+0`
otherwise, so the choice of NaN (rather than another non-finite value) does not affect the verdict.
The verdicts are `conditional n`, `n` the number of dots (programs × `cdiv(dstate, 32)`). -/

open TVA TVA.Validation Claims.SsdBmm

/-- A platform that relates nothing: the dots are answered by the oracle. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle: exact dots -/

/-- The `float32` quiet NaN. -/
def nan32 : F32 := FloatV.ofPat 0x7FC00000

/-- A `float32` accumulator lane: the exact value `q` when it is a `float32` (`none` otherwise), or
a NaN for a lane with a non-finite operand (`none` given). -/
def accR : (d : DType) → Option Rat → Option d.Val
  | .f32, some q => let v : F32 := FloatV.ofRat q; if v.toRat == q then some v else none
  | .f32, none => some nan32
  | _, _ => none

/-- The exact `acc + a·b` of a `tl.dot` request into `float32`, lane by lane: `none` unless every
finite lane's exact value is a `float32`; a NaN on a lane with a non-finite operand. -/
def exactDot : (q : PlatQ) → Option q.Out
  | @PlatQ.dot cfg M N K a b acc => do
    let A := ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => cfg.a.val? (a.get2 i k)).toArray
    let B := ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => cfg.b.val? (b.get2 k j)).toArray
    let C := ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => cfg.acc.val? (acc.get2 i j)).toArray
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      accR cfg.acc ((List.range K).foldl (fun s k => do
        let s ← s
        let x ← A[i * K + k]!
        let y ← B[k * N + j]!
        pure (if x == 0 || y == 0 then s else s + x * y)) C[e]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- The oracle: the exact dot at every platform request; nothing else. -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactDot q
  | _ => none

/-! ## The cases -/

def bfv (q : Rat) : BF16 := FloatV.ofRat q
def bfNaN : BF16 := FloatV.ofPat 0x7FC0
def bfInf : BF16 := FloatV.ofPat 0x7F80

/-- A sentinel in every element of `out` before the launch, and in `pad` elements on each side. -/
def sentinel : Rat := -7
def pad : Nat := 4

/-- One case: the instance's sizes for `mk T G K C cd`, the request of each token, and the
non-finite entries of `C` (a NaN) and `B` (a `+∞`) (token, group, `k`). -/
structure Case where
  T : Nat
  G : Nat
  K : Nat
  C : Nat
  cd : Nat
  req : Nat → Nat
  aBad : List (Nat × Nat × Nat)
  bBad : List (Nat × Nat × Nat)

def Case.inst (c : Case) : Inst := Claims.SsdBmm.mk c.T c.G c.K c.C c.cd

/-- `C[t, g, k]`, `B[t, g, k]` where finite: integers in `[-3, 3]`. -/
def aval (t g k : Nat) : Int := ((5 * t + 3 * g + 2 * k + 1) % 7 : Nat) - (3 : Int)
def bval (t g k : Nat) : Int := ((3 * t + 5 * g + 4 * k + 2) % 7 : Nat) - (3 : Int)

/-- Element `e` of a storage of `T` rows of `cd` elements: row `t`, group `g`, `k` when
`e % cd < G·K`; a padding element (NaN) otherwise. -/
def Case.elem (c : Case) (bad : List (Nat × Nat × Nat)) (badV : BF16) (v : Nat → Nat → Nat → Int)
    (e : Nat) : BF16 :=
  let t := e / c.cd
  let r := e % c.cd
  if r < c.G * c.K then
    let g := r / c.K
    let k := r % c.K
    if bad.contains (t, g, k) then badV else bfv (v t g k)
  else bfNaN

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .bf16 ((List.range d.a.len).map fun e : Nat => ((d.a.ptr e).addr, c.elem c.aBad bfNaN aval e))
  let m := CMem.ofElems m .bf16 ((List.range d.b.len).map fun e : Nat => ((d.b.ptr e).addr, c.elem c.bBad bfInf bval e))
  let m := CMem.ofElems m .i32 ((List.range c.T).map fun t : Nat => ((d.seqIdx.ptr t).addr, (IntV.ofNat (c.req t) : I32)))
  CMem.ofElems m .f32 ((List.range (d.out.len + 2 * pad)).map fun e : Nat =>
    ((d.out.ptr ((e : Int) - pad)).addr, (FloatV.ofRat sentinel : F32)))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) O fill0 10000 (model c.inst) (launch c.inst) c.init

/-- Row `tm` of `C` and row `tn` of `B` (group `g`) are finite. -/
def Case.finite (c : Case) (g tm tn : Nat) : Bool :=
  (List.range c.K).all fun k => !c.aBad.contains (tm, g, k) && !c.bBad.contains (tn, g, k)

/-- What `Post` asks of element `(c, g, m, n)`: `some (some q)` the finite value `q`; `some none`
any `float32`; `none` the `+0.0` bits. `same` decides `sameReq` from the case's own requests. -/
def Case.expect (c : Case) (ch g m n : Nat) : Option (Option Rat) :=
  let tm := ch * c.C + m
  let tn := ch * c.C + n
  if tm < c.T && tn < c.T && c.req tm == c.req tn then
    if c.finite g tm tn then
      some (some (((List.range c.K).map fun k => aval tm g k * bval tn g k).sum : Int))
    else some none
  else none

/-- The final memory against the expectation `ex`: every element of `out` as `ex` says; every
byte outside `out`'s elements unchanged; no byte added. -/
def Case.agree (c : Case) (ex : Nat → Nat → Nat → Nat → Option (Option Rat))
    (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let elems := (List.range (nch d)).flatMap fun ch => (List.range c.G).flatMap fun g =>
    (List.range c.C).flatMap fun i => (List.range c.C).map fun j => (ch, g, i, j)
  let bad := elems.filter fun (ch, g, i, j) =>
    let v := m.read .f32 (d.out.ptr (oIdx d ch g i j : Nat)).addr
    match ex ch g i j with
    | some (some q) => (v.bind (DType.f32.val? ·)) != some q
    | some none => v.isNone
    | none => v.map DType.f32.toBits != some 0
  let inOut (a : Int) : Bool := d.out.base ≤ a && a < d.out.base + 4 * (oB d : Int)
  let badElse := c.init.toList.filter fun (a, b) => !inOut a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"out elements {bad.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside out changed"
  else if cf.mem.size != c.init.size then some "the launch wrote outside the initial memory"
  else none

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch (c.agree c.expect) c.run

/-- Contiguous (`cd = G·K`): `seqlen = 5`, `ngroups = 2`, `dstate = 3`, chunks of `4`; requests
`[0, 0, 1, 1, 1]` (a boundary inside chunk `0`; chunk `1` holds one token, its other positions
past `seqlen`). Grid `(1, 1, 4)`, one dot step each. -/
def contig : Case :=
  { T := 5, G := 2, K := 3, C := 4, cd := 6, req := fun t => [0, 0, 1, 1, 1].getD t 0,
    aBad := [], bBad := [] }
theorem contig_ok : contig.inst.ok :=
  show (mk 5 2 3 4 6).ok from mk_ok (j := 2) rfl (by norm_num) (by norm_num)

/-- Strided views (`cd = 48 > G·K = 40`, eight NaN padding elements per row): `seqlen = 6`,
`ngroups = 1`, `dstate = 40` (two dot steps), chunks of `2`; requests `[0, 1, 1, 1, 2, 2]`; a NaN
at `C[1, 0, 5]` and a `+∞` at `B[4, 0, 37]` (in the second `k` step). Grid `(1, 1, 3)`. -/
def strided : Case :=
  { T := 6, G := 1, K := 40, C := 2, cd := 48, req := fun t => [0, 1, 1, 1, 2, 2].getD t 0,
    aBad := [(1, 0, 5)], bBad := [(4, 0, 37)] }
theorem strided_ok : strided.inst.ok :=
  show (mk 6 1 40 2 48).ok from mk_ok (j := 1) rfl (by norm_num) (by norm_num)

/-- Chunks of `128` (two programs per chunk and group, `pid_m = 0, 1`): `seqlen = 130`,
`ngroups = 1`, `dstate = 2`; requests `0` (tokens `0..49`), `1` (`50..119`), `2` (`120..129`,
crossing into chunk `1`). Grid `(2, 1, 2)`. -/
def bigChunk : Case :=
  { T := 130, G := 1, K := 2, C := 128, cd := 2,
    req := fun t => if t < 50 then 0 else if t < 120 then 1 else 2, aBad := [], bBad := [] }
theorem bigChunk_ok : bigChunk.inst.ok :=
  show (mk 130 1 2 128 2).ok from mk_ok (j := 7) rfl (by norm_num) (by norm_num)

def cases : List Case := [contig, strided, bigChunk]

/-! ## Coverage (`Pre`) and the expectations -/

-- `Pre`: the four buffers are aligned and inside `[0, 2^64)`, every element present in `init`,
-- and every `seq_idx` entry a non-negative `int32` (the case's requests).
#guard cases.all fun c =>
  let d := c.inst
  let m := CMem.toMem c.init
  let okB {dt : DType} (b : Buf dt) : Bool :=
    decide (0 ≤ b.base) && decide (b.base + b.len * dt.bytes ≤ 2 ^ 64) && decide (b.base % dt.bytes = 0) &&
    (List.range b.len).all fun e : Nat => (m.read dt (b.ptr e).addr).isSome
  okB d.a && okB d.b && okB d.seqIdx && okB d.out &&
    (List.range c.T).all fun t : Nat => (m.read .i32 (d.seqIdx.ptr t).addr).map (·.toInt) == some (c.req t : Int)

-- The inputs are encoded exactly; the non-finite entries are where the case says (and in the
-- padding of `strided`'s rows).
#guard (List.range 7).all fun z : Nat => (bfv (z - 3 : Int)).toRat == ((z - 3 : Int) : Rat)
#guard bfNaN.isNaN && !bfInf.isFinite && !bfInf.isNaN &&
  strided.elem strided.aBad bfNaN aval (1 * 48 + 5) == bfNaN && strided.elem strided.bBad bfInf bval (4 * 48 + 37) == bfInf &&
  strided.elem strided.aBad bfNaN aval 45 == bfNaN && (strided.elem strided.aBad bfNaN aval 4).isFinite

-- Every kind of element occurs: exact sums (some nonzero), unconstrained ones (`strided`'s
-- non-finite rows within a request), and zeroed ones (different requests, past `seqlen`) with a
-- non-finite row among them; and no exact sum equals the sentinel.
#guard cases.all fun c => let d := c.inst
  (List.range (nch d)).all fun ch => (List.range c.G).all fun g => (List.range c.C).all fun i =>
    (List.range c.C).all fun j => c.expect ch g i j != some (some sentinel)
#guard contig.expect 0 1 0 1 != some (some 0) && contig.expect 0 0 1 2 == none && contig.expect 1 0 0 1 == none &&
  strided.expect 0 0 1 1 == some none && strided.expect 0 0 1 0 == none &&
  strided.expect 2 0 0 0 == some none && strided.expect 1 0 0 1 != some none &&
  bigChunk.expect 0 0 49 50 == none && bigChunk.expect 1 0 1 0 != none && bigChunk.expect 1 0 2 0 == none

def verdictContig : Verdict := contig.verdict
def verdictStrided : Verdict := strided.verdict
def verdictBig : Verdict := bigChunk.verdict

#eval IO.println (verdictContig.line "ssd_bmm_chunk_bf16/contig")
#eval IO.println (verdictStrided.line "ssd_bmm_chunk_bf16/strided")
#eval IO.println (verdictBig.line "ssd_bmm_chunk_bf16/big_chunk")
-- Agreement resting on one assumed answer per dot (programs × `cdiv(dstate, 32)`).
#guard let v := verdictContig; v == .conditional 4 && v.agrees && v.assumed > 0
#guard let v := verdictStrided; v == .conditional 6 && v.agrees && v.assumed > 0
#guard let v := verdictBig; v == .conditional 4 && v.agrees && v.assumed > 0

-- A wrong expectation is caught: `agree` is not vacuous (the request mask ignored: the
-- cross-request pairs given their sums; the last `k` step dropped).
#guard
  let r := strided.run
  (Verdict.ofLaunch (strided.agree fun ch g i j =>
    let tm := ch * 2 + i
    let tn := ch * 2 + j
    if tm < 6 && tn < 6 && strided.finite g tm tn then
      some (some (((List.range 40).map fun k => aval tm g k * bval tn g k).sum : Int))
    else strided.expect ch g i j) r).isDiscrepancy &&
  (Verdict.ofLaunch (strided.agree fun ch g i j =>
    match strided.expect ch g i j with
    | some (some _) =>
      some (some (((List.range 32).map fun k => aval (ch * 2 + i) g k * bval (ch * 2 + j) g k).sum : Int))
    | e => e) r).isDiscrepancy
