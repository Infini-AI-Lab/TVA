import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `attn_fwd_f16` run by `execLaunch` on small covered instances. Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says on these inputs, both halves: (a) every element of `o` and of `M`
holds a value, finite and within `Post`'s tolerance of the exact attention `attnRef` and the exact
log-sum-exp `lseRef` (the heads are covered: `HeadOk`, checked below); (b) every other byte is
unchanged (`q`, `k`, `v`, sentinels on both sides of `o` and `M`), and no byte is added.

The claim's sizes are fixed (`BLOCK_M = 128`, `BLOCK_N = 64`, `HEAD_DIM ≥ 64`, `128 ∣ N_CTX`): the
smallest covered launch is one `128 × 64` query tile per head, with two key blocks of `64`.

**The inputs.** `sm_scale` is the double `0x3FE62E4300000000` (`0.6931471824645996`, the
tutorial's own `float32` `LN2`): the kernel's `qk_scale = sm_scale · 1.44269504` (line 219, a
`float32` product) is then exactly `1` (checked below), so the kernel works in base `2` on the
integer scores `s = q·k`. `q` is nonzero only in its first four columns, `q`, `k` in `{-1, 0, 1}`,
so every score is an integer in `[-4, 4]`; `v` is an integer in `[-7, 7]`. Every exponent the
kernel passes to `tl.math.exp2` is then an integer: `s - m` (`m` the running row maximum, `≥ -8`),
`m_old - m_new`, `-∞` (the first block's `alpha`), and on the causal mask's lanes about `-10^6`.

**The platform answers are assumed** (the verdicts are `conditional n`, `n` the number of
platform requests: four per key block, `q·kᵀ`, `exp2(qk − m)`, `exp2(m_old − m_new)` and
`p·v + acc`, and two per program, `log2` and `/`), each the value any correct device gives on
these operands, or the correctly rounded one where the result is not exact:
- `tl.dot` (`q·kᵀ`, and `p·v + acc`): the exact value, answered only when it is a `float32`
  (integer scores; `p` a power of two at least `2^-8`, so `acc` is a multiple of `2^-8` below
  `2^10`): any correct device gives it, whatever its order;
- `tl.math.exp2` at an integer `z ≥ -149`: `2^z`, exact; at `-∞`: `+0` (`Device.Exp2F32NegInf`);
  at an integer `z ≤ -150` (the masked lanes): `+0`, the correctly rounded value (`Device.Exp2F32`
  allows any value within `2^-126`);
- `tl.math.log2` (line 241, `log₂ l_i`, `l_i` a sum of powers of two): the `float32` rounding of
  Lean's double `log2` (within `2^-24·(1 + 2^-28)` relative, inside `Device.Log2F32`'s `2^-22`);
  exact when `l_i` is a power of two;
- `/` (line 242): the IEEE quotient (`FloatV.divRn`), the correctly rounded value.

**The check.** `attnRef`, `attnAbs` and `lseRef` involve `Real.exp` of `sm_scale · score`; they are
evaluated here in Lean's `Float` from the inputs (weights `exp(sm_scale·(s_j - max s))`, which
leave the quotients unchanged; relative error about `10^-15`), and each output must be within
`Post`'s tolerance minus a margin of `10^-9·(1 + |ref| + attnMag)` far above that error: never a
looser check than `Post`. The tolerance is `Post`'s at the smallest `X` it allows for the row,
`X = max_j sm_scale·scoreMag` over the row's keys (`epsO`, `epsM` grow with `X`, so every larger
`X` follows), with `epsO`, `epsM`, `scoreMag`'s `Device.liftMag` the functions `Post` names. -/

open TVA TVA.Validation Claims.AttnFwd

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

/-! ## The oracle -/

/-- Rationals with power-of-two (or any) denominators as integers over a common denominator. -/
def scaleInts (xs : Array Rat) : Nat × Array Int :=
  let den := xs.foldl (fun m q => Nat.lcm m q.den) 1
  (den, xs.map fun q => (q * (den : Rat)).num)

/-- A rational as an element of a float accumulator dtype, when it is exactly representable. -/
def accOf : (d : DType) → Rat → Option d.Val
  | .f32, q => let v : F32 := FloatV.ofRat q; if v.toRat == q then some v else none
  | _, _ => none

/-- The exact `acc + a·b` of a `tl.dot` request from its own operands (finite floats), answered
only when every element of the result is an exact `float32`. -/
def exactDot : (q : PlatQ) → Option q.Out
  | @PlatQ.dot cfg M N K a b acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k => a.get2 i k).toArray.mapM (cfg.a.val? ·)
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j => b.get2 k j).toArray.mapM (cfg.b.val? ·)
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM (cfg.acc.val? ·)
    let (da, Ai) := scaleInts A
    let (db, Bi) := scaleInts B
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      let s : Int := (List.range K).foldl (fun s k => s + Ai[i * K + k]! * Bi[k * N + j]!) 0
      accOf cfg.acc ((s : Rat) / ((da * db : Nat) : Rat) + C[e]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- `2^x` at an integer or `-∞` `x`: exact for `z ≥ -149`, `+0` (the correctly rounded value) for
`z ≤ -150` and at `-∞`; nothing otherwise. -/
def exp2Lane {f : Format} (x : FloatV f) : Option (FloatV f) :=
  if x.pat == (FloatV.negInf : FloatV f).pat then some (FloatV.ofPat 0)
  else if x.isFinite && x.toRat.den == 1 then
    let z := x.toRat.num
    if z ≤ -150 then some (FloatV.ofPat 0)
    else if z ≤ 100 then some (FloatV.ofRat ((2 : Rat) ^ z)) else none
  else none

/-- A rational as a `Float`. -/
def toF (q : Rat) : Float := Float.ofInt q.num / Float.ofNat q.den

/-- `log₂ x` for a finite positive `x`: the rounding of Lean's double `log2`. -/
def log2Lane {f : Format} (x : FloatV f) : Option (FloatV f) :=
  if x.isFinite && x.toRat > 0 then
    some (FloatV.cvt (FloatV.ofPat (Float.log2 (toF x.toRat)).toBits.toNat : F64))
  else none

/-- The position of an index in row-major order (as `Tensor.toList`). -/
def flatIdx : (s : Shape) → Idx s → Nat
  | [], _ => 0
  | _ :: s, (i, j) => i.1 * s.foldr (· * ·) 1 + flatIdx s j

/-- A tensor read from an array in row-major order: the oracle's answers are stored, not
recomputed from their operands at every read (which keeps the interpreter's time down). -/
def ofArr {s : Shape} {f : Format} (a : Array (FloatV f)) : Tensor s (FloatV f) :=
  ⟨fun i => a.getD (flatIdx s i) (FloatV.ofPat 0)⟩

/-- Lane by lane, when every lane has an answer. -/
def lanewise {s : Shape} {f : Format} (g : FloatV f → Option (FloatV f)) (x : Tensor s (FloatV f)) :
    Option (Tensor s (FloatV f)) :=
  (x.toList.mapM g).map fun l => ofArr l.toArray

/-- The oracle: exact dots, `exp2` and `log2` lane by lane, the IEEE quotient; nothing else
(trees, orders: defaults). -/
def O : Oracle := fun _ _ q => match q with
  | .plat (.math .exp2 _ x) => lanewise exp2Lane x
  | .plat (.math .log2 _ x) => lanewise log2Lane x
  | .plat (.div _ x y) => some (ofArr (List.zipWith FloatV.divRn x.toList y.toList).toArray)
  | .plat q => exactDot q
  | _ => none

/-! ## The inputs -/

/-- `sm_scale`: the double `0.6931471824645996`. -/
def sm : F64 := FloatV.ofPat 0x3FE62E4300000000

/-- `q[bh, n, e]`, `k[bh, j, e]`, `v[bh, j, e]` (integers; `k` in `{-1, 0}` in the first key
block, so that many rows find their maximum in the second one). -/
def qv (bh n e : Nat) : Int := if e < 4 then ((3 * n + 5 * e + 7 * bh) % 3 : Nat) - (1 : Int) else 0
def kv (bh j e : Nat) : Int :=
  if j < 64 then ((5 * j + 3 * e + bh + (j / 7) * e) % 2 : Nat) - (1 : Int)
  else ((5 * j + 3 * e + bh + (j / 7) * e) % 3 : Nat) - (1 : Int)
def vv (bh j e : Nat) : Int := ((7 * j + 11 * e + 3 * bh) % 15 : Nat) - (7 : Int)

/-- Sentinels: in `o` before the launch, and in `pad` elements on each side of `o` and `M`. -/
def sentO : F16 := FloatV.ofRat (-999)
def sentM : F32 := FloatV.ofRat (-12345)
def pad : Nat := 8

def f16v (z : Int) : F16 := FloatV.ofRat (z : Rat)

def initOf (d : Inst) : CMem :=
  let D := d.HEAD_DIM
  let all := (List.range (rows d)).flatMap fun r : Nat => (List.range D).map fun e : Nat =>
    (r / d.N_CTX, r % d.N_CTX, e)
  let m := CMem.ofElems ∅ .f16 (all.map fun (bh, n, e) => ((d.q.ptr (idx d bh n e)).addr, f16v (qv bh n e)))
  let m := CMem.ofElems m .f16 (all.map fun (bh, n, e) => ((d.k.ptr (idx d bh n e)).addr, f16v (kv bh n e)))
  let m := CMem.ofElems m .f16 (all.map fun (bh, n, e) => ((d.v.ptr (idx d bh n e)).addr, f16v (vv bh n e)))
  let m := CMem.ofElems m .f16 ((List.range (rows d * D + 2 * pad)).map fun j : Nat =>
    ((d.o.ptr ((j : Int) - pad)).addr, sentO))
  CMem.ofElems m .f32 ((List.range (rows d + 2 * pad)).map fun j : Nat =>
    ((d.M.ptr ((j : Int) - pad)).addr, sentM))

def runOf (d : Inst) : LRes (launch d).size × List Record :=
  execLaunch ((launch d).env P0) O fill0 1000000 (model d) (launch d) (initOf d)

/-! ## The expectation -/

/-- What the test expects of row `n` of head `bh`: the reference output `(attnRef, attnMag)` of
each column, and `lseRef`, in `Float`; and the tolerances' `X`. -/
structure RowRef where
  out : Array (Float × Float)
  lse : Float
  X : Rat
  deriving Inhabited

/-- The keys of row `n`, as `Post`'s `keys`. -/
def keyList (d : Inst) (n : Nat) : List Nat :=
  (List.range d.N_CTX).filter fun j => !d.causal || j ≤ n

def scoreI (d : Inst) (bh n j : Nat) : Int :=
  ((List.range d.HEAD_DIM).map fun e => qv bh n e * kv bh j e).sum

/-- `Σ_e liftMag q·liftMag k` (`scoreMag`), from the integer inputs: `liftMag` of an integer is
its magnitude (checked below on every input value), so this is `Σ_e |q·k|`, an integer. -/
def scoreMagI (d : Inst) (bh n j : Nat) : Nat :=
  ((List.range d.HEAD_DIM).map fun e => (qv bh n e).natAbs * (kv bh j e).natAbs).sum

/-- The reference of row `n` with the softmax scale `sig` (the test's is `sm_scale`) and the
values of row `j + vShift` (the test's is `0`). -/
def rowRef (d : Inst) (sig : Rat) (vShift : Nat) (bh n : Nat) : RowRef :=
  let ks := keyList d n
  let ss := ks.map fun j => scoreI d bh n j
  let smax := ss.foldl max (ss.headD 0)
  let sigF := toF sig
  let ws := ss.map fun s => Float.exp (sigF * Float.ofInt (s - smax))
  let den := ws.foldl (· + ·) 0
  let kw := ks.zip ws
  let out := (Array.range d.HEAD_DIM).map fun e =>
    let num := kw.foldl (fun a (j, w) => a + w * Float.ofInt (vv bh (j + vShift) e)) 0
    let numAbs := kw.foldl (fun a (j, w) => a + w * Float.ofInt (vv bh (j + vShift) e).natAbs) 0
    let vAbs := ks.foldl (fun a j => a + Float.ofInt (vv bh (j + vShift) e).natAbs) 0
    let mag := max (numAbs / den) (vAbs / Float.ofNat ks.length)
    (num / den, mag)
  let lse := sigF * Float.ofInt smax / Float.log 2 + Float.log2 den
  let X := sigma d * (ks.foldl (fun a j => max a (scoreMagI d bh n j)) 0 : Nat)
  { out, lse, X }

/-- The final memory against `Post`, with the references `R`: (a) every element of `o` finite
and within `epsO X·attnMag + 2^-24` of `attnRef`, every element of `M` finite and within
`epsM X` of `lseRef` (minus the margin); (b) every byte outside `o` and `M` unchanged, none
added. -/
def agreeOf (d : Inst) (init : CMem) (R : Array RowRef) (cf : CConfig (launch d).size) :
    Option String :=
  let m := CMem.toMem cf.mem
  let D := d.HEAD_DIM
  let pairs := (List.range (d.Z * d.H)).flatMap fun bh => (List.range d.N_CTX).map fun n => (bh, n)
  let bad := pairs.filter fun (bh, n) =>
    let r := R[bh * d.N_CTX + n]!
    let tO := toF (epsO D d.N_CTX r.X)
    let tM := toF (epsM D r.X)
    let badO := (List.range D).any fun e =>
      match m.read .f16 (d.o.ptr (idx d bh n e)).addr with
      | none => true
      | some c =>
        let (ref, mag) := r.out[e]!
        !(c.isFinite && (toF c.toRat - ref).abs + 1e-9 * (1 + ref.abs + mag) ≤ tO * mag + 1 / 16777216)
    let badM := match m.read .f32 (d.M.ptr (bh * d.N_CTX + n : Nat)).addr with
      | none => true
      | some c => !(c.isFinite && (toF c.toRat - r.lse).abs + 1e-9 * (1 + r.lse.abs) ≤ tM)
    badO || badM
  let outA (a : Int) : Bool :=
    (d.o.base ≤ a && a < d.o.base + 2 * (rows d * D : Nat)) ||
    (d.M.base ≤ a && a < d.M.base + 4 * (rows d : Nat))
  let badElse := init.toList.filter fun (a, b) => !outA a && cf.mem[a]? != some b
  if !bad.isEmpty then some s!"rows (head, n) {bad.take 8} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside o and M changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

/-- The references of every row (`(bh, n)` at `bh·N_CTX + n`), computed once: a table, not a
function (each `RowRef` is a loop over the row's keys). -/
def table (d : Inst) (R : Nat → Nat → RowRef) : Array RowRef :=
  (Array.range (d.Z * d.H * d.N_CTX)).map fun r => R (r / d.N_CTX) (r % d.N_CTX)

/-- The test's expectation: `Post`'s references, with `sm_scale` and the rows' own values. -/
def expectOf (d : Inst) : Array RowRef := table d (rowRef d (sigma d) 0)

/-! ## The cases -/

/-- Not causal, `Z = 1`, `H = 2`, `N_CTX = 128`, `HEAD_DIM = 64`: two programs (the second
head's offsets: `off_h = 1`, `offset_y = N_CTX`, in `q`, `k`, `v`, `o` and `M`), each two key
blocks (the second rescales the first's accumulator where its row maximum is larger). -/
def full : Inst := canon 1 2 128 64 sm false
/-- Causal, `Z = H = 1`: one program, which runs the diagonal stage only (`STAGE = 3`: the first
stage's loop is empty at `start_m = 0`), both key blocks under the mask, the second fully masked
for rows `n < 64`. -/
def causal : Inst := canon 1 1 128 64 sm true

theorem sm_ok : sm.isFinite = true ∧ (1 / 2 ^ 64 : ℚ) ≤ sm.toRat ∧ sm.toRat ≤ 2 ^ 64 :=
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩
theorem full_ok : full.ok :=
  canon_ok 1 2 128 64 sm false (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega) (by omega) sm_ok
theorem causal_ok : causal.ok :=
  canon_ok 1 1 128 64 sm true (by omega) (by omega) (by omega) (by omega) (by omega) (by omega)
    (by omega) (by omega) (by omega) sm_ok

-- `qk_scale` (line 219) is exactly `1`.
#guard (FloatV.mul (FloatV.hostF32 sm) (FloatV.litF32 (FloatV.ofPat 0x3FF7154764EE6C2F) : F32)).toRat == 1

-- `Pre`: the five buffers aligned and every byte present in `initOf`.
#guard [full, causal].all fun d =>
  let m := CMem.toMem (initOf d)
  d.q.base % 16 == 0 && d.k.base % 16 == 0 && d.v.base % 16 == 0 && d.o.base % 16 == 0 &&
  d.M.base % 4 == 0 &&
  ((List.range d.q.len).all fun e : Nat => (m.read .f16 (d.q.ptr e).addr).isSome &&
    (m.read .f16 (d.k.ptr e).addr).isSome && (m.read .f16 (d.v.ptr e).addr).isSome &&
    (m.read .f16 (d.o.ptr e).addr).isSome) &&
  ((List.range d.M.len).all fun e : Nat => (m.read .f32 (d.M.ptr e).addr).isSome)

-- `HeadOk` of every head: the inputs are finite (small integers, encoded exactly), `|v| ≤ 2^15`,
-- and every `sm_scale·scoreMag ≤ 2^10`. The scores are in `[-4, 4]` (so every `p ≥ 2^-8`).
#guard [full, causal].all fun d => (List.range (d.Z * d.H)).all fun bh =>
  (List.range d.N_CTX).all fun n =>
    ((List.range d.HEAD_DIM).all fun e =>
      (f16v (qv bh n e)).toRat == qv bh n e && (f16v (kv bh n e)).toRat == kv bh n e &&
      (f16v (vv bh n e)).toRat == vv bh n e && (vv bh n e).natAbs ≤ 2 ^ 15) &&
    (List.range d.N_CTX).all fun j =>
      sigma d * (scoreMagI d bh n j : Rat) ≤ 2 ^ 10 && (scoreI d bh n j).natAbs ≤ 4

-- `liftMag` of every input value (an integer in `[-7, 7]`) is its magnitude (`scoreMagI`).
#guard (List.range 15).all fun z : Nat => Device.liftMag .f16 ((z : Int) - 7 : Rat) == (((z : Int) - 7).natAbs : Rat)

-- The rows are not uniform: some row's maximum is in the second key block only (the rescaling
-- runs), and the scores vary along every row.
#guard (List.range 64).any fun n =>
  ((List.range 64).map fun j => scoreI full 0 n j).foldl max (-100) <
    ((List.range 64).map fun j => scoreI full 0 n (j + 64)).foldl max (-100)
#guard (List.range 128).all fun n =>
  ((List.range 128).map fun j => scoreI full 0 n j).eraseDups.length ≥ 3

/-- One case, on one run: print its verdict, and fail (an error, which fails the test as a failed
`#guard` does) unless the verdict is `conditional n` and every wrong expectation `bad E` (`E`
the test's expectation) is a discrepancy on the same run. A run of the kernel takes about 50 s
per program in the interpreter (100 s on the causal path), and a `#guard` after the `#eval`
would run it again: so the guards on a run are checked in the command that prints its
verdict. -/
def check (name : String) (d : Inst) (n : Nat)
    (bad : List (Array RowRef → Array RowRef)) : IO Unit := do
  let r := runOf d
  let E := expectOf d
  let v := Verdict.ofLaunch (agreeOf d (initOf d) E) r
  IO.println (v.line s!"attn_fwd_f16/{name}")
  unless v == .conditional n do throw (IO.userError s!"{name}: not conditional {n}")
  for (B, k) in bad.zipIdx do
    unless (Verdict.ofLaunch (agreeOf d (initOf d) (B E)) r).isDiscrepancy do
      throw (IO.userError s!"{name}: wrong expectation {k} not caught")

/-- `M`'s expectation shifted by `0.01` (about nine times `epsM` here), `o`'s kept. -/
def lseShift (E : Array RowRef) : Array RowRef := E.map fun r => { r with lse := r.lse + 0.01 }

-- Agreement resting on ten assumed answers per program (four per key block, `log2` and `/`). A
-- wrong expectation is caught: uniform weights (`sm_scale = 0`); `M` off by `0.01`; the causal
-- run against the expectation without the mask.
#eval check "full" full 20 [fun _ => table full (rowRef full 0 0), lseShift]
#eval check "causal" causal 10 [fun _ => table causal (rowRef { causal with causal := false } (sigma causal) 0)]
