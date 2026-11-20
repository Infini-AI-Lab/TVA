import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `block_scaled_matmul_mxfp8` run by `execLaunch` on small covered
instances. Not a test against the device (phase 4): it checks that `exec` runs the model end to end
(tensor-descriptor loads with zero padding, the 5-D scale blocks' reshape and permute, the
`tl.dot_scaled` steps, the `float16` store), and that the final memory is what `Post` says on these
inputs.

`Inst.ok` asks `M`, `N`, `K` to be positive multiples of `128`, and the tiles are `128 × 256` with
`k` steps of `128`: every program runs at least one full `128 × 256 × 128` `tl.dot_scaled`. Two
instances, from the claim's `hostInst`: `"mxfp8"` (`e4m3` by `e4m3`) with `N = 384` (two column
tiles, the second half outside `c`: zero-padded loads, `b`'s scale block at offset `2`, clipped
stores), and `"mixed"` (`e4m3` by `e2m1`, two elements per byte of `b`) with `M = K = 256` (two row
tiles, two `k` steps each, the scales' `k` offset advancing).

**The `tl.dot_scaled` values are assumed, not proved.** `tl.dot_scaled` is a platform operation
(`PlatQ.dotScaled`): only the oracle can supply its value. The oracle answers each step with the
exact `acc + Σ_k scaled(a)·scaled(b)` (each element's value times `2^(s − 127)` of its scale byte
`s`, decoded by the library's `scaledValue` from the request's own unpacked bits), and only when
that value is a `float32` (it answers nothing otherwise). The elements are small (`|a| ≤ 2`,
`|b| ≤ 3`, multiples of `1/2`) and the scale bytes are `126 … 128`, so every product is a multiple
of `2^-3` and every partial sum is at most `24·256 < 2^13` in magnitude: an exact `float32`, the only value
any correct device gives whatever its order of accumulation (`Device.DotScaledE4m3`,
`Device.DotScaledE4m3E2m1` bound the error by `relDot16 128·(|acc| + Σ|a·b|) + absDot16 128`, met
with `0` by the exact value). The padded lanes (rows of `b` past `N`) hold zero bits with zero scale
bytes, value `0`. The verdicts are `conditional n`, `n` the number of `tl.dot_scaled` steps
(programs × `K / 128`).

The check is `Post`'s, element by element, computed in the test from the inputs (the values the
test wrote, and the scale layout of the host, `[R / 128, K / 128, 32, 16]` with entry
`(r % 32, 4·((r % 128) / 32) + (k % 128) / 32)`): with `S = Σ_k p k` exact, `μ` (`stepMag`) and
`tolOut` (the step recursion `tol`, with the claim's `rho`, `alpha`), the element must be the
`float16` rounding of a finite `float32` `v` with `|v − S| ≤ vTol = 2^-14·μ + 2^-108` (searched
among the 129 `float32` values nearest `S`), and, when `|S| + tolOut ≤ 65504`, finite and within
`cTol = 2^-11·|S| + 2^-14·μ + 2^-24` of `S`. These are `Post`'s own definitions, restated on the
test's values. Every byte outside the matrix `c` (inputs, scales, sentinels around `c`) is
unchanged. `Pre`'s `ValsOk` is guarded: every element finite, every scale byte in `1 … 254`, every
scaled value `0` or within `[2^-126, 2^100]`, and room in `float32` (every `Σ|p| ≤ 2^20`). -/

open TVA TVA.Validation

namespace BlockScaledTest

open Claims.BlockScaledMxfp8

/-- A platform that relates nothing: the dots are answered by the oracle, never by `P0`. -/
def P0 : Platform where
  math _ _ _ _ := False
  div _ _ _ _ := False
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-! ## The oracle: exact scaled dots -/

/-- The fixed point of the oracle's integers: every scaled element times `2^8` is an integer. -/
def E : Nat := 8

/-- A rational times `2^n`, when that is an integer. -/
def scaleInt (q : Rat) (n : Nat) : Option Int :=
  let z := q * (2 ^ n : Nat)
  if z.den == 1 then some z.num else none

/-- An integer as a `float32`, divided by `2^(2E)`, when the `float32` holds it exactly. -/
def f32Of (z : Int) : Option F32 :=
  let q : Rat := (z : Rat) / ((2 ^ (2 * E) : Nat) : Rat)
  let v : F32 := FloatV.ofRat q
  if v.isFinite && v.toRat == q then some v else none

/-- The exact `acc + Σ_k scaled(a[i,k])·scaled(b[k,j])` of a `tl.dot_scaled` request, from its own
unpacked bits and scale bits: `none` unless every scaled element is a multiple of `2^-E` and every
result an exact `float32`. -/
def exactDotScaled : (q : PlatQ) → Option q.Out
  | @PlatQ.dotScaled cfg M N K a sa b sb acc => do
    let A ← ((List.finRange M).flatMap fun i => (List.finRange K).map fun k =>
        (a.get2 i k, sa.map fun t => t.get2 i k)).toArray.mapM fun (x, s) =>
      (scaledValue cfg.lhs cfg.lhsScale x s).bind (scaleInt · E)
    let B ← ((List.finRange K).flatMap fun k => (List.finRange N).map fun j =>
        (b.get2 k j, sb.map fun t => t.get2 k j)).toArray.mapM fun (x, s) =>
      (scaledValue cfg.rhs cfg.rhsScale x s).bind (scaleInt · E)
    let C ← ((List.finRange M).flatMap fun i => (List.finRange N).map fun j => acc.get2 i j).toArray.mapM
      fun v : F32 => if v.isFinite then scaleInt v.toRat (2 * E) else none
    let R ← (Array.range (M * N)).mapM fun e =>
      let i := e / N
      let j := e % N
      f32Of ((List.range K).foldl (fun s k => s + A[i * K + k]! * B[k * N + j]!) C[e]!)
    some (Tensor.of2 fun i j => R[i.1 * N + j.1]!)
  | _ => none

/-- The oracle: the exact scaled dot at every platform request; nothing else. -/
def O : Oracle := fun _ _ q => match q with
  | .plat q => exactDotScaled q
  | _ => none

/-! ## The inputs -/

/-- Twice the value of element `(i, k)` of `a` (`e4m3`): `-2 … 2`, i.e. values `-1 … 1`. -/
def a2 (i k : Nat) : Int := 2 * (((3 * i + 5 * k) % 5 : Nat) - (2 : Int))

/-- Twice the value of element `(j, k)` of `b`: for `"mxfp8"` (`e4m3`) values `-2 … 2`; for
`"mixed"` (`e2m1`) values `-2, -1, -1/2, 0, 1/2, 1, 3/2, 3`. -/
def b2 (mixed : Bool) (j k : Nat) : Int :=
  if mixed then [-4, -2, -1, 0, 1, 2, 3, 6].getD ((7 * k + 3 * j + 1) % 8) 0
  else 2 * (((7 * k + 3 * j + 1) % 5 : Nat) - (2 : Int))

/-- The scale bytes of row `r` of `a` and of `b`, group `g = k / 32`: `126 … 128`. -/
def saByte (r g : Nat) : Nat := 126 + (r + 2 * g) % 3
def sbByte (r g : Nat) : Nat := 126 + (2 * r + g + 1) % 3

/-- The byte of the host's scale tensor (`[R / 128, K / 128, 32, 16]`, contiguous) holding the
scale of element `(r, k)`: entry `[r / 128, k / 128, r % 32, 4·((r % 128) / 32) + (k % 128) / 32]`
(the 128 × 4 blocks of the `tcgen05` scale layout). -/
def scaleByte (K r k : Nat) : Nat :=
  ((r / 128 * (K / 128) + k / 128) * 32 + r % 32) * 16 + (4 * ((r % 128) / 32) + (k % 128) / 32)

/-- The `e4m3` bits of `x/2`. -/
def e4m3Bits (x : Int) : Nat := (FloatV.ofRat ((x : Rat) / 2) : F8E4).pat

/-- The `e2m1` bits of `x/2` (`0` if none). -/
def e2m1Bits (x : Int) : Nat := ((List.range 16).find? fun p => Format.e2m1.toRat p == (x : Rat) / 2).getD 0

/-- The sentinel in `c`'s storage and around it (`c` is `torch.empty`: `Pre` asks nothing of it). -/
def sentinel : Rat := -1 / 1024

/-- Sentinel elements on each side of `c`'s storage. -/
def pad : Nat := 4

/-- The bytes of `b`'s storage: element `(j, k)` at byte `j·K + k`, or for `"mixed"` the nibble
`k % 2` (low first) of byte `j·(K / 2) + k / 2`. -/
def bBytes (d : Inst) : List (Nat × Nat) :=
  if d.mixed then
    (List.range d.N).flatMap fun j => (List.range (d.K / 2)).map fun h =>
      (j * (d.K / 2) + h, e2m1Bits (b2 true j (2 * h)) + 16 * e2m1Bits (b2 true j (2 * h + 1)))
  else
    (List.range d.N).flatMap fun j => (List.range d.K).map fun k => (j * d.K + k, e4m3Bits (b2 false j k))

/-- The scale bytes of a matrix with `R` rows: `(byte index, value)`. -/
def scaleBytes (K R : Nat) (byte : Nat → Nat → Nat) : List (Nat × Nat) :=
  (List.range R).flatMap fun r => (List.range (K / 32)).map fun g => (scaleByte K r (32 * g), byte r g)

def initOf (d : Inst) : CMem :=
  let m := CMem.ofElems ∅ .f8e4nv ((List.range d.M).flatMap fun i : Nat => (List.range d.K).map fun k : Nat =>
    ((d.a.ptr (i * d.K + k : Nat)).addr, DType.f8e4nv.ofBits (e4m3Bits (a2 i k))))
  let m := CMem.ofElems m (bDT d.mixed) ((bBytes d).map fun (e, v) => ((d.b.ptr e).addr, (bDT d.mixed).ofBits v))
  let m := CMem.ofElems m .u8 ((scaleBytes d.K d.M saByte).map fun (e, v) => ((d.a_scale.ptr e).addr, DType.u8.ofBits v))
  let m := CMem.ofElems m .u8 ((scaleBytes d.K d.N sbByte).map fun (e, v) => ((d.b_scale.ptr e).addr, DType.u8.ofBits v))
  CMem.ofElems m .f16 ((List.range (d.c.len + 2 * pad)).map fun e : Nat =>
    ((d.c.ptr ((e : Int) - pad)).addr, (FloatV.ofRat sentinel : F16)))

def runOf (d : Inst) : LRes (launch d).size × List Record :=
  execLaunch ((launch d).env P0) O fill0 100000 (model d) (launch d) (initOf d)

/-! ## The expectation (`Post`, on the test's values) -/

/-- Scaled element `(r, k)` times `2^4`, an integer: `x2/2 · 2^(s − 127) · 2^4 = x2 · 2^(s − 124)`. -/
def sInt (x2 : Int) (s : Nat) : Int := x2 * 2 ^ (s - 124)

/-- Row `i` of `a`, scaled, times `2^4`: `scaled(a[i, k])·2^4` for `k < K`. -/
def aRow (K i : Nat) : Array Int := (Array.range K).map fun k => sInt (a2 i k) (saByte i (k / 32))

/-- Row `j` of `b`, scaled, times `2^4`. -/
def bRow (mixed : Bool) (K j : Nat) : Array Int :=
  (Array.range K).map fun k => sInt (b2 mixed j k) (sbByte j (k / 32))

/-- `(S, μ, tolOut)` of an element from its scaled rows: the exact sum `S = Σ_k p k`
(`p k = scaled(a[i,k])·scaled(b[j,k])`), `stepMag = Σ_{t<T} (|partSum t| + chunkAbs t)` and the
step recursion `tol (t + 1) = tol t + rho·(|partSum t| + tol t + chunkAbs t) + alpha` over the
`T = K / 128` steps, as `Post` defines them (`ref`, `magOut`, `tolOut`). -/
def statsRows (K : Nat) (ar br : Array Int) : Rat × Rat × Rat := Id.run do
  let mut part : Int := 0
  let mut mu : Rat := 0
  let mut tO : Rat := 0
  for t in [0:K / 128] do
    let mut s : Int := 0
    let mut ab : Nat := 0
    for e in [0:128] do
      let p := ar[128 * t + e]! * br[128 * t + e]!
      s := s + p
      ab := ab + p.natAbs
    let P : Rat := (part : Rat) / 256
    let C : Rat := (ab : Rat) / 256
    mu := mu + |P| + C
    tO := tO + rho * (|P| + tO + C) + alpha
    part := part + s
  return ((part : Rat) / 256, mu, tO)

/-- The table of `(S, μ, tolOut)` of every element `(i, j)`, at `i·N + j`. -/
def table (d : Inst) : Array (Rat × Rat × Rat) :=
  let A := (Array.range d.M).map (aRow d.K)
  let B := (Array.range d.N).map (bRow d.mixed d.K)
  (Array.range (d.M * d.N)).map fun e => statsRows d.K A[e / d.N]! B[e % d.N]!

/-- The `float32` patterns near `p0`, nearest first: `p0, p0 + 1, p0 − 1, …, p0 ± 64`. -/
def near (p0 : Nat) : List Int :=
  (List.range 129).map fun n => (p0 : Int) + (if n % 2 = 0 then ((n / 2 : Nat) : Int) else -(((n + 1) / 2 : Nat) : Int))

/-- `OutSpec` for element `(i, j)` holding `c`, with expected sum `S` (and `μ`, `tolOut`). -/
def elemOk (S mu tO : Rat) (c : F16) : Bool :=
  let vTol := (1 / 2 ^ 14 : Rat) * mu + 1 / 2 ^ 108
  let cTol := (1 / 2 ^ 11 : Rat) * |S| + (1 / 2 ^ 14 : Rat) * mu + 1 / 2 ^ 24
  let v0 : F32 := FloatV.ofRat S
  let c1 := (near v0.pat).any fun p =>
    if p < 0 ∨ 2 ^ 32 ≤ p then false else
      let v : F32 := FloatV.ofPat p.toNat
      let r : F16 := FloatV.cvt v
      v.isFinite && |v.toRat - S| ≤ vTol && (r == c || (r.isNaN && c.isNaN))
  let c2 := if |S| + tO ≤ 65504 then c.isFinite && |c.toRat - S| ≤ cTol else true
  c1 && c2

/-- The final memory against `Post`: every element of `c` as `elemOk` says for the expectation
`stat`; every byte outside it unchanged, and no byte added. -/
def agreeOf (d : Inst) (init : CMem) (stat : Nat → Nat → Rat × Rat × Rat) (cf : CConfig (launch d).size) :
    Option String :=
  let m := CMem.toMem cf.mem
  let cs := (List.range d.M).flatMap fun i : Nat => (List.range d.N).map fun j : Nat =>
    (i, j, (d.c.ptr (i * d.N + j : Nat)).addr)
  let badOut := cs.filter fun (i, j, a) => match m.read .f16 a with
    | some x => let (S, mu, tO) := stat i j; !elemOk S mu tO x
    | none => true
  let cBytes : Std.HashSet Int := cs.foldl (fun s (_, _, a) => (s.insert a).insert (a + 1)) {}
  let badElse := init.toList.filter fun (a, b) => !cBytes.contains a && cf.mem[a]? != some b
  if !badOut.isEmpty then some s!"c elements {(badOut.take 8).map fun (i, j, _) => (i, j)} differ from Post"
  else if !badElse.isEmpty then some s!"{badElse.length} bytes outside the matrix c changed"
  else if cf.mem.size != init.size then some "the launch wrote outside the initial memory"
  else none

def verdictOf (d : Inst) (stat : Nat → Nat → Rat × Rat × Rat) : Verdict :=
  Verdict.ofLaunch (agreeOf d (initOf d) stat) (runOf d)

/-! ## The cases -/

/-- `"mxfp8"`, `M = 128`, `N = 384`, `K = 128`: two programs (column tiles; the second has `128`
valid columns of `256`), one step each. -/
def mx : Inst := hostInst false 128 384 128
theorem mx_ok : mx.ok := hostInst_ok (by norm_num)

/-- `"mixed"`, `M = 256`, `N = 128`, `K = 256`: two programs (row tiles; each column tile half
outside `c`), two steps each. -/
def mix : Inst := hostInst true 256 128 256
theorem mix_ok : mix.ok := hostInst_ok (by norm_num)

end BlockScaledTest

open BlockScaledTest Claims.BlockScaledMxfp8

-- The grids: two programs each.
#guard (launch mx).size == 2 && (launch mix).size == 2

-- `Pre`: every buffer aligned (`16` bytes, `Inst.ok`), inside `[0, 2^64)`, and every byte present
-- in `initOf`; every scale byte written exactly once (the layout is a bijection).
#guard [mx, mix].all fun d =>
  let m := CMem.toMem (initOf d)
  let ok {dt : DType} (b : Buf dt) : Bool :=
    0 ≤ b.base && b.base + b.len * dt.bytes ≤ 2 ^ 64 && b.base % 16 == 0 &&
    (List.range b.len).all fun e : Nat => (m.read dt (b.ptr e).addr).isSome
  ok d.a && ok d.b && ok d.a_scale && ok d.b_scale && ok d.c &&
  ((scaleBytes d.K d.M saByte).map (·.1)).mergeSort == List.range d.a_scale.len &&
  ((scaleBytes d.K d.N sbByte).map (·.1)).mergeSort == List.range d.b_scale.len &&
  ((bBytes d).map (·.1)).mergeSort == List.range d.b.len

-- The encodings are exact (the elements are finite with the intended values), every scale byte is
-- in `1 … 254`, and every scaled value is `0` or in `[2^-126, 2^100]` (`InDom`).
#guard (List.range 5).all fun z : Nat =>
  let x : Int := 2 * ((z : Int) - 2)
  Format.e4m3.isFinite (e4m3Bits x) && Format.e4m3.toRat (e4m3Bits x) == (x : Rat) / 2
#guard [-4, -2, -1, 0, 1, 2, 3, 6].all fun x : Int =>
  Format.e2m1.isFinite (e2m1Bits x) && Format.e2m1.toRat (e2m1Bits x) == (x : Rat) / 2 && e2m1Bits x < 16
#guard (List.range 3).all fun s => let b := 126 + s; 1 ≤ b && b ≤ 254 &&
  (List.range 11).all fun z => let x : Rat := ((z : Int) - 4 : Int) / 2
    x == 0 || (1 / 2 ^ 126 ≤ |x| * 2 ^ (b - 126) / 2 && |x| * 2 ^ (b - 126) / 2 ≤ 2 ^ 100)

/-- The checks on one instance, from one run and one table: the verdict (`Post`), `ValsOk.room`
(every `Σ|p| ≤ μ ≤ 2^20`), some nonzero expectation, none the sentinel, and the wrong
expectations (`wrong`, each built from the table) all caught. -/
def checkOf (d : Inst) (n : Nat) (wrong : Array (Rat × Rat × Rat) → List (Nat → Nat → Rat × Rat × Rat)) : Bool :=
  let tbl := table d
  let stat := fun i j => tbl[i * d.N + j]!
  let r := runOf d
  let init := initOf d
  let v := Verdict.ofLaunch (agreeOf d init stat) r
  v == .conditional n && v.agrees && v.assumed > 0 &&
  tbl.all (fun (S, mu, _) => mu ≤ 2 ^ 20 && S != sentinel) && tbl.any (fun (S, _, _) => S != 0) &&
  (wrong tbl).all fun w => (Verdict.ofLaunch (agreeOf d init w) r).isDiscrepancy

/-- The sum of row `i` of `a` against row `j` of `b`, over `k < k1`, with `b`'s scales read from
row `j'`. -/
def sumWith (mixed : Bool) (i j j' k1 : Nat) : Rat :=
  ((((List.range k1).map fun k =>
    sInt (a2 i k) (saByte i (k / 32)) * sInt (b2 mixed j k) (sbByte j' (k / 32))).sum : Int) : Rat) / 256

def verdictMx : Verdict := let tbl := table mx; verdictOf mx fun i j => tbl[i * mx.N + j]!
def verdictMix : Verdict := let tbl := table mix; verdictOf mix fun i j => tbl[i * mix.N + j]!

#eval IO.println (verdictMx.line "block_scaled_matmul_mxfp8/mxfp8")
#eval IO.println (verdictMix.line "block_scaled_matmul_mxfp8/mixed")
-- Agreement resting on one assumed answer per `tl.dot_scaled` step (programs × `K / 128`), and
-- a wrong expectation is caught (`agree` is not vacuous): one element off by `1/4`; row 0 with
-- `b`'s scales read from row `j + 1` (a wrong scale layout); for `"mixed"`, row 0 without the
-- second `k` step.
#guard checkOf mx 2 fun tbl =>
  [fun i j => let (S, mu, tO) := tbl[i * mx.N + j]!; if i = 127 ∧ j = 383 then (S + 1 / 4, mu, tO) else (S, mu, tO),
   fun i j => let (S, mu, tO) := tbl[i * mx.N + j]!; (if i = 0 then sumWith false i j (j + 1) 128 else S, mu, tO)]
#guard checkOf mix 4 fun tbl =>
  [fun i j => let (S, mu, tO) := tbl[i * mix.N + j]!; if i = 255 ∧ j = 127 then (S + 1 / 4, mu, tO) else (S, mu, tO),
   fun i j => let (S, mu, tO) := tbl[i * mix.N + j]!; (if i = 0 then sumWith true i j (j + 1) 256 else S, mu, tO),
   fun i j => let (S, mu, tO) := tbl[i * mix.N + j]!; (if i = 0 then sumWith true i j j 128 else S, mu, tO)]
