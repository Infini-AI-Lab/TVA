import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `layer_norm_fwd_bf16` (fla's `layer_norm_fwd_kernel`, `BT = 32` rows
per program) run by `execLaunch` on three small covered instances, the claim's three paths. Not a
test against the device (phase 4): it checks that `exec` runs the model end to end, and that the
final memory is what `Post` says, both halves: for each row, with a residual `res_out` bit for
bit, then the stored mean, `rstd` and every output element within `Post`'s tolerances of the exact
real LayerNorm of `s` (`x + residual` rounded to `float32`, or `x`) with the weight and bias of the
row's group, all computed here from the inputs as rationals; and every byte outside the outputs'
elements unchanged (the inputs, and sentinels around `y`, `res_out`, `mean`, `rstd`), with no
byte added.

* `gn_exact`: `GroupNorm` without a residual: `T = 3` (one program, `29` masked rows), `D = BD =
  16`, `G = 2` (rows `0, 2` use group `0`'s weights, row `1` group `1`'s), `eps = 1/4` (the double
  `0x3FD0000000000000`; `Inst.ok` covers any `eps` in `[2^-125, 2^126]`). Row `0` is constant
  (`rstd = 2`), row `1` is `-1/2 ± 1` on `12` lanes and `-1/2` on `4` (variance `3/4`, `rstd = 1`),
  row `2` is `5 ± 3` on `4` lanes, `5 ± 2` on `6`, `5` on `6` (variance `15/4`, `rstd = 1/2`).
* `ln_res_exact`: `LayerNorm` with a residual (`HAS_RESIDUAL = STORE_RESIDUAL_OUT = True`), `T = 2`,
  `D = BD = 16`, `G = 1`, `eps = 1/4`: the sums `s = x + residual` are rows `1` and `2` of
  `gn_exact`, split into two `bfloat16` parts.
* `ln_res_host`: `T = 33`, `D = 12` (`BD = 16`: two programs, the second with `31` masked rows;
  `4` masked columns), with a residual, `eps = 1e-5` (the host's, `eps1e5`). (The host's test
  shape `33 × 50`, `BD = 64`, runs about 20 minutes per launch in the interpreter, against 24 s
  here: the cost grows about as `BD³`.) The inputs have more bits than `bfloat16` keeps of their sum: `res_out` checks the
  rounding of `s` to nearest even, ties included (`s = 1 + 2^-8` and its like).

On `gn_exact` and `ln_res_exact` every `float32` step is exact (the divisions by `16`, `sqrt` of
`1/4`, `1`, `4`, `1/·` of `1/2`, `1`, `2`; the masked rows' `sqrt(0 + 1/4)`), so the platform
answers are the only values any correct device gives, and `x_hat ∈ {0, ±1, ±3/2}` makes the
outputs exact. On `ln_res_host` the divisions by `12`, `tl.sqrt` and `1/·` are inexact: the
oracle answers them with the correctly rounded values (`FloatV.sqrtRn`, `FloatV.divRn`), one
value a correct device may give. The platform requests are per program: the mean's and the
variance's divisions by `D`, `tl.sqrt`, `1 / ·` (each a `[32]` tensor): `4` per program, so
`conditional 4`, `4`, `8`. Trees and fusion choices use the solvers' defaults.

How the tolerance is checked. `rstdRef = 1/sqrt(varRef + eps)` and `rmsS = sqrt(Σ s²/D)` are
bracketed from integer square roots (`rlo ≤ rstdRef ≤ rhi`, `qlo ≤ rmsS ≤ qhi`; `rlo = rhi` on the
exact cases). On the branch `4·cond ≤ 1`, `cond`, `rstdTol` and `zTol` are increasing in both, and
`yTol`'s terms other than `2^-7·|yRef|` too; each distance `|c − ref|` is convex in `rstdRef`
(`ref` is `rstdRef` or `zRef·w + b`, affine in it). The test requires `4·cond(rhi, qhi) ≤ 1`
(every value of the brackets is on that branch), and checks
`max(|c − ref(rlo)|, |c − ref(rhi)|) ≤ tol(rlo, qlo)`, with `|yRef|` in `yTol` replaced by its
minimum over the bracket: never looser than `Post`. `L = log₂ BD` is computed here (`Nat.log2`),
the claim's `depth` (`Nat.clog 2`) on a power of two. -/

open TVA TVA.Validation Claims.LayerNormFwd

/-- A platform whose square root and division are the IEEE ones. (`exec` does not consult it:
the answers come from `O`; it says what those answers assume.) -/
def P0 : Platform where
  math fn _ x r := fn = .sqrt ∧ r = FloatV.sqrtRn x
  div _ x y r := r = FloatV.divRn x y
  dot := @fun _ _ _ _ _ _ _ _ => False
  cvt _ _ _ _ _ := False
  atomicAdd _ _ _ _ := False
  dotScaled := @fun _ _ _ _ _ _ _ _ _ _ => False

/-- The oracle: the correctly rounded `tl.sqrt` and quotient, lane by lane; nothing else. -/
def O : Oracle := fun _ _ q => match q with
  | .plat (.math .sqrt _ x) => some (x.map FloatV.sqrtRn)
  | .plat (.div _ x y) => some (Tensor.map2 FloatV.divRn x y)
  | _ => none

/-- Disabled lanes read zero bits. -/
def fill0 : Fill := fun d _ _ => d.ofBits 0

/-- A sentinel in the outputs before the launch, and just outside them. -/
def sentinel : Rat := -7

/-- Sentinel elements on each side of each output. -/
def pad : Nat := 4

/-- One case: the sizes, the residual flag, `eps` (the double's pattern, and its value written
independently), the inputs as rationals (each exact in `bfloat16`, checked below): `x`,
`residual`, and `weight`, `bias` by their index in `[G·D]`. -/
structure Case where
  T : Nat
  D : Nat
  BD : Nat
  G : Nat
  hasRes : Bool
  epsBits : Nat
  epsQ : Rat
  xq : Nat → Nat → Rat
  rq : Nat → Nat → Rat
  wq : Nat → Rat
  bq : Nat → Rat

def Case.inst (c : Case) : Inst := mkInst c.T c.D c.BD c.G c.hasRes (FloatV.ofPat c.epsBits)

def bfv (q : Rat) : BF16 := FloatV.ofRat q

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .bf16 ((List.range (c.T * c.D)).map fun k : Nat =>
    ((d.x.ptr k).addr, bfv (c.xq (k / c.D) (k % c.D))))
  let m := if c.hasRes then CMem.ofElems m .bf16 ((List.range (c.T * c.D)).map fun k : Nat =>
    ((d.res.ptr k).addr, bfv (c.rq (k / c.D) (k % c.D)))) else m
  let m := CMem.ofElems m .bf16 ((List.range (c.G * c.D)).map fun k : Nat => ((d.w.ptr k).addr, bfv (c.wq k)))
  let m := CMem.ofElems m .bf16 ((List.range (c.G * c.D)).map fun k : Nat => ((d.b.ptr k).addr, bfv (c.bq k)))
  let m := CMem.ofElems m .bf16 ((List.range (c.T * c.D + 2 * pad)).map fun k : Nat =>
    ((d.y.ptr ((k : Int) - pad)).addr, bfv sentinel))
  let m := if c.hasRes then CMem.ofElems m .bf16 ((List.range (c.T * c.D + 2 * pad)).map fun k : Nat =>
    ((d.res_out.ptr ((k : Int) - pad)).addr, bfv sentinel)) else m
  let m := CMem.ofElems m .f32 ((List.range (c.T + 2 * pad)).map fun k : Nat =>
    ((d.mean.ptr ((k : Int) - pad)).addr, (FloatV.ofRat sentinel : F32)))
  CMem.ofElems m .f32 ((List.range (c.T + 2 * pad)).map fun k : Nat =>
    ((d.rstd.ptr ((k : Int) - pad)).addr, (FloatV.ofRat sentinel : F32)))

def Case.run (c : Case) : LRes (launch c.inst).size × List Record :=
  execLaunch ((launch c.inst).env P0) O fill0 1000 (model c.inst) (launch c.inst) c.init

/-! ## The reference and the tolerances, from the inputs -/

/-- A bracket `lo ≤ sqrt q ≤ hi` (`q ≥ 0`), from the integer square root at `2^-40` resolution;
`lo = hi` when `q` is the square of a rational of that resolution. -/
def sqrtBounds (q : Rat) : Rat × Rat :=
  let k : Nat := 2 ^ 40
  let n := q.num.toNat * q.den * k * k
  let s := Nat.sqrt n
  let den : Rat := ((q.den * k : Nat) : Rat)
  if s * s == n then ((s : Rat) / den, (s : Rat) / den) else ((s : Rat) / den, ((s + 1 : Nat) : Rat) / den)

/-- `s j`: with a residual, the `float32` rounding of the exact sum `x + residual` (the IEEE
addition of the two widened `bfloat16` values); without, `x`. -/
def Case.s (c : Case) (t j : Nat) : Rat :=
  if c.hasRes then (FloatV.ofRat (c.xq t j + c.rq t j) : F32).toRat else c.xq t j

/-- The weight and bias of row `t`'s group at column `j`. -/
def Case.w (c : Case) (t j : Nat) : Rat := c.wq ((t % c.G) * c.D + j)
def Case.b (c : Case) (t j : Nat) : Rat := c.bq ((t % c.G) * c.D + j)

/-- Row `t`'s exact quantities: `meanRef`, `Σ|s|/D`, the brackets of `rstdRef` and `rmsS`. -/
structure RowRef where
  mean : Rat
  absMean : Rat
  rlo : Rat
  rhi : Rat
  qlo : Rat
  qhi : Rat

def Case.row (c : Case) (t : Nat) : RowRef :=
  let ss := (List.range c.D).map (c.s t)
  let n : Rat := c.D
  let mean := ss.sum / n
  let var := (ss.map fun x => (x - mean) ^ 2).sum / n
  let (slo, shi) := sqrtBounds (var + c.epsQ)
  let (qlo, qhi) := sqrtBounds ((ss.map fun x => x ^ 2).sum / n)
  { mean, absMean := (ss.map fun x => |x|).sum / n, rlo := 1 / shi, rhi := 1 / slo, qlo, qhi }

/-- `L = log₂ BD`. -/
def Case.L (c : Case) : Rat := (Nat.log2 c.BD : Nat)

def Case.meanTol (c : Case) (R : RowRef) : Rat := (c.L + 4) * 2 ^ (-23 : Int) * R.absMean + 2 ^ (-140 : Int)
def Case.eTol (c : Case) (R : RowRef) (t j : Nat) : Rat :=
  c.meanTol R + 2 ^ (-23 : Int) * (|c.s t j| + R.absMean)
def Case.cond (c : Case) (R : RowRef) (r q : Rat) : Rat :=
  (c.meanTol R + 2 ^ (-23 : Int) * (R.absMean + q)) * r
def Case.gam (c : Case) : Rat := (c.L + 10) * 2 ^ (-23 : Int) + 2 ^ (-136 : Int) / c.epsQ
/-- `rstdTol` on the branch `4·cond ≤ 1`. -/
def Case.rstdTol (c : Case) (R : RowRef) (r q : Rat) : Rat := 2 * (c.cond R r q + c.gam) * r
def Case.zTol (c : Case) (R : RowRef) (t j : Nat) (r q : Rat) : Rat :=
  |c.s t j - R.mean| * c.rstdTol R r q + c.eTol R t j * (r + c.rstdTol R r q)
def Case.yRef (c : Case) (R : RowRef) (t j : Nat) (r : Rat) : Rat :=
  (c.s t j - R.mean) * r * c.w t j + c.b t j
/-- `yTol`, with `yAbs` standing for `|yRef|`. -/
def Case.yTol (c : Case) (R : RowRef) (t j : Nat) (r q yAbs : Rat) : Rat :=
  2 ^ (-7 : Int) * yAbs + 4 * |c.w t j| * c.zTol R t j r q + 2 ^ (-21 : Int) * |c.b t j| +
    2 ^ (-120 : Int)

/-- `|f r|`'s minimum over `[lo, hi]` for an affine `f`. -/
def absMin (f : Rat → Rat) (lo hi : Rat) : Rat :=
  if f lo * f hi ≤ 0 then 0 else min |f lo| |f hi|

/-! ## The check -/

/-- The final memory against `Post`, with the expectation computed from the case `e` (`e = c`
for the test). -/
def Case.agreeE (c e : Case) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let badRows := (List.range c.T).filter fun t : Nat =>
    let R := e.row t
    let okBranch := 4 * e.cond R R.rhi R.qhi ≤ 1
    let okRes := !c.hasRes || (List.range c.D).all fun j : Nat =>
      (m.read .bf16 (d.res_out.ptr (idx d t j)).addr).map DType.bf16.toBits ==
        some (DType.bf16.toBits (bfv (e.s t j)))
    let okMean := match m.read .f32 (d.mean.ptr t).addr with
      | some v => v.isFinite && |v.toRat - R.mean| ≤ e.meanTol R
      | none => false
    let okRstd := match m.read .f32 (d.rstd.ptr t).addr with
      | some v => v.isFinite && max |v.toRat - R.rlo| |v.toRat - R.rhi| ≤ e.rstdTol R R.rlo R.qlo
      | none => false
    let okY := (List.range c.D).all fun j : Nat =>
      match m.read .bf16 (d.y.ptr (idx d t j)).addr with
      | some v =>
        let f := e.yRef R t j
        v.isFinite && max |v.toRat - f R.rlo| |v.toRat - f R.rhi| ≤
          e.yTol R t j R.rlo R.qlo (absMin f R.rlo R.rhi)
      | none => false
    !(okBranch && okRes && okMean && okRstd && okY)
  let inside (b : Int) (n : Nat) (a : Int) : Bool := b ≤ a && a < b + (n : Int)
  let isOut (a : Int) : Bool :=
    inside d.y.base (c.T * c.D * 2) a || (c.hasRes && inside d.res_out.base (c.T * c.D * 2) a) ||
      inside d.mean.base (c.T * 4) a || inside d.rstd.base (c.T * 4) a
  let changed := c.init.toList.filter fun (a, b) => !isOut a && cf.mem[a]? != some b
  if !badRows.isEmpty then some s!"rows {badRows} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the outputs changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch c.agree c.run

/-! ## The cases -/

/-- `±1` on `12` of `16` lanes (`6` each), `0` on `4`: variance `3/4`. -/
def dev1 (j : Nat) : Rat := [1, -1, 1, 0, -1, 1, -1, 0, 1, -1, 1, 0, -1, 1, -1, 0].getD j 0

/-- `±3` on `4` lanes, `±2` on `6`, `0` on `6` (each sign equally often): variance `15/4`. -/
def dev2 (j : Nat) : Rat :=
  [3, -3, 2, -2, 0, 3, -3, 2, -2, 0, 2, -2, 0, 0, 0, 0].getD j 0

/-- The double `1/4`. -/
def quarter : Nat := 0x3FD0000000000000

def gnExact : Case :=
  { T := 3, D := 16, BD := 16, G := 2, hasRes := false, epsBits := quarter, epsQ := 1 / 4,
    xq := fun t j => match t with
      | 0 => 3
      | 1 => -1 / 2 + dev1 j
      | _ => 5 + dev2 j,
    rq := fun _ _ => 0,
    wq := fun k => (((k * 7) % 9 : Nat) - 4 : Rat) / 4,
    bq := fun k => (((k * 5) % 7 : Nat) - 3 : Rat) / 8 }

def lnResExact : Case :=
  { T := 2, D := 16, BD := 16, G := 1, hasRes := true, epsBits := quarter, epsQ := 1 / 4,
    xq := fun t j => (if t = 0 then -1 / 2 + dev1 j else 5 + dev2 j) - ((j % 5 : Nat) - 2 : Rat) / 4,
    rq := fun _ j => ((j % 5 : Nat) - 2 : Rat) / 4,
    wq := fun k => (((k * 3) % 7 : Nat) - 3 : Rat) / 2,
    bq := fun k => (((k * 2) % 5 : Nat) - 2 : Rat) / 4 }

/-- The double `1e-5`, its value. -/
def eps1e5Q : Rat := (0x14F8B588E368F1 : Nat) / 2 ^ 69

/-- `33 × 12`, a residual, `eps = 1e-5`. `x` has `8` significant bits (`bfloat16`'s), `residual`
is small, so `s` has more bits than `bfloat16` keeps: `res_out` rounds. Row `t`'s column `0` is
`1 + 2^-8` (a tie, to even `1`) when `t` is even, `1 + 3·2^-8` (a tie, to even `1 + 2^-6`) when
odd. -/
def lnResHost : Case :=
  { T := 33, D := 12, BD := 16, G := 1, hasRes := true, epsBits := 0x3EE4F8B588E368F1,
    epsQ := eps1e5Q,
    xq := fun t j => if j = 0 then 1 else (((t * 7 + j * 13) % 31 : Nat) - 15 : Rat) / 8 + (t % 3 : Nat),
    rq := fun t j => if j = 0 then (if t % 2 = 0 then 1 / 256 else 3 / 256)
      else (((t * 5 + j * 3) % 19 : Nat) - 9 : Rat) / 1024,
    wq := fun k => (((k * 3) % 11 : Nat) - 5 : Rat) / 8,
    bq := fun k => (((k * 5) % 13 : Nat) - 6 : Rat) / 16 }

/-! ## Coverage -/

/-- `Inst.ok` of a case's instance, by the claim's `mkInst_ok`. -/
macro "covered" : tactic => `(tactic| exact mkInst_ok (by norm_num) (by norm_num) ⟨_, rfl⟩
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  ⟨by decide +kernel, by decide +kernel, by decide +kernel⟩)

theorem gnExact_ok : gnExact.inst.ok := by
  show (mkInst 3 16 (2 ^ 4) 2 false (FloatV.ofPat quarter)).ok; covered
theorem lnResExact_ok : lnResExact.inst.ok := by
  show (mkInst 2 16 (2 ^ 4) 1 true (FloatV.ofPat quarter)).ok; covered
theorem lnResHost_ok : lnResHost.inst.ok := by
  show (mkInst 33 12 (2 ^ 4) 1 true eps1e5).ok
  exact mkInst_ok (by norm_num) (by norm_num) ⟨_, rfl⟩ (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) eps1e5_facts

-- `lnResHost`'s instance is the host's own `eps` (`eps1e5`).
#guard lnResHost.inst.eps == eps1e5

def cases : List (String × Case) :=
  [("gn_exact", gnExact), ("ln_res_exact", lnResExact), ("ln_res_host", lnResHost)]

-- The `eps` values are the doubles' (and the `float32` the kernel gets is exact on the exact ones).
#guard cases.all fun (_, c) => (FloatV.ofPat c.epsBits : F64).toRat == c.epsQ
#guard (FloatV.hostF32 gnExact.inst.eps).toRat == 1 / 4

-- `Pre`: the buffers the launch touches aligned and inside `[0, 2^64)`, every element present in
-- `init` (`residual`, `res_out` only with a residual); the inputs encoded exactly, finite, `|x|`,
-- `|residual| ≤ 2^56`, `|weight|`, `|bias| ≤ 2^16`.
#guard cases.all fun (_, c) =>
  let d := c.inst
  let mm := CMem.toMem c.init
  let all16 (b : Buf .bf16) : Bool := b.base % 2 == 0 && b.base + (b.len * 2 : Nat) ≤ 2 ^ 64 &&
    (List.range b.len).all fun k : Nat => (mm.read .bf16 (b.ptr k).addr).isSome
  let all32 (b : Buf .f32) : Bool := b.base % 4 == 0 && b.base + (b.len * 4 : Nat) ≤ 2 ^ 64 &&
    (List.range b.len).all fun k : Nat => (mm.read .f32 (b.ptr k).addr).isSome
  all16 d.x && (!c.hasRes || all16 d.res) && all16 d.w && all16 d.b && all16 d.y &&
    (!c.hasRes || all16 d.res_out) && all32 d.mean && all32 d.rstd
#guard cases.all fun (_, c) =>
  let ok (q : Rat) (B : Rat) := (bfv q).toRat == q && (bfv q).isFinite && |q| ≤ B
  ((List.range c.T).all fun t => (List.range c.D).all fun j =>
    ok (c.xq t j) (2 ^ 56) && (!c.hasRes || ok (c.rq t j) (2 ^ 56))) &&
  ((List.range (c.G * c.D)).all fun k => ok (c.wq k) (2 ^ 16) && ok (c.bq k) (2 ^ 16))

-- On the exact cases the reference is rational: `rstdRef = 2, 1, 1/2` and `1, 1/2` exactly.
#guard [0, 1, 2].map (fun t => ((gnExact.row t).rlo, (gnExact.row t).rhi, (gnExact.row t).mean)) ==
  [(2, 2, 3), (1, 1, -1 / 2), (1 / 2, 1 / 2, 5)]
#guard [0, 1].map (fun t => ((lnResExact.row t).rlo, (lnResExact.row t).rhi, (lnResExact.row t).mean)) ==
  [(1, 1, -1 / 2), (1 / 2, 1 / 2, 5)]
-- On `ln_res_host` the brackets are tight, and `res_out` rounds: `s` is not a `bfloat16` on some
-- lanes (column `0`: the ties, to even).
#guard (List.range 33).all fun t => (lnResHost.row t).rhi - (lnResHost.row t).rlo ≤ 2 ^ (-30 : Int)
#guard (bfv (lnResHost.s 0 0)).toRat == 1 && (bfv (lnResHost.s 1 0)).toRat == 1 + 1 / 64 &&
  (List.range 33).all (fun t => (lnResHost.s t 0) != (bfv (lnResHost.s t 0)).toRat) &&
  ((List.range 33).flatMap fun t => (List.range 12).filter fun j =>
    lnResHost.s t j != (bfv (lnResHost.s t j)).toRat).length > 33

/-- The residual's column `0` forgotten: the expected `res_out` of the odd rows is `1`, not the
rounded tie `1 + 2^-6`. -/
def noTie : Case :=
  { lnResHost with rq := fun t j => if j = 0 then 0 else lnResHost.rq t j }
def oneGroup : Case := { gnExact with G := 1 }
def epsHalf : Case := { lnResExact with epsQ := 1 / 2 }

#eval IO.println (gnExact.verdict.line "layer_norm_fwd_bf16/gn_exact")
#eval IO.println (lnResExact.verdict.line "layer_norm_fwd_bf16/ln_res_exact")
#eval IO.println (lnResHost.verdict.line "layer_norm_fwd_bf16/ln_res_host")
-- Agreement resting on four answers per program (two divisions by `D`, `tl.sqrt`, `1 / ·`); and,
-- on the same run (each command re-runs the launch), not vacuous: a wrong expectation is caught
-- (one group, not two: row 1 with group 0's weights; `eps = 1/2`; the residual's column `0`
-- forgotten).
#guard let r := gnExact.run
  Verdict.ofLaunch gnExact.agree r == .conditional 4 &&
    (Verdict.ofLaunch (gnExact.agreeE oneGroup) r).isDiscrepancy
#guard let r := lnResExact.run
  Verdict.ofLaunch lnResExact.agree r == .conditional 4 &&
    (Verdict.ofLaunch (lnResExact.agreeE epsHalf) r).isDiscrepancy
#guard let r := lnResHost.run
  Verdict.ofLaunch lnResHost.agree r == .conditional 8 &&
    (Verdict.ofLaunch (lnResHost.agreeE noTie) r).isDiscrepancy
