import Claim
import Solvers
import Exec
import Verdict

/-! Smoke test: the model of `layer_norm_fwd_f16` run by `execLaunch` on two small covered
instances. Not a test against the device (phase 4): it checks that `exec` runs the model end to
end, and that the final memory is what `Post` says, both halves: for each row, the stored mean,
`rstd` and every output element within `Post`'s tolerances of the exact real LayerNorm, computed
here from the inputs as rationals; and every byte outside the three outputs unchanged (the
inputs, and sentinels around `y`, `mean` and `rstd`), with no byte added.

* `exact`: `M = 3`, `N = BLOCK_SIZE = 128` (the smallest covered), `eps = 1/4` (the double
  `0x3FD0000000000000`; `Inst.ok` covers any `eps` in `[2^-125, 2^126]`). Row `0` is constant
  (`3`: variance `0`, `rstd = 1/sqrt(1/4) = 2`); row `1` is `-1/2 ± 1` on `96` lanes and `-1/2`
  on `32` (variance `3/4`, `rstd = 1`); row `2` is `5 ± 2` on `120` lanes and `5` on `8` (variance
  `15/4`, `rstd = 1/2`). Every `float32` step is exact on these inputs (the sums are small
  multiples of `1/2`, the divisions by `128`, `sqrt(1/4)`, `sqrt(1)`, `sqrt(4)` and `1/·` of
  `1/2`, `1`, `2` are exact), so the platform answers are the only values any correct device
  gives; `x_hat ∈ {0, ±1}`, and the outputs `x_hat·w + b` are exact in `float16`.
* `host_eps`: the host's `eps = 1e-5` (`mkInst` as the claim writes it), `M = 2`, `N = 384`,
  `BLOCK_SIZE = 512` (`128` masked lanes in each of the three loops), rows that are not
  symmetric (row `1` offset by `64`). Here the divisions by `384`, `var + eps`, `tl.sqrt` and
  `1/·` are inexact: the oracle answers `tl.sqrt` and `/` with the correctly rounded values
  (`FloatV.sqrtRn`, `FloatV.divRn`), one value a correct device may give.

The platform operations are `/` (the mean's and the variance's divisions by `N`, and `1 / sqrt`)
and `tl.sqrt`: four requests per row (one program per row), so the verdicts are `conditional 12`
and `conditional 8`. Trees and fusion choices use the solvers' defaults (one of the results
`Post` must cover).

How the tolerance is checked. `rstdRef = 1/sqrt(varRef + eps)` is irrational in general: the test
brackets it, `rlo ≤ rstdRef ≤ rhi`, from an integer square root (`rlo = rhi` exactly when
`varRef + eps` is a rational square, as on `exact`). Every tolerance of `Post` (`cond`, `rstdTol`,
`zTol`, `yTol`, on the branch `cond ≤ 1/2`) is increasing in `rstdRef`, and each distance
`|c − ref|` is convex in it (`ref` is `rstdRef` itself, or `zRef·w + b`, affine in it). So the test
requires `cond(rhi) ≤ 1/2` (the row is on that branch for every value in the bracket), and checks
`max(|c − ref(rlo)|, |c − ref(rhi)|) ≤ tol(rlo)`: for every `r` in the bracket, `|c − ref(r)| ≤
tol(r)`, so the check is never looser than `Post`. `D = log₂ BLOCK_SIZE` is computed here
(`Nat.log2`), which is the claim's `depth` (`Nat.clog 2`) on a power of two. `Pre`'s `YRoom` is
checked with the same bracket (at `rhi`, where its left side is largest). -/

open TVA TVA.Validation Claims.LayerNormFwdF16

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

/-- One case: the sizes, `eps` (the double's pattern, and its value written independently), the
inputs as rationals (each exact in `float16`, checked below). -/
structure Case where
  M : Nat
  N : Nat
  BLOCK : Nat
  epsBits : Nat
  epsQ : Rat
  xq : Nat → Nat → Rat
  wq : Nat → Rat
  bq : Nat → Rat

def Case.inst (c : Case) : Inst := { mkInst c.M c.N c.BLOCK with eps := FloatV.ofPat c.epsBits }

def f16v (q : Rat) : F16 := FloatV.ofRat q

def Case.init (c : Case) : CMem :=
  let d := c.inst
  let m := CMem.ofElems ∅ .f16 ((List.range (c.M * c.N)).map fun k : Nat =>
    ((d.x.ptr k).addr, f16v (c.xq (k / c.N) (k % c.N))))
  let m := CMem.ofElems m .f16 ((List.range c.N).map fun j : Nat => ((d.w.ptr j).addr, f16v (c.wq j)))
  let m := CMem.ofElems m .f16 ((List.range c.N).map fun j : Nat => ((d.b.ptr j).addr, f16v (c.bq j)))
  let m := CMem.ofElems m .f16 ((List.range (c.M * c.N + 2 * pad)).map fun k : Nat =>
    ((d.y.ptr ((k : Int) - pad)).addr, f16v sentinel))
  let m := CMem.ofElems m .f32 ((List.range (c.M + 2 * pad)).map fun k : Nat =>
    ((d.mean.ptr ((k : Int) - pad)).addr, (FloatV.ofRat sentinel : F32)))
  CMem.ofElems m .f32 ((List.range (c.M + 2 * pad)).map fun k : Nat =>
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

/-- Row `t`'s exact quantities: `meanRef`, `Σ|x|/N`, and the bracket `[rlo, rhi]` of `rstdRef`. -/
structure RowRef where
  mean : Rat
  absMean : Rat
  rlo : Rat
  rhi : Rat

def Case.row (c : Case) (t : Nat) : RowRef :=
  let xs := (List.range c.N).map (c.xq t)
  let n : Rat := c.N
  let mean := xs.sum / n
  let var := (xs.map fun x => (x - mean) ^ 2).sum / n
  let (slo, shi) := sqrtBounds (var + c.epsQ)
  { mean, absMean := (xs.map fun x => |x|).sum / n, rlo := 1 / shi, rhi := 1 / slo }

/-- `D = log₂ BLOCK_SIZE`. -/
def Case.D (c : Case) : Rat := (Nat.log2 c.BLOCK : Nat)

def Case.meanTol (c : Case) (R : RowRef) : Rat := (c.D + 4) * 2 ^ (-23 : Int) * R.absMean + 2 ^ (-140 : Int)
def Case.cond (c : Case) (R : RowRef) (r : Rat) : Rat := c.meanTol R * r
def Case.gam (c : Case) : Rat := (c.D + 8) * (2 ^ (-23 : Int) + 2 ^ (-145 : Int) / c.epsQ)
def Case.mTerm (c : Case) (R : RowRef) (r : Rat) : Rat := c.cond R r / 8 + c.cond R r ^ 2 / 4
/-- `rstdTol` on the branch `cond ≤ 1/2`. -/
def Case.rstdTol (c : Case) (R : RowRef) (r : Rat) : Rat :=
  (c.D * 2 ^ (-21 : Int) + 2 ^ (-140 : Int) / c.epsQ + c.cond R r / 4) * r + 2 ^ (-140 : Int)
def Case.zRef (c : Case) (R : RowRef) (t j : Nat) (r : Rat) : Rat := (c.xq t j - R.mean) * r
/-- `zTol` on the branch `cond ≤ 1/2`. -/
def Case.zTol (c : Case) (R : RowRef) (t j : Nat) (r : Rat) : Rat :=
  (1 + 2 * c.gam) * (|c.zRef R t j r| * (c.gam + c.mTerm R r) + c.cond R r) + 2 ^ (-140 : Int) * (1 + r)
def Case.yRef (c : Case) (R : RowRef) (t j : Nat) (r : Rat) : Rat := c.zRef R t j r * c.wq j + c.bq j
def Case.yTol (c : Case) (R : RowRef) (t j : Nat) (r : Rat) : Rat :=
  2 ^ (-9 : Int) * (|c.wq j| * |c.zRef R t j r| + |c.bq j|) + 2 * |c.wq j| * c.zTol R t j r +
    2 ^ (-24 : Int)

/-- `YRoom` for row `t`, at `rhi` (an upper bound over the bracket). -/
def Case.yRoom (c : Case) (t : Nat) : Bool :=
  let R := c.row t
  (List.range c.N).all fun j =>
    |c.wq j| * (|c.zRef R t j R.rhi| + c.zTol R t j R.rhi) + |c.bq j| ≤ 2 ^ 15

/-- `|v − f r| ≤ tol` for every `r` in `[lo, hi]`, when `f` is affine and `tol` increasing. -/
def within (v : Rat) (f : Rat → Rat) (tol : Rat → Rat) (lo hi : Rat) : Bool :=
  max |v - f lo| |v - f hi| ≤ tol lo

/-! ## The check -/

/-- The final memory against `Post`, with the expectation computed from the case `e` (`e = c`
for the test): for each row, the stored mean, `rstd` and each output element finite and within
tolerance; every byte outside the three outputs' elements unchanged; no byte added. -/
def Case.agreeE (c e : Case) (cf : CConfig (launch c.inst).size) : Option String :=
  let d := c.inst
  let m := CMem.toMem cf.mem
  let badRows := (List.range c.M).filter fun t : Nat =>
    let R := e.row t
    let okBranch := e.cond R R.rhi ≤ 1 / 2
    let okMean := match m.read .f32 (d.mean.ptr t).addr with
      | some v => v.isFinite && |v.toRat - R.mean| ≤ e.meanTol R
      | none => false
    let okRstd := match m.read .f32 (d.rstd.ptr t).addr with
      | some v => v.isFinite && within v.toRat id (e.rstdTol R) R.rlo R.rhi
      | none => false
    let okY := (List.range c.N).all fun j : Nat =>
      match m.read .f16 (d.y.ptr (idx d t j)).addr with
      | some v => v.isFinite && within v.toRat (e.yRef R t j) (e.yTol R t j) R.rlo R.rhi
      | none => false
    !(okBranch && okMean && okRstd && okY)
  let inside (b : Int) (n : Nat) (a : Int) : Bool := b ≤ a && a < b + (n : Int)
  let isOut (a : Int) : Bool :=
    inside d.y.base (c.M * c.N * 2) a || inside d.mean.base (c.M * 4) a || inside d.rstd.base (c.M * 4) a
  let changed := c.init.toList.filter fun (a, b) => !isOut a && cf.mem[a]? != some b
  if !badRows.isEmpty then some s!"rows {badRows} differ from Post"
  else if !changed.isEmpty then some s!"{changed.length} bytes outside the outputs changed"
  else if cf.mem.size != c.init.size then some "bytes were added"
  else none

def Case.agree (c : Case) := c.agreeE c

def Case.verdict (c : Case) : Verdict := Verdict.ofLaunch c.agree c.run

/-! ## The cases -/

/-- `±1` on `96` lanes (`48` each), `0` on `32`. -/
def dev1 (j : Nat) : Rat :=
  if j % 4 = 0 then 1 else if j % 4 = 1 then -1 else if j % 4 = 2 then 0
  else if j % 8 = 3 then 1 else -1

/-- `±2` on `120` lanes (`60` each), `0` on the `8` lanes `j ≡ 0 (mod 16)`. -/
def dev2 (j : Nat) : Rat :=
  if j % 16 = 0 then 0 else if (j % 16 + j / 16) % 2 = 0 then 2 else -2

def exact : Case :=
  { M := 3, N := 128, BLOCK := 128, epsBits := 0x3FD0000000000000, epsQ := 1 / 4,
    xq := fun t j => match t with
      | 0 => 3
      | 1 => -1 / 2 + dev1 j
      | _ => 5 + dev2 j,
    wq := fun j => (((j * 7) % 9 : Nat) - 4 : Rat) / 4,
    bq := fun j => (((j * 5) % 7 : Nat) - 3 : Rat) / 8 }

/-- The double `1e-5`, its value (`0x3EE4F8B588E368F1 = 1.0000000000000000818…·10^-5`). -/
def eps1e5Q : Rat := (0x14F8B588E368F1 : Nat) / 2 ^ 69

def hostEps : Case :=
  { M := 2, N := 384, BLOCK := 512, epsBits := 0x3EE4F8B588E368F1, epsQ := eps1e5Q,
    xq := fun t j => match t with
      | 0 => (((j * 13) % 17 : Nat) - 8 : Rat) / 4
      | _ => 64 + (((j * 11 + 5) % 23 : Nat) - 9 : Rat) / 8,
    wq := fun j => (((j * 3) % 11 : Nat) - 5 : Rat) / 8,
    bq := fun j => (((j * 5) % 13 : Nat) - 6 : Rat) / 16 }

/-! ## Coverage -/

/-- `Inst.ok` of the case's instance, from `mkInst_ok` (every field but `eps`'s) and `eps`'s
facts. -/
theorem Case.ok_of (c : Case) (h : (mkInst c.M c.N c.BLOCK).ok)
    (h1 : (FloatV.ofPat c.epsBits : F64).isFinite = true)
    (h2 : (2 : ℚ) ^ (-125 : ℤ) ≤ (FloatV.ofPat c.epsBits : F64).toRat)
    (h3 : (FloatV.ofPat c.epsBits : F64).toRat ≤ (2 : ℚ) ^ (126 : ℤ)) : c.inst.ok where
  n_ge := h.n_ge
  n_le := h.n_le
  block_pow2 := h.block_pow2
  n_le_block := h.n_le_block
  block_lt := h.block_lt
  m_pos := h.m_pos
  rows_lt := h.rows_lt
  n_dim := h.n_dim
  m_ctx := h.m_ctx
  eps_fin := h1
  eps_ge := h2
  eps_le := h3
  x_len := h.x_len
  w_len := h.w_len
  b_len := h.b_len
  y_len := h.y_len
  mean_len := h.mean_len
  rstd_len := h.rstd_len
  out_in := h.out_in
  y_mean := h.y_mean
  y_rstd := h.y_rstd
  mean_rstd := h.mean_rstd

theorem exact_ok : exact.inst.ok :=
  Case.ok_of exact (mkInst_ok (by norm_num [exact]) (by norm_num [exact]) ⟨7, rfl⟩
    (by norm_num [exact]) (by norm_num [exact]) (by norm_num [exact]) (by norm_num [exact])
    (by norm_num [exact]) (by norm_num [exact]))
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

theorem hostEps_ok : hostEps.inst.ok :=
  Case.ok_of hostEps (mkInst_ok (by norm_num [hostEps]) (by norm_num [hostEps]) ⟨9, rfl⟩
    (by norm_num [hostEps]) (by norm_num [hostEps]) (by norm_num [hostEps])
    (by norm_num [hostEps]) (by norm_num [hostEps]) (by norm_num [hostEps]))
    (by decide +kernel) (by decide +kernel) (by decide +kernel)

-- `hostEps`'s instance is the host's own (`mkInst`, `eps = 1e-5`).
#guard hostEps.inst.eps == eps1e5 && hostEps.inst.x.base == (mkInst 2 384 512).x.base

def cases : List (String × Case) := [("exact", exact), ("host_eps", hostEps)]

-- The `eps` values are the doubles' (and the `float32` the kernel gets is exact on `exact`).
#guard cases.all fun (_, c) => (FloatV.ofPat c.epsBits : F64).toRat == c.epsQ
#guard (FloatV.hostF32 exact.inst.eps).toRat == 1 / 4

-- `Pre`: the six buffers aligned and inside `[0, 2^64)`, every element present in `init`; the
-- inputs encoded exactly (so finite); every row has room (`YRoom`).
#guard cases.all fun (_, c) =>
  let d := c.inst
  let mm := CMem.toMem c.init
  let all16 (b : Buf .f16) : Bool := b.base % 2 == 0 && b.base + (b.len * 2 : Nat) ≤ 2 ^ 64 &&
    (List.range b.len).all fun k : Nat => (mm.read .f16 (b.ptr k).addr).isSome
  let all32 (b : Buf .f32) : Bool := b.base % 4 == 0 && b.base + (b.len * 4 : Nat) ≤ 2 ^ 64 &&
    (List.range b.len).all fun k : Nat => (mm.read .f32 (b.ptr k).addr).isSome
  all16 d.x && all16 d.w && all16 d.b && all16 d.y && all32 d.mean && all32 d.rstd
#guard cases.all fun (_, c) =>
  ((List.range c.M).all fun t => (List.range c.N).all fun j =>
    (f16v (c.xq t j)).toRat == c.xq t j && (f16v (c.xq t j)).isFinite) &&
  ((List.range c.N).all fun j => (f16v (c.wq j)).toRat == c.wq j && (f16v (c.bq j)).toRat == c.bq j)
#guard cases.all fun (_, c) => (List.range c.M).all c.yRoom

-- On `exact` the reference is rational: `rstdRef = 2, 1, 1/2` exactly, means `3, -1/2, 5`.
#guard [0, 1, 2].map (fun t => ((exact.row t).rlo, (exact.row t).rhi, (exact.row t).mean)) ==
  [(2, 2, 3), (1, 1, -1 / 2), (1 / 2, 1 / 2, 5)]
-- On `host_eps` the bracket is tight (`rstdRef` to `2^-30` or so).
#guard [0, 1].all fun t => (hostEps.row t).rhi - (hostEps.row t).rlo ≤ 2 ^ (-30 : Int)

def verdictExact : Verdict := exact.verdict
def verdictHostEps : Verdict := hostEps.verdict

#eval IO.println (verdictExact.line "layer_norm_fwd_f16/exact")
#eval IO.println (verdictHostEps.line "layer_norm_fwd_f16/host_eps")
-- Agreement resting on four answers per row (two divisions by `N`, `tl.sqrt`, `1 / ·`).
#guard verdictExact == .conditional 12
#guard verdictHostEps == .conditional 8

-- Not vacuous: a wrong expectation is caught (`eps = 1/2` instead of `1/4`: every `rstdRef`
-- moves; the weights shifted by one column; row 1's inputs shifted by `1/2` on `host_eps`).
#guard
  let r := exact.run
  (Verdict.ofLaunch (exact.agreeE { exact with epsQ := 1 / 2 }) r).isDiscrepancy &&
  (Verdict.ofLaunch (exact.agreeE { exact with wq := fun j => exact.wq (j + 1) }) r).isDiscrepancy
def shiftedRow1 : Case :=
  { hostEps with xq := fun t j => if t = 1 then hostEps.xq t j + 1 / 2 else hostEps.xq t j }
#guard (Verdict.ofLaunch (hostEps.agreeE shiftedRow1) hostEps.run).isDiscrepancy
