import AttnBwd.Model

/-!
# `attn_bwd_f16`: the specification and the assumptions

For one head `b`, write `Q`, `K`, `V`, `dO` for the head's `N_CTX × HEAD_DIM` blocks of the four
`float16` inputs, `L` for its row of `M` and `Δ` for its row of `delta` (all as the launch finds
them in memory). For a key row `n` and a query row `m`, with `Allowed n m` (`CAUSAL` is off, or
`n ≤ m`):

* `P[n, m] = 2^(Σ_c K[n, c]·Q[m, c] − L[m])` (`0` when not `Allowed`),
* `dS[n, m] = P[n, m]·(Σ_c V[n, c]·dO[m, c] − Δ[m])`,
* `dV[n, j] = Σ_m P[n, m]·dO[m, j]`,
* `dK[n, j] = sm_scale · Σ_m dS[n, m]·Q[m, j]`,
* `dQ[m, j] = LN2 · Σ_n dS[n, m]·K[n, j]` (`LN2` the kernel's `float32` constant).

`Post` says each stored `dV`, `dK`, `dQ` element is the `float16` rounding of a `float32` value
within an explicit tolerance of these exact values, whenever the head's inputs are finite and the
pairs fit (`HeadV.Ok`), and every byte outside the three outputs is unchanged. The tolerances count
a `float16` operand of a `tl.dot` by its lifted magnitude (`lm`: a subnormal at `2^-14`, as
`Rules.Num.DotF16` does since library round 25). With the host's `K`
(`arg_k = k·sm_scale/ln 2`), the forward's `M` (the row's base-2 log-sum-exp) and
`delta = Σ_j O·dO`, these are the gradients of attention (README, Specification).
-/

namespace Kernels.AttnBwd

open TVA

/-! ## The constants -/

/-- `float32`'s unit roundoff, `2^-24`. -/
noncomputable def u32 : ℝ := (Math.Float.Format.binary32.u : ℝ)

/-- `float32`'s smallest subnormal over two, `2^-150`. -/
noncomputable def eta32 : ℝ := (Math.Float.Format.binary32.eta : ℝ)

/-- `float16`'s unit roundoff, `2^-11`. -/
noncomputable def u16 : ℝ := (Math.Float.Format.binary16.u : ℝ)

/-- `float16`'s smallest subnormal over two, `2^-25`. -/
noncomputable def eta16 : ℝ := (Math.Float.Format.binary16.eta : ℝ)

/-- The relative constant of one `tl.dot` over `HEAD_DIM` products (`Rules.Num.relDot16`,
`(HEAD_DIM + 2)·2^-22`). -/
noncomputable def rH (d : Inst) : ℝ := (Rules.Num.relDot16 d.HEAD_DIM : ℝ)

/-- The absolute constant of one `tl.dot` over `HEAD_DIM` products (`Rules.Num.absDot16`,
`(HEAD_DIM + 2)·2^-126`). -/
noncomputable def aH (d : Inst) : ℝ := (Rules.Num.absDot16 d.HEAD_DIM : ℝ)

/-- The relative error of one accumulation step `acc += tl.dot(a, b)` over at most `32`
products: `DotLoop.stepRel relDot16 32 = 34·2^-22 + 3·2^-24·(1 + 34·2^-22)` (about `8.3·10^-6`). -/
noncomputable def rhoA : ℝ := (DotLoop.stepRel Rules.Num.relDot16 32 : ℝ)

/-- The absolute error of one accumulation step: `DotLoop.stepAbs absDot16 32`
(about `34·2^-126`). -/
noncomputable def alphaA : ℝ := (DotLoop.stepAbs Rules.Num.absDot16 32 : ℝ)

/-- The error after `T` accumulation steps whose products' magnitudes sum to `B`:
`((1 + rhoA)^T − 1)·B + T·(1 + rhoA)^T·alphaA` (`DotLoop.bound`, over the reals). -/
noncomputable def accB (T : ℕ) (B : ℝ) : ℝ := ((1 + rhoA) ^ T - 1) * B + T * (1 + rhoA) ^ T * alphaA

/-- The `float32` constant `LN2` (line 381): `0x3f317218 = 1453635/2097152` (`ln2_val`). -/
def ln2F : F32 := FloatV.litF32 (FloatV.ofPat ln2Pat)

/-! ## The values of a head -/

/-- The values of head `b` the kernel reads: rows `r < N_CTX`, columns `c < HEAD_DIM`. -/
structure HeadV where
  /-- `Q[b, r, c]`. -/
  q : Nat → Nat → F16
  /-- `K[b, r, c]` (`arg_k`). -/
  k : Nat → Nat → F16
  /-- `V[b, r, c]`. -/
  v : Nat → Nat → F16
  /-- `DO[b, r, c]`. -/
  dO : Nat → Nat → F16
  /-- `M[b, r]`. -/
  lse : Nat → F32
  /-- `D[b, r]` (`delta`). -/
  dl : Nat → F32

/-- The initial memory holds `hv` in head `b`. -/
def HeadV.Holds (d : Inst) (init : Mem) (b : Nat) (hv : HeadV) : Prop :=
  ∀ r, r < d.N_CTX →
    (∀ c, c < d.HEAD_DIM →
      init.read .f16 (d.Q.ptr ↑(eIdx d b r c)).addr = some (hv.q r c) ∧
      init.read .f16 (d.K.ptr ↑(eIdx d b r c)).addr = some (hv.k r c) ∧
      init.read .f16 (d.V.ptr ↑(eIdx d b r c)).addr = some (hv.v r c) ∧
      init.read .f16 (d.DO.ptr ↑(eIdx d b r c)).addr = some (hv.dO r c)) ∧
    init.read .f32 (d.M.ptr ↑(rIdx d b r)).addr = some (hv.lse r) ∧
    init.read .f32 (d.D.ptr ↑(rIdx d b r)).addr = some (hv.dl r)

/-- Key row `n` and query row `m` interact: `CAUSAL` is off, or `n ≤ m` (the mask
`offs_m >= offs_n`, lines 297 and 350). -/
abbrev Allowed (d : Inst) (n m : Nat) : Prop := d.CAUSAL = false ∨ n ≤ m

/-- The lifted magnitude of a `float16` value, by which `Rules.Num.DotF16` counts a `tl.dot`
operand (`Rules.Num.liftMag`): `0` for `0`, `|x|` for a normal value, `2^-14` for a subnormal one. -/
noncomputable def lm (x : F16) : ℝ := (Rules.Num.liftMag .f16 x.toRat : ℝ)

/-- A bound on the lifted magnitude of a `float16` value of magnitude at most `B`: `0` when
`B < 2^-24` (the value is then `0`), else `max B 2^-14`. -/
noncomputable def lift16 (B : ℝ) : ℝ := if B < 2 ^ (-24 : ℤ) then 0 else max B (2 ^ (-14 : ℤ))

/-- `Σ_c K[n, c]·Q[m, c]`. -/
noncomputable def sQK (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.HEAD_DIM, (hv.k n c).toReal * (hv.q m c).toReal

/-- `Σ_c lm K[n, c]·lm Q[m, c]`: the lifted magnitude of the dot (`Σ_c |K·Q|` when no operand is
subnormal). -/
noncomputable def aQK (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.HEAD_DIM, lm (hv.k n c) * lm (hv.q m c)

/-- `Σ_c V[n, c]·dO[m, c]`. -/
noncomputable def sVD (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.HEAD_DIM, (hv.v n c).toReal * (hv.dO m c).toReal

/-- `Σ_c lm V[n, c]·lm dO[m, c]`, as `aQK`. -/
noncomputable def aVD (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  ∑ c ∈ Finset.range d.HEAD_DIM, lm (hv.v n c) * lm (hv.dO m c)

/-- The error bound of a computed `tl.dot(a, b) − x` (a dot over `HEAD_DIM` products of exact
sum `s` and lifted magnitude sum `a`, then the `float32` subtraction of `x`):
`rH·a + aH + 3u32·(|s| + rH·a + aH + |x|) + 4eta32`. -/
noncomputable def eSub (d : Inst) (s a x : ℝ) : ℝ :=
  rH d * a + aH d + 3 * u32 * (|s| + rH d * a + aH d + |x|) + 4 * eta32

/-- The exact exponent `Σ_c K[n, c]·Q[m, c] − L[m]`. -/
noncomputable def xE (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := sQK d hv n m - (hv.lse m).toReal

/-- The bound on the computed exponent's error. -/
noncomputable def eX (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  eSub d (sQK d hv n m) (aQK d hv n m) (hv.lse m).toReal

/-- The exact `dP[n, m] − Δ[m] = Σ_c V[n, c]·dO[m, c] − Δ[m]`. -/
noncomputable def yE (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := sVD d hv n m - (hv.dl m).toReal

/-- The bound on its computed value's error. -/
noncomputable def eY (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  eSub d (sVD d hv n m) (aVD d hv n m) (hv.dl m).toReal

/-- A bound on the computed `dP − Δ`'s magnitude. -/
noncomputable def yB (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := |yE d hv n m| + eY d hv n m

/-- A bound on the computed `tl.math.exp2`'s magnitude (`Rules.Num.Exp2F32`'s `2^-22` and
`2^-126`, at the largest exponent the error allows). -/
noncomputable def pBr (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  (1 + 2 ^ (-22 : ℤ)) * (2 : ℝ) ^ (xE d hv n m + eX d hv n m) + 2 ^ (-126 : ℤ)

/-- The exact `P[n, m]`. -/
noncomputable def pE (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then (2 : ℝ) ^ xE d hv n m else 0

/-- The exact `dS[n, m] = P[n, m]·(dP[n, m] − Δ[m])`. -/
noncomputable def dsE (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := pE d hv n m * yE d hv n m

/-- The error bound of the `float16` `P` (line 302): the exponent's and `tl.math.exp2`'s errors
(`pBr − 2^x`), and one rounding to `float16`. `0` when not `Allowed` (the masked lanes are
exactly `0`). -/
noncomputable def p16Err (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then pBr d hv n m - (2 : ℝ) ^ xE d hv n m + u16 * pBr d hv n m + eta16 else 0

/-- A bound on the `float16` `P`'s magnitude. -/
noncomputable def p16B (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then (1 + u16) * pBr d hv n m + eta16 else 0

/-- A bound on the `float16` `P`'s lifted magnitude, the operand of line 303's `tl.dot`. -/
noncomputable def p16L (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := lift16 (p16B d hv n m)

/-- The error bound of the `float32` `dS` (line 308), before its rounding to `float16`. -/
noncomputable def dsErr (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  (pBr d hv n m - (2 : ℝ) ^ xE d hv n m) * yB d hv n m + (2 : ℝ) ^ xE d hv n m * eY d hv n m +
    u32 * (pBr d hv n m * yB d hv n m) + eta32

/-- A bound on the `float32` `dS`'s magnitude. -/
noncomputable def dsB (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  (1 + u32) * (pBr d hv n m * yB d hv n m) + eta32

/-- The error bound of the `float16` `dS` (line 309). `0` when not `Allowed`. -/
noncomputable def ds16Err (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then dsErr d hv n m + u16 * dsB d hv n m + eta16 else 0

/-- A bound on the `float16` `dS`'s magnitude. -/
noncomputable def ds16B (d : Inst) (hv : HeadV) (n m : Nat) : ℝ :=
  if Allowed d n m then (1 + u16) * dsB d hv n m + eta16 else 0

/-- A bound on the `float16` `dS`'s lifted magnitude, the operand of lines 310 and 358's `tl.dot`. -/
noncomputable def ds16L (d : Inst) (hv : HeadV) (n m : Nat) : ℝ := lift16 (ds16B d hv n m)

/-- The inputs of head `b` that `Post`'s accuracy covers: every value finite; `|L|` and `|Δ|` at
most `2^126`; and for every `Allowed` pair, the exponent at most `15` with its error
(`xE + eX ≤ 15`, so `P ≤ 2^15` fits `float16` and `float32`) and `|P·(dP − Δ)|`'s bound at most
`2^15` (it fits `float16`). On real inputs `P ≤ 1` (the forward's `M` is the row's log-sum-exp)
and `|dS|` is small. -/
structure HeadV.Ok (d : Inst) (hv : HeadV) : Prop where
  /-- Finite `Q`. -/
  q : ∀ r c, r < d.N_CTX → c < d.HEAD_DIM → (hv.q r c).isFinite = true
  /-- Finite `K`. -/
  k : ∀ r c, r < d.N_CTX → c < d.HEAD_DIM → (hv.k r c).isFinite = true
  /-- Finite `V`. -/
  v : ∀ r c, r < d.N_CTX → c < d.HEAD_DIM → (hv.v r c).isFinite = true
  /-- Finite `dO`. -/
  dO : ∀ r c, r < d.N_CTX → c < d.HEAD_DIM → (hv.dO r c).isFinite = true
  /-- Finite `L`, at most `2^126`. -/
  lse : ∀ r, r < d.N_CTX → (hv.lse r).isFinite = true ∧ |(hv.lse r).toRat| ≤ 2 ^ 126
  /-- Finite `Δ`, at most `2^126`. -/
  dl : ∀ r, r < d.N_CTX → (hv.dl r).isFinite = true ∧ |(hv.dl r).toRat| ≤ 2 ^ 126
  /-- Each `Allowed` pair fits. -/
  fits : ∀ n m, n < d.N_CTX → m < d.N_CTX → Allowed d n m →
    xE d hv n m + eX d hv n m ≤ 15 ∧ pBr d hv n m * yB d hv n m ≤ 2 ^ 15

/-! ## The outputs -/

/-- The number of accumulation steps for key row `n` (lines 415–443): `N_CTX/32`; with `CAUSAL`,
`8` masked steps and `(N_CTX − start_n − 128)/32`, where `start_n = 128·⌊n/128⌋`. -/
def stepsKV (d : Inst) (n : Nat) : Nat :=
  if d.CAUSAL then 8 + (d.N_CTX - 128 * (n / 128) - 128) / 32 else d.N_CTX / 32

/-- The number of accumulation steps for query row `m` (lines 468–496): `N_CTX/32`; with
`CAUSAL`, `8` masked steps and `start_m/32`, where `start_m = 128·⌊m/128⌋`. -/
def stepsQ (d : Inst) (m : Nat) : Nat :=
  if d.CAUSAL then 8 + 128 * (m / 128) / 32 else d.N_CTX / 32

/-- The exact `dV[n, j] = Σ_m P[n, m]·dO[m, j]`. -/
noncomputable def dVE (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  ∑ m ∈ Finset.range d.N_CTX, pE d hv n m * (hv.dO m j).toReal

/-- The tolerance of `dV[n, j]`'s `float32` value: the `float16` `P`'s errors weighted by
`|dO|`, and the accumulation's error over the bounds of those products' lifted magnitudes
(`p16L·lm dO`). -/
noncomputable def dVTol (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  ∑ m ∈ Finset.range d.N_CTX, p16Err d hv n m * |(hv.dO m j).toReal| +
    accB (stepsKV d n) (∑ m ∈ Finset.range d.N_CTX, p16L d hv n m * lm (hv.dO m j))

/-- `Σ_m dS[n, m]·Q[m, j]`. -/
noncomputable def dKS (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  ∑ m ∈ Finset.range d.N_CTX, dsE d hv n m * (hv.q m j).toReal

/-- The tolerance of the accumulated `Σ_m dS[n, m]·Q[m, j]`. -/
noncomputable def dKAcc (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  ∑ m ∈ Finset.range d.N_CTX, ds16Err d hv n m * |(hv.q m j).toReal| +
    accB (stepsKV d n) (∑ m ∈ Finset.range d.N_CTX, ds16L d hv n m * lm (hv.q m j))

/-- The exact `dK[n, j] = sm_scale · Σ_m dS[n, m]·Q[m, j]` (`sm_scale` as the kernel receives it). -/
noncomputable def dKE (d : Inst) (hv : HeadV) (n j : Nat) : ℝ := (smF d).toReal * dKS d hv n j

/-- The tolerance of `dK[n, j]`'s `float32` value: the accumulation's, scaled, and the rounding
of the product by `sm_scale` (line 449). -/
noncomputable def dKTol (d : Inst) (hv : HeadV) (n j : Nat) : ℝ :=
  |(smF d).toReal| * dKAcc d hv n j +
    u32 * (|(smF d).toReal| * (|dKS d hv n j| + dKAcc d hv n j)) + eta32

/-- `Σ_n dS[n, m]·K[n, j]`. -/
noncomputable def dQS (d : Inst) (hv : HeadV) (m j : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N_CTX, dsE d hv n m * (hv.k n j).toReal

/-- The tolerance of the accumulated `Σ_n dS[n, m]·K[n, j]`. -/
noncomputable def dQAcc (d : Inst) (hv : HeadV) (m j : Nat) : ℝ :=
  ∑ n ∈ Finset.range d.N_CTX, ds16Err d hv n m * |(hv.k n j).toReal| +
    accB (stepsQ d m) (∑ n ∈ Finset.range d.N_CTX, ds16L d hv n m * lm (hv.k n j))

/-- The exact `dQ[m, j] = LN2 · Σ_n dS[n, m]·K[n, j]`. -/
noncomputable def dQE (d : Inst) (hv : HeadV) (m j : Nat) : ℝ := ln2F.toReal * dQS d hv m j

/-- The tolerance of `dQ[m, j]`'s `float32` value: the accumulation's, scaled, and the rounding
of the product by `LN2` (line 499). -/
noncomputable def dQTol (d : Inst) (hv : HeadV) (m j : Nat) : ℝ :=
  ln2F.toReal * dQAcc d hv m j + u32 * (ln2F.toReal * (|dQS d hv m j| + dQAcc d hv m j)) + eta32

/-- What the launch leaves in element `(b, r, j)` of the output `B`: some `float16` value `c`;
and when head `b`'s inputs are `hv` and covered (`HeadV.Ok`), `c` is the `float16` rounding (to
nearest even, up to NaN payload) of a finite `float32` value within `T hv` of the exact `E hv`;
and when there is room in `float16` (`|E| + T ≤ 65504`), `c` is finite and within `T` plus one
`float16` rounding (`(|E| + T)·2^-11 + 2^-25`) of `E`. -/
def OutSpec (d : Inst) (init fin : Mem) (B : Buf .f16) (b r j : Nat) (E T : HeadV → ℝ) : Prop :=
  ∃ c : F16, fin.read .f16 (B.ptr ↑(eIdx d b r j)).addr = some c ∧
    ∀ hv : HeadV, hv.Holds d init b → hv.Ok d →
      ∃ v : F32, v.isFinite = true ∧ |v.toReal - E hv| ≤ T hv ∧ FloatV.NaNEq (FloatV.cvt v) c ∧
        (|E hv| + T hv ≤ 65504 →
          c.isFinite = true ∧ |c.toReal - E hv| ≤ T hv + (|E hv| + T hv) * u16 + eta16)

/-- What the launch leaves: every element of `dv`, `dk` and `dq` as `OutSpec` says, and every
byte outside the three outputs unchanged. -/
def Post (d : Inst) (init fin : Mem) : Prop :=
  (∀ b r j, b < G d → r < d.N_CTX → j < d.HEAD_DIM →
    OutSpec d init fin d.DV b r j (fun hv => dVE d hv r j) (fun hv => dVTol d hv r j) ∧
    OutSpec d init fin d.DK b r j (fun hv => dKE d hv r j) (fun hv => dKTol d hv r j) ∧
    OutSpec d init fin d.DQ b r j (fun hv => dQE d hv r j) (fun hv => dQTol d hv r j)) ∧
  ∀ z, z ∉ outBytes d → fin z = init z

def Spec (K : Claim.Behaviour Inst) : Prop := Claim.Meets Inst.ok Pre Post K

/-! ## The assumptions -/

structure Assumptions (K : Claim.Behaviour Inst) (P : Platform) : Prop where
  /-- [rule] `Compiler.Faithful`: on every covered instance, when `launch d` has `debug = false`
  and is valid, every outcome of the compiled kernel launched as `launch d`, from a memory from
  which the model's launch is well behaved, is one of the outcomes of the model's launch. The rule
  covers a launch within its scope: part (A), the kernel and its launches, which the host's launches
  meet (README, Conditions), and part (B), the environment, which the claim relies on. -/
  faithful : ∀ d, d.ok → Rules.Compiler.Faithful P (model d) (launch d) (K d)
  /-- [rule] `Num.DotF16`: `tl.dot` of `float16` operands into `float32` (`maxImpreciseAcc = 0`,
  any `input_precision`; lines 293, 303, 307, 310, 345, 353, 358): each output element whose
  operands and accumulator are finite is finite and within `relDot16 K·L + absDot16 K` of
  `acc + Σ a·b`, where `L = |acc| + Σ liftMag a·liftMag b` counts a subnormal operand at `2^-14`
  (`Rules.Num.DotLiftAcc`, library round 25), with room in `float32`. -/
  dot : Rules.Num.DotF16 P
  /-- [rule] `Num.DotTotal`: the model's `tl.dot` always has a result. -/
  dot_total : Rules.Num.DotTotal P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 }
  /-- [rule] `Num.Exp2F32`: `tl.math.exp2` on `float32` (lines 294, 346) is within
  `2^-22·2^x + 2^-126` of `2^x`, with room in `float32`. -/
  exp2 : Rules.Num.Exp2F32 P
  /-- [rule] `Num.MathTotal`: `tl.math.exp2` on `float32` always has a result. -/
  exp2_total : Rules.Num.MathTotal P .exp2 .binary32

end Kernels.AttnBwd
