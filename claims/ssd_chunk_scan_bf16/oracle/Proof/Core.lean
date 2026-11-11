import Proof.Defs

/-!
# The kernel's core, in three phases

Both variants of the kernel (`chunk_scan_kernel`, `chunk_scan_kernel_noinit`) end in the same
computation (lines 319–488) once the pointers, the scale of the entering state and the limits are
known. This file states it once, generic in the integer width `w` of the limits (`32` with initial
states, `64` without: the `widen` function turns an `int32` tensor into the width in which the
variant compares it), and cut into three continuation-passing phases:

* `phaseA`: the state term `acc₀ = tl.dot(C, bf16(S)) * scale` (lines 324–361);
* `phaseB`: the loop of lines 383–424 over the chunk's tokens, `acc += tl.dot(cb, x)`;
* `phaseC`: `D·x`, the conversion and the store (lines 426–488).

`core` runs them in order. Each variant's walk ends by `core`, whose definition unfolds to the
variant's own text.
-/

namespace Claims.SsdChunkScan

open TVA

/-- Phase A (lines 324–361): the `[BM, BSD]` tile of `C` and the `[BSD, BN]` tile of the state
(converted to `bfloat16`), their `tl.dot`, and the product with `scale_m[:, None]`. -/
def phaseA {w : Nat} {α : Type} (d : Inst) (C_ptr : Ptr .bf16) (prev_ptr : Ptr .f32)
    (hdimS dstS : I32) (offs_m : Tensor [BM] I32) (offs_n : Tensor [BN] I32)
    (offs_mw : Tensor [BM] (IntV true w)) (limit : IntV true w) (scale_m : Tensor [BM] F32)
    (k : Tensor [BM, BN] F32 → Prog α) : Prog α := do
  let offs_k_dstate : Tensor [d.BSD] I32 := tl.arange 0 d.BSD
  let C_ptrs : Tensor [BM, d.BSD] (Ptr .bf16) :=
    C_ptr + ((offs_m.col d.BSD) * (IntV.ofNat d.cT : I32) +
      (offs_k_dstate.row BM) * (IntV.ofNat 1 : I32))
  let prev_states_ptrs : Tensor [d.BSD, BN] (Ptr .f32) :=
    prev_ptr + ((offs_n.row d.BSD) * hdimS + (offs_k_dstate.col BN) * dstS)
  let C : Tensor [BM, d.BSD] BF16 ← tl.load C_ptrs
    (some (((offs_mw.col d.BSD).ltS limit).and
      ((offs_k_dstate.row BM).ltS (IntV.ofNat d.N : I32))))
    (some (← tl.convert .f32 .bf16 (tl.full [BM, d.BSD] zf32)))
  let prev_states : Tensor [d.BSD, BN] F32 ← tl.load prev_states_ptrs
    (some (((offs_k_dstate.col BN).ltS (IntV.ofNat d.N : I32)).and
      ((offs_n.row d.BSD).ltS (IntV.ofNat d.P : I32))))
    (some (bc zf32))
  let prev_states₁ : Tensor [d.BSD, BN] BF16 ← tl.cvt prev_states
  let acc₀ : Tensor [BM, BN] F32 ← tl.mul
    (← tl.dot cfgBF16 C prev_states₁ (tl.zeros [BM, BN] .f32))
    (scale_m.col BN)
  k acc₀

/-- The loop's carried state (line 397): the accumulator and the pointer tiles of `cb`, `x`, `dt`
and `dA_cumsum`. -/
abbrev LoopSt : Type :=
  Tensor [BM, BN] F32 × Tensor [BM, BK] (Ptr .f32) × Tensor [BK, BN] (Ptr .bf16) ×
    Tensor [BK] (Ptr .f32) × Tensor [BK] (Ptr .f32)

/-- One trip of the loop of line 397 (lines 398–424), with loop variable `kk`. -/
def bBody {w : Nat} (d : Inst) (offs_m : Tensor [BM] I32) (offs_n : Tensor [BN] I32)
    (offs_k : Tensor [BK] I32) (widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w))
    (csw limit : IntV true w) (dA_cs_m : Tensor [BM] F32) (kk : IntV true w) (st : LoopSt) :
    Prog LoopSt := do
  let (acc, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs) := st
  let cb : Tensor [BM, BK] F32 ← tl.load cb_ptrs
    (some (((offs_m.col BK).ltS (IntV.ofNat d.cs : I32)).and
      (((widen offs_k).row BM).ltS (csw - kk))))
    (some (bc zf32))
  let dA_cs_k : Tensor [BK] F32 ← tl.load dA_cumsum_ptrs
    (some ((widen offs_k).ltS (csw - kk))) (some (bc zf32))
  let cb₁ : Tensor [BM, BK] F32 ← tl.mul cb
    (← tl.exp (← tl.sub (dA_cs_m.col BK) (dA_cs_k.row BM)))
  let dt_k : Tensor [BK] F32 ← tl.load dt_ptrs
    (some ((widen offs_k).ltS (csw - kk))) (some (bc zf32))
  let cb₂ : Tensor [BM, BK] F32 ← tl.mul cb₁ (dt_k.row BM)
  let mask : Tensor [BM, BK] Bool :=
    Tensor.le ((kk + widen offs_k).row BM) ((widen offs_m).col BK)
  let cb₃ : Tensor [BM, BK] F32 := tl.where mask cb₂ (bc zf32)
  let cb₄ : Tensor [BM, BK] BF16 ← tl.cvt cb₃
  let x : Tensor [BK, BN] BF16 ← tl.load x_ptrs
    (some ((((widen offs_k).col BN).ltS (limit - kk)).and
      ((offs_n.row BK).ltS (IntV.ofNat d.P : I32))))
    (some (← tl.convert .f32 .bf16 (tl.full [BK, BN] zf32)))
  let acc : Tensor [BM, BN] F32 ← tl.dot_add cfgBF16 cb₄ x acc
  let cb_ptrs := cb_ptrs.map (fun (q : Ptr .f32) =>
    q + (IntV.ofNat BK : I32) * (IntV.ofNat 1 : I32))
  let x_ptrs := x_ptrs.map (fun (q : Ptr .bf16) =>
    q + (IntV.ofNat BK : I32) * (IntV.ofNat d.xT : I32))
  let dt_ptrs := dt_ptrs.map (fun (q : Ptr .f32) =>
    q + (IntV.ofNat BK : I32) * (IntV.ofNat 1 : I32))
  let dA_cumsum_ptrs := dA_cumsum_ptrs.map (fun (q : Ptr .f32) =>
    q + (IntV.ofNat BK : I32) * (IntV.ofNat 1 : I32))
  pure (acc, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs)

/-- Phase B (lines 383–424): the loop over the chunk's tokens `acc += tl.dot(cb, x)`, with the
weights `cb · exp(dA_m − dA_k) · dt_k` masked to `k ≤ m` and rounded to `bfloat16`. -/
def phaseB {w : Nat} {α : Type} (d : Inst) (cb_ptr : Ptr .f32) (x_ptr : Ptr .bf16)
    (dt_ptr dA_ptr : Ptr .f32) (offs_m : Tensor [BM] I32) (offs_n : Tensor [BN] I32) (c_off : I32)
    (widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w))
    (csw limit kmaxarg : IntV true w) (dA_cs_m : Tensor [BM] F32) (acc₀ : Tensor [BM, BN] F32)
    (k : Tensor [BM, BN] F32 → Prog α) : Prog α := do
  let offs_k : Tensor [BK] I32 := tl.arange 0 BK + c_off
  let cb_ptrs : Tensor [BM, BK] (Ptr .f32) :=
    cb_ptr + ((offs_m.col BK) * (IntV.ofNat d.cs : I32) +
      (offs_k.row BM) * (IntV.ofNat 1 : I32))
  let x_ptrs : Tensor [BK, BN] (Ptr .bf16) :=
    x_ptr + ((offs_k.col BN) * (IntV.ofNat d.xT : I32) +
      (offs_n.row BK) * (IntV.ofNat 1 : I32))
  let dt_ptrs : Tensor [BK] (Ptr .f32) := dt_ptr + offs_k * (IntV.ofNat 1 : I32)
  let dA_cumsum_ptrs : Tensor [BK] (Ptr .f32) := dA_ptr + offs_k * (IntV.ofNat 1 : I32)
  let K_MAX : IntV true w :=
    (tl.iminimum (Tensor.scalar kmaxarg) (Tensor.scalar limit)).item
  let (acc, _cb_ptrs, _x_ptrs, _dt_ptrs, _dA_cumsum_ptrs) ←
    tl.forRange (IntV.ofInt 0 : IntV true w) K_MAX (IntV.ofNat BK : IntV true w)
      (acc₀, cb_ptrs, x_ptrs, dt_ptrs, dA_cumsum_ptrs)
      (bBody d offs_m offs_n offs_k widen csw limit dA_cs_m)
  k acc

/-- Phase C (lines 426–488): `acc + x·D`, converted to `bfloat16` and stored. -/
def phaseC {w : Nat} (d : Inst) (x_ptr : Ptr .bf16) (out_ptr : Ptr .bf16) (D_ptr : Ptr .bf16)
    (pid_h pid_m pid_n c_off : I32) (offs_m : Tensor [BM] I32) (offs_n : Tensor [BN] I32)
    (widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w)) (limit : IntV true w)
    (acc : Tensor [BM, BN] F32) : Prog Unit := do
  let offs_out_m : Tensor [BM] I32 :=
    (pid_m * (IntV.ofNat BM : I32) + c_off) + tl.arange 0 BM
  let offs_out_n : Tensor [BN] I32 := pid_n * (IntV.ofNat BN : I32) + tl.arange 0 BN
  let D₀ : BF16 ← tl.load1 (D_ptr + pid_h * (IntV.ofNat 1 : I32))
  let D : F32 := (← tl.cvt (Tensor.scalar D₀)).item
  let x_residual₀ : Tensor [BM, BN] BF16 ← tl.load
    (x_ptr + ((offs_m.col BN) * (IntV.ofNat d.xT : I32) +
      (offs_n.row BM) * (IntV.ofNat 1 : I32)))
    (some ((((widen offs_m).col BN).ltS limit).and
      ((offs_n.row BM).ltS (IntV.ofNat d.P : I32))))
    (some (← tl.convert .f32 .bf16 (tl.full [BM, BN] zf32)))
  let x_residual : Tensor [BM, BN] F32 ← tl.cvt x_residual₀
  let acc ← tl.mul_add x_residual (bc D) acc
  let out_ptrs : Tensor [BM, BN] (Ptr .bf16) :=
    out_ptr + ((IntV.ofNat (d.H * d.P) : I32) * (offs_out_m.col BN) +
      (offs_out_n.row BM) * (IntV.ofNat 1 : I32))
  let out16 : Tensor [BM, BN] BF16 ← tl.cvt acc
  tl.store out_ptrs out16
    (some ((((widen offs_out_m).col BN).ltS limit).and
      ((offs_out_n.row BM).ltS (IntV.ofNat d.P : I32))))

/-- The whole core: phase A, then B, then C. -/
def core {w : Nat} (d : Inst) (C_ptr : Ptr .bf16) (prev_ptr : Ptr .f32) (hdimS dstS : I32)
    (cb_ptr : Ptr .f32) (x_ptr : Ptr .bf16) (dt_ptr dA_ptr : Ptr .f32) (out_ptr : Ptr .bf16)
    (D_ptr : Ptr .bf16) (pid_h pid_m pid_n c_off : I32)
    (offs_m : Tensor [BM] I32) (offs_n : Tensor [BN] I32)
    (widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w))
    (csw limit kmaxarg : IntV true w) (dA_cs_m scale_m : Tensor [BM] F32) : Prog Unit :=
  phaseA d C_ptr prev_ptr hdimS dstS offs_m offs_n (widen offs_m) limit scale_m fun acc₀ =>
    phaseB d cb_ptr x_ptr dt_ptr dA_ptr offs_m offs_n c_off widen csw limit kmaxarg dA_cs_m acc₀
      fun acc => phaseC d x_ptr out_ptr D_ptr pid_h pid_m pid_n c_off offs_m offs_n widen limit acc

end Claims.SsdChunkScan
