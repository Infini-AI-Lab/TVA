import Proof.SetupDefs

/-!
# The set-up of the variant with initial states, as programs

`setupI` is `chunk_scan_kernel` (lines 172–260, up to the first runtime `if`), `tail1I` lines
261–310 (`offs_n`, `dA_cs_m`, `c_idx_n`, the second runtime `if`) and `tail2I` the scale (lines
337–344) and the `core`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- The scale and the core (lines 337–488), given the merged limit and decay boundary. -/
def tail2I (d : Inst) (c_off pid_h pid_m pid_n : I32) (prev_ptr : Ptr .f32) (hdimS dstS : I32)
    (cb_ptr : Ptr .f32) (x_ptr : Ptr .bf16) (dt_ptr dA_ptr : Ptr .f32) (C_ptr out_ptr : Ptr .bf16)
    (offs_m : Tensor [BM] I32) (offs_n : Tensor [BN] I32) (dA_cs_m : Tensor [BM] F32)
    (limit : I32) (boundary : F32) : Prog Unit := do
  let scale_m : Tensor [BM] F32 ←
    tl.exp (← tl.sub dA_cs_m (bc boundary))
  core d C_ptr prev_ptr hdimS dstS cb_ptr x_ptr dt_ptr dA_ptr out_ptr
    (d.D.ptr 0) pid_h pid_m pid_n c_off offs_m offs_n (fun x => x) (IntV.ofNat d.cs) limit
    ((pid_m + IntV.ofInt 1) * (IntV.ofNat BM : I32)) dA_cs_m scale_m

/-- Lines 261–310: `offs_n`, `dA_cs_m`, `c_idx_n` and the second runtime `if` (the limit and the
decay boundary), then `tail2I`. -/
def tail1I (d : Inst) (pid_c : I64) (c_idx c_off pid_h pid_m pid_n : I32) (limit : I32)
    (prev_ptr : Ptr .f32) (hdimS dstS : I32) (cb_ptr : Ptr .f32) (x_ptr : Ptr .bf16) (dt_ptr dA_ptr : Ptr .f32)
    (C_ptr out_ptr : Ptr .bf16) (offs_m : Tensor [BM] I32) : Prog Unit := do
  let offs_n : Tensor [BN] I32 := pid_n * (IntV.ofNat BN : I32) + tl.arange 0 BN
  let dA_cs_m : Tensor [BM] F32 ← tl.load (dA_ptr + offs_m * (IntV.ofNat 1 : I32))
    (some (offs_m.ltS (IntV.ofNat d.cs : I32))) (some (bc zf32))
  let c_idx_n : I32 := (← tl.load
    (Tensor.scalar (d.chunkIdx.ptr 0 + (pid_c + (IntV.ofInt 1 : I64))))
    (some (Tensor.and (Tensor.scalar (IntV.lt (IntV.ofInt (-1) : I64) pid_c))
      (Tensor.scalar (IntV.lt (pid_c + (IntV.ofInt 1 : I64))
        (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)))))
    (some (Tensor.scalar (IntV.ofInt (-1) : I32)))).item
  let (limit', boundary) ←
    if IntV.eq c_idx c_idx_n || IntV.lt (IntV.ofInt 0) c_off then do
      let c_off_n : I32 := (← tl.load
        (Tensor.scalar (d.chunkOffs.ptr 0 + (pid_c + (IntV.ofInt 1 : I64))))
        (some (Tensor.and (Tensor.scalar (IntV.lt (IntV.ofInt (-1) : I64) pid_c))
          (Tensor.scalar (IntV.lt (pid_c + (IntV.ofInt 1 : I64))
            (IntV.cast (IntV.ofNat d.nmeta : I32) : I64)))))
        (some (Tensor.scalar (IntV.ofNat d.cs : I32)))).item
      let limit' : I32 :=
        if IntV.eq c_idx c_idx_n then
          (tl.iminimum (Tensor.scalar c_off_n) (Tensor.scalar limit)).item
        else limit
      let boundary : F32 := (← tl.load
        (Tensor.scalar (dA_ptr + (c_off - IntV.ofInt 1) * (IntV.ofNat 1 : I32)))
        (some (Tensor.and (Tensor.scalar (IntV.lt (IntV.ofInt (-1)) (c_off - IntV.ofInt 1)))
          (Tensor.scalar (IntV.lt c_off (IntV.ofNat d.cs : I32)))))
        (some (Tensor.scalar zf32))).item
      pure (limit', boundary)
    else
      pure (limit, zf32)
  tail2I d c_off pid_h pid_m pid_n prev_ptr hdimS dstS cb_ptr x_ptr dt_ptr dA_ptr C_ptr out_ptr offs_m offs_n
    dA_cs_m limit' boundary

/-- Lines 172–260: coordinates, the metadata loads, the pointers, `chunk_size_limit`,
`seq_idx_prev` and the first runtime `if` (the state's pointer); then `tail1I`. -/
def setupI (d : Inst) (ctx : Ctx) : Prog Unit := do
  let pid_bc : I64 := IntV.cast (tl.program_id ctx 1)
  let pid_c : I64 :=
    (← tl.idiv (Tensor.scalar pid_bc) (Tensor.scalar (IntV.ofNat 1 : I64))).item
  let pid_b : I64 := pid_bc - pid_c * (IntV.ofNat 1 : I64)
  let c_idx : I32 := (← tl.load (Tensor.scalar (d.chunkIdx.ptr 0 + pid_c))
    (some (Tensor.scalar (IntV.lt (IntV.ofInt (-1) : I64) pid_c)))
    (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
  let c_off : I32 := (← tl.load (Tensor.scalar (d.chunkOffs.ptr 0 + pid_c))
    (some (Tensor.scalar (IntV.lt (IntV.ofInt (-1) : I64) pid_c)))
    (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
  let pid_h := tl.program_id ctx 2
  let num_pid_n : I32 := (← tl.idiv
    (Tensor.scalar (IntV.ofNat d.P + (IntV.ofNat (BN - 1) : I32)))
    (Tensor.scalar (IntV.ofNat BN : I32))).item
  let pid_m : I32 :=
    (← tl.idiv (Tensor.scalar (tl.program_id ctx 0)) (Tensor.scalar num_pid_n)).item
  let pid_n : I32 :=
    (← tl.irem (Tensor.scalar (tl.program_id ctx 0)) (Tensor.scalar num_pid_n)).item
  let g_cb : I32 :=
    (← tl.idiv (Tensor.scalar pid_h) (Tensor.scalar (IntV.ofNat (ratio d) : I32))).item
  let cb_ptr := d.cb.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (nc d * (d.G * (d.cs * d.cs))) : I32) : I64) +
    (IntV.cast (c_idx * (IntV.ofNat (d.G * (d.cs * d.cs)) : I32)) : I64) +
    (IntV.cast (g_cb * (IntV.ofNat (d.cs * d.cs) : I32)) : I64))
  let x_ptr := d.x.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.T * d.xT) : I32) : I64) +
    (IntV.cast (c_idx * (IntV.ofNat d.cs : I32) * (IntV.ofNat d.xT : I32)) : I64) +
    (IntV.cast (pid_h * (IntV.ofNat d.P : I32)) : I64))
  let dt_ptr := d.dt.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.H * (nc d * d.cs)) : I32) : I64) +
    (IntV.cast (c_idx * (IntV.ofNat d.cs : I32)) : I64) +
    (IntV.cast (pid_h * (IntV.ofNat (nc d * d.cs) : I32)) : I64))
  let dA_cumsum_ptr := d.dA.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.H * (nc d * d.cs)) : I32) : I64) +
    (IntV.cast (c_idx * (IntV.ofNat d.cs : I32)) : I64) +
    (IntV.cast (pid_h * (IntV.ofNat (nc d * d.cs) : I32)) : I64))
  let g_C : I32 :=
    (← tl.idiv (Tensor.scalar pid_h) (Tensor.scalar (IntV.ofNat (ratio d) : I32))).item
  let C_ptr := d.C.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.T * d.cT) : I32) : I64) +
    (IntV.cast (c_idx * (IntV.ofNat d.cs : I32) * (IntV.ofNat d.cT : I32)) : I64) +
    (IntV.cast (g_C * (IntV.ofNat d.N : I32)) : I64))
  let offs_m : Tensor [BM] I32 :=
    (pid_m * (IntV.ofNat BM : I32) + c_off) + tl.arange 0 BM
  let prev_states_ptr := d.states.ptr 0 + pid_b * (IntV.cast (IntV.ofNat (nc d * (d.H * PN d)) : I32) : I64) +
    c_idx * (IntV.ofNat (d.H * PN d) : I32) + pid_h * (IntV.ofNat (PN d) : I32)
  let prev_states_hdim : I32 := IntV.ofNat d.N
  let prev_states_dstate : I32 := IntV.ofNat 1
  let chunk_size_limit : I32 := (tl.iminimum (Tensor.scalar (IntV.ofNat d.cs : I32))
    (Tensor.scalar ((IntV.ofNat d.T : I32) - c_idx * (IntV.ofNat d.cs : I32)))).item
  let seq_idx_ptr := d.seqIdx.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat d.T : I32) : I64) +
    (IntV.cast (c_idx * (IntV.ofNat d.cs : I32) * (IntV.ofNat 1 : I32)) : I64))
  let seq_idx_prev : I32 := (← tl.load
    (Tensor.scalar (seq_idx_ptr + (- (IntV.ofNat 1 : I32))))
    (some (Tensor.scalar (IntV.le (IntV.ofInt 1 : I32) c_idx)))
    (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
  let out_ptr := d.out.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.T * (d.H * d.P)) : I32) : I64) +
    (IntV.cast (c_idx * (IntV.ofNat d.cs : I32) * (IntV.ofNat (d.H * d.P) : I32)) : I64) +
    (IntV.cast (pid_h * (IntV.ofNat d.P : I32)) : I64))
  let (prev_ptr, hdimS, dstS) ←
    if IntV.lt (pid_m * (IntV.ofNat BM : I32) + c_off) chunk_size_limit then do
      let seq_idx_m : I32 ← tl.load1
        (seq_idx_ptr + (pid_m * (IntV.ofNat BM : I32) + c_off) * (IntV.ofNat 1 : I32))
      if (IntV.eq c_off (IntV.ofInt 0) && !(IntV.eq seq_idx_prev seq_idx_m)) ||
          IntV.lt (IntV.ofInt 0) c_off then
        pure (d.initst.ptr 0 + seq_idx_m * (IntV.ofNat (d.H * PN d) : I32) +
            pid_h * (IntV.ofNat (PN d) : I32), (IntV.ofNat d.N : I32), (IntV.ofNat 1 : I32))
      else pure (prev_states_ptr, prev_states_hdim, prev_states_dstate)
    else pure (prev_states_ptr, prev_states_hdim, prev_states_dstate)
  tail1I d pid_c c_idx c_off pid_h pid_m pid_n chunk_size_limit prev_ptr hdimS dstS cb_ptr x_ptr dt_ptr
    dA_cumsum_ptr C_ptr out_ptr offs_m

/-- The model's body with initial states is `setupI`. -/
theorem body_eq_init (d : Inst) (h : d.hasInit = true) (ctx : Ctx) :
    (model d).body ctx = setupI d ctx := by
  simp only [model, h]
  rfl

end Claims.SsdChunkScan
