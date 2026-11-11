import Proof.Decode

/-!
# The two set-ups as programs

`setupN d ctx` and `setupI d ctx` are the text of `chunk_scan_kernel_noinit` and `chunk_scan_kernel`
up to the point where the core takes over (lines 172–344); the model's body is exactly this
followed by `core` (`body_eq_noinit`, `body_eq_init`).
-/

namespace Claims.SsdChunkScan

open TVA

/-- The set-up of `chunk_scan_kernel_noinit`: coordinates, pointers, `chunk_size_limit`,
`seq_idx_prev`, `dA_cs_m`, `seq_idx_m` and the scale; then `core`. -/
def setupN (d : Inst) (ctx : Ctx) : Prog Unit := do
    let pid_bc : I64 := IntV.cast (tl.program_id ctx 1)
    let pid_c : I64 :=
      (← tl.idiv (Tensor.scalar pid_bc) (Tensor.scalar (IntV.ofNat 1 : I64))).item
    let pid_b : I64 := pid_bc - pid_c * (IntV.ofNat 1 : I64)
    let c_idx : I64 := pid_c
    let c_off : I32 := IntV.ofInt 0
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
      c_idx * (IntV.cast (IntV.ofNat (d.G * (d.cs * d.cs)) : I32) : I64) +
      (IntV.cast (g_cb * (IntV.ofNat (d.cs * d.cs) : I32)) : I64))
    let x_ptr := d.x.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.T * d.xT) : I32) : I64) +
      c_idx * (IntV.cast (IntV.ofNat d.cs : I32) : I64) * (IntV.cast (IntV.ofNat d.xT : I32) : I64) +
      (IntV.cast (pid_h * (IntV.ofNat d.P : I32)) : I64))
    let dt_ptr := d.dt.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.H * (nc d * d.cs)) : I32) : I64) +
      c_idx * (IntV.cast (IntV.ofNat d.cs : I32) : I64) +
      (IntV.cast (pid_h * (IntV.ofNat (nc d * d.cs) : I32)) : I64))
    let dA_cumsum_ptr := d.dA.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.H * (nc d * d.cs)) : I32) : I64) +
      c_idx * (IntV.cast (IntV.ofNat d.cs : I32) : I64) +
      (IntV.cast (pid_h * (IntV.ofNat (nc d * d.cs) : I32)) : I64))
    let g_C : I32 :=
      (← tl.idiv (Tensor.scalar pid_h) (Tensor.scalar (IntV.ofNat (ratio d) : I32))).item
    let C_ptr := d.C.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.T * d.cT) : I32) : I64) +
      c_idx * (IntV.cast (IntV.ofNat d.cs : I32) : I64) * (IntV.cast (IntV.ofNat d.cT : I32) : I64) +
      (IntV.cast (g_C * (IntV.ofNat d.N : I32)) : I64))
    let offs_m : Tensor [BM] I32 :=
      (pid_m * (IntV.ofNat BM : I32) + c_off) + tl.arange 0 BM
    let offs_m64 : Tensor [BM] I64 := offs_m.castInt
    let prev_states_ptr := d.states.ptr 0 +
      pid_b * (IntV.cast (IntV.ofNat (nc d * (d.H * PN d)) : I32) : I64) +
      c_idx * (IntV.cast (IntV.ofNat (d.H * PN d) : I32) : I64) + pid_h * (IntV.ofNat (PN d) : I32)
    let prev_states_hdim : I32 := IntV.ofNat d.N
    let prev_states_dstate : I32 := IntV.ofNat 1
    let chunk_size_limit : I64 := (tl.iminimum (Tensor.scalar (IntV.cast (IntV.ofNat d.cs : I32) : I64))
      (Tensor.scalar ((IntV.cast (IntV.ofNat d.T : I32) : I64) -
        c_idx * (IntV.cast (IntV.ofNat d.cs : I32) : I64)))).item
    let seq_idx_ptr := d.seqIdx.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat d.T : I32) : I64) +
      c_idx * (IntV.cast (IntV.ofNat d.cs : I32) : I64) * (IntV.cast (IntV.ofNat 1 : I32) : I64))
    let seq_idx_prev : I32 := (← tl.load
      (Tensor.scalar (seq_idx_ptr + (- (IntV.ofNat 1 : I32))))
      (some (Tensor.scalar (IntV.le (IntV.ofInt 1 : I64) c_idx)))
      (some (Tensor.scalar (IntV.ofInt 0 : I32)))).item
    let offs_n : Tensor [BN] I32 :=
      pid_n * (IntV.ofNat BN : I32) + tl.arange 0 BN
    let dA_cs_m : Tensor [BM] F32 ← tl.load (dA_cumsum_ptr + offs_m * (IntV.ofNat 1 : I32))
      (some (offs_m.ltS (IntV.ofNat d.cs : I32))) (some (bc zf32))
    let seq_idx_m : Tensor [BM] I32 ← tl.load
      (seq_idx_ptr + offs_m * (IntV.ofNat 1 : I32))
      (some (offs_m64.ltS chunk_size_limit)) (some (bc (IntV.ofInt (-1) : I32)))
    let scale_m : Tensor [BM] F32 :=
      tl.where (Tensor.eq seq_idx_m (bc seq_idx_prev)) (← tl.exp dA_cs_m) (bc zf32)
    let out_ptr := d.out.ptr 0 + (pid_b * (IntV.cast (IntV.ofNat (d.T * (d.H * d.P)) : I32) : I64) +
      c_idx * (IntV.cast (IntV.ofNat d.cs : I32) : I64) *
        (IntV.cast (IntV.ofNat (d.H * d.P) : I32) : I64) +
      (IntV.cast (pid_h * (IntV.ofNat d.P : I32)) : I64))
    core d C_ptr prev_states_ptr prev_states_hdim prev_states_dstate cb_ptr x_ptr dt_ptr
      dA_cumsum_ptr out_ptr (d.D.ptr 0) pid_h pid_m pid_n c_off offs_m offs_n (fun x => x.castInt)
      (IntV.cast (IntV.ofNat d.cs : I32) : I64) chunk_size_limit
      (IntV.cast ((pid_m + IntV.ofInt 1) * (IntV.ofNat BM : I32)) : I64) dA_cs_m scale_m

/-- The model's body without initial states is `setupN`. -/
theorem body_eq_noinit (d : Inst) (h : d.hasInit = false) (ctx : Ctx) :
    (model d).body ctx = setupN d ctx := by
  simp only [model, h]
  rfl

end Claims.SsdChunkScan
