import Claim
import TVA.Proof

/-!
# The launch and the programs' footprints

The kernel runs on a 3-D grid `(gx, 1, gz)`; program `(pid, ·, pid_ch)` owns one `64 × 128`
tile of chunk `pid_c = pid_ch / G` and group `pid_h = pid_ch % G`. This file shows that the
launch is valid (`launch_valid`), gives the ids of program `i`, and defines its chunk, group and
tile (`cOf`, `hOf`, `mOf`, `nOf`), its read footprint `R` (the inputs), its write footprint `W`
(its tile of `out`) and its post-condition `Q`.
-/

namespace Claims.SsdBmm

open TVA

/-! ## The program's ids -/

/-- Program `c`'s body, with the instance's arguments substituted. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let pid_b := tl.program_id c 1
        let pid_ch : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id c 2))).item
        let pid_c : I64 := (← tl.idiv (Tensor.scalar pid_ch) (Tensor.scalar (← wide (IntV.ofNat d.G)))).item
        let pid_h : I64 := pid_ch - pid_c * (← wide (IntV.ofNat d.G))
        let num_pid_n : I32 := (← tl.idiv
          (Tensor.scalar ((IntV.ofNat d.C) + (IntV.ofNat (BN - 1) : I32)))
          (Tensor.scalar (IntV.ofNat BN : I32))).item
        let pid_m : I32 := (← tl.idiv (Tensor.scalar (tl.program_id c 0))
          (Tensor.scalar num_pid_n)).item
        let pid_n : I32 := (← tl.irem (Tensor.scalar (tl.program_id c 0))
          (Tensor.scalar num_pid_n)).item
        let a_ptr := d.a.ptr 0 + ((← batchOff pid_b (intArg d.aB)) +
          pid_c * (← wide (IntV.ofNat d.C)) * (← wide (IntV.ofNat d.aS)) + pid_h * (← wide (IntV.ofNat d.aH)))
        let b_ptr := d.b.ptr 0 + ((← batchOff pid_b (intArg d.bB)) +
          pid_c * (← wide (IntV.ofNat d.C)) * (← wide (IntV.ofNat d.bS)) + pid_h * (← wide (IntV.ofNat d.bH)))
        let seq_idx_ptr := d.seqIdx.ptr 0 + ((← wide (pid_b * (IntV.ofNat d.T))) +
          pid_c * (← wide (IntV.ofNat d.C)) * (← wide (IntV.ofNat 1)))
        let offs_m : Tensor [BM] I32 :=
          pid_m * (IntV.ofNat BM : I32) + tl.arange 0 BM
        let offs_n : Tensor [BN] I32 :=
          pid_n * (IntV.ofNat BN : I32) + tl.arange 0 BN
        let offs_k : Tensor [BK] I32 := tl.arange 0 BK
        let a_ptrs : Tensor [BM, BK] (Ptr .bf16) :=
          a_ptr + (offs_m.col BK * (IntV.ofNat d.aS : I32) + offs_k.row BM * (IntV.ofNat d.aK : I32))
        let b_ptrs : Tensor [BK, BN] (Ptr .bf16) :=
          b_ptr + (offs_k.col BN * (IntV.ofNat d.bK : I32) + offs_n.row BK * (IntV.ofNat d.bS : I32))
        let chunk_size_limit : I64 := (tl.iminimum (Tensor.scalar (← wide (IntV.ofNat d.C)))
          (Tensor.scalar ((← wide (IntV.ofNat d.T)) - pid_c * (← wide (IntV.ofNat d.C))))).item
        let acc := tl.zeros [BM, BN] .f32
        let (_a_ptrs, _b_ptrs, acc) ← tl.forRange (IntV.ofInt 0 : I32)
            (← tl.idiv (Tensor.scalar ((IntV.ofNat d.K) + (IntV.ofNat (BK - 1) : I32)))
              (Tensor.scalar (IntV.ofNat BK : I32))).item
            (IntV.ofInt 1 : I32) (a_ptrs, b_ptrs, acc)
            fun k (a_ptrs, b_ptrs, acc) => do
          let a ← tl.load a_ptrs
            (some (((← tl.convert .i32 .i64 (offs_m.col BK)).ltS chunk_size_limit).and
              ((offs_k.row BM).ltS ((IntV.ofNat d.K) - k * (IntV.ofNat BK : I32)))))
            (some (← tl.convert .f32 .bf16 (tl.full [BM, BK] zf32)))
          let b ← tl.load b_ptrs
            (some (((offs_k.col BN).ltS ((IntV.ofNat d.K) - k * (IntV.ofNat BK : I32))).and
              ((← tl.convert .i32 .i64 (offs_n.row BK)).ltS chunk_size_limit)))
            (some (← tl.convert .f32 .bf16 (tl.full [BK, BN] zf32)))
          let acc ← tl.dot_add cfgBF16 a b acc
          let a_ptrs := a_ptrs +
            (bc ((IntV.ofNat BK : I32) * (IntV.ofNat d.aK)) : Tensor [BM, BK] I32)
          let b_ptrs := b_ptrs +
            (bc ((IntV.ofNat BK : I32) * (IntV.ofNat d.bK)) : Tensor [BK, BN] I32)
          return (a_ptrs, b_ptrs, acc)
        let offs_m : Tensor [BM] I32 :=
          pid_m * (IntV.ofNat BM : I32) + tl.arange 0 BM
        let offs_n : Tensor [BN] I32 :=
          pid_n * (IntV.ofNat BN : I32) + tl.arange 0 BN
        let chunk_size_limit : I64 := (tl.iminimum (Tensor.scalar (← wide (IntV.ofNat d.C)))
          (Tensor.scalar ((← wide (IntV.ofNat d.T)) - pid_c * (← wide (IntV.ofNat d.C))))).item
        let seq_idx_m : Tensor [BM] I32 ←
          tl.load (seq_idx_ptr + offs_m * (IntV.ofNat 1 : I32))
            (some ((← tl.convert .i32 .i64 offs_m).ltS chunk_size_limit))
            (some (tl.full [BM] (IntV.ofInt (-1 : Int) : I32)))
        let seq_idx_n : Tensor [BN] I32 ←
          tl.load (seq_idx_ptr + offs_n * (IntV.ofNat 1 : I32))
            (some ((← tl.convert .i32 .i64 offs_n).ltS chunk_size_limit))
            (some (tl.full [BN] (IntV.ofInt (-2 : Int) : I32)))
        let acc := tl.where ((seq_idx_m.col BN).eq (seq_idx_n.row BM)) acc
          (tl.full [BM, BN] zf32)
        let out := acc
        let out_ptr := d.out.ptr 0 + ((← batchOff pid_b (intArg (oB d))) +
          pid_c * (← wide (IntV.ofNat (oC d))) + pid_h * (← wide (IntV.ofNat (oH d))))
        let out_ptrs : Tensor [BM, BN] (Ptr .f32) :=
          out_ptr + ((IntV.ofNat d.C : I32) * offs_m.col BN + offs_n.row BM * (IntV.ofNat 1 : I32))
        tl.store out_ptrs out
          (some ((offs_m.col BN).ltS (IntV.ofNat d.C) |>.and ((offs_n.row BM).ltS (IntV.ofNat d.C))))) := rfl

/-! ## The launch -/

/-- The number of programs is `gx · 1 · gz`. -/
theorem launch_size (d : Inst) : (launch d).size = gx d * 1 * gz d := rfl

/-- The number of programs is `gx · gz`. -/
theorem launch_size' (d : Inst) : (launch d).size = gx d * gz d := by
  rw [launch_size, Nat.mul_one]

/-- Axis 0 of program `i`'s id is `i % gx`. -/
theorem launch_pid0 (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i % gx d := by
  simp [Launch.ctx, launch]

/-- Axis 1 of program `i`'s id is `0` (the grid's `y` is `1`). -/
theorem launch_pid1 (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 1 = 0 := by
  simp [Launch.ctx, launch, Nat.mod_one]

/-- Axis 2 of program `i`'s id is `i / gx`. -/
theorem launch_pid2 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 2 = i / gx d := by
  simp [Launch.ctx, launch]

/-- There is at least one chunk. -/
theorem nch_pos {d : Inst} (hd : d.ok) : 0 < nch d := by
  have hC : 0 < d.C := by have := hd.C_ge; omega
  have h : d.C ≤ d.T + d.C - 1 := by have := hd.T_pos; omega
  exact Nat.div_pos h hC

/-- The grid's `z` is positive. -/
theorem gz_pos {d : Inst} (hd : d.ok) : 0 < gz d := by
  have h1 := nch_pos hd
  have h2 := hd.G_pos
  unfold gz
  exact Nat.mul_pos h1 h2

/-- The grid's `x` is below `2^31`. -/
theorem gx_lt {d : Inst} (hd : d.ok) : gx d < 2 ^ 31 := by
  have hle : gx d ≤ gx d * gz d := Nat.le_mul_of_pos_right _ (gz_pos hd)
  have h := hd.grid_lt
  omega

/-- Every block shape of the launch is accepted by Triton (powers of two, at most `2^20` elements). -/
theorem tritonOk_blocks :
    ∀ s ∈ ([[BM], [BN], [BK], [BM, BK], [BK, BN], [BM, BN]] : List Shape), Shape.TritonOk s := by
  have h1 : Shape.TritonOk [BM] := by
    refine ⟨?_, by simp [BM]⟩
    intro j hj
    simp only [List.mem_singleton] at hj
    subst hj; exact ⟨6, by simp [BM]⟩
  have h2 : Shape.TritonOk [BN] := by
    refine ⟨?_, by simp [BN]⟩
    intro j hj
    simp only [List.mem_singleton] at hj
    subst hj; exact ⟨7, by simp [BN]⟩
  have h3 : Shape.TritonOk [BK] := by
    refine ⟨?_, by simp [BK]⟩
    intro j hj
    simp only [List.mem_singleton] at hj
    subst hj; exact ⟨5, by simp [BK]⟩
  have h4 : Shape.TritonOk [BM, BK] := by
    refine ⟨?_, by simp [BM, BK]⟩
    intro j hj
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
    rcases hj with rfl | rfl
    · exact ⟨6, by simp [BM]⟩
    · exact ⟨5, by simp [BK]⟩
  have h5 : Shape.TritonOk [BK, BN] := by
    refine ⟨?_, by simp [BK, BN]⟩
    intro j hj
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
    rcases hj with rfl | rfl
    · exact ⟨5, by simp [BK]⟩
    · exact ⟨7, by simp [BN]⟩
  have h6 : Shape.TritonOk [BM, BN] := by
    refine ⟨?_, by simp [BM, BN]⟩
    intro j hj
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hj
    rcases hj with rfl | rfl
    · exact ⟨6, by simp [BM]⟩
    · exact ⟨7, by simp [BN]⟩
  intro s hs
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl
  exacts [h1, h2, h3, h4, h5, h6]

/-- **The launch is valid**: the grid, the program count and the block shapes are within Triton's and CUDA's limits. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hgz := gz_pos hd
  have hgx := gx_lt hd
  refine ⟨by simpa [launch] using hgx, by simp [launch], by simpa [launch] using hd.gz_lt, ?_,
    rfl, rfl, rfl, ?_⟩
  · rw [launch_size']
    exact hd.grid_lt
  · intro s hs
    exact tritonOk_blocks s (by simpa [launch] using hs)

/-! ## The programs' footprints -/

/-- `num_pid_n = cdiv(chunk_size, BLOCK_SIZE_N)`, the number of column blocks. -/
def numPidN (d : Inst) : Nat := (d.C + BN - 1) / BN

/-- `pid_c = pid_ch // ngroups`: the chunk. -/
def cOf (d : Inst) (i : Fin (launch d).size) : Nat := i / gx d / d.G

/-- `pid_h = pid_ch % ngroups`: the group. -/
def hOf (d : Inst) (i : Fin (launch d).size) : Nat := i / gx d % d.G

/-- `pid_m`: the row block of the program's tile. -/
def mOf (d : Inst) (i : Fin (launch d).size) : Nat := (i % gx d) / numPidN d

/-- `pid_n`: the column block of the program's tile. -/
def nOf (d : Inst) (i : Fin (launch d).size) : Nat := (i % gx d) % numPidN d

/-- The read footprint of every program: the whole of `a`, `b` and `seq_idx`. -/
def R (d : Inst) (_init : Mem) (_i : Fin (launch d).size) : Set Int :=
  (d.a.bytes ∪ d.b.bytes) ∪ d.seqIdx.bytes

/-- The elements of `out` that program `(c, g, pm, pn)` stores: its enabled lanes. -/
def tileElems (d : Inst) (c g pm pn : Nat) : Set Nat :=
  {j | ∃ m n, m < BM ∧ n < BN ∧ pm * BM + m < d.C ∧ pn * BN + n < d.C ∧
        j = oIdx d c g (pm * BM + m) (pn * BN + n)}

/-- The write footprint of a program: its tile of `out`. -/
def W (d : Inst) (_init : Mem) (i : Fin (launch d).size) : Set Int :=
  d.out.elems (tileElems d (cOf d i) (hOf d i) (mOf d i) (nOf d i))

/-- What a program leaves in its own memory: `OutElem` for each of its enabled lanes. -/
def Q (d : Inst) (init : Mem) (i : Fin (launch d).size) (mem : Mem) : Prop :=
  ∀ m n, m < BM → n < BN → mOf d i * BM + m < d.C → nOf d i * BN + n < d.C →
    OutElem d init mem (cOf d i) (hOf d i) (mOf d i * BM + m) (nOf d i * BN + n)


end Claims.SsdBmm



