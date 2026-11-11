import Proof.Qi

/-!
# The grid, the footprints and the post-condition of one program

Program `i` of the grid `(batch, num_heads, cdiv(max_len_extend, BLOCK_M))` is the triple
`(b, h, m) = (i % batch, i / batch % H, i / (batch * H))`: request `b`, query head `h`, block `m`
of `BLOCK_M` new tokens. It reads the inputs and writes the tile of `o_extend` of its block, cut at
the request's number of new tokens (`tile`); `Q` says each of its output elements is within the
claim's tolerance of the exact attention.
-/

namespace Claims.ExtendAttn

open TVA

/-! ## The grid -/

/-- The request of program `i`. -/
def bOf (d : Inst) (i : Nat) : Nat := i % d.batch
/-- The query head of program `i`. -/
def hOf (d : Inst) (i : Nat) : Nat := i / d.batch % d.H
/-- The query block of program `i`. -/
def mOf (d : Inst) (i : Nat) : Nat := i / (d.batch * d.H)

/-- The launch has `batch * H * nBlocks` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.batch * d.H * nBlocks d := rfl

/-- Program axis 0 of program `i` is its batch index. -/
theorem launch_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = bOf d i := rfl
/-- Program axis 1 of program `i` is its head index. -/
theorem launch_pid1 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 1 = hOf d i := rfl
/-- Program axis 2 of program `i` is its query-block index. -/
theorem launch_pid2 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 2 = mOf d i := rfl

private theorem tritonOk1 {x : Nat} (hx : ∃ e, x = 2 ^ e) (h : x ≤ 2 ^ 20) :
    Shape.TritonOk [x] := by
  refine ⟨?_, by simpa using h⟩
  intro k hk
  simp only [List.mem_singleton] at hk
  subst hk; exact hx

private theorem tritonOk2 {x y : Nat} (hx : ∃ e, x = 2 ^ e) (hy : ∃ e, y = 2 ^ e)
    (h : x * y ≤ 2 ^ 20) : Shape.TritonOk [x, y] := by
  refine ⟨?_, by simpa using h⟩
  intro k hk
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
  rcases hk with rfl | rfl
  exacts [hx, hy]

/-- `BLOCK_M` is `128` or `64`, a power of two. -/
theorem bm_pow2 {d : Inst} (hd : d.ok) : ∃ e, d.BM = 2 ^ e := by
  rw [hd.bm]; split_ifs
  exacts [⟨7, by norm_num⟩, ⟨6, by norm_num⟩]

/-- The shapes of the launch are valid Triton block shapes. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  obtain ⟨a, ha⟩ := hd.bd_pow2
  have hbd := hd.bd_le
  have hbm := hd.bm
  obtain ⟨c, hc⟩ := bm_pow2 hd
  have hBM : d.BM ≤ 128 := by rw [hbm]; split_ifs <;> omega
  have hBDp : ∃ e, d.BD = 2 ^ e := ⟨a, ha⟩
  have h64 : ∃ e, BN = 2 ^ e := ⟨6, by norm_num [BN]⟩
  have h1 : ∃ e, (1 : Nat) = 2 ^ e := ⟨0, by norm_num⟩
  have hBN : BN = 64 := rfl
  have hBD1 : 1 ≤ d.BD := by rw [ha]; exact Nat.one_le_two_pow
  refine ⟨by simpa [launch] using hd.batch_lt, by simpa [launch] using hd.H_lt,
    by simpa [launch] using hd.nb_lt, hd.grid_lt, rfl, rfl, rfl, ?_⟩
  intro s hs
  simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact tritonOk1 ⟨c, hc⟩ (by omega)
  · exact tritonOk1 hBDp (by omega)
  · exact tritonOk1 h64 (by omega)
  · exact tritonOk2 ⟨c, hc⟩ hBDp (by nlinarith)
  · exact tritonOk2 hBDp h64 (by rw [hBN]; omega)
  · exact tritonOk2 ⟨c, hc⟩ h64 (by rw [hBN]; omega)
  · exact tritonOk2 h64 hBDp (by rw [hBN]; omega)
  · exact tritonOk2 ⟨c, hc⟩ h1 (by omega)
  · exact tritonOk2 h1 hBDp (by omega)
  · exact tritonOk2 h1 h64 (by rw [hBN]; omega)
  · exact tritonOk2 hBDp h1 (by omega)
  · exact tritonOk2 h64 h1 (by rw [hBN]; omega)

/-- A program of the grid has a request, a head and a block in range. -/
theorem prog_range {d : Inst} {i : Nat} (hi : i < (launch d).size) :
    bOf d i < d.batch ∧ hOf d i < d.H ∧ mOf d i < nBlocks d := by
  have hsize : i < d.batch * d.H * nBlocks d := hi
  have hbatch : 0 < d.batch := by
    rcases Nat.eq_zero_or_pos d.batch with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hH : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  refine ⟨Nat.mod_lt _ hbatch, Nat.mod_lt _ hH, ?_⟩
  unfold mOf
  rw [Nat.div_lt_iff_lt_mul (Nat.mul_pos hbatch hH)]
  rw [Nat.mul_comm (nBlocks d)]; exact hsize

/-! ## Footprints -/

/-- Every program reads (parts of) the inputs: the tensors, the indices, and with the custom mask
its data and, with a window too, the window offsets. -/
def Rf (d : Inst) : Set Int :=
  d.q.bytes ∪ d.k.bytes ∪ d.v.bytes ∪ d.kb.bytes ∪ d.vb.bytes ∪ d.qo.bytes ∪ d.kvp.bytes ∪
    d.kvi.bytes ∪ (if d.cm = true then d.cmask.bytes ∪ d.mip.bytes else ∅) ∪
    (if d.cm = true ∧ 0 < d.sw then d.wko.bytes else ∅)

/-- When the custom-mask flag is set, the custom-mask buffer lies in the readable region. -/
theorem rf_cmask {d : Inst} (h : d.cm = true) : d.cmask.bytes ⊆ Rf d := by
  intro x hx; simp [Rf, h, hx]

/-- When the custom-mask flag is set, the mask-pointer buffer lies in the readable region. -/
theorem rf_mip {d : Inst} (h : d.cm = true) : d.mip.bytes ⊆ Rf d := by
  intro x hx; simp [Rf, h, hx]

/-- When the custom-mask flag is set and the sliding window is positive, the window offset buffer lies in the readable region. -/
theorem rf_wko {d : Inst} (h : d.cm = true) (hsw : 0 < d.sw) : d.wko.bytes ⊆ Rf d := by
  intro x hx; simp [Rf, h, hsw, hx]

/-- The elements of `o_extend` of the rows `m * BM + r` (`r < BM`, below the request's new tokens)
of request `b`, head `h`, all `D` components. -/
def tile (d : Inst) (init : Mem) (b h m : Nat) : Set Nat :=
  {x | ∃ r e, r < d.BM ∧ e < d.D ∧ m * d.BM + r < extOf d init b ∧
    x = oIdx d (qsOf d init b + (m * d.BM + r)) h e}

/-- Program `i` writes its tile of `o_extend`. -/
def Wf (d : Inst) (init : Mem) (i : Nat) : Set Int :=
  d.o.elems (tile d init (bOf d i) (hOf d i) (mOf d i))

/-- **What a program leaves**: each element of its tile of `o_extend` holds a finite `bfloat16`
within the claim's tolerance of the exact attention. -/
def Q (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  ∀ r e, r < d.BM → e < d.D → mOf d i * d.BM + r < extOf d init (bOf d i) →
    ∃ c : BF16, rdB m d.o (oIdx d (qsOf d init (bOf d i) + (mOf d i * d.BM + r)) (hOf d i) e) =
        some c ∧ c.isFinite = true ∧
      |c.toReal - attnRef d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e| ≤
        tolRel d.BD (tripsPre d init (bOf d i)) (tripsExt d init (bOf d i) (mOf d i * d.BM + r))
            (Xrow d init (bOf d i) (hOf d i) (mOf d i * d.BM + r)) *
          attnAbs d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e +
        tolAbs (vSum d init (bOf d i) (hOf d i) (mOf d i * d.BM + r) e)

end Claims.ExtendAttn
