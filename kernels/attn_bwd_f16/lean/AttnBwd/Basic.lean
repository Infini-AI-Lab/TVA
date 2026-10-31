import AttnBwd.Acc

/-!
# `attn_bwd_f16`: lanes, tiles, covered rows and sizes

* `RowsAt`: an `int32` block holding `r0, r0 + 1, …` (`offs_m`, `offs_n`, `offs_k`).
* `tile_rc`, `tile_cr`, `vec_r`, `of2_add_bc`: the pointer tiles of the source
  (`B + offs[:, None] * stride_tok + offs_k[None, :] * stride_d` and its transpose, `M + offs_m`,
  and the increments `ptrs += step * stride_tok`) as tiles of buffer elements.
* `sum_cover`, `cover_succ`: the rows a loop has covered, as a `Finset`.
* `KVInv`, `QInv`: the accumulators' invariants over the covered rows.
* Sizes of the `Spec` quantities on covered inputs (`aQK_le`, `p16B_le`, `ds16B_le`, …).
-/

namespace Kernels.AttnBwd

open TVA

/-! ## `int32` lanes -/

theorem i32_nat {n : Nat} (h : n < 2 ^ 31) : (IntV.ofNat n : I32).toInt = n := IntV.I32.toInt_ofNat h

/-- Lanes `r0, r0 + 1, …` of an `int32` block. -/
def RowsAt {B : Nat} (x : Tensor [B] I32) (r0 : Nat) : Prop :=
  ∀ i : Fin B, (x.get1 i).toInt = ((r0 + i : Nat) : Int)

theorem rowsAt_arange0 {B : Nat} (hB : B < 2 ^ 31) : RowsAt (tl.arange 0 B : Tensor [B] I32) 0 := by
  intro i
  show (IntV.ofInt (((0 : Nat) : Int) + ((i : Nat) : Int)) : I32).toInt = ((0 + (i : Nat) : Nat) : Int)
  have := i.2
  rw [IntV.I32.toInt_ofInt (by constructor <;> push_cast <;> omega)]
  push_cast; ring

theorem rowsAt_add {B : Nat} {x0 : I32} {r0 : Nat} (hx : x0.toInt = r0) (h : r0 + B < 2 ^ 31) :
    RowsAt (x0 + tl.arange 0 B : Tensor [B] I32) r0 := by
  intro i
  show (x0 + (tl.arange 0 B : Tensor [B] I32).get1 i).toInt = _
  have := rowsAt_arange0 (B := B) (by omega) i
  have hi := i.2
  rw [IntV.I32.add_nat hx this (by omega)]
  push_cast; ring

/-! ## Pointer tiles -/

/-- The `[R, HD]` tile `B + rows[:, None] * HD + cols[None, :] * 1` of a contiguous tensor:
row `r0 + i`, column `c`. -/
theorem tile_rc {dt : DType} (Bf : Buf dt) (base : Nat) {R HD : Nat} {rows : Tensor [R] I32}
    {cols : Tensor [HD] I32} {r0 : Nat} (hr : RowsAt rows r0) (hc : RowsAt cols 0)
    (hfit : (r0 + R) * HD < 2 ^ 31) :
    (Bf.ptr ↑base + rows.col HD * (IntV.ofNat HD : I32) + cols.row R * (IntV.ofNat 1 : I32) :
      Tensor [R, HD] (Ptr dt)) = Tensor.of2 fun i c => Bf.ptr ↑(base + (r0 + i) * HD + c) := by
  refine Tensor.ext (funext fun ⟨i, c, ⟨⟩⟩ => ?_)
  show ((Bf.ptr ↑base).offset ((rows.get1 i * IntV.ofNat HD).toInt)).offset
    ((cols.get1 c * IntV.ofNat 1).toInt) = Bf.ptr ↑(base + (r0 + i) * HD + c)
  have hi := i.2
  have hc' := c.2
  have h1 : (r0 + i) * HD < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (by omega)) hfit
  have h2 : HD < 2 ^ 31 := lt_of_le_of_lt (Nat.le_mul_of_pos_left _ (by omega)) hfit
  rw [IntV.I32.mul_nat (hr i) (i32_nat h2) h1, IntV.I32.mul_nat (hc c) (i32_nat (by norm_num)) (by omega),
    Buf.ptr_offset, Buf.ptr_offset]
  congr 1; push_cast; ring

/-- The transposed `[HD, R]` tile `B + rows[None, :] * HD + cols[:, None] * 1`. -/
theorem tile_cr {dt : DType} (Bf : Buf dt) (base : Nat) {R HD : Nat} {rows : Tensor [R] I32}
    {cols : Tensor [HD] I32} {r0 : Nat} (hr : RowsAt rows r0) (hc : RowsAt cols 0)
    (hfit : (r0 + R) * HD < 2 ^ 31) :
    (Bf.ptr ↑base + rows.row HD * (IntV.ofNat HD : I32) + cols.col R * (IntV.ofNat 1 : I32) :
      Tensor [HD, R] (Ptr dt)) = Tensor.of2 fun c i => Bf.ptr ↑(base + (r0 + i) * HD + c) := by
  refine Tensor.ext (funext fun ⟨c, i, ⟨⟩⟩ => ?_)
  show ((Bf.ptr ↑base).offset ((rows.get1 i * IntV.ofNat HD).toInt)).offset
    ((cols.get1 c * IntV.ofNat 1).toInt) = Bf.ptr ↑(base + (r0 + i) * HD + c)
  have hi := i.2
  have hc' := c.2
  have h1 : (r0 + i) * HD < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (by omega)) hfit
  have h2 : HD < 2 ^ 31 := lt_of_le_of_lt (Nat.le_mul_of_pos_left _ (by omega)) hfit
  rw [IntV.I32.mul_nat (hr i) (i32_nat h2) h1, IntV.I32.mul_nat (hc c) (i32_nat (by norm_num)) (by omega),
    Buf.ptr_offset, Buf.ptr_offset]
  congr 1; push_cast; ring

/-- The `[R]` block `B + rows` (`M + offs_m`, `D + offs_m`). -/
theorem vec_r {dt : DType} (Bf : Buf dt) (base : Nat) {R : Nat} {rows : Tensor [R] I32} {r0 : Nat}
    (hr : RowsAt rows r0) :
    (Bf.ptr ↑base + rows : Tensor [R] (Ptr dt)) = Tensor.of1 fun i => Bf.ptr ↑(base + (r0 + i)) := by
  refine Tensor.ext (funext fun ⟨i, ⟨⟩⟩ => ?_)
  show (Bf.ptr ↑base).offset (rows.get1 i).toInt = Bf.ptr ↑(base + (r0 + i))
  rw [hr i, Buf.ptr_offset]
  congr 1

/-- `ptrs += k` on a tile of buffer elements. -/
theorem of2_add_bc {dt : DType} (Bf : Buf dt) {R C : Nat} (F : Fin R → Fin C → Nat) {a : I32} {k : Nat}
    (ha : a.toInt = k) :
    (Tensor.of2 (fun i c => Bf.ptr ↑(F i c)) + (bc a : Tensor [R, C] I32) : Tensor [R, C] (Ptr dt)) =
      Tensor.of2 fun i c => Bf.ptr ↑(F i c + k) := by
  refine Tensor.ext (funext fun ⟨i, c, ⟨⟩⟩ => ?_)
  show (Bf.ptr ↑(F i c)).offset a.toInt = Bf.ptr ↑(F i c + k)
  rw [ha, Buf.ptr_offset]
  congr 1

/-! ## Covered rows -/

theorem sum_cover {C : Finset ℕ} {lo B : ℕ} (hC : ∀ x ∈ C, x < lo ∨ lo + B ≤ x) (f : ℕ → ℝ) :
    ∑ m ∈ C ∪ Finset.Ico lo (lo + B), f m = ∑ m ∈ C, f m + ∑ i : Fin B, f (lo + i) := by
  rw [Finset.sum_union]
  · rw [Finset.sum_Ico_eq_sum_range, show lo + B - lo = B by omega,
      ← Fin.sum_univ_eq_sum_range (fun i => f (lo + i)) B]
  · rw [Finset.disjoint_left]
    intro x hx hx'
    rw [Finset.mem_Ico] at hx'
    rcases hC x hx with h | h <;> omega

theorem cover_succ (C : Finset ℕ) (m0 BM t : ℕ) :
    C ∪ Finset.Ico m0 (m0 + (t + 1) * BM) =
      (C ∪ Finset.Ico m0 (m0 + t * BM)) ∪ Finset.Ico (m0 + t * BM) (m0 + t * BM + BM) := by
  rw [Finset.union_assoc, Finset.Ico_union_Ico_eq_Ico (by omega) (by omega)]
  congr 2; rw [Nat.add_mul]; ring

theorem cover_apart {C : Finset ℕ} {m0 BM T t : ℕ} (hC : ∀ x ∈ C, x < m0 ∨ m0 + T * BM ≤ x)
    (ht : t < T) : ∀ x ∈ C ∪ Finset.Ico m0 (m0 + t * BM), x < m0 + t * BM ∨ m0 + t * BM + BM ≤ x := by
  intro x hx
  rcases Finset.mem_union.mp hx with h | h
  · rcases hC x h with h' | h'
    · left; omega
    · right
      have : (t + 1) * BM ≤ T * BM := Nat.mul_le_mul_right _ ht
      rw [Nat.add_mul] at this; omega
  · rw [Finset.mem_Ico] at h; left; omega

theorem cover_sub {C : Finset ℕ} {m0 BM t N : ℕ} (hC : C ⊆ Finset.range N) (h : m0 + t * BM ≤ N) :
    C ∪ Finset.Ico m0 (m0 + t * BM) ⊆ Finset.range N := by
  intro x hx
  rcases Finset.mem_union.mp hx with h' | h'
  · exact hC h'
  · rw [Finset.mem_Ico] at h'; rw [Finset.mem_range]; omega

/-! ## The invariants of the accumulators -/

/-- The `dv` and `dk` accumulators of program `p` of head `b`, after `T` steps covering the query
rows `C`: element `(r, c)` (key row `n = 128p + r`) satisfies `AccI` with the exact sums over
`m ∈ C` of `P[n, m]·dO[m, c]` and of `dS[n, m]·Q[m, c]`, their errors and bounds. -/
def KVInv (d : Inst) (init : Mem) (b p : Nat) (C : Finset ℕ) (T : ℕ)
    (dk dv : Tensor [128, d.HEAD_DIM] F32) : Prop :=
  ∀ (r : Fin 128) (c : Fin d.HEAD_DIM) (hv : HeadV), hv.Holds d init b → hv.Ok d →
    AccI (dv.get2 r c) (∑ m ∈ C, pE d hv (128 * p + r) m * (hv.dO m c).toReal)
      (∑ m ∈ C, p16Err d hv (128 * p + r) m * |(hv.dO m c).toReal|)
      (∑ m ∈ C, p16B d hv (128 * p + r) m * |(hv.dO m c).toReal|) T ∧
    AccI (dk.get2 r c) (∑ m ∈ C, dsE d hv (128 * p + r) m * (hv.q m c).toReal)
      (∑ m ∈ C, ds16Err d hv (128 * p + r) m * |(hv.q m c).toReal|)
      (∑ m ∈ C, ds16B d hv (128 * p + r) m * |(hv.q m c).toReal|) T

/-- The `dq` accumulator of program `p` of head `b`, after `T` steps covering the key rows `C`:
element `(r, c)` (query row `m = 128p + r`) satisfies `AccI` with the exact sum over `n ∈ C` of
`dS[n, m]·K[n, c]`, its error and bound. -/
def QInv (d : Inst) (init : Mem) (b p : Nat) (C : Finset ℕ) (T : ℕ)
    (dq : Tensor [128, d.HEAD_DIM] F32) : Prop :=
  ∀ (r : Fin 128) (c : Fin d.HEAD_DIM) (hv : HeadV), hv.Holds d init b → hv.Ok d →
    AccI (dq.get2 r c) (∑ n ∈ C, dsE d hv n (128 * p + r) * (hv.k n c).toReal)
      (∑ n ∈ C, ds16Err d hv n (128 * p + r) * |(hv.k n c).toReal|)
      (∑ n ∈ C, ds16B d hv n (128 * p + r) * |(hv.k n c).toReal|) T

theorem kvInv_zero (d : Inst) (init : Mem) (b p : Nat) :
    KVInv d init b p ∅ 0 (tl.zeros [128, d.HEAD_DIM] .f32) (tl.zeros [128, d.HEAD_DIM] .f32) := by
  intro r c hv _ _
  simp only [Finset.sum_empty]
  exact ⟨accI_zero, accI_zero⟩

theorem qInv_zero (d : Inst) (init : Mem) (b p : Nat) :
    QInv d init b p ∅ 0 (tl.zeros [128, d.HEAD_DIM] .f32) := by
  intro r c hv _ _
  simp only [Finset.sum_empty]
  exact accI_zero

/-! ## Sizes -/

section Sizes
variable {d : Inst} {hv : HeadV}

theorem hd_le (hd : d.ok) : d.HEAD_DIM ≤ 256 := by rcases hd.hd with h | h | h | h | h <;> omega

theorem hd_pos (hd : d.ok) : 0 < d.HEAD_DIM := by rcases hd.hd with h | h | h | h | h <;> omega

theorem prodR_le {x y : F16} (hx : x.isFinite = true) (hy : y.isFinite = true) :
    |x.toReal * y.toReal| ≤ 2 ^ 32 := by
  rw [abs_mul]
  have h1 := f16_abs_leR hx
  have h2 := f16_abs_leR hy
  have : |x.toReal| * |y.toReal| ≤ 65504 * 65504 := mul_le_mul h1 h2 (abs_nonneg _) (by norm_num)
  linarith

theorem aQK_le (hd : d.ok) (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX) (hm : m < d.N_CTX) :
    aQK d hv n m ≤ 2 ^ 40 := by
  unfold aQK
  calc ∑ c ∈ Finset.range d.HEAD_DIM, |(hv.k n c).toReal * (hv.q m c).toReal|
      ≤ ∑ _c ∈ Finset.range d.HEAD_DIM, (2 : ℝ) ^ 32 := Finset.sum_le_sum fun c hc =>
        prodR_le (hok.k n c hn (Finset.mem_range.mp hc)) (hok.q m c hm (Finset.mem_range.mp hc))
    _ = d.HEAD_DIM * 2 ^ 32 := by simp
    _ ≤ 2 ^ 40 := by
      have : (d.HEAD_DIM : ℝ) ≤ 256 := by exact_mod_cast hd_le hd
      nlinarith

theorem aVD_le (hd : d.ok) (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX) (hm : m < d.N_CTX) :
    aVD d hv n m ≤ 2 ^ 40 := by
  unfold aVD
  calc ∑ c ∈ Finset.range d.HEAD_DIM, |(hv.v n c).toReal * (hv.dO m c).toReal|
      ≤ ∑ _c ∈ Finset.range d.HEAD_DIM, (2 : ℝ) ^ 32 := Finset.sum_le_sum fun c hc =>
        prodR_le (hok.v n c hn (Finset.mem_range.mp hc)) (hok.dO m c hm (Finset.mem_range.mp hc))
    _ = d.HEAD_DIM * 2 ^ 32 := by simp
    _ ≤ 2 ^ 40 := by
      have : (d.HEAD_DIM : ℝ) ≤ 256 := by exact_mod_cast hd_le hd
      nlinarith

theorem sQK_abs (n m : Nat) : |sQK d hv n m| ≤ aQK d hv n m := Finset.abs_sum_le_sum_abs _ _

theorem sVD_abs (n m : Nat) : |sVD d hv n m| ≤ aVD d hv n m := Finset.abs_sum_le_sum_abs _ _

theorem eSub_nn (hd : d.ok) {s a x : ℝ} (ha : 0 ≤ a) : 0 ≤ eSub d s a x := by
  unfold eSub
  have := (rH_le hd).1
  have := (aH_le hd).1
  have : 0 ≤ u32 := by rw [u32_eq]; positivity
  have : 0 ≤ eta32 := by rw [eta32_eq]; positivity
  positivity

theorem aQK_nn (n m : Nat) : 0 ≤ aQK d hv n m := Finset.sum_nonneg fun _ _ => abs_nonneg _

theorem aVD_nn (n m : Nat) : 0 ≤ aVD d hv n m := Finset.sum_nonneg fun _ _ => abs_nonneg _

theorem pBr_le (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX) (hm : m < d.N_CTX) (hal : Allowed d n m) :
    0 ≤ pBr d hv n m ∧ pBr d hv n m ≤ 2 ^ 16 := by
  obtain ⟨h1, -⟩ := hok.fits n m hn hm hal
  unfold pBr
  have h2 : (2 : ℝ) ^ (xE d hv n m + eX d hv n m) ≤ 2 ^ 15 := by
    have := two_rpow_le h1; rwa [two_rpow_15] at this
  have h3 := two_rpow_pos (xE d hv n m + eX d hv n m)
  constructor
  · positivity
  · have : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 ^ (xE d hv n m + eX d hv n m) ≤ (1 + 2 ^ (-22 : ℤ)) * 2 ^ 15 :=
      mul_le_mul_of_nonneg_left h2 (by positivity)
    have : (1 + (2 : ℝ) ^ (-22 : ℤ)) * 2 ^ 15 + 2 ^ (-126 : ℤ) ≤ 2 ^ 16 := by norm_num
    linarith

theorem yB_nn (hd : d.ok) (n m : Nat) : 0 ≤ yB d hv n m := by
  unfold yB eY
  have := eSub_nn (s := sVD d hv n m) (x := (hv.dl m).toReal) hd (aVD_nn (d := d) (hv := hv) n m)
  positivity

theorem p16B_bd (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX) (hm : m < d.N_CTX) :
    0 ≤ p16B d hv n m ∧ p16B d hv n m ≤ 2 ^ 17 := by
  unfold p16B
  split_ifs with hal
  · obtain ⟨h0, h1⟩ := pBr_le hok hn hm hal
    rw [u16_eq, eta16_eq]
    constructor
    · positivity
    · have : (1 + (2 : ℝ) ^ (-11 : ℤ)) * pBr d hv n m ≤ (1 + 2 ^ (-11 : ℤ)) * 2 ^ 16 :=
        mul_le_mul_of_nonneg_left h1 (by positivity)
      have : (1 + (2 : ℝ) ^ (-11 : ℤ)) * 2 ^ 16 + 2 ^ (-25 : ℤ) ≤ 2 ^ 17 := by norm_num
      linarith
  · exact ⟨le_rfl, by norm_num⟩

theorem dsB_bd (hd : d.ok) (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX) (hm : m < d.N_CTX)
    (hal : Allowed d n m) : 0 ≤ dsB d hv n m ∧ dsB d hv n m ≤ 2 ^ 16 := by
  obtain ⟨-, h2⟩ := hok.fits n m hn hm hal
  obtain ⟨h0, -⟩ := pBr_le hok hn hm hal
  have hy := yB_nn (hv := hv) hd n m
  unfold dsB
  rw [u32_eq, eta32_eq]
  constructor
  · positivity
  · have : (1 + (2 : ℝ) ^ (-24 : ℤ)) * (pBr d hv n m * yB d hv n m) ≤ (1 + 2 ^ (-24 : ℤ)) * 2 ^ 15 :=
      mul_le_mul_of_nonneg_left h2 (by positivity)
    have : (1 + (2 : ℝ) ^ (-24 : ℤ)) * 2 ^ 15 + 2 ^ (-150 : ℤ) ≤ 2 ^ 16 := by norm_num
    linarith

theorem ds16B_bd (hd : d.ok) (hok : hv.Ok d) {n m : Nat} (hn : n < d.N_CTX) (hm : m < d.N_CTX) :
    0 ≤ ds16B d hv n m ∧ ds16B d hv n m ≤ 2 ^ 17 := by
  unfold ds16B
  split_ifs with hal
  · obtain ⟨h0, h1⟩ := dsB_bd hd hok hn hm hal
    rw [u16_eq, eta16_eq]
    constructor
    · positivity
    · have : (1 + (2 : ℝ) ^ (-11 : ℤ)) * dsB d hv n m ≤ (1 + 2 ^ (-11 : ℤ)) * 2 ^ 16 :=
        mul_le_mul_of_nonneg_left h1 (by positivity)
      have : (1 + (2 : ℝ) ^ (-11 : ℤ)) * 2 ^ 16 + 2 ^ (-25 : ℤ) ≤ 2 ^ 17 := by norm_num
      linarith
  · exact ⟨le_rfl, by norm_num⟩

/-- A sum over covered rows of bounds `≤ 2^17` times `float16` magnitudes: at most `2^47`. -/
theorem sum_bd_le (hd : d.ok) {C : Finset ℕ} (hC : C ⊆ Finset.range d.N_CTX) (g : ℕ → ℝ) (x : ℕ → F16)
    (hg : ∀ m, m < d.N_CTX → 0 ≤ g m ∧ g m ≤ 2 ^ 17) (hx : ∀ m, m < d.N_CTX → (x m).isFinite = true) :
    ∑ m ∈ C, g m * |(x m).toReal| ≤ 2 ^ 47 := by
  have h1 : ∑ m ∈ C, g m * |(x m).toReal| ≤ ∑ _m ∈ C, (2 : ℝ) ^ 33 := by
    refine Finset.sum_le_sum fun m hm => ?_
    have hmN : m < d.N_CTX := Finset.mem_range.mp (hC hm)
    obtain ⟨h0, h17⟩ := hg m hmN
    have := f16_abs_leR (hx m hmN)
    have : g m * |(x m).toReal| ≤ 2 ^ 17 * 65504 := mul_le_mul h17 this (abs_nonneg _) (by norm_num)
    linarith
  have h2 : (C.card : ℝ) ≤ 2 ^ 14 := by
    have := Finset.card_le_card hC
    rw [Finset.card_range] at this
    have h := hd.n_le
    exact_mod_cast this.trans h
  rw [Finset.sum_const, nsmul_eq_mul] at h1
  nlinarith

end Sizes

end Kernels.AttnBwd
