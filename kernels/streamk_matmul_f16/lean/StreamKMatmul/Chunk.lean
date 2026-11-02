import StreamKMatmul.Parts

/-!
# `streamk_matmul_f16`: one trip of the inner loop

`safe_loadDot`: lines 53–57 read only the matrices `a` and `b`, write nothing, and take the
accumulator from `cnt` chunks to `cnt + 1` (`AccInv`), by `Rules.Num.DotF16` (`acc_step`).
-/

namespace Kernels.StreamKMatmul

open TVA

/-- The element of `a`'s storage lane `(i, k)` of chunk `c` of tile `t` loads. -/
def gA (d : Inst) (t c : Nat) (x : Idx [128, 64]) : Nat :=
  rowOf d t x.1 * d.stride_am + (64 * c + (x.2.1 : Nat)) * d.stride_ak

/-- Lane `(i, k)` of chunk `c` of tile `t` loads from `a` (inside the matrix). -/
def enA (d : Inst) (t c : Nat) (x : Idx [128, 64]) : Bool :=
  decide (rowOf d t x.1 < d.M) && decide (64 * c + (x.2.1 : Nat) < d.K)

/-- The element of `b`'s storage lane `(k, j)` of chunk `c` of tile `t` loads. -/
def gB (d : Inst) (t c : Nat) (x : Idx [64, 128]) : Nat :=
  (64 * c + (x.1 : Nat)) * d.stride_bk + colOf d t x.2.1 * d.stride_bn

/-- Lane `(k, j)` of chunk `c` of tile `t` loads from `b`. -/
def enB (d : Inst) (t c : Nat) (x : Idx [64, 128]) : Bool :=
  decide (64 * c + (x.1 : Nat) < d.K) && decide (colOf d t x.2.1 < d.N)

/-- A two-term offset below a bound. -/
theorem lin2 {x X y Y a b B : Nat} (hx : x < X) (hy : y < Y) (ha : 0 < a) (hB : X * a + Y * b ≤ B) :
    x * a + y * b < B := by
  have h1 : x * a < X * a := Nat.mul_lt_mul_of_pos_right hx ha
  have h2 : y * b ≤ Y * b := Nat.mul_le_mul_right _ hy.le
  omega

theorem mul_le_fit {x X a : Nat} (hx : x < X) : x * a ≤ (X - 1) * a :=
  Nat.mul_le_mul_right _ (by omega)

/-- **One trip of the inner loop** (lines 53–57): it reads only `a` and `b`, writes nothing, and
the accumulator goes from `cnt` to `cnt + 1` chunks. -/
theorem safe_loadDot {d : Inst} (hd : d.ok) {E : Env} (hdot : Rules.Num.DotF16 E.P)
    (htot : Rules.Num.DotTotal E.P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 })
    {init m : Mem} (hva : d.a.Valid m) (hvb : d.b.Valid m) (hag : ∀ x ∈ R d, m x = init x)
    {t lo cnt : Nat} (hc : lo + cnt < steps d.K)
    {offs_m offs_n : Tensor [128] I32} {offs_k : Tensor [64] I32}
    (hm : ∀ i : Fin 128, (offs_m.get (i, ())).toInt = (rowOf d t i : Int))
    (hn : ∀ j : Fin 128, (offs_n.get (j, ())).toInt = (colOf d t j : Int))
    (hk : ∀ k : Fin 64, (offs_k.get (k, ())).toInt = ((64 * (lo + cnt) + k : Nat) : Int))
    {acc : Tensor [128, 128] F32} (hacc : AccInv d init t lo cnt acc) {W : Set Int} :
    Safe E (R d) W (fun acc' m' => m' = m ∧ AccInv d init t lo (cnt + 1) acc') m
      (loadDot d offs_m offs_n offs_k acc) := by
  have hK := hd.K_le
  have hK0 := hd.K_pos
  have hc64 : 64 * (lo + cnt) < d.K := by unfold steps at hc; omega
  have hMv : (IntV.ofNat d.M : I32).toInt = d.M := i32_nat (by have := hd.M_le; omega)
  have hNv : (IntV.ofNat d.N : I32).toInt = d.N := i32_nat (by have := hd.N_le; omega)
  have hKv : (IntV.ofNat d.K : I32).toInt = d.K := i32_nat (by omega)
  have hsam := i32_nat hd.am
  have hsak := i32_nat hd.ak
  have hsbk := i32_nat hd.bk
  have hsbn := i32_nat hd.bn
  unfold loadDot
  -- line 54
  refine safe_convert (fun _ => zero_conv_ex E.P) fun za hza => ?_
  refine safe_load_buf d.a (g := gA d t (lo + cnt)) (en := enA d t (lo + cnt)) ?_ ?_ hva ?_
    fun A hA1 hA2 => ?_
  · refine Buf.addrs_of_get d.a (fun ⟨i, k, ⟨⟩⟩ => ?_) fun ⟨i, k, ⟨⟩⟩ he => ?_
    · show (IntV.lt (offs_m.get (i, ())) (IntV.ofNat d.M) && IntV.lt (offs_k.get (k, ())) (IntV.ofNat d.K)) =
        enA d t (lo + cnt) (i, k, ())
      rw [i32_lt (hm i) hMv, i32_lt (hk k) hKv]
      simp only [enA, Nat.cast_lt]
    · simp only [enA, Bool.and_eq_true, decide_eq_true_eq] at he
      show ((d.a.ptr 0).offset ((offs_m.get (i, ()) * IntV.ofNat d.stride_am).toInt)).offset
        ((offs_k.get (k, ()) * IntV.ofNat d.stride_ak).toInt) = d.a.ptr ↑(gA d t (lo + cnt) (i, k, ()))
      have h1 : rowOf d t i * d.stride_am < 2 ^ 31 :=
        lt_of_le_of_lt (mul_le_fit he.1) hd.am_fit
      have h2 : (64 * (lo + cnt) + k) * d.stride_ak < 2 ^ 31 :=
        lt_of_le_of_lt (mul_le_fit he.2) hd.ak_fit
      rw [IntV.I32.mul_nat (hm i) hsam h1, IntV.I32.mul_nat (hk k) hsak h2, Buf.ptr_offset, Buf.ptr_offset]
      unfold gA
      congr 1; push_cast; ring
  · rintro ⟨i, k, ⟨⟩⟩ he
    simp only [enA, Bool.and_eq_true, decide_eq_true_eq] at he
    exact hd.a_len _ _ he.1 he.2
  · refine Set.Subset.trans (d.a.elems_mono ?_) Set.subset_union_left
    rintro _ ⟨⟨i, k, ⟨⟩⟩, he, rfl⟩
    simp only [enA, Bool.and_eq_true, decide_eq_true_eq] at he
    exact ⟨_, _, he.1, he.2, rfl⟩
  -- line 56
  refine safe_convert (fun _ => zero_conv_ex E.P) fun zb hzb => ?_
  refine safe_load_buf d.b (g := gB d t (lo + cnt)) (en := enB d t (lo + cnt)) ?_ ?_ hvb ?_
    fun B hB1 hB2 => ?_
  · refine Buf.addrs_of_get d.b (fun ⟨k, j, ⟨⟩⟩ => ?_) fun ⟨k, j, ⟨⟩⟩ he => ?_
    · show (IntV.lt (offs_k.get (k, ())) (IntV.ofNat d.K) && IntV.lt (offs_n.get (j, ())) (IntV.ofNat d.N)) =
        enB d t (lo + cnt) (k, j, ())
      rw [i32_lt (hk k) hKv, i32_lt (hn j) hNv]
      simp only [enB, Nat.cast_lt]
    · simp only [enB, Bool.and_eq_true, decide_eq_true_eq] at he
      show ((d.b.ptr 0).offset ((offs_k.get (k, ()) * IntV.ofNat d.stride_bk).toInt)).offset
        ((offs_n.get (j, ()) * IntV.ofNat d.stride_bn).toInt) = d.b.ptr ↑(gB d t (lo + cnt) (k, j, ()))
      have h1 : (64 * (lo + cnt) + k) * d.stride_bk < 2 ^ 31 :=
        lt_of_le_of_lt (mul_le_fit he.1) hd.bk_fit
      have h2 : colOf d t j * d.stride_bn < 2 ^ 31 :=
        lt_of_le_of_lt (mul_le_fit he.2) hd.bn_fit
      rw [IntV.I32.mul_nat (hk k) hsbk h1, IntV.I32.mul_nat (hn j) hsbn h2, Buf.ptr_offset, Buf.ptr_offset]
      unfold gB
      congr 1; push_cast; ring
  · rintro ⟨k, j, ⟨⟩⟩ he
    simp only [enB, Bool.and_eq_true, decide_eq_true_eq] at he
    exact hd.b_len _ _ he.1 he.2
  · refine Set.Subset.trans (d.b.elems_mono ?_) Set.subset_union_right
    rintro _ ⟨⟨k, j, ⟨⟩⟩, he, rfl⟩
    simp only [enB, Bool.and_eq_true, decide_eq_true_eq] at he
    exact ⟨_, _, he.1, he.2, rfl⟩
  -- line 57
  refine safe_dot (htot A B acc) fun r hr => ?_
  refine safe_ret ⟨rfl, ?_⟩
  rintro ⟨i, j, ⟨⟩⟩ he va vb hra hcb
  simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he
  have hread : ∀ {dd : DType} {S : Set Nat} (bf : Buf dd) (e : Nat), e ∈ S → bf.elems S ⊆ R d →
      m.read dd (bf.ptr (e : Int)).addr = init.read dd (bf.ptr (e : Int)).addr := by
    intro dd S bf e he hS
    exact Mem.read_congr fun x hx => hag x (hS (bf.span_sub_elems he hx))
  refine acc_step hdot hr i j (by omega) hc64 (fun e he => (hra e he).2) (fun e he => (hcb e he).2)
    (fun k => ⟨fun h => ?_, fun h => ?_⟩) (fun k => ⟨fun h => ?_, fun h => ?_⟩) (hacc (i, j, ()) (by
      simp only [cEn, Bool.and_eq_true, decide_eq_true_eq]; exact he) va vb hra hcb)
  · have hen : enA d t (lo + cnt) (i, k, ()) = true := by
      simp only [enA, Bool.and_eq_true, decide_eq_true_eq]; exact ⟨he.1, h⟩
    have h1 := hA1 (i, k, ()) hen
    have h2 := (hra _ h).1
    rw [hread d.a _ (show gA d t (lo + cnt) (i, k, ()) ∈ aIdx d from ⟨_, _, he.1, h, rfl⟩)
      Set.subset_union_left] at h1
    simp only [gA] at h1
    exact Option.some.inj (h1.symm.trans h2)
  · have h1 := hA2 (i, k, ()) (by simp only [enA, Bool.and_eq_false_iff, decide_eq_false_iff_not]; omega) za rfl
    show (A.get (i, k, ())).isFinite = true ∧ (A.get (i, k, ())).toRat = 0
    rw [h1]
    exact zero_conv (hza _)
  · have hen : enB d t (lo + cnt) (k, j, ()) = true := by
      simp only [enB, Bool.and_eq_true, decide_eq_true_eq]; exact ⟨h, he.2⟩
    have h1 := hB1 (k, j, ()) hen
    have h2 := (hcb _ h).1
    rw [hread d.b _ (show gB d t (lo + cnt) (k, j, ()) ∈ bIdx d from ⟨_, _, h, he.2, rfl⟩)
      Set.subset_union_right] at h1
    simp only [gB] at h1
    exact Option.some.inj (h1.symm.trans h2)
  · have h1 := hB2 (k, j, ()) (by simp only [enB, Bool.and_eq_false_iff, decide_eq_false_iff_not]; omega) zb rfl
    show (B.get (k, j, ())).isFinite = true ∧ (B.get (k, j, ())).toRat = 0
    rw [h1]
    exact zero_conv (hzb _)

end Kernels.StreamKMatmul
