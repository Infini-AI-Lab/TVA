import GroupedGemm.Defs

/-!
# Grouped GEMM: the `K` loop of one tile

`safe_kloop`: from the tile's starting pointers, `T = K / BK` iterations of `kstep` leave
every accumulator element finite and within `errC K S` of the exact `Σₖ A[i,k]·B[k,j]`. Each
step is `DotLoop.dotAdd_step` (either form of `accumulator += tl.dot(a, b)`), accumulated with
`DotLoop.accum_step`.
-/
namespace TVAExamples.GroupedGemm

open TVA TVA.Math.Float

/-! ## Constants -/

theorem rel_nonneg (K : Nat) : 0 ≤ rel K := by unfold rel; positivity
theorem abs_nonneg' (K : Nat) : 0 ≤ abs K := by unfold abs; positivity

theorem binary32_eta : Format.binary32.eta = 2 ^ (-150 : ℤ) := by unfold Format.eta; rfl

theorem ρ_nonneg (d : Inst) : 0 ≤ ρ d := by
  unfold ρ DotLoop.stepRel
  have := rel_nonneg d.BK
  have := Format.u_nonneg (f := Format.binary32)
  positivity

theorem α_nonneg (d : Inst) : 0 ≤ α d := by
  unfold α DotLoop.stepAbs
  have := abs_nonneg' d.BK
  have := Format.u_nonneg (f := Format.binary32)
  have := Format.eta_nonneg (f := Format.binary32)
  positivity

theorem rel_le_one {K : Nat} (hK : K ≤ 2 ^ 20) : rel K ≤ 1 := by
  unfold rel
  have : (K : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hK
  have h2 : (2 : ℚ) ^ (-22 : ℤ) = 1 / 2 ^ 22 := by norm_num
  rw [h2]
  have : ((K : ℚ) + 2) ≤ 2 ^ 22 := by linarith
  have h3 : (0 : ℚ) < 2 ^ 22 := by positivity
  rw [mul_one_div, div_le_one h3]
  exact this

theorem abs_le_one {K : Nat} (hK : K ≤ 2 ^ 20) : abs K ≤ 1 := by
  unfold abs
  have : (K : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hK
  have h2 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  rw [h2, mul_one_div, div_le_one (by positivity)]
  have : (2 : ℚ) ^ 20 + 2 ≤ 2 ^ 126 := by norm_num
  linarith

theorem α_le (d : Inst) (hK : d.BK ≤ 2 ^ 20) : α d ≤ 2 ^ (-100 : ℤ) := by
  unfold α DotLoop.stepAbs abs
  rw [DotLoop.binary32_u, binary32_eta]
  have : (d.BK : ℚ) ≤ 2 ^ 20 := by exact_mod_cast hK
  have e1 : (2 : ℚ) ^ (-126 : ℤ) = 1 / 2 ^ 126 := by norm_num
  have e2 : (2 : ℚ) ^ (-24 : ℤ) = 1 / 2 ^ 24 := by norm_num
  have e3 : (2 : ℚ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
  have e4 : (2 : ℚ) ^ (-100 : ℤ) = 1 / 2 ^ 100 := by norm_num
  rw [e1, e2, e3, e4]
  have hB : ((d.BK : ℚ) + 2) * (1 / 2 ^ 126) ≤ (2 ^ 20 + 2) * (1 / 2 ^ 126) :=
    mul_le_mul_of_nonneg_right (by linarith) (by positivity)
  have hp : (0 : ℚ) ≤ ((d.BK : ℚ) + 2) * (1 / 2 ^ 126) := by positivity
  nlinarith

theorem f16_abs_le {x : F16} (hx : x.isFinite = true) : |x.toRat| ≤ 65504 := by
  have := FloatV.abs_toRat_le_maxVal Format.binary16_regular rfl hx
  rwa [Format.binary16_maxVal] at this

theorem f32_maxVal_ge : (2 : ℚ) ^ 127 ≤ Format.binary32.maxVal := by
  have h := Format.two_pow_emax_le (f := Format.binary32) rfl
  have he : Format.binary32.emax = 127 := by decide
  rw [he] at h
  exact_mod_cast h

/-- The room `DotLoop.dotAdd_step` needs, for magnitudes up to `2^66`. -/
theorem room {K : Nat} (hK : K ≤ 2 ^ 20) {c A : ℚ} (hc : 0 ≤ c) (hA : 0 ≤ A) (h : c + A ≤ 2 ^ 66) :
    2 * ((1 + rel K) * (c + A) + abs K + 2 * Format.binary32.eta) ≤ Format.binary32.maxVal := by
  have h1 := rel_le_one hK
  have h2 := abs_le_one hK
  have h3 := rel_nonneg K
  have he : Format.binary32.eta ≤ 1 := by rw [binary32_eta]; norm_num
  have h4 : (1 + rel K) * (c + A) ≤ 2 * 2 ^ 66 := by nlinarith
  have := f32_maxVal_ge
  have : (2 : ℚ) * (2 * 2 ^ 66 + 1 + 2 * 1) ≤ 2 ^ 127 := by norm_num
  linarith

/-! ## Exact partial sums -/

/-- `Σ_{k < n} A[i,k]·B[k,j]`. -/
def Gemm.S (p : Gemm) (m : Mem) (i j n : Nat) : ℚ := ∑ k ∈ Finset.range n, p.A m i k * p.B m k j
/-- `Σ_{k < n} |A[i,k]·B[k,j]|`. -/
def Gemm.Sabs (p : Gemm) (m : Mem) (i j n : Nat) : ℚ := ∑ k ∈ Finset.range n, |p.A m i k * p.B m k j|

theorem Gemm.abs_S_le (p : Gemm) (m : Mem) (i j n : Nat) : |p.S m i j n| ≤ p.Sabs m i j n :=
  Finset.abs_sum_le_sum_abs _ _

theorem Gemm.Sabs_nonneg (p : Gemm) (m : Mem) (i j n : Nat) : 0 ≤ p.Sabs m i j n :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

/-- The entries of `A` and `B` the tile reads are finite in `m`. -/
def Gemm.Finite (p : Gemm) (m : Mem) : Prop :=
  (∀ i k, i < p.M → k < p.K → ∃ x : F16, m.read .f16 (p.a.ptr ((i * p.lda + k : ℕ) : ℤ)).addr = some x ∧
      x.isFinite = true) ∧
  (∀ k j, k < p.K → j < p.N → ∃ x : F16, m.read .f16 (p.b.ptr ((k * p.ldb + j : ℕ) : ℤ)).addr = some x ∧
      x.isFinite = true)

theorem val16_of_read {m : Mem} {q : Ptr .f16} {x : F16} (h : m.read .f16 q.addr = some x) :
    val16 m q = x.toRat := by
  unfold val16; rw [h]

theorem Gemm.abs_A_le {p : Gemm} {m : Mem} (hf : p.Finite m) {i k : Nat} (hi : i < p.M) (hk : k < p.K) :
    |p.A m i k| ≤ 65504 := by
  obtain ⟨x, hx, hfin⟩ := hf.1 i k hi hk
  unfold Gemm.A; rw [val16_of_read hx]; exact f16_abs_le hfin

theorem Gemm.abs_B_le {p : Gemm} {m : Mem} (hf : p.Finite m) {k j : Nat} (hk : k < p.K) (hj : j < p.N) :
    |p.B m k j| ≤ 65504 := by
  obtain ⟨x, hx, hfin⟩ := hf.2 k j hk hj
  unfold Gemm.B; rw [val16_of_read hx]; exact f16_abs_le hfin

theorem Gemm.Sabs_le {p : Gemm} {m : Mem} (hf : p.Finite m) {i j n : Nat} (hi : i < p.M) (hj : j < p.N)
    (hn : n ≤ p.K) : p.Sabs m i j n ≤ n * 65504 ^ 2 := by
  unfold Gemm.Sabs
  have : ∀ k ∈ Finset.range n, |p.A m i k * p.B m k j| ≤ 65504 ^ 2 := by
    intro k hk
    have hk' : k < p.K := lt_of_lt_of_le (Finset.mem_range.mp hk) hn
    rw [abs_mul, sq]
    exact mul_le_mul (p.abs_A_le hf hi hk') (p.abs_B_le hf hk' hj) (abs_nonneg _) (by norm_num)
  calc ∑ k ∈ Finset.range n, |p.A m i k * p.B m k j| ≤ ∑ _k ∈ Finset.range n, (65504 : ℚ) ^ 2 :=
        Finset.sum_le_sum this
    _ = n * 65504 ^ 2 := by simp

/-! ## The loop invariant -/

/-- The `K` loop after `s` steps, for the tile with rows `row r` and columns `col c`: the
pointers have advanced `s` blocks, and each accumulator element is finite and within
`bound (ρ d) (α d) s` of the partial sum over the first `s·BK` products. -/
def KInv (d : Inst) (p : Gemm) (m : Mem) (row : Fin d.BM → Nat) (col : Fin d.BN → Nat) (s : Nat)
    (st : Tensor [d.BM, d.BN] F32 × Tensor [d.BM, d.BK] (Ptr .f16) × Tensor [d.BK, d.BN] (Ptr .f16)) :
    Prop :=
  (∀ r k, st.2.1.get2 r k = p.a.ptr ((row r * p.lda + (s * d.BK + k) : ℕ) : ℤ)) ∧
  (∀ k c, st.2.2.get2 k c = p.b.ptr (((s * d.BK + k) * p.ldb + col c : ℕ) : ℤ)) ∧
  ∀ r c, (st.1.get2 r c).isFinite = true ∧
    |(st.1.get2 r c).toRat - p.S m (row r) (col c) (s * d.BK)| ≤
      DotLoop.bound (ρ d) (α d) s (p.Sabs m (row r) (col c) (s * d.BK))

section Loop

variable {E : Env} {R W : Set Int} {γ : Type} {Q : γ → Mem → Prop} {m : Mem}

/-- The data a tile's `K` loop needs: the problem's conditions, its bounds, and the rows and
columns of the tile inside `C`. -/
structure KCtx (d : Inst) (p : Gemm) (m : Mem) (row : Fin d.BM → Nat) (col : Fin d.BN → Nat)
    (R : Set Int) : Prop where
  ok : Gemm.ok d p
  BK_le : d.BK ≤ 2 ^ 20
  row_lt : ∀ r, row r < p.M
  col_lt : ∀ c, col c < p.N
  fin : p.Finite m
  va : p.a.Valid m
  vb : p.b.Valid m
  ra : p.a.bytes ⊆ R
  rb : p.b.bytes ⊆ R

theorem KCtx.K_lt {d : Inst} {p : Gemm} {row col} (h : KCtx d p m row col R) : p.K < 2 ^ 31 := by
  have := h.ok.2.2.2.2.2.2.2.2.2.2.2.1; omega

theorem KCtx.T_rho {d : Inst} {p : Gemm} {row col} (h : KCtx d p m row col R) :
    ((p.K / d.BK : Nat) : ℚ) * ρ d ≤ 1 / 2 := h.ok.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2

/-- One step of the `K` loop keeps the invariant (memory is unchanged: the step only loads). -/
theorem safe_kstep {d : Inst} {p : Gemm} {row : Fin d.BM → Nat} {col : Fin d.BN → Nat}
    (hdot : Rules.Num.DotTotal E.P cfg) (hacc : Rules.Num.DotAcc E.P cfg (fun _ => True) rel abs)
    (hc : KCtx d p m row col R) {ldb : I32} (hldb : ldb.toInt = (p.ldb : Int)) {s : Nat}
    (hs : s < p.K / d.BK) (kk : I32) {st} (hI : KInv d p m row col s st) :
    Safe E R W (fun st' m₂ => m₂ = m ∧ KInv d p m row col (s + 1) st') m (kstep d.BM d.BN d.BK ldb kk st) := by
  obtain ⟨acc, ap, bp⟩ := st
  obtain ⟨hap, hbp, hinv⟩ := hI
  obtain ⟨hal, hbl, -, hKa, hNb, -, -, -, -, -, -, hKfit, -, -, -, -, hBKldb, -, -⟩ := hc.ok
  have hBK : 0 < d.BK := by
    rcases Nat.eq_zero_or_pos d.BK with h | h
    · simp [h] at hs
    · exact h
  have hsK : (s + 1) * d.BK ≤ p.K := by
    have := Nat.lt_div_iff_mul_lt hBK |>.mp hs
    have : (s + 1) * d.BK ≤ p.K / d.BK * d.BK := Nat.mul_le_mul_right _ hs
    exact this.trans (Nat.div_mul_le_self _ _)
  simp only [kstep]
  -- `a = tl.load(a_ptrs)`
  have hgA : ∀ i : Fin d.BM × Fin d.BK × Unit, row i.1 * p.lda + (s * d.BK + i.2.1) < p.a.len := by
    rintro ⟨r, k, ⟨⟩⟩
    have hr := hc.row_lt r
    have hk : s * d.BK + k < p.lda := by have := k.isLt; nlinarith
    show row r * p.lda + (s * d.BK + k) < p.a.len
    rw [hal]
    calc row r * p.lda + (s * d.BK + k) < row r * p.lda + p.lda := by omega
      _ = (row r + 1) * p.lda := by ring
      _ ≤ p.M * p.lda := Nat.mul_le_mul_right _ hr
  have hRA : p.a.elems (Buf.img (s := [d.BM, d.BK])
      (fun i => row i.1 * p.lda + (s * d.BK + i.2.1)) fun _ => true) ⊆ R :=
    (p.a.elems_sub_bytes fun _ hj => by obtain ⟨i, -, rfl⟩ := hj; exact hgA i).trans hc.ra
  refine safe_load_buf p.a (Buf.addrs_of_get_all (s := [d.BM, d.BK]) _ fun ⟨r, k, ()⟩ => hap r k)
    (fun i _ => hgA i) hc.va hRA fun a ha _ => ?_
  -- `b = tl.load(b_ptrs)`
  have hgB : ∀ i : Fin d.BK × Fin d.BN × Unit, (s * d.BK + i.1) * p.ldb + col i.2.1 < p.b.len := by
    rintro ⟨k, c, ⟨⟩⟩
    have hcl := hc.col_lt c
    have hk : s * d.BK + k < p.K := by have := k.isLt; nlinarith
    show (s * d.BK + k) * p.ldb + col c < p.b.len
    rw [hbl]
    calc (s * d.BK + k) * p.ldb + col c < (s * d.BK + k) * p.ldb + p.ldb := by omega
      _ = (s * d.BK + k + 1) * p.ldb := by ring
      _ ≤ p.K * p.ldb := Nat.mul_le_mul_right _ hk
  have hRB : p.b.elems (Buf.img (s := [d.BK, d.BN])
      (fun i => (s * d.BK + i.1) * p.ldb + col i.2.1) fun _ => true) ⊆ R :=
    (p.b.elems_sub_bytes fun _ hj => by obtain ⟨i, -, rfl⟩ := hj; exact hgB i).trans hc.rb
  refine safe_load_buf p.b (Buf.addrs_of_get_all (s := [d.BK, d.BN]) _ fun ⟨k, c, ()⟩ => hbp k c)
    (fun i _ => hgB i) hc.vb hRB fun b hb _ => ?_
  -- `accumulator += tl.dot(a, b)`
  refine safe_dot_add (hdot _ _ _) fun acc' hr => .ret ⟨rfl, fun r k => ?_, fun k c => ?_, fun r c => ?_⟩
  · -- `a_ptrs += BLOCK_SIZE_K`
    show (ap.get2 r k).offset (IntV.ofNat d.BK : I32).toInt = _
    rw [hap r k, Buf.ptr_offset, IntV.I32.ofNat_nat (by have := hc.BK_le; omega)]
    congr 1; push_cast; ring
  · -- `b_ptrs += BLOCK_SIZE_K * ldb`
    show (bp.get2 k c).offset ((IntV.ofNat d.BK : I32) * ldb).toInt = _
    rw [hbp k c, Buf.ptr_offset, IntV.I32.mul_nat (IntV.I32.ofNat_nat (by have := hc.BK_le; omega)) hldb hBKldb]
    congr 1; push_cast; ring
  · -- the accumulator element `(r, c)`
    obtain ⟨hcf, herr⟩ := hinv r c
    set i := row r
    set j := col c
    have hi := hc.row_lt r
    have hj := hc.col_lt c
    -- the loaded values are the entries of `A` and `B`
    have hka : ∀ k : Fin d.BK, s * d.BK + k < p.K := fun k => by have := k.isLt; nlinarith
    have hva : ∀ k : Fin d.BK, DType.val? .f16 (a.get2 r k) = some (p.A m i (s * d.BK + k)) := by
      intro k
      obtain ⟨x, hx, hxf⟩ := hc.fin.1 i (s * d.BK + k) hi (hka k)
      have hrd := ha (r, k, ()) rfl
      have hxa : a.get2 r k = x := by
        have : m.read .f16 (p.a.ptr ((i * p.lda + (s * d.BK + k) : ℕ) : ℤ)).addr = some (a.get2 r k) := hrd
        rw [hx] at this; exact (Option.some.inj this).symm
      unfold Gemm.A
      rw [val16_of_read hx, hxa]
      show (if x.isFinite = true then some x.toRat else none) = _
      simp [hxf]
    have hvb : ∀ k : Fin d.BK, DType.val? .f16 (b.get2 k c) = some (p.B m (s * d.BK + k) j) := by
      intro k
      obtain ⟨x, hx, hxf⟩ := hc.fin.2 (s * d.BK + k) j (hka k) hj
      have hrd := hb (k, c, ()) rfl
      have hxb : b.get2 k c = x := by
        have : m.read .f16 (p.b.ptr (((s * d.BK + k) * p.ldb + j : ℕ) : ℤ)).addr = some (b.get2 k c) := hrd
        rw [hx] at this; exact (Option.some.inj this).symm
      unfold Gemm.B
      rw [val16_of_read hx, hxb]
      show (if x.isFinite = true then some x.toRat else none) = _
      simp [hxf]
    set va : Fin d.BK → ℚ := fun k => p.A m i (s * d.BK + k)
    set vb : Fin d.BK → ℚ := fun k => p.B m (s * d.BK + k) j
    -- the sums over the block are the next `BK` terms of the partial sums
    have hsum : p.S m i j (s * d.BK) + ∑ k, va k * vb k = p.S m i j ((s + 1) * d.BK) := by
      unfold Gemm.S
      rw [show (s + 1) * d.BK = s * d.BK + d.BK by ring, Finset.sum_range_add]
      congr 1
      exact Fin.sum_univ_eq_sum_range (fun k => p.A m i (s * d.BK + k) * p.B m (s * d.BK + k) j) d.BK
    have hsabs : p.Sabs m i j (s * d.BK) + ∑ k, |va k * vb k| = p.Sabs m i j ((s + 1) * d.BK) := by
      unfold Gemm.Sabs
      rw [show (s + 1) * d.BK = s * d.BK + d.BK by ring, Finset.sum_range_add]
      congr 1
      exact Fin.sum_univ_eq_sum_range (fun k => |p.A m i (s * d.BK + k) * p.B m (s * d.BK + k) j|) d.BK
    -- room in `float32`
    have hρ := ρ_nonneg d
    have hα := α_nonneg d
    have hsρ : (s : ℚ) * ρ d ≤ 1 / 2 := by
      have : (s : ℚ) ≤ (p.K / d.BK : Nat) := by exact_mod_cast hs.le
      exact (mul_le_mul_of_nonneg_right this hρ).trans hc.T_rho
    have hB0 := p.Sabs_nonneg m i j (s * d.BK)
    have hA0 : 0 ≤ ∑ k, |va k * vb k| := Finset.sum_nonneg fun _ _ => abs_nonneg _
    have hbnd := DotLoop.bound_le hρ hα hsρ hB0
    have hcabs : |(acc.get2 r c).toRat| ≤ p.Sabs m i j (s * d.BK) +
        2 * s * (ρ d * p.Sabs m i j (s * d.BK) + α d) := by
      have := abs_sub_abs_le_abs_sub (acc.get2 r c).toRat (p.S m i j (s * d.BK))
      have := p.abs_S_le m i j (s * d.BK)
      linarith
    have hKs := p.Sabs_le hc.fin hi hj hsK
    have hK31 := hc.K_lt
    have hαs : 2 * (s : ℚ) * α d ≤ 1 := by
      have h1 := α_le d hc.BK_le
      have h2 : (s : ℚ) ≤ 2 ^ 31 := by
        have : s ≤ p.K := (Nat.le_mul_of_pos_right s hBK).trans (by nlinarith)
        exact_mod_cast (this.trans hK31.le)
      have h3 : (2 : ℚ) ^ (-100 : ℤ) = 1 / 2 ^ 100 := by norm_num
      rw [h3] at h1
      have h4 : 2 * (s : ℚ) * α d ≤ 2 * 2 ^ 31 * (1 / 2 ^ 100) :=
        mul_le_mul (by linarith) h1 hα (by positivity)
      linarith [show (2 : ℚ) * 2 ^ 31 * (1 / 2 ^ 100) ≤ 1 by norm_num]
    have hroom := room hc.BK_le (abs_nonneg (acc.get2 r c).toRat) hA0 (by
      have hK : ((s + 1) * d.BK : ℕ) ≤ (2 ^ 31 : ℚ) := by exact_mod_cast (hsK.trans hK31.le)
      have : ((s + 1) * d.BK : ℕ) * (65504 : ℚ) ^ 2 ≤ 2 ^ 31 * 65504 ^ 2 :=
        mul_le_mul_of_nonneg_right hK (by positivity)
      have h65 : (2 : ℚ) ^ 31 * 65504 ^ 2 ≤ 2 ^ 64 := by norm_num
      have hρs : 2 * (s : ℚ) * (ρ d * p.Sabs m i j (s * d.BK)) ≤ p.Sabs m i j (s * d.BK) := by
        have := mul_le_mul_of_nonneg_right hsρ hB0
        nlinarith
      nlinarith)
    obtain ⟨hfin, hstep⟩ := DotLoop.dotAdd_step hacc (rel_nonneg _) (abs_nonneg' _) hr r c hva hvb
      (fun _ => ⟨trivial, trivial⟩) hcf hroom
    refine ⟨hfin, ?_⟩
    have := DotLoop.accum_step hρ hα (p.abs_S_le m i j (s * d.BK)) (Finset.abs_sum_le_sum_abs _ _) herr hstep
    rw [hsum, hsabs] at this
    exact this

/-- **The `K` loop**: after `T = K / BK` steps from zero and the tile's first pointers, every
accumulator element is finite and within `errC K S` of the exact sum over `K`. -/
theorem safe_kloop {d : Inst} {p : Gemm} {row : Fin d.BM → Nat} {col : Fin d.BN → Nat}
    (hdot : Rules.Num.DotTotal E.P cfg) (hacc : Rules.Num.DotAcc E.P cfg (fun _ => True) rel abs)
    (hc : KCtx d p m row col R) {ldb kt : I32} (hldb : ldb.toInt = (p.ldb : Int))
    (hkt : kt.toInt = ((p.K / d.BK : ℕ) : Int))
    {a0 : Tensor [d.BM, d.BK] (Ptr .f16)} {b0 : Tensor [d.BK, d.BN] (Ptr .f16)}
    (ha0 : ∀ r k, a0.get2 r k = p.a.ptr ((row r * p.lda + k : ℕ) : ℤ))
    (hb0 : ∀ k c, b0.get2 k c = p.b.ptr ((k * p.ldb + col c : ℕ) : ℤ))
    {k : Tensor [d.BM, d.BN] F32 × Tensor [d.BM, d.BK] (Ptr .f16) × Tensor [d.BK, d.BN] (Ptr .f16) → Prog γ}
    (hk : ∀ st, (∀ r c, (st.1.get2 r c).isFinite = true ∧
        |(st.1.get2 r c).toRat - p.S m (row r) (col c) p.K| ≤ errC d p.K (p.Sabs m (row r) (col c) p.K)) →
      Safe E R W Q m (k st)) :
    Safe E R W Q m (tl.forRange (0 : I32) kt 1 (tl.zeros [d.BM, d.BN] .f32, a0, b0)
      (kstep d.BM d.BN d.BK ldb) >>= k) := by
  have hK31 := hc.K_lt
  have hT : p.K / d.BK < 2 ^ 31 := lt_of_le_of_lt (Nat.div_le_self _ _) hK31
  have h1 : (1 : I32).toInt = 1 := IntV.I32.toInt_ofNat (by norm_num)
  have h0 : (0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)
  have hrange : tl.RangeOk (0 : I32) kt 1 := by
    refine ⟨by rw [h1]; norm_num, ?_, ?_, ?_⟩ <;>
      simp only [Format.intRange, ite_true, h0, h1, hkt] <;> omega
  have htrip : tl.tripCount (0 : I32).toInt kt.toInt (1 : I32).toInt = p.K / d.BK := by
    rw [h0, h1, hkt]; unfold tl.tripCount
    rw [sub_zero, add_sub_cancel_right, Int.ediv_one, Int.toNat_natCast]
  have hBK : d.BK ∣ p.K := hc.ok.2.2.2.2.2.2.2.2.1
  refine safe_forRange hrange (fun s st m₁ => m₁ = m ∧ KInv d p m row col s st) ⟨rfl, ?_, ?_, ?_⟩
    (fun s hs st m₁ ⟨hm, hI⟩ => ?_) (fun st m₁ ⟨hm, hI⟩ => ?_)
  · intro r kk; rw [ha0 r kk]; simp
  · intro kk c; rw [hb0 kk c]; simp
  · intro r c
    have hz : (tl.zeros [d.BM, d.BN] .f32 : Tensor [d.BM, d.BN] F32).get2 r c = FloatV.ofPat 0 := rfl
    refine ⟨?_, ?_⟩
    · rw [hz]
      have := DotLoop.zero_val
      exact (DotLoop.val?_f32.mp this).1
    · rw [hz, (DotLoop.val?_f32.mp DotLoop.zero_val).2]
      simp [Gemm.S, Gemm.Sabs, DotLoop.bound]
  · subst hm
    rw [htrip] at hs
    exact safe_kstep hdot hacc hc hldb hs _ hI
  · subst hm
    rw [htrip] at hI
    obtain ⟨-, -, hv⟩ := hI
    refine hk st fun r c => ?_
    rw [Nat.div_mul_cancel hBK] at hv
    exact hv r c

end Loop

end TVAExamples.GroupedGemm
