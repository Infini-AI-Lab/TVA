import Proof.Defs

/-!
# The launch around the programs

The grid is `(G, Z·H, 1)`: program `i` handles tile `i % G` of head `i / G`. This file shows the
launch is valid, the programs' footprints are disjoint, and the programs' results make up `Post`.
-/

namespace Claims.AttnFwd

open TVA

variable {d : Inst}

/-- For valid instances, the number of row groups `G` is `N_CTX / 128`. -/
theorem G_eq (hd : d.ok) : G d = d.N_CTX / 128 := by
  unfold G; have := hd.n_mul.2; omega

/-- For valid instances, `G` is positive. -/
theorem G_pos (hd : d.ok) : 0 < G d := by
  have := hd.n_mul.1; have := hd.n_mul.2; unfold G; omega

/-- For valid instances, `G * 128` equals `N_CTX`. -/
theorem G_mul (hd : d.ok) : G d * 128 = d.N_CTX := by
  have := hd.n_mul.2; unfold G; omega

/-- The launch has `G * (Z * H)` programs. -/
theorem launch_size (d : Inst) : (launch d).size = G d * (d.Z * d.H) := by
  simp [launch, Launch.size, G]

/-- For valid instances, the program count times 128 equals the total number of rows. -/
theorem rows_eq (hd : d.ok) : G d * (d.Z * d.H) * 128 = rows d := by
  have hG := G_mul hd
  unfold rows; rw [mul_right_comm, hG]; ring

/-- For valid instances, the launch is valid. -/
theorem launch_valid (hd : d.ok) : (launch d).Valid := by
  have hrows := rows_eq hd
  have hzh := hd.zh_pos
  have hzh' : 1 ≤ d.Z * d.H := Nat.one_le_iff_ne_zero.mpr (Nat.mul_ne_zero hzh.1.ne' hzh.2.ne')
  have hGzh : G d ≤ G d * (d.Z * d.H) := Nat.le_mul_of_pos_right _ hzh'
  have hlt := hd.rows_lt
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, ?_⟩
  · show G d < 2 ^ 31
    omega
  · show d.Z * d.H < 2 ^ 16
    exact hd.zh_lt
  · show (1 : Nat) < 2 ^ 16
    norm_num
  · rw [launch_size]
    omega
  · intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    have h2 : ∀ k, (k = 1 ∨ k = 64 ∨ k = 128 ∨ k = 256) → ∃ e, k = 2 ^ e := by
      rintro k (rfl | rfl | rfl | rfl)
      exacts [⟨0, by norm_num⟩, ⟨6, by norm_num⟩, ⟨7, by norm_num⟩, ⟨8, by norm_num⟩]
    have hD : d.HEAD_DIM = 1 ∨ d.HEAD_DIM = 64 ∨ d.HEAD_DIM = 128 ∨ d.HEAD_DIM = 256 := by
      rcases hd.head_dim with h | h | h <;> simp [h]
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    all_goals
      refine ⟨?_, ?_⟩
      · intro k hk
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
        rcases hk with rfl | rfl | rfl <;> first
          | exact h2 _ (by simp) | exact h2 _ hD
      · rcases hd.head_dim with h | h | h <;> simp [h]

/-! ## Tiles -/

theorem sm_row_lt (hd : d.ok) {sm r : Nat} (hsm : sm < G d) (hr : r < 128) : sm * 128 + r < d.N_CTX := by
  have := G_mul hd; nlinarith

/-- A row index within a head lies below the total row count. -/
theorem row_bound (_hd : d.ok) {bh n : Nat} (hbh : bh < d.Z * d.H) (hn : n < d.N_CTX) :
    bh * d.N_CTX + n < rows d := by
  unfold rows; nlinarith

/-- The rows of a head end within the total row count. -/
theorem row_bound' (_hd : d.ok) {bh : Nat} (hbh : bh < d.Z * d.H) : bh * d.N_CTX + d.N_CTX ≤ rows d := by
  unfold rows
  calc bh * d.N_CTX + d.N_CTX = (bh + 1) * d.N_CTX := by ring
    _ ≤ d.Z * d.H * d.N_CTX := Nat.mul_le_mul_right _ hbh

/-- A flattened row-and-column index lies below the buffer size. -/
theorem idx_lt (hd : d.ok) {bh n e : Nat} (hbh : bh < d.Z * d.H) (hn : n < d.N_CTX)
    (he : e < d.HEAD_DIM) : idx d bh n e < rows d * d.HEAD_DIM := by
  unfold idx
  have := row_bound hd hbh hn
  nlinarith

/-- Distinct (head, row) pairs give distinct flattened rows. -/
theorem row_inj (hd : d.ok) {bh bh' n n' : Nat} (hn : n < d.N_CTX) (hn' : n' < d.N_CTX)
    (h : bh * d.N_CTX + n = bh' * d.N_CTX + n') : bh = bh' ∧ n = n' := by
  have hN := hd.n_mul.1
  have h1 : (bh * d.N_CTX + n) / d.N_CTX = bh := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hn, zero_add]
  have h2 : (bh' * d.N_CTX + n') / d.N_CTX = bh' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hN, Nat.div_eq_of_lt hn', zero_add]
  have hb : bh = bh' := by rw [← h1, ← h2, h]
  subst hb
  exact ⟨rfl, by omega⟩

/-- Distinct (head, row, column) triples give distinct flattened indices. -/
theorem idx_inj (hd : d.ok) {bh bh' n n' e e' : Nat} (hn : n < d.N_CTX) (hn' : n' < d.N_CTX)
    (he : e < d.HEAD_DIM) (he' : e' < d.HEAD_DIM) (h : idx d bh n e = idx d bh' n' e') :
    bh = bh' ∧ n = n' ∧ e = e' := by
  unfold idx at h
  have hD : 0 < d.HEAD_DIM := by omega
  have h1 : ((bh * d.N_CTX + n) * d.HEAD_DIM + e) / d.HEAD_DIM = bh * d.N_CTX + n := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt he, zero_add]
  have h2 : ((bh' * d.N_CTX + n') * d.HEAD_DIM + e') / d.HEAD_DIM = bh' * d.N_CTX + n' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt he', zero_add]
  have hrow : bh * d.N_CTX + n = bh' * d.N_CTX + n' := by rw [← h1, ← h2, h]
  obtain ⟨hb, hnn⟩ := row_inj hd hn hn' hrow
  subst hb; subst hnn
  exact ⟨rfl, rfl, by omega⟩

/-- The output tile of a program lies inside the output buffer. -/
theorem oTile_sub (hd : d.ok) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d) :
    oTile d bh sm ⊆ below (rows d * d.HEAD_DIM) := by
  rintro x ⟨r, hr, e, he, rfl⟩
  exact idx_lt hd hbh (sm_row_lt hd hsm hr) he

/-- The statistics tile of a program lies inside its buffer. -/
theorem mTile_sub (hd : d.ok) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d) :
    mTile d bh sm ⊆ below (rows d) := by
  rintro x ⟨r, hr, rfl⟩
  have := row_bound hd hbh (sm_row_lt hd hsm hr)
  show _ < rows d
  omega

/-- Write sets of different programs in the same head are disjoint. -/
theorem Wset_disjoint_R (hd : d.ok) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d) :
    Disjoint (Wset d bh sm) (Rset d) := by
  obtain ⟨oq, ok', ov, oM, Mq, Mk, Mv⟩ := hd.disj
  have ho := d.o.elems_mono (oTile_sub hd hbh hsm)
  have hm := d.M.elems_mono (mTile_sub hd hbh hsm)
  unfold Wset Rset
  refine Disjoint.union_left ?_ ?_
  · exact Disjoint.union_right (Disjoint.union_right (oq.mono ho le_rfl) (ok'.mono ho le_rfl))
      (ov.mono ho le_rfl)
  · exact Disjoint.union_right (Disjoint.union_right (Mq.mono hm le_rfl) (Mk.mono hm le_rfl))
      (Mv.mono hm le_rfl)

/-- Write sets of different programs are disjoint. -/
theorem Wset_disjoint (hd : d.ok) {bh sm bh' sm' : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d)
    (hbh' : bh' < d.Z * d.H) (hsm' : sm' < G d) (hne : ¬ (bh = bh' ∧ sm = sm')) :
    Disjoint (Wset d bh sm) (Wset d bh' sm') := by
  have ho := hd.disj.2.2.2.1
  have hoo : Disjoint (oTile d bh sm) (oTile d bh' sm') := by
    rw [Set.disjoint_left]
    rintro x ⟨r, hr, e, he, rfl⟩ ⟨r', hr', e', he', h⟩
    obtain ⟨h1, h2, -⟩ := idx_inj hd (sm_row_lt hd hsm hr) (sm_row_lt hd hsm' hr') he he' h
    exact hne ⟨h1, by omega⟩
  have hmm : Disjoint (mTile d bh sm) (mTile d bh' sm') := by
    rw [Set.disjoint_left]
    rintro x ⟨r, hr, rfl⟩ ⟨r', hr', h⟩
    obtain ⟨h1, h2⟩ := row_inj hd (sm_row_lt hd hsm hr) (sm_row_lt hd hsm' hr') h
    exact hne ⟨h1, by omega⟩
  unfold Wset
  refine Disjoint.union_left (Disjoint.union_right ?_ ?_) (Disjoint.union_right ?_ ?_)
  · exact d.o.elems_disjoint hoo
  · exact ho.mono (d.o.elems_mono (oTile_sub hd hbh hsm)) (d.M.elems_mono (mTile_sub hd hbh' hsm'))
  · exact (ho.mono (d.o.elems_mono (oTile_sub hd hbh' hsm')) (d.M.elems_mono (mTile_sub hd hbh hsm))).symm
  · exact d.M.elems_disjoint hmm
