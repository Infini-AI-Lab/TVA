import Proof.Geom

/-!
# Arithmetic of the instance

Small facts about `Inst.ok` that the walk repeatedly needs: positivity of the sizes, the group of a
head, and the range of the logical chunk's rows and of a tile's rows and columns.
-/

namespace Claims.SsdChunkScan

open TVA

/-- `chunk_size` is positive. -/
theorem cs_pos {d : Inst} (hd : d.ok) : 0 < d.cs := by
  obtain ⟨a, ha⟩ := hd.cs_pow2
  rw [ha]; exact Nat.two_pow_pos a

/-- `nheads // ngroups` is positive. -/
theorem ratio_pos {d : Inst} (hd : d.ok) : 0 < ratio d := by
  obtain ⟨m, hm⟩ := hd.G_dvd
  have : 0 < d.H := hd.H_pos
  unfold ratio
  have hG := hd.G_pos
  rw [hm, Nat.mul_div_cancel_left _ hG]
  rcases Nat.eq_zero_or_pos m with h | h
  · rw [h] at hm; omega
  · exact h

/-- The group of a head is below `ngroups`. -/
theorem grp_lt {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.H) : grp d h < d.G := by
  obtain ⟨m, hm⟩ := hd.G_dvd
  have hG := hd.G_pos
  have hr : ratio d = m := by unfold ratio; rw [hm, Nat.mul_div_cancel_left _ hG]
  unfold grp; rw [hr]
  have hm0 : 0 < m := by have := hd.H_pos; rcases Nat.eq_zero_or_pos m with h | h <;> [(rw [h] at hm; omega); exact h]
  rw [Nat.div_lt_iff_lt_mul hm0]
  rw [hm] at hh
  linarith [Nat.mul_comm d.G m]

/-- The head's group times `x` is inside `G·x`. -/
theorem grp_mul_le {d : Inst} (hd : d.ok) {h : Nat} (hh : h < d.H) (x : Nat) :
    (grp d h + 1) * x ≤ d.G * x := Nat.mul_le_mul_right _ (grp_lt hd hh)

/-- A tile row is below `2·cs + 64`. -/
theorem rowOf_lt {d : Inst} (hd : d.ok) {k pm ii : Nat} (hk : k < d.nmeta) (hpm : pm < nmt d)
    (hii : ii < BM) : rowOf d k pm ii < 2 * d.cs + 64 := by
  have h1 := lo_lt hd hk
  have h2 : BM * pm < d.cs := by
    unfold nmt at hpm
    have := cs_pos hd
    unfold BM at *
    omega
  unfold rowOf; unfold BM at *; omega

/-- A buffer disjoint from `out` is inside the read footprint. -/
theorem bytes_sub_R {d : Inst} {α : DType} {b : Buf α} (h : Disjoint d.out.bytes b.bytes) :
    b.bytes ⊆ R d := fun _ hx hy => Set.disjoint_left.mp h hy hx

/-- The state buffer the kernel reads is one of the inputs: valid, disjoint from `out`, and long
enough for the states of a head. -/
theorem stBuf_valid {d : Inst} {init : Mem} (hpre : Pre d init) (u : Bool) :
    (stBuf d u).Valid init := by
  obtain ⟨-, -, -, -, -, -, hst, -, hin, -⟩ := hpre
  cases u <;> simpa [stBuf]

/-- The state buffer the kernel reads is disjoint from `out`. -/
theorem stBuf_disj {d : Inst} (hd : d.ok) (u : Bool) : Disjoint d.out.bytes (stBuf d u).bytes := by
  cases u
  · exact hd.disj.2.2.2.2.2.2.1
  · exact hd.disj.2.2.2.2.2.2.2.2.1

/-- The last state element a tile of head `h` reads is inside the buffer. -/
theorem st_len {d : Inst} (hd : d.ok) {u : Bool} {c r h : Nat} (hh : h < d.H)
    (hc : u = false → c < nc d) (hur : u = true → d.hasInit = true ∧ r < d.nreq) :
    stBase d u c r h + (d.N - 1) * 1 + (d.P - 1) * d.N < (stBuf d u).len := by
  have hN := hd.N_pos
  have hP := hd.P_pos
  have hPN : d.N - 1 + (d.P - 1) * d.N < PN d := by
    unfold PN
    have : (d.P - 1) * d.N + d.N = d.P * d.N := by
      rw [← Nat.succ_mul]; congr 1; omega
    rw [Nat.mul_comm d.P d.N] at *; rw [Nat.mul_comm] at this ⊢; omega
  have e2 : (h + 1) * PN d ≤ d.H * PN d := Nat.mul_le_mul_right _ hh
  have e2' : (h + 1) * PN d = h * PN d + PN d := by rw [Nat.add_mul, Nat.one_mul]
  cases u
  · have e1 : (c + 1) * (d.H * PN d) ≤ nc d * (d.H * PN d) := Nat.mul_le_mul_right _ (hc rfl)
    have e1' : (c + 1) * (d.H * PN d) = c * (d.H * PN d) + d.H * PN d := by
      rw [Nat.add_mul, Nat.one_mul]
    have := hd.states_len
    simp only [stBase, stBuf, Bool.false_eq_true, ite_false] at *
    omega
  · obtain ⟨hi, hr⟩ := hur rfl
    have e1 : (r + 1) * (d.H * PN d) ≤ d.nreq * (d.H * PN d) := Nat.mul_le_mul_right _ hr
    have e1' : (r + 1) * (d.H * PN d) = r * (d.H * PN d) + d.H * PN d := by
      rw [Nat.add_mul, Nat.one_mul]
    have := hd.init_len hi
    simp only [stBase, stBuf, ite_true] at *
    omega

/-- A packed batch has a token. -/
theorem T_pos {d : Inst} (hd : d.ok) : 0 < d.T := by
  have key : ∀ n, 1 ≤ n → n ≤ d.nreq → d.qsl 1 ≤ d.qsl n := by
    intro n hn
    induction n, hn using Nat.le_induction with
    | base => intro _; exact le_rfl
    | succ n hn ih =>
      intro h
      exact (ih (by omega)).trans (hd.qsl_lt n (by omega)).le
  have h1 : d.qsl 0 < d.qsl 1 := hd.qsl_lt 0 hd.nreq_pos
  have h2 := key d.nreq hd.nreq_pos le_rfl
  have := hd.qsl_end
  have := hd.qsl_zero
  omega

/-- There is a chunk. -/
theorem nc_pos {d : Inst} (hd : d.ok) : 0 < nc d := by
  have := T_pos hd
  have := cs_pos hd
  unfold nc
  exact Nat.div_pos (by omega) (by omega)

/-- A `cs × cs` block of `cb` of chunk `lc k` and the group of head `h` is inside `cb`. -/
theorem cb_tile_len {d : Inst} (hd : d.ok) {k h : Nat} (hk : k < d.nmeta) (hh : h < d.H) :
    cbIdx d (lc d k) (grp d h) 0 0 + (d.cs - 1) * d.cs + (d.cs - 1) * 1 < d.cb.len := by
  have hc := lc_lt hd hk
  have hg := grp_lt hd hh
  have hcs := cs_pos hd
  have e1 : (lc d k + 1) * (d.G * (d.cs * d.cs)) ≤ nc d * (d.G * (d.cs * d.cs)) :=
    Nat.mul_le_mul_right _ hc
  have e1' : (lc d k + 1) * (d.G * (d.cs * d.cs)) =
      lc d k * (d.G * (d.cs * d.cs)) + d.G * (d.cs * d.cs) := by rw [Nat.add_mul, Nat.one_mul]
  have e2 : (grp d h + 1) * (d.cs * d.cs) ≤ d.G * (d.cs * d.cs) := Nat.mul_le_mul_right _ hg
  have e2' : (grp d h + 1) * (d.cs * d.cs) = grp d h * (d.cs * d.cs) + d.cs * d.cs := by
    rw [Nat.add_mul, Nat.one_mul]
  have e3 : (d.cs - 1) * d.cs + (d.cs - 1) * 1 < d.cs * d.cs := by
    have : (d.cs - 1) * d.cs + d.cs = d.cs * d.cs := by
      rw [← Nat.succ_mul]; congr 1; omega
    omega
  have := hd.cb_len
  unfold cbIdx
  omega

/-- Element `u < cs` of the `dt`/`dA_cumsum` row of chunk `lc k` and head `h` is inside the buffer. -/
theorem dtIdx_lt {d : Inst} (hd : d.ok) {k h u : Nat} (hk : k < d.nmeta) (hh : h < d.H)
    (hu : u < d.cs) : dtIdx d h (lc d k) u < d.H * (nc d * d.cs) := by
  have hc := lc_lt hd hk
  have e1 : (lc d k + 1) * d.cs ≤ nc d * d.cs := Nat.mul_le_mul_right _ hc
  have e1' : (lc d k + 1) * d.cs = lc d k * d.cs + d.cs := by rw [Nat.add_mul, Nat.one_mul]
  have e2 : (h + 1) * (nc d * d.cs) ≤ d.H * (nc d * d.cs) := Nat.mul_le_mul_right _ hh
  have e2' : (h + 1) * (nc d * d.cs) = h * (nc d * d.cs) + nc d * d.cs := by
    rw [Nat.add_mul, Nat.one_mul]
  unfold dtIdx
  omega

/-- The `x` tile of rows `< lend` and channels `< P` of head `h` in chunk `lc k` is inside `x`. -/
theorem x_tile_len {d : Inst} (hd : d.ok) {k h : Nat} (hk : k < d.nmeta) (hh : h < d.H)
    (hl : 0 < lend d k) :
    xIdx d (lc d k * d.cs) h 0 + (lend d k - 1) * d.xT + (d.P - 1) * 1 < d.x.len := by
  have hlend := lend_le hd hk
  have e1 : (lc d k * d.cs + (lend d k - 1)) * d.xT ≤ (d.T - 1) * d.xT :=
    Nat.mul_le_mul_right _ (by omega)
  have e1' : (lc d k * d.cs + (lend d k - 1)) * d.xT =
      lc d k * d.cs * d.xT + (lend d k - 1) * d.xT := Nat.add_mul _ _ _
  have e2 : (h + 1) * d.P ≤ d.H * d.P := Nat.mul_le_mul_right _ hh
  have e2' : (h + 1) * d.P = h * d.P + d.P := by rw [Nat.add_mul, Nat.one_mul]
  have := hd.x_len
  have := hd.P_pos
  unfold xIdx
  omega

/-- `headdim` fits `int32`. -/
theorem P_lt31 {d : Inst} (hd : d.ok) : d.P < 2 ^ 31 := by
  have h1 := hd.x_off
  have h2 := hd.xT_ge
  have h3 : d.P ≤ d.H * d.P := Nat.le_mul_of_pos_left _ hd.H_pos
  have h4 : d.xT ≤ (2 * d.cs + 64) * d.xT := Nat.le_mul_of_pos_left _ (by omega)
  omega

/-- A token's logical chunk starts at its chunk's first row plus `lOff`. -/
theorem lTok_eq {d : Inst} (t : Nat) : lTok d t = chk d t * d.cs + lOff d t := by
  unfold lOff
  have : chk d t * d.cs ≤ lTok d t := by unfold lTok; exact le_max_left _ _
  omega

/-- A valid tile row's logical chunk starts at or after row `lo k` of the chunk (at it, with initial
states; without, `lo k = 0`). -/
theorem lTok_tile {d : Inst} (hd : d.ok) {k pm ii : Nat} (hk : k < d.nmeta)
    (hr : RowOk d k pm ii) : lc d k * d.cs + lo d k ≤ lTok d (tokOf d k pm ii) := by
  have hrow : lo d k ≤ rowOf d k pm ii ∧ rowOf d k pm ii < lend d k := ⟨by unfold rowOf; omega, hr⟩
  obtain ⟨-, hchk, -, hlOff⟩ := tok_of_row hd hk hrow
  have h1 : chk d (tokOf d k pm ii) = lc d k := hchk
  have h0 := lTok_eq (d := d) (tokOf d k pm ii)
  rw [h1] at h0
  by_cases hI : d.hasInit = true
  · have h2 : lOff d (tokOf d k pm ii) = lo d k := hlOff hI
    omega
  · have : d.hasInit = false := by simpa using hI
    have := lo_noinit this k
    omega

/-- `headdim` leaves room for the `int32` tile offsets: `headdim + 256 < 2^31`. -/
theorem P_small {d : Inst} (hd : d.ok) : d.P + 256 < 2 ^ 31 := by
  have h1 := hd.x_off
  have h2 := hd.xT_ge
  have h3 : d.P ≤ d.H * d.P := Nat.le_mul_of_pos_left _ hd.H_pos
  have h4 : 66 * d.xT ≤ (2 * d.cs + 64) * d.xT :=
    Nat.mul_le_mul_right _ (by have := cs_pos hd; omega)
  omega

/-- Without initial states, chunk `k < nmeta = nc` starts below `seqlen`. -/
theorem noinit_start_lt {d : Inst} (hd : d.ok) (hN : d.hasInit = false) {k : Nat}
    (hk : k < d.nmeta) : k * d.cs < d.T := by
  have h1 := lc_lt hd hk
  rw [lc_noinit hN] at h1
  unfold nc at h1
  have hcs := cs_pos hd
  have h2 : (k + 1) * d.cs ≤ d.T + d.cs - 1 := (Nat.le_div_iff_mul_le hcs).mp h1
  rw [Nat.succ_mul] at h2
  omega

/-- A logical chunk's physical chunk starts below `seqlen`. -/
theorem lc_start_lt {d : Inst} (hd : d.ok) {k : Nat} (hk : k < d.nmeta) : lc d k * d.cs < d.T := by
  have h1 := lc_lt hd hk
  unfold nc at h1
  have hcs := cs_pos hd
  have h2 : (lc d k + 1) * d.cs ≤ d.T + d.cs - 1 := (Nat.le_div_iff_mul_le hcs).mp h1
  rw [Nat.succ_mul] at h2
  omega

/-- The request before a chunk is an `int32` value. -/
theorem prevReq_lt {d : Inst} (hd : d.ok) (c : Nat) : prevReq d c < 2 ^ 31 := by
  unfold prevReq; split_ifs
  · exact reqOf_lt hd _
  · norm_num

/-- The grid's `y` is positive. -/
theorem nmeta_pos {d : Inst} (hd : d.ok) : 0 < d.nmeta := by
  have h := hd.nmeta_eq
  have hn := nc_pos hd
  split_ifs at h with hi
  · rw [h]; unfold hostN; unfold nc at hn; omega
  · rw [h]; exact hn

end Claims.SsdChunkScan
