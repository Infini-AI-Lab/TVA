import Proof.Lanes

namespace Claims.LayerNormFwd

open TVA


/-! The setup of the proof: the footprints of program `p` (`R`, `W`, `yRows`, `mRows`, ...), the
post-condition `Q` of one program, and the bounds and disjointness facts about the buffers' elements
that the walk of the kernel and the composition of the programs need. -/

/-! ## Element bounds -/

/-- The flat index `t·D + j` of an element of `x`, `y` is below `T·D`. -/
theorem idx_lt_TD {d : Inst} (_hd : d.ok) {t j : Nat} (ht : t < d.T) (hj : j < d.D) :
    idx d t j < d.T * d.D := by
  have h1 : t + 1 ≤ d.T := ht
  have h3 : idx d t j < t * d.D + d.D := by
    show t * d.D + j < t * d.D + d.D
    omega
  have h4 : t * d.D + d.D = (t + 1) * d.D := by ring
  have h5 : (t + 1) * d.D ≤ d.T * d.D := Nat.mul_le_mul_right _ h1
  omega

/-- The flat index of an element of `weight`, `bias` is below `G·D`. -/
theorem gidx_lt_GD {d : Inst} (hd : d.ok) {t j : Nat} (hj : j < d.D) :
    gidx d t j < d.G * d.D := by
  have hG : 0 < d.G := hd.g_pos
  have hm : t % d.G < d.G := Nat.mod_lt _ hG
  have h3 : gidx d t j < (t % d.G) * d.D + d.D := by
    show (t % d.G) * d.D + j < (t % d.G) * d.D + d.D
    omega
  have h4 : (t % d.G) * d.D + d.D = (t % d.G + 1) * d.D := by ring
  have h5 : (t % d.G + 1) * d.D ≤ d.G * d.D := Nat.mul_le_mul_right _ hm
  omega

/-- Element `(t, j)` of a `T * D` layout lies inside any buffer of length at least `T * D`. -/
theorem idx_lt_buf {d : Inst} (hd : d.ok) {t j : Nat} (ht : t < d.T) (hj : j < d.D)
    {dy : DType} {B : Buf dy} (hlen : d.T * d.D ≤ B.len) : idx d t j < B.len :=
  lt_of_lt_of_le (idx_lt_TD hd ht hj) hlen

/-- Element `(t, j)` of a `G * D` layout lies inside any buffer of length at least `G * D`. -/
theorem gidx_lt_buf {d : Inst} (hd : d.ok) {t j : Nat} (hj : j < d.D)
    {dy : DType} {B : Buf dy} (hlen : d.G * d.D ≤ B.len) : gidx d t j < B.len :=
  lt_of_lt_of_le (gidx_lt_GD hd hj) hlen

/-- Integer division by 32 of `32 * p + r` with `r < 32` is `p`. -/
theorem div32_add {p r : Nat} (hr : r < 32) : (32 * p + r) / 32 = p := by
  rw [Nat.mul_add_div (by norm_num : 0 < 32), Nat.div_eq_of_lt hr, Nat.add_zero]

/-! ## Accessors stated with `get1`/`get2` -/

theorem get1_eq_get {n : Nat} {α : Type} (x : Tensor [n] α) (i : Fin n) :
    x.get1 i = x.get (i, ()) := rfl

/-! ## `Buf.Addrs` for the computed pointer blocks -/

/-- The `[T, D]` block pointer of a `[32, BD]` access (`x`, `res`, `y`, `res_out`). -/
theorem mat_addrs {d : Inst} (hd : d.ok) (p : Nat) {dy : DType} (B : Buf dy)
    {ot : Tensor [32] I64} {od : Tensor [d.BD] I32}
    {mt : Tensor [32] Bool} {md : Tensor [d.BD] Bool}
    (hp : p < 2 ^ 31)
    (hot : ∀ r : Fin 32, (ot.get1 r).toInt = 32 * (p : Int) + (r : Int))
    (hod : ∀ j : Fin d.BD, (od.get1 j).toInt = (j : Int))
    (hmt : ∀ r : Fin 32, mt.get1 r = decide (32 * p + (r : Nat) < d.T))
    (hmd : ∀ j : Fin d.BD, md.get1 j = decide ((j : Nat) < d.D)) :
    B.Addrs (tl.addrs ((B.ptr 0 + ot.col d.BD * (IntV.ofNat d.D : I64)) + od.row 32)
        (some ((mt.col d.BD).and (md.row 32))))
      (fun i : Idx [32, d.BD] => idx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat))
      (fun i : Idx [32, d.BD] => decide (32 * p + (i.1 : Nat) < d.T ∧ (i.2.1 : Nat) < d.D)) := by
  refine Buf.addrs_of_get B (fun i => ?_) (fun i _ => ?_)
  · obtain ⟨r, j, u⟩ := i
    cases u
    show ((mt.col d.BD).and (md.row 32)).get2 r j = decide (32 * p + (r : Nat) < d.T ∧ (j : Nat) < d.D)
    rw [mX_get r j (hmt r) (hmd j)]
  · obtain ⟨r, j, u⟩ := i
    cases u
    show (((B.ptr 0 + ot.col d.BD * (IntV.ofNat d.D : I64)) + od.row 32)).get2 r j
        = B.ptr (idx d (32 * p + (r : Nat)) (j : Nat))
    rw [pmat_get hd B p hp hot hod r j]

/-- The `[G·D]` block pointer of a `[32, BD]` access (`w`, `b`), masked only by `m_d`. -/
theorem vec_addrs {d : Inst} (hd : d.ok) (p : Nat) {dy : DType} (B : Buf dy)
    {og : Tensor [32] I64} {od : Tensor [d.BD] I32} {md : Tensor [d.BD] Bool}
    (hp : p < 2 ^ 31)
    (hog : ∀ r : Fin 32, (og.get1 r).toInt = (((32 * p + r) % d.G : Nat) : Int))
    (hod : ∀ j : Fin d.BD, (od.get1 j).toInt = (j : Int))
    (hmd : ∀ j : Fin d.BD, md.get1 j = decide ((j : Nat) < d.D)) :
    B.Addrs (tl.addrs ((B.ptr 0 + og.col d.BD * (IntV.ofNat d.D : I64)) + od.row 32)
        (some (md.row 32)))
      (fun i : Idx [32, d.BD] => gidx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat))
      (fun i : Idx [32, d.BD] => decide ((i.2.1 : Nat) < d.D)) := by
  refine Buf.addrs_of_get B (fun i => ?_) (fun i _ => ?_)
  · obtain ⟨r, j, u⟩ := i
    cases u
    show (md.row 32).get2 r j = decide ((j : Nat) < d.D)
    rw [get2_row, hmd j]
  · obtain ⟨r, j, u⟩ := i
    cases u
    show (((B.ptr 0 + og.col d.BD * (IntV.ofNat d.D : I64)) + od.row 32)).get2 r j
        = B.ptr (gidx d (32 * p + (r : Nat)) (j : Nat))
    rw [pvec_get hd B p hp hog hod r j]

/-- The `[T]` block pointer of a `[32]` access (`mean`, `rstd`). -/
theorem row_addrs {d : Inst} {dy : DType} (B : Buf dy) (p : Nat)
    {ot : Tensor [32] I64} {mt : Tensor [32] Bool}
    (hp : p < 2 ^ 31)
    (hot : ∀ r : Fin 32, (ot.get1 r).toInt = 32 * (p : Int) + (r : Int))
    (hmt : ∀ r : Fin 32, mt.get1 r = decide (32 * p + (r : Nat) < d.T)) :
    B.Addrs (tl.addrs (B.ptr 0 + ot) (some mt))
      (fun i : Idx [32] => 32 * p + (i.1 : Nat))
      (fun i : Idx [32] => decide (32 * p + (i.1 : Nat) < d.T)) := by
  refine Buf.addrs_of_get B (fun i => ?_) (fun i _ => ?_)
  · obtain ⟨r, u⟩ := i
    cases u
    show mt.get1 r = decide (32 * p + (r : Nat) < d.T)
    rw [hmt r]
  · obtain ⟨r, u⟩ := i
    cases u
    show (B.ptr 0 + ot).get1 r = B.ptr (32 * p + (r : Nat))
    rw [prow_get B p hp hot r]

/-! ## Footprints -/

/-- The elements of `x`, `residual`, `y` and `res_out` program `p` touches: row `t` with
`t / 32 = p`, column `j < D`. -/
def yRows (d : Inst) (p : Nat) : Set Nat :=
  {k | ∃ t, t < d.T ∧ t / 32 = p ∧ ∃ j, j < d.D ∧ k = idx d t j}

/-- The rows program `p` owns. -/
def mRows (d : Inst) (p : Nat) : Set Nat := {t | t < d.T ∧ t / 32 = p}

/-- The elements of `residual` and `res_out` the launch touches, restricted to program `p`. -/
def resRows (d : Inst) (p : Nat) : Set Nat := {k | d.hasRes = true ∧ k ∈ yRows d p}

/-- The elements of `weight` and `bias` program `p` loads: row group `(32p + r) % G`, column
`j < D`, for every `r < 32`. -/
def wRows (d : Inst) (p : Nat) : Set Nat :=
  {k | ∃ r, r < 32 ∧ ∃ j, j < d.D ∧ k = gidx d (32 * p + r) j}

/-- The bytes program `p` reads. -/
def R (d : Inst) (p : Nat) : Set Int :=
  d.x.elems (yRows d p) ∪ d.res.elems (resRows d p) ∪ d.w.elems (wRows d p) ∪
    d.b.elems (wRows d p)

/-- The bytes program `p` writes. -/
def W (d : Inst) (p : Nat) : Set Int :=
  d.y.elems (yRows d p) ∪ d.res_out.elems (resRows d p) ∪ d.mean.elems (mRows d p) ∪
    d.rstd.elems (mRows d p)

/-- What program `p` leaves behind: its own rows are done. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ t, t < d.T → t / 32 = p → RowDone d init m t

/-- Program `p`'s `T×D` rows lie in the whole matrix. -/
theorem yRows_subset_matSet (d : Inst) (p : Nat) : yRows d p ⊆ matSet d := by
  rintro k ⟨t, ht, _, j, hj, rfl⟩
  show t * d.D + j < d.T * d.D
  have h1 : t + 1 ≤ d.T := ht
  have h2 : t * d.D + j < t * d.D + d.D := by omega
  have h3 : t * d.D + d.D = (t + 1) * d.D := by ring
  have h4 : (t + 1) * d.D ≤ d.T * d.D := Nat.mul_le_mul_right _ h1
  omega

/-- The mean/rstd rows of program `p` are among all rows. -/
theorem mRows_subset_rowSet (d : Inst) (p : Nat) : mRows d p ⊆ rowSet d := fun _t ht => ht.1

/-- The residual rows of program `p` are among its output rows. -/
theorem resRows_sub_yRows (d : Inst) (p : Nat) : resRows d p ⊆ yRows d p := fun _k hk => hk.2

/-- The residual rows of program `p` are among all residual rows. -/
theorem resRows_subset_resSet (d : Inst) (p : Nat) : resRows d p ⊆ resSet d :=
  fun _k hk => ⟨hk.1, yRows_subset_matSet d p hk.2⟩

/-- Program `p`'s weight rows lie in the whole weight table. -/
theorem wRows_subset_vecSet {d : Inst} (hd : d.ok) (p : Nat) : wRows d p ⊆ vecSet d := by
  rintro k ⟨r, _, j, hj, rfl⟩
  show ((32 * p + r) % d.G) * d.D + j < d.G * d.D
  have hG : 0 < d.G := hd.g_pos
  have hm : (32 * p + r) % d.G < d.G := Nat.mod_lt _ hG
  have h2 : ((32 * p + r) % d.G) * d.D + j < ((32 * p + r) % d.G) * d.D + d.D := by omega
  have h3 : ((32 * p + r) % d.G) * d.D + d.D = ((32 * p + r) % d.G + 1) * d.D := by ring
  have h4 : ((32 * p + r) % d.G + 1) * d.D ≤ d.G * d.D := Nat.mul_le_mul_right _ hm
  omega

/-- Program `p` reads only input bytes. -/
theorem R_subset_inBytes {d : Inst} (hd : d.ok) (p : Nat) : R d p ⊆ inBytes d := by
  refine Set.union_subset (Set.union_subset (Set.union_subset ?_ ?_) ?_) ?_
  · exact (d.x.elems_mono (yRows_subset_matSet d p)).trans
      (Set.subset_union_left.trans (Set.subset_union_left.trans Set.subset_union_left))
  · exact (d.res.elems_mono (resRows_subset_resSet d p)).trans
      (Set.subset_union_right.trans (Set.subset_union_left.trans Set.subset_union_left))
  · exact (d.w.elems_mono (wRows_subset_vecSet hd p)).trans
      (Set.subset_union_right.trans Set.subset_union_left)
  · exact (d.b.elems_mono (wRows_subset_vecSet hd p)).trans Set.subset_union_right

/-- Program `p` writes only output bytes. -/
theorem W_subset_outBytes (d : Inst) (p : Nat) : W d p ⊆ outBytes d := by
  refine Set.union_subset (Set.union_subset (Set.union_subset ?_ ?_) ?_) ?_
  · exact (d.y.elems_mono (yRows_subset_matSet d p)).trans
      (Set.subset_union_left.trans (Set.subset_union_left.trans Set.subset_union_left))
  · exact (d.res_out.elems_mono (resRows_subset_resSet d p)).trans
      (Set.subset_union_right.trans (Set.subset_union_left.trans Set.subset_union_left))
  · exact (d.mean.elems_mono (mRows_subset_rowSet d p)).trans
      (Set.subset_union_right.trans Set.subset_union_left)
  · exact (d.rstd.elems_mono (mRows_subset_rowSet d p)).trans Set.subset_union_right

/-- Element positions of a `[T, D]` tensor are one-to-one. -/
theorem idx_inj {d : Inst} (hD : 0 < d.D) {t j t' j' : Nat} (hj : j < d.D) (hj' : j' < d.D)
    (h : idx d t j = idx d t' j') : t = t' ∧ j = j' := by
  have h1 : (t * d.D + j) / d.D = t := by
    rw [Nat.mul_comm t d.D, Nat.mul_add_div hD, Nat.div_eq_of_lt hj, Nat.add_zero]
  have h2 : (t * d.D + j) % d.D = j := by
    rw [Nat.mul_comm t d.D, Nat.mul_add_mod, Nat.mod_eq_of_lt hj]
  have h1' : (t' * d.D + j') / d.D = t' := by
    rw [Nat.mul_comm t' d.D, Nat.mul_add_div hD, Nat.div_eq_of_lt hj', Nat.add_zero]
  have h2' : (t' * d.D + j') % d.D = j' := by
    rw [Nat.mul_comm t' d.D, Nat.mul_add_mod, Nat.mod_eq_of_lt hj']
  have h' : t * d.D + j = t' * d.D + j' := h
  constructor
  · have hc := congrArg (fun x => x / d.D) h'
    rwa [h1, h1'] at hc
  · have hc := congrArg (fun x => x % d.D) h'
    rwa [h2, h2'] at hc

/-- Different programs have disjoint rows of the matrices. -/
theorem yRows_disjoint {d : Inst} (hD : 0 < d.D) {p q : Nat} (hpq : p ≠ q) :
    Disjoint (yRows d p) (yRows d q) := by
  rw [Set.disjoint_left]
  rintro k ⟨t, ht, htp, j, hj, rfl⟩ ⟨t', ht', htq, j', hj', he⟩
  obtain ⟨rfl, rfl⟩ := idx_inj hD hj hj' he
  exact hpq (htp.symm.trans htq)

/-- Different programs have disjoint rows of `mean`, `rstd`. -/
theorem mRows_disjoint {d : Inst} {p q : Nat} (hpq : p ≠ q) :
    Disjoint (mRows d p) (mRows d q) := by
  rw [Set.disjoint_left]
  rintro t ⟨_, htp⟩ ⟨_, htq⟩
  exact hpq (htp.symm.trans htq)

/-- Disjointness transfers to subsets. -/
theorem disj_sub {α : Type} {a b c e : Set α} (h : Disjoint a b) (h1 : c ⊆ a) (h2 : e ⊆ b) :
    Disjoint c e :=
  Disjoint.mono h1 h2 h

/-- Distinct programs have disjoint residual row sets. -/
theorem resRows_disjoint {d : Inst} (hD : 0 < d.D) {p q : Nat} (hpq : p ≠ q) :
    Disjoint (resRows d p) (resRows d q) :=
  disj_sub (yRows_disjoint hD hpq) (resRows_sub_yRows d p) (resRows_sub_yRows d q)

/-! ## Footprint disjointness -/

/-- Read and write footprints never overlap. -/
theorem R_disjoint_W {d : Inst} (hd : d.ok) (i j : Nat) : Disjoint (R d i) (W d j) :=
  disj_sub hd.out_in.symm (R_subset_inBytes hd i) (W_subset_outBytes d j)

/-- Two different programs write disjoint bytes. -/
theorem W_disjoint_W {d : Inst} (hd : d.ok) {i j : Nat} (hij : i ≠ j) :
    Disjoint (W d i) (W d j) := by
  simp only [W]
  refine Disjoint.union_left (Disjoint.union_left (Disjoint.union_left ?_ ?_) ?_) ?_
  · refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_
    · exact d.y.elems_disjoint (yRows_disjoint hd.d_pos hij)
    · exact disj_sub hd.y_res_out (d.y.elems_mono (yRows_subset_matSet d i))
        (d.res_out.elems_mono (resRows_subset_resSet d j))
    · exact disj_sub hd.y_mean (d.y.elems_mono (yRows_subset_matSet d i))
        (d.mean.elems_mono (mRows_subset_rowSet d j))
    · exact disj_sub hd.y_rstd (d.y.elems_mono (yRows_subset_matSet d i))
        (d.rstd.elems_mono (mRows_subset_rowSet d j))
  · refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_
    · exact disj_sub hd.y_res_out.symm (d.res_out.elems_mono (resRows_subset_resSet d i))
        (d.y.elems_mono (yRows_subset_matSet d j))
    · exact d.res_out.elems_disjoint (resRows_disjoint hd.d_pos hij)
    · exact disj_sub hd.res_out_mean (d.res_out.elems_mono (resRows_subset_resSet d i))
        (d.mean.elems_mono (mRows_subset_rowSet d j))
    · exact disj_sub hd.res_out_rstd (d.res_out.elems_mono (resRows_subset_resSet d i))
        (d.rstd.elems_mono (mRows_subset_rowSet d j))
  · refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_
    · exact disj_sub hd.y_mean.symm (d.mean.elems_mono (mRows_subset_rowSet d i))
        (d.y.elems_mono (yRows_subset_matSet d j))
    · exact disj_sub hd.res_out_mean.symm (d.mean.elems_mono (mRows_subset_rowSet d i))
        (d.res_out.elems_mono (resRows_subset_resSet d j))
    · exact d.mean.elems_disjoint (mRows_disjoint hij)
    · exact disj_sub hd.mean_rstd (d.mean.elems_mono (mRows_subset_rowSet d i))
        (d.rstd.elems_mono (mRows_subset_rowSet d j))
  · refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_
    · exact disj_sub hd.y_rstd.symm (d.rstd.elems_mono (mRows_subset_rowSet d i))
        (d.y.elems_mono (yRows_subset_matSet d j))
    · exact disj_sub hd.res_out_rstd.symm (d.rstd.elems_mono (mRows_subset_rowSet d i))
        (d.res_out.elems_mono (resRows_subset_resSet d j))
    · exact disj_sub hd.mean_rstd.symm (d.rstd.elems_mono (mRows_subset_rowSet d i))
        (d.mean.elems_mono (mRows_subset_rowSet d j))
    · exact d.rstd.elems_disjoint (mRows_disjoint hij)

/-! ## Rows and the specification -/

/-- Row `t < T` belongs to a program of the grid. -/
theorem div32_lt {d : Inst} (_hd : d.ok) {t : Nat} (ht : t < d.T) :
    t / 32 < (d.T + 31) / 32 := by
  have h1 := Nat.div_add_mod t 32
  have h2 := Nat.mod_lt t (by norm_num : 0 < 32)
  have h3 := Nat.div_add_mod (d.T + 31) 32
  have h4 := Nat.mod_lt (d.T + 31) (by norm_num : 0 < 32)
  omega

/-- Element `(t, j)` with `t / 32 = p` belongs to the output elements of program `p`. -/
theorem yRows_mem {d : Inst} {p t j : Nat} (ht : t < d.T) (hp : t / 32 = p) (hj : j < d.D) :
    idx d t j ∈ yRows d p := ⟨t, ht, hp, j, hj, rfl⟩

/-- Row `t` with `t / 32 = p` belongs to the mean/rstd rows of program `p`. -/
theorem mRows_mem {d : Inst} {p t : Nat} (ht : t < d.T) (hp : t / 32 = p) : t ∈ mRows d p :=
  ⟨ht, hp⟩

/-- With a residual present, element `(t, j)` with `t / 32 = p` belongs to the residual elements of program `p`. -/
theorem resRows_mem {d : Inst} {p t j : Nat} (hres : d.hasRes = true) (ht : t < d.T)
    (hp : t / 32 = p) (hj : j < d.D) : idx d t j ∈ resRows d p :=
  ⟨hres, yRows_mem ht hp hj⟩

/-- The output-buffer cells of program `p` lie in its write footprint. -/
theorem y_elems_sub_W (d : Inst) (p : Nat) : d.y.elems (yRows d p) ⊆ W d p :=
  Set.subset_union_left.trans (Set.subset_union_left.trans Set.subset_union_left)

/-- The residual-output cells of program `p` lie in its write footprint. -/
theorem res_out_elems_sub_W (d : Inst) (p : Nat) : d.res_out.elems (resRows d p) ⊆ W d p :=
  Set.subset_union_right.trans (Set.subset_union_left.trans Set.subset_union_left)

/-- The mean-buffer cells of program `p` lie in its write footprint. -/
theorem mean_elems_sub_W (d : Inst) (p : Nat) : d.mean.elems (mRows d p) ⊆ W d p :=
  Set.subset_union_right.trans Set.subset_union_left

/-- The rstd-buffer cells of program `p` lie in its write footprint. -/
theorem rstd_elems_sub_W (d : Inst) (p : Nat) : d.rstd.elems (mRows d p) ⊆ W d p :=
  Set.subset_union_right

/-- A row's result survives any other program's writes. -/
theorem rowDone_congr {d : Inst} {init m fin : Mem} {t p : Nat} (ht : t < d.T)
    (hp : t / 32 = p) (hagree : ∀ x ∈ R d p ∪ W d p, fin x = m x) (h : RowDone d init m t) :
    RowDone d init fin t := by
  obtain ⟨h1, h2, h3, h4⟩ := h
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro hres j hj
    rw [Mem.read_congr (d := .bf16) (fun x hx => hagree x (Or.inr (res_out_elems_sub_W d p
      ((Buf.span_sub_elems d.res_out (resRows_mem hres ht hp hj)) hx))))]
    exact h1 hres j hj
  · obtain ⟨c, hc, hf, hb⟩ := h2
    exact ⟨c, by
      rw [Mem.read_congr (d := .f32) (fun x hx => hagree x (Or.inr (mean_elems_sub_W d p
        ((Buf.span_sub_elems d.mean (mRows_mem ht hp)) hx))))]
      exact hc, hf, hb⟩
  · obtain ⟨c, hc, hf, hb⟩ := h3
    exact ⟨c, by
      rw [Mem.read_congr (d := .f32) (fun x hx => hagree x (Or.inr (rstd_elems_sub_W d p
        ((Buf.span_sub_elems d.rstd (mRows_mem ht hp)) hx))))]
      exact hc, hf, hb⟩
  · intro j hj
    obtain ⟨c, hc, hf, hb⟩ := h4 j hj
    exact ⟨c, by
      rw [Mem.read_congr (d := .bf16) (fun x hx => hagree x (Or.inr (y_elems_sub_W d p
        ((Buf.span_sub_elems d.y (yRows_mem ht hp hj)) hx))))]
      exact hc, hf, hb⟩
/-! ## The `hin` bounds -/

/-- Every enabled lane of the `[32, BD]` matrix access is inside the buffer. -/
theorem mat_hin {d : Inst} (hd : d.ok) (p : Nat) {dy : DType} {B : Buf dy}
    (hlen : d.T * d.D ≤ B.len) :
    ∀ i : Idx [32, d.BD],
      (fun i : Idx [32, d.BD] => decide (32 * p + (i.1 : Nat) < d.T ∧ (i.2.1 : Nat) < d.D)) i
        = true →
      (fun i : Idx [32, d.BD] => idx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat)) i < B.len := by
  rintro ⟨r, j, u⟩ he
  cases u
  have h := of_decide_eq_true he
  exact idx_lt_buf hd h.1 h.2 hlen

/-- Every enabled lane of the weight access is inside the buffer. -/
theorem vec_hin {d : Inst} (hd : d.ok) (p : Nat) {dy : DType} {B : Buf dy}
    (hlen : d.G * d.D ≤ B.len) :
    ∀ i : Idx [32, d.BD],
      (fun i : Idx [32, d.BD] => decide ((i.2.1 : Nat) < d.D)) i = true →
      (fun i : Idx [32, d.BD] => gidx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat)) i < B.len := by
  rintro ⟨r, j, u⟩ he
  cases u
  exact gidx_lt_buf hd (of_decide_eq_true he) hlen

/-- Every enabled lane of the `[32]` row access is inside the buffer. -/
theorem row_hin {d : Inst} (_hd : d.ok) (p : Nat) {dy : DType} {B : Buf dy}
    (hlen : d.T ≤ B.len) :
    ∀ i : Idx [32],
      (fun i : Idx [32] => decide (32 * p + (i.1 : Nat) < d.T)) i = true →
      (fun i : Idx [32] => 32 * p + (i.1 : Nat)) i < B.len := by
  rintro ⟨r, u⟩ he
  cases u
  exact lt_of_lt_of_le (of_decide_eq_true he) hlen

/-! ## The images of the access patterns -/

/-- The matrix lanes of program `p` hit only its rows. -/
theorem img_mat_sub_yRows {d : Inst} (p : Nat) :
    Buf.img (fun i : Idx [32, d.BD] => idx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat))
        (fun i : Idx [32, d.BD] => decide (32 * p + (i.1 : Nat) < d.T ∧ (i.2.1 : Nat) < d.D))
      ⊆ yRows d p := by
  rintro k ⟨i, he, rfl⟩
  obtain ⟨r, j, u⟩ := i
  cases u
  have h := of_decide_eq_true he
  exact yRows_mem h.1 (div32_add r.isLt) h.2

/-- Distinct enabled matrix lanes address distinct elements. -/
theorem mat_img_inj {d : Inst} (p : Nat) :
    ∀ i j : Idx [32, d.BD],
      (fun i : Idx [32, d.BD] => decide (32 * p + (i.1 : Nat) < d.T ∧ (i.2.1 : Nat) < d.D)) i = true →
      (fun i : Idx [32, d.BD] => decide (32 * p + (i.1 : Nat) < d.T ∧ (i.2.1 : Nat) < d.D)) j = true →
      (fun i : Idx [32, d.BD] => idx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat)) i =
        (fun i : Idx [32, d.BD] => idx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat)) j → i = j := by
  rintro ⟨ri, ji, ui⟩ ⟨rj, jj, uj⟩ hi hj he
  have hi' := of_decide_eq_true hi
  have hj' := of_decide_eq_true hj
  have hji : (ji : Nat) < d.D := hi'.2
  have hjj : (jj : Nat) < d.D := hj'.2
  simp only [] at he
  obtain ⟨h1, h2⟩ := idx_inj (d := d) (by omega) hji hjj he
  have h3 : (ri : Nat) = (rj : Nat) := by omega
  refine Prod.ext (Fin.ext h3) (Prod.ext (Fin.ext h2) ?_)
  cases ui; cases uj; rfl

/-- Distinct enabled row lanes address distinct elements. -/
theorem row_img_inj {d : Inst} (p : Nat) :
    ∀ i j : Idx [32],
      (fun i : Idx [32] => decide (32 * p + (i.1 : Nat) < d.T)) i = true →
      (fun i : Idx [32] => decide (32 * p + (i.1 : Nat) < d.T)) j = true →
      (fun i : Idx [32] => 32 * p + (i.1 : Nat)) i =
        (fun i : Idx [32] => 32 * p + (i.1 : Nat)) j → i = j := by
  rintro ⟨ri, ui⟩ ⟨rj, uj⟩ _ _ he
  simp only [] at he
  have h3 : (ri : Nat) = (rj : Nat) := by omega
  refine Prod.ext (Fin.ext h3) ?_
  cases ui; cases uj; rfl

/-- If `A` and `B` sit inside disjoint sets `A'` and `B'`, an element of `A` is not in `B`. -/
theorem not_mem_right {A A' B B' : Set Int} (h : Disjoint A' B') (hA : A ⊆ A') (hB : B ⊆ B')
    {x : Int} (hx : x ∈ A) : x ∉ B :=
  fun hb => Set.disjoint_left.mp (h.mono hA hB) hx hb

/-- With a residual, the matrix lanes hit only the residual rows of `p`. -/
theorem img_mat_sub_resRows {d : Inst} (p : Nat) (hres : d.hasRes = true) :
    Buf.img (fun i : Idx [32, d.BD] => idx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat))
        (fun i : Idx [32, d.BD] => decide (32 * p + (i.1 : Nat) < d.T ∧ (i.2.1 : Nat) < d.D))
      ⊆ resRows d p := by
  intro k hk
  exact ⟨hres, img_mat_sub_yRows p hk⟩

/-- The weight lanes hit only the weight rows of `p`. -/
theorem img_vec_sub_wRows {d : Inst} (p : Nat) :
    Buf.img (fun i : Idx [32, d.BD] => gidx d (32 * p + (i.1 : Nat)) (i.2.1 : Nat))
        (fun i : Idx [32, d.BD] => decide ((i.2.1 : Nat) < d.D))
      ⊆ wRows d p := by
  rintro k ⟨i, he, rfl⟩
  obtain ⟨r, j, u⟩ := i
  cases u
  exact ⟨r, r.isLt, j, of_decide_eq_true he, rfl⟩

/-- The row lanes hit only the `mean`/`rstd` rows of `p`. -/
theorem img_row_sub_mRows {d : Inst} (p : Nat) :
    Buf.img (fun i : Idx [32] => 32 * p + (i.1 : Nat))
        (fun i : Idx [32] => decide (32 * p + (i.1 : Nat) < d.T))
      ⊆ mRows d p := by
  rintro k ⟨i, he, rfl⟩
  obtain ⟨r, u⟩ := i
  cases u
  exact mRows_mem (of_decide_eq_true he) (div32_add r.isLt)

/-! ## Footprint inclusions -/

theorem x_elems_sub_R (d : Inst) (p : Nat) : d.x.elems (yRows d p) ⊆ R d p :=
  Set.subset_union_left.trans (Set.subset_union_left.trans Set.subset_union_left)

/-- The residual-input cells of program `p` lie in its read footprint. -/
theorem res_elems_sub_R (d : Inst) (p : Nat) : d.res.elems (resRows d p) ⊆ R d p :=
  Set.subset_union_right.trans (Set.subset_union_left.trans Set.subset_union_left)

/-- The weight cells of program `p` lie in its read footprint. -/
theorem w_elems_sub_R (d : Inst) (p : Nat) : d.w.elems (wRows d p) ⊆ R d p :=
  Set.subset_union_right.trans Set.subset_union_left

/-- The bias cells of program `p` lie in its read footprint. -/
theorem b_elems_sub_R (d : Inst) (p : Nat) : d.b.elems (wRows d p) ⊆ R d p :=
  Set.subset_union_right

/-! ## Validity across stores -/

/-- A buffer whose bytes are all still present is still valid. -/
theorem valid_of_present {d : DType} {b : Buf d} {m m' : Mem} (hv : b.Valid m)
    (h : ∀ x, m x ≠ none → m' x ≠ none) : b.Valid m' :=
  Buf.valid_of_agree hv fun x hx => h x (hv.2.2.2 x hx)

/-- Every buffer of the instance is valid, unconditionally. -/
def Vals (d : Inst) (m : Mem) : Prop :=
  d.x.Valid m ∧ d.res.Valid m ∧ d.w.Valid m ∧ d.b.Valid m ∧ d.y.Valid m ∧
    d.res_out.Valid m ∧ d.mean.Valid m ∧ d.rstd.Valid m

/-- `Vals` survives any memory that keeps every present byte present. -/
theorem Vals.of_present {d : Inst} {m m' : Mem} (h : Vals d m)
    (hpr : ∀ x, m x ≠ none → m' x ≠ none) : Vals d m' :=
  ⟨valid_of_present h.1 hpr, valid_of_present h.2.1 hpr, valid_of_present h.2.2.1 hpr,
    valid_of_present h.2.2.2.1 hpr, valid_of_present h.2.2.2.2.1 hpr,
    valid_of_present h.2.2.2.2.2.1 hpr, valid_of_present h.2.2.2.2.2.2.1 hpr,
    valid_of_present h.2.2.2.2.2.2.2 hpr⟩

/-! ## The write footprints left after the earlier stores -/

/-- After the `res_out` store, the row store to `mean` still fits in `W`. -/
theorem mean_elems_sub_W_sdiff {d : Inst} (hd : d.ok) (p : Nat)
    {S S' : Set Nat} (hS : S ⊆ resRows d p) (hS' : S' ⊆ mRows d p) :
    d.mean.elems S' ⊆ W d p \ d.res_out.elems S :=
  Set.subset_sdiff.mpr ⟨(d.mean.elems_mono hS').trans (mean_elems_sub_W d p),
    (disj_sub hd.res_out_mean
      (d.res_out.elems_mono (hS.trans (resRows_subset_resSet d p)))
      (d.mean.elems_mono (hS'.trans (mRows_subset_rowSet d p)))).symm⟩

/-- What remains of `W p` after `res_out` and `mean` stores contains the `rstd` elements. -/
theorem rstd_elems_sub_W_sdiff {d : Inst} (hd : d.ok) (p : Nat)
    {S S' S'' : Set Nat} (hS : S ⊆ resRows d p) (hS' : S' ⊆ mRows d p)
    (hS'' : S'' ⊆ mRows d p) :
    d.rstd.elems S'' ⊆ (W d p \ d.res_out.elems S) \ d.mean.elems S' :=
  Set.subset_sdiff.mpr
    ⟨Set.subset_sdiff.mpr ⟨(d.rstd.elems_mono hS'').trans (rstd_elems_sub_W d p),
      (disj_sub hd.res_out_rstd
        (d.res_out.elems_mono (hS.trans (resRows_subset_resSet d p)))
        (d.rstd.elems_mono (hS''.trans (mRows_subset_rowSet d p)))).symm⟩,
     (disj_sub hd.mean_rstd
       (d.mean.elems_mono (hS'.trans (mRows_subset_rowSet d p)))
       (d.rstd.elems_mono (hS''.trans (mRows_subset_rowSet d p)))).symm⟩

/-- What remains of `W p` after `res_out`, `mean`, `rstd` stores contains the `y` elements. -/
theorem y_elems_sub_W_sdiff {d : Inst} (hd : d.ok) (p : Nat)
    {S S' S'' S''' : Set Nat} (hS : S ⊆ resRows d p) (hS' : S' ⊆ mRows d p)
    (hS'' : S'' ⊆ mRows d p) (hS''' : S''' ⊆ yRows d p) :
    d.y.elems S''' ⊆ ((W d p \ d.res_out.elems S) \ d.mean.elems S') \ d.rstd.elems S'' :=
  Set.subset_sdiff.mpr
    ⟨Set.subset_sdiff.mpr
      ⟨Set.subset_sdiff.mpr ⟨(d.y.elems_mono hS''').trans (y_elems_sub_W d p),
        disj_sub hd.y_res_out
          (d.y.elems_mono (hS'''.trans (yRows_subset_matSet d p)))
          (d.res_out.elems_mono (hS.trans (resRows_subset_resSet d p)))⟩,
       disj_sub hd.y_mean
         (d.y.elems_mono (hS'''.trans (yRows_subset_matSet d p)))
         (d.mean.elems_mono (hS'.trans (mRows_subset_rowSet d p)))⟩,
     disj_sub hd.y_rstd
       (d.y.elems_mono (hS'''.trans (yRows_subset_matSet d p)))
       (d.rstd.elems_mono (hS''.trans (mRows_subset_rowSet d p)))⟩


/-- After the `mean` store only (no residual), the row store to `rstd` still fits in `W`. -/
theorem rstd_elems_sub_W_sdiff0 {d : Inst} (hd : d.ok) (p : Nat)
    {S' S'' : Set Nat} (hS' : S' ⊆ mRows d p) (hS'' : S'' ⊆ mRows d p) :
    d.rstd.elems S'' ⊆ W d p \ d.mean.elems S' :=
  Set.subset_sdiff.mpr ⟨(d.rstd.elems_mono hS'').trans (rstd_elems_sub_W d p),
    (disj_sub hd.mean_rstd
      (d.mean.elems_mono (hS'.trans (mRows_subset_rowSet d p)))
      (d.rstd.elems_mono (hS''.trans (mRows_subset_rowSet d p)))).symm⟩

/-- After the `mean` and `rstd` stores (no residual), the row store to `y` still fits in `W`. -/
theorem y_elems_sub_W_sdiff0 {d : Inst} (hd : d.ok) (p : Nat)
    {S' S'' S''' : Set Nat} (hS' : S' ⊆ mRows d p) (hS'' : S'' ⊆ mRows d p)
    (hS''' : S''' ⊆ yRows d p) :
    d.y.elems S''' ⊆ (W d p \ d.mean.elems S') \ d.rstd.elems S'' :=
  Set.subset_sdiff.mpr
    ⟨Set.subset_sdiff.mpr ⟨(d.y.elems_mono hS''').trans (y_elems_sub_W d p),
      disj_sub hd.y_mean
        (d.y.elems_mono (hS'''.trans (yRows_subset_matSet d p)))
        (d.mean.elems_mono (hS'.trans (mRows_subset_rowSet d p)))⟩,
     disj_sub hd.y_rstd
       (d.y.elems_mono (hS'''.trans (yRows_subset_matSet d p)))
       (d.rstd.elems_mono (hS''.trans (mRows_subset_rowSet d p)))⟩

end Claims.LayerNormFwd
