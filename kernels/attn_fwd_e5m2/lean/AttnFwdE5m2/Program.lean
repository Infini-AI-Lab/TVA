import AttnFwdE5m2.Inner

/-!
# `attn_fwd_e5m2`: one program of the launch

* `row0`: program `i`'s query tile starts at row `128·i` of the `[Z·H·N_CTX, HEAD_DIM]` view
  (`off_hz·N_CTX + 128·start_m`);
* `oTile`, `mTile`, `Wr`: the elements of `o` and `M` program `i` writes;
* `psum_below`, `blk_sum`, `ew_weight`, `acR_accA`, `ecR_errC`, `centC_final`: the row's sums and
  error bounds in the specification's terms (`weight`, `partW`, `centC`, `blkC`, `blkA`, `blkLW`,
  `blkVL`, `accA`, `errC`), and the centred sum after the last block (`0`);
* `OutO`, `OutM`, `Qp`: what `Post` asks of an element, and of a program's tile;
* `row_out`: the end of a row (lines 241–245) from its state after the loop (the centred bound at
  `r = attnRef`, `l_final`, `final_v`, `final_M`);
* `safe_epi`: lines 241–245 of a program;
* `safe_program`: a whole program (lines 189–245).
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

/-! ## The tiles -/

theorem n_eq {d : Inst} (hd : d.ok) : d.N_CTX = 128 * g0 d := by
  rw [g0_eq hd]; have := hd.n_mul; omega

/-- **Program `i`'s first row** in the `[Z·H·N_CTX, HEAD_DIM]` view: `off_hz·N_CTX + 128·start_m`
is `128·i`. -/
theorem row0 {d : Inst} (hd : d.ok) (i : ℕ) : pbh d i * d.N_CTX + 128 * pm d i = 128 * i := by
  have h2 := Nat.div_add_mod i (g0 d)
  unfold pbh pm
  rw [n_eq hd]
  calc i / g0 d * (128 * g0 d) + 128 * (i % g0 d) = 128 * (g0 d * (i / g0 d) + i % g0 d) := by ring
    _ = 128 * i := by rw [h2]

theorem tile_le {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) : 128 * i + 128 ≤ rows d := by
  rw [launch_size] at hi
  have h1 := n_eq hd
  have : i + 1 ≤ g0 d * (d.Z * d.H) := hi
  calc 128 * i + 128 = 128 * (i + 1) := by ring
    _ ≤ 128 * (g0 d * (d.Z * d.H)) := Nat.mul_le_mul_left _ this
    _ = rows d := by unfold rows; rw [h1]; ring

/-- The elements of `o` program `i` writes (line 245): rows `128·i …` of the view. -/
def oTile (d : Inst) (i : ℕ) : Set ℕ :=
  {x | ∃ r < 128, ∃ e < d.HEAD_DIM, x = (128 * i + r) * d.HEAD_DIM + e}

/-- The elements of `M` program `i` writes (line 244). -/
def mTile (i : ℕ) : Set ℕ := {x | ∃ r < 128, x = 128 * i + r}

/-- Program `i`'s write footprint. -/
def Wr (d : Inst) (i : ℕ) : Set Int := d.o.elems (oTile d i) ∪ d.M.elems (mTile i)

theorem oTile_sub {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) :
    oTile d i ⊆ below (rows d * d.HEAD_DIM) := by
  rintro _ ⟨r, hr, e, he, rfl⟩
  have h1 := tile_le hd hi
  show (128 * i + r) * d.HEAD_DIM + e < rows d * d.HEAD_DIM
  calc (128 * i + r) * d.HEAD_DIM + e < (128 * i + r) * d.HEAD_DIM + d.HEAD_DIM := by omega
    _ = (128 * i + r + 1) * d.HEAD_DIM := by ring
    _ ≤ rows d * d.HEAD_DIM := Nat.mul_le_mul_right _ (by omega)

theorem mTile_sub {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) :
    mTile i ⊆ below (rows d) := by
  rintro _ ⟨r, hr, rfl⟩
  have h1 := tile_le hd hi
  show 128 * i + r < rows d
  omega

theorem Wr_sub {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) : Wr d i ⊆ outBytes d :=
  Set.union_subset_union (d.o.elems_mono (oTile_sub hd hi)) (d.M.elems_mono (mTile_sub hd hi))

theorem row_inj {D a b e e' : ℕ} (h : a * D + e = b * D + e') (he : e < D) (he' : e' < D) : a = b := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
  · have : (a + 1) * D ≤ b * D := Nat.mul_le_mul_right _ hlt
    nlinarith
  · have : (b + 1) * D ≤ a * D := Nat.mul_le_mul_right _ hlt
    nlinarith

theorem Wr_disj {d : Inst} (hd : d.ok) {i j : ℕ} (hi : i < (launch d).size) (hj : j < (launch d).size)
    (hij : i ≠ j) : Disjoint (Wr d i) (Wr d j) := by
  have hoo : Disjoint (d.o.elems (oTile d i)) (d.o.elems (oTile d j)) := by
    refine d.o.elems_disjoint (Set.disjoint_left.mpr ?_)
    rintro _ ⟨r, hr, e, he, rfl⟩ ⟨r', hr', e', he', h⟩
    have := row_inj h he he'
    omega
  have hmm : Disjoint (d.M.elems (mTile i)) (d.M.elems (mTile j)) := by
    refine d.M.elems_disjoint (Set.disjoint_left.mpr ?_)
    rintro _ ⟨r, hr, rfl⟩ ⟨r', hr', h⟩
    omega
  have hoM := hd.disj.2.2.2.1
  refine Disjoint.union_left (Disjoint.union_right hoo ?_) (Disjoint.union_right ?_ hmm)
  · exact hoM.mono (d.o.elems_mono (oTile_sub hd hi)) (d.M.elems_mono (mTile_sub hd hj))
  · exact hoM.symm.mono (d.M.elems_mono (mTile_sub hd hi)) (d.o.elems_mono (oTile_sub hd hj))

theorem keys_lt {d : Inst} {n j : ℕ} (hj : j ∈ keys d n) : j < d.N_CTX := by
  unfold keys at hj
  simp only [Finset.mem_filter, Finset.mem_range] at hj
  exact hj.1

/-! ## The row's sums in the specification's terms -/

/-- The row's sums over the first `t` blocks are its sums over `keysBelow`. -/
theorem psum_below {d : Inst} {n : ℕ} (t : ℕ) (f : ℕ → ℝ) :
    psum (alOf d n) t f = ∑ j ∈ keysBelow d n t, f j := by
  unfold psum keysBelow
  rw [← Finset.sum_filter]
  refine Finset.sum_congr ?_ fun _ _ => rfl
  ext j
  simp only [Finset.mem_filter, Finset.mem_range, keys, alOf, decide_eq_true_eq]
  tauto

/-- The row's sums over block `t` are its sums over `keysBlk`. -/
theorem blk_sum {d : Inst} {n : ℕ} (t : ℕ) (f : ℕ → ℝ) :
    ∑ jj : Fin 64, (if alOf d n (64 * t + jj) = true then f (64 * t + jj) else 0) =
      ∑ j ∈ keysBlk d n t, f j := by
  have h1 := psum_succ (alOf d n) t f
  rw [psum_below, psum_below] at h1
  have h2 : ∑ j ∈ keysBelow d n (t + 1), f j = ∑ j ∈ keysBelow d n t, f j + ∑ j ∈ keysBlk d n t, f j := by
    rw [← Finset.sum_filter_add_sum_filter_not (keysBelow d n (t + 1)) (fun j => j < 64 * t)]
    congr 1
    · refine Finset.sum_congr ?_ fun _ _ => rfl
      ext j
      simp only [keysBelow, Finset.mem_filter]
      constructor
      · rintro ⟨⟨hk, -⟩, h2⟩; exact ⟨hk, h2⟩
      · rintro ⟨hk, h2⟩; exact ⟨⟨hk, by omega⟩, h2⟩
    · refine Finset.sum_congr ?_ fun _ _ => rfl
      ext j
      simp only [keysBelow, keysBlk, Finset.mem_filter]
      constructor
      · rintro ⟨⟨hk, h1⟩, h2⟩; exact ⟨hk, by omega, by omega⟩
      · rintro ⟨hk, h1, h2⟩; exact ⟨⟨hk, by omega⟩, by omega⟩
  linarith

/-- The exact weight `2^(w·s_j)` of `RowCtx` is the specification's `exp(sm_scale·score)`. -/
theorem ew_weight (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (j : ℕ) :
    ew (rowCtx d init c bh n X Z) j = weight d init bh n j := by
  show (2 : ℝ) ^ ((sigma d : ℝ) / Real.log 2 * (score d init bh n j : ℝ)) =
    Real.exp ((sigma d : ℝ) * (score d init bh n j : ℝ))
  rw [Real.rpow_def_of_pos (by norm_num)]
  congr 1
  have : Real.log 2 ≠ 0 := (Real.log_pos (by norm_num)).ne'
  field_simp

theorem veOf_key {d : Inst} {init : Mem} {bh n j : ℕ} (e : Fin d.HEAD_DIM) (hj : j ∈ keys d n) :
    veOf d init bh e j = (vEff d init bh j e : ℝ) := by
  unfold veOf; rw [ite_eq_left (keys_lt hj)]

theorem vlOf_key {d : Inst} {init : Mem} {bh n j : ℕ} (e : Fin d.HEAD_DIM) (hj : j ∈ keys d n) :
    vlOf d init bh e j = ((Rules.Num.liftMag .f8e5 (vEff d init bh j e) : ℚ) : ℝ) := by
  unfold vlOf; rw [ite_eq_left (keys_lt hj)]

/-- `Cen.Cx` at `r = attnRef` is `Model.centC`. -/
theorem Cx_centC (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (e : Fin d.HEAD_DIM) (t : ℕ) :
    Cx (rowCtx d init c bh n X Z) (veOf d init bh e) (attnRef d init bh n e) t = centC d init bh n e t := by
  unfold Cx centC
  rw [show (rowCtx d init c bh n X Z).al = alOf d n from rfl, psum_below]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' : j ∈ keys d n := (Finset.mem_filter.mp hj).1
  rw [ew_weight, veOf_key e hj']

theorem psum_partW (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (t : ℕ) :
    psum (rowCtx d init c bh n X Z).al t (ew (rowCtx d init c bh n X Z)) = partW d init bh n t := by
  rw [show (rowCtx d init c bh n X Z).al = alOf d n from rfl, psum_below]
  exact Finset.sum_congr rfl fun j _ => ew_weight d init c bh n X Z j

theorem bC_blkC (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (e : Fin d.HEAD_DIM) (t : ℕ) :
    bC (rowCtx d init c bh n X Z) (veOf d init bh e) (attnRef d init bh n e) t = blkC d init bh n e t := by
  unfold bC blkC
  rw [show (rowCtx d init c bh n X Z).al = alOf d n from rfl,
    blk_sum t (fun j => ew (rowCtx d init c bh n X Z) j * |veOf d init bh e j - attnRef d init bh n e|)]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' : j ∈ keys d n := (Finset.mem_filter.mp hj).1
  rw [ew_weight, veOf_key e hj']

theorem bA_blkA (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (e : Fin d.HEAD_DIM) (t : ℕ) :
    bA (rowCtx d init c bh n X Z) (veOf d init bh e) (attnRef d init bh n e) t = blkA d init bh n e t := by
  unfold bA blkA
  rw [show (rowCtx d init c bh n X Z).al = alOf d n from rfl,
    blk_sum t (fun j => |veOf d init bh e j - attnRef d init bh n e|)]
  refine Finset.sum_congr rfl fun j hj => ?_
  have hj' : j ∈ keys d n := (Finset.mem_filter.mp hj).1
  rw [veOf_key e hj']

theorem bVL_blkVL (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (e : Fin d.HEAD_DIM) (t : ℕ) :
    bVL (rowCtx d init c bh n X Z) (vlOf d init bh e) t = blkVL d init bh n e t := by
  unfold bVL blkVL
  refine Finset.sum_congr rfl fun c' _ => ?_
  congr 1
  funext k
  show (if alOf d n (64 * t + k) = true then vlOf d init bh e (64 * t + k) else 0) = _
  by_cases h : alOf d n (64 * t + k) = true
  · have hk := mem_keys.mpr h
    rw [if_pos h, if_pos hk, vlOf_key e hk]
  · have hk : 64 * t + (k : ℕ) ∉ keys d n := fun hk => h (mem_keys.mp hk)
    rw [if_neg h, if_neg hk]

theorem bLW_blkLW (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (e : Fin d.HEAD_DIM) (t : ℕ) :
    bLW (rowCtx d init c bh n X Z) (vlOf d init bh e) t = blkLW d init bh n e t := by
  unfold bLW blkLW
  refine Finset.sum_congr rfl fun c' _ => ?_
  congr 1
  funext k
  show (if alOf d n (64 * t + k) = true then
    ew (rowCtx d init c bh n X Z) (64 * t + k) * vlOf d init bh e (64 * t + k) else 0) = _
  by_cases h : alOf d n (64 * t + k) = true
  · have hk := mem_keys.mpr h
    rw [if_pos h, if_pos hk, vlOf_key e hk, ew_weight]
  · have hk : 64 * t + (k : ℕ) ∉ keys d n := fun hk => h (mem_keys.mp hk)
    rw [if_neg h, if_neg hk]

/-- `Cen.acR` at `r = attnRef` is `Model.accA`. -/
theorem acR_accA (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (e : Fin d.HEAD_DIM) :
    ∀ t, acR (rowCtx d init c bh n X Z) (veOf d init bh e) (vlOf d init bh e) (attnRef d init bh n e) t =
      accA d init bh n e X Z t
  | 0 => rfl
  | t + 1 => by
    rw [acR, accA, acR_accA d init c bh n X Z e t, bA_blkA, bVL_blkVL,
      ← dmax_r (R := rowCtx d init c bh n X Z) (D := d.HEAD_DIM) (X := X) (Z := Z) rfl rfl]

/-- `Cen.ecR` at `r = attnRef` is `Model.errC`. -/
theorem ecR_errC (d : Inst) (init : Mem) (c : ℝ) (bh n : ℕ) (X Z : ℚ) (e : Fin d.HEAD_DIM) :
    ∀ t, ecR (rowCtx d init c bh n X Z) (veOf d init bh e) (vlOf d init bh e) (attnRef d init bh n e) t =
      errC d init bh n e X Z t
  | 0 => rfl
  | t + 1 => by
    rw [ecR, errC, ecR_errC d init c bh n X Z e t, acR_accA, Cx_centC, bC_blkC, bLW_blkLW, psum_partW,
      ← dmax_r (R := rowCtx d init c bh n X Z) (D := d.HEAD_DIM) (X := X) (Z := Z) rfl rfl,
      ← pE_r (R := rowCtx d init c bh n X Z) (D := d.HEAD_DIM) (X := X) (Z := Z) rfl,
      ← FhT_r (R := rowCtx d init c bh n X Z) (D := d.HEAD_DIM) (X := X) (Z := Z) rfl rfl]

/-- After the last block the centred sum is `0`: `attnRef` is the weights' mean. -/
theorem centC_final {d : Inst} {init : Mem} {bh n T : ℕ} (hTk : ∀ j, alOf d n j = true → j < 64 * T)
    (hW : 0 < wsum d init bh n) (c : ℝ) (X Z : ℚ) (e : Fin d.HEAD_DIM) :
    Cx (rowCtx d init c bh n X Z) (veOf d init bh e) (attnRef d init bh n e) T = 0 := by
  unfold Cx
  rw [show (rowCtx d init c bh n X Z).al = alOf d n from rfl, psum_keys hTk]
  have h1 : ∑ j ∈ keys d n, ew (rowCtx d init c bh n X Z) j * (veOf d init bh e j - attnRef d init bh n e) =
      ∑ j ∈ keys d n, weight d init bh n j * (vEff d init bh j e : ℝ) -
        attnRef d init bh n e * wsum d init bh n := by
    unfold wsum
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [ew_weight, veOf_key e hj]; ring
  rw [h1]
  unfold attnRef
  field_simp
  ring

/-! ## What a program leaves -/

/-- What `Post` asks of element `(bh, n, e)` of `o` holding `c`. -/
def OutO (d : Inst) (init : Mem) (bh n e : ℕ) (c : F8E5) : Prop :=
  HeadOk d init bh → c.isFinite = true ∧
    (∀ X Z : ℚ, X ≤ Xh d.HEAD_DIM →
      (∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X) →
      (∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) →
      |c.toReal - attnRef d init bh n e| ≤ tolO d init bh n e X Z) ∧
    ∀ K : ℝ, (∀ j ∈ keys d n, |(vEff d init bh j e : ℝ) - attnRef d init bh n e| ≤ K) →
      |c.toReal - attnRef d init bh n e| ≤ tolV d init bh n e K

/-- What `Post` asks of element `(bh, n)` of `M` holding `c`. -/
def OutM (d : Inst) (init : Mem) (bh n : ℕ) (c : F32) : Prop :=
  HeadOk d init bh → c.isFinite = true ∧ ∀ X Z : ℚ, X ≤ Xh d.HEAD_DIM →
    (∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X) →
    (∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) →
    |c.toReal - lseRef d init bh n| ≤ tolM d.HEAD_DIM d.N_CTX X Z + (u32 : ℝ) * |lseRef d init bh n|

/-- What program `(sm, bh)` leaves in memory `m`: its rows of `o` and `M` as `Post` asks. -/
def Qp (d : Inst) (init : Mem) (sm bh : ℕ) (m : Mem) : Prop :=
  ∀ r : Fin 128,
    (∀ e : Fin d.HEAD_DIM, ∃ c : F8E5,
      m.read .f8e5 (d.o.ptr ((idx d bh (128 * sm + r) e : ℕ) : ℤ)).addr = some c ∧
        OutO d init bh (128 * sm + r) e c) ∧
    ∃ c : F32, m.read .f32 (d.M.ptr ((bh * d.N_CTX + (128 * sm + r) : ℕ) : ℤ)).addr = some c ∧
      OutM d init bh (128 * sm + r) c

/-- **The end of a row** (lines 241–245): from the row's state after the `T = nblk` blocks that
hold all its keys, `(acc / l_i).to(float8e5)` and `m_i + log2(l_i)` are as `Post` asks. The first
bound on `o` is the centred bound (`CInv`) at `r = attnRef`, where the centred sum is `0`
(`centC_final`), with `l_i ≥ Fp` times the weights' sum (`l_final`); the second is `RowSt`'s
magnitude bound at `X = Xh HEAD_DIM`. -/
theorem row_out {d : Inst} (hd : d.ok) {init : Mem} {P : Platform} (hdivF : Rules.Num.DivFull P)
    (hcvt : Rules.Num.CvtSatF32E5m2 P) (hlog : Rules.Num.Log2F32 P) {c : F32} (hc : CFacts d c)
    {bh n : ℕ} (hn : n < d.N_CTX) {T : ℕ} (hTn : nblk d n = T) (hT0 : 0 < T) (hT : T ≤ nblocks d.N_CTX)
    (hTk : ∀ j, alOf d n j = true → j < 64 * T)
    {acc : Fin d.HEAD_DIM → F32} {l m : F32}
    (hst : HeadOk d init bh → ∀ X Z : ℚ, X ≤ Xh d.HEAD_DIM →
      (∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X) →
      (∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) →
      RowSt (rowCtx d init c.toReal bh n X Z) (veOf d init bh) (vlOf d init bh) T m l acc)
    {lg Mv : F32} (hlg : P.math .log2 .binary32 l lg) (hMv : FloatV.NaNEq (FloatV.add m lg) Mv)
    {o' : Fin d.HEAD_DIM → F32} (ho' : ∀ e, P.div .binary32 (acc e) l (o' e))
    {o8 : Fin d.HEAD_DIM → F8E5} (ho8 : ∀ e, P.cvt Format.binary32 Format.e5m2 .rn (o' e) (o8 e)) :
    (∀ e : Fin d.HEAD_DIM, OutO d init bh n e (o8 e)) ∧ OutM d init bh n Mv := by
  have hN := hd.n_le
  have hTle : T ≤ 1024 := hT.trans (by unfold nblocks; omega)
  have hfin : HeadOk d init bh → ∀ X Z : ℚ, X ≤ Xh d.HEAD_DIM →
      (∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X) →
      (∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) →
      (rowCtx d init c.toReal bh n X Z).Ok ∧ MInv (rowCtx d init c.toReal bh n X Z) T m ∧
        LInv (rowCtx d init c.toReal bh n X Z) T m l ∧ (∀ e, ACr T (acc e)) ∧ lminB T ≤ l.toReal ∧
        (∀ e, CInv (rowCtx d init c.toReal bh n X Z) (veOf d init bh e) (vlOf d init bh e) T m (acc e) l) ∧
        ∀ e (r K : ℝ), 0 ≤ K → (∀ j, (rowCtx d init c.toReal bh n X Z).al j = true →
          |veOf d init bh e j - r| ≤ K) →
          |(acc e).toReal - r * l.toReal| ≤ magA K |r| T * l.toReal + magD K |r| T := by
    intro hH X Z hX hXk hZk
    obtain ⟨hRok, -⟩ := rowCtx_ok hd init hc.2.1 hc.2.2.1 hc.2.2.2.1 (qk_scale_w hc.2.2.2.2) bh hn hX hXk hZk
    rcases hst hH X Z hX hXk hZk with ⟨h0, -⟩ | ⟨-, hM, hL, hA, hlm, hC, hE⟩
    · omega
    · exact ⟨hRok, hM, hL, hA, hlm, hC, hE⟩
  have hW : ∀ X Z : ℚ, psum (rowCtx d init c.toReal bh n X Z).al T (ew (rowCtx d init c.toReal bh n X Z)) =
      wsum d init bh n := fun X Z => by
    rw [show (rowCtx d init c.toReal bh n X Z).al = alOf d n from rfl, psum_keys hTk]; unfold wsum
    exact Finset.sum_congr rfl fun j _ => ew_weight d init c.toReal bh n X Z j
  have hSx : ∀ (X Z : ℚ) (e : Fin d.HEAD_DIM),
      Sx (rowCtx d init c.toReal bh n X Z) (veOf d init bh e) T / psum (rowCtx d init c.toReal bh n X Z).al T
        (ew (rowCtx d init c.toReal bh n X Z)) = attnRef d init bh n e := fun X Z e => by
    rw [hW]; unfold Sx attnRef
    rw [show (rowCtx d init c.toReal bh n X Z).al = alOf d n from rfl, psum_keys hTk]
    congr 1
    exact Finset.sum_congr rfl fun j hj => by rw [ew_weight, veOf_key e hj]
  have hOo : ∀ e : Fin d.HEAD_DIM, HeadOk d init bh → ∀ X Z : ℚ, X ≤ Xh d.HEAD_DIM →
      (∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X) →
      (∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) →
      (o8 e).isFinite = true ∧ |(o8 e).toReal - attnRef d init bh n e| ≤ tolO d init bh n e X Z := by
    intro e hH X Z hX hXk hZk
    obtain ⟨hRok, hM, hL, hA, hlm, hC, -⟩ := hfin hH X Z hX hXk hZk
    obtain ⟨hFh, hFl, haL, hFp, -⟩ :=
      model_factors (D := d.HEAD_DIM) (N := d.N_CTX) (X := X) (Z := Z) hRok rfl rfl hN hT
    obtain ⟨-, -, hlo, -, -⟩ := l_final hRok hFh hFl haL hM hL
    have hW0 : 0 < wsum d init bh n := (hW X Z) ▸ psum_ew_pos hM
    have hI := hC e (attnRef d init bh n e)
    rw [centC_final hTk hW0, mul_zero, mul_zero, sub_zero, ecR_errC, acR_accA] at hI
    have hEC0 : 0 ≤ errC d init bh n e X Z T := by
      rw [← ecR_errC d init c.toReal bh n X Z e T]; exact ecR_nonneg hRok _ _ _ T
    rw [psum_wt_eq, hW X Z, ← pE_r (R := rowCtx d init c.toReal bh n X Z) (D := d.HEAD_DIM) (X := X)
      (Z := Z) rfl] at hlo
    have hFpd : Fp d.HEAD_DIM d.N_CTX X Z = Flo d.HEAD_DIM d.N_CTX X Z - ((aL d.N_CTX : ℚ) : ℝ) *
        pE d.HEAD_DIM X Z := rfl
    rw [← pE_r (R := rowCtx d init c.toReal bh n X Z) (D := d.HEAD_DIM) (X := X) (Z := Z) rfl] at hFp
    rw [← hFpd] at hlo hFp
    have hFp0 : 0 < Fp d.HEAD_DIM d.N_CTX X Z := lt_of_lt_of_le (by norm_num) hFp
    set q2 := (2 : ℝ) ^ (-m.toReal) with hq2
    have hq20 : 0 < q2 := Real.rpow_pos_of_pos (by norm_num) _
    have hkey : q2 * errC d init bh n e X Z T ≤
        errC d init bh n e X Z T / wsum d init bh n / Fp d.HEAD_DIM d.N_CTX X Z * l.toReal := by
      rw [div_div, div_mul_eq_mul_div, le_div_iff₀ (mul_pos hW0 hFp0)]
      have := mul_le_mul_of_nonneg_left hlo hEC0
      linarith [show errC d init bh n e X Z T * (Fp d.HEAD_DIM d.N_CTX X Z * (q2 * wsum d init bh n)) =
        q2 * errC d init bh n e X Z T * (wsum d init bh n * Fp d.HEAD_DIM d.N_CTX X Z) by ring]
    have hE' : |(acc e).toReal - attnRef d init bh n e * l.toReal| ≤
        (errC d init bh n e X Z T / wsum d init bh n / Fp d.HEAD_DIM d.N_CTX X Z + accA d init bh n e X Z T) *
          l.toReal + 0 := by
      linarith [add_mul (errC d init bh n e X Z T / wsum d init bh n / Fp d.HEAD_DIM d.N_CTX X Z)
        (accA d init bh n e X Z T) l.toReal]
    have hl : 1 / 2 ≤ l.toReal := (lminB_half hTle).trans hlm
    have hr : |attnRef d init bh n e| ≤ 57344 := by
      rw [← hSx X Z e]; exact ref_le hM (veOf_le hH e)
    obtain ⟨hf, hb⟩ := final_v hdivF hcvt (hA e).1 hL.1 hl ((hA e).2.trans (crA_le hTle)) hr le_rfl hE'
      (ho' e) (ho8 e)
    refine ⟨hf, ?_⟩
    rw [mul_zero, add_zero] at hb
    unfold tolO
    rw [hTn]
    exact hb
  have hOM : HeadOk d init bh → ∀ X Z : ℚ, X ≤ Xh d.HEAD_DIM →
      (∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ X) →
      (∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤ Z) →
      Mv.isFinite = true ∧ |Mv.toReal - lseRef d init bh n| ≤
        tolM d.HEAD_DIM d.N_CTX X Z + (u32 : ℝ) * |lseRef d init bh n| := by
    intro hH X Z hX hXk hZk
    obtain ⟨hRok, hM, hL, -⟩ := hfin hH X Z hX hXk hZk
    obtain ⟨hFh, hFl, haL, hFp, hFh2⟩ :=
      model_factors (D := d.HEAD_DIM) (N := d.N_CTX) (X := X) (Z := Z) hRok rfl rfl hN hT
    obtain ⟨hf, hb⟩ := final_M hlog hRok hTle hFh hFl haL hFp hFh2 hM hL hlg hMv
    refine ⟨hf, ?_⟩
    rw [hW, ← pE_r (R := rowCtx d init c.toReal bh n X Z) (D := d.HEAD_DIM) (X := X) (Z := Z) rfl] at hb
    unfold tolM lgB Fp lseRef
    rw [u32_r]
    exact hb
  have hall : HeadOk d init bh → ∀ j ∈ keys d n, sigma d * scoreLift d init bh n j ≤ Xh d.HEAD_DIM :=
    fun hH j hj => hH.2.2 n j hn (keys_lt hj)
  -- a bound on the row's scaled score errors
  have hZ0 : ∀ j ∈ keys d n, sigma d * scoreErr d init bh n j ≤
      ∑ i ∈ keys d n, |sigma d * scoreErr d init bh n i| := fun j hj =>
    (le_abs_self _).trans (Finset.single_le_sum (f := fun i => |sigma d * scoreErr d init bh n i|)
      (fun i _ => abs_nonneg _) hj)
  -- the second bound, at `X = Xh HEAD_DIM`
  have hOv : ∀ e : Fin d.HEAD_DIM, HeadOk d init bh → ∀ K : ℝ,
      (∀ j ∈ keys d n, |(vEff d init bh j e : ℝ) - attnRef d init bh n e| ≤ K) →
      |(o8 e).toReal - attnRef d init bh n e| ≤ tolV d init bh n e K := by
    intro e hH K hK
    obtain ⟨-, hM, hL, hA, hlm, -, hE⟩ := hfin hH _ _ le_rfl (hall hH) hZ0
    have hK0 : 0 ≤ K := (abs_nonneg _).trans (hK 0 (mem_keys.mpr (by unfold alOf; simp; omega)))
    have hKv : ∀ j, (rowCtx d init c.toReal bh n (Xh d.HEAD_DIM)
        (∑ i ∈ keys d n, |sigma d * scoreErr d init bh n i|)).al j = true →
        |veOf d init bh e j - attnRef d init bh n e| ≤ K := fun j hj => by
      have hj' : j ∈ keys d n := mem_keys.mpr hj
      rw [veOf_key e hj']; exact hK j hj'
    have hl : 1 / 2 ≤ l.toReal := (lminB_half hTle).trans hlm
    have hr : |attnRef d init bh n e| ≤ 57344 := by
      rw [← hSx (Xh d.HEAD_DIM) (∑ i ∈ keys d n, |sigma d * scoreErr d init bh n i|) e]
      exact ref_le hM (veOf_le hH e)
    obtain ⟨-, hb⟩ := final_v hdivF hcvt (hA e).1 hL.1 hl ((hA e).2.trans (crA_le hTle)) hr
      (magD_nonneg hK0 (abs_nonneg _) T) (hE e (attnRef d init bh n e) K hK0 hKv) (ho' e) (ho8 e)
    unfold tolV; rw [hTn]; exact hb
  refine ⟨fun e hH => ⟨(hOo e hH _ _ le_rfl (hall hH) hZ0).1, fun X Z hX hXk hZk => (hOo e hH X Z hX hXk hZk).2,
      hOv e hH⟩,
    fun hH => ⟨(hOM hH _ _ le_rfl (hall hH) hZ0).1, fun X Z hX hXk hZk => (hOM hH X Z hX hXk hZk).2⟩⟩

/-! ## The end of a program (lines 241–245) -/

/-- **Lines 241–245** of program `i` (tile `(sm, bh)`), from every row's state after the `T` blocks
that hold all its keys: the stores of `M` and `o` leave `Qp`. -/
theorem safe_epi {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hdivF : Rules.Num.DivFull P) (hdivT : Rules.Num.DivTotal P .binary32) (hlog : Rules.Num.Log2F32 P)
    (hlt : Rules.Num.MathTotal P .log2 .binary32) (hcvt : Rules.Num.CvtSatF32E5m2 P)
    (hcvtt : Rules.Num.CvtTotal P Format.binary32 Format.e5m2 .rn) {i sm bh : ℕ} (hi : i < (launch d).size)
    (hsm : 128 * sm + 128 ≤ d.N_CTX) (hrow : bh * d.N_CTX + 128 * sm = 128 * i)
    {c : F32} (hc : CFacts d c) {T : ℕ} (hTn : ∀ r : Fin 128, nblk d (128 * sm + r) = T) (hT0 : 0 < T)
    (hT : T ≤ nblocks d.N_CTX)
    (hTk : ∀ (r : Fin 128) (j : ℕ), alOf d (128 * sm + r) j = true → j < 64 * T)
    {acc : Tensor [128, d.HEAD_DIM] F32} {l m : Tensor [128] F32} (hinv : StInv d init c sm bh T acc l m)
    {mp : Tensor [128] (Ptr .f32)}
    (hmp : ∀ r : Fin 128, mp.get (r, ()) = d.M.ptr ((bh * d.N_CTX + (128 * sm + r) : ℕ) : ℤ))
    {qo : I32} (hqo : qo.toInt = ((128 * i : ℕ) : ℤ)) :
    Safe ((launch d).env P) (Rd d) (Wr d i) (fun _ m' => Qp d init sm bh m') init (do
      let lg ← tl.log2 l
      let mv ← tl.add m lg
      let o' ← tl.div acc (l.col d.HEAD_DIM)
      tl.store mp mv
      let o8 : Tensor [128, d.HEAD_DIM] F8E5 ← tl.cvt8 o'
      tl.desc_store (descOf d d.o 128) [qo, IntV.ofInt 0] o8) := by
  have hO := tile_le hd hi
  have hMl := hd.lens.2.2.2.2
  have hoM := hd.disj.2.2.2.1
  -- line 241
  refine safe_math (fun _ => hlt _) fun lg hlg => ?_
  refine safe_add fun mv hmv => ?_
  -- line 242
  refine safe_div (fun _ => hdivT _ _) fun o' ho' => ?_
  -- line 244
  have himg : Buf.img (fun ri : Idx [128] => bh * d.N_CTX + (128 * sm + ri.1)) (fun _ => true) ⊆ mTile i := by
    rintro _ ⟨⟨r, ⟨⟩⟩, -, rfl⟩
    exact ⟨r, r.2, by simp only; omega⟩
  have hMsub : d.M.elems (Buf.img (fun ri : Idx [128] => bh * d.N_CTX + (128 * sm + ri.1)) (fun _ => true)) ⊆
      d.M.elems (below (rows d)) := d.M.elems_mono (himg.trans (mTile_sub hd hi))
  refine safe_store_buf d.M (g := fun ri : Idx [128] => bh * d.N_CTX + (128 * sm + ri.1)) (en := fun _ => true)
    (Buf.addrs_of_get_all d.M fun ⟨r, ⟨⟩⟩ => hmp r) ?_ ?_ hpre.2.2.2.2
    (fun x hx => Or.inr (d.M.elems_mono himg hx)) fun m1 hm1 hfr1 hpr1 => ?_
  · rintro ⟨r, ⟨⟩⟩ -
    show bh * d.N_CTX + (128 * sm + r) < d.M.len
    have := r.2; omega
  · rintro ⟨a, ⟨⟩⟩ ⟨b, ⟨⟩⟩ - - h
    have : a = b := Fin.ext (by simp only at h; omega)
    rw [this]
  -- line 245
  refine safe_cvt8 f32_e5m2_ok (fun _ => hcvtt _) fun o8 ho8 => ?_
  rw [← bind_pure (tl.desc_store (descOf d d.o 128) [qo, IntV.ofInt 0] o8)]
  have hov : d.o.Valid m1 := Buf.valid_of_agree hpre.2.2.2.1 fun x hx => hpr1 x (hpre.2.2.2.1.2.2.2 x hx)
  refine safe_store_rows hd d.o hd.align.2.2.2 hd.lens.2.2.2.1 hov hqo hO ?_ fun m2 hm2 hfr2 _ => ?_
  · intro x hx
    refine ⟨Or.inl hx, fun hxM => ?_⟩
    exact Set.disjoint_left.mp hoM (d.o.elems_mono (oTile_sub hd hi) hx) (hMsub hxM)
  refine safe_ret fun r => ?_
  have hn : 128 * sm + r < d.N_CTX := by have := r.2; omega
  obtain ⟨hOo, hOM⟩ := row_out hd hdivF hcvt hlog hc hn (hTn r) hT0 hT (hTk r) (hinv r) (hlg (r, ()))
    (hmv (r, ())) (o' := fun e => o'.get2 r e) (fun e => ho' (r, e, ())) (o8 := fun e => o8.get2 r e)
    (fun e => ho8 (r, e, ()))
  refine ⟨fun e => ⟨o8.get2 r e, ?_, hOo e⟩, mv.get (r, ()), ?_, hOM⟩
  · have hidx : idx d bh (128 * sm + r) e = (128 * i + r) * d.HEAD_DIM + e := by
      unfold idx; rw [← Nat.add_assoc, hrow]
    rw [hidx]; exact hm2 r e
  · refine (Mem.read_congr fun x hx => hfr2 x fun hxo => ?_).trans (hm1 (r, ()) rfl)
    have hmem : bh * d.N_CTX + (128 * sm + r) ∈ below (rows d) := by
      show _ < rows d; have := r.2; omega
    exact Set.disjoint_left.mp hoM (d.o.elems_mono (oTile_sub hd hi) hxo) (d.M.span_sub_elems hmem hx)

/-! ## A program -/

theorem offs_m_val {sm : ℕ} (hsm : 128 * sm + 128 < 2 ^ 31) (r : Fin 128) :
    (((IntV.ofNat sm : I32) * (IntV.ofNat 128 : I32) + tl.arange 0 128).get1 r).toInt =
      ((128 * sm + r : ℕ) : ℤ) := by
  show ((IntV.ofNat sm : I32) * IntV.ofNat 128 + IntV.ofInt (((0 : ℕ) : ℤ) + ((r : ℕ) : ℤ))).toInt = _
  have h1 : (IntV.ofInt (((0 : ℕ) : ℤ) + ((r : ℕ) : ℤ)) : I32).toInt = ((r : ℕ) : ℤ) := by
    rw [IntV.I32.toInt_ofInt (by have := r.2; constructor <;> push_cast <;> omega)]; push_cast; ring
  rw [IntV.I32.add_nat (IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)) h1
    (by have := r.2; omega)]
  congr 1; ring

theorem offs_n_val (jj : Fin 64) : ((tl.arange 0 64).get1 jj).toInt = ((jj : ℕ) : ℤ) := by
  show (IntV.ofInt (((0 : ℕ) : ℤ) + ((jj : ℕ) : ℤ)) : I32).toInt = _
  rw [IntV.I32.toInt_ofInt (by have := jj.2; constructor <;> push_cast <;> omega)]; push_cast; ring

/-- **One program** (lines 189–245): program `i` reads only `q`, `k`, `v`, writes only its tile of
`o` and `M`, and leaves them as `Post` asks (`Qp`). -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hdot : Rules.Num.DotE5m2Chunk P) (hdtot : Rules.Num.DotTotal P cfg8) (hexp : Rules.Num.Exp2F32 P)
    (hni : Rules.Num.Exp2F32NegInf P) (hext : Rules.Num.MathTotal P .exp2 .binary32)
    (hcvt : Rules.Num.CvtSatF32E5m2 P) (hcvtt : Rules.Num.CvtTotal P Format.binary32 Format.e5m2 .rn)
    (hlog : Rules.Num.Log2F32 P) (hlt : Rules.Num.MathTotal P .log2 .binary32)
    (hdivF : Rules.Num.DivFull P) (hdivT : Rules.Num.DivTotal P .binary32)
    {i : ℕ} (hi : i < (launch d).size) :
    Safe ((launch d).env P) (Rd d) (Wr d i) (fun _ m => Qp d init (pm d i) (pbh d i) m) init
      ((model d).body ((launch d).ctx i)) := by
  have hsm := pm_row hd i
  have hbh := pbh_lt hd hi
  have hrow := row0 hd i
  simp only [model, attn_fwd]
  generalize hS : (if d.causal = true then 3 else 1) = STAGE
  simp only [tl.program_id, pid0, pid1 hd hi]
  generalize pm d i = sm at hsm hrow ⊢
  generalize pbh d i = bh at hbh hrow ⊢
  have hrows := rows_lt hd
  have hzh := hd.zh_lt
  have hN := hd.n_le
  have hNm := hd.n_mul
  have hZH := hd.zh_pos
  have hO := tile_le hd hi
  have hH1 : d.H ≤ d.Z * d.H := Nat.le_mul_of_pos_left _ hZH.1
  have hbhN : bh * d.N_CTX + d.N_CTX ≤ rows d := by
    unfold rows
    calc bh * d.N_CTX + d.N_CTX = (bh + 1) * d.N_CTX := by ring
      _ ≤ (d.Z * d.H) * d.N_CTX := Nat.mul_le_mul_right _ hbh
  have hNH : d.N_CTX * d.H ≤ rows d := by
    unfold rows
    calc d.N_CTX * d.H = 1 * d.H * d.N_CTX := by ring
      _ ≤ d.Z * d.H * d.N_CTX := Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hZH.1)
  -- lines 191–192
  refine safe_idiv1 (by rw [i32_nat (by omega)]; positivity) (by rw [i32_nat (by omega)]; exact_mod_cast hZH.2)
    fun z hz => ?_
  refine safe_irem1 (by rw [i32_nat (by omega)]; positivity) (by rw [i32_nat (by omega)]; exact_mod_cast hZH.2)
    fun h hh => ?_
  rw [i32_nat (by omega), i32_nat (by omega)] at hz hh
  -- lines 195–206
  refine safe_maybe_desc hd d.q 128 ?_
  refine safe_maybe_descV hd ?_
  refine safe_maybe_desc hd d.k 64 ?_
  refine safe_maybe_desc hd d.o 128 ?_
  -- lines 214–215
  refine safe_sub fun mi hmi => ?_
  refine safe_add fun li hli => ?_
  -- line 219
  refine safe_mul fun cs hcs => ?_
  have hc : CFacts d cs.item := qk_scale_facts hd (hcs ())
  -- lines 208–209
  have hdm := Nat.div_add_mod bh d.H
  have e1 : bh / d.H * (d.N_CTX * d.H) + bh % d.H * d.N_CTX = bh * d.N_CTX := by
    calc bh / d.H * (d.N_CTX * d.H) + bh % d.H * d.N_CTX = (d.H * (bh / d.H) + bh % d.H) * d.N_CTX := by ring
      _ = bh * d.N_CTX := by rw [hdm]
  have hzv : z.item.toInt = ((bh / d.H : ℕ) : ℤ) := by rw [hz]; push_cast; rfl
  have hhv : h.item.toInt = ((bh % d.H : ℕ) : ℤ) := by rw [hh]; push_cast; rfl
  have hNHv : (IntV.ofNat d.N_CTX * IntV.ofNat d.H : I32).toInt = ((d.N_CTX * d.H : ℕ) : ℤ) :=
    IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega)) (by omega)
  have hle1 : bh / d.H * (d.N_CTX * d.H) ≤ bh * d.N_CTX := e1 ▸ Nat.le_add_right _ _
  have hle2 : bh % d.H * d.N_CTX ≤ bh * d.N_CTX := e1 ▸ Nat.le_add_left _ _
  have hoy : (z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) + h.item * IntV.ofNat d.N_CTX : I32).toInt =
      ((bh * d.N_CTX : ℕ) : ℤ) := by
    rw [IntV.I32.add_nat (IntV.I32.mul_nat hzv hNHv (by omega)) (IntV.I32.mul_nat hhv (i32_nat (by omega)) (by omega))
      (by omega), e1]
  have hqo : (z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) + h.item * IntV.ofNat d.N_CTX +
      IntV.ofNat sm * IntV.ofNat 128 : I32).toInt = ((128 * i : ℕ) : ℤ) := by
    rw [IntV.I32.add_nat hoy (IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)) (by omega)]
    congr 1; omega
  -- line 221
  refine safe_load_rows hd d.q hd.align.1 hd.lens.1 hpre.1 (fun x hx => Or.inl (Or.inl hx)) hqo hO
    fun q hq => ?_
  have hQ : QReads d init sm bh q := fun r e => by
    have hidx : idx d bh (128 * sm + r) e = (128 * i + r) * d.HEAD_DIM + e := by
      unfold idx; rw [← Nat.add_assoc, hrow]
    rw [hidx]; exact hq r e
  have hcall : CallOk d init sm bh cs.item q (z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) +
      h.item * IntV.ofNat d.N_CTX) (IntV.ofNat sm) (IntV.ofNat d.N_CTX)
      ((IntV.ofNat sm : I32) * IntV.ofNat 128 + tl.arange 0 128) (tl.arange 0 64) :=
    { sm_le := hsm, bh_lt := hbh, c_ok := hc, q_ok := hQ, oy := hoy, start := i32_nat (by omega),
      n := i32_nat (by omega), om := offs_m_val (by omega), on := offs_n_val }
  have hinv0 : StInv d init cs.item sm bh 0 (tl.zeros [128, d.HEAD_DIM] .f32) li mi := by
    intro r _ X Z _ _ _
    obtain ⟨hl1, hl2⟩ := l_init_eq (hli (r, ()))
    have hl2' : (li.get1 r).toReal = 1 := hl2
    exact Or.inl ⟨rfl, m_init_eq (hmi (r, ())), hl1, hl2',
      fun e => ⟨zero32.1, by rw [toReal_def]; exact (congrArg (fun x : ℚ => (x : ℝ)) zero32.2).trans Rat.cast_zero⟩⟩
  have hmp : ∀ r : Fin 128, (d.M.ptr 0 + (IntV.ofNat bh : I32) * IntV.ofNat d.N_CTX +
      ((IntV.ofNat sm : I32) * IntV.ofNat 128 + tl.arange 0 128)).get (r, ()) =
      d.M.ptr ((bh * d.N_CTX + (128 * sm + r) : ℕ) : ℤ) := fun r => by
    have h1 := offs_m_val (sm := sm) (by omega) r
    have h2 : ((IntV.ofNat bh : I32) * IntV.ofNat d.N_CTX).toInt = ((bh * d.N_CTX : ℕ) : ℤ) :=
      IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega)) (by omega)
    show ((d.M.ptr 0).offset ((IntV.ofNat bh : I32) * IntV.ofNat d.N_CTX).toInt).offset
      (((IntV.ofNat sm : I32) * IntV.ofNat 128 + tl.arange 0 128).get1 r).toInt = _
    rw [h1, h2, Buf.ptr_offset, Buf.ptr_offset]
    congr 1; push_cast; ring
  -- lines 225–239
  by_cases hcz : d.causal = true
  · rw [ite_eq_left hcz] at hS
    subst hS
    rw [ite_eq_left (show (3 : ℕ) &&& 1 ≠ 0 by decide)]
    refine safe_inner1 hd hpre hdot hdtot hexp hni hext hcvt hcvtt subset_rfl hcall hinv0 fun acc l m hst => ?_
    rw [ite_eq_left (show (3 : ℕ) &&& 2 ≠ 0 by decide)]
    refine safe_inner2 hd hcz hpre hdot hdtot hexp hni hext hcvt hcvtt subset_rfl hcall hst fun acc l m hst => ?_
    refine safe_epi hd hpre hdivF hdivT hlog hlt hcvt hcvtt hi hsm hrow hc (fun r => ?_) (by omega)
      (by unfold nblocks; omega) (fun r j hj => ?_) hst hmp hqo
    · unfold nblk; rw [ite_eq_left hcz]
      have := r.2; omega
    · unfold alOf at hj
      simp only [hcz, Bool.true_eq_false, false_or, decide_eq_true_eq] at hj
      have := r.2; omega
  · rw [ite_eq_right hcz] at hS
    subst hS
    rw [ite_eq_left (show (1 : ℕ) &&& 1 ≠ 0 by decide)]
    have hcf : d.causal = false := by simpa using hcz
    refine safe_inner3 hd hcf hpre hdot hdtot hexp hni hext hcvt hcvtt subset_rfl hcall hinv0 fun acc l m hst => ?_
    rw [ite_eq_right (show ¬((1 : ℕ) &&& 2 ≠ 0) by decide)]
    simp only [pure_bind]
    refine safe_epi hd hpre hdivF hdivT hlog hlt hcvt hcvtt hi hsm hrow hc (fun r => ?_) (by omega)
      (by unfold nblocks; omega) (fun r j hj => ?_) hst hmp hqo
    · unfold nblk; rw [ite_eq_right hcz]
    · unfold alOf at hj
      simp only [decide_eq_true_eq] at hj
      omega

end Kernels.AttnFwdE5m2
