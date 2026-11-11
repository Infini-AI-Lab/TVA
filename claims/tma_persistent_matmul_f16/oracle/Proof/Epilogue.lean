import Proof.Setup

/-!
# `tma_persistent_matmul_f16`: The epilogue: the two stored halves of a tile, the conversion, and `Post` from the programs' tiles.
-/

namespace Claims.TmaPersistentMatmul

open TVA

/-!
## The descriptors and the epilogue tensors

Two kinds of facts, all elementary:

* the two `[128, 128]` halves `Tensor.split` cuts out of the `[128, 256]` accumulator are, after
  the `reshape [128, 2, 128]` and the `permute [0, 2, 1]`, the columns `[0, 128)` and
  `[128, 256)` of the accumulator;
* the descriptors `aDesc`, `bDesc`, `cDesc` meet Triton's requirements on an access, their
  `elemAt` and `inbAt` are the storage index and the bounds test of an element, and the lanes of
  one access are the elements of the index sets `aIdx`, `bIdx`, `cIdx` (or of one tile `tileIdx`).
-/


/-- A lane of a two-offset block is inside a descriptor of shape `[A, B]` iff both coordinates are
in range. -/
private theorem inbAt_two {δ : DType} (D : TensorDesc δ) {A B : Nat}
    (hshape : D.shape = [A, B]) (o0 o1 : Int) {bs : Shape} (i : Idx bs) :
    D.inbAt (TensorDesc.coord [o0, o1] i) = true ↔
      (0 ≤ o0 + (i.coord 0 : Int) ∧ o0 + (i.coord 0 : Int) < (A : Int) ∧
       0 ≤ o1 + (i.coord 1 : Int) ∧ o1 + (i.coord 1 : Int) < (B : Int)) := by
  simp only [TensorDesc.inbAt]
  rw [decide_eq_true_eq]
  simp only [TensorDesc.coord, TensorDesc.rank, hshape, List.length_cons, List.length_nil]
  constructor
  · intro h
    have h0 := h 0 (by omega)
    have h1 := h 1 (by omega)
    simpa [List.getD_cons_zero, List.getD_cons_succ, List.getD_nil] using
      ⟨h0.1, h0.2, h1.1, h1.2⟩
  · rintro ⟨h0, h1, h2, h3⟩ k hk
    interval_cases k
    · simpa [List.getD_cons_zero] using ⟨h0, h1⟩
    · simpa [List.getD_cons_zero, List.getD_cons_succ, List.getD_nil] using ⟨h2, h3⟩

/-! ### The halves of the accumulator -/

/-- Element `(r, col)` of the first stored half of the accumulator is `A[r, col]`. -/
theorem acc0_get2 (A : Tensor [128, 256] F32) (r : Fin 128) (col : Fin 128) :
    (Tensor.split (s := [128, 128])
        ((A.reshape [128, 2, 128] (by norm_num)).permute [0, 2, 1])).1.get2 r col
      = A.get2 r ⟨col.val, by omega⟩ := by
  simp only [Tensor.split, Tensor.get2]
  have h1 := Tensor.get_permute (x := A.reshape [128, 2, 128] (by norm_num)) (dims := [0, 2, 1])
      (by decide) (o := ((r, (col, (0, ()))) : Idx [128, 128, 2])) (i := (r, (0, (col, ()))))
      (by intro k hk
          simp only [List.length_cons, List.length_nil] at hk
          interval_cases k <;> rfl)
  refine h1.trans ?_
  have hpos : Idx.pos [128, 2, 128] (r, (0, (col, ()))) =
      Idx.pos [128, 256] (r, (⟨col.val, by omega⟩, ())) := by
    simp [Idx.pos]
  have key : ∀ (hh : Idx.pos [128, 2, 128] (r, (0, (col, ()))) < 128 * 256),
      Idx.ofPos [128, 256] (Idx.pos [128, 2, 128] (r, (0, (col, ())))) hh
        = ((r, (⟨col.val, by omega⟩, ())) : Idx [128, 256]) := by
    intro hh
    exact (Idx.ofPos_congr (s := [128, 256]) hpos hh (Idx.pos_lt [128, 256] _)).trans
      (Idx.ofPos_pos [128, 256] _ _)
  exact congrArg A.get (key _)

/-- Element `(r, col)` of the second stored half of the accumulator is `A[r, 128 + col]`. -/
theorem acc1_get2 (A : Tensor [128, 256] F32) (r : Fin 128) (col : Fin 128) :
    (Tensor.split (s := [128, 128])
        ((A.reshape [128, 2, 128] (by norm_num)).permute [0, 2, 1])).2.get2 r col
      = A.get2 r ⟨128 + col.val, by omega⟩ := by
  simp only [Tensor.split, Tensor.get2]
  have h1 := Tensor.get_permute (x := A.reshape [128, 2, 128] (by norm_num)) (dims := [0, 2, 1])
      (by decide) (o := ((r, (col, (1, ()))) : Idx [128, 128, 2])) (i := (r, (1, (col, ()))))
      (by intro k hk
          simp only [List.length_cons, List.length_nil] at hk
          interval_cases k <;> rfl)
  refine h1.trans ?_
  have hpos : Idx.pos [128, 2, 128] (r, (1, (col, ()))) =
      Idx.pos [128, 256] (r, (⟨128 + col.val, by omega⟩, ())) := by
    simp [Idx.pos]
  have key : ∀ (hh : Idx.pos [128, 2, 128] (r, (1, (col, ()))) < 128 * 256),
      Idx.ofPos [128, 256] (Idx.pos [128, 2, 128] (r, (1, (col, ())))) hh
        = ((r, (⟨128 + col.val, by omega⟩, ())) : Idx [128, 256]) := by
    intro hh
    exact (Idx.ofPos_congr (s := [128, 256]) hpos hh (Idx.pos_lt [128, 256] _)).trans
      (Idx.ofPos_pos [128, 256] _ _)
  exact congrArg A.get (key _)

/-! ### The descriptor of `a` -/

/-- The host descriptor of `a` is valid. -/
theorem aDesc_valid {d : Inst} (hd : d.ok) : (aDesc d).Valid := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, rfl, ?_, ?_, ?_⟩
  · simp [TensorDesc.rank, aDesc]
  · simp [TensorDesc.rank, aDesc]
  · simp [TensorDesc.rank, aDesc]
  · simp [TensorDesc.rank, aDesc]
  · intro k hk
    simp only [TensorDesc.rank, aDesc, List.length_cons, List.length_nil] at hk
    interval_cases k
    · simp only [aDesc, List.getD_cons_zero]
      exact ⟨hd.pos.1, by have := hd.M_le; omega⟩
    · simp only [aDesc, List.getD_cons_zero, List.getD_cons_succ]
      refine ⟨hd.pos.2.2, ?_⟩
      have hp : (2 : Nat) ^ 13 < 2 ^ 31 := by norm_num
      have := hd.K_le
      omega
  · simp only [aDesc, Buf.ptr_addr]
    simpa using hd.base_al.1
  · intro k hk
    simp only [TensorDesc.rank, aDesc, List.length_cons, List.length_nil] at hk
    have hk0 : k = 0 := by omega
    subst hk0
    obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hd.ld8.1
    simp only [aDesc, List.getD_cons_zero, DType.bytes]
    rw [hq]
    omega
  · simp [aDesc]
  · intro h
    simp [aDesc] at h

/-- The storage index of a lane of a block of `a`. -/
theorem aDesc_elemAt (d : Inst) (o0 o1 : Int) (i : Idx [128, 64]) :
    (aDesc d).elemAt (TensorDesc.coord [o0, o1] i)
      = (o0 + (i.coord 0 : Int)) * (d.lda : Int) + (o1 + (i.coord 1 : Int)) :=
  elemAt_two (aDesc d) rfl rfl o0 o1 i

/-- A lane of a block of `a` is inside the matrix iff both coordinates are in range. -/
theorem aDesc_inbAt (d : Inst) (o0 o1 : Int) (i : Idx [128, 64]) :
    (aDesc d).inbAt (TensorDesc.coord [o0, o1] i) = true ↔
      (0 ≤ o0 + (i.coord 0 : Int) ∧ o0 + (i.coord 0 : Int) < (d.M : Int) ∧
       0 ≤ o1 + (i.coord 1 : Int) ∧ o1 + (i.coord 1 : Int) < (d.K : Int)) :=
  inbAt_two (aDesc d) rfl o0 o1 i

/-- A load of a `128 × 64` block of `a` meets Triton's requirements. -/
theorem aDesc_access {d : Inst} (hd : d.ok) (o0 o1 : Int) (h : (o1 * 2) % 16 = 0) :
    (aDesc d).Access [o0, o1] [128, 64] := by
  refine ⟨aDesc_valid hd, rfl, rfl, rfl, ?_⟩
  show (o1 * (DType.f16.bytes : Int)) % 16 = 0
  simpa [DType.bytes] using h

/-- A lane of `a` inside the matrix has its element inside the buffer. -/
theorem aDesc_len {d : Inst} (hd : d.ok) (o0 o1 : Int) (i : Idx [128, 64])
    (hinb : (aDesc d).inbAt (TensorDesc.coord [o0, o1] i) = true) :
    ((aDesc d).elemAt (TensorDesc.coord [o0, o1] i)).toNat < d.a.len := by
  have h := (inbAt_two (aDesc d) rfl o0 o1 i).mp hinb
  obtain ⟨h0, h1, h2, h3⟩ := h
  rw [elemAt_toNat_two (aDesc d) rfl rfl o0 o1 i h0 h2]
  exact hd.a_len _ _ (by rw [Int.toNat_lt h0]; exact h1) (by rw [Int.toNat_lt h2]; exact h3)

/-- The elements a block of `a` touches lie among the elements of `a`'s matrix. -/
theorem aDesc_img_subset (d : Inst) (o0 o1 : Int) :
    Buf.img (fun i : Idx [128, 64] => ((aDesc d).elemAt (TensorDesc.coord [o0, o1] i)).toNat)
      (fun i => (aDesc d).inbAt (TensorDesc.coord [o0, o1] i)) ⊆ aIdx d := by
  rintro e ⟨i, hinb, rfl⟩
  have h := (inbAt_two (aDesc d) rfl o0 o1 i).mp hinb
  obtain ⟨h0, h1, h2, h3⟩ := h
  exact ⟨(o0 + (i.coord 0 : Int)).toNat, (o1 + (i.coord 1 : Int)).toNat,
    by rw [Int.toNat_lt h0]; exact h1,
    by rw [Int.toNat_lt h2]; exact h3,
    elemAt_toNat_two (aDesc d) rfl rfl o0 o1 i h0 h2⟩

/-! ### The descriptor of `b` -/

/-- The host descriptor of `b` is valid. -/
theorem bDesc_valid {d : Inst} (hd : d.ok) : (bDesc d).Valid := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, rfl, ?_, ?_, ?_⟩
  · simp [TensorDesc.rank, bDesc]
  · simp [TensorDesc.rank, bDesc]
  · simp [TensorDesc.rank, bDesc]
  · simp [TensorDesc.rank, bDesc]
  · intro k hk
    simp only [TensorDesc.rank, bDesc, List.length_cons, List.length_nil] at hk
    interval_cases k
    · simp only [bDesc, List.getD_cons_zero]
      exact ⟨hd.pos.2.1, by have := hd.N_le; omega⟩
    · simp only [bDesc, List.getD_cons_zero, List.getD_cons_succ]
      refine ⟨hd.pos.2.2, ?_⟩
      have hp : (2 : Nat) ^ 13 < 2 ^ 31 := by norm_num
      have := hd.K_le
      omega
  · simp only [bDesc, Buf.ptr_addr]
    simpa using hd.base_al.2.1
  · intro k hk
    simp only [TensorDesc.rank, bDesc, List.length_cons, List.length_nil] at hk
    have hk0 : k = 0 := by omega
    subst hk0
    obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hd.ld8.2.1
    simp only [bDesc, List.getD_cons_zero, DType.bytes]
    rw [hq]
    omega
  · simp [bDesc]
  · intro h
    simp [bDesc] at h

/-- The storage index of a lane of a block of `b`. -/
theorem bDesc_elemAt (d : Inst) (o0 o1 : Int) (i : Idx [256, 64]) :
    (bDesc d).elemAt (TensorDesc.coord [o0, o1] i)
      = (o0 + (i.coord 0 : Int)) * (d.ldb : Int) + (o1 + (i.coord 1 : Int)) :=
  elemAt_two (bDesc d) rfl rfl o0 o1 i

/-- A lane of a block of `b` is inside the matrix iff both coordinates are in range. -/
theorem bDesc_inbAt (d : Inst) (o0 o1 : Int) (i : Idx [256, 64]) :
    (bDesc d).inbAt (TensorDesc.coord [o0, o1] i) = true ↔
      (0 ≤ o0 + (i.coord 0 : Int) ∧ o0 + (i.coord 0 : Int) < (d.N : Int) ∧
       0 ≤ o1 + (i.coord 1 : Int) ∧ o1 + (i.coord 1 : Int) < (d.K : Int)) :=
  inbAt_two (bDesc d) rfl o0 o1 i

/-- A load of a `256 × 64` block of `b` meets Triton's requirements. -/
theorem bDesc_access {d : Inst} (hd : d.ok) (o0 o1 : Int) (h : (o1 * 2) % 16 = 0) :
    (bDesc d).Access [o0, o1] [256, 64] := by
  refine ⟨bDesc_valid hd, rfl, rfl, rfl, ?_⟩
  show (o1 * (DType.f16.bytes : Int)) % 16 = 0
  simpa [DType.bytes] using h

/-- A lane of `b` inside the matrix has its element inside the buffer. -/
theorem bDesc_len {d : Inst} (hd : d.ok) (o0 o1 : Int) (i : Idx [256, 64])
    (hinb : (bDesc d).inbAt (TensorDesc.coord [o0, o1] i) = true) :
    ((bDesc d).elemAt (TensorDesc.coord [o0, o1] i)).toNat < d.b.len := by
  have h := (inbAt_two (bDesc d) rfl o0 o1 i).mp hinb
  obtain ⟨h0, h1, h2, h3⟩ := h
  rw [elemAt_toNat_two (bDesc d) rfl rfl o0 o1 i h0 h2]
  exact hd.b_len _ _ (by rw [Int.toNat_lt h0]; exact h1) (by rw [Int.toNat_lt h2]; exact h3)

/-- The elements a block of `b` touches lie among the elements of `b`'s matrix. -/
theorem bDesc_img_subset (d : Inst) (o0 o1 : Int) :
    Buf.img (fun i : Idx [256, 64] => ((bDesc d).elemAt (TensorDesc.coord [o0, o1] i)).toNat)
      (fun i => (bDesc d).inbAt (TensorDesc.coord [o0, o1] i)) ⊆ bIdx d := by
  rintro e ⟨i, hinb, rfl⟩
  have h := (inbAt_two (bDesc d) rfl o0 o1 i).mp hinb
  obtain ⟨h0, h1, h2, h3⟩ := h
  exact ⟨(o0 + (i.coord 0 : Int)).toNat, (o1 + (i.coord 1 : Int)).toNat,
    by rw [Int.toNat_lt h0]; exact h1,
    by rw [Int.toNat_lt h2]; exact h3,
    elemAt_toNat_two (bDesc d) rfl rfl o0 o1 i h0 h2⟩

/-! ### The descriptor of `c` -/

/-- The host descriptor of `c` is valid. -/
theorem cDesc_valid {d : Inst} (hd : d.ok) : (cDesc d).Valid := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, rfl, ?_, ?_, ?_⟩
  · simp [TensorDesc.rank, cDesc]
  · simp [TensorDesc.rank, cDesc]
  · simp [TensorDesc.rank, cDesc]
  · simp [TensorDesc.rank, cDesc]
  · intro k hk
    simp only [TensorDesc.rank, cDesc, List.length_cons, List.length_nil] at hk
    interval_cases k
    · simp only [cDesc, List.getD_cons_zero]
      exact ⟨hd.pos.1, by have := hd.M_le; omega⟩
    · simp only [cDesc, List.getD_cons_zero, List.getD_cons_succ]
      refine ⟨hd.pos.2.1, by have := hd.N_le; omega⟩
  · simp only [cDesc, Buf.ptr_addr]
    simpa using hd.base_al.2.2
  · intro k hk
    simp only [TensorDesc.rank, cDesc, List.length_cons, List.length_nil] at hk
    have hk0 : k = 0 := by omega
    subst hk0
    obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero hd.ld8.2.2
    simp only [cDesc, List.getD_cons_zero, DType.bytes]
    rw [hq]
    omega
  · simp [cDesc]
  · intro h
    simp [cDesc] at h

/-- The storage index of a lane of a block of `c`. -/
theorem cDesc_elemAt (d : Inst) (o0 o1 : Int) (i : Idx [128, 128]) :
    (cDesc d).elemAt (TensorDesc.coord [o0, o1] i)
      = (o0 + (i.coord 0 : Int)) * (d.ldc : Int) + (o1 + (i.coord 1 : Int)) :=
  elemAt_two (cDesc d) rfl rfl o0 o1 i

/-- A lane of a block of `c` is inside the matrix iff both coordinates are in range. -/
theorem cDesc_inbAt (d : Inst) (o0 o1 : Int) (i : Idx [128, 128]) :
    (cDesc d).inbAt (TensorDesc.coord [o0, o1] i) = true ↔
      (0 ≤ o0 + (i.coord 0 : Int) ∧ o0 + (i.coord 0 : Int) < (d.M : Int) ∧
       0 ≤ o1 + (i.coord 1 : Int) ∧ o1 + (i.coord 1 : Int) < (d.N : Int)) :=
  inbAt_two (cDesc d) rfl o0 o1 i

/-- A store of a `128 × 128` block of `c` meets Triton's requirements. -/
theorem cDesc_access {d : Inst} (hd : d.ok) (o0 o1 : Int) (h : (o1 * 2) % 16 = 0) :
    (cDesc d).Access [o0, o1] [128, 128] := by
  refine ⟨cDesc_valid hd, rfl, rfl, rfl, ?_⟩
  show (o1 * (DType.f16.bytes : Int)) % 16 = 0
  simpa [DType.bytes] using h

/-- A lane of `c` inside the matrix has its element inside the buffer. -/
theorem cDesc_len {d : Inst} (hd : d.ok) (o0 o1 : Int) (i : Idx [128, 128])
    (hinb : (cDesc d).inbAt (TensorDesc.coord [o0, o1] i) = true) :
    ((cDesc d).elemAt (TensorDesc.coord [o0, o1] i)).toNat < d.c.len := by
  have h := (inbAt_two (cDesc d) rfl o0 o1 i).mp hinb
  obtain ⟨h0, h1, h2, h3⟩ := h
  rw [elemAt_toNat_two (cDesc d) rfl rfl o0 o1 i h0 h2]
  exact hd.c_len _ _ (by rw [Int.toNat_lt h0]; exact h1) (by rw [Int.toNat_lt h2]; exact h3)

/-- Distinct in-bounds lanes of one block of `c` have distinct storage indices. -/
theorem cDesc_toNat_inj (d : Inst) (hd : d.ok) (o0 o1 : Int) (i j : Idx [128, 128])
    (hi : (cDesc d).inbAt (TensorDesc.coord [o0, o1] i) = true)
    (hj : (cDesc d).inbAt (TensorDesc.coord [o0, o1] j) = true)
    (h : ((cDesc d).elemAt (TensorDesc.coord [o0, o1] i)).toNat =
      ((cDesc d).elemAt (TensorDesc.coord [o0, o1] j)).toNat) :
    i = j := by
  have hi' := (inbAt_two (cDesc d) rfl o0 o1 i).mp hi
  have hj' := (inbAt_two (cDesc d) rfl o0 o1 j).mp hj
  obtain ⟨hi0, hi1, hi2, hi3⟩ := hi'
  obtain ⟨hj0, hj1, hj2, hj3⟩ := hj'
  have hI : (o0 + (i.coord 0 : Int)).toNat < d.M := by rw [Int.toNat_lt hi0]; exact hi1
  have hK : (o1 + (i.coord 1 : Int)).toNat < d.N := by rw [Int.toNat_lt hi2]; exact hi3
  have hI' : (o0 + (j.coord 0 : Int)).toNat < d.M := by rw [Int.toNat_lt hj0]; exact hj1
  have hK' : (o1 + (j.coord 1 : Int)).toNat < d.N := by rw [Int.toNat_lt hj2]; exact hj3
  rw [elemAt_toNat_two (cDesc d) rfl rfl o0 o1 i hi0 hi2,
      elemAt_toNat_two (cDesc d) rfl rfl o0 o1 j hj0 hj2] at h
  obtain ⟨hIeq, hKeq⟩ := elem_inj hd hI hK hI' hK' h
  have hc0 : i.coord 0 = j.coord 0 := by
    have he : (o0 + (i.coord 0 : Int)) = (o0 + (j.coord 0 : Int)) := by
      rw [← Int.toNat_of_nonneg hi0, ← Int.toNat_of_nonneg hj0, hIeq]
    omega
  have hc1 : i.coord 1 = j.coord 1 := by
    have he : (o1 + (i.coord 1 : Int)) = (o1 + (j.coord 1 : Int)) := by
      rw [← Int.toNat_of_nonneg hi2, ← Int.toNat_of_nonneg hj2, hKeq]
    omega
  apply Idx.ext_coord
  intro k
  by_cases hk : k < 2
  · interval_cases k
    · exact hc0
    · exact hc1
  · have h2 : (2 : Nat) ≤ k := by omega
    rw [Idx.coord_of_ge (s := [128, 128]) (i := i) (k := k) (by simpa using h2),
        Idx.coord_of_ge (s := [128, 128]) (i := j) (k := k) (by simpa using h2)]

/-- The elements the first stored half of tile `t` touches are the first half of the tile's
elements. -/
theorem cDesc_img_tile1 (d : Inst) (t : Nat) :
    Buf.img (fun i : Idx [128, 128] => ((cDesc d).elemAt (TensorDesc.coord
          [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int)] i)).toNat)
      (fun i => (cDesc d).inbAt (TensorDesc.coord
          [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int)] i))
      ⊆ tileIdx d t := by
  rintro e ⟨i, hinb, rfl⟩
  have h := (inbAt_two (cDesc d) rfl _ _ i).mp hinb
  obtain ⟨h0, h1, h2, h3⟩ := h
  have hr : i.coord 0 < 128 := by
    have hcl := Idx.coord_lt i (k := 0) (by norm_num : 0 < [128, 128].length)
    simpa [Shape.dim_eq_getD] using hcl
  have hn : i.coord 1 < 256 := by
    have hcl := Idx.coord_lt i (k := 1) (by norm_num : 1 < [128, 128].length)
    have h128 : i.coord 1 < 128 := by simpa [Shape.dim_eq_getD] using hcl
    omega
  have hM : 128 * pidM d t + i.coord 0 < d.M := by omega
  have hN : 256 * pidN d t + i.coord 1 < d.N := by omega
  refine ⟨i.coord 0, i.coord 1, hr, hn, hM, hN, ?_⟩
  have hInt : (cDesc d).elemAt (TensorDesc.coord
        [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int)] i)
      = (((128 * pidM d t + i.coord 0) * d.ldc
          + (256 * pidN d t + i.coord 1) : Nat) : Int) := by
    rw [elemAt_two (cDesc d) rfl rfl _ _ i]
    push_cast
    ring
  dsimp only
  rw [hInt, Int.toNat_natCast]

/-- The elements the second stored half of tile `t` touches are the second half of the tile's
elements. -/
theorem cDesc_img_tile2 (d : Inst) (t : Nat) :
    Buf.img (fun i : Idx [128, 128] => ((cDesc d).elemAt (TensorDesc.coord
          [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int) + 128] i)).toNat)
      (fun i => (cDesc d).inbAt (TensorDesc.coord
          [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int) + 128] i))
      ⊆ tileIdx d t := by
  rintro e ⟨i, hinb, rfl⟩
  have h := (inbAt_two (cDesc d) rfl _ _ i).mp hinb
  obtain ⟨h0, h1, h2, h3⟩ := h
  have hr : i.coord 0 < 128 := by
    have hcl := Idx.coord_lt i (k := 0) (by norm_num : 0 < [128, 128].length)
    simpa [Shape.dim_eq_getD] using hcl
  have hn : 128 + i.coord 1 < 256 := by
    have hcl := Idx.coord_lt i (k := 1) (by norm_num : 1 < [128, 128].length)
    have h128 : i.coord 1 < 128 := by simpa [Shape.dim_eq_getD] using hcl
    omega
  have hM : 128 * pidM d t + i.coord 0 < d.M := by omega
  have hN : 256 * pidN d t + (128 + i.coord 1) < d.N := by omega
  refine ⟨i.coord 0, 128 + i.coord 1, hr, hn, hM, hN, ?_⟩
  have hInt : (cDesc d).elemAt (TensorDesc.coord
        [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int) + 128] i)
      = (((128 * pidM d t + i.coord 0) * d.ldc
          + (256 * pidN d t + (128 + i.coord 1)) : Nat) : Int) := by
    rw [elemAt_two (cDesc d) rfl rfl _ _ i]
    push_cast
    ring
  dsimp only
  rw [hInt, Int.toNat_natCast]

/-- The indices `aIdx d` are inside the buffer `a`. -/
theorem aIdx_subset_bytes {d : Inst} (hd : d.ok) : d.a.elems (aIdx d) ⊆ d.a.bytes := by
  apply Buf.elems_sub_bytes
  rintro j ⟨i, k, hi, hk, rfl⟩
  exact hd.a_len i k hi hk

/-- The indices `bIdx d` are inside the buffer `b`. -/
theorem bIdx_subset_bytes {d : Inst} (hd : d.ok) : d.b.elems (bIdx d) ⊆ d.b.bytes := by
  apply Buf.elems_sub_bytes
  rintro j ⟨i, k, hi, hk, rfl⟩
  exact hd.b_len i k hi hk

/-!
## The two halves of a tile are disjoint, and facts survive a later store

The epilogue of one tile stores its two `128`-column halves with two separate `desc_store`s; the
two write disjoint elements. A store to one tile does not change the values another tile holds,
which lets the outer loop carry the facts of the tiles it already wrote.
-/


/-- The storage index of an in-bounds lane of a block of `c`, as a natural number. -/
theorem cDesc_elemAt_toNat (d : Inst) (o0 o1 : Int) (i : Idx [128, 128])
    (h0 : 0 ≤ o0 + (i.coord 0 : Int)) (h2 : 0 ≤ o1 + (i.coord 1 : Int)) :
    ((cDesc d).elemAt (TensorDesc.coord [o0, o1] i)).toNat
      = (o0 + (i.coord 0 : Int)).toNat * d.ldc + (o1 + (i.coord 1 : Int)).toNat := by
  rw [cDesc_elemAt]
  rw [Int.toNat_add (mul_nonneg h0 (Int.natCast_nonneg _)) h2]
  rw [Int.toNat_mul h0 (Int.natCast_nonneg _), Int.toNat_natCast]

/-- The two halves of one tile write disjoint elements of `c`. -/
theorem cDesc_img12_disjoint {d : Inst} (hd : d.ok) (t : Nat) :
    Disjoint
      (Buf.img (fun i : Idx [128, 128] => ((cDesc d).elemAt (TensorDesc.coord
          [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int)] i)).toNat)
        (fun i => (cDesc d).inbAt (TensorDesc.coord
          [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int)] i)))
      (Buf.img (fun i : Idx [128, 128] => ((cDesc d).elemAt (TensorDesc.coord
          [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int) + 128] i)).toNat)
        (fun i => (cDesc d).inbAt (TensorDesc.coord
          [((128 * pidM d t : Nat) : Int), ((256 * pidN d t : Nat) : Int) + 128] i))) := by
  rw [Set.disjoint_left]
  rintro e ⟨i, hi, rfl⟩ ⟨j, hj, he⟩
  obtain ⟨hi0, hi1, hi2, hi3⟩ :=
    (cDesc_inbAt d (((128 * pidM d t : Nat) : Int)) (((256 * pidN d t : Nat) : Int)) i).mp hi
  obtain ⟨hj0, hj1, hj2, hj3⟩ :=
    (cDesc_inbAt d (((128 * pidM d t : Nat) : Int))
      ((((256 * pidN d t : Nat) : Int)) + 128) j).mp hj
  have e1 := cDesc_elemAt_toNat d (((128 * pidM d t : Nat) : Int))
    (((256 * pidN d t : Nat) : Int)) i hi0 hi2
  have e2 := cDesc_elemAt_toNat d (((128 * pidM d t : Nat) : Int))
    ((((256 * pidN d t : Nat) : Int)) + 128) j hj0 hj2
  dsimp only at he
  rw [e1] at he
  rw [e2] at he
  have hj1' : (((128 * pidM d t : Nat) : Int) + (j.coord 0 : Int)).toNat < d.M := by
    rw [Int.toNat_lt hj0]; exact hj1
  have hj3' : (((((256 * pidN d t : Nat) : Int)) + 128) + (j.coord 1 : Int)).toNat < d.N := by
    rw [Int.toNat_lt hj2]; exact hj3
  have hi1' : (((128 * pidM d t : Nat) : Int) + (i.coord 0 : Int)).toNat < d.M := by
    rw [Int.toNat_lt hi0]; exact hi1
  have hi3' : ((((256 * pidN d t : Nat) : Int)) + (i.coord 1 : Int)).toNat < d.N := by
    rw [Int.toNat_lt hi2]; exact hi3
  obtain ⟨-, hcol⟩ := elem_inj hd hj1' hj3' hi1' hi3' he
  have hci : i.coord 1 < 128 := by
    have hcl := Idx.coord_lt i (k := 1) (by norm_num : 1 < [128, 128].length)
    simpa [Shape.dim_eq_getD] using hcl
  have hcolInt : ((((256 * pidN d t : Nat) : Int)) + 128) + (j.coord 1 : Int)
      = (((256 * pidN d t : Nat) : Int)) + (i.coord 1 : Int) := by
    rw [← Int.toNat_of_nonneg hj2, ← Int.toNat_of_nonneg hi2, hcol]
  have hj1n : (0 : Int) ≤ (j.coord 1 : Int) := Int.natCast_nonneg _
  have hci' : (i.coord 1 : Int) < 128 := by exact_mod_cast hci
  omega

/-- A read of an element of tile `T'` is unchanged by a write to the elements of tile `T`. -/
theorem read_frame {d : Inst} (hd : d.ok) {T T' : Nat} (hT : T < tiles d) (hT' : T' < tiles d)
    (hne : T ≠ T') {img : Set Nat} (himg : img ⊆ tileIdx d T) {j : Nat}
    (hj : j ∈ tileIdx d T') {m m' : Mem}
    (hfr : ∀ x, x ∉ d.c.elems img → m' x = m x) :
    m'.read .f16 (d.c.ptr (j : Int)).addr = m.read .f16 (d.c.ptr (j : Int)).addr := by
  apply Mem.read_congr
  intro x hx
  exact hfr x fun hximg =>
    (Set.disjoint_left.mp (Buf.elems_disjoint d.c (tileIdx_disjoint hd hT hT' hne))
      (Buf.elems_mono d.c himg hximg)) (Buf.span_sub_elems d.c hj hx)

/-- What one tile holds after the program wrote it: every in-bounds lane is present, and a
finite row and column give the specified element. -/
noncomputable def Qfact (d : Inst) (init : Mem) (T : Nat) (m : Mem) : Prop :=
  ∀ r n, r < 128 → n < 256 →
    128 * pidM d T + r < d.M → 256 * pidN d T + n < d.N →
    (∃ c : F16, m.read .f16
      (d.c.ptr (↑((128 * pidM d T + r) * d.ldc + (256 * pidN d T + n)))).addr = some c) ∧
    (RowFin d init (128 * pidM d T + r) → ColFin d init (256 * pidN d T + n) →
      LaneSpec d init (128 * pidM d T + r) (256 * pidN d T + n) m)

/-- `Q` is the conjunction of the tile facts of the program's tiles. -/
theorem Q_iff_Qfact (d : Inst) (init : Mem) (i : Fin (launch d).size) (m : Mem) :
    Q d init i m ↔ ∀ t ∈ progTiles d (i : Nat), Qfact d init t m :=
  Iff.rfl

/-- A tile fact survives a write to another tile. -/
theorem Qfact_transport {d : Inst} (hd : d.ok) {init : Mem} {T T' : Nat}
    (hT : T < tiles d) (hT' : T' < tiles d) (hne : T ≠ T')
    {img : Set Nat} (himg : img ⊆ tileIdx d T) {m m' : Mem}
    (hfr : ∀ x, x ∉ d.c.elems img → m' x = m x)
    (h : Qfact d init T' m) : Qfact d init T' m' := by
  intro r n hr hn hrM hnN
  have hj : (128 * pidM d T' + r) * d.ldc + (256 * pidN d T' + n) ∈ tileIdx d T' :=
    ⟨r, n, hr, hn, hrM, hnN, rfl⟩
  have hread := read_frame hd hT hT' hne himg hj (m := m) (m' := m') hfr
  obtain ⟨h1, h2⟩ := h r n hr hn hrM hnN
  refine ⟨?_, ?_⟩
  · rw [hread]; exact h1
  · intro hR hC
    unfold LaneSpec at h2 ⊢
    rw [hread]
    exact h2 hR hC

/-!
## Converting a finite `float32` into `float16`

The epilogue of the kernel converts both halves of the `float32` accumulator into `float16`. This
file records that a finite value in the range of `float16` converts to a finite `float16`.
-/


/-- Converting a finite `float32` value in the `float16` range to `float16` gives a finite value. -/
theorem cvt_f16_finite {x : F32} (hx : x.isFinite = true)
    (h : |x.toRat| ≤ Format.binary16.maxVal) :
    (FloatV.cvt x : F16).isFinite = true :=
  (FloatV.cvt_err (f := Format.binary32) (g := Format.binary16)
    Format.binary16_regular hx h).1

/-!
## From the per-tile facts to the postcondition

The program facts `Q` are stated over the values the initial memory holds (`aGet`, `bGet`,
`dSum`, `dTol`), while `Post` quantifies over any row and column values present in `init`. This
file converts between the two: functions that agree on `[0, K)` give the same `dotSum`, `pathMag`
and `tol`, and each element of `c` in the matrix gets its `OutSpec` from the tile fact that wrote
it.
-/


/-! ## The numeric quantities depend on the row and column values only below `K` -/

/-- The products of two pairs of rows that agree below `K` agree. -/
theorem prodV_congr {va va' vb vb' : Nat → F16} {K : Nat}
    (ha : ∀ k, k < K → va k = va' k) (hb : ∀ k, k < K → vb k = vb' k) :
    ∀ e, prodV va vb K e = prodV va' vb' K e := by
  intro e
  unfold prodV
  by_cases h : e < K
  · simp only [h, ↓reduceIte, ha e h, hb e h]
  · simp only [h, ↓reduceIte]

/-- The lifted products of two pairs of rows that agree below `K` agree. -/
theorem liftV_congr {va va' vb vb' : Nat → F16} {K : Nat}
    (ha : ∀ k, k < K → va k = va' k) (hb : ∀ k, k < K → vb k = vb' k) :
    ∀ e, liftV va vb K e = liftV va' vb' K e := by
  intro e
  unfold liftV
  by_cases h : e < K
  · simp only [h, ↓reduceIte, ha e h, hb e h]
  · simp only [h, ↓reduceIte]

/-- The path magnitude depends only on the rows below `K`. -/
theorem pathMag_congr {va va' vb vb' : Nat → F16} {K : Nat}
    (ha : ∀ k, k < K → va k = va' k) (hb : ∀ k, k < K → vb k = vb' k) :
    pathMag va vb K = pathMag va' vb' K := by
  unfold pathMag
  refine Finset.sum_congr rfl fun t _ => ?_
  have h1 : partSum (prodV va vb K) t = partSum (prodV va' vb' K) t := by
    unfold partSum
    exact Finset.sum_congr rfl fun e _ => prodV_congr ha hb e
  have h2 : stepMag (liftV va vb K) t = stepMag (liftV va' vb' K) t := by
    unfold stepMag
    exact Finset.sum_congr rfl fun k _ => liftV_congr ha hb (64 * t + k)
  rw [h1, h2]

/-- The exact dot product `dSum` of the proof is the claim's `dotSum` of any rows that agree with
the loaded ones. -/
theorem dSum_congr {d : Inst} {init : Mem} {i j : Nat} {va vb : Nat → F16}
    (ha : ∀ k, k < d.K → aGet d init i k = va k)
    (hb : ∀ k, k < d.K → bGet d init j k = vb k) :
    dSum d init i j = dotSum va vb d.K := by
  unfold dSum dotSum
  refine Finset.sum_congr rfl fun k hk => ?_
  rw [Finset.mem_range] at hk
  rw [ha k hk, hb k hk]

/-- The tolerance `dTol` of the proof is the claim's `tol` of any rows that agree with the loaded
ones. -/
theorem dTol_congr {d : Inst} {init : Mem} {i j : Nat} {va vb : Nat → F16}
    (ha : ∀ k, k < d.K → aGet d init i k = va k)
    (hb : ∀ k, k < d.K → bGet d init j k = vb k) :
    dTol d init i j = tol va vb d.K := by
  unfold dTol tol
  rw [pathMag_congr ha hb]

/-! ## One element of `c` -/

/-- A lane fact in the program's memory gives the `OutSpec` of the element in the final memory, as
long as the final memory agrees with the program's on the bytes the element occupies. -/
theorem outSpec_of_laneSpec {d : Inst} {init fin m : Mem} {i0 : Fin (launch d).size}
    {i j : Nat} {c : F16}
    (hspan : Mem.span .f16 (d.c.ptr (↑(i * d.ldc + j))).addr ⊆ R d ∪ W d i0)
    (hfin : ∀ x ∈ R d ∪ W d i0, fin x = m x)
    (hread : m.read .f16 (d.c.ptr (↑(i * d.ldc + j))).addr = some c)
    (hls : RowFin d init i → ColFin d init j → LaneSpec d init i j m) :
    OutSpec d init fin i j := by
  refine ⟨c, ?_, ?_⟩
  · rw [Mem.read_congr (m₁ := fin) (m₂ := m) fun x hx => hfin x (hspan hx)]
    exact hread
  · intro va vb hva hvb
    have ha : ∀ k, k < d.K → aGet d init i k = va k := by
      intro k hk
      obtain ⟨hr, -⟩ := hva k hk
      simp only [aGet]
      rw [hr]
      rfl
    have hb : ∀ k, k < d.K → bGet d init j k = vb k := by
      intro k hk
      obtain ⟨hr, -⟩ := hvb k hk
      simp only [bGet]
      rw [hr]
      rfl
    have hrow : RowFin d init i := by
      intro k hk
      obtain ⟨hr, hf⟩ := hva k hk
      simp only [aGet]
      rw [hr]
      simpa using hf
    have hcol : ColFin d init j := by
      intro k hk
      obtain ⟨hr, hf⟩ := hvb k hk
      simp only [bGet]
      rw [hr]
      simpa using hf
    obtain ⟨v, hv, herr, c', hread', hnae, hcfin⟩ := hls hrow hcol
    have hcc' : c' = c := by
      rw [hread] at hread'
      exact (Option.some.inj hread').symm
    have hs := dSum_congr ha hb
    have ht := dTol_congr ha hb
    refine ⟨v, hv, ?_, ?_, ?_⟩
    · rw [← hs, ← ht]; exact herr
    · simpa only [hcc'] using hnae
    · intro hroom
      rw [← hs, ← ht] at hroom
      simpa only [hcc'] using hcfin hroom

/-! ## Every element of `c`, and the bytes outside it -/

/-- **`Post` from the programs' facts**: if each program's final memory satisfies its tile facts
on its own footprint and the other bytes are unchanged, the final memory satisfies `Post`. -/
theorem post_of_locs {d : Inst} (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d init i m ∧ ∀ x ∈ R d ∪ W d i, fin x = m x)
    (hfr : ∀ x, (∀ i, x ∉ W d i) → fin x = init x) :
    Post d init fin := by
  refine ⟨?_, ?_⟩
  · intro I J hI hJ
    obtain ⟨t, ht, hIeq, hJeq⟩ := pid_surj' hd hI hJ
    rw [← hIeq, ← hJeq]
    have hS : 0 < d.NUM_SMS := hd.sms.1
    have hi0lt : t % d.NUM_SMS < (launch d).size := by
      rw [launch_size]
      exact lt_min (Nat.mod_lt _ hS) (lt_of_le_of_lt (Nat.mod_le _ _) ht)
    let i0 : Fin (launch d).size := ⟨t % d.NUM_SMS, hi0lt⟩
    have hprog : t ∈ progTiles d (i0 : Nat) := ⟨ht, rfl⟩
    obtain ⟨m, hQm, hfin⟩ := hloc i0
    have hQf := (Q_iff_Qfact d init i0 m).mp hQm t hprog
    have hI' : I % 128 < 128 := Nat.mod_lt _ (by norm_num)
    have hJ' : J % 256 < 256 := Nat.mod_lt _ (by norm_num)
    have hMi : 128 * pidM d t + I % 128 < d.M := by rw [hIeq]; exact hI
    have hNj : 256 * pidN d t + J % 256 < d.N := by rw [hJeq]; exact hJ
    obtain ⟨⟨c, hread⟩, hls⟩ := hQf (I % 128) (J % 256) hI' hJ' hMi hNj
    have hmem : ((128 * pidM d t + I % 128) * d.ldc + (256 * pidN d t + J % 256))
        ∈ Wset d i0 :=
      ⟨t, hprog, I % 128, J % 256, hI', hJ', hMi, hNj, rfl⟩
    have hspan : Mem.span .f16 (d.c.ptr (↑((128 * pidM d t + I % 128) * d.ldc
        + (256 * pidN d t + J % 256)))).addr ⊆ R d ∪ W d i0 :=
      fun x hx => Or.inr (Buf.span_sub_elems d.c hmem hx)
    exact outSpec_of_laneSpec (i0 := i0) hspan hfin hread hls
  · intro z hz
    exact hfr z fun i hzi => hz (W_subset_c i hzi)

end Claims.TmaPersistentMatmul
