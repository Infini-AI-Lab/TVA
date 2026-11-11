import Claim
import TVA.Proof
import Proof.Basic
import Proof.Arith

/-!
# The scale descriptors at the walk's offsets

The two scale descriptors' `Valid`, the `Access` of the block each step loads, which lanes are
inside the tensor, and the element (`elemAt`) a lane names.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-! ## Numeric helpers -/

theorem scaleOff_Mpos {d : Inst} (hd : d.ok) : 0 < d.M / 128 := by
  have h := M_div d hd; have h2 := M_pos hd; omega

/-- `M / 128 < 2^31`. -/
theorem scaleOff_Mlt {d : Inst} (hd : d.ok) : d.M / 128 < 2 ^ 31 := by
  have h := M_div d hd; have h2 := hd.M_ok.2.2; omega

/-- `0 < N / 128`. -/
theorem scaleOff_Npos {d : Inst} (hd : d.ok) : 0 < d.N / 128 := by
  have h := N_div d hd; have h2 := N_pos hd; omega

/-- `N / 128 < 2^31`. -/
theorem scaleOff_Nlt {d : Inst} (hd : d.ok) : d.N / 128 < 2 ^ 31 := by
  have h := N_div d hd; have h2 := hd.N_ok.2.2; omega

/-- `0 < K / 128`. -/
theorem scaleOff_Kpos {d : Inst} (hd : d.ok) : 0 < d.K / 128 := by
  have h := K_div d hd; have h2 := K_pos hd; omega

/-- `K / 128 < 2^31`. -/
theorem scaleOff_Klt {d : Inst} (hd : d.ok) : d.K / 128 < 2 ^ 31 := by
  have h := K_div d hd; have h2 := hd.K_ok.2.2; omega

/-- A `uint8` is one byte. -/
theorem scaleOff_u8bytes : (DType.u8).bytes = 1 := rfl

/-- `x * 512` is a multiple of `16`. -/
theorem scaleOff_mod16_512 (x : Nat) : (x * 512) % 16 = 0 :=
  Nat.dvd_iff_mod_eq_zero.mp (dvd_mul_of_dvd_right (by norm_num : (16 : Nat) ∣ 512) x)

/-! ## `a`'s scales (block `[1, 1, 1, 2, 256]`) -/

theorem aScaleDesc_valid {d : Inst} (hd : d.ok) : (aScaleDesc d).Valid := by
  have hMp := scaleOff_Mpos hd; have hMl := scaleOff_Mlt hd
  have hKp := scaleOff_Kpos hd; have hKl := scaleOff_Klt hd
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [aScaleDesc, TensorDesc.rank]
  · simp [aScaleDesc, TensorDesc.rank]
  · simp [aScaleDesc, TensorDesc.rank]
  · simp [aScaleDesc, TensorDesc.rank]
  · intro k hk
    have hk5 : k < 5 := by simpa [aScaleDesc, TensorDesc.rank] using hk
    interval_cases k <;> simp [aScaleDesc] <;> omega
  · show (d.a_scale.ptr 0).addr % 16 = 0
    rw [Buf.ptr_addr]
    simpa using hd.base_al.2.2.1
  · simp [aScaleDesc]
  · intro k hk
    have hk4 : k < 4 := by
      have h : k + 1 < 5 := by simpa [aScaleDesc, TensorDesc.rank] using hk
      omega
    interval_cases k <;> simp [aScaleDesc, scaleOff_u8bytes] <;> exact scaleOff_mod16_512 _
  · simp [aScaleDesc]
  · simp [aScaleDesc]


/-- `Access` contains `Valid`, which needs `0 < d.M / 128`, so the model's hypotheses are
essential here. -/
theorem aScaleDesc_access {d : Inst} (hd : d.ok) (pm t : Nat) :
    (aScaleDesc d).Access [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] [1, 1, 1, 2, 256] := by
  refine ⟨aScaleDesc_valid hd, rfl, rfl, rfl, ?_⟩
  simp

/-- A lane of `a_scale`'s block at a tile of the grid is in bounds. -/
theorem aScaleDesc_inbAt {d : Inst} (_hd : d.ok) (pm t : Nat) (i : Idx [1, 1, 1, 2, 256])
    (hpm : pm < d.M / 128) (ht : t < d.K / 128) :
    (aScaleDesc d).inbAt (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] i) =
      true := by
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk5 : k < 5 := by simpa [aScaleDesc, TensorDesc.rank] using hk
  have c0 : i.coord 0 < 1 := i.coord_lt (k := 0) (by simp)
  have c1 : i.coord 1 < 1 := i.coord_lt (k := 1) (by simp)
  have c2 : i.coord 2 < 1 := i.coord_lt (k := 2) (by simp)
  have c3 : i.coord 3 < 2 := i.coord_lt (k := 3) (by simp)
  have c4 : i.coord 4 < 256 := i.coord_lt (k := 4) (by simp)
  interval_cases k <;> simp [TensorDesc.coord, aScaleDesc] <;> omega

/-- The element of `a_scale` that lane `i` of the block at tile `(pm, t)` addresses. -/
theorem aScaleDesc_elemAt {d : Inst} (_hd : d.ok) (pm t : Nat) (i : Idx [1, 1, 1, 2, 256])
    (_hpm : pm < d.M / 128) (_ht : t < d.K / 128) :
    ((aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] i)).toNat =
      pm * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4 := by
  have c0 : i.coord 0 < 1 := i.coord_lt (k := 0) (by simp)
  have c1 : i.coord 1 < 1 := i.coord_lt (k := 1) (by simp)
  have c2 : i.coord 2 < 1 := i.coord_lt (k := 2) (by simp)
  have e0 : i.coord 0 = 0 := by omega
  have e1 : i.coord 1 = 0 := by omega
  have e2 : i.coord 2 = 0 := by omega
  have h : (aScaleDesc d).elemAt (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] i)
      = ((pm * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4 : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, aScaleDesc, List.length_cons, List.length_nil,
      List.range_succ, List.range_zero, List.map_append, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, List.sum_append, List.nil_append, TensorDesc.coord,
      List.getD_cons_zero, List.getD_cons_succ, e0, e1, e2]
    push_cast
    ring
  rw [h, Int.toNat_natCast]


/-- Every lane of `a_scale`'s block in bounds addresses an element below its length. -/
theorem aScaleDesc_hin {d : Inst} (hd : d.ok) (pm t : Nat) (i : Idx [1, 1, 1, 2, 256])
    (hpm : pm < d.M / 128) (ht : t < d.K / 128) :
    ((aScaleDesc d).elemAt
        (TensorDesc.coord [0, ((pm : Nat) : Int), ((t : Nat) : Int), 0, 0] i)).toNat <
      d.a_scale.len := by
  rw [aScaleDesc_elemAt hd pm t i hpm ht]
  have hK : 0 < d.K / 128 := scaleOff_Kpos hd
  have c3 : i.coord 3 < 2 := i.coord_lt (k := 3) (by simp)
  have c4 : i.coord 4 < 256 := i.coord_lt (k := 4) (by simp)
  set A := (d.K / 128) * 512 with hA
  have hApos : 0 < A := Nat.mul_pos hK (by norm_num)
  have h1 : t * 512 + i.coord 3 * 256 + i.coord 4 < A := by
    rw [hA]; omega
  have h2 : pm * A + (t * 512 + i.coord 3 * 256 + i.coord 4) < (pm + 1) * A := by
    rw [Nat.succ_mul]
    exact Nat.add_lt_add_left h1 (pm * A)
  have h3 : (pm + 1) * A ≤ d.M / 128 * A :=
    Nat.mul_le_mul_right A (by omega)
  have h4 : d.M / 128 * A = d.M / 128 * (d.K / 128) * 512 := by
    rw [hA, Nat.mul_assoc]
  have h5 := hd.as_len
  have heq : pm * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4
      = pm * A + (t * 512 + i.coord 3 * 256 + i.coord 4) := by
    rw [hA, Nat.mul_assoc]
    omega
  calc pm * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4
      = pm * A + (t * 512 + i.coord 3 * 256 + i.coord 4) := heq
    _ < (pm + 1) * A := h2
    _ ≤ d.M / 128 * A := h3
    _ = d.M / 128 * (d.K / 128) * 512 := h4
    _ ≤ d.a_scale.len := h5

/-! ## `b`'s scales (block `[1, 2, 1, 2, 256]`) -/

theorem bScaleDesc_valid {d : Inst} (hd : d.ok) : (bScaleDesc d).Valid := by
  have hNp := scaleOff_Npos hd; have hNl := scaleOff_Nlt hd
  have hKp := scaleOff_Kpos hd; have hKl := scaleOff_Klt hd
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [bScaleDesc, TensorDesc.rank]
  · simp [bScaleDesc, TensorDesc.rank]
  · simp [bScaleDesc, TensorDesc.rank]
  · simp [bScaleDesc, TensorDesc.rank]
  · intro k hk
    have hk5 : k < 5 := by simpa [bScaleDesc, TensorDesc.rank] using hk
    interval_cases k <;> simp [bScaleDesc] <;> omega
  · show (d.b_scale.ptr 0).addr % 16 = 0
    rw [Buf.ptr_addr]
    simpa using hd.base_al.2.2.2.1
  · simp [bScaleDesc]
  · intro k hk
    have hk4 : k < 4 := by
      have h : k + 1 < 5 := by simpa [bScaleDesc, TensorDesc.rank] using hk
      omega
    interval_cases k <;> simp [bScaleDesc, scaleOff_u8bytes] <;> exact scaleOff_mod16_512 _
  · simp [bScaleDesc]
  · simp [bScaleDesc]


/-- `Access` contains `Valid`, which needs `0 < d.N / 128`, so the model's hypotheses are
essential here. -/
theorem bScaleDesc_access {d : Inst} (hd : d.ok) (pn t : Nat) :
    (bScaleDesc d).Access [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0]
      [1, 2, 1, 2, 256] := by
  refine ⟨bScaleDesc_valid hd, rfl, rfl, rfl, ?_⟩
  simp

/-- `2 * (N / 256) ≤ N / 128`. -/
theorem scaleOff_two_N256 {d : Inst} (hd : d.ok) : 2 * (d.N / 256) ≤ d.N / 128 := by
  have hN := N_div d hd
  have h1 : d.N / 256 = (d.N / 128) / 2 := by
    conv_lhs => rw [hN]
    rw [show (256 : Nat) = 128 * 2 from rfl]
    exact Nat.mul_div_mul_left (d.N / 128) 2 (m := 128) (by norm_num)
  rw [h1]
  omega

/-- A lane of `b_scale`'s block at a tile of the grid is in bounds. -/
theorem bScaleDesc_inbAt {d : Inst} (hd : d.ok) (pn t : Nat) (i : Idx [1, 2, 1, 2, 256])
    (hpn : pn < d.N / 256) (ht : t < d.K / 128) :
    (bScaleDesc d).inbAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i) = true := by
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk5 : k < 5 := by simpa [bScaleDesc, TensorDesc.rank] using hk
  have c0 : i.coord 0 < 1 := i.coord_lt (k := 0) (by simp)
  have c1 : i.coord 1 < 2 := i.coord_lt (k := 1) (by simp)
  have c2 : i.coord 2 < 1 := i.coord_lt (k := 2) (by simp)
  have c3 : i.coord 3 < 2 := i.coord_lt (k := 3) (by simp)
  have c4 : i.coord 4 < 256 := i.coord_lt (k := 4) (by simp)
  have hN := scaleOff_two_N256 hd
  interval_cases k <;> simp [TensorDesc.coord, bScaleDesc] <;> omega

/-- The element of `b_scale` that lane `i` of the block at tile `(pn, t)` addresses. -/
theorem bScaleDesc_elemAt {d : Inst} (_hd : d.ok) (pn t : Nat) (i : Idx [1, 2, 1, 2, 256])
    (_hpn : pn < d.N / 256) (_ht : t < d.K / 128) :
    ((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i)).toNat =
      (2 * pn + i.coord 1) * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4 := by
  have c0 : i.coord 0 < 1 := i.coord_lt (k := 0) (by simp)
  have c2 : i.coord 2 < 1 := i.coord_lt (k := 2) (by simp)
  have e0 : i.coord 0 = 0 := by omega
  have e2 : i.coord 2 = 0 := by omega
  have h : (bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i)
      = (((2 * pn + i.coord 1) * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 +
          i.coord 4 : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bScaleDesc, List.length_cons, List.length_nil,
      List.range_succ, List.range_zero, List.map_append, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, List.sum_append, List.nil_append, TensorDesc.coord,
      List.getD_cons_zero, List.getD_cons_succ, e0, e2]
    push_cast
    ring
  rw [h, Int.toNat_natCast]

/-- Every lane of `b_scale`'s block in bounds addresses an element below its length. -/
theorem bScaleDesc_hin {d : Inst} (hd : d.ok) (pn t : Nat) (i : Idx [1, 2, 1, 2, 256])
    (hpn : pn < d.N / 256) (ht : t < d.K / 128) :
    ((bScaleDesc d).elemAt
        (TensorDesc.coord [0, ((2 * pn : Nat) : Int), ((t : Nat) : Int), 0, 0] i)).toNat <
      d.b_scale.len := by
  rw [bScaleDesc_elemAt hd pn t i hpn ht]
  have hK : 0 < d.K / 128 := scaleOff_Kpos hd
  have c1 : i.coord 1 < 2 := i.coord_lt (k := 1) (by simp)
  have c3 : i.coord 3 < 2 := i.coord_lt (k := 3) (by simp)
  have c4 : i.coord 4 < 256 := i.coord_lt (k := 4) (by simp)
  have hN := scaleOff_two_N256 hd
  have hpm' : 2 * pn + i.coord 1 < d.N / 128 := by omega
  set A := (d.K / 128) * 512 with hA
  have h1 : t * 512 + i.coord 3 * 256 + i.coord 4 < A := by
    rw [hA]; omega
  have h2 : (2 * pn + i.coord 1) * A + (t * 512 + i.coord 3 * 256 + i.coord 4) <
      (2 * pn + i.coord 1 + 1) * A := by
    have hs : (2 * pn + i.coord 1 + 1) * A = (2 * pn + i.coord 1) * A + A := by
      rw [Nat.add_mul, one_mul]
    rw [hs]
    exact Nat.add_lt_add_left h1 _
  have h3 : (2 * pn + i.coord 1 + 1) * A ≤ d.N / 128 * A :=
    Nat.mul_le_mul_right A (by omega)
  have h4 : d.N / 128 * A = d.N / 128 * (d.K / 128) * 512 := by
    rw [hA, Nat.mul_assoc]
  have h5 := hd.bs_len
  have heq : (2 * pn + i.coord 1) * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 +
        i.coord 4
      = (2 * pn + i.coord 1) * A + (t * 512 + i.coord 3 * 256 + i.coord 4) := by
    rw [hA, Nat.mul_assoc]
    omega
  calc (2 * pn + i.coord 1) * (d.K / 128) * 512 + t * 512 + i.coord 3 * 256 + i.coord 4
      = (2 * pn + i.coord 1) * A + (t * 512 + i.coord 3 * 256 + i.coord 4) := heq
    _ < (2 * pn + i.coord 1 + 1) * A := h2
    _ ≤ d.N / 128 * A := h3
    _ = d.N / 128 * (d.K / 128) * 512 := h4
    _ ≤ d.b_scale.len := h5

end Claims.BlockScaledMxfp8
