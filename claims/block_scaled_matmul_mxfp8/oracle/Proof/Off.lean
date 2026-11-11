import Claim
import TVA.Proof
import Proof.Basic
import Proof.Arith

/-!
# The descriptors at the walk's offsets

The five descriptors' `Valid`, the `Access` of each block the loop body loads (and of the block
the epilogue stores), which lanes are inside the tensor, and the element (`elemAt`) a lane names.
-/

namespace Claims.BlockScaledMxfp8

open TVA

/-! ## Coordinates of a lane on literal offset lists -/

theorem coord_l2_0 {n m : Nat} (i : Idx [n, m]) (a b : Int) :
    TensorDesc.coord [a, b] i 0 = a + (i.coord 0 : Int) := by
  rfl

/-- Coordinate 1 of a lane of a 2-D block at offsets `[a, b]`. -/
theorem coord_l2_1 {n m : Nat} (i : Idx [n, m]) (a b : Int) :
    TensorDesc.coord [a, b] i 1 = b + (i.coord 1 : Int) := by
  rfl

/-- The coordinates of a 2-D block beyond the second are `0`. -/
theorem coord_l2_ge {n m : Nat} (i : Idx [n, m]) (a b : Int) {k : Nat} (hk : 2 ≤ k) :
    TensorDesc.coord [a, b] i k = 0 := by
  rcases k with _ | _ | k
  · omega
  · omega
  · rfl

/-- Coordinate 0 of a lane of a 5-D block at the offsets `[a, b, c, e, f]`. -/
theorem coord_l5_0 {p q r s u : Nat} (i : Idx [p, q, r, s, u]) (a b c e f : Int) :
    TensorDesc.coord [a, b, c, e, f] i 0 = a + (i.coord 0 : Int) := by
  rfl

/-- Coordinate 1 of a lane of a 5-D block. -/
theorem coord_l5_1 {p q r s u : Nat} (i : Idx [p, q, r, s, u]) (a b c e f : Int) :
    TensorDesc.coord [a, b, c, e, f] i 1 = b + (i.coord 1 : Int) := by
  rfl

/-- Coordinate 2 of a lane of a 5-D block. -/
theorem coord_l5_2 {p q r s u : Nat} (i : Idx [p, q, r, s, u]) (a b c e f : Int) :
    TensorDesc.coord [a, b, c, e, f] i 2 = c + (i.coord 2 : Int) := by
  rfl

/-- Coordinate 3 of a lane of a 5-D block. -/
theorem coord_l5_3 {p q r s u : Nat} (i : Idx [p, q, r, s, u]) (a b c e f : Int) :
    TensorDesc.coord [a, b, c, e, f] i 3 = e + (i.coord 3 : Int) := by
  rfl

/-- Coordinate 4 of a lane of a 5-D block. -/
theorem coord_l5_4 {p q r s u : Nat} (i : Idx [p, q, r, s, u]) (a b c e f : Int) :
    TensorDesc.coord [a, b, c, e, f] i 4 = f + (i.coord 4 : Int) := by
  rfl

/-- The coordinates of a 5-D block beyond the fifth are `0`. -/
theorem coord_l5_ge {p q r s u : Nat} (i : Idx [p, q, r, s, u]) (a b c e f : Int) {k : Nat}
    (hk : 5 ≤ k) : TensorDesc.coord [a, b, c, e, f] i k = 0 := by
  rcases k with _ | _ | _ | _ | _ | k
  · omega
  · omega
  · omega
  · omega
  · omega
  · rfl

/-! ## Arithmetic helpers for the block coordinates -/

/-- `p * c + l < M` when `p < M / c` and `l < c`. -/
theorem off_mul_add_lt_of_lt_div {p l M c : Nat} (h : p < M / c) (hl : l < c) :
    p * c + l < M := by
  have h1 : p * c + l < (p + 1) * c := by
    have he : (p + 1) * c = p * c + c := by ring
    rw [he]; omega
  have h2 : (p + 1) * c ≤ M := by
    have h3 : p + 1 ≤ M / c := by omega
    calc (p + 1) * c ≤ (M / c) * c := Nat.mul_le_mul_right c h3
      _ ≤ M := Nat.div_mul_le_self M c
  omega

/-- `A * K + B < M * K` when `A < M` and `B < K`. -/
theorem off_mul_add_lt_mul {A B M K : Nat} (hA : A < M) (hB : B < K) :
    A * K + B < M * K := by
  have h1 : A * K + B < A * K + K := by omega
  have h2 : A * K + K ≤ M * K := by
    have he : A * K + K = (A + 1) * K := by ring
    rw [he]
    exact Nat.mul_le_mul_right K (by omega)
  omega

/-! ## `a` -/

theorem aDesc_valid {d : Inst} (hd : d.ok) : (aDesc d).Valid := by
  have hM := hd.M_ok
  have hK := hd.K_ok
  simp only [TensorDesc.Valid, TensorDesc.rank, aDesc, List.length_cons, List.length_nil,
    List.getLast?_cons_cons, Buf.ptr, DType.bytes, Nat.reduceAdd, Nat.reduceLeDiff]
  refine ⟨trivial, trivial, trivial, trivial, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk
    interval_cases k
    · simp only [List.getD_cons_zero]
      exact ⟨by omega, by omega⟩
    · simp only [List.getD_cons_succ, List.getD_cons_zero]
      exact ⟨by omega, by omega⟩
  · simpa using hd.base_al.1
  · rfl
  · intro k hk
    have hk0 : k = 0 := by omega
    subst hk0
    simp only [List.getD_cons_zero, Nat.mul_one]
    have hKd := K_div d hd
    omega
  · simp
  · simp

/-- `a`'s block at a tile of the grid is an access the descriptor allows. -/
theorem aDesc_access {d : Inst} (hd : d.ok) (pm t : Nat) :
    (aDesc d).Access [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] [128, 128] := by
  refine ⟨aDesc_valid hd, rfl, rfl, rfl, ?_⟩
  simp only [List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some, DType.bytes]
  rw [← Int.dvd_iff_emod_eq_zero]
  exact ⟨8 * (t : Int), by push_cast; ring⟩

/-- A lane of `a`'s block at a tile of the grid is in bounds. -/
theorem aDesc_inbAt {d : Inst} (hd : d.ok) (pm t : Nat) (l : Idx [128, 128])
    (hpm : pm < d.M / 128) (ht : t < d.K / 128) :
    (aDesc d).inbAt (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l) =
      true := by
  have hMd := M_div d hd
  have hKd := K_div d hd
  have hl0 : l.coord 0 < 128 := l.coord_lt (by simp)
  have hl1 : l.coord 1 < 128 := l.coord_lt (by simp)
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk2 : k < 2 := by simpa [TensorDesc.rank, aDesc] using hk
  interval_cases k
  all_goals
    simp only [TensorDesc.coord, aDesc, List.getD_cons_zero, List.getD_cons_succ]
    constructor <;> push_cast <;> omega

/-- The element of `a` that lane `l` of the block at tile `(pm, t)` addresses. -/
theorem aDesc_elemAt {d : Inst} (_hd : d.ok) (pm t : Nat) (l : Idx [128, 128])
    (_hpm : pm < d.M / 128) (_ht : t < d.K / 128) :
    ((aDesc d).elemAt
        (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat =
      (pm * 128 + l.coord 0) * d.K + (128 * t + l.coord 1) := by
  have key : ((aDesc d).elemAt
        (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l))
      = (((pm * 128 + l.coord 0) * d.K + (128 * t + l.coord 1) : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, aDesc, List.length_cons, List.length_nil,
      List.range_succ, List.range_zero, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
      TensorDesc.coord, List.getD_cons_zero, List.getD_cons_succ, List.nil_append,
      List.cons_append]
    push_cast
    ring
  rw [key, Int.toNat_natCast]

/-- Every lane of `a`'s block in bounds addresses an element below the length of `a`. -/
theorem aDesc_hin {d : Inst} (hd : d.ok) (pm t : Nat) (l : Idx [128, 128])
    (hpm : pm < d.M / 128) (ht : t < d.K / 128) :
    ((aDesc d).elemAt
        (TensorDesc.coord [((pm * 128 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat <
      d.a.len := by
  rw [aDesc_elemAt hd pm t l hpm ht]
  have hl0 : l.coord 0 < 128 := l.coord_lt (by simp)
  have hl1 : l.coord 1 < 128 := l.coord_lt (by simp)
  have hA : pm * 128 + l.coord 0 < d.M := off_mul_add_lt_of_lt_div hpm hl0
  have hB : 128 * t + l.coord 1 < d.K := by
    have h := off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := d.K) (l := l.coord 1) ht hl1
    omega
  have hlen : d.M * d.K ≤ d.a.len := hd.a_len
  have hm' := off_mul_add_lt_mul hA hB
  omega

/-! ## `b`, `"mixed"` (two elements per byte: block `[256, 64]`) -/

theorem bDesc_valid_mixed {d : Inst} (hd : d.ok) (hm : d.mixed = true) : (bDesc d).Valid := by
  have hep : epbB true = 2 := by simp [epbB]
  have hN := hd.N_ok
  have hK := hd.K_ok
  simp only [TensorDesc.Valid, TensorDesc.rank, bDesc, hm, hep, bDT, List.length_cons,
    List.length_nil, List.getLast?_cons_cons, Buf.ptr, DType.bytes, Nat.reduceAdd, Nat.reduceLeDiff]
  refine ⟨trivial, trivial, trivial, trivial, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk
    interval_cases k
    · simp only [List.getD_cons_zero]
      exact ⟨by omega, by omega⟩
    · simp only [List.getD_cons_succ, List.getD_cons_zero]
      exact ⟨by omega, by omega⟩
  · simpa using hd.base_al.2.1
  · rfl
  · intro k hk
    have hk0 : k = 0 := by omega
    subst hk0
    simp only [List.getD_cons_zero, Nat.mul_one]
    have hKd := K_div d hd
    omega
  · simp
  · simp

/-- `b`'s block at a tile of the grid is an access the descriptor allows (`mixed`). -/
theorem bDesc_access_mixed {d : Inst} (hd : d.ok) (hm : d.mixed = true) (pn t : Nat) :
    (bDesc d).Access [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] [256, 64] := by
  refine ⟨bDesc_valid_mixed hd hm, ?_, rfl, rfl, ?_⟩
  · simp [bDesc, hm, epbB]
  · simp only [List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some, DType.bytes,
      bDT, hm]
    rw [← Int.dvd_iff_emod_eq_zero]
    exact ⟨4 * (t : Int), by push_cast; ring⟩

/-- A lane of `b`'s block at a tile of the grid is in bounds (`mixed`). -/
theorem bDesc_inbAt_mixed {d : Inst} (hd : d.ok) (hm : d.mixed = true) (pn t : Nat)
    (l : Idx [256, 64]) (hpn : pn < d.N / 256) (ht : t < d.K / 128) :
    (bDesc d).inbAt (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l) =
      true := by
  have hep : epbB true = 2 := by simp [epbB]
  have hKd := K_div d hd
  have hK2 : d.K / 2 = 64 * (d.K / 128) := by
    conv_lhs => rw [hKd]
    rw [show 128 * (d.K / 128) = 2 * (64 * (d.K / 128)) by ring]
    exact Nat.mul_div_cancel_left (64 * (d.K / 128)) (n := 2) (by norm_num)
  have hl0 : l.coord 0 < 256 := l.coord_lt (by simp)
  have hl1 : l.coord 1 < 64 := l.coord_lt (by simp)
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk2 : k < 2 := by simpa [TensorDesc.rank, bDesc, hm, hep] using hk
  interval_cases k
  all_goals
    simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ]
    constructor <;> push_cast <;> omega

/-- The element of `b` that lane `l` of the block at tile `(pn, t)` addresses (`mixed`). -/
theorem bDesc_elemAt_mixed {d : Inst} (_hd : d.ok) (hm : d.mixed = true) (pn t : Nat)
    (l : Idx [256, 64]) (_hpn : pn < d.N / 256) (_ht : t < d.K / 128) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat =
      (pn * 256 + l.coord 0) * (d.K / 2) + (64 * t + l.coord 1) := by
  have hep : epbB true = 2 := by simp [epbB]
  have key : ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l))
      = (((pn * 256 + l.coord 0) * (d.K / 2) + (64 * t + l.coord 1) : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bDesc, hm, hep, List.length_cons,
      List.length_nil, List.range_succ, List.range_zero, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, TensorDesc.coord, List.getD_cons_zero, List.getD_cons_succ,
      List.nil_append, List.cons_append]
    push_cast
    ring
  rw [key, Int.toNat_natCast]

/-- Every lane of `b`'s block in bounds addresses an element below the length of `b` (`mixed`). -/
theorem bDesc_hin_mixed {d : Inst} (hd : d.ok) (hm : d.mixed = true) (pn t : Nat)
    (l : Idx [256, 64]) (hpn : pn < d.N / 256) (ht : t < d.K / 128) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((64 * t : Nat) : Int)] l)).toNat <
      d.b.len := by
  rw [bDesc_elemAt_mixed hd hm pn t l hpn ht]
  have hlen : d.N * (d.K / 2) ≤ d.b.len := by simpa [epbB, hm] using hd.b_len
  have hKd := K_div d hd
  have hK2 : d.K / 2 = 64 * (d.K / 128) := by
    conv_lhs => rw [hKd]
    rw [show 128 * (d.K / 128) = 2 * (64 * (d.K / 128)) by ring]
    exact Nat.mul_div_cancel_left (64 * (d.K / 128)) (n := 2) (by norm_num)
  have hl0 : l.coord 0 < 256 := l.coord_lt (by simp)
  have hl1 : l.coord 1 < 64 := l.coord_lt (by simp)
  have hA : pn * 256 + l.coord 0 < d.N := off_mul_add_lt_of_lt_div hpn hl0
  have hB : 64 * t + l.coord 1 < d.K / 2 := by
    have h1 : 64 * t + l.coord 1 < (t + 1) * 64 := by
      have he : (t + 1) * 64 = t * 64 + 64 := by ring
      rw [he]; omega
    have h2 : (t + 1) * 64 ≤ d.K / 2 := by
      rw [hK2]
      calc (t + 1) * 64 ≤ (d.K / 128) * 64 := Nat.mul_le_mul_right 64 (by omega)
        _ = 64 * (d.K / 128) := Nat.mul_comm _ _
    omega
  have hm' := off_mul_add_lt_mul hA hB
  omega

/-! ## `b`, `"mxfp8"` (block `[256, 128]`) -/

theorem bDesc_valid_mx {d : Inst} (hd : d.ok) (hm : d.mixed = false) : (bDesc d).Valid := by
  have hep : epbB false = 1 := by simp [epbB]
  have hN := hd.N_ok
  have hK := hd.K_ok
  simp only [TensorDesc.Valid, TensorDesc.rank, bDesc, hm, hep, bDT, List.length_cons,
    List.length_nil, List.getLast?_cons_cons, Buf.ptr, DType.bytes, Nat.reduceAdd,
    Nat.reduceLeDiff, Nat.div_one]
  refine ⟨trivial, trivial, trivial, trivial, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk
    interval_cases k
    · simp only [List.getD_cons_zero]
      exact ⟨by omega, by omega⟩
    · simp only [List.getD_cons_succ, List.getD_cons_zero]
      exact ⟨by omega, by omega⟩
  · simpa using hd.base_al.2.1
  · rfl
  · intro k hk
    have hk0 : k = 0 := by omega
    subst hk0
    simp only [List.getD_cons_zero, Nat.mul_one]
    have hKd := K_div d hd
    omega
  · simp
  · simp

/-- `b`'s block at a tile of the grid is an access the descriptor allows (`mxfp8`). -/
theorem bDesc_access_mx {d : Inst} (hd : d.ok) (hm : d.mixed = false) (pn t : Nat) :
    (bDesc d).Access [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] [256, 128] := by
  refine ⟨bDesc_valid_mx hd hm, ?_, rfl, rfl, ?_⟩
  · simp [bDesc, hm, epbB]
  · simp only [List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some, DType.bytes,
      bDT, hm]
    rw [← Int.dvd_iff_emod_eq_zero]
    exact ⟨8 * (t : Int), by push_cast; ring⟩

/-- A lane of `b`'s block at a tile of the grid is in bounds (`mxfp8`). -/
theorem bDesc_inbAt_mx {d : Inst} (hd : d.ok) (hm : d.mixed = false) (pn t : Nat)
    (l : Idx [256, 128]) (hpn : pn < d.N / 256) (ht : t < d.K / 128) :
    (bDesc d).inbAt (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l) =
      true := by
  have hep : epbB false = 1 := by simp [epbB]
  have hKd := K_div d hd
  have hl0 : l.coord 0 < 256 := l.coord_lt (by simp)
  have hl1 : l.coord 1 < 128 := l.coord_lt (by simp)
  rw [TensorDesc.inbAt, decide_eq_true_iff]
  intro k hk
  have hk2 : k < 2 := by simpa [TensorDesc.rank, bDesc, hm, hep] using hk
  interval_cases k
  all_goals
    simp only [TensorDesc.coord, bDesc, hm, hep, List.getD_cons_zero, List.getD_cons_succ,
      Nat.div_one]
    constructor <;> push_cast <;> omega

/-- The element of `b` that lane `l` of the block at tile `(pn, t)` addresses (`mxfp8`). -/
theorem bDesc_elemAt_mx {d : Inst} (_hd : d.ok) (hm : d.mixed = false) (pn t : Nat)
    (l : Idx [256, 128]) (_hpn : pn < d.N / 256) (_ht : t < d.K / 128) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat =
      (pn * 256 + l.coord 0) * d.K + (128 * t + l.coord 1) := by
  have hep : epbB false = 1 := by simp [epbB]
  have key : ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l))
      = (((pn * 256 + l.coord 0) * d.K + (128 * t + l.coord 1) : Nat) : Int) := by
    simp only [TensorDesc.elemAt, TensorDesc.rank, bDesc, hm, hep, List.length_cons,
      List.length_nil, List.range_succ, List.range_zero, List.map_cons, List.map_nil,
      List.sum_cons, List.sum_nil, TensorDesc.coord, List.getD_cons_zero, List.getD_cons_succ,
      List.nil_append, List.cons_append, Nat.div_one]
    push_cast
    ring
  rw [key, Int.toNat_natCast]

/-- Every lane of `b`'s block in bounds addresses an element below the length of `b` (`mxfp8`). -/
theorem bDesc_hin_mx {d : Inst} (hd : d.ok) (hm : d.mixed = false) (pn t : Nat)
    (l : Idx [256, 128]) (hpn : pn < d.N / 256) (ht : t < d.K / 128) :
    ((bDesc d).elemAt
        (TensorDesc.coord [((pn * 256 : Nat) : Int), ((128 * t : Nat) : Int)] l)).toNat <
      d.b.len := by
  rw [bDesc_elemAt_mx hd hm pn t l hpn ht]
  have hlen : d.N * d.K ≤ d.b.len := by simpa [epbB, hm] using hd.b_len
  have hl0 : l.coord 0 < 256 := l.coord_lt (by simp)
  have hl1 : l.coord 1 < 128 := l.coord_lt (by simp)
  have hA : pn * 256 + l.coord 0 < d.N := off_mul_add_lt_of_lt_div hpn hl0
  have hB : 128 * t + l.coord 1 < d.K := by
    have h := off_mul_add_lt_of_lt_div (p := t) (c := 128) (M := d.K) (l := l.coord 1) ht hl1
    omega
  have hm' := off_mul_add_lt_mul hA hB
  omega

/-! ## `c` (the block a program stores, at its lane offsets) -/

theorem cDesc_valid {d : Inst} (hd : d.ok) : (cDesc d).Valid := by
  have hM := hd.M_ok
  have hN := hd.N_ok
  simp only [TensorDesc.Valid, TensorDesc.rank, cDesc, List.length_cons, List.length_nil,
    List.getLast?_cons_cons, Buf.ptr, DType.bytes, Nat.reduceAdd, Nat.reduceLeDiff]
  refine ⟨trivial, trivial, trivial, trivial, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro k hk
    interval_cases k
    · simp only [List.getD_cons_zero]
      exact ⟨by omega, by omega⟩
    · simp only [List.getD_cons_succ, List.getD_cons_zero]
      exact ⟨by omega, by omega⟩
  · simpa using hd.base_al.2.2.2.2
  · rfl
  · intro k hk
    have hk0 : k = 0 := by omega
    subst hk0
    simp only [List.getD_cons_zero]
    have hNd := N_div d hd
    omega
  · simp
  · simp

/-- The store of the program's tile is an access the descriptor allows. -/
theorem cDesc_access {d : Inst} (hd : d.ok) (i : Fin (launch d).size) :
    (cDesc d).Access (laneOffs d i) [128, 256] := by
  refine ⟨cDesc_valid hd, rfl, rfl, rfl, ?_⟩
  simp only [laneOffs, List.getLast?_cons_cons, List.getLast?_singleton, Option.getD_some,
    DType.bytes]
  rw [← Int.dvd_iff_emod_eq_zero]
  exact ⟨((i : Nat) / gridM d: Nat) * 32, by push_cast; ring⟩

/-- A lane of the store is in bounds exactly when `laneMask` says so. -/
theorem cDesc_laneMask {d : Inst} (i : Fin (launch d).size) (l : Idx [128, 256]) :
    (cDesc d).inbAt (laneCoord d i l) = laneMask d i l := by
  rfl

end Claims.BlockScaledMxfp8
