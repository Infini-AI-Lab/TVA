import Claim
import TVA.Proof

/-!
# The geometry of one program

Program `t` of the grid `(n_rows,)` handles row `t`. Lane `i` of its block is column `i`, enabled
when `i < d`. This file gives the grid facts, the value of `row` and of the mask, the `Buf.Addrs`
facts of the two loads and the store (with their index maps `gG`, `gU`, `gO`), the bounds of the
accessed elements, and the footprints `R`, `W` with their disjointness.
-/

namespace Claims.GeluAndMulSparse

open TVA

/-! ## The grid -/

/-- The launch has one program per row. -/
theorem launch_size (d : Inst) : (launch d).size = d.T := Launch.size_grid1 _ _ _

/-- Program `i` of the launch has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = i := by
  have hi : (i : Nat) < d.T := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- `BLOCK_SIZE < 2^16`. -/
theorem b_lt16 {d : Inst} (hd : d.ok) : d.B < 2 ^ 16 := by
  have h1 := hd.b_lt; have h2 := hd.d_le; omega

/-- The launch is valid: fewer than `2^31` programs and the block `[BLOCK_SIZE]` a power of two
within `2^20` elements. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 hd.t_lt (fun s hs => ?_)
  simp only [List.mem_singleton] at hs
  subst hs
  obtain ⟨e, he⟩ := hd.b_pow2
  refine ⟨fun k hk => ⟨e, ?_⟩, ?_⟩
  · simp only [List.mem_singleton] at hk
    rw [hk, he]
  · have := b_lt16 hd
    simp only [List.prod_cons, List.prod_nil, Nat.mul_one]
    omega

/-! ## Lanes and index maps -/

/-- Lane `i` is enabled: its column is below `d`. -/
def en (d : Inst) (i : Idx [d.B]) : Bool := decide (i.1.val < d.d)

/-- The element of `x` lane `i` of program `t` reads as `gate`. -/
def gG (d : Inst) (t : Nat) (i : Idx [d.B]) : Nat := gidx d t i.1.val

/-- The element of `x` lane `i` of program `t` reads as `up`. -/
def gU (d : Inst) (t : Nat) (i : Idx [d.B]) : Nat := uidx d t i.1.val

/-- The element of `output` lane `i` of program `t` writes. -/
def gO (d : Inst) (t : Nat) (i : Idx [d.B]) : Nat := oidx d t i.1.val

/-- The lane of column `j`. -/
def lane (d : Inst) (j : Nat) (h : j < d.B) : Idx [d.B] := (⟨j, h⟩, ())

/-- An enabled lane's column is below `d`. -/
theorem lt_of_en {d : Inst} {i : Idx [d.B]} (h : en d i = true) : i.1.val < d.d :=
  of_decide_eq_true h

/-! ## Values -/

/-- A natural below `2^63` fits a signed 64-bit integer. -/
theorem sfits64 {n : Nat} (h : n < 2 ^ 63) : IntV.SFits 64 (n : Int) := by
  have h' : (n : Int) < 2 ^ 63 := by exact_mod_cast h
  simp only [IntV.SFits]
  constructor <;> norm_num; omega

/-- The sign-extended program id `row` has value `i`. -/
theorem row_toInt {d : Inst} (hd : d.ok) {P : Platform} (i : Fin (launch d).size)
    {r : Tensor [] I64}
    (hr : ∀ j, ConvRel P .i32 .i64 ((Tensor.scalar (tl.program_id ((launch d).ctx i) 0)).get j)
      (r.get j)) :
    r.item.toInt = (i.val : Int) := by
  have hi : i.val < 2 ^ 31 := lt_of_lt_of_le (lt_of_lt_of_eq i.isLt (launch_size d)) hd.t_lt.le
  have hi32 : IntV.toInt (tl.program_id ((launch d).ctx i) 0) = (i.val : Int) := by
    simp only [tl.program_id, launch_pid]
    exact IntV.I32.toInt_ofNat hi
  have h1 : r.item = IntV.ofInt (IntV.toInt (tl.program_id ((launch d).ctx i) 0)) := hr ()
  rw [h1, hi32]
  exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩

/-- Lane `i` of `tl.arange(0, BLOCK_SIZE)` has value `i`. -/
theorem arange_get {d : Inst} (hd : d.ok) (i : Idx [d.B]) :
    ((tl.arange 0 d.B).get i).toInt = (i.1.val : Int) := by
  rcases i with ⟨i, ⟨⟩⟩
  exact IntV.I32.toInt_tl_arange (n := d.B) (by have := b_lt16 hd; omega) i

/-- The mask `offsets < d` is `en`. -/
theorem mask_get {d : Inst} (hd : d.ok) (i : Idx [d.B]) :
    ((tl.arange 0 d.B).ltS (IntV.ofNat d.d : I32)).get i = en d i := by
  rw [Tensor.get_ltS, IntV.lt_eq, arange_get hd i,
    IntV.I32.toInt_ofNat (lt_of_le_of_lt hd.d_le (by norm_num))]
  simp only [en, Nat.cast_lt]

/-- `row * 2 * d` and `row * d` in `int64` have their exact values. -/
theorem rowStart_toInt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) {row : I64}
    (hrow : row.toInt = t) :
    (row * IntV.ofNat 2 * IntV.ofNat d.d).toInt = ((t * (2 * d.d) : Nat) : Int) ∧
      (row * (IntV.ofNat d.d : I64)).toInt = ((t * d.d : Nat) : Int) := by
  have ht31 := hd.t_lt
  have hdl := hd.d_le
  have h1 : t * (2 * d.d) < 2 ^ 63 := by
    have : t * (2 * d.d) ≤ 2 ^ 31 * 65536 := Nat.mul_le_mul (by omega) (by omega)
    omega
  have h2 : t * d.d < 2 ^ 63 := by
    have : t * d.d ≤ 2 ^ 31 * 32768 := Nat.mul_le_mul (by omega) hdl
    omega
  have hd64 : (IntV.ofNat d.d : I64).toInt = (d.d : Int) := IntV.I64.toInt_ofNat (by omega)
  have h2' : (row * IntV.ofNat 2 : I64).toInt = (t : Int) * 2 :=
    IntV.toInt_mul hrow (IntV.I64.toInt_ofNat (by norm_num)) (by
      have := sfits64 (n := t * 2) (by omega); push_cast at this; exact this)
  refine ⟨?_, ?_⟩
  · rw [IntV.toInt_mul h2' hd64]
    · push_cast; ring
    · have := sfits64 h1; push_cast at this; convert this using 1; ring
  · rw [IntV.toInt_mul hrow hd64]
    · push_cast; ring
    · have := sfits64 h2; push_cast at this; exact this

/-! ## Addresses -/

/-- The addresses of the `gate` load. -/
theorem addrs_gate {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) {row : I64}
    (hrow : row.toInt = t) :
    d.x.Addrs (tl.addrs (d.x.ptr 0 + row * IntV.ofNat 2 * IntV.ofNat d.d + tl.arange 0 d.B)
      (some ((tl.arange 0 d.B).ltS (IntV.ofNat d.d)))) (gG d t) (en d) := by
  refine Buf.addrs_of_get d.x (fun p => mask_get hd p) (fun p _ => ?_)
  rw [Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset, Buf.ptr_offset,
    (rowStart_toInt hd ht hrow).1, arange_get hd p]
  simp only [gG, gidx]
  congr 1; push_cast; ring

/-- The addresses of the `up` load. -/
theorem addrs_up {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) {row : I64}
    (hrow : row.toInt = t) :
    d.x.Addrs (tl.addrs (d.x.ptr 0 + row * IntV.ofNat 2 * IntV.ofNat d.d + (IntV.ofNat d.d : I32) +
      tl.arange 0 d.B) (some ((tl.arange 0 d.B).ltS (IntV.ofNat d.d)))) (gU d t) (en d) := by
  refine Buf.addrs_of_get d.x (fun p => mask_get hd p) (fun p _ => ?_)
  rw [Tensor.get_ptrAdd, Ptr.add_int, Ptr.add_int, Buf.ptr_offset, Buf.ptr_offset,
    Buf.ptr_offset, (rowStart_toInt hd ht hrow).1, arange_get hd p,
    IntV.I32.toInt_ofNat (lt_of_le_of_lt hd.d_le (by norm_num))]
  simp only [gU, uidx]
  congr 1; push_cast; ring

/-- The addresses of the `output` store. -/
theorem addrs_out {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) {row : I64}
    (hrow : row.toInt = t) :
    d.out.Addrs (tl.addrs (d.out.ptr 0 + row * (IntV.ofNat d.d : I64) + tl.arange 0 d.B)
      (some ((tl.arange 0 d.B).ltS (IntV.ofNat d.d)))) (gO d t) (en d) := by
  refine Buf.addrs_of_get d.out (fun p => mask_get hd p) (fun p _ => ?_)
  rw [Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset, Buf.ptr_offset,
    (rowStart_toInt hd ht hrow).2, arange_get hd p]
  simp only [gO, oidx]
  congr 1; push_cast; ring

/-! ## Bounds and injectivity -/

/-- Row `t < T` of `x` ends within the `T` rows. -/
theorem rowX_end {d : Inst} {t : Nat} (ht : t < d.T) : t * (2 * d.d) + 2 * d.d ≤ d.T * (2 * d.d) := by
  have := Nat.mul_le_mul_right (2 * d.d) (Nat.succ_le_of_lt ht)
  rw [Nat.succ_mul] at this; exact this

/-- Row `t < T` of `output` ends within the `T` rows. -/
theorem rowO_end {d : Inst} {t : Nat} (ht : t < d.T) : t * d.d + d.d ≤ d.T * d.d := by
  have := Nat.mul_le_mul_right d.d (Nat.succ_le_of_lt ht)
  rw [Nat.succ_mul] at this; exact this

/-- The `gate` element of an enabled lane is an element of `x`. -/
theorem gG_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    ∀ i, en d i = true → gG d t i < d.x.len := by
  intro i hi
  have := lt_of_en hi; have := rowX_end (d := d) ht; have := hd.x_len
  simp only [gG, gidx]; omega

/-- The `up` element of an enabled lane is an element of `x`. -/
theorem gU_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    ∀ i, en d i = true → gU d t i < d.x.len := by
  intro i hi
  have := lt_of_en hi; have := rowX_end (d := d) ht; have := hd.x_len
  simp only [gU, uidx]; omega

/-- The `output` element of an enabled lane is an element of `output`. -/
theorem gO_lt {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    ∀ i, en d i = true → gO d t i < d.out.len := by
  intro i hi
  have := lt_of_en hi; have := rowO_end (d := d) ht; have := hd.out_len
  simp only [gO, oidx]; omega

/-- Distinct enabled lanes write distinct elements. -/
theorem gO_inj {d : Inst} (t : Nat) :
    ∀ i j, en d i = true → en d j = true → gO d t i = gO d t j → i = j := by
  rintro ⟨i, ⟨⟩⟩ ⟨j, ⟨⟩⟩ _ _ h
  simp only [gO, oidx] at h
  have : i = j := Fin.ext (by omega)
  subst this; rfl

/-! ## Footprints -/

/-- The elements of row `t` of `x` (its `gate` and `up`). -/
def rowX (d : Inst) (t : Nat) : Set Nat := {k | t * (2 * d.d) ≤ k ∧ k < t * (2 * d.d) + 2 * d.d}

/-- The elements of row `t` of `output`. -/
def rowO (d : Inst) (t : Nat) : Set Nat := {k | t * d.d ≤ k ∧ k < t * d.d + d.d}

/-- The bytes program `i` reads: row `i` of `x`. -/
def R (d : Inst) (i : Fin (launch d).size) : Set Int := d.x.elems (rowX d i.val)

/-- The bytes program `i` writes: row `i` of `output`. -/
def W (d : Inst) (i : Fin (launch d).size) : Set Int := d.out.elems (rowO d i.val)

/-- The `gate` load touches row `t` of `x`. -/
theorem imgG_sub (d : Inst) (t : Nat) : Buf.img (gG d t) (en d) ⊆ rowX d t := by
  rintro k ⟨i, hi, rfl⟩
  have := lt_of_en hi
  simp only [rowX, gG, gidx, Set.mem_ofPred_eq]; omega

/-- The `up` load touches row `t` of `x`. -/
theorem imgU_sub (d : Inst) (t : Nat) : Buf.img (gU d t) (en d) ⊆ rowX d t := by
  rintro k ⟨i, hi, rfl⟩
  have := lt_of_en hi
  simp only [rowX, gU, uidx, Set.mem_ofPred_eq]; omega

/-- The store writes exactly row `t` of `output`. -/
theorem imgO_eq (d : Inst) (t : Nat) (hd : d.ok) : Buf.img (gO d t) (en d) = rowO d t := by
  ext k
  constructor
  · rintro ⟨i, hi, rfl⟩
    have := lt_of_en hi
    simp only [rowO, gO, oidx, Set.mem_ofPred_eq]; omega
  · rintro ⟨h1, h2⟩
    have hb := hd.d_le_b
    refine ⟨lane d (k - t * d.d) (by omega), ?_, ?_⟩
    · exact decide_eq_true (show k - t * d.d < d.d by omega)
    · simp only [gO, oidx, lane]; omega

/-- Row `t` of `x` lies in the elements the launch reads. -/
theorem rowX_sub {d : Inst} {t : Nat} (ht : t < d.T) : rowX d t ⊆ inSet d := by
  rintro k ⟨_, h2⟩
  have := rowX_end (d := d) ht
  simp only [inSet, Set.mem_ofPred_eq]; omega

/-- Row `t` of `output` lies in the elements the launch writes. -/
theorem rowO_sub {d : Inst} {t : Nat} (ht : t < d.T) : rowO d t ⊆ outSet d := by
  rintro k ⟨_, h2⟩
  have := rowO_end (d := d) ht
  simp only [outSet, Set.mem_ofPred_eq]; omega

/-- Distinct rows of `output` are disjoint. -/
theorem rowO_disjoint {d : Inst} {s t : Nat} (h : s ≠ t) : Disjoint (rowO d s) (rowO d t) := by
  have key : ∀ a b : Nat, a < b → ∀ k, k ∈ rowO d a → k ∉ rowO d b := by
    rintro a b hab k ⟨_, h2⟩ ⟨h3, _⟩
    have := Nat.mul_le_mul_right d.d (Nat.succ_le_of_lt hab)
    rw [Nat.succ_mul] at this; omega
  rw [Set.disjoint_left]
  intro k hs ht
  rcases Nat.lt_or_gt_of_ne h with h' | h'
  · exact key s t h' k hs ht
  · exact key t s h' k ht hs

/-- A program's index is a row. -/
theorem fin_lt {d : Inst} (i : Fin (launch d).size) : i.val < d.T :=
  lt_of_lt_of_eq i.isLt (launch_size d)

/-- What program `i` writes is disjoint from what any other program `j` reads or writes. -/
theorem hdisj {d : Inst} (hd : d.ok) (i j : Fin (launch d).size) (hij : i ≠ j) :
    Disjoint (W d i) (R d j ∪ W d j) := by
  have hij' : i.val ≠ j.val := fun h => hij (Fin.ext h)
  refine Disjoint.union_right ?_ (Buf.elems_disjoint _ (rowO_disjoint hij'))
  exact hd.out_x.mono (Buf.elems_mono _ (rowO_sub (fin_lt i))) (Buf.elems_mono _ (rowX_sub (fin_lt j)))

/-- A program does not read what it writes. -/
theorem hRW {d : Inst} (hd : d.ok) (i : Fin (launch d).size) : Disjoint (R d i) (W d i) :=
  hd.out_x.symm.mono (Buf.elems_mono _ (rowX_sub (fin_lt i)))
    (Buf.elems_mono _ (rowO_sub (fin_lt i)))

/-- A byte outside the written elements of `output` is in no program's write footprint. -/
theorem not_mem_W {d : Inst} {z : Int} (hz : z ∉ d.out.elems (outSet d)) (i : Fin (launch d).size) :
    z ∉ W d i := fun h => hz (Buf.elems_mono _ (rowO_sub (fin_lt i)) h)

/-- The bytes of `output[t, j]` lie in program `t`'s write footprint. -/
theorem span_out_W {d : Inst} (_hd : d.ok) {t j : Nat} (ht : t < (launch d).size) (hj : j < d.d) :
    Mem.span .bf16 (d.out.ptr (oidx d t j : Int)).addr ⊆ W d ⟨t, ht⟩ :=
  Buf.span_sub_elems d.out (show oidx d t j ∈ rowO d t by simp only [rowO, oidx, Set.mem_ofPred_eq]; omega)

end Claims.GeluAndMulSparse
