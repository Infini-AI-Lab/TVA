import TVA.Proof
import Claim
import Proof.Dot

/-!
# The tiles are disjoint and away from the inputs, and the programs' results assemble into `Post`.

The tiles are disjoint and away from the inputs, and the programs' results assemble into `Post`.
-/


/-!
# A program does not write what it reads

Its tile of `c` lies in the matrix `c`, which `c_a` and `c_b` put away from `a` and `b`.
-/

namespace Claims.Matmul

open TVA

/-- A program does not write what it reads. -/
theorem read_write_disjoint {d : Inst} (hd : d.ok) (init : Mem) (_hpre : Pre d init)
    (i : Fin (launch d).size) : Disjoint (R d init i) (W d init i) := by
  have ha : aSet d ⊆ aIdx d := by
    rintro e ⟨r, k, hr, hk, rfl⟩
    exact ⟨r, k, hr, hk, rfl⟩
  have hb : bSet d ⊆ bIdx d := by
    rintro e ⟨k, c, hk, hc, rfl⟩
    exact ⟨k, c, hk, hc, rfl⟩
  have hc : tileSet d (i : Nat) ⊆ cIdx d := by
    rintro e ⟨r, c, _hr, _hc, hM, hN, rfl⟩
    exact ⟨cRow d (i : Nat) r, cCol d (i : Nat) c, hM, hN, rfl⟩
  show Disjoint (d.a.elems (aSet d) ∪ d.b.elems (bSet d)) (d.c.elems (tileSet d (i : Nat)))
  rw [Set.disjoint_union_left]
  exact ⟨Disjoint.mono (Buf.elems_mono d.a ha) (Buf.elems_mono d.c hc) hd.c_a.symm,
    Disjoint.mono (Buf.elems_mono d.b hb) (Buf.elems_mono d.c hc) hd.c_b.symm⟩

end Claims.Matmul

/- ===== TD ===== -/

/-!
# Two programs' tiles are disjoint

Tiles of distinct programs hold distinct matrix elements (`c_inj`), so their bytes are disjoint;
and a tile is away from the inputs (`c_a`, `c_b`).
-/

namespace Claims.Matmul

open TVA

/-- A tile of `c`'s storage lies in the matrix `c`. -/
theorem tileSet_subset_cIdx (d : Inst) (p : Nat) : tileSet d p ⊆ cIdx d := by
  rintro e ⟨r, c, hr, hc, hM, hN, rfl⟩
  exact ⟨cRow d p r, cCol d p c, hM, hN, rfl⟩

/-- Two distinct programs' tiles of `c` share no element. -/
theorem tileSet_disjoint {d : Inst} (hd : d.ok) {p q : Nat}
    (hp : p < grid d) (hq : q < grid d) (hpq : p ≠ q) :
    Disjoint (tileSet d p) (tileSet d q) := by
  rw [Set.disjoint_left]
  rintro e ⟨r, c, hr, hc, hM, hN, he⟩ ⟨r', c', hr', hc', hM', hN', he'⟩
  have heq : cRow d p r * d.stride_cm + cCol d p c * d.stride_cn
      = cRow d q r' * d.stride_cm + cCol d q c' * d.stride_cn :=
    he.symm.trans he'
  obtain ⟨h1, h2⟩ := hd.c_inj (cRow d p r) (cCol d p c) (cRow d q r') (cCol d q c')
    hM hN hM' hN' heq
  simp only [cRow] at h1
  simp only [cCol] at h2
  have hpM : pidM d p = pidM d q := by omega
  have hpN : pidN d p = pidN d q := by omega
  exact hpq (pid_inj hp hq hpM hpN)

/-- One program's tile is disjoint from everything another program touches. -/
theorem tiles_disjoint {d : Inst} (hd : d.ok) (init : Mem) (_hpre : Pre d init)
    (i j : Fin (launch d).size) (hij : i ≠ j) :
    Disjoint (W d init i) (R d init j ∪ W d init j) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hj : (j : Nat) < grid d := lt_of_lt_of_eq j.isLt (launch_size d)
  have hsubA : tileSet d (i : Nat) ⊆ cIdx d := tileSet_subset_cIdx d (i : Nat)
  have hca : Disjoint (d.c.elems (tileSet d (i : Nat))) (d.a.elems (aSet d)) :=
    hd.c_a.mono (d.c.elems_mono hsubA) (le_refl _)
  have hcb : Disjoint (d.c.elems (tileSet d (i : Nat))) (d.b.elems (bSet d)) :=
    hd.c_b.mono (d.c.elems_mono hsubA) (le_refl _)
  have hidx : Disjoint (tileSet d (i : Nat)) (tileSet d (j : Nat)) :=
    tileSet_disjoint hd hi hj fun h => hij (Fin.ext h)
  have hW : Disjoint (d.c.elems (tileSet d (i : Nat))) (d.c.elems (tileSet d (j : Nat))) :=
    Buf.elems_disjoint d.c hidx
  show Disjoint (d.c.elems (tileSet d (i : Nat)))
    ((d.a.elems (aSet d) ∪ d.b.elems (bSet d)) ∪ d.c.elems (tileSet d (j : Nat)))
  rw [Set.disjoint_union_right]
  refine ⟨?_, hW⟩
  rw [Set.disjoint_union_right]
  exact ⟨hca, hcb⟩

end Claims.Matmul

/- ===== PO ===== -/

/-!
# The programs' postconditions assemble

Every in-range `(i, j)` lies in exactly one program's tile, and every element of `c`'s storage
outside the matrix lies in no tile.
-/

namespace Claims.Matmul

open TVA

/-- **The tile map is onto the tile grid**: every pair of row and column tiles is some program's
tile. -/
theorem tile_surj (d : Inst) {a b : Nat} (ha : a < npm d) (hb : b < npn d) :
    ∃ p, p < grid d ∧ pidM d p = a ∧ pidN d p = b := by
  have hnpn : 0 < npn d := Nat.lt_of_le_of_lt (Nat.zero_le b) hb
  have hG : 0 < 8 * npn d := Nat.mul_pos (by norm_num) hnpn
  set g := a / 8 with hg
  set f := 8 * g with hf
  have hfa : f ≤ a := by
    have h := Nat.div_add_mod a 8
    omega
  set r := a - f with hr
  have hfr : f + r = a := by
    rw [hr]
    exact Nat.add_sub_cancel' hfa
  have hr8 : r < 8 := by
    have h := Nat.div_add_mod a 8
    have h8 := Nat.mod_lt a (by norm_num : (0 : Nat) < 8)
    omega
  set gs := min (npm d - f) 8 with hgs
  have hgs8 : gs ≤ 8 := by
    rw [hgs]
    exact Nat.min_le_right _ _
  have hgsnf : gs ≤ npm d - f := by
    rw [hgs]
    exact Nat.min_le_left _ _
  have hgs_pos : 0 < gs := by
    rw [hgs]
    exact Nat.lt_min.mpr ⟨Nat.sub_pos_of_lt (by omega), by norm_num⟩
  have hfgs : f + gs ≤ npm d := by omega
  set m := r + gs * b with hm
  set q := 8 * npn d * g + m with hq
  have hm_lt8 : m < 8 * npn d := by
    have h1 : gs * b ≤ 8 * b := Nat.mul_le_mul_right b hgs8
    have h2 : 8 * (b + 1) ≤ 8 * npn d := Nat.mul_le_mul_left 8 (by omega)
    rw [hm]
    omega
  have hqdiv : q / (8 * npn d) = g := by
    have hq' : q = m + 8 * npn d * g := by
      rw [hq]
      omega
    rw [hq', Nat.add_mul_div_left m g hG, Nat.div_eq_of_lt hm_lt8, Nat.zero_add]
  have hqmod : q % (8 * npn d) = m := by
    have h := Nat.div_add_mod q (8 * npn d)
    rw [hqdiv] at h
    omega
  have hfpm : fpm d q = f := by
    have h1 : grp d q = g := hqdiv
    show grp d q * 8 = f
    rw [h1]
    exact Nat.mul_comm g 8
  have hgsize : gsize d q = gs := by
    show min (npm d - fpm d q) 8 = gs
    rw [hfpm, ← hgs]
  have hrgs : r < gs := by
    rw [hgs]
    exact Nat.lt_min.mpr ⟨by omega, hr8⟩
  have hmodm : m % gs = r := by
    rw [hm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hrgs]
  have hdivm : m / gs = b := by
    rw [hm, Nat.add_mul_div_left r b hgs_pos, Nat.div_eq_of_lt hrgs, Nat.zero_add]
  refine ⟨q, ?_, ?_, ?_⟩
  · rw [grid]
    have hmlt : m < gs * npn d := by
      have h1 : r + gs * b < gs * (b + 1) := by
        rw [Nat.mul_add, Nat.mul_one]
        omega
      have h2 : gs * (b + 1) ≤ gs * npn d := Nat.mul_le_mul_left gs (by omega)
      rw [hm]
      omega
    have hsum : 8 * npn d * g + m < (f + gs) * npn d := by
      have h3 : (f + gs) * npn d = 8 * npn d * g + gs * npn d := by
        rw [hf]
        ring
      rw [h3]
      omega
    have h4 : (f + gs) * npn d ≤ npm d * npn d := Nat.mul_le_mul_right _ hfgs
    rw [hq]
    omega
  · simp only [pidM]
    rw [hfpm, hqmod, hgsize, hmodm]
    exact hfr
  · simp only [pidN]
    rw [hqmod, hgsize, hdivm]

/-- The programs' postconditions assemble into `Post`. -/
theorem post_of_tiles {d : Inst} (_hd : d.ok) (init : Mem) (_hpre : Pre d init) (fin : Mem)
    (hQ : ∀ i, ∃ m, Q d init i m ∧ ∀ x ∈ R d init i ∪ W d init i, fin x = m x)
    (hfr : ∀ x, (∀ i, x ∉ W d init i) → fin x = init x) : Post d init fin := by
  constructor
  · intro i j hi hj
    have hanpm : i / 128 < npm d := by
      simp only [npm]
      have h := Nat.div_add_mod (d.M + 127) 128
      have hr := Nat.mod_lt (d.M + 127) (by norm_num : (0 : Nat) < 128)
      exact Nat.div_lt_of_lt_mul (by omega)
    have hbnpn : j / 256 < npn d := by
      simp only [npn]
      have h := Nat.div_add_mod (d.N + 255) 256
      have hr := Nat.mod_lt (d.N + 255) (by norm_num : (0 : Nat) < 256)
      exact Nat.div_lt_of_lt_mul (by omega)
    obtain ⟨p, hp_grid, hpM, hpN⟩ := tile_surj d hanpm hbnpn
    have hp_size : p < (launch d).size := by
      rw [launch_size]
      exact hp_grid
    obtain ⟨m, hQm, hagree⟩ := hQ ⟨p, hp_size⟩
    have hrow : cRow d p (i % 128) = i := by
      simp only [cRow]
      rw [hpM]
      have h := Nat.div_add_mod i 128
      omega
    have hcol : cCol d p (j % 256) = j := by
      simp only [cCol]
      rw [hpN]
      have h := Nat.div_add_mod j 256
      omega
    have hr : i % 128 < 128 := Nat.mod_lt _ (by norm_num)
    have hc : j % 256 < 256 := Nat.mod_lt _ (by norm_num)
    have hout : OutSpec d init m i j := by
      have h := hQm (i % 128) (j % 256) hr hc (by rw [hrow]; exact hi) (by rw [hcol]; exact hj)
      rwa [hrow, hcol] at h
    obtain ⟨v, hread, hrest⟩ := hout
    refine ⟨v, ?_, hrest⟩
    have hk : i * d.stride_cm + j * d.stride_cn ∈ tileSet d p := by
      refine ⟨i % 128, j % 256, hr, hc, ?_, ?_, ?_⟩
      · rw [hrow]; exact hi
      · rw [hcol]; exact hj
      · rw [hrow, hcol]
    have hspan : Mem.span .f16
          (d.c.ptr (↑(i * d.stride_cm + j * d.stride_cn) : Int)).addr ⊆
        R d init ⟨p, hp_size⟩ ∪ W d init ⟨p, hp_size⟩ :=
      fun x hx => Or.inr (Buf.span_sub_elems d.c hk hx)
    have hcongr : fin.read .f16
          (d.c.ptr (↑(i * d.stride_cm + j * d.stride_cn) : Int)).addr =
        m.read .f16 (d.c.ptr (↑(i * d.stride_cm + j * d.stride_cn) : Int)).addr :=
      Mem.read_congr (fun x hx => hagree x (hspan hx))
    rw [hcongr]
    exact hread
  · intro z hz
    refine hfr z (fun i hzi => hz ?_)
    have hsub : tileSet d (i : Nat) ⊆ cIdx d := by
      rintro e ⟨r, c, hr, hc, hrowm, hcolm, rfl⟩
      exact ⟨cRow d (i : Nat) r, cCol d (i : Nat) c, hrowm, hcolm, rfl⟩
    simpa only [W] using Buf.elems_mono d.c hsub hzi

end Claims.Matmul
