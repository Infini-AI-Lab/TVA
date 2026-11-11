import Proof.Defs

/-!
# The programs' footprints are disjoint

Each program stores only its own block of `c`, and `c` is disjoint from `a` and `b`.
-/

namespace Claims.MatmulE5m2

open TVA

/-- The indices of a program's block are matrix indices. -/
theorem cBlk_subset_cIdx (d : Inst) (p : Nat) : cBlk d p ⊆ cIdx d := by
  rintro e ⟨x, y, hx, hy, hxm, hyn, rfl⟩
  exact ⟨ocm d p x, ocn d p y, hxm, hyn, rfl⟩

/-- Two different programs store disjoint blocks. -/
theorem cBlk_disjoint {d : Inst} (hd : d.ok) {p q : Nat} (hp : p < grid d) (hq : q < grid d)
    (hne : p ≠ q) : Disjoint (cBlk d p) (cBlk d q) := by
  rw [Set.disjoint_left]
  rintro e ⟨x, y, hx, hy, hxm, hyn, rfl⟩ ⟨x', y', hx', hy', hxm', hyn', he⟩
  -- the two addresses are equal, so the matrix indices are equal (`c_inj`)
  obtain ⟨hm, hn⟩ :=
    hd.c_inj (ocm d p x) (ocn d p y) (ocm d q x') (ocn d q y') hxm hyn hxm' hyn' he
  -- `ocm d p x = pidm d p * 128 + x` with `x < 128` recovers `pidm d p`
  have hdm : pidm d p = pidm d q := by
    have h1 : ocm d p x / 128 = pidm d p := by
      unfold ocm
      rw [Nat.mul_comm (pidm d p) 128, Nat.mul_add_div (by norm_num), Nat.div_eq_of_lt hx,
        Nat.add_zero]
    have h2 : ocm d q x' / 128 = pidm d q := by
      unfold ocm
      rw [Nat.mul_comm (pidm d q) 128, Nat.mul_add_div (by norm_num), Nat.div_eq_of_lt hx',
        Nat.add_zero]
    rw [← h1, ← h2, hm]
  -- and `ocn d p y = pidn d p * 256 + y` with `y < 256` recovers `pidn d p`
  have hdn : pidn d p = pidn d q := by
    have h1 : ocn d p y / 256 = pidn d p := by
      unfold ocn
      rw [Nat.mul_comm (pidn d p) 256, Nat.mul_add_div (by norm_num), Nat.div_eq_of_lt hy,
        Nat.add_zero]
    have h2 : ocn d q y' / 256 = pidn d q := by
      unfold ocn
      rw [Nat.mul_comm (pidn d q) 256, Nat.mul_add_div (by norm_num), Nat.div_eq_of_lt hy',
        Nat.add_zero]
    rw [← h1, ← h2, hn]
  exact hne (pid_inj hp hq hdm hdn)

/-- Two different programs' writes are disjoint. -/
theorem W_disjoint {d : Inst} {init : Mem} (hd : d.ok) {p q : Nat} (hp : p < grid d)
    (hq : q < grid d) (hne : p ≠ q) : Disjoint (W d init p) (W d init q) := by
  unfold W
  exact Buf.elems_disjoint d.c (cBlk_disjoint hd hp hq hne)

/-- A program's writes are disjoint from another program's reads. -/
theorem W_R_disjoint {d : Inst} {init : Mem} (hd : d.ok) {p q : Nat} :
    Disjoint (W d init p) (R d init q) := by
  unfold W R
  rw [Set.disjoint_union_right]
  refine ⟨hd.c_a.mono (Buf.elems_mono d.c (cBlk_subset_cIdx d p)) le_rfl,
    hd.c_b.mono (Buf.elems_mono d.c (cBlk_subset_cIdx d p)) le_rfl⟩

/-- A program's writes are disjoint from its own reads.

**NOTE**: this is the claim of `R_W_disjoint` below, which is missing the hypothesis
`hd : d.ok`; without it the conclusion is false (`d.a` and `d.c` may occupy the same
bytes, and `R`/`W` are both nonempty).  `R_W_disjoint` needs `d.ok` (or the two
disjointness hypotheses `hd.c_a`, `hd.c_b`) added to its statement. -/
theorem R_W_disjoint_of_ok (d : Inst) (init : Mem) (p : Nat) (hd : d.ok) :
    Disjoint (R d init p) (W d init p) := by
  unfold W R
  rw [Set.disjoint_union_left]
  refine ⟨hd.c_a.symm.mono le_rfl (Buf.elems_mono d.c (cBlk_subset_cIdx d p)),
    hd.c_b.symm.mono le_rfl (Buf.elems_mono d.c (cBlk_subset_cIdx d p))⟩

end Claims.MatmulE5m2
