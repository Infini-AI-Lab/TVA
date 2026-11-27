import Claim
import TVA.Proof

/-!
# `swiglu_fwd_bf16`: the geometry of one program

Program `p` of the grid `cdiv(T, B)` owns the flat elements `blk d p = [p·B, p·B + B) ∩ [0, T)`;
lane `i` is element `gk d p i = p·B + i`, enabled when it is below `T = N·D`. Flat element `k` is
row `k / D`, column `k % D`: element `inIdx d s k = (k / D)·s + k % D` of an input with row stride
`s`, and element `k` of `z` (row stride `D`). This file names those maps and sets, the footprints
`Rset`, `Wset` and the local postcondition `Q`, and proves that the footprints of distinct
programs do not meet, that no program reads what it writes, and that the programs' `Q` give
`Post`.
-/

namespace Claims.SwigluFwd

open TVA

/-- The number of programs, `cdiv(T, B)`. -/
def G (d : Inst) : Nat := (d.N * d.D + d.B - 1) / d.B

/-- The flat element of lane `i` of program `p`. -/
def gk (d : Inst) (p : Nat) (i : Idx [d.B]) : Nat := p * d.B + i.1.val

/-- Lane `i` of program `p` is enabled: its flat element is below `T = N·D`. -/
def en (d : Inst) (p : Nat) (i : Idx [d.B]) : Bool := decide (p * d.B + i.1.val < d.N * d.D)

/-- The element of an input with row stride `s` at flat element `k`: `(k / D)·s + k % D`. -/
def inIdx (d : Inst) (s k : Nat) : Nat := k / d.D * s + k % d.D

/-- The element of an input with row stride `s` that lane `i` of program `p` reads. -/
def gin (d : Inst) (s p : Nat) (i : Idx [d.B]) : Nat := inIdx d s (gk d p i)

/-- The flat elements program `p` owns. -/
def blk (d : Inst) (p : Nat) : Set Nat :=
  {k | p * d.B ≤ k ∧ k < p * d.B + d.B ∧ k < d.N * d.D}

/-- Program `p` reads its elements of `x` and `y`. -/
def Rset (d : Inst) (p : Nat) : Set Int :=
  d.x.elems (Buf.img (gin d d.sx p) (en d p)) ∪ d.y.elems (Buf.img (gin d d.sy p) (en d p))

/-- Program `p` writes its elements of `z`. -/
def Wset (d : Inst) (p : Nat) : Set Int := d.z.elems (blk d p)

/-- After program `p`, each of its elements of `z` is right for the initial inputs. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ k ∈ blk d p, ∃ a b o : BF16,
    init.read .bf16 (d.x.ptr (inIdx d d.sx k)).addr = some a ∧
    init.read .bf16 (d.y.ptr (inIdx d d.sy k)).addr = some b ∧
    m.read .bf16 (d.z.ptr k).addr = some o ∧ ZOk a b o

/-! ## Sets -/

theorem b_pos {d : Inst} (hd : d.ok) : 0 < d.B := by
  obtain ⟨e, he⟩ := hd.b_pow2; rw [he]; exact Nat.two_pow_pos e

/-- The enabled lanes address exactly the block. -/
theorem img_gk (d : Inst) (p : Nat) : Buf.img (gk d p) (en d p) = blk d p := by
  ext k
  constructor
  · rintro ⟨i, hi, rfl⟩
    simp only [en, decide_eq_true_eq] at hi
    exact ⟨Nat.le_add_right _ _, by have := i.1.isLt; simp only [gk]; omega, hi⟩
  · rintro ⟨h1, h2, h3⟩
    refine ⟨(⟨k - p * d.B, by omega⟩, ()), ?_, ?_⟩
    · simp only [en, decide_eq_true_eq]; omega
    · simp only [gk]; omega

/-- A flat element below `N·D` is row `k / D < N`, column `k % D < D`. -/
theorem row_col {d : Inst} (hd : d.ok) {k : Nat} (hk : k < d.N * d.D) :
    k / d.D < d.N ∧ k % d.D < d.D :=
  ⟨(Nat.div_lt_iff_lt_mul hd.d_pos).mpr hk, Nat.mod_lt _ hd.d_pos⟩

/-- The input elements a program reads are among the launch's. -/
theorem img_gin_sub {d : Inst} (hd : d.ok) (s p : Nat) :
    Buf.img (gin d s p) (en d p) ⊆ inSet d s := by
  rintro _ ⟨i, hi, rfl⟩
  simp only [en, decide_eq_true_eq] at hi
  obtain ⟨h1, h2⟩ := row_col hd hi
  exact ⟨_, _, h1, h2, rfl⟩

/-- A block is among the first `N·D` elements. -/
theorem blk_sub (d : Inst) (p : Nat) : blk d p ⊆ outSet d := fun _ h => h.2.2

/-- The blocks of distinct programs are disjoint. -/
theorem blk_disjoint {d : Inst} (hd : d.ok) {p q : Nat} (hpq : p ≠ q) :
    Disjoint (blk d p) (blk d q) := by
  rw [Set.disjoint_left]
  rintro k ⟨hp1, hp2, -⟩ ⟨hq1, hq2, -⟩
  have hB := b_pos hd
  have e1 : k / d.B = p := Nat.div_eq_of_lt_le hp1 (by rw [Nat.succ_mul]; exact hp2)
  have e2 : k / d.B = q := Nat.div_eq_of_lt_le hq1 (by rw [Nat.succ_mul]; exact hq2)
  exact hpq (e1.symm.trans e2)

/-- Element `k < N·D` is in the block of program `k / B`, which is in the grid. -/
theorem mem_blk {d : Inst} (hd : d.ok) {k : Nat} (hk : k < d.N * d.D) :
    k ∈ blk d (k / d.B) ∧ k / d.B < G d := by
  have hB := b_pos hd
  have h1 := Nat.div_mul_le_self k d.B
  have h2 := Nat.lt_div_mul_add (a := k) hB
  refine ⟨⟨h1, h2, hk⟩, ?_⟩
  apply Nat.lt_of_succ_le
  rw [G, Nat.le_div_iff_mul_le hB, Nat.succ_mul]
  omega

/-- A program's footprints: it writes inside `z`'s first `N·D` elements and reads inside the
inputs' elements, which do not meet. -/
theorem w_sub (d : Inst) (p : Nat) : Wset d p ⊆ d.z.elems (outSet d) :=
  d.z.elems_mono (blk_sub d p)

theorem r_sub {d : Inst} (hd : d.ok) (p : Nat) :
    Rset d p ⊆ d.x.elems (inSet d d.sx) ∪ d.y.elems (inSet d d.sy) :=
  Set.union_subset_union (d.x.elems_mono (img_gin_sub hd _ _)) (d.y.elems_mono (img_gin_sub hd _ _))

theorem w_r_disjoint {d : Inst} (hd : d.ok) (p q : Nat) : Disjoint (Wset d p) (Rset d q) := by
  refine Disjoint.mono (w_sub d p) (r_sub hd q) ?_
  exact Disjoint.union_right hd.z_x hd.z_y

/-- The flat element of row `r`, column `c`. -/
theorem div_mod_flat {d : Inst} (hd : d.ok) {r c : Nat} (hc : c < d.D) :
    (r * d.D + c) / d.D = r ∧ (r * d.D + c) % d.D = c := by
  have hD := hd.d_pos
  constructor
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hD, Nat.div_eq_of_lt hc, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc]

end Claims.SwigluFwd
