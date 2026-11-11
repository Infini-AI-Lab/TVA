import Claim
import TVA.Proof
import Proof.Geometry
import Proof.Numeric

/-!
# The invariants of the loops: memory invariants, partial magnitudes, and the `OutSpec` of a finished
tile from the accumulator's bound.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-- The magnitude sum of the first `t` steps of the reduction: `accMag` cut after `t` steps. -/
def partMag (va vb : Nat → F16) (K t : Nat) : ℚ :=
  ∑ u ∈ Finset.range t, (|partSum (prodV va vb K) u| + stepMag (liftV va vb K) u)

/-- The bytes of the matrices `a` and `b` (the reads of every program). -/
def abSet (d : Inst) : Set Int := d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)

/-- **The accumulator invariant.** After `t` steps of the reduction of the tile `T`, every element
`(i, n)` is finite and within `errB` of the partial dot product of the inputs it has summed. -/
def CellInv (d : Inst) (init : Mem) (T t : Nat) (acc : Tensor [128, 256] F32) : Prop :=
  ∀ (iX : Fin 128) (nX : Fin 256) (va vb : Nat → F16),
    (∀ k, k < d.K →
      init.read .f16 (d.a.ptr ↑((pm d T * 128 + iX.val) * d.stride_am + k * d.stride_ak)).addr =
        some (va k) ∧ (va k).isFinite = true) →
    (∀ k, k < d.K →
      init.read .f16 (d.b.ptr ↑(k * d.stride_bk + (pn d T * 256 + nX.val) * d.stride_bn)).addr =
        some (vb k) ∧ (vb k).isFinite = true) →
    (acc.get2 iX nX).isFinite = true ∧
      |(acc.get2 iX nX).toRat - partSum (prodV va vb d.K) t| ≤
        errB rho' alpha' (partMag va vb d.K t) t

/-- **The memory invariant.** The reads of `a` and `b` still hold their initial values (they are
never written), and every byte that was present still is. -/
def MemInv (d : Inst) (init m : Mem) : Prop :=
  (∀ x ∈ abSet d, m x = init x) ∧ (∀ x, init x ≠ none → m x ≠ none)

/-- **The stores made so far.** Each of the first `T` trips of program `i` left its `c` tile as
`OutSpec` says. -/
def Done (d : Inst) (init : Mem) (i T : Nat) (m : Mem) : Prop :=
  ∀ u, u < T → ∀ a b : Nat, a < 128 → b < 256 →
    OutSpec d init m (pm d (i + u * d.NUM_SMS) * 128 + a) (pn d (i + u * d.NUM_SMS) * 256 + b)



/-- The lifted magnitude of a step is nonnegative (helper for the results below). -/
private theorem stepMag_liftV_nonneg_aux (va vb : Nat → F16) (K t : Nat) :
    0 ≤ stepMag (liftV va vb K) t := by
  unfold stepMag
  refine Finset.sum_nonneg fun k _ => ?_
  unfold liftV
  split_ifs with hh
  · have h1 := Device.liftMag_nonneg .f16 (va (64 * t + (k : Nat))).toRat
    have h2 := Device.liftMag_nonneg .f16 (vb (64 * t + (k : Nat))).toRat
    nlinarith
  · simp

/-- No steps, no magnitude. -/
theorem partMag_zero (va vb : Nat → F16) (K : Nat) : partMag va vb K 0 = 0 := by
  unfold partMag
  simp

/-- One more step adds its magnitude. -/
theorem partMag_succ (va vb : Nat → F16) (K t : Nat) :
    partMag va vb K (t + 1) =
      partMag va vb K t + (|partSum (prodV va vb K) t| + stepMag (liftV va vb K) t) := by
  unfold partMag
  rw [Finset.sum_range_succ]

/-- All steps give `accMag`. -/
theorem partMag_steps (va vb : Nat → F16) (K : Nat) :
    partMag va vb K (steps K) = accMag va vb K := by
  unfold partMag accMag
  rfl

/-- Partial magnitudes are nonnegative. -/
theorem partMag_nonneg (va vb : Nat → F16) (K t : Nat) : 0 ≤ partMag va vb K t := by
  unfold partMag
  refine Finset.sum_nonneg fun u _ => ?_
  have h1 : 0 ≤ |partSum (prodV va vb K) u| := abs_nonneg _
  have h2 := stepMag_liftV_nonneg_aux va vb K u
  linarith

/-- Partial magnitudes grow with the steps. -/
theorem partMag_mono {va vb : Nat → F16} {K t t' : Nat} (h : t ≤ t') :
    partMag va vb K t ≤ partMag va vb K t' := by
  unfold partMag
  refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_subset_range.2 h) ?_
  intro x _ _
  have h1 : 0 ≤ |partSum (prodV va vb K) x| := abs_nonneg _
  have h2 := stepMag_liftV_nonneg_aux va vb K x
  linarith

/-- A step's lifted magnitude is nonnegative. -/
theorem stepMag_liftV_nonneg (va vb : Nat → F16) (K t : Nat) :
    0 ≤ stepMag (liftV va vb K) t :=
  stepMag_liftV_nonneg_aux va vb K t

/-- `accMag` is nonnegative. -/
theorem accMag_nonneg (va vb : Nat → F16) (K : Nat) : 0 ≤ accMag va vb K := by
  rw [accMag_eq]
  refine Finset.sum_nonneg fun t _ => ?_
  have h1 : 0 ≤ |partSum (prodV va vb K) t| := abs_nonneg _
  have h2 := stepMag_liftV_nonneg_aux va vb K t
  linarith

/-- A partial sum is at most `accMag`. -/
theorem partSum_le_accMag {K : Nat} {va vb : Nat → F16} {t : Nat} (ht : t < steps K) :
    |partSum (prodV va vb K) t| ≤ accMag va vb K := by
  rw [accMag_eq]
  have hmem : t ∈ Finset.range (steps K) := Finset.mem_range.2 ht
  have hnn : ∀ x ∈ Finset.range (steps K),
      0 ≤ |partSum (prodV va vb K) x| + stepMag (liftV va vb K) x := by
    intro x _
    have h1 : 0 ≤ |partSum (prodV va vb K) x| := abs_nonneg _
    have h2 := stepMag_liftV_nonneg_aux va vb K x
    linarith
  have hsingle := Finset.single_le_sum (f := fun x =>
      |partSum (prodV va vb K) x| + stepMag (liftV va vb K) x) hnn hmem
  have h2 := stepMag_liftV_nonneg_aux va vb K t
  linarith



/-- The inner loop runs `steps K` trips. -/
theorem tripCount_steps (K : Nat) :
    tl.tripCount (0 : Int) ((steps K : Nat) : Int) (1 : Int) = steps K := by
  unfold tl.tripCount
  rw [Int.ediv_one]
  simp

/-- `OutSpec` reads only the two bytes of its own output element: a memory that agrees with `m`
there satisfies it too. -/
theorem outSpec_frame {d : Inst} {init m m' : Mem} {i j : Nat}
    (h : ∀ x ∈ Mem.span .f16 (d.c.ptr ((i * d.stride_cm + j * d.stride_cn : Nat) : Int)).addr,
      m' x = m x)
    (hs : OutSpec d init m i j) : OutSpec d init m' i j := by
  obtain ⟨c, hc, hrest⟩ := hs
  exact ⟨c, (Mem.read_congr h).trans hc, hrest⟩

/-- Storing a tile outside the reads of `a` and `b` keeps the memory invariant. -/
theorem memInv_store {d : Inst} (hd : d.ok) {init m m' : Mem} {T : Nat} (hT : T < nT d)
    (hm : MemInv d init m) (hframe : ∀ x, x ∉ tileSet d T → m' x = m x)
    (hpres : ∀ x, m x ≠ none → m' x ≠ none) : MemInv d init m' := by
  unfold MemInv at hm ⊢
  constructor
  · intro x hx
    have hxle : x ∉ tileSet d T := by
      intro hxc
      have hsub : tileSet d T ⊆ d.c.elems (cIdx d) :=
        Buf.elems_mono d.c (tileOf_sub_cIdx hd hT)
      have hc := hsub hxc
      simp only [abSet, Set.mem_union] at hx
      rcases hx with hxa | hxb
      · exact Set.disjoint_left.1 hd.c_a hc hxa
      · exact Set.disjoint_left.1 hd.c_b hc hxb
    rw [hframe x hxle]
    exact hm.1 x (by simp only [abSet, Set.mem_union] at hx ⊢; exact hx)
  · intro x hx
    exact hpres x (hm.2 x hx)

/-- `Done` grows by one trip: the new tile's elements satisfy `OutSpec`, and the earlier tiles are
untouched by the store. -/
theorem done_succ {d : Inst} {init : Mem} {i T t : Nat} {m m' : Mem}
    (hT : T = i + t * d.NUM_SMS)
    (hdone : Done d init i t m)
    (hdisj : ∀ u, u < t → Disjoint (tileSet d (i + u * d.NUM_SMS)) (tileSet d T))
    (hframe : ∀ x, x ∉ tileSet d T → m' x = m x)
    (hnew : ∀ a b : Nat, a < 128 → b < 256 →
      OutSpec d init m' (pm d T * 128 + a) (pn d T * 256 + b)) :
    Done d init i (t + 1) m' := by
  unfold Done at hdone ⊢
  intro u hu a b ha hb
  rcases Nat.lt_succ_iff_lt_or_eq.mp hu with hlt | heq
  · refine outSpec_frame (i := pm d (i + u * d.NUM_SMS) * 128 + a)
      (j := pn d (i + u * d.NUM_SMS) * 256 + b) ?_ (hdone u hlt a b ha hb)
    intro x hx
    have hmem : (pm d (i + u * d.NUM_SMS) * 128 + a) * d.stride_cm +
        (pn d (i + u * d.NUM_SMS) * 256 + b) * d.stride_cn ∈ tileOf d (i + u * d.NUM_SMS) :=
      ⟨a, b, ha, hb, rfl⟩
    have hsub := Buf.span_sub_elems d.c hmem
    exact hframe x (fun hxc => Set.disjoint_left.1 (hdisj u hlt) (hsub hx) hxc)
  · rw [heq, ← hT]
    exact hnew a b ha hb

/-- The `OutSpec` of the tile `T` of the accumulator: `CellInv` at the end of the reduction plus
the rounded `float16` value stored in each element. -/
theorem outspec_body {d : Inst} (hd : d.ok) {init m : Mem} {T : Nat}
    {acc : Tensor [128, 256] F32} {vals : Tensor [128, 256] F16}
    (hcell : CellInv d init T (steps d.K) acc)
    (hvals : ∀ (iX : Fin 128) (nX : Fin 256),
      FloatV.NaNEq (FloatV.cvt (acc.get2 iX nX)) (vals.get2 iX nX))
    (hread : ∀ (iX : Fin 128) (nX : Fin 256),
      m.read .f16 (d.c.ptr (((pm d T * 128 + iX.val) * d.stride_cm +
        (pn d T * 256 + nX.val) * d.stride_cn : Nat) : Int)).addr = some (vals.get2 iX nX))
    (a b : Nat) (ha : a < 128) (hb : b < 256) :
    OutSpec d init m (pm d T * 128 + a) (pn d T * 256 + b) := by
  refine ⟨vals.get2 ⟨a, ha⟩ ⟨b, hb⟩, hread ⟨a, ha⟩ ⟨b, hb⟩, ?_⟩
  intro va vb hva hvb
  obtain ⟨hfin, hbound⟩ := hcell ⟨a, ha⟩ ⟨b, hb⟩ va vb hva hvb
  have hKdvd : 64 ∣ d.K := by have h := hd.K_dim.2.1; omega
  have hps : partSum (prodV va vb d.K) (steps d.K) = dotSum va vb d.K :=
    partSum_steps hKdvd va vb
  have hpmg : partMag va vb d.K (steps d.K) = accMag va vb d.K := partMag_steps va vb d.K
  have herr : errB rho' alpha' (partMag va vb d.K (steps d.K)) (steps d.K) ≤
      tol va vb d.K := by
    have h := errB_le_tol hd.K_le (fun k hk => (hva k hk).2) (fun k hk => (hvb k hk).2)
    rw [hpmg]
    exact h
  have hle : |(acc.get2 ⟨a, ha⟩ ⟨b, hb⟩).toRat - dotSum va vb d.K| ≤ tol va vb d.K := by
    rw [← hps]
    exact le_trans hbound herr
  refine ⟨acc.get2 ⟨a, ha⟩ ⟨b, hb⟩, hfin, hle, hvals ⟨a, ha⟩ ⟨b, hb⟩, ?_⟩
  intro hroom
  obtain ⟨hcf, hcb⟩ := cvt_out (x := acc.get2 ⟨a, ha⟩ ⟨b, hb⟩) hle hroom
  have hEq : vals.get2 ⟨a, ha⟩ ⟨b, hb⟩ = FloatV.cvt (acc.get2 ⟨a, ha⟩ ⟨b, hb⟩) :=
    FloatV.NaNEq.eq (hvals ⟨a, ha⟩ ⟨b, hb⟩) (Format.not_isNaN_of_isFinite hcf)
  exact ⟨by rw [hEq]; exact hcf, by rw [hEq]; exact hcb⟩

end Claims.PersistentMatmul
