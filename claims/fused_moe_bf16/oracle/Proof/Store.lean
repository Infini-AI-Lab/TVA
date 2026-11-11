import TVA.Proof
import Claim
import Proof.Pid
import Proof.Foot
import Proof.Defs

/-!
# Entries, masks and the store of a tile

The widening conversion, the bytes of the input buffers, the lane values of the block's entries
and mask (`TokV`, `TokLanes`), and `safe_storeC`: the store of a `BM x BN` tile to the rows of
`C` that the block routes.
-/

namespace Claims.FusedMoe

open TVA

variable {P : Platform} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- `tl.convert .i32 .i64` is the exact widening. -/
theorem safe_conv32 {s : Shape} {x : Tensor s I32} {k : Tensor s I64 → Prog α} {E : Env}
    (h : Safe E R W Q m (k x.castInt)) : Safe E R W Q m (tl.convert .i32 .i64 x >>= k) :=
  (safe_convert_iff (fun i r => ConvRel.iff_cast_i32_i64.trans (by rw [Tensor.get_castInt]))).mpr h

/-- The bytes of the six input buffers are in `Rfp`. -/
theorem Rfp_ntp (d : Inst) : d.ntpb.bytes ⊆ Rfp d := fun _ h => Or.inr h
/-- The bytes of `expert_ids` are read bytes. -/
theorem Rfp_eid (d : Inst) : d.eid.bytes ⊆ Rfp d := fun _ h => Or.inl (Or.inr h)
/-- The bytes of `sorted_token_ids` are read bytes. -/
theorem Rfp_s (d : Inst) : d.s.bytes ⊆ Rfp d := fun _ h => Or.inl (Or.inl (Or.inr h))
/-- The bytes of `topk_weights` are read bytes. -/
theorem Rfp_w (d : Inst) : d.w.bytes ⊆ Rfp d := fun _ h => Or.inl (Or.inl (Or.inl (Or.inr h)))
/-- The bytes of `B` are read bytes. -/
theorem Rfp_b (d : Inst) : d.b.bytes ⊆ Rfp d :=
  fun _ h => Or.inl (Or.inl (Or.inl (Or.inl (Or.inr h))))
/-- The bytes of `A` are read bytes. -/
theorem Rfp_a (d : Inst) : d.a.bytes ⊆ Rfp d :=
  fun _ h => Or.inl (Or.inl (Or.inl (Or.inl (Or.inl h))))

/-- The lane values of the block's entries. -/
def TokV (d : Inst) (init : Mem) (pm : Nat) (v : Tensor [d.BM] I32) : Prop :=
  ∀ l : Fin d.BM, (v.get1 l).toInt = entryOf d init (pm * d.BM + l)

/-- The lanes of the widened entries and of the mask. -/
def TokLanes (d : Inst) (init : Mem) (pmN : Nat) (ot : Tensor [d.BM] I64)
    (tm : Tensor [d.BM] Bool) : Prop :=
  (∀ l : Fin d.BM, (ot.get1 l).toInt = entryOf d init (pmN * d.BM + l)) ∧
  (∀ l : Fin d.BM, tm.get1 l = decide (entryOf d init (pmN * d.BM + l) < (nv d : Int)))

/-- **The store of a tile to the rows of the block** (`storeC`): the elements of the tile hold
the tile's lanes, and nothing else changes. -/
theorem safe_storeC {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    {R W : Set Int} {Q : Unit → Mem → Prop} {pmN pnN : Nat} {pn : I32}
    (hpn : pn.toInt = (pnN : Int)) (hpnb : pnN * d.BN + d.BN ≤ d.N + d.BN - 1)
    (hact : pmN * d.BM + d.BM ≤ ntpOf d init)
    {ot : Tensor [d.BM] I64} {tm : Tensor [d.BM] Bool} (hl : TokLanes d init pmN ot tm)
    (c : Tensor [d.BM, d.BN] BF16) {k : Unit → Prog Unit}
    (hW : d.c.elems (Wset d init pmN pnN) ⊆ W)
    (hk : ∀ m' : Mem, (∀ (l : Fin d.BM) (j : Fin d.BN),
        entryOf d init (pmN * d.BM + l) < (nv d : Int) → pnN * d.BN + j < d.N →
        m'.read .bf16 (d.c.ptr (((entryOf d init (pmN * d.BM + l)).toNat * d.N + (pnN * d.BN + j) : Nat) : Int)).addr
          = some (c.get2 l j)) →
      (∀ x, x ∉ d.c.elems (Wset d init pmN pnN) → m' x = init x) →
      Safe E R (W \ d.c.elems (Wset d init pmN pnN)) Q m' (k ())) :
    Safe E R W Q init (storeC d pn ot tm c >>= k) := by
  obtain ⟨hv, hm⟩ := hl
  have hBM := (cfg_facts hd).1
  have hBN := (cfg_facts hd).2.1
  have hN := hd.N_pos
  have hcm := hd.cm
  have hnv := hd.nv_le
  have hNle := hd.N_le
  unfold storeC
  dsimp only
  simp only [bind_assoc]
  refine safe_conv32 ?_
  refine safe_conv32 ?_
  have hpnv : (pn * (IntV.ofNat d.BN : I32)).toInt = ((pnN * d.BN : Nat) : Int) := by
    rw [IntV.toInt_mul rfl rfl (by rw [hpn, i32_ofNat (by omega)]; exact sfits_mul_nat (by omega)),
      hpn, i32_ofNat (by omega)]
    push_cast; ring
  have hcol : ∀ j : Fin d.BN, ((pn * (IntV.ofNat d.BN : I32) + tl.arange 0 d.BN).get1 j).toInt =
      ((pnN * d.BN + (j : Nat) : Nat) : Int) :=
    fun j => IntV.I32.toInt_sadd_arange (n := d.BN) hpnv (by omega) j
  have hsc : ((Tensor.scalar (IntV.ofNat (stride_cm d) : I32)).castInt : Tensor [] I64).item.toInt =
      (d.N : Int) := by
    show (IntV.cast (IntV.ofNat (stride_cm d) : I32) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), i32_ofNat (by have := hd.cm; omega)]; rfl
  have hfitc : (nv d - 1) * d.N < 2 ^ (64 - 1) := by
    have h1 : (nv d - 1) * d.N ≤ 2 ^ 31 * 2 ^ 17 :=
      Nat.mul_le_mul (by omega) hd.N_le17
    have h2 : (2 : Nat) ^ 31 * 2 ^ 17 < 2 ^ (64 - 1) := by norm_num
    omega
  have hent0 : ∀ l : Fin d.BM, 0 ≤ entryOf d init (pmN * d.BM + l) := fun l =>
    hpre.s1 (pmN * d.BM + l) (by have := l.isLt; omega)
  have hrow : ∀ l : Fin d.BM, (ot.get1 l).toInt = (((entryOf d init (pmN * d.BM + l)).toNat : Nat) : Int) :=
    fun l => by rw [hv l]; exact (Int.toNat_of_nonneg (hent0 l)).symm
  refine safe_store_buf d.c
    (g := fun idx : Idx [d.BM, d.BN] =>
      (entryOf d init (pmN * d.BM + idx.1)).toNat * d.N + (pnN * d.BN + idx.2.1))
    (en := fun idx : Idx [d.BM, d.BN] =>
      decide (entryOf d init (pmN * d.BM + idx.1) < (nv d : Int)) && decide (pnN * d.BN + idx.2.1 < d.N))
    ?_ ?_ ?_ hpre.vc ?_ fun m' hread hframe _ => ?_
  · -- the addresses
    refine Buf.addrs_of_get d.c ?_ ?_
    · rintro ⟨l, j, ⟨⟩⟩
      show (tm.get1 l && IntV.lt ((pn * (IntV.ofNat d.BN : I32) + tl.arange 0 d.BN).get1 j) (IntV.ofNat d.N : I32)) = _
      rw [hm l, IntV.lt_eq, hcol j, i32_ofNat (by omega)]
      congr 1
      exact decide_eq_decide.mpr (by norm_cast)
    · rintro ⟨l, j, ⟨⟩⟩ hen
      have hen' := hen
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hen'
      have hy : ∀ (i : Fin d.BM) (j : Fin d.BN), pnN * d.BN + (j : Nat) < d.N →
          ((((pn * (IntV.ofNat d.BN : I32) + tl.arange 0 d.BN).row d.BM * (IntV.ofNat 1 : I32)).castInt :
            Tensor [d.BM, d.BN] I64).get2 i j).toInt = (((pnN * d.BN + (j : Nat)) * 1 : Nat) : Int) := by
        intro i j hj
        show (IntV.cast (((pn * (IntV.ofNat d.BN : I32) + tl.arange 0 d.BN).row d.BM * (IntV.ofNat 1 : I32)).get2 i j) : I64).toInt = _
        rw [IntV.toInt_cast _ (by norm_num)]
        exact Tile2.rowMul_toInt_on (κ := fun j : Fin d.BN => pnN * d.BN + (j : Nat)) (N := d.N) hcol
          (i32_ofNat (by norm_num)) (by omega) i j hj
      have hx : ∀ (i : Fin d.BM) (j : Fin d.BN),
          (entryOf d init (pmN * d.BM + i)).toNat < nv d →
          ((((Tensor.scalar (IntV.ofNat (stride_cm d) : I32)).castInt : Tensor [] I64).item *
            ot.col d.BN).get2 i j).toInt = (((entryOf d init (pmN * d.BM + i)).toNat * d.N : Nat) : Int) := by
        intro i j hi
        exact Tile2.mulCol_toInt_on (ρ := fun l : Fin d.BM => (entryOf d init (pmN * d.BM + l)).toNat)
          (M := nv d) hrow hsc hfitc i j hi
      have := Tile2.ptr_lane_seq_of_on d.c (e := 0) (s0 := d.N) (s1 := 1) (M := nv d) (N := d.N)
        (ρ := fun l : Fin d.BM => (entryOf d init (pmN * d.BM + l)).toNat)
        (κ := fun j : Fin d.BN => pnN * d.BN + (j : Nat)) (base := d.c.ptr 0) (by simp [Buf.ptr]) hx hy
        l j (by have := hent0 l; omega) hen'.2
      refine this.trans ?_
      congr 1
      push_cast; ring
  · -- the elements are inside `C`
    rintro ⟨l, j, ⟨⟩⟩ hen
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    have := hd.c_len (entryOf d init (pmN * d.BM + l)).toNat (pnN * d.BN + j) (by have := hent0 l; omega) hen.2
    exact this
  · -- distinct lanes, distinct elements
    rintro ⟨l, j, ⟨⟩⟩ ⟨l', j', ⟨⟩⟩ hen hen' heq
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen hen'
    obtain ⟨h1, h2⟩ := mul_add_inj (f := (entryOf d init (pmN * d.BM + l)).toNat)
      (f' := (entryOf d init (pmN * d.BM + l')).toNat) hen.2 hen'.2 heq
    have e1 : entryOf d init (pmN * d.BM + l') = entryOf d init (pmN * d.BM + l) := by
      have := hent0 l; have := hent0 l'; omega
    have hll := hpre.s2 (pmN * d.BM + l) (pmN * d.BM + l')
      (by have := l.isLt; omega) (by have := l'.isLt; omega) hen.1 e1
    have hl : l = l' := Fin.ext (by omega)
    have hj : j = j' := Fin.ext (by omega)
    subst hl; subst hj; rfl
  · -- the footprint
    refine Set.Subset.trans (d.c.elems_mono ?_) hW
    rintro x ⟨⟨l, j, ⟨⟩⟩, hen, rfl⟩
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
    exact ⟨l, j, l.isLt, j.isLt, by have := l.isLt; omega, hen.1, hen.2, rfl⟩
  · -- the rest of the program
    have himg : Buf.img (fun idx : Idx [d.BM, d.BN] =>
        (entryOf d init (pmN * d.BM + idx.1)).toNat * d.N + (pnN * d.BN + idx.2.1))
        (fun idx : Idx [d.BM, d.BN] => decide (entryOf d init (pmN * d.BM + idx.1) < (nv d : Int)) &&
          decide (pnN * d.BN + idx.2.1 < d.N)) ⊆ Wset d init pmN pnN := by
      rintro x ⟨⟨l, j, ⟨⟩⟩, hen, rfl⟩
      simp only [Bool.and_eq_true, decide_eq_true_eq] at hen
      exact ⟨l, j, l.isLt, j.isLt, by have := l.isLt; omega, hen.1, hen.2, rfl⟩
    refine (hk m' (fun l j hv hc => ?_) (fun x hx => hframe x (fun hx' => hx (d.c.elems_mono himg hx')))).mono
      (fun _ _ h => h) ?_
    · exact hread (l, j, ()) (by simp [hv, hc])
    · exact Set.sdiff_subset_sdiff_right (d.c.elems_mono himg)


end Claims.FusedMoe
