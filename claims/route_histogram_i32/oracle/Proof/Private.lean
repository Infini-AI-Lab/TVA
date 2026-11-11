import Proof.AddCounts

/-!
# `route_histogram_i32`: one program of the launch is `SafeA`

`_route_histogram_kernel`'s body reads `token_slots` (and `group_ids`) privately, routes each
lane to its bucket (`route_bucket_ids`), and adds the counts (`add_counts`, `safe_add_counts`).
The reads are ordinary loads of the input buffers, whose tiles lie in `Rpriv d`.
-/

namespace Claims.RouteHistogram

open TVA
open TVA.Block

/-- Replacing the negative lanes of `t` by `b` twice is replacing them once. -/
private theorem where_self {B : Nat} (t : Tensor [B] I32) (b : Tensor [B] I32) :
    tl.where ((tl.where (t.ltS (IntV.ofInt 0 : I32)) b t).ltS (IntV.ofInt 0 : I32)) b
      (tl.where (t.ltS (IntV.ofInt 0 : I32)) b t)
    = tl.where (t.ltS (IntV.ofInt 0 : I32)) b t := by
  ext i
  simp only [tl.where, Tensor.where_, Tensor.get_ltS]
  by_cases h : IntV.lt (t.get i) (IntV.ofInt 0 : I32) = true
  · simp [h]
  · have hf : IntV.lt (t.get i) (IntV.ofInt 0 : I32) = false := by
      cases hv : IntV.lt (t.get i) (IntV.ofInt 0 : I32) <;> simp_all
    simp [hf]

/-- `add_counts` already replaces negative lanes by `NUM_BUCKETS - 1`, so replacing them first changes nothing. -/
private theorem add_counts_where_self {B : Nat} (ptr : Ptr .i32) (t : Tensor [B] I32)
    (mask : Tensor [B] Bool) (nb bins : Nat) :
    add_counts ptr (tl.where (t.ltS (IntV.ofInt 0 : I32))
      (tl.full [B] (IntV.ofNat (nb - 1) : I32)) t) mask nb bins
      = add_counts ptr t mask nb bins := by
  simp only [add_counts]
  rw [where_self]

/-- **One program** of the launch is `SafeA`: it reads only `Rpriv`, writes nothing privately, updates the cells of `counts` with its atomics, and ends with `Qset`. -/
theorem safe_private {d : Inst} (hd : d.ok) {E : Env} {m : Mem} {p : Nat} (hp : p < grid d)
    (hpre : Pre d m) :
    SafeA E (Rpriv d) ∅ (cells d)
      (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Qset d p log m) [] m
      (do
        let pair_ids := offs p d.block
        let pair_mask := mask p d.block d.n
        let bucket_ids ← route_bucket_ids (d.groupIds.ptr 0) (d.tokenSlots.ptr 0) pair_ids
          pair_mask d.gps d.maxLoras d.width d.hasGroups
        add_counts (d.counts.ptr 0) bucket_ids pair_mask d.numBuckets d.bins) := by
  obtain ⟨htokV, hcntV, htokH, hcntH, hgrp⟩ := hpre
  have hfit : Fits p d.block := fits hd hp
  have hn : d.n < 2 ^ 31 := by have := n_max_lt hd; omega
  have hw : 0 < d.width := hd.width_pos
  have hwlt : d.width < 2 ^ 31 := hd.width_lt
  have hmain : SafeA E (Rpriv d) (∅ ∪ ∅) (cells d)
      (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Qset d p log m) [] m
      ((let pair_ids := offs p d.block
        let pair_mask := mask p d.block d.n
        route_bucket_ids (d.groupIds.ptr 0) (d.tokenSlots.ptr 0) pair_ids
          pair_mask d.gps d.maxLoras d.width d.hasGroups).bind
        (fun bucket_ids =>
          add_counts (d.counts.ptr 0) bucket_ids (mask p d.block d.n) d.numBuckets d.bins)) := by
    refine SafeA.of_safe (W₁ := ∅) (W₂ := ∅)
      (f := fun bucket_ids =>
        add_counts (d.counts.ptr 0) bucket_ids (mask p d.block d.n) d.numBuckets d.bins)
      ?_ (Set.disjoint_empty ∅)
    simp only [route_bucket_ids]
    refine safe_idiv ?_ ?_
    · rintro ⟨l, u⟩
      have hu : u = () := by cases u; rfl
      rw [hu]
      have hx0 : 0 ≤ ((offs p d.block).get (l, ())).toInt := by
        rw [lane_get hfit]
        exact Int.natCast_nonneg _
      have hy0 : 0 < ((tl.full [d.block - 0] (IntV.ofNat d.width : I32)).get (l, ())).toInt := by
        show 0 < (IntV.ofNat d.width : I32).toInt
        rw [IntV.I32.toInt_ofNat hwlt]
        exact_mod_cast hw
      obtain ⟨q, hq, -⟩ := IntV.div?_of_nonneg hx0 hy0
      exact ⟨q, hq⟩
    · intro r hdiv
      have hrval : ∀ i : Idx [d.block - 0],
          (r.get i).toInt = ((p * d.block + i.1) / d.width : Nat) := by
        rintro ⟨l, u⟩
        have hu : u = () := by cases u; rfl
        rw [hu]
        have hx0 : 0 ≤ ((offs p d.block).get (l, ())).toInt := by
          rw [lane_get hfit]
          exact Int.natCast_nonneg _
        have hy0 : 0 < ((tl.full [d.block - 0] (IntV.ofNat d.width : I32)).get (l, ())).toInt := by
          show 0 < (IntV.ofNat d.width : I32).toInt
          rw [IntV.I32.toInt_ofNat hwlt]
          exact_mod_cast hw
        obtain ⟨q, hq, hqv⟩ := IntV.div?_of_nonneg hx0 hy0
        have h := hdiv (l, ())
        rw [hq] at h
        have hrq : r.get (l, ()) = q := (Option.some.inj h).symm
        rw [hrq, hqv]
        have hnum0 : ((offs p d.block).get (l, ())).toInt = (↑p * ↑(d.block) + ↑↑l : Int) :=
          lane_get hfit l
        have hden0 : ((tl.full [d.block - 0] (IntV.ofNat d.width : I32)).get (l, ())).toInt
            = (d.width : Int) := by
          show (IntV.ofNat d.width : I32).toInt = (d.width : Int)
          rw [IntV.I32.toInt_ofNat hwlt]
        rw [hnum0, hden0]
        have hnum : (↑p * ↑(d.block) + ↑((l : Nat)) : Int) = ↑(p * d.block + (l : Nat)) := by
          push_cast
          ring
        rw [hnum]
        exact (Int.natCast_div (p * d.block + (l : Nat)) d.width).symm
      have hA : d.tokenSlots.Addrs
          (tl.addrs (d.tokenSlots.ptr 0 + r) (some (mask p d.block d.n)))
          (fun i : Idx [d.block - 0] => (p * d.block + i.1) / d.width)
          (fun i : Idx [d.block - 0] => decide (p * d.block + i.1 < d.n)) := by
        refine Buf.addrs_of_get d.tokenSlots ?_ ?_
        · rintro ⟨l, u⟩
          cases u
          exact mask_get hfit hn l
        · rintro ⟨l, u⟩ hi
          have hu : u = () := by cases u; rfl
          rw [hu] at hi ⊢
          show (d.tokenSlots.ptr 0).offset ((r.get (l, ())).toInt)
              = d.tokenSlots.ptr ↑((p * d.block + (l : Nat)) / d.width)
          rw [hrval (l, ())]
          rw [Buf.ptr_offset]
          rw [zero_add]
      have hin : ∀ i : Idx [d.block - 0],
          decide (p * d.block + i.1 < d.n) = true →
            (p * d.block + i.1) / d.width < d.tokenSlots.len := by
        rintro ⟨l, u⟩ hi
        have hu : u = () := by cases u; rfl
        rw [hu] at hi ⊢
        show (p * d.block + (l : Nat)) / d.width < d.tokenSlots.len
        have hj : p * d.block + (l : Nat) < d.n := of_decide_eq_true hi
        have h1 : (p * d.block + (l : Nat)) / d.width ≤ (d.n - 1) / d.width :=
          Nat.div_le_div_right (by omega)
        have h2 : (d.n - 1) / d.width < (d.n + d.width - 1) / d.width := by
          have he : d.n + d.width - 1 = (d.n - 1) + d.width := by omega
          rw [he, Nat.add_div_right _ hw]
          omega
        exact lt_of_lt_of_le (lt_of_le_of_lt h1 h2) hd.tokenSlots_len
      have hR : d.tokenSlots.elems
          (Buf.img (fun i : Idx [d.block - 0] => (p * d.block + i.1) / d.width)
            (fun i : Idx [d.block - 0] => decide (p * d.block + i.1 < d.n))) ⊆ Rpriv d := by
        have hsub : d.tokenSlots.elems
            (Buf.img (fun i : Idx [d.block - 0] => (p * d.block + i.1) / d.width)
              (fun i : Idx [d.block - 0] => decide (p * d.block + i.1 < d.n)))
            ⊆ d.tokenSlots.bytes := by
          refine d.tokenSlots.elems_sub_bytes ?_
          rintro j ⟨i, hi, rfl⟩
          exact hin i hi
        intro x hx
        simp only [Rpriv]
        exact Or.inl (hsub hx)
      refine safe_load_buf (b := d.tokenSlots) (ptrs := d.tokenSlots.ptr 0 + r)
        (mask := some (mask p d.block d.n))
        (other := some (tl.full [d.block - 0] (IntV.ofInt (-1))))
        (g := fun i : Idx [d.block - 0] => (p * d.block + i.1) / d.width)
        (en := fun i : Idx [d.block - 0] => decide (p * d.block + i.1 < d.n))
        hA hin htokV hR ?_
      intro v hread hother
      have hsl : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
          (v.get (l, ())).toInt = (d.tsVal ((p * d.block + (l : Nat)) / d.width)).toInt := by
        intro l hl
        have he : decide (p * d.block + (l : Nat) < d.n) = true := decide_eq_true hl
        have hr : m.read DType.i32
            (d.tokenSlots.ptr ↑((fun i : Idx [d.block - 0] => (p * d.block + i.1) / d.width)
              (l, ()))).addr = some (v.get (l, ())) := hread (l, ()) he
        have hgl : ((fun i : Idx [d.block - 0] => (p * d.block + i.1) / d.width) (l, ()))
            < d.tokenSlots.len := hin (l, ()) he
        have hts : m.read DType.i32
            (d.tokenSlots.ptr ↑(((fun i : Idx [d.block - 0] => (p * d.block + i.1) / d.width)
              (l, ())))).addr
            = some (d.tsVal (((fun i : Idx [d.block - 0] => (p * d.block + i.1) / d.width)
              (l, ())))) := htokH _ hgl
        have hv : v.get (l, ()) = d.tsVal (((fun i : Idx [d.block - 0] =>
            (p * d.block + i.1) / d.width) (l, ()))) := Option.some.inj (hr.symm.trans hts)
        rw [hv]
      by_cases hg : d.hasGroups = true
      · rw [ite_eq_left hg]
        have hgiV : d.groupIds.Valid m := (hgrp hg).1
        have hgiH : d.groupIds.Holds m d.giVal := (hgrp hg).2
        have hA2 : d.groupIds.Addrs
            (tl.addrs (d.groupIds.ptr 0 + offs p d.block) (some (mask p d.block d.n)))
            (fun i : Idx [d.block - 0] => p * d.block + i.1)
            (fun i : Idx [d.block - 0] => decide (p * d.block + i.1 < d.n)) := by
          refine Buf.addrs_of_get d.groupIds ?_ ?_
          · rintro ⟨l, u⟩
            cases u
            exact mask_get hfit hn l
          · rintro ⟨l, u⟩ hi
            have hu : u = () := by cases u; rfl
            rw [hu] at hi ⊢
            show (d.groupIds.ptr 0).offset ((offs p d.block).get (l, ())).toInt
                = d.groupIds.ptr ↑(p * d.block + (l : Nat))
            rw [lane_get hfit]
            rw [Buf.ptr_offset, zero_add]
        have hin2 : ∀ i : Idx [d.block - 0],
            decide (p * d.block + i.1 < d.n) = true →
              p * d.block + i.1 < d.groupIds.len := by
          rintro ⟨l, u⟩ hi
          have hu : u = () := by cases u; rfl
          rw [hu] at hi ⊢
          show p * d.block + (l : Nat) < d.groupIds.len
          have hj : p * d.block + (l : Nat) < d.n := of_decide_eq_true hi
          have hlen := hd.groupIds_len hg
          omega
        have hR2 : d.groupIds.elems
            (Buf.img (fun i : Idx [d.block - 0] => p * d.block + i.1)
              (fun i : Idx [d.block - 0] => decide (p * d.block + i.1 < d.n))) ⊆ Rpriv d := by
          have hsub : d.groupIds.elems
              (Buf.img (fun i : Idx [d.block - 0] => p * d.block + i.1)
                (fun i : Idx [d.block - 0] => decide (p * d.block + i.1 < d.n)))
              ⊆ d.groupIds.bytes := by
            refine d.groupIds.elems_sub_bytes ?_
            rintro j ⟨i, hi, rfl⟩
            exact hin2 i hi
          intro x hx
          have hx' : x ∈ d.groupIds.bytes := hsub hx
          simp only [Rpriv, ite_eq_left hg]
          exact Or.inr hx'
        refine safe_load_buf (b := d.groupIds) (ptrs := d.groupIds.ptr 0 + offs p d.block)
          (mask := some (mask p d.block d.n))
          (other := some (tl.full [d.block - 0] (IntV.ofInt (-1))))
          (g := fun i : Idx [d.block - 0] => p * d.block + i.1)
          (en := fun i : Idx [d.block - 0] => decide (p * d.block + i.1 < d.n))
          hA2 hin2 hgiV hR2 ?_
        intro v2 hread2 hother2
        have hgr : ∀ l : Fin d.block, p * d.block + (l : Nat) < d.n →
            (v2.get (l, ())).toInt = (d.giVal (p * d.block + (l : Nat))).toInt := by
          intro l hl
          have he : decide (p * d.block + (l : Nat) < d.n) = true := decide_eq_true hl
          have hr : m.read DType.i32
              (d.groupIds.ptr ↑((fun i : Idx [d.block - 0] => p * d.block + i.1) (l, ()))).addr
              = some (v2.get (l, ())) := hread2 (l, ()) he
          have hgl : ((fun i : Idx [d.block - 0] => p * d.block + i.1) (l, ()))
              < d.groupIds.len := hin2 (l, ()) he
          have hts : m.read DType.i32
              (d.groupIds.ptr ↑(((fun i : Idx [d.block - 0] => p * d.block + i.1) (l, ())))).addr
              = some (d.giVal (((fun i : Idx [d.block - 0] => p * d.block + i.1) (l, ())))) :=
            hgiH _ hgl
          have hv : v2.get (l, ()) = d.giVal (((fun i : Idx [d.block - 0] =>
              p * d.block + i.1) (l, ()))) := Option.some.inj (hr.symm.trans hts)
          rw [hv]
        by_cases hgps : d.gps = 1
        · rw [ite_eq_left hgps]
          have hraw : RawOk d p
              (tl.where
                ((((tl.full [d.block - 0] (IntV.ofInt 0 : I32)).le v).and
                    (v.ltS (IntV.ofNat d.maxLoras))).and
                  ((tl.full [d.block - 0] (IntV.ofInt 0 : I32)).le v2))
                (v.castInt : Tensor [d.block - 0] I32)
                (tl.full [d.block - 0] (IntV.ofInt (-1) : I32))) :=
            raw_g1 hd hg hgps hsl hgr
          rw [Prog.pure_eq]
          refine Safe.ret ?_
          have hsa := safe_add_counts (E := E) hd hp hcntV (laneOf_raw hd hraw)
          rw [add_counts_where_self] at hsa
          exact hsa
        · rw [ite_eq_right hgps]
          have hraw : RawOk d p
              (tl.where
                (((((tl.full [d.block - 0] (IntV.ofInt 0 : I32)).le v).and
                    (v.ltS (IntV.ofNat d.maxLoras))).and
                  ((tl.full [d.block - 0] (IntV.ofInt 0 : I32)).le v2)).and
                  (v2.ltS (IntV.ofNat d.gps)))
                ((v.castInt : Tensor [d.block - 0] I32) * (IntV.ofNat d.gps : I32) +
                  (v2.castInt : Tensor [d.block - 0] I32))
                (tl.full [d.block - 0] (IntV.ofInt (-1) : I32))) :=
            raw_gN hd hg hgps hsl hgr
          rw [Prog.pure_eq]
          refine Safe.ret ?_
          have hsa := safe_add_counts (E := E) hd hp hcntV (laneOf_raw hd hraw)
          rw [add_counts_where_self] at hsa
          exact hsa
      · rw [ite_eq_right hg]
        have hnofalse : d.hasGroups = false := Bool.eq_false_iff.mpr hg
        have hraw : RawOk d p
            (tl.where
              (((tl.full [d.block - 0] (IntV.ofInt 0 : I32)).le v).and
                (v.ltS (IntV.ofNat d.maxLoras)))
              (v.castInt : Tensor [d.block - 0] I32)
              (tl.full [d.block - 0] (IntV.ofInt (-1) : I32))) :=
          raw_nogroups hd hnofalse hsl
        rw [Prog.pure_eq]
        refine Safe.ret ?_
        have hsa := safe_add_counts (E := E) hd hp hcntV (laneOf_raw hd hraw)
        rw [add_counts_where_self] at hsa
        exact hsa
  rw [show (∅ : Set Int) ∪ ∅ = ∅ from Set.empty_union ∅] at hmain
  exact hmain

end Claims.RouteHistogram
