import Proof.Trip

/-!
# One trip of the loop, in each branch

The masked load of the ids, the optional gather through the map, and the increment of the lanes
whose loaded entry is the program's expert. Both branches leave the trip's pointer block advanced
by `BLOCK_SIZE` and read only the ids' (and the map's) bytes.
-/

namespace Claims.CountExpertTokens

open TVA

/-! ## The body of one trip, in each branch -/

/-- Without a map, one trip loads the ids and compares them with the expert. -/
theorem loadBody_false (dd : DType) (B : Nat) (em : Ptr .i32) (numel curr x : I32)
    (acc : Tensor [B] I32) (ptrs : Tensor [B] (Ptr dd)) :
    loadBody dd false B em numel curr x acc ptrs =
      ((tl.load ptrs (some ((tl.arange 0 B).ltS (numel - x * (IntV.ofNat B : I32))))
        (some (Tensor.full (m1 dd)))) >>= fun v => tailOf acc ptrs (eqCurr dd v curr)) := rfl

/-- With a map, one trip loads the ids, gathers through the map and compares with the expert. -/
theorem loadBody_true (dd : DType) (B : Nat) (em : Ptr .i32) (numel curr x : I32)
    (acc : Tensor [B] I32) (ptrs : Tensor [B] (Ptr dd)) :
    loadBody dd true B em numel curr x acc ptrs =
      ((tl.load ptrs (some ((tl.arange 0 B).ltS (numel - x * (IntV.ofNat B : I32))))
        (some (Tensor.full (m1 dd)))) >>= fun v =>
        (tl.load (mapPtrs dd em v) (some (geZero dd v)) (some (Tensor.full (m1 .i32)))) >>= fun v' =>
          tailOf acc ptrs (v'.eq (bc curr))) := rfl

/-! ## `Rin` in each branch -/

/-- With a map the read footprint is the ids and the map. -/
theorem Rin_map {d : Inst} (h : d.hasMap = true) :
    Rin d = d.ids.elems {j | j < d.numel} ∪ d.emap.elems {j | j < d.numGlobal} := by
  simp [Rin, h]

/-! ## One more position at a lane -/

/-- One trip adds the lane's indicator to its count. -/
theorem count_step {d : Inst} (hd : d.ok) {e t : Nat} {l : Fin d.B} {acc : Tensor [d.B] I32}
    {eq : Tensor [d.B] Bool} (ht : t < numTrips d)
    (hacc : (acc.get1 l).toInt = (cnt d e t l.val : Int))
    (heq : (eq.get1 l = true) ↔
      (t * d.B + l.val < d.numel ∧ routed d (t * d.B + l.val) = (e : Int))) :
    ((acc + tl.where eq (Tensor.full (1 : I32)) (Tensor.full (0 : I32))).get1 l).toInt
      = (cnt d e (t + 1) l.val : Int) := by
  have hb : cnt d e t l.val
      + (if t * d.B + l.val < d.numel ∧ routed d (t * d.B + l.val) = (e : Int) then 1 else 0)
      < 2 ^ 31 := by
    rw [← cnt_succ d e t l.val]
    have h1 : cnt d e (t + 1) l.val ≤ d.numel :=
      cnt_le_numel hd e (Nat.succ_le_of_lt ht) l.val
    have h2 := numel_lt hd
    omega
  have hy : (if eq.get1 l = true then (1 : I32) else (0 : I32)).toInt
      = ((if t * d.B + l.val < d.numel ∧ routed d (t * d.B + l.val) = (e : Int)
          then 1 else 0 : Nat) : Int) := by
    by_cases h : eq.get1 l = true
    · rw [ite_eq_left h, toInt_one_i32]
      have hc : t * d.B + l.val < d.numel ∧ routed d (t * d.B + l.val) = (e : Int) := heq.mp h
      rw [ite_eq_left hc]
      norm_num
    · rw [ite_eq_right h, toInt_zero_i32]
      have hc : ¬(t * d.B + l.val < d.numel ∧ routed d (t * d.B + l.val) = (e : Int)) :=
        fun hc => h (heq.mpr hc)
      rw [ite_eq_right hc]
      norm_num
  rw [Tensor.get1_add]
  simp only [tl.where, get1_where, get1_full]
  rw [IntV.I32.add_nat hacc hy hb, cnt_succ]

/-! ## Advancing the pointer block -/

/-- Advancing a lane's pointer by `B` moves it to the next trip's position. -/
theorem ptr_next {d : Inst} (hd : d.ok) {t : Nat} (l : Fin d.B) :
    (d.ids.ptr (tripG d t (l, ()) : Int)).offset ((IntV.ofNat d.B : I32).toInt)
      = d.ids.ptr (tripG d (t + 1) (l, ()) : Int) := by
  rw [B_val hd, Buf.ptr_offset]
  have hnat : tripG d t (l, ()) + d.B = tripG d (t + 1) (l, ()) := by
    simp only [tripG]
    rw [Nat.add_mul, one_mul]
    omega
  rw [← Nat.cast_add, hnat]

/-- Before the first trip lane `l` points at element `l` of the ids. -/
theorem initPtrs {d : Inst} (hd : d.ok) (l : Fin d.B) :
    (d.ids.ptr 0 + tl.arange 0 d.B).get1 l = d.ids.ptr (tripG d 0 (l, ()) : Int) := by
  rw [get1_ptrAdd]
  have hl : ((tl.arange 0 d.B).get1 l).toInt = (l.val : Int) := by
    rw [tl.arange, Tensor.get1_arange]
    norm_num
    refine IntV.I32.toInt_ofInt ?_
    have hB := B_lt hd
    constructor <;> omega
  rw [hl, Buf.ptr_offset]
  congr 1
  simp only [tripG]
  omega

/-- The initial accumulator counts nothing. -/
theorem acc_zero {d : Inst} (e : Nat) (l : Fin d.B) :
    ((tl.zeros [d.B] .i32).get1 l).toInt = (cnt d e 0 l.val : Int) := by
  rw [cnt_zero, tl.zeros, get1_full]
  exact IntV.I32.toInt_ofNat (by norm_num)

/-! ## A lane's equality test -/

/-- Without a map, lane `l` of the equality test holds exactly when its position is in range and routed to the expert. -/
theorem lane_eq_plain {d : Inst} (hd : d.ok) (hmap : d.hasMap = false) {init : Mem}
    (hpre : Pre d init) {e t : Nat} {v : Tensor [d.B] d.dtype.Val}
    (hv : ∀ i, tripEn d t i = true →
      init.read d.dtype (d.ids.ptr (tripG d t i : Int)).addr = some (v.get i))
    (ho : ∀ i, tripEn d t i = false → v.get i = m1 d.dtype)
    (curr : I32) (hc : curr.toInt = (e : Int)) (l : Fin d.B) :
    ((eqCurr d.dtype v curr).get1 l = true) ↔
      (t * d.B + l.val < d.numel ∧ routed d (t * d.B + l.val) = (e : Int)) := by
  rw [eqCurr_true hd.dtype_ok v curr (e : Int) hc l]
  rw [show v.get1 l = v.get (l, ()) from rfl]
  rw [ids_entry hd hpre v hv ho (l, ())]
  by_cases hen : tripEn d t (l, ()) = true
  · rw [ite_eq_left hen]
    have hlt : t * d.B + l.val < d.numel := by
      have h := (tripEn_iff (l, ())).mp hen
      rwa [tripG_pair] at h
    rw [show tripG d t (l, ()) = t * d.B + l.val from rfl, routed_noMap hmap]
    constructor
    · intro h; exact ⟨hlt, h⟩
    · intro h; exact h.2
  · have hf : tripEn d t (l, ()) = false := by
      cases h : tripEn d t (l, ()) <;> simp_all
    rw [ite_eq_right hen]
    have hnlt : ¬(t * d.B + l.val < d.numel) := by
      intro hc'
      exact (tripEn_false_iff (l, ())).mp hf (by rw [tripG_pair]; exact hc')
    constructor
    · intro h; exact absurd h (by omega)
    · intro h; exact absurd h.1 hnlt

/-- With a map, the same for the gathered ids. -/
theorem lane_eq_map {d : Inst} (hd : d.ok) (hmap : d.hasMap = true) {init : Mem}
    (hpre : Pre d init) {e t : Nat} {v : Tensor [d.B] d.dtype.Val}
    (hv : ∀ i, tripEn d t i = true →
      init.read d.dtype (d.ids.ptr (tripG d t i : Int)).addr = some (v.get i))
    (ho : ∀ i, tripEn d t i = false → v.get i = m1 d.dtype)
    (v' : Tensor [d.B] I32)
    (hv' : ∀ i, gatherEn d v i = true →
      init.read .i32 (d.emap.ptr (gatherG d v i : Int)).addr = some (v'.get i))
    (ho' : ∀ i, gatherEn d v i = false → v'.get i = m1 .i32)
    (curr : I32) (hc : curr.toInt = (e : Int)) (l : Fin d.B) :
    ((v'.eq (bc curr)).get1 l = true) ↔
      (t * d.B + l.val < d.numel ∧ routed d (t * d.B + l.val) = (e : Int)) := by
  rw [Tensor.get1_eq, show (bc curr).get1 l = curr from rfl, IntV.eq_eq, decide_eq_true_iff]
  rw [get1_get v' l]
  rw [emap_entry hd hpre v hv ho v' hv' ho' hmap (l, ()) , hc]
  by_cases hen : tripEn d t (l, ()) = true
  · rw [ite_eq_left hen]
    have hlt : t * d.B + l.val < d.numel := by
      have h := (tripEn_iff (l, ())).mp hen
      rwa [tripG_pair] at h
    rw [show tripG d t (l, ()) = t * d.B + l.val from rfl]
    constructor
    · intro h; exact ⟨hlt, h⟩
    · intro h; exact h.2
  · have hf : tripEn d t (l, ()) = false := by
      cases h : tripEn d t (l, ()) <;> simp_all
    rw [ite_eq_right hen]
    have hnlt : ¬(t * d.B + l.val < d.numel) := by
      intro hc'
      exact (tripEn_false_iff (l, ())).mp hf (by rw [tripG_pair]; exact hc')
    constructor
    · intro h; exact absurd h (by omega)
    · intro h; exact absurd h.1 hnlt

/-! ## What an enabled lane of the gather points at -/

/-- An enabled gather lane indexes inside the map. -/
theorem gather_bound {d : Inst} (hd : d.ok) (hmap : d.hasMap = true) {init : Mem}
    (hpre : Pre d init) {t : Nat} {v : Tensor [d.B] d.dtype.Val}
    (hv : ∀ i, tripEn d t i = true →
      init.read d.dtype (d.ids.ptr (tripG d t i : Int)).addr = some (v.get i))
    (ho : ∀ i, tripEn d t i = false → v.get i = m1 d.dtype)
    (i : Idx [d.B]) (hi : gatherEn d v i = true) :
    gatherG d v i < d.numGlobal := by
  obtain ⟨l, u⟩ := i
  cases u
  have hentry := ids_entry hd hpre v hv ho (l, ())
  have hge : 0 ≤ entryInt d.dtype (v.get (l, ())) := by
    have h := (geZero_true hd.dtype_ok v l).mp hi
    exact h
  have hen : tripEn d t (l, ()) = true := by
    by_cases h : tripEn d t (l, ()) = true
    · exact h
    · have hf : tripEn d t (l, ()) = false := by
        cases h' : tripEn d t (l, ()) <;> simp_all
      rw [hentry, ite_eq_right h] at hge
      omega
  have hlt : tripG d t (l, ()) < d.numel := (tripEn_iff (l, ())).mp hen
  have hbnd : entryInt d.dtype (d.idsVal (tripG d t (l, ()))) < d.numGlobal :=
    (hpre.2.2.2 hmap).2.2 _ hlt
  have heq2 : entryInt d.dtype (v.get (l, ()))
      = entryInt d.dtype (d.idsVal (tripG d t (l, ()))) := by
    rw [hentry, ite_eq_left hen]
  have hge2 : 0 ≤ entryInt d.dtype (d.idsVal (tripG d t (l, ()))) := by
    rw [← heq2]; exact hge
  rw [show gatherG d v (l, ()) = (entryInt d.dtype (v.get (l, ()))).toNat from rfl, heq2,
    Int.toNat_lt hge2]
  exact hbnd

/-! ## One trip with no map -/

/-- One trip without a map is safe, reads inside the ids, writes nothing and advances the invariant. -/
theorem safe_trip_false {d : Inst} (hd : d.ok) (hmap : d.hasMap = false) {init : Mem}
    (hpre : Pre d init) (E : Env) {e t : Nat} (ht : t < numTrips d) (curr : I32)
    (hc : curr.toInt = (e : Int)) (acc : Tensor [d.B] I32) (ptrs : Tensor [d.B] (Ptr d.dtype))
    (hacc : ∀ l : Fin d.B, (acc.get1 l).toInt = (cnt d e t l.val : Int))
    (hptr : ∀ l : Fin d.B, ptrs.get1 l = d.ids.ptr (tripG d t (l, ()) : Int)) :
    Safe E (Rin d) ∅ (fun st' m₂ => m₂ = init ∧
      (∀ l : Fin d.B, (st'.1.get1 l).toInt = (cnt d e (t + 1) l.val : Int)) ∧
      (∀ l : Fin d.B, st'.2.get1 l = d.ids.ptr (tripG d (t + 1) (l, ()) : Int)))
      init
      (loadBody d.dtype false d.B (d.emap.ptr 0) (IntV.ofNat d.numel) curr
        (IntV.ofInt (t : Int)) acc ptrs) := by
  rw [loadBody_false]
  have hx : (IntV.ofInt (t : Int) : I32).toInt = (t : Int) := by
    refine IntV.I32.toInt_ofInt ?_
    have := numTrips_lt hd
    constructor <;> omega
  refine safe_load_buf (g := tripG d t) (en := tripEn d t) d.ids ?_ ?_ ?_ ?_ ?_
  · refine Buf.addrs_of_get _ ?_ ?_
    · intro i
      exact maskIdx hd ht _ hx i
    · intro i hi
      obtain ⟨l, u⟩ := i
      cases u
      exact hptr l
  · intro i hi
    exact lt_of_lt_of_le ((tripEn_iff i).mp hi) hd.ids_len
  · exact hpre.1
  · intro x hx'
    obtain ⟨j, ⟨i, hi, rfl⟩, hxb⟩ := hx'
    exact Or.inl ⟨tripG d t i, (tripEn_iff i).mp hi, hxb⟩
  · intro v hv ho
    have ho₂ : ∀ i, tripEn d t i = false → v.get i = m1 d.dtype := fun i hi =>
      ho i hi (Tensor.full (m1 d.dtype)) rfl
    simp only [tailOf]
    refine safe_ret ⟨rfl, ?_, ?_⟩
    · intro l
      refine count_step hd ht (hacc l) ?_
      exact lane_eq_plain hd hmap hpre hv ho₂ curr hc l
    · intro l
      show (ptrs + (Tensor.full (IntV.ofNat d.B : I32) : Tensor [d.B] I32)).get1 l
        = d.ids.ptr (tripG d (t + 1) (l, ()) : Int)
      rw [get1_ptrAddT, get1_full, hptr l, ptr_next hd]

/-! ## One trip through the map -/

/-- One trip with a map is safe, reads inside the ids and the map, writes nothing and advances the invariant. -/
theorem safe_trip_true {d : Inst} (hd : d.ok) (hmap : d.hasMap = true) {init : Mem}
    (hpre : Pre d init) (E : Env) {e t : Nat} (ht : t < numTrips d) (curr : I32)
    (hc : curr.toInt = (e : Int)) (acc : Tensor [d.B] I32) (ptrs : Tensor [d.B] (Ptr d.dtype))
    (hacc : ∀ l : Fin d.B, (acc.get1 l).toInt = (cnt d e t l.val : Int))
    (hptr : ∀ l : Fin d.B, ptrs.get1 l = d.ids.ptr (tripG d t (l, ()) : Int)) :
    Safe E (Rin d) ∅ (fun st' m₂ => m₂ = init ∧
      (∀ l : Fin d.B, (st'.1.get1 l).toInt = (cnt d e (t + 1) l.val : Int)) ∧
      (∀ l : Fin d.B, st'.2.get1 l = d.ids.ptr (tripG d (t + 1) (l, ()) : Int)))
      init
      (loadBody d.dtype true d.B (d.emap.ptr 0) (IntV.ofNat d.numel) curr
        (IntV.ofInt (t : Int)) acc ptrs) := by
  rw [loadBody_true]
  have hx : (IntV.ofInt (t : Int) : I32).toInt = (t : Int) := by
    refine IntV.I32.toInt_ofInt ?_
    have := numTrips_lt hd
    constructor <;> omega
  refine safe_load_buf (g := tripG d t) (en := tripEn d t) d.ids ?_ ?_ ?_ ?_ ?_
  · refine Buf.addrs_of_get _ ?_ ?_
    · intro i
      exact maskIdx hd ht _ hx i
    · intro i hi
      obtain ⟨l, u⟩ := i
      cases u
      exact hptr l
  · intro i hi
    exact lt_of_lt_of_le ((tripEn_iff i).mp hi) hd.ids_len
  · exact hpre.1
  · intro x hx'
    obtain ⟨j, ⟨i, hi, rfl⟩, hxb⟩ := hx'
    exact Or.inl ⟨tripG d t i, (tripEn_iff i).mp hi, hxb⟩
  · intro v hv ho
    have ho₂ : ∀ i, tripEn d t i = false → v.get i = m1 d.dtype := fun i hi =>
      ho i hi (Tensor.full (m1 d.dtype)) rfl
    have hpv : ∀ i, gatherEn d v i = true → gatherG d v i < d.emap.len := by
      intro i hi
      exact lt_of_lt_of_le (gather_bound hd hmap hpre hv ho₂ i hi) (hd.emap_len hmap)
    have hRv : d.emap.elems (Buf.img (gatherG d v) (gatherEn d v)) ⊆ Rin d := by
      rw [Rin_map hmap]
      intro x hx'
      obtain ⟨j, ⟨i, hi, rfl⟩, hxb⟩ := hx'
      exact Or.inr ⟨gatherG d v i, gather_bound hd hmap hpre hv ho₂ i hi, hxb⟩
    refine safe_load_buf (g := gatherG d v) (en := gatherEn d v) d.emap ?_ hpv (hpre.2.2.2 hmap).1 hRv ?_
    · refine Buf.addrs_of_get _ ?_ ?_
      · intro i
        rfl
      · intro i hi
        obtain ⟨l, u⟩ := i
        cases u
        show (mapPtrs d.dtype (d.emap.ptr 0) v).get1 l
          = d.emap.ptr (gatherG d v (l, ()) : Int)
        rw [mapPtrs_get hd.dtype_ok]
        rw [Buf.ptr_offset]
        have hge : 0 ≤ entryInt d.dtype (v.get1 l) := (geZero_true hd.dtype_ok v l).mp hi
        rw [Int.zero_add]
        rw [show gatherG d v (l, ()) = (entryInt d.dtype (v.get1 l)).toNat from rfl,
          Int.toNat_of_nonneg hge]
    · intro v' hv' ho'
      have ho'₂ : ∀ i, gatherEn d v i = false → v'.get i = m1 .i32 := fun i hi =>
        ho' i hi (Tensor.full (m1 .i32)) rfl
      simp only [tailOf]
      refine safe_ret ⟨rfl, ?_, ?_⟩
      · intro l
        refine count_step hd ht (hacc l) ?_
        exact lane_eq_map hd hmap hpre hv ho₂ v' hv' ho'₂ curr hc l
      · intro l
        show (ptrs + (Tensor.full (IntV.ofNat d.B : I32) : Tensor [d.B] I32)).get1 l
          = d.ids.ptr (tripG d (t + 1) (l, ()) : Int)
        rw [get1_ptrAddT, get1_full, hptr l, ptr_next hd]

end Claims.CountExpertTokens
