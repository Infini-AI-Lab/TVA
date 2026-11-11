import Proof.Lane

import Proof.Sum

/-!
# The loop of `count_expert_tokens_i32`

One trip of the loop, the trip count `tl.cdiv(numel, B)`, and the invariant of the loop: after
`t` trips, lane `l` of the accumulator is the number of positions `x * B + l` (`x < t`) in range
whose routed expert is `e`, and the pointer block is the block of trip `t`.

This file proves that each trip is `Safe` with the ids' (and the map's) bytes as its read
footprint and no writes, and that the whole program of a grid is `Safe` with the ids', the map's
and element `e` of the output as footprints.
-/

namespace Claims.CountExpertTokens

open TVA

/-! ## The trip count -/

/-- The `int32` one has value `1`. -/
theorem toInt_one_i32 : (1 : I32).toInt = 1 := IntV.I32.toInt_ofNat (by norm_num)

/-- The trip argument `0 + t * 1` of `tl.forRange`, as an `int32`. -/
theorem tripArg (t : Nat) :
    (IntV.ofInt ((0 : I32).toInt + (t : Int) * (1 : I32).toInt) : I32)
      = IntV.ofInt (t : Int) := by
  rw [toInt_zero_i32, toInt_one_i32]
  norm_num

/-- `numel + (B - 1)` does not wrap. -/
theorem sum_numel_B {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.numel + IntV.ofNat (d.B - 1) : I32).toInt
      = ((d.numel + d.B - 1 : Nat) : Int) := by
  have hB := B_pos hd
  have h1 : (IntV.ofNat d.numel : I32).toInt = d.numel :=
    IntV.I32.toInt_ofNat (numel_lt hd)
  have h2 : (IntV.ofNat (d.B - 1) : I32).toInt = ((d.B - 1 : Nat) : Int) := by
    refine IntV.I32.toInt_ofNat ?_
    have := B_lt hd
    omega
  have h3 : d.numel + (d.B - 1) < 2 ^ 31 := by
    have := hd.numel_b
    omega
  have h := IntV.I32.add_nat h1 h2 h3
  rwa [show ((d.numel + (d.B - 1) : Nat) : Int)
      = ((d.numel + d.B - 1 : Nat) : Int) by
    rw [show d.numel + (d.B - 1) = d.numel + d.B - 1 by omega]] at h

/-- `BLOCK_SIZE` as an `int32` has its value. -/
theorem B_val {d : Inst} (hd : d.ok) : (IntV.ofNat d.B : I32).toInt = (d.B : Int) := by
  refine IntV.I32.toInt_ofNat ?_
  exact B_lt hd

/-- `tl.cdiv(numel, B)`'s value, as the trip count. -/
theorem trips_val {d : Inst} (hd : d.ok) {r : I32}
    (h : r.toInt = ((IntV.ofNat d.numel + IntV.ofNat (d.B - 1) : I32).toInt)
      / ((IntV.ofNat d.B : I32).toInt)) :
    r.toInt = (numTrips d : Int) := by
  rw [h, sum_numel_B hd, B_val hd]
  rw [numTrips, Int.natCast_div]

/-- The largest `int32` is `2^31 - 1`. -/
theorem intRange_32 : IntV.maxInt true 32 = 2 ^ 31 - 1 := by
  norm_num [IntV.maxInt]

/-! ## One trip, written out -/

/-- The body of one trip of the loop, with the state's components as parameters. -/
def loadBody (dd : DType) (hasMap : Bool) (B : Nat) (em : Ptr .i32) (numel curr x : I32)
    (acc : Tensor [B] I32) (ptrs : Tensor [B] (Ptr dd)) :
    Prog (Tensor [B] I32 × Tensor [B] (Ptr dd)) :=
  (tl.load ptrs (some ((tl.arange 0 B).ltS (numel - x * (IntV.ofNat B : I32))))
    (some (Tensor.full (m1 dd)))) >>= fun v =>
    match hasMap with
    | true =>
      (tl.load (mapPtrs dd em v) (some (geZero dd v)) (some (Tensor.full (m1 .i32)))) >>= fun v' =>
        tailOf acc ptrs (v'.eq (bc curr))
    | false => tailOf acc ptrs (eqCurr dd v curr)

/-- The loop body on a pair state is `loadBody`. -/
theorem loopBody_eq (dd : DType) (hasMap : Bool) (B : Nat) (em : Ptr .i32) (numel curr x : I32)
    (acc : Tensor [B] I32) (ptrs : Tensor [B] (Ptr dd)) :
    loopBody dd hasMap B em numel curr x (acc, ptrs) =
      loadBody dd hasMap B em numel curr x acc ptrs := rfl

/-- The position of trip `t`, lane `i`. -/
def tripG (d : Inst) (t : Nat) (i : Idx [d.B]) : Nat := t * d.B + (i.1 : Nat)

/-- The mask value of trip `t`, lane `i`. -/
def tripEn (d : Inst) (t : Nat) (i : Idx [d.B]) : Bool :=
  decide (t * d.B + (i.1 : Nat) < d.numel)

/-- The gather index of lane `i`, given the loaded ids of a trip. -/
def gatherG (d : Inst) (v : Tensor [d.B] d.dtype.Val) (i : Idx [d.B]) : Nat :=
  (entryInt d.dtype (v.get i)).toNat

/-- The gather's `geZero` test at lane `i`. -/
def gatherEn (d : Inst) (v : Tensor [d.B] d.dtype.Val) (i : Idx [d.B]) : Bool :=
  (geZero d.dtype v).get i

/-- `tripG` on a lane written as a pair. -/
theorem tripG_pair (d : Inst) (t : Nat) (l : Fin d.B) :
    tripG d t (l, ()) = t * d.B + l.val := rfl

/-- The trip's mask at lane `i` is `tripEn`. -/
theorem maskIdx {d : Inst} (hd : d.ok) {t : Nat} (ht : t < numTrips d) (x : I32)
    (hx : x.toInt = (t : Int)) (i : Idx [d.B]) :
    ((tl.arange 0 d.B).ltS (IntV.ofNat d.numel - x * (IntV.ofNat d.B : I32))).get i
      = tripEn d t i := by
  obtain ⟨l, u⟩ := i
  cases u
  exact mask_lane hd ht x hx l

/-! ## What a routed position is -/

/-- Without a map a position is routed to its entry. -/
theorem routed_noMap {d : Inst} (hm : d.hasMap = false) (j : Nat) :
    routed d j = entryInt d.dtype (d.idsVal j) := by
  simp only [routed, hm]
  rfl

/-- With a map a position is routed through the map (`-1` for a negative entry). -/
theorem routed_map {d : Inst} (hm : d.hasMap = true) (j : Nat) :
    routed d j =
      if 0 ≤ entryInt d.dtype (d.idsVal j) then
        (d.mapVal (entryInt d.dtype (d.idsVal j)).toNat).toInt
      else -1 := by
  simp only [routed, hm]
  rfl

/-- `tripEn` holds iff the position is below `numel`. -/
theorem tripEn_iff {d : Inst} {t : Nat} (i : Idx [d.B]) :
    tripEn d t i = true ↔ tripG d t i < d.numel := by
  simp only [tripEn]
  exact decide_eq_true_iff

/-- `tripEn` fails iff the position is not below `numel`. -/
theorem tripEn_false_iff {d : Inst} {t : Nat} (i : Idx [d.B]) :
    tripEn d t i = false ↔ ¬(tripG d t i < d.numel) := by
  simp only [tripEn]
  exact decide_eq_false_iff_not

/-! ## What a lane of the loaded ids means -/

/-- The loaded ids are the entries on enabled lanes and `-1` elsewhere. -/
theorem ids_entry {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {t : Nat}
    (v : Tensor [d.B] d.dtype.Val)
    (hv : ∀ i, tripEn d t i = true →
      init.read d.dtype (d.ids.ptr (tripG d t i : Int)).addr = some (v.get i))
    (ho : ∀ i, tripEn d t i = false → v.get i = m1 d.dtype) (i : Idx [d.B]) :
    entryInt d.dtype (v.get i)
      = if tripEn d t i then entryInt d.dtype (d.idsVal (tripG d t i)) else -1 := by
  by_cases hen : tripEn d t i = true
  · rw [ite_eq_left hen]
    have hlt : tripG d t i < d.numel := (tripEn_iff i).mp hen
    have hlen : tripG d t i < d.ids.len := lt_of_lt_of_le hlt hd.ids_len
    have h1 : init.read d.dtype (d.ids.ptr (tripG d t i : Int)).addr
        = some (d.idsVal (tripG d t i)) := hpre.2.2.1 _ hlen
    have h2 := hv i hen
    rw [h1] at h2
    have h3 : d.idsVal (tripG d t i) = v.get i := Option.some.inj h2
    rw [← h3]
  · have hf : tripEn d t i = false := by
      cases h : tripEn d t i <;> simp_all
    rw [ite_eq_right hen, ho i hf, entryInt_m1 hd.dtype_ok]

/-! ## What a lane of the gathered map means -/

/-- The gathered values are the mapped entries on enabled lanes and `-1` elsewhere. -/
theorem emap_entry {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {t : Nat}
    (v : Tensor [d.B] d.dtype.Val)
    (hv : ∀ i, tripEn d t i = true →
      init.read d.dtype (d.ids.ptr (tripG d t i : Int)).addr = some (v.get i))
    (ho : ∀ i, tripEn d t i = false → v.get i = m1 d.dtype)
    (v' : Tensor [d.B] I32)
    (hv' : ∀ i, gatherEn d v i = true →
      init.read .i32 (d.emap.ptr (gatherG d v i : Int)).addr = some (v'.get i))
    (ho' : ∀ i, gatherEn d v i = false → v'.get i = m1 .i32)
    (hmap : d.hasMap = true) (i : Idx [d.B]) :
    (v'.get i).toInt = if tripEn d t i then routed d (tripG d t i) else -1 := by
  obtain ⟨l, u⟩ := i
  cases u
  by_cases hen : tripEn d t (l, ()) = true
  · rw [ite_eq_left hen]
    have hlt : tripG d t (l, ()) < d.numel := (tripEn_iff (l, ())).mp hen
    have hlen : tripG d t (l, ()) < d.ids.len := lt_of_lt_of_le hlt hd.ids_len
    have h1 : init.read d.dtype (d.ids.ptr (tripG d t (l, ()) : Int)).addr
        = some (d.idsVal (tripG d t (l, ()))) := hpre.2.2.1 _ hlen
    have h2 := hv (l, ()) hen
    rw [h1] at h2
    have h3 : d.idsVal (tripG d t (l, ())) = v.get (l, ()) := Option.some.inj h2
    have hbnd : entryInt d.dtype (d.idsVal (tripG d t (l, ()))) < d.numGlobal :=
      (hpre.2.2.2 hmap).2.2 _ hlt
    by_cases hge : 0 ≤ entryInt d.dtype (v.get (l, ()))
    · have hge' : 0 ≤ entryInt d.dtype (d.idsVal (tripG d t (l, ()))) := by rw [h3]; exact hge
      have hg : gatherG d v (l, ()) = (entryInt d.dtype (d.idsVal (tripG d t (l, ())))).toNat := by
        simp only [gatherG]
        rw [h3]
      have hentrue : gatherEn d v (l, ()) = true := by
        show (geZero d.dtype v).get1 l = true
        rw [geZero_true hd.dtype_ok]
        exact hge
      have hread := hv' (l, ()) hentrue
      rw [hg] at hread
      have hlen2 : (entryInt d.dtype (d.idsVal (tripG d t (l, ())))).toNat < d.emap.len :=
        lt_of_lt_of_le (by
          rw [Int.toNat_lt hge']
          exact hbnd) (hd.emap_len hmap)
      have h4 : init.read .i32
            (d.emap.ptr ((entryInt d.dtype (d.idsVal (tripG d t (l, ())))).toNat : Int)).addr
          = some (d.mapVal (entryInt d.dtype (d.idsVal (tripG d t (l, ())))).toNat) :=
        (hpre.2.2.2 hmap).2.1 _ hlen2
      rw [h4] at hread
      have h5 : d.mapVal (entryInt d.dtype (d.idsVal (tripG d t (l, ())))).toNat
          = v'.get (l, ()) := Option.some.inj hread
      rw [← h5, routed_map hmap, ite_eq_left hge']
    · have hge' : ¬(0 ≤ entryInt d.dtype (d.idsVal (tripG d t (l, ())))) := by
        rw [h3]; exact hge
      have hentrue : gatherEn d v (l, ()) = false := by
        show (geZero d.dtype v).get1 l = false
        rw [geZero_false hd.dtype_ok]
        exact hge
      rw [ho' (l, ()) hentrue, routed_map hmap, ite_eq_right hge']
      exact IntV.I32.toInt_ofInt (by norm_num)
  · rw [ite_eq_right hen]
    have hentry : entryInt d.dtype (v.get (l, ()))
        = if tripEn d t (l, ()) then entryInt d.dtype (d.idsVal (tripG d t (l, ()))) else -1 :=
      ids_entry hd hpre v hv ho (l, ())
    have hneg : ¬(0 ≤ entryInt d.dtype (v.get (l, ()))) := by
      rw [hentry, ite_eq_right hen]
      norm_num
    have hentrue : gatherEn d v (l, ()) = false := by
      show (geZero d.dtype v).get1 l = false
      rw [geZero_false hd.dtype_ok]
      exact hneg
    rw [ho' (l, ()) hentrue]
    exact IntV.I32.toInt_ofInt (by norm_num)

end Claims.CountExpertTokens
