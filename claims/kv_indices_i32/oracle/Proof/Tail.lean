import Proof.Trip

/-!
# `kv_indices_i32`: the tail of the program

After the row pointer `p` (`req_pool_indices[b]`), the request's first position `x` (`kv_start`)
and the output offset `off` (`kv_indptr[b]`) are known, the program loads `seq_lens[b]`, computes
the trip count and runs the masked copy loop (`Proof/Trip.lean`).
-/

namespace Claims.KvIndices

open TVA

/-- Trips `t` and `u` (`t < u`) of a program write disjoint elements of `kv_indices`. -/
theorem tripElems_disjoint {d : Inst} {init : Mem} {b : Nat} {t u : Nat} (htu : t < u) :
    Disjoint (tripElems d init b t) (tripElems d init b u) := by
  rw [Set.disjoint_left]
  rintro j ⟨l, hl, -, hj⟩ ⟨l', hl', -, hj'⟩
  have h : (t : Int) * 512 + (l : Int) = (u : Int) * 512 + (l' : Int) := by linarith
  have hu : (t : Int) + 1 ≤ (u : Int) := by exact_mod_cast htu
  have hl512 : (l : Int) < 512 := by exact_mod_cast hl
  have hl'0 : (0 : Int) ≤ (l' : Int) := by exact_mod_cast Nat.zero_le l'
  nlinarith

/-- The trip count `(s + 511) / 512` of a nonnegative length `s` covers `s`. -/
theorem lt_trip_mul {s : Int} (_h0 : 0 ≤ s) {j : Nat} (hj : (j : Int) < s) :
    (j : Int) < (((s + 511) / 512).toNat : Int) * 512 := by
  have hq0 : 0 ≤ (s + 511) / 512 := Int.ediv_nonneg (by linarith) (by norm_num)
  have hid : (s + 511) / 512 * 512 + (s + 511) % 512 = s + 511 := Int.ediv_mul_add_emod _ _
  have hlt : (s + 511) % 512 < 512 := Int.emod_lt_of_pos _ (by norm_num)
  have hge : s ≤ (s + 511) / 512 * 512 := by linarith
  have hcast : (((s + 511) / 512).toNat : Int) = (s + 511) / 512 := Int.toNat_of_nonneg hq0
  rw [hcast]
  linarith

/-- A nonnegative length below `2^31 - 511` has fewer than `2^31` positions after `512` trips. -/
theorem trip_mul_lt {s : Int} (h0 : 0 ≤ s) (hs : s + 511 < 2 ^ 31) {t : Nat}
    (ht : (t : Int) < (((s + 511) / 512).toNat : Int)) : (t : Int) * 512 < 2 ^ 31 := by
  have hq0 : 0 ≤ (s + 511) / 512 := Int.ediv_nonneg (by linarith) (by norm_num)
  have hcast : (((s + 511) / 512).toNat : Int) = (s + 511) / 512 := Int.toNat_of_nonneg hq0
  rw [hcast] at ht
  have hid : (s + 511) / 512 * 512 + (s + 511) % 512 = s + 511 := Int.ediv_mul_add_emod _ _
  have hmod : 0 ≤ (s + 511) % 512 := Int.emod_nonneg _ (by norm_num)
  have hle : (s + 511) / 512 * 512 ≤ s + 511 := by linarith
  have ht1 : (t : Int) + 1 ≤ (s + 511) / 512 := by omega
  nlinarith

/-- The tail of the program, run from the initial memory: it loads `seq_lens[b]`, computes the
trip count and runs the loop, leaving `Q`. -/
theorem safe_tail {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (b : Nat) (hb : b < d.bs) (p : I64) (x off : I32)
    (hp : p.toInt = rpi d init b) (hx : x.toInt = sidx d init b)
    (hoff : off.toInt = ind d init b) :
    Safe E (R d init b) (W d init b) (fun _ m' => Q d init b m') init
      (tailProg d b p x off) := by
  have hb31 : b < 2 ^ 31 := by have := hd.grid_lt; omega
  have hlen : b < d.seq_lens.len := by have := hd.llen; omega
  obtain ⟨v0, hv0⟩ := Buf.read_some d.seq_lens hpre.2.2.1 hlen
  have hv0' : init.read d.dL (d.seq_lens.ptr 0 + (IntV.ofNat b : I32)).addr = some v0 := by
    rw [ptr_zero_add d.seq_lens b hb31]
    exact hv0
  obtain ⟨hs0, hs1, hi0, hi1, hw⟩ := hpre.2.2.2.2.2.2.1 b hb
  simp only [tailProg]
  refine safe_load1 (v0 := v0) hv0' ?_ ?_
  · intro z hz
    rw [ptr_zero_add d.seq_lens b hb31] at hz
    have hsub := Buf.span_sub_elems d.seq_lens (S := {b}) (j := b) (by simp)
    simp only [R]
    exact Or.inl (Or.inl (Or.inr (hsub hz)))
  refine safe_convert (d := d.dL) (d' := .i32) (x := Tensor.scalar v0) ?_ ?_
  · intro u
    simpa only [Tensor.scalar] using exists_convRel_i32 hd.dL_ok E.P v0
  · intro r hr
    have hslen : slen d init b = ivOf v0 := by simp only [slen, valOf, hv0]
    have hfit : IntV.SFits 32 (ivOf v0) := by
      rw [← hslen]
      exact IntV.I32.sfits (by linarith) (by linarith)
    have hs32 : r.item.toInt = slen d init b := by
      have h := hr ()
      have hv : r.item.toInt = ivOf v0 := by
        simpa only [Tensor.scalar, Tensor.item] using convRel_i32_ivOf hd.dL_ok h hfit
      rw [hv, hslen]
    have h511 : (IntV.ofNat 511 : I32).toInt = 511 := IntV.I32.toInt_ofNat (by norm_num)
    have h512 : (IntV.ofNat 512 : I32).toInt = 512 := IntV.I32.toInt_ofNat (by norm_num)
    have hlo1 : (IntV.ofInt 0 : I32).toInt = 0 :=
      IntV.toInt_ofInt (w := 32) (by norm_num) (IntV.I32.sfits (by norm_num) (by norm_num))
    have hst1 : (IntV.ofInt 1 : I32).toInt = 1 :=
      IntV.toInt_ofInt (w := 32) (by norm_num) (IntV.I32.sfits (by norm_num) (by norm_num))
    have hsub : ((x + r.item) - x).toInt = slen d init b := by
      have h1 : r.item.toInt < (2 : Int) ^ (32 - 1) := by
        rw [hs32]
        have e : ((2 : Int) ^ (32 - 1 : Nat)) = 2 ^ 31 := by norm_num
        rw [e]
        linarith
      rw [sub_add_self_toInt (w := 32) (by norm_num) (by rw [hs32]; exact hs0) h1, hs32]
    have hnum : (x + r.item - x + (IntV.ofNat 511 : I32)).toInt = slen d init b + 511 := by
      have hf : IntV.SFits 32 (((x + r.item) - x).toInt + (IntV.ofNat 511 : I32).toInt) := by
        rw [hsub, h511]
        exact IntV.I32.sfits (by linarith) (by linarith)
      rw [IntV.toInt_add rfl rfl hf, hsub, h511]
    have hn0 : 0 ≤ (x + r.item - x + (IntV.ofNat 511 : I32)).toInt := by
      rw [hnum]
      linarith
    have hy0 : 0 < (IntV.ofNat 512 : I32).toInt := by
      rw [h512]
      norm_num
    refine safe_idiv1 (x := x + r.item - x + (IntV.ofNat 511 : I32))
      (y := (IntV.ofNat 512 : I32)) hn0 hy0 ?_
    intro n hn
    have hnval : n.item.toInt = (slen d init b + 511) / 512 := by
      rw [hn, hnum, h512]
    have hq0 : 0 ≤ (slen d init b + 511) / 512 := Int.ediv_nonneg (by linarith) (by norm_num)
    have hqid : (slen d init b + 511) / 512 * 512 + (slen d init b + 511) % 512
        = slen d init b + 511 := Int.ediv_mul_add_emod _ _
    have hqmod0 : 0 ≤ (slen d init b + 511) % 512 := Int.emod_nonneg _ (by norm_num)
    have hqmodlt : (slen d init b + 511) % 512 < 512 := Int.emod_lt_of_pos _ (by norm_num)
    have hqle : (slen d init b + 511) / 512 * 512 ≤ slen d init b + 511 := by linarith
    have hq31 : (slen d init b + 511) / 512 ≤ 2 ^ 31 - 1 := by
      have h1 : (slen d init b + 511) / 512 ≤ (slen d init b + 511) / 512 * 512 := by linarith
      linarith
    have hir : IntV.maxInt true 32 = 2 ^ 31 - 1 := by
      norm_num [IntV.maxInt]
    have hN : tl.tripCount (IntV.ofInt 0 : I32).toInt n.item.toInt (IntV.ofInt 1 : I32).toInt
        = ((slen d init b + 511) / 512).toNat := by
      simp only [tl.tripCount, hlo1, hst1, hnval]
      norm_num
    have hrange : tl.RangeOk (IntV.ofInt 0 : I32) n.item (IntV.ofInt 1 : I32) := by
      refine ⟨?_, ?_, ?_, ?_⟩
      · rw [hst1]
        norm_num
      · rw [hnval, hst1, hir]
        linarith
      · rw [hlo1, hir]
        linarith
      · rw [hst1, hir]
        norm_num
    let I : Nat → Unit → Mem → Prop := fun t _ m =>
      Qpart d init b t m ∧ (∀ z ∈ d.req_to_token.bytes, m z = init z) ∧
        d.kv_indices.Valid m ∧ d.req_to_token.Valid m
    let Wt : Nat → Set Int := fun t => d.kv_indices.elems (tripElems d init b t)
    refine Safe.mono (Q := fun _ m' => Q d init b m')
      (W := iterW Wt (tl.tripCount (IntV.ofInt 0 : I32).toInt n.item.toInt
        (IntV.ofInt 1 : I32).toInt) ∪ ∅) ?_ (fun _ _ h => h) ?_
    · refine safe_forRange (lo := (IntV.ofInt 0 : I32)) (hi := n.item)
        (step := (IntV.ofInt 1 : I32)) (init := ())
        (body := fun i _ => tripProg d p x r.item i off) (k := fun _ => pure ())
        (I := I) (Wt := Wt) (Wk := ∅) hrange ?_ ?_ ?_ ?_ ?_
      · intro t u htu _
        exact Buf.elems_disjoint d.kv_indices (tripElems_disjoint htu)
      · exact Set.disjoint_empty _
      · dsimp only [I]
        refine ⟨?_, (fun _ _ => rfl), hpre.2.2.2.2.1, hpre.1⟩
        intro j hj hj0
        simp only [Nat.cast_zero, Int.zero_mul] at hj0
        have hj0' : (0 : Int) ≤ (j : Int) := by exact_mod_cast Nat.zero_le j
        omega
      · intro t ht st m₁ hI
        rw [hN] at ht
        have htN : (t : Int) < (((slen d init b + 511) / 512).toNat : Int) := by
          exact_mod_cast ht
        have htq : (t : Int) < (slen d init b + 511) / 512 := by
          have h2 := htN
          rwa [Int.toNat_of_nonneg hq0] at h2
        have ht1 : (t : Int) * 512 < 2 ^ 31 :=
          trip_mul_lt (s := slen d init b) (by linarith) hs1 htN
        have ht0 : (0 : Int) ≤ (t : Int) := by exact_mod_cast Nat.zero_le t
        have htsmall : (0 : Int) + (t : Int) * 1 < 2 ^ 31 := by nlinarith
        have hi' : ((IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) *
            (IntV.ofInt 1 : I32).toInt) : I32)).toInt = (t : Int) := by
          rw [hlo1, hst1]
          have hsf : IntV.SFits 32 ((0 : Int) + (t : Int) * 1) :=
            IntV.I32.sfits (by linarith) (by linarith)
          have h := IntV.toInt_ofInt (w := 32) (by norm_num) hsf
          simpa using h
        dsimp only [I] at hI
        obtain ⟨hQ, hT, hVi, hVt⟩ := hI
        exact safe_trip hd hpre E b hb p x r.item
          (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt))
          off hp hx hs32 hoff t hi' ht1 m₁ hQ hT hVi hVt
      · intro st m₁ hI
        rw [hN] at hI
        dsimp only [I] at hI
        obtain ⟨hQ, -, -, -⟩ := hI
        refine safe_ret ?_
        intro j hj
        have hjq : (j : Int) < (((slen d init b + 511) / 512).toNat : Int) * 512 :=
          lt_trip_mul (s := slen d init b) (by linarith) hj
        exact hQ j hj hjq
    · intro z hz
      rcases hz with hz | hz
      · obtain ⟨t, -, hzt⟩ := hz
        exact Buf.elems_mono d.kv_indices (tripElems_sub d init b t) hzt
      · exact absurd hz (by simp)

end Claims.KvIndices
