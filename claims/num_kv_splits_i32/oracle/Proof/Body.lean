import Proof.Loop

/-!
# The program

`safe_program` walks the body: the two loads (`safe_load_seq`), the folds (`max_val`, `min_val`),
the rule of line 77, the store of the uneven flag (`safe_uneven`), then lines 81-108 with the
stages of `Proof/Tail.lean` and the loop of `Proof/Loop.lean`.
-/

namespace Claims.NumKvSplits

open TVA

/-- **Lines 79-80**: with `uneven_batch`, its element `0` holds the flag `c`; the rest of `Wu` is
untouched, and the output stays valid. -/
theorem safe_uneven {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    {R : Set Int} (c : Bool) (hc : c = decide (minLen d init < maxLen d init)) {P : Mem → Prop}
    (hP : ∀ m1, (d.hasUneven = true → ∃ u : I32, m1.read .i32 (d.uneven.ptr 0).addr = some u ∧
        u.toInt = if minLen d init < maxLen d init then 1 else 0) →
      (∀ x, x ∉ Wu d → m1 x = init x) → d.out.Valid m1 → P m1) :
    Safe E R (Wu d) (fun _ m1 => P m1) init
      (match (if d.hasUneven = true then some (d.uneven.ptr 0) else none) with
        | some p => do
          let u ← tl.convert .i1 .i32 (Tensor.scalar c)
          tl.store1 p u.item
        | none => pure () : Prog Unit) := by
  cases hu : d.hasUneven with
  | false =>
    simp only [Bool.false_eq_true, ite_false]
    exact safe_ret (hP init (fun h => by rw [hu] at h; exact absurd h (by decide)) (fun _ _ => rfl)
      hpre.2.1)
  | true =>
    simp only [ite_true]
    refine safe_convert (fun _ => ⟨IntV.ofInt (if c then 1 else 0), rfl⟩) fun u hu' => ?_
    refine safe_last ?_
    have hU : Wu d = d.uneven.elems {0} := by simp [Wu, hu]
    have hv := hpre.2.2.1 hu
    have hlen := hd.uneven_len hu
    have hok : init.Writable .i32 (d.uneven.ptr 0).addr :=
      Buf.writable d.uneven hv (i := 0) (by omega)
    have hspan : Mem.span .i32 (d.uneven.ptr 0).addr ⊆ Wu d := by
      rw [hU]; exact Buf.span_sub_elems d.uneven (S := {0}) (j := 0) rfl
    refine safe_store1 hok hspan fun m' hread hframe => safe_ret ?_
    refine hP m' (fun _ => ⟨u.item, hread, ?_⟩) (fun x hx => hframe x fun hx' => hx (hspan hx'))
      (Buf.valid_of_agree hpre.2.1 fun x hx => ?_)
    · have h := hu' ()
      change u.item = IntV.ofInt (if c then 1 else 0) at h
      rw [h, hc]
      split_ifs with h1 h2 h2
      · exact IntV.toInt_ofInt_one (by norm_num)
      · simp at h1; exact absurd h1 h2
      · simp at h1; exact absurd h2 (not_lt.mpr h1)
      · exact IntV.toInt_ofInt_zero
    · have hxo : x ∉ Mem.span .i32 (d.uneven.ptr 0).addr := by
        intro hx'
        exact Set.disjoint_left.mp (hd.uneven_out hu)
          (Buf.span_sub d.uneven (j := 0) (by omega) hx') hx
      rw [hframe x hxo]; exact hpre.2.1.2.2.2 x hx

/-- The disjointness of the two write footprints. -/
theorem Wu_Wout {d : Inst} (hd : d.ok) : Disjoint (Wu d) (Wout d) := by
  unfold Wu
  split_ifs with hu
  · refine Disjoint.mono ?_ ?_ (hd.uneven_out hu)
    · exact Buf.elems_sub_bytes d.uneven fun j hj => by
        rw [Set.mem_singleton_iff.mp hj]; have := hd.uneven_len hu; omega
    · exact Buf.elems_sub_bytes d.out fun j hj => lt_of_lt_of_le hj hd.out_len
  · exact Set.empty_disjoint _

/-- **The program** is safe from `init`: it reads `Rd`, writes `Wu ∪ Wout`, and ends with `Qf`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    (hP : Device E.P) (c : Ctx) :
    Safe E (Rd d) (Wu d ∪ Wout d) (fun _ m => Qf d init m) init ((model d).body c) := by
  simp only [model, get_num_kv_splits_triton]
  refine safe_load_seq hd hpre.1 subset_rfl _ fun v0 hv0a hv0b => ?_
  refine safe_load_seq hd hpre.1 subset_rfl _ fun v1 hv1a hv1b => ?_
  have hmx := max_val hd hpre hv0a hv0b
  have hmn0 := min_val hd hpre hmx hv1a hv1b
  set MX := Tensor.foldAll IntV.max (wMin d.dL) (v0.map (toW d.dL)) with hMX
  set MN0 := Tensor.foldAll IntV.min (wMax d.dL) (v1.map (toW d.dL)) with hMN0
  obtain ⟨hM1, hM2, -⟩ := maxLen_bounds hd hpre
  obtain ⟨hm1, hm2⟩ := minLen0_bounds hd hpre
  have hp := pow_wZ hd.dL_ok
  have hw := w_two hd
  have h8 : (MX * IntV.ofInt 8).toInt = maxLen d init * 8 := by
    refine IntV.toInt_mul hmx (IntV.toInt_ofInt (by omega)
      (by rw [IntV.sfits_iff]; constructor <;> omega)) ?_
    rw [IntV.sfits_iff]; constructor <;> omega
  have h10 : (MN0 * IntV.ofInt 10).toInt = minLen0 d init * 10 := by
    refine IntV.toInt_mul hmn0 (IntV.toInt_ofInt (by omega)
      (by rw [IntV.sfits_iff]; constructor <;> omega)) ?_
    rw [IntV.sfits_iff]; constructor <;> omega
  set MN := (if (MX * IntV.ofInt 8).lt (MN0 * IntV.ofInt 10) = true then MX else MN0) with hMN
  have hmn : MN.toInt = minLen d init := by
    rw [hMN, IntV.lt_eq, h8, h10]
    unfold minLen
    by_cases h : maxLen d init * 8 < minLen0 d init * 10
    · simp [h, hmx]
    · simp [h, hmn0]
  refine Safe.seq (W₁ := Wu d) (W₂ := Wout d) ?_ (Wu_Wout hd)
  refine safe_uneven hd hpre (MN.lt MX) (by rw [IntV.lt_eq, hmn, hmx]) fun m1 hu hfr hv => ?_
  -- lines 81-108, from `m1`
  refine Safe.mono (Q := fun _ m => Qout d init m ∧ ∀ x, x ∉ Wout d → m x = m1 x) ?_ ?_ subset_rfl
  · have hsl : ∀ i : Idx [d.MAX - 0], ((v1.map (toW d.dL)).get i).toInt =
        if i.1.val < d.S then len d init i.1.val else maxLen d init := by
      intro i
      show (toW d.dL (v1.get i)).toInt = _
      split_ifs with hi
      · exact lane_len hd (hv1a i hi)
      · rw [hv1b i (Nat.le_of_not_lt hi), toW_ofW hd.dL_ok, hmx]
    refine safe_chunk1 hd hpre hmx hmn fun c1t hc1 => ?_
    refine safe_core hd hpre hP hmx fun et hcore hE21 => ?_
    have hE1 : 1 ≤ et.item.toInt := le_trans (by exact_mod_cast hd.C_pos) hcore.1
    refine safe_chunk2 hd hpre hmx hE1 hE21 fun c2t hc2 => ?_
    refine safe_splits hd hpre hE1 hsl hc1 hc2 fun n1 n2 hn => ?_
    exact (safe_loop hd hpre hv hE1 hn).mono (fun _ m h => ⟨⟨_, hcore, h.1⟩, h.2⟩) subset_rfl
  · rintro _ m ⟨hQ, hfr'⟩
    refine ⟨hQ, fun hU => ?_⟩
    obtain ⟨u, hr, hv'⟩ := hu hU
    refine ⟨u, ?_, hv'⟩
    rw [Mem.read_congr (m₂ := m1) (fun x hx => hfr' x ?_)]
    · exact hr
    · intro hxo
      have hxu : x ∈ Wu d := by
        simp only [Wu, hU, ite_true]
        exact Buf.span_sub_elems d.uneven (S := {0}) (j := 0) rfl hx
      exact Set.disjoint_left.mp (Wu_Wout hd) hxu hxo

end Claims.NumKvSplits
