import StreamKMatmul.Chunk

/-!
# `streamk_matmul_f16`: the end of an iteration

`safe_storeAdd`: lines 60–64 of the iteration from `it = n`. When the program owns the tile
(`owns`, exactly when line 60's condition holds, `iter_owns`), it stores the accumulator, which
is within `tolDot K A` of the exact sum (`TileOut`); otherwise the tile is split (`iter_split`)
and it adds the accumulator into the tile's elements with `tl.atomic_add`, one logged step per
element, paired with its operand (`LogInv`). Either way the while loop's invariant holds at the
end of the tile's part (`thi`), and the variant decreased.
-/

namespace Kernels.StreamKMatmul

open TVA

/-! ## Facts about one iteration -/

section
variable {d : Inst} {p n : Nat}

theorem pend_lt (hd : d.ok) : pend d p < 2 ^ 31 := by
  have h1 := hd.cover
  have h2 := hd.progs_lt
  unfold pend; omega

theorem iter_tile_lt (hd : d.ok) (h3 : n < pend d p) : n / steps d.K < ntiles d := by
  have hT := steps_pos hd
  have h := hd.total
  unfold pend at h3
  rw [Nat.div_lt_iff_lt_mul hT]
  unfold ntiles
  omega

theorem owns_kfirst {t : Nat} (h : owns d p t) : kfirst d p t = 0 := by
  unfold owns at h; unfold kfirst tlo; omega

theorem owns_klast {t : Nat} (h : owns d p t) : klast d p t = steps d.K := by
  unfold owns at h; unfold klast thi; omega

theorem kfirst_le_klast {t : Nat} (h : touches d p t) : kfirst d p t ≤ klast d p t := by
  unfold touches at h; unfold kfirst klast; omega

theorem iter_num (hd : d.ok) (h1 : pstart d p ≤ n) (h3 : n < pend d p) :
    pstart d p ≤ thi d p (n / steps d.K) ∧ thi d p (n / steps d.K) ≤ pend d p ∧
      (thi d p (n / steps d.K) % steps d.K = 0 ∨ pend d p ≤ thi d p (n / steps d.K)) := by
  have hT := steps_pos hd
  have hdm := Nat.div_add_mod n (steps d.K)
  rw [Nat.mul_comm] at hdm
  have hml := Nat.mod_lt n hT
  unfold thi
  refine ⟨by omega, by omega, ?_⟩
  by_cases h : pend d p ≤ n / steps d.K * steps d.K + steps d.K
  · right; omega
  · left
    rw [min_eq_right (by omega), ← Nat.succ_mul]
    exact Nat.mul_mod_left _ _

/-- The next value of `it` (line 64). -/
theorem iter_it (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0)
    (h3 : n < pend d p) {it k_first k_last : I32} (hit : it.toInt = n)
    (hkf : k_first.toInt = (kfirst d p (n / steps d.K) : Int))
    (hkl : k_last.toInt = (klast d p (n / steps d.K) : Int)) :
    (it + (k_last - k_first)).toInt = (thi d p (n / steps d.K) : Int) := by
  have hlo := iter_lo hd h1 h2
  have htc := iter_touches hd h1 h2 h3
  have hpl := pend_lt (p := p) hd
  have hkle := part_lt (d := d) p (n / steps d.K)
  have hTle : steps d.K ≤ 128 := by have := hd.K_le; unfold steps; omega
  have hdml := Nat.div_mul_le_self n (steps d.K)
  have hle := kfirst_le_klast htc
  have hthi : thi d p (n / steps d.K) ≤ pend d p := by unfold thi; omega
  unfold touches at htc
  have e1 : kfirst d p (n / steps d.K) = n - n / steps d.K * steps d.K := by unfold kfirst; omega
  have e2 : klast d p (n / steps d.K) = thi d p (n / steps d.K) - n / steps d.K * steps d.K := rfl
  have hsub : (k_last - k_first).toInt = ((klast d p (n / steps d.K) - kfirst d p (n / steps d.K) : Nat) : Int) :=
    IntV.I32.sub_nat hkl hkf hle (by omega)
  have hval : n + (klast d p (n / steps d.K) - kfirst d p (n / steps d.K)) = thi d p (n / steps d.K) := by
    rw [e1, e2]; omega
  rw [IntV.I32.add_nat hit hsub (by omega), hval]

theorem iter_mu (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0)
    (h3 : n < pend d p) {it' : I32} (hit' : it'.toInt = (thi d p (n / steps d.K) : Int)) :
    mu d p it' < pend d p - n := by
  have := iter_not_done hd h1 h2 h3
  unfold mu
  rw [hit', Int.toNat_natCast]
  omega

theorem iter_done' (hd : d.ok) (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0)
    (h3 : n < pend d p) {t : Nat} (ht : t ≠ n / steps d.K) :
    doneAt d p (thi d p (n / steps d.K)) t ↔ doneAt d p n t :=
  ⟨fun h => ((iter_done hd h1 h2 h3 t).mp h).resolve_left ht, fun h => (iter_done hd h1 h2 h3 t).mpr (Or.inr h)⟩

end

/-- A read of an element of tile `t` is unchanged by writes to another tile `t'`. -/
theorem read_other_tile {d : Inst} (hd : d.ok) {t t' : Nat} (h : t ≠ t') {x : Idx [128, 128]}
    (hx : cEn d t x = true) {m m' : Mem}
    (hfr : ∀ z, z ∉ d.c.elems (Buf.img (cG d t') (cEn d t')) → m' z = m z) :
    m'.read .f32 (d.c.ptr ↑(cG d t x)).addr = m.read .f32 (d.c.ptr ↑(cG d t x)).addr :=
  Mem.read_congr fun z hz => hfr z fun hz' =>
    Set.disjoint_left.mp (d.c.elems_disjoint (img_disj hd h)) (d.c.span_sub_elems (S := Buf.img (cG d t) (cEn d t)) ⟨x, hx, rfl⟩ hz) hz'

/-- A store to a tile of `c` keeps `a`, `b` and the buffers' validity. -/
theorem frame_tile {d : Inst} {init m m' : Mem} {t : Nat} (hd : d.ok) (hR : ∀ x ∈ R d, m x = init x)
    (hva : d.a.Valid m) (hvb : d.b.Valid m) (hvc : d.c.Valid m)
    (hfr : ∀ z, z ∉ d.c.elems (Buf.img (cG d t) (cEn d t)) → m' z = m z)
    (hpres : ∀ z, m z ≠ none → m' z ≠ none) :
    (∀ x ∈ R d, m' x = init x) ∧ d.a.Valid m' ∧ d.b.Valid m' ∧ d.c.Valid m' := by
  refine ⟨fun x hx => ?_, Buf.valid_of_agree hva fun x hx => hpres x (hva.2.2.2 x hx),
    Buf.valid_of_agree hvb fun x hx => hpres x (hvb.2.2.2 x hx),
    Buf.valid_of_agree hvc fun x hx => hpres x (hvc.2.2.2 x hx)⟩
  rw [← hR x hx]
  refine hfr x fun hc => ?_
  have hc' : x ∈ d.c.elems (cIdx d) := d.c.elems_mono (img_sub_cIdx t) hc
  rcases hx with ha | hb
  · exact Set.disjoint_left.mp hd.c_a hc' ha
  · exact Set.disjoint_left.mp hd.c_b hc' hb

/-- The operands of a tensor atomic's lanes, one lane `x` with operand `w`: the multiset `{w}`. -/
theorem filterMap_single {ι β : Type} [DecidableEq ι] (l : List ι) (hl : l.Nodup) {x : ι} (hx : x ∈ l)
    (w : β) : ((l.filterMap fun y => if y = x then some w else none : List β) : Multiset β) = {w} := by
  induction l with
  | nil => simp at hx
  | cons a l ih =>
    rw [List.nodup_cons] at hl
    by_cases ha : a = x
    · subst ha
      have : l.filterMap (fun y => if y = a then some w else none) = [] := by
        rw [List.filterMap_eq_nil_iff]
        intro y hy
        rw [ite_eq_right (fun h : y = a => hl.1 (h ▸ hy))]
      simp [this]
    · have hx' : x ∈ l := by
        rcases List.mem_cons.mp hx with h | h
        · exact absurd h.symm ha
        · exact h
      simp [ha, ih hl.2 hx']

/-- **Lines 60–64 of an iteration** from `it = n`: the store of an owned tile, or the atomic
addition into a split tile; the invariant of the while loop at the end of the tile's part. -/
theorem safe_storeAdd {d : Inst} (hd : d.ok) {init : Mem} {E : Env}
    (hat : Rules.Num.AtomicAddF32 E.P) (hatot : Rules.Num.AtomicAddTotal E.P Math.Float.Format.binary32)
    {p n : Nat} (hp : p < d.num_programs)
    (h1 : pstart d p ≤ n) (h2 : n = pstart d p ∨ n % steps d.K = 0) (h3 : n < pend d p)
    {log : List AtomicStep} {m : Mem}
    (hR : ∀ x ∈ R d, m x = init x) (hva : d.a.Valid m) (hvb : d.b.Valid m) (hvc : d.c.Valid m)
    (hmem : MemInv d init p (doneAt d p n) m) (hlog : LogInv d init p (doneAt d p n) log)
    {c_ptrs : Tensor [128, 128] (Ptr .f32)} {c_mask : Tensor [128, 128] Bool}
    (hcm : ∀ x, c_mask.get x = cEn d (n / steps d.K) x)
    (hcp : ∀ x, cEn d (n / steps d.K) x = true → c_ptrs.get x = d.c.ptr ↑(cG d (n / steps d.K) x))
    {it k_first k_last T : I32} (hit : it.toInt = n)
    (hkf : k_first.toInt = (kfirst d p (n / steps d.K) : Int))
    (hkl : k_last.toInt = (klast d p (n / steps d.K) : Int)) (hT : T.toInt = steps d.K)
    {acc : Tensor [128, 128] F32}
    (hacc : AccInv d init (n / steps d.K) (kfirst d p (n / steps d.K))
      (klast d p (n / steps d.K) - kfirst d p (n / steps d.K)) acc) :
    SafeA E (R d) (Wv d p (pend d p - n)) (cells d)
      (fun it' log' m' => LoopInv d init p it' log' m' ∧ mu d p it' < pend d p - n) log m
      (storeAdd c_ptrs c_mask it k_first k_last T acc) := by
  have hT0 := steps_pos hd
  have htc := iter_touches hd h1 h2 h3
  have hnum := iter_num hd h1 h3
  have hit' := iter_it hd h1 h2 h3 hit hkf hkl
  have hmu := iter_mu hd h1 h2 h3 hit'
  have hA : d.c.Addrs (tl.addrs c_ptrs (some c_mask)) (cG d (n / steps d.K)) (cEn d (n / steps d.K)) :=
    Buf.addrs_of_get d.c hcm hcp
  have hin : ∀ x, cEn d (n / steps d.K) x = true → cG d (n / steps d.K) x < d.c.len := by
    intro x he
    simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he
    exact hd.c_len _ _ he.1 he.2
  have hcond : (IntV.eq k_first (IntV.ofInt 0 : I32) && IntV.eq k_last T) = true ↔ owns d p (n / steps d.K) := by
    rw [i32_eq hkf i32_zero, i32_eq hkl hT, ← iter_owns hd h1 h2 h3, ← iter_klast hd h3, ← iter_kfirst hd h1 h2]
    simp only [Bool.and_eq_true, decide_eq_true_eq, Nat.cast_eq_zero, Nat.cast_inj]
  -- the invariant's numbers at the end of the part
  have hnext : ∀ log' m', (∀ x ∈ R d, m' x = init x) → d.a.Valid m' → d.b.Valid m' → d.c.Valid m' →
      MemInv d init p (doneAt d p (thi d p (n / steps d.K))) m' → LogInv d init p (doneAt d p (thi d p (n / steps d.K))) log' →
      LoopInv d init p (it + (k_last - k_first)) log' m' ∧ mu d p (it + (k_last - k_first)) < pend d p - n :=
    fun log' m' hR' hva' hvb' hvc' hmem' hlog' =>
      ⟨by unfold LoopInv; exact ⟨thi d p (n / steps d.K), hit', hnum.1, Or.inl hnum.2.1, Or.inr hnum.2.2, hR', hva', hvb', hvc', hmem', hlog'⟩, hmu⟩
  unfold storeAdd
  dsimp only
  by_cases how : owns d p (n / steps d.K)
  · -- line 61: the store of an owned tile
    rw [ite_eq_left (hcond.mpr how)]
    have hown := (iter_owns hd h1 h2 h3).mpr how
    have hsub : d.c.elems (Buf.img (cG d (n / steps d.K)) (cEn d (n / steps d.K))) ⊆ Wv d p (pend d p - n) := by
      unfold Wv
      rw [ite_eq_left ⟨Nat.sub_le _ _, by rw [Nat.sub_sub_self h3.le]; exact hown.1, by
        have := owns_klast how; unfold owns at how; omega⟩, Nat.sub_sub_self h3.le]
    have hinj : ∀ x y, cEn d (n / steps d.K) x = true → cEn d (n / steps d.K) y = true → cG d (n / steps d.K) x = cG d (n / steps d.K) y → x = y :=
      fun x y hx hy hg => (lane_inj hd hx hy hg).2
    have h := safe_store_buf (E := E) (R := R d) (Q := fun a m' => SafeA E (R d) ∅ (cells d)
        (fun it' log' m' => LoopInv d init p it' log' m' ∧ mu d p it' < pend d p - n) log m' (pure a))
      (k := fun _ => pure (it + (k_last - k_first))) (vals := acc) d.c hA hin hinj hvc hsub
      fun m' hm1 hm2 hm3 => safe_ret (SafeA.ret ?_)
    · have := SafeA.of_safe h (Set.disjoint_empty _)
      rw [Set.union_empty] at this
      exact this
    obtain ⟨hR', hva', hvb', hvc'⟩ := frame_tile hd hR hva hvb hvc hm2 hm3
    refine hnext log m' hR' hva' hvb' hvc' ?_ ?_
    · intro t x hot hdone hen
      rcases (iter_done hd h1 h2 h3 t).mp hdone with rfl | hdone'
      · refine ⟨acc.get x, hm1 x hen, fun va vb hra hcb => ?_⟩
        obtain ⟨hfin, herr⟩ := hacc x hen va vb hra hcb
        rw [owns_kfirst how, owns_klast how] at herr
        simp only [Nat.sub_zero, Nat.zero_add] at herr
        obtain ⟨hS, hA'⟩ := chunks_end va vb d.K
        unfold partS partA at herr
        rw [← Finset.range_eq_Ico, hS, hA'] at herr
        exact ⟨hfin, herr⟩
      · have hne : t ≠ (n / steps d.K) := by
          intro h; rw [h] at hdone'; exact iter_not_done hd h1 h2 h3 hdone'.2
        obtain ⟨c, hc, hcs⟩ := hmem t x hot hdone' hen
        exact ⟨c, by rw [read_other_tile hd hne hen hm2]; exact hc, hcs⟩
    · intro t x hw hen va vb hra hcb
      obtain ⟨ops, hpd, hdn, hnd⟩ := hlog t x hw hen va vb hra hcb
      have hne : t ≠ (n / steps d.K) := by
        intro h; rw [h] at hw; exact hw ⟨p, hp, how⟩
      have hiff := iter_done' hd h1 h2 h3 hne
      exact ⟨ops, hpd, fun h => hdn (hiff.mp h), fun h => hnd fun h' => h (hiff.mpr h')⟩
  · -- line 63: the atomic addition into a split tile
    rw [ite_eq_right (fun h => how (hcond.mp h))]
    have hsplit : ¬ whole d (n / steps d.K) := iter_split hd h1 h2 h3 fun h => how ((iter_owns hd h1 h2 h3).mp h)
    have hen_of : ∀ i, (tl.addrs c_ptrs (some c_mask) i).isSome = true → cEn d (n / steps d.K) i = true := by
      intro i hi
      rw [hA i] at hi
      by_contra hne
      rw [ite_eq_right hne] at hi
      exact absurd hi (by simp)
    refine SafeA.env ?_
    refine SafeA.atomic (fun i hi => ?_) rfl (by decide) (fun i old => hatot old (acc.get i))
      fun order hperm junk olds news hup => SafeA.ret (hnext _ m hR hva hvb hvc ?_ ?_)
    · have hen := hen_of i hi
      rw [hcp i hen]
      exact ⟨d.c.aligned hvc (hin i hen), ⟨(n / steps d.K), cG d (n / steps d.K) i, hsplit, ⟨i, hen, rfl⟩, rfl⟩⟩
    · intro t x hot hdone hen
      rcases (iter_done hd h1 h2 h3 t).mp hdone with rfl | hdone'
      · exact absurd hot how
      · exact hmem t x hot hdone' hen
    · have hnd' : order.Nodup := hperm.nodup_iff.mpr ((Idx.nodup_all _).filter _)
      have heny : ∀ y ∈ order, cEn d (n / steps d.K) y = true := fun y hy =>
        hen_of y (by simpa using (List.mem_filter.mp (hperm.subset hy)).2)
      intro t x hw hen va vb hra hcb
      obtain ⟨ops, hpd, hdn, hnd⟩ := hlog t x hw hen va vb hra hcb
      by_cases ht : t = (n / steps d.K)
      · subst ht
        have hops : ops = 0 := hnd fun h => iter_not_done hd h1 h2 h3 h.2
        obtain ⟨hfin, herr⟩ := hacc x hen va vb hra hcb
        have hxo : x ∈ order := hperm.mem_iff.mpr (List.mem_filter.mpr ⟨Idx.mem_all _ x, by
          rw [hA x, ite_eq_left hen]; rfl⟩)
        have hpair := Paired.append hpd (Paired.lanes order (fun i => (⟨.f32, (c_ptrs.get i).addr, olds i, news i⟩ : AtomicStep))
          (fun y => if y = x then some (acc.get x).toRat else none) (fun y hy => ?_))
        · rw [filterMap_single order hnd' hxo, hops, zero_add] at hpair
          refine ⟨{(acc.get x).toRat}, hpair, fun _ => ⟨(acc.get x).toRat, rfl, ?_⟩,
            fun h => absurd ((iter_done hd h1 h2 h3 _).mpr (Or.inl rfl)) h⟩
          have hkk : kfirst d p (n / steps d.K) + (klast d p (n / steps d.K) - kfirst d p (n / steps d.K)) =
              klast d p (n / steps d.K) := by have := kfirst_le_klast htc; omega
          rw [hkk] at herr
          exact herr
        · by_cases hyx : y = x
          · subst hyx
            rw [ite_eq_left rfl]
            show AccStep .f32 _ _ _ _ _ _ ⟨.f32, (c_ptrs.get y).addr, olds y, news y⟩ (some (acc.get y).toRat)
            rw [hcp y hen]
            exact accStep_atomicAdd hat (hup y hy) hfin
          · rw [ite_eq_right hyx]
            refine accStep_other ?_
            show Disjoint (Mem.span .f32 (c_ptrs.get y).addr) _
            rw [hcp y (heny y hy)]
            exact d.c.span_disjoint fun h => hyx (lane_inj hd (heny y hy) hen (by exact_mod_cast h)).2
      · have hiff := iter_done' hd h1 h2 h3 ht
        have hpair := Paired.append hpd (Paired.lanes order (fun i => (⟨.f32, (c_ptrs.get i).addr, olds i, news i⟩ : AtomicStep))
          (fun _ => (none : Option ℚ)) (fun y hy => ?_))
        · rw [show order.filterMap (fun _ => (none : Option ℚ)) = [] from
            List.filterMap_eq_nil_iff.mpr fun _ _ => rfl, Multiset.coe_nil, add_zero] at hpair
          exact ⟨ops, hpair, fun h => hdn (hiff.mp h), fun h => hnd fun h' => h (hiff.mpr h')⟩
        · refine accStep_other ?_
          show Disjoint (Mem.span .f32 (c_ptrs.get y).addr) _
          rw [hcp y (heny y hy)]
          exact d.c.span_disjoint fun h => ht (lane_inj hd hen (heny y hy) (by exact_mod_cast h.symm)).1

end Kernels.StreamKMatmul
