import Proof.Inv

/-!
# Running one program of the launch

The walk of one greedy request line by line (`safe_afterStart`, `safe_rest`), the case on the
`is_greedy` mask, and the final `safe_program`.
-/

namespace Claims.RejectionGreedy

open TVA

/-! ## `afterStart` in explicit form -/

/-- `afterStart` in explicit form: load `end_idx`, run the loop, then the bonus. -/
theorem afterStart_eq (d : Inst) (r : Nat) (start_idx : I32) :
    afterStart d r start_idx =
      (tl.load1 (d := .i32) (d.cu.ptr 0 + (IntV.ofNat r : I32)) >>= fun end_idx =>
        tl.forRange (0 : I32) (end_idx - start_idx) (1 : I32) false (tripBody d r start_idx)
          >>= fun rej =>
        if !rej then bonusStep d r (end_idx - start_idx) else pure ()) := by
  unfold afterStart
  rfl

/-! ## What the columns must hold

`EV` (the invariant's columns once the loop has run to the end) is `expect`'s value: always when
the loop rejected, and away from the bonus column when it did not. -/

/-- For a greedy request `expect` is the three-way case without the greedy test. -/
theorem expect_of_greedy {d : Inst} {init : Mem} {r : Nat} (hg : greedyAt d init r = true)
    (j : Nat) :
    expect d init r j =
      (if (j : Int) < min (matchLen d init r + 1) (nOf d init r) then
        some (rdt64 d.target init (startOf d init r + (j : Int)))
      else if (j : Int) = nOf d init r ∧ matchLen d init r = nOf d init r then
        some (rd32 d.bonus init (r : Int)) else none) := by
  unfold expect
  dsimp only
  rw [hg]
  simp only [Bool.true_eq_false, ite_false]

/-- When some draft mismatches, the columns after the loop are `expect`. -/
theorem EV_eq_expect_lt {d : Inst} {init : Mem} {r : Nat} (hg : greedyAt d init r = true)
    (hn0 : 0 ≤ nOf d init r) (hk : matchLen d init r < nOf d init r) (j : Nat) :
    EV d init r (nOf d init r).toNat j = expect d init r j := by
  have hnt : ((nOf d init r).toNat : Int) = nOf d init r := Int.toNat_of_nonneg hn0
  rw [expect_of_greedy hg]
  by_cases hc : (j : Int) ≤ matchLen d init r ∧ (j : Int) < nOf d init r
  · rw [EV_some hc.1 (by rw [hnt]; exact hc.2)]
    rw [ite_eq_left ((by
      have h1 : (j : Int) < matchLen d init r + 1 := by omega
      exact lt_min_iff.mpr ⟨h1, hc.2⟩))]
  · rw [EV_none (by rintro ⟨h1, h2⟩; exact hc ⟨h1, by rw [hnt] at h2; exact h2⟩)]
    rw [ite_eq_right ((by
      intro hcon
      rw [lt_min_iff] at hcon
      exact hc ⟨by omega, hcon.2⟩))]
    rw [ite_eq_right ((by rintro ⟨-, h2⟩; omega))]

/-- When every draft matches, the columns after the loop other than the bonus column are `expect`. -/
theorem EV_eq_expect_eq {d : Inst} {init : Mem} {r : Nat} (hg : greedyAt d init r = true)
    (hn0 : 0 ≤ nOf d init r) (hkn : matchLen d init r = nOf d init r) {j : Nat}
    (hjn : (j : Int) ≠ nOf d init r) :
    EV d init r (nOf d init r).toNat j = expect d init r j := by
  have hnt : ((nOf d init r).toNat : Int) = nOf d init r := Int.toNat_of_nonneg hn0
  rw [expect_of_greedy hg]
  by_cases hc : (j : Int) < nOf d init r
  · rw [EV_some (by rw [hkn]; omega) (by rw [hnt]; exact hc)]
    rw [ite_eq_left ((by
      rw [hkn]
      rw [min_eq_right (by omega)]
      exact hc))]
  · rw [EV_none (by rintro ⟨h1, h2⟩; rw [hkn] at h1; rw [hnt] at h2; omega)]
    rw [ite_eq_right ((by
      rw [hkn]
      rw [min_eq_right (by omega)]
      omega))]
    rw [ite_eq_right ((by rintro ⟨h1, -⟩; exact hjn h1))]

/-! ## The bonus step -/

/-- `bonusStep` in explicit form. -/
theorem bonusStep_eq (d : Inst) (r : Nat) (num : I32) :
    bonusStep d r num =
      (do
        let bonus_token_id ← tl.load1 (d := .i32) (d.bonus.ptr 0 + (IntV.ofNat r : I32))
        tl.store1 (d.out.ptr 0 + rowOff d r + num) bonus_token_id) := by
  unfold bonusStep
  rfl

/-- At the bonus column, `expect` wants `bonus[r]`. -/
theorem expect_at_end {d : Inst} {init : Mem} {r : Nat} (hg : greedyAt d init r = true)
    (hn0 : 0 ≤ nOf d init r) (hkn : matchLen d init r = nOf d init r) :
    expect d init r (nOf d init r).toNat = some (rd32 d.bonus init (r : Int)) := by
  have hnt : ((nOf d init r).toNat : Int) = nOf d init r := Int.toNat_of_nonneg hn0
  rw [expect_of_greedy hg]
  rw [ite_eq_right ((by
    rw [hkn, min_eq_right (show nOf d init r ≤ nOf d init r + 1 by omega), hnt]
    exact lt_irrefl _))]
  rw [ite_eq_left ((show ((nOf d init r).toNat : Int) = nOf d init r ∧
      matchLen d init r = nOf d init r from ⟨hnt, hkn⟩))]

/-! ## The body of a greedy request, from `startOf` on -/

/-- The body of a greedy request from `start_idx` on is safe, with footprints `Rf` and the row. -/
theorem safe_afterStart {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r : Nat}
    (hr : r < d.B) (hg : greedyAt d init r = true) (E : Env) {start_idx : I32}
    (hst : start_idx.toInt = startOf d init r) :
    Safe E (Rf d) (iterW (Wt d r) (nOf d init r).toNat ∪ Wt d r (nOf d init r).toNat)
      (fun _ m => Qf d init r m) init (afterStart d r start_idx) := by
  have hn0 : 0 ≤ nOf d init r := nOf_nonneg hpre hr
  have hnt : ((nOf d init r).toNat : Int) = nOf d init r := Int.toNat_of_nonneg hn0
  have hnS : (nOf d init r).toNat ≤ d.S := by
    have := hpre.nOf_le hr
    omega
  have hr31 : r < 2 ^ 31 := lt_trans hr hd.hgrid
  have hi0 : (0 : Int) ≤ ((r : Nat) : Int) := Int.natCast_nonneg _
  have hcu_len : ((r : Nat) : Int) < (d.cu.len : Int) := by
    exact_mod_cast lt_of_lt_of_le hr hd.hcu
  obtain ⟨v, hvread⟩ := buf_read_some_int d.cu hpre.cu_valid (i := ((r : Nat) : Int)) hi0 hcu_len
  rw [afterStart_eq, buf_ptr_add d.cu hr31]
  refine safe_load1 (d := .i32) hvread ?_ ?_
  · intro x hx
    exact cu_sub_Rf d (buf_span_sub_int d.cu hi0 hcu_len hx)
  have hnum : (v - start_idx).toInt = nOf d init r := by
    have hvs : IntV.SFits 32 (v.toInt - start_idx.toInt) := by
      have h1 : v.toInt = rd32 d.cu init ((r : Nat) : Int) := (rd32_eq hvread).symm
      have h3 := cu_le_22 hpre hr
      have h4 := startOf_nonneg hpre hr
      have h5 := startOf_add_nOf (d := d) init r
      simp only [IntV.SFits, h1, hst]
      omega
    rw [IntV.toInt_sub rfl rfl hvs, (rd32_eq hvread).symm, hst]
    have h := startOf_add_nOf (d := d) init r
    omega
  have htrip : tl.tripCount (0 : I32).toInt (v - start_idx).toInt (1 : I32).toInt
      = (nOf d init r).toNat := by
    rw [i32_zero_toInt, i32_one_toInt, hnum, tripCount_zero]
  rw [← htrip]
  refine safe_forRange (σ := Bool) (lo := (0 : I32)) (hi := v - start_idx) (step := (1 : I32))
    (init := false) (body := tripBody d r start_idx)
    (k := fun st => if !st then bonusStep d r (v - start_idx) else pure ())
    (I := fun t st m => Invs d init r t st m) (Wt := Wt d r)
    (Wk := Wt d r (tl.tripCount (0 : I32).toInt (v - start_idx).toInt (1 : I32).toInt))
    (rangeOk_zero_num (by rw [hnum]; exact hn0) (by rw [hnum]; exact nOf_lt_2p31 hd hpre hr))
    ?_ ?_ ?_ ?_ ?_
  · intro t u htu _
    exact Wt_disjoint (by omega)
  · refine Set.disjoint_left.mpr ?_
    rintro x ⟨t, ht, hxt⟩ hx
    refine Set.disjoint_left.mp (Wt_disjoint (d := d) (r := r) (t := t)
      (u := tl.tripCount (0 : I32).toInt (v - start_idx).toInt (1 : I32).toInt) ?_) hxt hx
    omega
  · exact Invs_init hpre hd hr
  · intro t ht st m₁ hI
    rw [htrip] at ht
    refine safe_trip hd hpre hr hst (n := (nOf d init r).toNat) rfl ht ?_ hI
    have ht31 : t < 2 ^ 31 := by
      have h := nOf_lt_2p31 hd hpre hr
      omega
    show (IntV.ofInt ((0 : I32).toInt + (t : Int) * (1 : I32).toInt) : I32).toInt = (t : Int)
    rw [i32_zero_toInt, i32_one_toInt, Int.zero_add, Int.mul_one]
    exact IntV.I32.toInt_ofInt ⟨by omega, by omega⟩
  · intro st m₁ hI
    rw [htrip] at hI ⊢
    cases st with
    | true =>
      have hk : matchLen d init r < nOf d init r := by
        rw [← hnt]
        exact hI.1.mp rfl
      refine safe_ret ?_
      intro j hj
      rw [← EV_eq_expect_lt hg hn0 hk j]
      exact hI.2.1 j hj
    | false =>
      have hk : ¬ matchLen d init r < ((nOf d init r).toNat : Int) := by
        intro h
        have h' := hI.1.mpr h
        simp at h'
      have hkle : ((nOf d init r).toNat : Int) ≤ matchLen d init r := not_lt.mp hk
      have heq : matchLen d init r = nOf d init r := by
        have h1 := matchLen_le_nOf d init r hn0
        omega
      show Safe E (Rf d) (Wt d r (nOf d init r).toNat) (fun _ m => Qf d init r m) m₁
        (bonusStep d r (v - start_idx))
      rw [bonusStep_eq]
      have hb_len : ((r : Nat) : Int) < (d.bonus.len : Int) := by
        exact_mod_cast lt_of_lt_of_le hr hd.hbonus
      obtain ⟨vb, hvb_init⟩ := buf_read_some_int d.bonus hpre.bonus_valid
        (i := ((r : Nat) : Int)) hi0 hb_len
      have hvb : m₁.read .i32 (d.bonus.ptr ((r : Nat) : Int)).addr = some vb := by
        rw [read_frame (r := r) hI.2.2.1 (Wf_disjoint_bonus hd hr) hi0 hb_len]
        exact hvb_init
      rw [buf_ptr_add d.bonus hr31]
      refine safe_load1 (d := .i32) hvb ?_ ?_
      · intro x hx
        exact bonus_sub_Rf d (buf_span_sub_int d.bonus hi0 hb_len hx)
      have hS1 : (nOf d init r).toNat < d.S + 1 := by omega
      have hoidx : oidx d r (nOf d init r).toNat < d.out.len := by
        have h1 : oidx d r (nOf d init r).toNat < (r + 1) * (d.S + 1) := by
          have he : (r + 1) * (d.S + 1) = r * (d.S + 1) + (d.S + 1) := Nat.succ_mul _ _
          rw [he, oidx]
          exact Nat.add_lt_add_left hS1 _
        exact Nat.lt_of_lt_of_le h1
          (le_trans (Nat.mul_le_mul_right _ (by omega : r + 1 ≤ d.B)) hd.hout_len)
      have hw : m₁.Writable .i32 (d.out.ptr (oidx d r (nOf d init r).toNat)).addr := by
        refine ⟨?_, ?_⟩
        · exact buf_aligned_int d.out hpre.out_valid (Int.natCast_nonneg _) (by exact_mod_cast hoidx)
        · intro k hkk
          exact hI.2.2.2 _ (Wt_sub_Wf hS1 (Mem.mem_span hkk))
      rw [out_ptr_eq (r := r) (t := (nOf d init r).toNat) (rowOff_toInt hd hr)
        (by rw [hnum, hnt])]
      show Safe E (Rf d) (Wt d r (nOf d init r).toNat) (fun _ m => Qf d init r m) m₁
        (tl.store (Tensor.scalar (d.out.ptr (oidx d r (nOf d init r).toNat)))
          (Tensor.scalar vb) none >>= fun _ => pure ())
      refine safe_store_any (ptrs := Tensor.scalar (d.out.ptr (oidx d r (nOf d init r).toNat)))
        (vals := Tensor.scalar vb) (mask := none) ?_ ?_ ?_
      · rw [storeOk_scalar]; exact hw
      · rw [accBytes_scalar]; exact subset_rfl
      intro order hperm
      refine safe_ret ?_
      have hread : (writeAll m₁ .i32
          (tl.addrs (Tensor.scalar (d.out.ptr (oidx d r (nOf d init r).toNat))) none)
          (Tensor.scalar vb) order).read .i32 (d.out.ptr (oidx d r (nOf d init r).toNat)).addr
          = some vb :=
        writeAll_read (distinctLanes_scalar _) ((storeOk_scalar _).mpr hw) order
          (hperm.nodup_iff.mpr (enabled_nodup _)) () (d.out.ptr (oidx d r (nOf d init r).toNat)).addr
          (hperm.mem_iff.mpr (mem_enabled rfl)) rfl
      intro j hj
      by_cases hjn : j = (nOf d init r).toNat
      · subst j
        rw [expect_at_end hg hn0 heq]
        exact ⟨vb, hread, (rd32_eq hvb_init).symm⟩
      · have hjn' : (j : Int) ≠ nOf d init r := by
          intro h
          apply hjn
          omega
        have hfr : (writeAll m₁ .i32
            (tl.addrs (Tensor.scalar (d.out.ptr (oidx d r (nOf d init r).toNat))) none)
            (Tensor.scalar vb) order).read .i32 (d.out.ptr (oidx d r j)).addr
            = m₁.read .i32 (d.out.ptr (oidx d r j)).addr := by
          apply Mem.read_congr
          intro x hx
          have hspan : Disjoint (Mem.span .i32 (d.out.ptr (oidx d r j)).addr)
              (accBytes .i32
                (tl.addrs (Tensor.scalar (d.out.ptr (oidx d r (nOf d init r).toNat))) none)) := by
            rw [accBytes_scalar]
            exact Wt_disjoint hjn
          exact writeAll_frame m₁ _ _ order (Set.disjoint_left.mp hspan hx)
        rw [← EV_eq_expect_eq hg hn0 heq hjn', hfr]
        exact hI.2.1 j hj

/-- A row shares no byte with the read footprint. -/
theorem Wf_disjoint_Rf {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.B) :
    Disjoint (Wf d r) (Rf d) := by
  refine Set.disjoint_left.mpr ?_
  intro x hx hxr
  have hxo : x ∈ d.out.bytes := Wf_sub_out_bytes hd hr hx
  unfold Rf at hxr
  rcases hxr with ((((h1 | h2) | h3) | h4) | h5)
  · exact Set.disjoint_left.mp hd.hout_cu hxo h1
  · exact Set.disjoint_left.mp hd.hout_draft hxo h2
  · exact Set.disjoint_left.mp hd.hout_target hxo h3
  · exact Set.disjoint_left.mp hd.hout_bonus hxo h4
  · cases hgr : d.greedy with
    | none => simp [hgr] at h5
    | some g => exact Set.disjoint_left.mp (hd.hout_greedy g hgr) hxo (by simpa [hgr] using h5)

/-- The body of a greedy request, from the `start_idx` branch on, is safe. -/
theorem safe_rest {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r : Nat}
    (hr : r < d.B) (hg : greedyAt d init r = true) (E : Env) :
    Safe E (Rf d) (iterW (Wt d r) (nOf d init r).toNat ∪ Wt d r (nOf d init r).toNat)
      (fun _ m => Qf d init r m) init (rest d r) := by
  have hr31 : r < 2 ^ 31 := lt_trans hr hd.hgrid
  unfold rest
  dsimp only
  split
  case isTrue =>
    rename_i hcond
    have hz : (IntV.ofNat r : I32).toInt = (0 : I32).toInt := (IntV.cmpKey_eq _ _).mp hcond
    rw [IntV.I32.toInt_ofNat hr31, i32_zero_toInt] at hz
    have hr0 : r = 0 := by omega
    have hst : ((tl.zeros [] .i32).item : I32).toInt = startOf d init r := by
      rw [hr0]
      have h0 : ((IntV.ofNat 0 : I32).toInt) = (0 : Int) := IntV.I32.toInt_ofNat (by norm_num)
      have h1 : startOf d init 0 = 0 := by simp [startOf]
      rw [h1]
      simp [tl.zeros, Tensor.full, Tensor.item, DType.ofBits] at h0 ⊢
    show Safe E (Rf d) (iterW (Wt d r) (nOf d init r).toNat ∪ Wt d r (nOf d init r).toNat)
      (fun _ m => Qf d init r m) init
      ((pure (tl.zeros [] .i32).item) >>= fun start_idx => afterStart d r start_idx)
    rw [show ((pure (tl.zeros [] .i32).item) >>= fun start_idx => afterStart d r start_idx)
        = afterStart d r (tl.zeros [] .i32).item from rfl]
    exact safe_afterStart hd hpre hr hg E hst
  case isFalse =>
    rename_i hne
    have hr0 : 0 < r := by
      rcases Nat.eq_zero_or_pos r with h | h
      · exfalso
        subst h
        exact hne rfl
      · exact h
    have hrne : r ≠ 0 := Nat.pos_iff_ne_zero.mp hr0
    have hi0 : (0 : Int) ≤ ((r - 1 : Nat) : Int) := Int.natCast_nonneg _
    have hcu_len : ((r - 1 : Nat) : Int) < (d.cu.len : Int) := by
      have h1 : r - 1 < d.B := by omega
      exact_mod_cast lt_of_lt_of_le h1 hd.hcu
    obtain ⟨v, hvread⟩ := buf_read_some_int d.cu hpre.cu_valid (i := ((r - 1 : Nat) : Int)) hi0 hcu_len
    have hs : startOf d init r = rd32 d.cu init ((r - 1 : Nat) : Int) := by
      simp [startOf, hrne]
    have hstart : v.toInt = startOf d init r := by
      rw [hs]
      exact (rd32_eq hvread).symm
    show Safe E (Rf d) (iterW (Wt d r) (nOf d init r).toNat ∪ Wt d r (nOf d init r).toNat)
      (fun _ m => Qf d init r m) init
      ((tl.load1 (d := .i32) (d.cu.ptr 0 + (IntV.ofNat r : I32) - (1 : I32))) >>= fun start_idx =>
        afterStart d r start_idx)
    rw [cu_ptr_sub hr0 hr31]
    refine safe_load1 (d := .i32) hvread ?_ ?_
    · intro x hx
      exact cu_sub_Rf d (buf_span_sub_int d.cu hi0 hcu_len hx)
    exact safe_afterStart hd hpre hr hg E hstart



/-- Without a mask, the program is `rest`. -/
theorem prog_none_eq (d : Inst) (r : Nat) (h : d.greedy = none) : prog d r = rest d r := by
  simp [prog, h]

/-- With a mask, the program loads the flag, then runs `rest` when it is set. -/
theorem prog_some_eq (d : Inst) (r : Nat) (g : Buf .i1) (h : d.greedy = some g) :
    prog d r = (do let is_greedy ← tl.load1 (d := .i1) (g.ptr 0 + (IntV.ofNat r : I32))
                   if !is_greedy then return () else rest d r) := by
  simp [prog, h]

/-- **One program** of the launch, request `r`, is safe from `init`: it reads inside `Rf`, writes inside its row, and leaves `Qf`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {r : Nat}
    (hr : r < d.B) :
    Safe E (Rf d) (Wf d r) (fun _ m => Qf d init r m) init (prog d r) := by
  have hr31 : r < 2 ^ 31 := lt_trans hr hd.hgrid
  have hn0 : 0 ≤ nOf d init r := nOf_nonneg hpre hr
  have hnS : (nOf d init r).toNat < d.S + 1 := by
    have h1 := hpre.nOf_le hr
    have h2 : (nOf d init r).toNat ≤ d.S := by omega
    omega
  have hwid : iterW (Wt d r) (nOf d init r).toNat ∪ Wt d r (nOf d init r).toNat ⊆ Wf d r := by
    intro x hx
    rcases hx with ⟨t, ht, hxt⟩ | hxt
    · exact Wt_sub_Wf (by omega) hxt
    · exact Wt_sub_Wf hnS hxt
  cases hgr : d.greedy with
  | none =>
    have hg : greedyAt d init r = true := by simp [greedyAt, hgr]
    rw [prog_none_eq d r hgr]
    exact Safe.mono (safe_rest hd hpre hr hg E) (fun _ _ h => h) hwid
  | some g =>
    have hglen : r < g.len := lt_of_lt_of_le hr (hd.hgreedy g hgr)
    have hgl : ((r : Nat) : Int) < (g.len : Int) := by exact_mod_cast hglen
    have hi0 : (0 : Int) ≤ ((r : Nat) : Int) := Int.natCast_nonneg _
    obtain ⟨v, hvread⟩ := buf_read_some_int g (hpre.greedy_valid hgr) (i := ((r : Nat) : Int)) hi0 hgl
    rw [prog_some_eq d r g hgr]
    rw [buf_ptr_add g hr31]
    refine safe_load1 (d := .i1) hvread ?_ ?_
    · intro x hx
      exact greedy_sub_Rf hgr (buf_span_sub_int g hi0 hgl hx)
    cases v with
    | false =>
      have hgf : greedyAt d init r = false := by
        rw [greedyAt_eq hgr hvread]
      refine safe_ret ?_
      intro j hj
      simp [expect, hgf]
    | true =>
      have hgt : greedyAt d init r = true := by
        rw [greedyAt_eq hgr hvread]
      exact Safe.mono (safe_rest hd hpre hr hgt E) (fun _ _ h => h) hwid

end Claims.RejectionGreedy
