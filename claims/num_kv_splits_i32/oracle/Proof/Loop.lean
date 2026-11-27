import Proof.Tail

/-!
# Source lines 105-108: the store loop

Trip `t` of `for i in range(0, num_group)` stores lane `s < num_seq` at element `s·num_group + t`
of `num_kv_splits`: the trips write the disjoint sets `{j | j % num_group = t}`, and after the
last trip every element `j < num_seq·num_group` holds the split count of request
`j / num_group`.
-/

namespace Claims.NumKvSplits

open TVA

/-- The elements trip `t` writes. -/
def tripS (d : Inst) (t : Nat) : Set Nat := {j | j < d.S * d.G ∧ j % d.G = t}

/-- The elements the first `t` trips write. -/
def doneS (d : Inst) (t : Nat) : Set Nat := {j | j < d.S * d.G ∧ j % d.G < t}

/-- `num_group ≤ 2^30`. -/
theorem G_le {d : Inst} (hd : d.ok) : d.G ≤ 2 ^ 30 := by
  have := hd.grid_le
  have : d.G ≤ d.S * d.G * d.H := by
    have := Nat.mul_pos hd.S_pos (le_trans hd.Hkv_pos hd.Hkv_le)
    nlinarith
  omega

/-- `num_group` as an `int32`. -/
theorem G_val {d : Inst} (hd : d.ok) : (IntV.ofNat d.G : I32).toInt = d.G :=
  ofNat_val (by have := G_le hd; omega)

/-- `num_seq * num_group` as an `int32`. -/
theorem SG_val {d : Inst} (hd : d.ok) :
    (IntV.ofNat d.S * IntV.ofNat d.G : I32).toInt = ((d.S * d.G : ℕ) : ℤ) := by
  have h1 : d.S * d.G ≤ d.S * d.G * d.H :=
    Nat.le_mul_of_pos_right _ (le_trans hd.Hkv_pos hd.Hkv_le)
  have := hd.grid_le
  exact IntV.I32.mul_nat (S_val hd) (G_val hd) (by omega)

/-- Lane `i` of `offs_token = offs_seq * num_group` has value `i · num_group`. -/
theorem offs_val {d : Inst} (hd : d.ok) (i : Idx [d.MAX - 0]) :
    ((tl.arange 0 d.MAX * (IntV.ofNat d.G : I32)).get i).toInt = ((i.1.val * d.G : ℕ) : ℤ) := by
  rw [Tensor.get_muls]
  have hi : i.1.val < d.MAX := by have := i.1.isLt; omega
  have : i.1.val * d.G < 2 ^ 31 := by
    have := hd.MAX_G
    have : i.1.val * d.G < d.MAX * d.G := Nat.mul_lt_mul_of_pos_right hi hd.G_pos
    omega
  exact IntV.I32.mul_nat (arange_val hd i) (G_val hd) this

/-- The store's lanes in trip `t`: lane `i` writes element `i · num_group + t`, enabled for
`i < num_seq`. -/
theorem out_addrs {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.G) :
    d.out.Addrs (tl.addrs (d.out.ptr 0 + (IntV.ofInt (0 + (t : ℤ) * 1) : I32) +
        tl.arange 0 d.MAX * (IntV.ofNat d.G : I32))
      (some ((tl.arange 0 d.MAX * (IntV.ofNat d.G : I32)).ltS
        (IntV.ofNat d.S * IntV.ofNat d.G))))
      (fun i => i.1.val * d.G + t) (fun i => decide (i.1.val < d.S)) := by
  have htv : (IntV.ofInt (0 + (t : ℤ) * 1) : I32).toInt = t := by
    rw [IntV.I32.toInt_ofInt (by have := G_le hd; constructor <;> omega)]
    ring
  refine Buf.addrs_of_get d.out (fun i => ?_) (fun i _ => ?_)
  · rw [Tensor.get_ltS, IntV.lt_eq, offs_val hd i, SG_val hd]
    simp only [Nat.cast_lt, decide_eq_decide]
    exact Nat.mul_lt_mul_right hd.G_pos
  · show ((d.out.ptr 0).offset (IntV.ofInt (0 + (t : ℤ) * 1) : I32).toInt).offset _ = _
    rw [htv, offs_val hd i, Buf.ptr_offset, Buf.ptr_offset]
    congr 1; push_cast; ring

/-- Element `j` is lane `j / num_group`'s in trip `j % num_group`. -/
theorem split_j {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.S * d.G) :
    j / d.G < d.S ∧ j / d.G * d.G + j % d.G = j := by
  have hG := hd.G_pos
  refine ⟨?_, Nat.div_add_mod' j d.G⟩
  exact (Nat.div_lt_iff_lt_mul hG).mpr hj

/-- The value a split count keeps as `int32`. -/
theorem cast_val {d : Inst} {x : IntV true (wOf d.dL)} (h1 : 1 ≤ x.toInt) (h2 : x.toInt ≤ 2 ^ 22) :
    (IntV.cast x : I32).toInt = x.toInt :=
  IntV.toInt_cast_of_sfits x (by norm_num) (by rw [IntV.I32.sfits_iff]; constructor <;> omega)

/-- **The store loop** (lines 107-108): from a memory `m1` with a valid output, the loop writes
`Wout`, and ends with every element `j < num_seq·num_group` holding `numSplits Ev (j / G)`, and
every other byte as in `m1`. -/
theorem safe_loop {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env} {R : Set Int}
    {m1 : Mem} (hv : d.out.Valid m1) {Ev : ℤ} (hE : 1 ≤ Ev)
    {nks : Tensor [d.MAX - 0] (IntV true (wOf d.dL))}
    (hn : ∀ i : Idx [d.MAX - 0], i.1.val < d.S → (nks.get i).toInt = numSplits d init Ev i.1.val) :
    Safe E R (Wout d) (fun _ m => (∀ j : Nat, j < d.S * d.G → ∃ v : I32,
        m.read .i32 (d.out.ptr (j : ℤ)).addr = some v ∧ v.toInt = numSplits d init Ev (j / d.G)) ∧
        ∀ x, x ∉ Wout d → m x = m1 x) m1
      (tl.forRange (0 : I32) (IntV.ofNat d.G) (1 : I32) () fun i _ =>
        tl.store (d.out.ptr 0 + i + tl.arange 0 d.MAX * (IntV.ofNat d.G : I32))
          (Tensor.castInt nks : Tensor [d.MAX - 0] I32)
          (some ((tl.arange 0 d.MAX * (IntV.ofNat d.G : I32)).ltS
            (IntV.ofNat d.S * IntV.ofNat d.G)))) := by
  have hG := hd.G_pos
  have hGv := G_val hd
  have h0 : (0 : I32).toInt = 0 := rfl
  have h1 : (1 : I32).toInt = 1 := rfl
  have htc : tl.tripCount (0 : I32).toInt (IntV.ofNat d.G : I32).toInt (1 : I32).toInt = d.G := by
    rw [h0, h1, hGv]; simp [tl.tripCount]
  have hrange : tl.RangeOk (0 : I32) (IntV.ofNat d.G) (1 : I32) := by
    refine ⟨by rw [h1]; norm_num, ?_, ?_, ?_⟩
    · rw [hGv, h1]
      have := G_le hd
      simp [IntV.maxInt]; omega
    · rw [h0]; simp [IntV.maxInt]
    · rw [h1]; simp [IntV.maxInt]
  -- the values stored
  have hval : ∀ i : Idx [d.MAX - 0], i.1.val < d.S →
      ((Tensor.castInt nks : Tensor [d.MAX - 0] I32).get i).toInt = numSplits d init Ev i.1.val := by
    intro i hi
    obtain ⟨a, b⟩ := numSplits_bounds hd hpre hE hi
    rw [Tensor.get_castInt, cast_val (by rw [hn i hi]; exact a) (by rw [hn i hi]; exact b), hn i hi]
  let I : Nat → Unit → Mem → Prop := fun t _ m =>
    (∀ j : Nat, j ∈ doneS d t → ∃ v : I32,
      m.read .i32 (d.out.ptr (j : ℤ)).addr = some v ∧ v.toInt = numSplits d init Ev (j / d.G)) ∧
    (∀ x, x ∉ d.out.elems (doneS d t) → m x = m1 x) ∧ d.out.Valid m
  have hmain : Safe E R (iterW (fun t => d.out.elems (tripS d t))
      (tl.tripCount (0 : I32).toInt (IntV.ofNat d.G : I32).toInt (1 : I32).toInt) ∪ ∅)
      (fun _ m => I d.G () m) m1
      (tl.forRange (0 : I32) (IntV.ofNat d.G) (1 : I32) () (fun i _ =>
        tl.store (d.out.ptr 0 + i + tl.arange 0 d.MAX * (IntV.ofNat d.G : I32))
          (Tensor.castInt nks : Tensor [d.MAX - 0] I32)
          (some ((tl.arange 0 d.MAX * (IntV.ofNat d.G : I32)).ltS
            (IntV.ofNat d.S * IntV.ofNat d.G)))) >>= pure) := by
    refine safe_forRange hrange I (fun t => d.out.elems (tripS d t)) (Wk := ∅) ?_ ?_ ?_ ?_ ?_
    · intro t u htu _
      refine d.out.elems_disjoint (Set.disjoint_left.mpr ?_)
      rintro j ⟨-, hj⟩ ⟨-, hj'⟩; omega
    · exact Set.disjoint_empty _
    · refine ⟨fun j ⟨_, hj⟩ => absurd hj (by omega), fun x _ => rfl, hv⟩
    · intro t ht _ m₁ ⟨hI1, hI2, hI3⟩
      rw [htc] at ht
      refine safe_last ?_
      refine safe_store_buf d.out (out_addrs hd ht) (fun i hi => ?_) (fun i j hi hj hij => ?_)
        hI3 ?_ fun m' hrd hfr hpres => safe_ret ?_
      · simp only [decide_eq_true_eq] at hi
        have := hd.out_len
        have : (i.1.val + 1) * d.G ≤ d.S * d.G := Nat.mul_le_mul_right _ hi
        show i.1.val * d.G + t < d.out.len
        nlinarith
      · rcases i with ⟨i, ⟨⟩⟩; rcases j with ⟨j, ⟨⟩⟩
        have : (i : ℕ) * d.G = j * d.G := by simpa using hij
        exact Prod.ext (Fin.ext (Nat.eq_of_mul_eq_mul_right hG this)) rfl
      · rintro x ⟨j, ⟨i, hi, rfl⟩, hx⟩
        simp only [decide_eq_true_eq] at hi
        refine ⟨i.1.val * d.G + t, ⟨?_, ?_⟩, hx⟩
        · have : (i.1.val + 1) * d.G ≤ d.S * d.G := Nat.mul_le_mul_right _ hi
          nlinarith
        · rw [Nat.mul_comm, Nat.mul_add_mod]; exact Nat.mod_eq_of_lt ht
      · -- the invariant after trip `t`
        have himg : ∀ j, j ∈ Buf.img (fun i : Idx [d.MAX - 0] => i.1.val * d.G + t)
            (fun i => decide (i.1.val < d.S)) → j % d.G = t := by
          rintro j ⟨i, -, rfl⟩
          show (i.1.val * d.G + t) % d.G = t
          rw [Nat.mul_comm, Nat.mul_add_mod]; exact Nat.mod_eq_of_lt ht
        refine ⟨fun j ⟨hj, hjt⟩ => ?_, fun x hx => ?_, ?_⟩
        · obtain ⟨hjS, hjd⟩ := split_j hd hj
          by_cases hjt' : j % d.G = t
          · have hjM : j / d.G < d.MAX - 0 := by have := hd.S_MAX; omega
            have := hrd (⟨j / d.G, hjM⟩, ()) (by simpa using hjS)
            refine ⟨_, ?_, hval (⟨j / d.G, hjM⟩, ()) hjS⟩
            have e : ((⟨j / d.G, hjM⟩, ()) : Idx [d.MAX - 0]).1.val * d.G + t = j := by
              show j / d.G * d.G + t = j; omega
            rw [e] at this; exact this
          · obtain ⟨v, hrv, hvv⟩ := hI1 j ⟨hj, by omega⟩
            refine ⟨v, ?_, hvv⟩
            rw [Mem.read_congr (m₂ := m₁) (fun x hx => hfr x ?_)]
            · exact hrv
            · rintro ⟨j', hj', hx'⟩
              have hj'm := himg j' hj'
              have hne : (j : ℤ) ≠ j' := by
                intro h
                have hjj : j = j' := by exact_mod_cast h
                exact hjt' (hjj ▸ hj'm)
              exact Set.disjoint_left.mp (d.out.span_disjoint hne) hx hx'
        · have hx1 : x ∉ d.out.elems (Buf.img (fun i : Idx [d.MAX - 0] => i.1.val * d.G + t)
              (fun i => decide (i.1.val < d.S))) := by
            rintro ⟨j, hj, hxj⟩
            refine hx ⟨j, ⟨?_, by have := himg j hj; omega⟩, hxj⟩
            obtain ⟨i, hi, rfl⟩ := hj
            simp only [decide_eq_true_eq] at hi
            have : (i.1.val + 1) * d.G ≤ d.S * d.G := Nat.mul_le_mul_right _ hi
            show i.1.val * d.G + t < d.S * d.G
            nlinarith
          rw [hfr x hx1]
          refine hI2 x fun ⟨j, ⟨hj, hjt⟩, hxj⟩ => hx ⟨j, ⟨hj, by omega⟩, hxj⟩
        · exact Buf.valid_of_agree hI3 fun x hx => hpres x (hI3.2.2.2 x hx)
    · intro _ m₂ hI
      rw [htc] at hI
      exact safe_ret hI
  rw [htc] at hmain
  refine safe_last ((hmain.mono (fun _ m hI => ⟨fun j hj => hI.1 j ⟨hj, Nat.mod_lt _ hG⟩,
    fun x hx => hI.2.1 x fun hx' => hx (d.out.elems_mono (fun j hj => hj.1) hx')⟩) ?_))
  rintro x (⟨t, -, ⟨j, ⟨hj, -⟩, hx⟩⟩ | hx)
  · exact ⟨j, hj, hx⟩
  · exact absurd hx (Set.notMem_empty x)

end Claims.NumKvSplits
