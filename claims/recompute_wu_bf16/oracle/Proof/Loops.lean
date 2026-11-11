import Claim
import TVA.Proof
import Proof.WTrip
import Proof.Stable

/-!
# The two loops

`uLoop_safe` and `wLoop_safe`: the loop over the column blocks of `u` (lines 109–117) and of `w`
(lines 123–133), then the rest of the program. The invariant after `t` trips is that the inputs are
unchanged, the outputs present, and the cells of the first `t` blocks meet `CellU` (`CellW`);
each trip writes its own tile.
-/

namespace Claims.RecomputeWU

open TVA

/-! ## Cells outside a tile are untouched by its stores -/

/-- A cell of `u` whose element is outside the set `S` of elements is unchanged by a change of the
memory that keeps the bytes of `S` ... outside `S`. -/
theorem cellU_stable {d : Inst} {init m m' : Mem} {S : Set Int} (hfr : ∀ x, x ∉ S → m' x = m x)
    {s0 len t' h c : Nat} (hdis : Disjoint (Mem.span .bf16 (d.u.ptr (vIdx d (s0 + t') h c : Nat)).addr) S)
    (hc : CellU d init m s0 len t' h c) : CellU d init m' s0 len t' h c := by
  obtain ⟨o, ho, hf, hb⟩ := hc
  refine ⟨o, ?_, hf, hb⟩
  unfold bfAt at ho ⊢
  rw [Mem.read_congr (m₂ := m) fun x hx => hfr x (Set.disjoint_left.mp hdis hx)]
  exact ho

/-- A cell of `w` is unchanged by a change of the memory away from its element. -/
theorem cellW_stable {d : Inst} {init m m' : Mem} {S : Set Int} (hfr : ∀ x, x ∉ S → m' x = m x)
    {s0 len t' h c : Nat} (hdis : Disjoint (Mem.span .bf16 (d.w.ptr (wIdx d (s0 + t') h c : Nat)).addr) S)
    (hc : CellW d init m s0 len t' h c) : CellW d init m' s0 len t' h c := by
  obtain ⟨o, ho, hf, hb⟩ := hc
  refine ⟨o, ?_, hf, hb⟩
  unfold bfAt at ho ⊢
  rw [Mem.read_congr (m₂ := m) fun x hx => hfr x (Set.disjoint_left.mp hdis hx)]
  exact ho

/-- A cell of `u` outside the columns of the tile of block `t` is disjoint from the tile. -/
theorem span_disj_tileU {d : Inst} {s0 len it h t i c : Nat} (hh : h < d.HV) (hc : c < d.V)
    (hout : c < t * 64 ∨ (t + 1) * 64 ≤ c) :
    Disjoint (Mem.span .bf16 (d.u.ptr (vIdx d (s0 + (it * 64 + i)) h c : Nat)).addr)
      (tileU d s0 len it h t) := by
  refine Set.disjoint_of_subset_left (d.u.span_sub_elems (S := {vIdx d (s0 + (it * 64 + i)) h c})
    rfl) ?_
  refine d.u.elems_disjoint (Set.disjoint_singleton_left.mpr ?_)
  rintro ⟨i', j', hi', hj', e⟩
  have key : ((s0 + (it * 64 + i)) * d.HV + h) * d.V + c =
      ((s0 + it * 64 + i') * d.HV + h) * d.V + (t * 64 + j') :=
    calc _ = vIdx d (s0 + (it * 64 + i)) h c := rfl
      _ = _ := e
      _ = _ := by unfold eU; ring
  obtain ⟨-, -, h3⟩ := idx_inj (f := s0 + (it * 64 + i)) (f' := s0 + it * 64 + i') hh hh hc hj' key
  have h4 : c = t * 64 + (j' : Nat) := h3
  have := j'.isLt
  omega

/-- A cell of `w` outside the columns of the tile of block `t` is disjoint from the tile. -/
theorem span_disj_tileW {d : Inst} {s0 len it h t i c : Nat} (hh : h < d.HV) (hc : c < d.K)
    (hout : c < t * 64 ∨ (t + 1) * 64 ≤ c) :
    Disjoint (Mem.span .bf16 (d.w.ptr (wIdx d (s0 + (it * 64 + i)) h c : Nat)).addr)
      (tileW d s0 len it h t) := by
  refine Set.disjoint_of_subset_left (d.w.span_sub_elems (S := {wIdx d (s0 + (it * 64 + i)) h c})
    rfl) ?_
  refine d.w.elems_disjoint (Set.disjoint_singleton_left.mpr ?_)
  rintro ⟨i', j', hi', hj', e⟩
  have key : ((s0 + (it * 64 + i)) * d.HV + h) * d.K + c =
      ((s0 + it * 64 + i') * d.HV + h) * d.K + (t * 64 + j') :=
    calc _ = wIdx d (s0 + (it * 64 + i)) h c := rfl
      _ = _ := e
      _ = _ := by unfold eW; ring
  obtain ⟨-, -, h3⟩ := idx_inj (f := s0 + (it * 64 + i)) (f' := s0 + it * 64 + i') hh hh hc hj' key
  have h4 : c = t * 64 + (j' : Nat) := h3
  have := j'.isLt
  omega


/-! ## Invariants -/

theorem u_disj_R {d : Inst} (hd : d.ok) : Disjoint d.u.bytes (Rall d) := by
  unfold Rall
  simp only [Set.disjoint_union_right]
  exact ⟨⟨⟨⟨⟨⟨⟨hd.u_k, hd.u_v⟩, hd.u_beta⟩, hd.u_betab⟩, hd.u_A⟩, hd.u_g⟩, hd.u_cu⟩, hd.u_ci⟩

/-- The `w` buffer shares no byte with the inputs. -/
theorem w_disj_R {d : Inst} (hd : d.ok) : Disjoint d.w.bytes (Rall d) := by
  unfold Rall
  simp only [Set.disjoint_union_right]
  exact ⟨⟨⟨⟨⟨⟨⟨hd.w_k, hd.w_v⟩, hd.w_beta⟩, hd.w_betab⟩, hd.w_A⟩, hd.w_g⟩, hd.w_cu⟩, hd.w_ci⟩

/-- The inputs unchanged and the outputs present survive a change confined to `S`, a set of bytes
of an output buffer. -/
theorem AgPres_stable {d : Inst} {init m m' : Mem} {S : Set Int}
    (hSR : Disjoint S (Rall d)) (hfr : ∀ x, x ∉ S → m' x = m x)
    (hpr : ∀ x, m x ≠ none → m' x ≠ none) (h : Ag d init m ∧ Pres d m) :
    Ag d init m' ∧ Pres d m' := by
  refine ⟨fun x hx => ?_, ?_, ?_⟩
  · rw [hfr x (fun hxS => Set.disjoint_left.mp hSR hxS hx)]; exact h.1 x hx
  · exact Buf.valid_of_agree h.2.1 fun x hx => hpr x (h.2.1.2.2.2 x hx)
  · exact Buf.valid_of_agree h.2.2 fun x hx => hpr x (h.2.2.2.2.2 x hx)

/-- After `t` trips of the `u` loop: the cells of the first `t` column blocks. -/
def UDone (d : Inst) (init : Mem) (s0 len it h t : Nat) (m : Mem) : Prop :=
  ∀ i : Fin 64, rowEn it len i = true → ∀ c, c < min d.V (t * 64) →
    CellU d init m s0 len (it * 64 + i) h c

/-- The `u` loop's invariant. -/
def UInv (d : Inst) (init : Mem) (s0 len it h t : Nat) (m : Mem) : Prop :=
  Ag d init m ∧ Pres d m ∧ UDone d init s0 len it h t m

/-- After `t` trips of the `w` loop, with the `u` cells already done. -/
def WDone (d : Inst) (init : Mem) (s0 len it h t : Nat) (m : Mem) : Prop :=
  ∀ i : Fin 64, rowEn it len i = true → ∀ c, c < min d.K (t * 64) →
    CellW d init m s0 len (it * 64 + i) h c

/-- All the `u` cells. -/
def UFull (d : Inst) (init : Mem) (s0 len it h : Nat) (m : Mem) : Prop :=
  ∀ i : Fin 64, rowEn it len i = true → ∀ c, c < d.V → CellU d init m s0 len (it * 64 + i) h c

/-- The `w` loop's invariant. -/
def WInv (d : Inst) (init : Mem) (s0 len it h t : Nat) (m : Mem) : Prop :=
  Ag d init m ∧ Pres d m ∧ UFull d init s0 len it h m ∧ WDone d init s0 len it h t m

/-- The induction variable of trip `t` is `t`. -/
theorem iv_val {w : Nat} (hw : 2 ≤ w) {t : Nat} (ht : t < 2 ^ (w - 1)) :
    (IntV.ofInt ((IntV.ofInt 0 : IntV true w).toInt + (t : Int) * (IntV.ofInt 1 : IntV true w).toInt) :
      IntV true w).toInt = t := by
  rw [IntV.toInt_ofInt_zero, IntV.toInt_ofInt_one hw]
  rw [IntV.toInt_ofInt (by omega)]
  · simp
  · simp only [IntV.sfits_iff]
    constructor
    · have : (0 : Int) ≤ t := by positivity
      have : (0 : Int) < 2 ^ (w - 1) := by positivity
      linarith
    · simp only [zero_add, mul_one]; exact_mod_cast ht

/-- `range(n)` over `int32` is a covered loop. -/
theorem range_ok_n {n : Nat} (hn : n < 2 ^ 30) :
    tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat n : I32) (IntV.ofInt 1 : I32) := by
  unfold tl.RangeOk
  rw [IntV.toInt_ofInt_zero, IntV.toInt_ofInt_one (by norm_num), IntV.I32.toInt_ofNat (by omega)]
  simp only [IntV.maxInt]
  norm_num
  omega

/-- `range(n)` runs `n` trips. -/
theorem trip_n {n : Nat} (hn : n < 2 ^ 30) :
    tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat n : I32).toInt (IntV.ofInt 1 : I32).toInt = n := by
  rw [IntV.toInt_ofInt_zero, IntV.toInt_ofInt_one (by norm_num), IntV.I32.toInt_ofNat (by omega)]
  unfold tl.tripCount
  simp


/-- A trip's tile of `u` lies in the buffer. -/
theorem tileU_sub_bytes {d : Inst} (hd : d.ok) {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d)
    (hl : it * 64 < len) (hh : h < d.HV) : tileU d s0 len it h t ⊆ d.u.bytes :=
  d.u.tileAt_sub_bytes (Or.inr (Or.inr (lt_of_lt_of_le
    (tile_lt hs (by omega) hh hd.v_pos le_rfl le_rfl) hd.u_len)))

/-- A trip's tile of `u` is among the program's elements. -/
theorem tileU_sub_UE {d : Inst} {s0 len it h t : Nat} :
    tileU d s0 len it h t ⊆ d.u.elems (UE d s0 len it h) := by
  refine d.u.elems_mono ?_
  rintro x ⟨i, j, hi, hj, e⟩
  refine ⟨i, t * 64 + j, i.isLt, hi, hj, ?_⟩
  rw [e]; unfold eU vIdx; ring

/-- One trip of the `u` loop keeps the invariant. -/
theorem uInv_step {E : Env} (hP : Device E.P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) {bos i_h : I64} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool}
    {b_A : Tensor [64, 64] BF16} (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hA : ALane d init s0 len it h b_A)
    {vbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit}
    (hvb : ∀ t, t * 64 < d.V → VbSpec E d init s0 len it h t vbK) {t : Nat}
    (ht : t * 64 < d.V) {i_v : I32} (hiv : i_v.toInt = t) {m : Mem}
    (hI : UInv d init s0 len it h t m) :
    Safe E (Rall d) (tileU d s0 len it h t) (fun _ m' => UInv d init s0 len it h (t + 1) m') m
      (uTrip (d.v.ptr 0) (d.u.ptr 0) d.HV d.V bos i_h o_t m_t b_A vbK i_v) := by
  have tr := uTrip_safe hP hd hpre hs hl hh ht hbos hih ho hmt hA (hvb t ht) hiv hI.1 hI.2.1
  have hSR : Disjoint (tileU d s0 len it h t) (Rall d) :=
    Set.disjoint_of_subset_left (tileU_sub_bytes hd hs hl hh) (u_disj_R hd)
  have tr' := safe_and_inv tr (UInv d init s0 len it h t) (S := tileU d s0 len it h t)
    (fun m₀ m' hfr hpr hI₀ => by
      obtain ⟨hAg, hPr⟩ := AgPres_stable hSR hfr hpr ⟨hI₀.1, hI₀.2.1⟩
      refine ⟨hAg, hPr, fun i hi c hc => ?_⟩
      have hc' : c < d.V := lt_of_lt_of_le hc (min_le_left _ _)
      have hc'' : c < t * 64 := lt_of_lt_of_le hc (min_le_right _ _)
      exact cellU_stable hfr (by
        have := span_disj_tileU (d := d) (s0 := s0) (len := len) (it := it) (h := h) (t := t)
          (i := i) hh hc' (Or.inl hc'')
        convert this using 3)
        (hI₀.2.2 i hi c hc)) hI le_rfl
  refine tr'.mono (fun _ m' ⟨hnew, hinv⟩ => ⟨hinv.1, hinv.2.1, fun i hi c hc => ?_⟩) le_rfl
  by_cases hct : c < t * 64
  · exact hinv.2.2 i hi c (lt_min (lt_of_lt_of_le hc (min_le_left _ _)) hct)
  · have hcV : c < d.V := lt_of_lt_of_le hc (min_le_left _ _)
    have h64 : c - t * 64 < 64 := by
      have := lt_of_lt_of_le hc (min_le_right _ _); omega
    have := hnew i ⟨c - t * 64, h64⟩ hi (by simp only [colEn, decide_eq_true_eq]; omega)
    have e : t * 64 + ((⟨c - t * 64, h64⟩ : Fin 64) : Nat) = c := by
      show t * 64 + (c - t * 64) = c
      omega
    rw [e] at this
    exact this


/-- The tiles of different trips of the `u` loop are disjoint. -/
theorem tileU_disjoint {d : Inst} (hd : d.ok) {s0 len it h t u : Nat} (htu : t < u) :
    Disjoint (tileU d s0 len it h t) (tileU d s0 len it h u) :=
  d.u.tileAt_disjoint (Tile2.inj_rowMajor1 (Nat.le_mul_of_pos_left d.V hd.hv_pos))
    (Or.inr fun j j' e => by
      have e' : t * 64 + (j : Nat) = u * 64 + (j' : Nat) := e
      have := j.isLt; have := j'.isLt
      nlinarith)

/-- **The `u` loop** (lines 109–117) and the rest of the program `k`. -/
theorem uLoop_safe {E : Env} (hP : Device E.P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) {bos i_h : I64} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool}
    {b_A : Tensor [64, 64] BF16} (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hA : ALane d init s0 len it h b_A)
    {vbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit}
    (hvb : ∀ t, t * 64 < d.V → VbSpec E d init s0 len it h t vbK) {Wk : Set Int}
    (hWk : Disjoint (d.u.elems (UE d s0 len it h)) Wk) {k : Unit → Prog Unit}
    {Q : Unit → Mem → Prop}
    (hk : ∀ m₁, UInv d init s0 len it h ((d.V + 63) / 64) m₁ → Safe E (Rall d) Wk Q m₁ (k ()))
    {m : Mem} (hI0 : UInv d init s0 len it h 0 m) :
    Safe E (Rall d) (d.u.elems (UE d s0 len it h) ∪ Wk) Q m
      (tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((d.V + (64 - 1)) / 64) : I32)
        (IntV.ofInt 1 : I32) ()
        (fun i_v _ => uTrip (d.v.ptr 0) (d.u.ptr 0) d.HV d.V bos i_h o_t m_t b_A vbK i_v) >>= k) := by
  have hV31 := hd.V_lt
  have hn : (d.V + (64 - 1)) / 64 < 2 ^ 30 := by omega
  have hrange := range_ok_n hn
  have htc := trip_n hn
  have hiter : ∀ n, iterW (fun t => tileU d s0 len it h t) n ⊆ d.u.elems (UE d s0 len it h) := by
    rintro n x ⟨t, -, hx⟩
    exact tileU_sub_UE hx
  refine (safe_forRange (E := E) (R := Rall d) (Q := Q) (m := m) (k := k) hrange
    (fun t _ m => UInv d init s0 len it h t m) (fun t => tileU d s0 len it h t)
    (Wk := Wk) ?_ ?_ hI0 ?_ ?_).mono (fun _ _ h => h)
    (Set.union_subset_union_left _ (hiter _))
  · intro t u htu _; exact tileU_disjoint hd htu
  · exact Set.disjoint_of_subset_left (hiter _) hWk
  · intro t ht _ m₁ hI
    have ht1 : t < (d.V + (64 - 1)) / 64 := htc ▸ ht
    exact uInv_step hP hd hpre hs hl hh hbos hih ho hmt hA hvb (by omega)
      (iv_val (w := 32) (by norm_num) (by norm_num; omega)) hI
  · intro _ m₁ hI
    exact hk m₁ (by rw [htc] at hI; exact hI)


/-- A trip's tile of `w` lies in the buffer. -/
theorem tileW_sub_bytes {d : Inst} (hd : d.ok) {s0 len it h t : Nat} (hs : s0 + len ≤ nTok d)
    (hl : it * 64 < len) (hh : h < d.HV) : tileW d s0 len it h t ⊆ d.w.bytes :=
  d.w.tileAt_sub_bytes (Or.inr (Or.inr (lt_of_lt_of_le
    (tile_lt hs (by omega) hh hd.k_pos le_rfl le_rfl) hd.w_len)))

/-- A trip's tile of `w` is among the program's elements. -/
theorem tileW_sub_WE {d : Inst} {s0 len it h t : Nat} :
    tileW d s0 len it h t ⊆ d.w.elems (WE d s0 len it h) := by
  refine d.w.elems_mono ?_
  rintro x ⟨i, j, hi, hj, e⟩
  refine ⟨i, t * 64 + j, i.isLt, hi, hj, ?_⟩
  rw [e]; unfold eW wIdx; ring

/-- The tiles of different trips of the `w` loop are disjoint. -/
theorem tileW_disjoint {d : Inst} (hd : d.ok) {s0 len it h t u : Nat} (htu : t < u) :
    Disjoint (tileW d s0 len it h t) (tileW d s0 len it h u) :=
  d.w.tileAt_disjoint (Tile2.inj_rowMajor1 (Nat.le_mul_of_pos_left d.K hd.hv_pos))
    (Or.inr fun j j' e => by
      have e' : t * 64 + (j : Nat) = u * 64 + (j' : Nat) := e
      have := j.isLt; have := j'.isLt
      nlinarith)

/-- A cell of `u` is disjoint from every tile of `w`. -/
theorem span_u_disj_tileW {d : Inst} (hd : d.ok) {s0 len it h t i c : Nat}
    (hs : s0 + len ≤ nTok d) (hl : it * 64 < len) (hrow : it * 64 + i < len) (hh : h < d.HV)
    (hc : c < d.V) :
    Disjoint (Mem.span .bf16 (d.u.ptr (vIdx d (s0 + (it * 64 + i)) h c : Nat)).addr)
      (tileW d s0 len it h t) := by
  have hlt : vIdx d (s0 + (it * 64 + i)) h c < d.u.len := by
    refine lt_of_lt_of_le (idx_lt (n := nTok d) (by omega) hh hc) hd.u_len
  exact Set.disjoint_of_subset_right (tileW_sub_bytes hd hs hl hh)
    (Set.disjoint_of_subset_left (d.u.span_sub hlt) hd.w_u.symm)

/-- One trip of the `w` loop keeps the invariant. -/
theorem wInv_step {E : Env} (hP : Device E.P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) {bos i_h : I64} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool}
    {b_A : Tensor [64, 64] BF16} (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hA : ALane d init s0 len it h b_A)
    {kbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit}
    (hkb : ∀ t, t * 64 < d.K → KbSpec E d init s0 len it h t kbK) {t : Nat}
    (ht : t * 64 < d.K) {i_k : I32} (hik : i_k.toInt = t) {m : Mem}
    (hI : WInv d init s0 len it h t m) :
    Safe E (Rall d) (tileW d s0 len it h t) (fun _ m' => WInv d init s0 len it h (t + 1) m') m
      (wTrip (d.k.ptr 0) (d.w.ptr 0) d.H d.HV d.K bos i_h o_t m_t b_A kbK i_k) := by
  have tr := wTrip_safe hP hd hpre hs hl hh ht hbos hih ho hmt hA (hkb t ht) hik hI.1 hI.2.1
  have hSR : Disjoint (tileW d s0 len it h t) (Rall d) :=
    Set.disjoint_of_subset_left (tileW_sub_bytes hd hs hl hh) (w_disj_R hd)
  have tr' := safe_and_inv tr (WInv d init s0 len it h t) (S := tileW d s0 len it h t)
    (fun m₀ m' hfr hpr hI₀ => by
      obtain ⟨hAg, hPr⟩ := AgPres_stable hSR hfr hpr ⟨hI₀.1, hI₀.2.1⟩
      refine ⟨hAg, hPr, fun i hi c hc => ?_, fun i hi c hc => ?_⟩
      · exact cellU_stable hfr (span_u_disj_tileW hd hs hl (by simpa [rowEn] using hi) hh hc)
          (hI₀.2.2.1 i hi c hc)
      · have hc' : c < d.K := lt_of_lt_of_le hc (min_le_left _ _)
        have hc'' : c < t * 64 := lt_of_lt_of_le hc (min_le_right _ _)
        exact cellW_stable hfr (by
          have := span_disj_tileW (d := d) (s0 := s0) (len := len) (it := it) (h := h) (t := t)
            (i := i) hh hc' (Or.inl hc'')
          convert this using 3)
          (hI₀.2.2.2 i hi c hc)) hI le_rfl
  refine tr'.mono (fun _ m' ⟨hnew, hinv⟩ => ⟨hinv.1, hinv.2.1, hinv.2.2.1, fun i hi c hc => ?_⟩) le_rfl
  by_cases hct : c < t * 64
  · exact hinv.2.2.2 i hi c (lt_min (lt_of_lt_of_le hc (min_le_left _ _)) hct)
  · have hcK : c < d.K := lt_of_lt_of_le hc (min_le_left _ _)
    have h64 : c - t * 64 < 64 := by
      have := lt_of_lt_of_le hc (min_le_right _ _); omega
    have := hnew i ⟨c - t * 64, h64⟩ hi (by simp only [colEn, decide_eq_true_eq]; omega)
    have e : t * 64 + ((⟨c - t * 64, h64⟩ : Fin 64) : Nat) = c := by
      show t * 64 + (c - t * 64) = c
      omega
    rw [e] at this
    exact this

/-- **The `w` loop** (lines 123–133), the last statement of the body. -/
theorem wLoop_safe {E : Env} (hP : Device E.P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {s0 len it h : Nat} (hs : s0 + len ≤ nTok d) (hl : it * 64 < len)
    (hh : h < d.HV) {bos i_h : I64} {o_t : Tensor [64] I64} {m_t : Tensor [64] Bool}
    {b_A : Tensor [64, 64] BF16} (hbos : bos.toInt = s0) (hih : i_h.toInt = h)
    (ho : ∀ r : Fin 64, (o_t.get1 r).toInt = ((it * 64 + r : Nat) : Int))
    (hmt : ∀ r : Fin 64, m_t.get1 r = rowEn it len r) (hA : ALane d init s0 len it h b_A)
    {kbK : Tensor [64, 64] BF16 → (Tensor [64, 64] BF16 → Prog Unit) → Prog Unit}
    (hkb : ∀ t, t * 64 < d.K → KbSpec E d init s0 len it h t kbK) {m : Mem}
    (hI0 : WInv d init s0 len it h 0 m) :
    Safe E (Rall d) (d.w.elems (WE d s0 len it h))
      (fun _ m' => WInv d init s0 len it h ((d.K + 63) / 64) m') m
      (tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat ((d.K + (64 - 1)) / 64) : I32)
        (IntV.ofInt 1 : I32) ()
        (fun i_k _ => wTrip (d.k.ptr 0) (d.w.ptr 0) d.H d.HV d.K bos i_h o_t m_t b_A kbK i_k)) := by
  have hK31 := hd.K_lt
  have hn : (d.K + (64 - 1)) / 64 < 2 ^ 30 := by omega
  have hrange := range_ok_n hn
  have htc := trip_n hn
  have hiter : ∀ n, iterW (fun t => tileW d s0 len it h t) n ⊆ d.w.elems (WE d s0 len it h) := by
    rintro n x ⟨t, -, hx⟩
    exact tileW_sub_WE hx
  refine safe_last ?_
  refine (safe_forRange (E := E) (R := Rall d) (Q := fun _ m' => WInv d init s0 len it h
    ((d.K + 63) / 64) m') (m := m) (k := pure) hrange
    (fun t _ m => WInv d init s0 len it h t m) (fun t => tileW d s0 len it h t)
    (Wk := ∅) ?_ ?_ hI0 ?_ ?_).mono (fun _ _ h => h)
    (by rintro x (hx | hx)
        · exact hiter _ hx
        · exact hx.elim)
  · intro t u htu _; exact tileW_disjoint hd htu
  · exact Set.disjoint_empty _
  · intro t ht _ m₁ hI
    have ht1 : t < (d.K + (64 - 1)) / 64 := htc ▸ ht
    exact wInv_step hP hd hpre hs hl hh hbos hih ho hmt hA hkb (by omega)
      (iv_val (w := 32) (by norm_num) (by norm_num; omega)) hI
  · intro _ m₁ hI
    exact safe_ret (by rw [htc] at hI; exact hI)

end Claims.RecomputeWU
