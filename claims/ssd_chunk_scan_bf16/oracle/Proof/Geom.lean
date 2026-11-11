import Proof.Defs

/-!
# The logical chunks

The combinatorics of the host's metadata (`hostMeta`, `reqOf`): the logical chunks `k < nmeta`
(physical chunk `lc k`, rows `lo k ≤ r < lend k`) partition the tokens `t < seqlen`, every token of
a logical chunk belongs to one request, and the kernel's tests on `seq_idx` pick the state the
reference `fromInit` names.
-/

namespace Claims.SsdChunkScan

open TVA

/-! ## Arithmetic of the chunk size -/

/-- A number below `b` has a chunk below `⌈b / cs⌉`: `c < ⌊b / cs⌋ + [b mod cs > 0]` iff `c · cs < b`. -/
theorem geom_lt_cdiv {cs c b : Nat} (hcs : 0 < cs) :
    c < b / cs + (if b % cs > 0 then 1 else 0) ↔ c * cs < b := by
  have hb := Nat.div_add_mod' b cs
  have hr := Nat.mod_lt b hcs
  generalize b / cs = q at *
  generalize b % cs = m at *
  subst hb
  rcases lt_trichotomy c q with h | h | h
  · have := Nat.mul_le_mul_right cs (show c + 1 ≤ q from h)
    rw [Nat.add_mul, one_mul] at this
    split <;> omega
  · subst h
    split <;> omega
  · have := Nat.mul_le_mul_right cs (show q + 1 ≤ c from h)
    rw [Nat.add_mul, one_mul] at this
    split <;> omega

/-- `⌈b / cs⌉ = ⌊b / cs⌋ + [b mod cs > 0]`. -/
theorem geom_cdiv {cs b : Nat} (hcs : 0 < cs) :
    (b + cs - 1) / cs = b / cs + (if b % cs > 0 then 1 else 0) := by
  have hb := Nat.div_add_mod' b cs
  have hr := Nat.mod_lt b hcs
  generalize b / cs = q at *
  generalize b % cs = m at *
  subst hb
  have e1 : (q + 1) * cs = q * cs + cs := by ring
  have e2 : (q + 1 + 1) * cs = q * cs + cs + cs := by ring
  split
  · apply Nat.div_eq_of_lt_le (k := q + 1) <;> omega
  · rw [show q * cs + m + cs - 1 = q * cs + (cs - 1) by omega]
    apply Nat.div_eq_of_lt_le (k := q) <;> omega

/-! ## The request of a token -/

theorem geom_qsl_lt {d : Inst} (hd : d.ok) {r r' : Nat} (h : r < r') (h' : r' ≤ d.nreq) :
    d.qsl r < d.qsl r' := by
  induction r', h using Nat.le_induction with
  | base => exact hd.qsl_lt r (by omega)
  | succ r' hr ih => exact lt_trans (ih (by omega)) (hd.qsl_lt r' (by omega))

/-- Auxiliary fact: geom qsl le. -/
theorem geom_qsl_le {d : Inst} (hd : d.ok) {r r' : Nat} (h : r ≤ r') (h' : r' ≤ d.nreq) :
    d.qsl r ≤ d.qsl r' := by
  rcases h.eq_or_lt with rfl | h
  · exact le_rfl
  · exact (geom_qsl_lt hd h h').le

/-- Counting the requests `≤ r` among `1, …, m`. -/
theorem geom_count (r m : Nat) :
    ((List.range' 1 m).filter (fun x => decide (x ≤ r))).length = min m r := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [List.range'_concat, Nat.one_mul, List.filter_append, List.length_append, ih]
    by_cases h : 1 + m ≤ r
    · have : (List.filter (fun x => decide (x ≤ r)) [1 + m]).length = 1 := by simp [h]
      omega
    · have : (List.filter (fun x => decide (x ≤ r)) [1 + m]).length = 0 := by simp [h]
      omega

/-- A token's request, from the bounds. -/
theorem geom_reqOf_eq {d : Inst} (hd : d.ok) {t r : Nat} (hr : r < d.nreq) (h1 : d.qsl r ≤ t)
    (h2 : t < d.qsl (r + 1)) : reqOf d t = r := by
  unfold reqOf laterReqs
  have : (List.filter (fun r' => decide (d.qsl r' ≤ t)) (List.range' 1 (d.nreq - 1))) =
      List.filter (fun x => decide (x ≤ r)) (List.range' 1 (d.nreq - 1)) := by
    apply List.filter_congr
    intro x hx
    rw [List.mem_range'_1] at hx
    by_cases hxr : x ≤ r
    · have := geom_qsl_le hd hxr hr.le
      simp only [hxr]; simp; omega
    · have := geom_qsl_le hd (show r + 1 ≤ x by omega) (by omega)
      simp only [hxr]; simp; omega
  rw [this, geom_count]
  omega

/-- Every token lies between the starts of two consecutive requests. -/
theorem geom_exists_req {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    ∃ r, r < d.nreq ∧ d.qsl r ≤ t ∧ t < d.qsl (r + 1) := by
  have : ∀ j, j ≤ d.nreq → t < d.qsl j → ∃ r, r < j ∧ d.qsl r ≤ t ∧ t < d.qsl (r + 1) := by
    intro j
    induction j with
    | zero => intro _ h; rw [hd.qsl_zero] at h; omega
    | succ j ih =>
      intro hj h
      by_cases h' : t < d.qsl j
      · obtain ⟨r, hr, hr'⟩ := ih (by omega) h'
        exact ⟨r, by omega, hr'⟩
      · exact ⟨j, by omega, by omega, h⟩
  obtain ⟨r, hr, h⟩ := this d.nreq le_rfl (by rw [hd.qsl_end]; exact ht)
  exact ⟨r, hr, h⟩

/-- Auxiliary fact: geom cs pos. -/
theorem geom_cs_pos {d : Inst} (hd : d.ok) : 0 < d.cs := by
  obtain ⟨a, ha⟩ := hd.cs_pow2
  rw [ha]; exact Nat.two_pow_pos a

/-! ## The host's loop

`geomIns d r` is the host's `p` after request `r` (the number of requests among `1, …, r` that do
not start on a chunk boundary), and `geomS d r = qsl r / cs + geomIns d r` is the first logical
chunk of request `r` (`_s` in the host's loop); the logical chunks of request `r` are
`geomS d r ≤ k < geomS d (r + 1)`. -/

/-- The host's `p` after request `r`. -/
def geomIns (d : Inst) : Nat → Nat
  | 0 => 0
  | r + 1 => geomIns d r + (if d.qsl (r + 1) % d.cs > 0 then 1 else 0)

/-- The first logical chunk of request `r`. -/
def geomS (d : Inst) (r : Nat) : Nat := d.qsl r / d.cs + geomIns d r

/-- Auxiliary fact: geomS zero. -/
theorem geomS_zero {d : Inst} (hd : d.ok) : geomS d 0 = 0 := by
  simp [geomS, geomIns, hd.qsl_zero]

/-- Auxiliary fact: geomIns le. -/
theorem geomIns_le {d : Inst} (r : Nat) : geomIns d r ≤ geomS d r := by
  exact Nat.le_add_left _ _

/-- Auxiliary fact: geomS succ. -/
theorem geomS_succ (d : Inst) (r : Nat) : geomS d (r + 1) =
    d.qsl (r + 1) / d.cs + geomIns d r + (if d.qsl (r + 1) % d.cs > 0 then 1 else 0) := by
  simp only [geomS, geomIns]; omega

/-- Consecutive requests have consecutive, disjoint, non-empty ranges of logical chunks. -/
theorem geomS_lt {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.nreq) : geomS d r < geomS d (r + 1) := by
  have hcs := geom_cs_pos hd
  have hlt := hd.qsl_lt r hr
  rw [geomS_succ]
  unfold geomS
  have h1 : d.qsl r / d.cs ≤ d.qsl (r + 1) / d.cs := Nat.div_le_div_right hlt.le
  have h2 : d.qsl r / d.cs < d.qsl (r + 1) / d.cs + (if d.qsl (r + 1) % d.cs > 0 then 1 else 0) := by
    rw [geom_lt_cdiv hcs]
    exact lt_of_le_of_lt (Nat.div_mul_le_self _ _) hlt
  omega

/-- Auxiliary fact: geomS lt of lt. -/
theorem geomS_lt_of_lt {d : Inst} (hd : d.ok) {r r' : Nat} (h : r < r') (h' : r' ≤ d.nreq) :
    geomS d r < geomS d r' := by
  induction r', h using Nat.le_induction with
  | base => exact geomS_lt hd (by omega)
  | succ r' hr ih => exact lt_trans (ih (by omega)) (geomS_lt hd (by omega))

/-- Auxiliary fact: geomS le. -/
theorem geomS_le {d : Inst} (hd : d.ok) {r r' : Nat} (h : r ≤ r') (h' : r' ≤ d.nreq) :
    geomS d r ≤ geomS d r' := by
  rcases h.eq_or_lt with rfl | h
  · exact le_rfl
  · exact (geomS_lt_of_lt hd h h').le

/-- Every logical chunk `k < geomS d nreq` belongs to a request. -/
theorem geom_exists_meta_req {d : Inst} (hd : d.ok) {k : Nat} (hk : k < geomS d d.nreq) :
    ∃ r, r < d.nreq ∧ geomS d r ≤ k ∧ k < geomS d (r + 1) := by
  have : ∀ j, j ≤ d.nreq → k < geomS d j → ∃ r, r < j ∧ geomS d r ≤ k ∧ k < geomS d (r + 1) := by
    intro j
    induction j with
    | zero => intro _ h; rw [geomS_zero hd] at h; omega
    | succ j ih =>
      intro hj h
      by_cases h' : k < geomS d j
      · obtain ⟨r, hr, hr'⟩ := ih (by omega) h'
        exact ⟨r, by omega, hr'⟩
      · exact ⟨j, by omega, by omega, h⟩
  obtain ⟨r, hr, h⟩ := this d.nreq le_rfl hk
  exact ⟨r, hr, h⟩

/-- The state of the host's loop after requests `1, …, j`. -/
def geomSt (d : Inst) (j : Nat) : Nat × (Nat → Nat) × (Nat → Nat) :=
  (List.range' 1 j).foldl (metaStep d) (0, fun k => k, fun _ => 0)

/-- Auxiliary fact: geomSt succ. -/
theorem geomSt_succ (d : Inst) (j : Nat) : geomSt d (j + 1) = metaStep d (geomSt d j) (j + 1) := by
  unfold geomSt
  rw [List.range'_concat, List.foldl_append]
  simp [Nat.add_comm]

/-- Auxiliary fact: geom metaStep. -/
theorem geom_metaStep (d : Inst) (st : Nat × (Nat → Nat) × (Nat → Nat)) (j : Nat)
    (h : st.1 = geomIns d j) : metaStep d st (j + 1) =
      (geomIns d (j + 1),
       fun k => if geomS d (j + 1) ≤ k ∧ k < geomS d (j + 1 + 1)
         then st.2.1 k - geomIns d (j + 1) else st.2.1 k,
       Function.update st.2.2 (geomS d (j + 1)) (d.qsl (j + 1) % d.cs)) := by
  have hp : st.1 + (if d.qsl (j + 1) % d.cs > 0 then 1 else 0) = geomIns d (j + 1) := by
    rw [h]; rfl
  have hs : d.qsl (j + 1) / d.cs + geomIns d (j + 1) = geomS d (j + 1) := rfl
  have he : d.qsl (j + 1 + 1) / d.cs + geomIns d (j + 1) +
      (if d.qsl (j + 1 + 1) % d.cs > 0 then 1 else 0) = geomS d (j + 1 + 1) := by
    simp only [geomS, geomIns]; omega
  simp only [metaStep]
  rw [hp, hs, he]

/-- The invariant of the host's loop after `j` later requests: `p`, the indices and offsets of the
logical chunks of requests `0, …, j`, and the untouched tail. -/
def GeomInv (d : Inst) (j : Nat) (st : Nat × (Nat → Nat) × (Nat → Nat)) : Prop :=
  st.1 = geomIns d j ∧
  (∀ r, r ≤ j → ∀ k, geomS d r ≤ k → k < geomS d (r + 1) →
    st.2.1 k = k - geomIns d r ∧ st.2.2 k = if k = geomS d r then d.qsl r % d.cs else 0) ∧
  (∀ k, geomS d (j + 1) ≤ k → st.2.1 k = k ∧ st.2.2 k = 0)

/-- Auxiliary fact: geomSt inv. -/
theorem geomSt_inv {d : Inst} (hd : d.ok) : ∀ j, j < d.nreq → GeomInv d j (geomSt d j) := by
  intro j
  induction j with
  | zero =>
    intro _
    refine ⟨rfl, ?_, fun k _ => ⟨rfl, rfl⟩⟩
    intro r hr k _ _
    obtain rfl : r = 0 := by omega
    simp [geomSt, geomIns, geomS_zero hd, hd.qsl_zero]
  | succ j ih =>
    intro hj
    obtain ⟨h1, h2, h3⟩ := ih (by omega)
    have hlt := geomS_lt hd hj
    rw [GeomInv, geomSt_succ, geom_metaStep d _ j h1]
    refine ⟨rfl, ?_, ?_⟩
    · intro r hr k hk1 hk2
      rcases Nat.lt_or_ge r (j + 1) with hr' | hr'
      · have hle := geomS_le hd (show r + 1 ≤ j + 1 by omega) (by omega)
        obtain ⟨e1, e2⟩ := h2 r (by omega) k hk1 hk2
        have hne : k ≠ geomS d (j + 1) := by omega
        simp only [Function.update_of_ne hne]
        have hc : ¬(geomS d (j + 1) ≤ k ∧ k < geomS d (j + 1 + 1)) := by omega
        simp only [hc, ↓reduceIte]
        exact ⟨e1, e2⟩
      · obtain rfl : r = j + 1 := by omega
        obtain ⟨e1, e2⟩ := h3 k hk1
        simp only [Function.update_apply]
        simp only [hk1, hk2, and_self, ↓reduceIte, e1, e2]
    · intro k hk
      obtain ⟨e1, e2⟩ := h3 k (by omega)
      have hne : k ≠ geomS d (j + 1) := by omega
      simp only [Function.update_of_ne hne]
      have hc : ¬(geomS d (j + 1) ≤ k ∧ k < geomS d (j + 1 + 1)) := by omega
      simp only [hc, ↓reduceIte]
      exact ⟨e1, e2⟩

/-- The host's insertions counted: `p` after the last request. -/
theorem geom_filter_count (d : Inst) (m : Nat) :
    ((List.range' 1 m).filter (fun r => decide (d.qsl r % d.cs ≠ 0))).length = geomIns d m := by
  induction m with
  | zero => simp [geomIns]
  | succ m ih =>
    rw [List.range'_concat, Nat.one_mul, Nat.add_comm 1 m, List.filter_append, List.length_append,
      ih, geomIns]
    by_cases h : d.qsl (m + 1) % d.cs = 0
    · simp [h]
    · simp [h]

/-- The grid's `y` with initial states is `geomS d nreq`. -/
theorem geom_hostN {d : Inst} (hd : d.ok) : hostN d = geomS d d.nreq := by
  have hcs := geom_cs_pos hd
  have hn := hd.nreq_pos
  obtain ⟨m, hm⟩ : ∃ m, d.nreq = m + 1 := ⟨d.nreq - 1, by omega⟩
  unfold hostN laterReqs
  rw [geom_filter_count, hm, Nat.add_sub_cancel, geomS_succ, ← hm, hd.qsl_end, geom_cdiv hcs]
  omega

/-- What the host's loop leaves for the logical chunks of request `r`. -/
theorem geom_hostMeta {d : Inst} (hd : d.ok) {r k : Nat} (hr : r < d.nreq) (h1 : geomS d r ≤ k)
    (h2 : k < geomS d (r + 1)) :
    hostIdx d k = k - geomIns d r ∧
      hostOff d k = if k = geomS d r then d.qsl r % d.cs else 0 := by
  have := (geomSt_inv hd (d.nreq - 1) (by have := hd.nreq_pos; omega)).2.1 r (by omega) k h1 h2
  exact this

/-! ## The logical chunks with initial states -/

/-- **The logical chunk `k` of request `r`** (with initial states): its physical chunk `c = k - p`
meets the request, its first token is the later of the chunk's and the request's first token, and
its end is the earlier of the chunk's end and the request's end. -/
theorem geom_chunk {d : Inst} (hd : d.ok) (hI : d.hasInit = true) {r k : Nat} (hr : r < d.nreq)
    (h1 : geomS d r ≤ k) (h2 : k < geomS d (r + 1)) :
    lc d k = k - geomIns d r ∧ geomIns d r ≤ k ∧ lc d k * d.cs < d.qsl (r + 1) ∧
      d.qsl r < lc d k * d.cs + d.cs ∧
      lc d k * d.cs + lo d k = max (lc d k * d.cs) (d.qsl r) ∧
      lc d k * d.cs + lend d k = min (lc d k * d.cs + d.cs) (d.qsl (r + 1)) := by
  have hcs := geom_cs_pos hd
  obtain ⟨hi, ho⟩ := geom_hostMeta hd hr h1 h2
  have hlc : lc d k = k - geomIns d r := by simp [lc, hI, hi]
  have hlo : lo d k = if k = geomS d r then d.qsl r % d.cs else 0 := by simp [lo, hI, ho]
  have hins : geomIns d r ≤ k := le_trans (geomIns_le r) h1
  have hnm : d.nmeta = geomS d d.nreq := by simp [hd.nmeta_eq, hI, geom_hostN hd]
  have hbT : d.qsl (r + 1) ≤ d.T := by
    rw [← hd.qsl_end]; exact geom_qsl_le hd (by omega) le_rfl
  have hSr : geomS d r = d.qsl r / d.cs + geomIns d r := rfl
  have hSr1 := geomS_succ d r
  obtain ⟨c, hc⟩ : ∃ c, c = k - geomIns d r := ⟨_, rfl⟩
  rw [hlc, ← hc]
  have e1 : (c + 1) * d.cs = c * d.cs + d.cs := by ring
  have hq : c * d.cs < d.qsl (r + 1) := by
    rw [← geom_lt_cdiv hcs]; omega
  have hdiv := Nat.div_add_mod' (d.qsl r) d.cs
  have hmod := Nat.mod_lt (d.qsl r) hcs
  have hd1 : d.qsl r < (d.qsl r / d.cs + 1) * d.cs := (Nat.div_lt_iff_lt_mul hcs).1 (Nat.lt_succ_self _)
  have e2 : (d.qsl r / d.cs + 1) * d.cs = d.qsl r / d.cs * d.cs + d.cs := by ring
  have ha : d.qsl r < c * d.cs + d.cs := by
    have : d.qsl r / d.cs ≤ c := by omega
    have := Nat.mul_le_mul_right d.cs this
    omega
  refine ⟨rfl, hins, hq, ha, ?_, ?_⟩
  · rw [hlo]
    by_cases hk : k = geomS d r
    · have : c = d.qsl r / d.cs := by omega
      subst this
      simp only [hk, ↓reduceIte]
      omega
    · have : d.qsl r / d.cs + 1 ≤ c := by omega
      have := Nat.mul_le_mul_right d.cs this
      simp only [hk, ↓reduceIte]
      omega
  · -- the end row
    have hb := Nat.div_add_mod' (d.qsl (r + 1)) d.cs
    have hbm := Nat.mod_lt (d.qsl (r + 1)) hcs
    have hB : k + 1 = geomS d (r + 1) →
        (d.qsl (r + 1) % d.cs > 0 ∧ c * d.cs + d.qsl (r + 1) % d.cs = d.qsl (r + 1)) ∨
        (d.qsl (r + 1) % d.cs = 0 ∧ d.qsl (r + 1) = c * d.cs + d.cs) := by
      intro hk
      by_cases hm : d.qsl (r + 1) % d.cs > 0
      · left
        have : c = d.qsl (r + 1) / d.cs := by simp only [hm, ↓reduceIte] at hSr1; omega
        subst this
        exact ⟨hm, by omega⟩
      · right
        have : c + 1 = d.qsl (r + 1) / d.cs := by simp only [hm, ↓reduceIte] at hSr1; omega
        rw [← this] at hb
        omega
    have hlend : k + 1 < d.nmeta → k + 1 = geomS d (r + 1) → r + 1 < d.nreq := by
      intro hk hkk
      by_contra hcon
      have : r + 1 = d.nreq := by omega
      rw [this] at hkk
      omega
    unfold lend lm
    rw [hlc, ← hc]
    simp only [hI, true_and]
    by_cases hcond : k + 1 < d.nmeta ∧ hostIdx d (k + 1) = hostIdx d k
    · obtain ⟨hk1, hid⟩ := hcond
      simp only [hk1, hid, and_self, ↓reduceIte]
      rcases Nat.lt_or_ge (k + 1) (geomS d (r + 1)) with hA | hA
      · have := (geom_hostMeta hd hr (by omega) hA).1
        omega
      · have hkk : k + 1 = geomS d (r + 1) := by omega
        have hr1 : r + 1 < d.nreq := hlend hk1 hkk
        have hlt := geomS_lt hd hr1
        obtain ⟨hi', ho'⟩ := geom_hostMeta hd hr1 (by omega) (by omega)
        simp only [hkk, ↓reduceIte] at ho'
        have hins1 : geomIns d (r + 1) = geomIns d r + (if d.qsl (r + 1) % d.cs > 0 then 1 else 0) := rfl
        rcases hB hkk with ⟨hm, hh⟩ | ⟨hm, hh⟩
        · rw [hkk, ho']
          omega
        · have hm' : ¬ d.qsl (r + 1) % d.cs > 0 := by omega
          simp only [hm', ↓reduceIte] at hins1
          omega
    · simp only [hcond, ↓reduceIte]
      rcases Nat.lt_or_ge (k + 1) (geomS d (r + 1)) with hA | hA
      · have hq1 : (c + 1) * d.cs < d.qsl (r + 1) := by
          rw [← geom_lt_cdiv hcs]; omega
        omega
      · have hkk : k + 1 = geomS d (r + 1) := by omega
        rcases hB hkk with ⟨hm, hh⟩ | ⟨hm, hh⟩
        · have hr1 : r + 1 = d.nreq := by
            by_contra hne
            have hr1 : r + 1 < d.nreq := by omega
            have hlt := geomS_lt hd hr1
            have hlt2 := geomS_lt_of_lt hd hr1 le_rfl
            obtain ⟨hi', -⟩ := geom_hostMeta hd hr1 (by omega) (by omega)
            have hins1 : geomIns d (r + 1) = geomIns d r + (if d.qsl (r + 1) % d.cs > 0 then 1 else 0) := rfl
            simp only [hm, ↓reduceIte] at hins1
            exact hcond ⟨by omega, by omega⟩
          rw [hr1, hd.qsl_end] at *
          omega
        · omega

/-- The request of a token. -/
theorem reqOf_spec {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    reqOf d t < d.nreq ∧ d.qsl (reqOf d t) ≤ t ∧ t < d.qsl (reqOf d t + 1) := by
  obtain ⟨r, hr, h1, h2⟩ := geom_exists_req hd ht
  rw [geom_reqOf_eq hd hr h1 h2]
  exact ⟨hr, h1, h2⟩

/-- `qsl (r + 1)` is at most `seqlen`. -/
theorem geom_qsl_succ_le {d : Inst} (hd : d.ok) {r : Nat} (hr : r < d.nreq) :
    d.qsl (r + 1) ≤ d.T := by
  rw [← hd.qsl_end]; exact geom_qsl_le hd (by omega) le_rfl

/-- A chunk is below `nchunks` iff it starts below `seqlen`. -/
theorem geom_lt_nc {d : Inst} (hcs : 0 < d.cs) {c : Nat} : c < nc d ↔ c * d.cs < d.T := by
  unfold nc
  rw [Nat.lt_iff_add_one_le, Nat.le_div_iff_mul_le hcs]
  have : (c + 1) * d.cs = c * d.cs + d.cs := by ring
  omega

/-- Without initial states, logical chunk `k` is physical chunk `k`. -/
theorem lc_noinit {d : Inst} (h : d.hasInit = false) (k : Nat) : lc d k = k := by
  simp [lc, h]

/-- Auxiliary fact: lo noinit. -/
theorem lo_noinit {d : Inst} (h : d.hasInit = false) (k : Nat) : lo d k = 0 := by
  simp [lo, h]

/-- Auxiliary fact: lend noinit. -/
theorem lend_noinit {d : Inst} (h : d.hasInit = false) (k : Nat) :
    lend d k = min d.cs (d.T - k * d.cs) := by
  simp [lend, lm, lc, h]

/-- Without initial states, the grid's `y` is `nchunks`. -/
theorem geom_nmeta_noinit {d : Inst} (hd : d.ok) (h : d.hasInit = false) : d.nmeta = nc d := by
  rw [hd.nmeta_eq, h]; rfl

/-- With initial states, the grid's `y` is `geomS d nreq`. -/
theorem geom_nmeta_init {d : Inst} (hd : d.ok) (h : d.hasInit = true) :
    d.nmeta = geomS d d.nreq := by
  simp [hd.nmeta_eq, h, geom_hostN hd]

/-- With initial states, the logical chunk `k < nmeta` belongs to a request. -/
theorem geom_init_req {d : Inst} (hd : d.ok) (hI : d.hasInit = true) {k : Nat} (hk : k < d.nmeta) :
    ∃ r, r < d.nreq ∧ geomS d r ≤ k ∧ k < geomS d (r + 1) := by
  apply geom_exists_meta_req hd
  rw [geom_nmeta_init hd hI] at hk
  exact hk

/-- **The request of a logical chunk** (with initial states): the tokens of its rows belong to one
request `r`, its physical chunk is `k - p`, and it starts at the request's first token or the
chunk's. -/
theorem geom_row {d : Inst} (hd : d.ok) (hI : d.hasInit = true) {k : Nat} (hk : k < d.nmeta) :
    ∃ r, r < d.nreq ∧ geomIns d r ≤ k ∧ lc d k = k - geomIns d r ∧
      (∀ row, lo d k ≤ row → row < lend d k → reqOf d (lc d k * d.cs + row) = r) ∧
      lc d k * d.cs + lo d k = max (lc d k * d.cs) (d.qsl r) := by
  obtain ⟨r, hr, h1, h2⟩ := geom_init_req hd hI hk
  obtain ⟨hlc, hins, -, -, hlo, hle⟩ := geom_chunk hd hI hr h1 h2
  refine ⟨r, hr, hins, hlc, fun row hrow1 hrow2 => ?_, hlo⟩
  have := geom_qsl_succ_le hd hr
  apply geom_reqOf_eq hd hr <;> omega

/-- A token's logical chunk starts no later than the token. -/
theorem lTok_le {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) : lTok d t ≤ t := by
  obtain ⟨-, h1, -⟩ := reqOf_spec hd ht
  have := Nat.div_mul_le_self t d.cs
  unfold lTok chk
  omega

/-- Tokens of `t`'s chunk before its logical chunk belong to earlier requests. -/
theorem reqOf_ne_of_lt_lTok {d : Inst} (hd : d.ok) {t s : Nat} (ht : t < d.T)
    (h1 : chk d t * d.cs ≤ s) (h2 : s < lTok d t) : reqOf d s ≠ reqOf d t := by
  intro h
  have hs : s < d.T := lt_of_lt_of_le h2 (le_trans (lTok_le hd ht) ht.le)
  obtain ⟨-, hq, -⟩ := reqOf_spec hd hs
  rw [h] at hq
  unfold lTok at h2
  omega

/-- The logical chunks are inside the physical chunks and inside `seqlen`. -/
theorem lc_lt {d : Inst} (hd : d.ok) {k : Nat} (hk : k < d.nmeta) : lc d k < nc d := by
  have hcs := geom_cs_pos hd
  by_cases hI : d.hasInit = true
  · obtain ⟨r, hr, h1, h2⟩ := geom_init_req hd hI hk
    obtain ⟨-, -, hq, -⟩ := geom_chunk hd hI hr h1 h2
    have := geom_qsl_succ_le hd hr
    rw [geom_lt_nc hcs]
    omega
  · have h : d.hasInit = false := by simpa using hI
    rw [lc_noinit h]
    rwa [geom_nmeta_noinit hd h] at hk

/-- Auxiliary fact: lo lt. -/
theorem lo_lt {d : Inst} (hd : d.ok) {k : Nat} (hk : k < d.nmeta) : lo d k < d.cs := by
  have hcs := geom_cs_pos hd
  by_cases hI : d.hasInit = true
  · obtain ⟨r, hr, h1, h2⟩ := geom_init_req hd hI hk
    obtain ⟨-, -, -, ha, hlo, -⟩ := geom_chunk hd hI hr h1 h2
    omega
  · have h : d.hasInit = false := by simpa using hI
    rw [lo_noinit h]
    exact hcs

/-- Auxiliary fact: lend le. -/
theorem lend_le {d : Inst} (hd : d.ok) {k : Nat} (hk : k < d.nmeta) :
    lend d k ≤ d.cs ∧ lc d k * d.cs + lend d k ≤ d.T := by
  have hcs := geom_cs_pos hd
  by_cases hI : d.hasInit = true
  · obtain ⟨r, hr, h1, h2⟩ := geom_init_req hd hI hk
    obtain ⟨-, -, -, -, -, hle⟩ := geom_chunk hd hI hr h1 h2
    have := geom_qsl_succ_le hd hr
    omega
  · have h : d.hasInit = false := by simpa using hI
    have := (geom_lt_nc hcs).1 (by rwa [geom_nmeta_noinit hd h] at hk)
    rw [lend_noinit h, lc_noinit h]
    omega

/-- **Every token is in a logical chunk.** -/
theorem cover {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    ∃ k, k < d.nmeta ∧ lc d k = chk d t ∧ lo d k ≤ t % d.cs ∧ t % d.cs < lend d k := by
  have hcs := geom_cs_pos hd
  have hdm := Nat.div_add_mod' t d.cs
  have hmod := Nat.mod_lt t hcs
  by_cases hI : d.hasInit = true
  · obtain ⟨hr, h1, h2⟩ := reqOf_spec hd ht
    generalize reqOf d t = r at *
    have hS : geomS d r ≤ t / d.cs + geomIns d r := by
      have := Nat.div_le_div_right (c := d.cs) h1
      unfold geomS; omega
    have hS' : t / d.cs + geomIns d r < geomS d (r + 1) := by
      rw [geomS_succ]
      have : t / d.cs < d.qsl (r + 1) / d.cs + (if d.qsl (r + 1) % d.cs > 0 then 1 else 0) := by
        rw [geom_lt_cdiv hcs]
        exact lt_of_le_of_lt (Nat.div_mul_le_self _ _) h2
      omega
    have hk : t / d.cs + geomIns d r < d.nmeta := by
      have := geomS_le hd (show r + 1 ≤ d.nreq by omega) le_rfl
      rw [geom_nmeta_init hd hI]
      omega
    obtain ⟨hlc, -, -, -, hlo, hle⟩ := geom_chunk hd hI hr hS hS'
    have hlc' : lc d (t / d.cs + geomIns d r) = t / d.cs := by rw [hlc, Nat.add_sub_cancel]
    rw [hlc'] at hlo hle
    refine ⟨_, hk, hlc', ?_, ?_⟩
    · generalize t / d.cs = c at *
      omega
    · generalize t / d.cs = c at *
      omega
  · have h : d.hasInit = false := by simpa using hI
    have hk : t / d.cs < d.nmeta := by
      rw [geom_nmeta_noinit hd h, geom_lt_nc hcs]
      have := Nat.div_mul_le_self t d.cs
      omega
    refine ⟨t / d.cs, hk, lc_noinit h _, ?_, ?_⟩
    · rw [lo_noinit h]; exact Nat.zero_le _
    · rw [lend_noinit h]
      have := Nat.div_mul_le_self t d.cs
      omega

/-- **No token is in two logical chunks.** -/
theorem unique {d : Inst} (hd : d.ok) {k k' r : Nat} (hk : k < d.nmeta) (hk' : k' < d.nmeta)
    (hc : lc d k = lc d k') (h1 : lo d k ≤ r ∧ r < lend d k) (h2 : lo d k' ≤ r ∧ r < lend d k') :
    k = k' := by
  by_cases hI : d.hasInit = true
  · obtain ⟨ρ, -, hi, hlc, hreq, -⟩ := geom_row hd hI hk
    obtain ⟨ρ', -, hi', hlc', hreq', -⟩ := geom_row hd hI hk'
    have e := (hreq r h1.1 h1.2).symm.trans (hc ▸ hreq' r h2.1 h2.2)
    subst e
    omega
  · have h : d.hasInit = false := by simpa using hI
    rwa [lc_noinit h, lc_noinit h] at hc

/-- **A row of a logical chunk.** Token `t = lc k · cs + r` with `lo k ≤ r < lend k` lies in
physical chunk `lc k` at row `r`, its logical chunk starts at row `lo k` (with initial states), and
it is below `seqlen`. -/
theorem tok_of_row {d : Inst} (hd : d.ok) {k r : Nat} (hk : k < d.nmeta)
    (hr : lo d k ≤ r ∧ r < lend d k) :
    lc d k * d.cs + r < d.T ∧ chk d (lc d k * d.cs + r) = lc d k ∧
      (lc d k * d.cs + r) % d.cs = r ∧
      (d.hasInit = true → lOff d (lc d k * d.cs + r) = lo d k) := by
  have hcs := geom_cs_pos hd
  obtain ⟨hl1, hl2⟩ := lend_le hd hk
  have hchk : chk d (lc d k * d.cs + r) = lc d k := by
    unfold chk
    apply Nat.div_eq_of_lt_le (k := lc d k)
    · omega
    · have : (lc d k + 1) * d.cs = lc d k * d.cs + d.cs := by ring
      omega
  have hdm := Nat.div_add_mod' (lc d k * d.cs + r) d.cs
  refine ⟨by omega, hchk, by unfold chk at hchk; rw [hchk] at hdm; omega, fun hI => ?_⟩
  obtain ⟨ρ, -, -, -, hreq, hlo⟩ := geom_row hd hI hk
  unfold lOff lTok
  rw [hreq r hr.1 hr.2, hchk]
  omega

/-- All the tokens of a logical chunk belong to one request (with initial states). -/
theorem req_same {d : Inst} (hd : d.ok) (hI : d.hasInit = true) {k r r' : Nat} (hk : k < d.nmeta)
    (hr : lo d k ≤ r ∧ r < lend d k) (hr' : lo d k ≤ r' ∧ r' < lend d k) :
    reqOf d (lc d k * d.cs + r) = reqOf d (lc d k * d.cs + r') := by
  obtain ⟨ρ, -, -, -, hreq, -⟩ := geom_row hd hI hk
  rw [hreq r hr.1 hr.2, hreq r' hr'.1 hr'.2]

/-- Token `t`'s previous chunk ends in another request exactly when the request of `t` is not the
first and starts in `t`'s chunk. -/
theorem geom_prev_ne_iff {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    prevReq d (chk d t) ≠ reqOf d t ↔ 1 ≤ reqOf d t ∧ chk d t * d.cs ≤ d.qsl (reqOf d t) := by
  have hcs := geom_cs_pos hd
  have hle : chk d t * d.cs ≤ t := Nat.div_mul_le_self t d.cs
  obtain ⟨hρ, hρ1, hρ2⟩ := reqOf_spec hd ht
  unfold prevReq
  by_cases hc : 1 ≤ chk d t
  · have hpos : d.cs ≤ chk d t * d.cs := by simpa using Nat.mul_le_mul_right d.cs hc
    have hu : chk d t * d.cs - 1 < d.T := by omega
    obtain ⟨hσ, hσ1, hσ2⟩ := reqOf_spec hd hu
    simp only [hc, ↓reduceIte]
    generalize reqOf d (chk d t * d.cs - 1) = σ at *
    generalize reqOf d t = ρ at *
    constructor
    · intro hne
      have hlt : σ < ρ := by
        by_contra hnot
        have := geom_qsl_le hd (show ρ + 1 ≤ σ by omega) hσ.le
        omega
      have := geom_qsl_le hd (show σ + 1 ≤ ρ by omega) hρ.le
      omega
    · rintro ⟨-, h⟩ rfl
      omega
  · have h0 : chk d t = 0 := by omega
    rw [h0]
    simp
    omega

/-- **With initial states**, the kernel's test (lines 245–251) names `fromInit`: the logical chunk
starts after its chunk's first row, or the token before the chunk belongs to another request. -/
theorem fromInit_init {d : Inst} (hd : d.ok) (h : d.hasInit = true) {k r : Nat}
    (hk : k < d.nmeta) (hr : lo d k ≤ r ∧ r < lend d k) :
    fromInit d (lc d k * d.cs + r) = true ↔
      (0 < lo d k ∨ prevReq d (lc d k) ≠ reqOf d (lc d k * d.cs + r)) := by
  obtain ⟨ht, hchk, -, -⟩ := tok_of_row hd hk hr
  obtain ⟨ρ, hρ, -, -, hreq, hlo⟩ := geom_row hd h hk
  have key := geom_prev_ne_iff hd ht
  rw [hchk] at key
  have hf : fromInit d (lc d k * d.cs + r) = true ↔ 1 ≤ reqOf d (lc d k * d.cs + r) ∧
      lc d k * d.cs ≤ d.qsl (reqOf d (lc d k * d.cs + r)) := by
    unfold fromInit
    rw [hchk]
    simp
  rw [hf, ← key]
  constructor
  · intro hh; exact Or.inr hh
  · rintro (hpos | hne)
    · rw [key, hreq r hr.1 hr.2]
      refine ⟨?_, by omega⟩
      by_contra h0
      obtain rfl : ρ = 0 := by omega
      have := hd.qsl_zero
      omega
    · exact hne

/-- **Without initial states**, a token is `fromInit` exactly when the token before its chunk
belongs to another request (the `tl.where` of line 337). -/
theorem fromInit_noinit {d : Inst} (hd : d.ok) {t : Nat} (ht : t < d.T) :
    fromInit d t = true ↔ prevReq d (chk d t) ≠ reqOf d t := by
  rw [geom_prev_ne_iff hd ht]
  unfold fromInit
  simp

/-- A token not `fromInit` has its logical chunk start at its chunk's first row. -/
theorem lOff_of_not_fromInit {d : Inst} (hd : d.ok) {t : Nat} (_ht : t < d.T)
    (h : fromInit d t = false) : lOff d t = 0 := by
  have h' : ¬(1 ≤ reqOf d t ∧ chk d t * d.cs ≤ d.qsl (reqOf d t)) := by
    unfold fromInit at h
    simpa using h
  have := hd.qsl_zero
  unfold lOff lTok
  by_cases h0 : reqOf d t = 0
  · rw [h0]; omega
  · omega

/-- The tokens whose weights the kernel's loop computes: from row `lo k` up to the token. -/
theorem kToks_eq {d : Inst} (hd : d.ok) {k r : Nat} (hk : k < d.nmeta)
    (hr : lo d k ≤ r ∧ r < lend d k) :
    kToks d (lc d k * d.cs + r) = Finset.Ico (lc d k * d.cs + lo d k) (lc d k * d.cs + r + 1) := by
  obtain ⟨-, hchk, -, hlOff⟩ := tok_of_row hd hk hr
  unfold kToks
  by_cases hI : d.hasInit = true
  · have hl := hlOff hI
    have hle : chk d (lc d k * d.cs + r) * d.cs ≤ lTok d (lc d k * d.cs + r) := by
      unfold lTok; exact le_max_left _ _
    unfold toks lOff at *
    rw [hchk] at hle hl
    simp only [hI, ↓reduceIte]
    congr 1
    omega
  · have h : d.hasInit = false := by simpa using hI
    simp only [h, Bool.false_eq_true, ↓reduceIte, hchk, lo_noinit h, Nat.add_zero]

/-- `seq_idx[t] = reqOf t` for every token (`MetaOk`), and a request number is below `2^31`. -/
theorem reqOf_lt {d : Inst} (hd : d.ok) (t : Nat) : reqOf d t < 2 ^ 31 := by
  have h1 := hd.nreq_batch
  have h2 : (List.filter (fun r => decide (d.qsl r ≤ t)) (laterReqs d)).length ≤ d.nreq - 1 := by
    refine le_trans (List.length_filter_le _ _) ?_
    simp [laterReqs]
  unfold reqOf
  omega

/-- The metadata of the launch with initial states, as `int32` values below `2^31`: the physical
chunk and row of every logical chunk. -/
theorem meta_lt {d : Inst} (hd : d.ok) {k : Nat} (hk : k < d.nmeta) :
    lc d k < 2 ^ 31 ∧ lo d k < 2 ^ 31 := by
  have h1 := lc_lt hd hk
  have h2 := lo_lt hd hk
  have h3 := hd.cs_le
  have h4 := hd.T_ctx
  have h5 : nc d ≤ d.T + d.cs := by
    unfold nc
    exact le_trans (Nat.div_le_self _ _) (by have := geom_cs_pos hd; omega)
  constructor <;> omega

/-- The next logical chunk's entry (line 272–276, 289–293): the physical chunk is `-1` past the
end; otherwise it is the host's `chunk_indices[k + 1]`. And the kernel's cut of the limit. -/
theorem lend_eq_init {d : Inst} (_hd : d.ok) (h : d.hasInit = true) {k : Nat} (_hk : k < d.nmeta) :
    (k + 1 < d.nmeta ∧ hostIdx d (k + 1) = hostIdx d k → lend d k = min (hostOff d (k + 1)) (lm d k)) ∧
    (¬ (k + 1 < d.nmeta ∧ hostIdx d (k + 1) = hostIdx d k) → lend d k = lm d k) := by
  constructor
  · intro hc
    simp [lend, h, hc]
  · intro hc
    simp only [lend, h, true_and]
    simp only [hc, ↓reduceIte]

end Claims.SsdChunkScan
