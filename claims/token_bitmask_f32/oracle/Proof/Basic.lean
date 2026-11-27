import Claim
import TVA.Proof

/-!
# `token_bitmask_f32`: shared definitions

The work items, the cells a launch sets to `-∞` (a processed row `k` and a token `c`, at element
`cell k c` of the logits), the footprints of a program, its post-condition and its loop invariant,
and the facts about rows and cells the other files use.
-/

namespace Claims.TokenBitmask

open TVA

/-! ## Items and cells -/

/-- The blocks of a row: `cdiv(vocab_size, 4096)`. -/
def nb (d : Inst) : Nat := (d.vocab_size + 4095) / 4096

/-- The work item of row `k`, token `c`. -/
def itemOf (d : Inst) (k c : Nat) : Nat := k * nb d + c / 4096

/-- Row `k`, token `c` is set to `-∞`: a processed row, a token below `vocab_size`, bit `0`. -/
def Mk (d : Inst) (init : Mem) (k c : Nat) : Prop :=
  k < d.num_rows ∧ c < d.vocab_size ∧ bitOf d init (rowOf d init k) c = false

/-- The logits element of row `k`, token `c`. -/
def cell (d : Inst) (init : Mem) (k c : Nat) : Int :=
  rowOf d init k * (d.logits_strides : Int) + (c : Int)

/-- The logits elements work item `w` writes. -/
def itemElems (d : Inst) (init : Mem) (w : Nat) : Set Nat :=
  {e | ∃ k c, Mk d init k c ∧ itemOf d k c = w ∧ (e : Int) = cell d init k c}

/-- The logits elements program `p` writes: those of its items `p, p + NUM_SMS, …`. -/
def progElems (d : Inst) (init : Mem) (p : Nat) : Set Nat :=
  {e | ∃ k c, Mk d init k c ∧ itemOf d k c % d.NUM_SMS = p ∧ (e : Int) = cell d init k c}

/-! ## Footprints, post-condition, invariant -/

/-- The read footprint of every program: the bitmask and the indices. -/
def R (d : Inst) : Set Int := d.bitmask.bytes ∪ (if d.hasIdx then d.indices.bytes else ∅)

/-- The write footprint of program `p`. -/
def W (d : Inst) (init : Mem) (p : Nat) : Set Int := d.logits.elems (progElems d init p)

/-- What program `p` leaves: each of its cells holds `-∞`. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ k c, Mk d init k c → itemOf d k c % d.NUM_SMS = p →
    m.read .f32 (d.logits.ptr (cell d init k c)).addr = some (FloatV.negInf : F32)

/-- The loop invariant of program `p` after `t` trips: the cells of its items below
`p + t * NUM_SMS` hold `-∞`, every byte outside the logits is as at the launch, and the logits are
still valid. -/
def Inv (d : Inst) (init : Mem) (p t : Nat) (m : Mem) : Prop :=
  (∀ k c, Mk d init k c → itemOf d k c % d.NUM_SMS = p → itemOf d k c < p + t * d.NUM_SMS →
    m.read .f32 (d.logits.ptr (cell d init k c)).addr = some (FloatV.negInf : F32)) ∧
  (∀ z, z ∉ d.logits.bytes → m z = init z) ∧ d.logits.Valid m

/-! ## The launch -/

/-- The grid has `grid` programs. -/
theorem launch_size (d : Inst) : (launch d).size = d.grid := Launch.size_grid1 _ _ _

/-- Program `i` of the grid has `program_id(0) = i`. -/
theorem launch_pid (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = (i : Nat) := by
  have hi : (i : Nat) < d.grid := lt_of_lt_of_eq i.isLt (launch_size d)
  simp [Launch.ctx, launch, Launch.grid1, Nat.mod_eq_of_lt hi]

/-- The launch is valid: fewer than `2^31` programs, the source's block shapes. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine Launch.valid_grid1 (by have := hd.grid_le; have := hd.sms_lt; omega) ?_
  intro s hs
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl
  · exact ⟨by intro k hk; simp at hk; subst hk; exact ⟨12, by norm_num⟩, by norm_num⟩
  · exact ⟨by intro k hk; simp at hk; subst hk; exact ⟨7, by norm_num⟩, by norm_num⟩
  · exact ⟨by intro k hk; simp at hk; subst hk; exact ⟨5, by norm_num⟩, by norm_num⟩
  · refine ⟨?_, by norm_num⟩
    intro k hk
    simp at hk
    rcases hk with rfl | rfl
    · exact ⟨7, by norm_num⟩
    · exact ⟨5, by norm_num⟩

/-! ## Rows -/

/-- A processed row is a row of both tensors. -/
theorem rowOf_bounds {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {k : Nat}
    (hk : k < d.num_rows) :
    0 ≤ rowOf d init k ∧ rowOf d init k < d.logits_rows ∧ rowOf d init k < d.bitmask_rows := by
  unfold rowOf
  cases h : d.hasIdx
  · obtain ⟨h1, h2⟩ := hd.rows_le h
    simp only [Bool.false_eq_true, ↓reduceIte]
    refine ⟨by positivity, by exact_mod_cast (by omega : k < d.logits_rows),
      by exact_mod_cast (by omega : k < d.bitmask_rows)⟩
  · simp only [↓reduceIte]
    exact hpre.2.2.2.1 h k hk

/-- Distinct processed rows are distinct rows. -/
theorem rowOf_inj {d : Inst} {init : Mem} (hpre : Pre d init) {k k' : Nat}
    (hk : k < d.num_rows) (hk' : k' < d.num_rows) (h : rowOf d init k = rowOf d init k') :
    k = k' := by
  unfold rowOf at h
  cases hI : d.hasIdx
  · simp only [hI, Bool.false_eq_true, ↓reduceIte] at h
    exact_mod_cast h
  · simp only [hI, ↓reduceIte] at h
    exact hpre.2.2.2.2 hI k k' hk hk' h

/-! ## Cells -/

/-- `r * s + c = r' * s + c'` with `0 ≤ c, c' < s` and `0 ≤ r, r'` gives `r = r'`, `c = c'`. -/
theorem rowcol_inj {r r' s c c' : Int} (hc0 : 0 ≤ c) (hc : c < s) (hc0' : 0 ≤ c') (hc' : c' < s)
    (h : r * s + c = r' * s + c') : r = r' ∧ c = c' := by
  have hs : 0 < s := lt_of_le_of_lt hc0 hc
  have h1 : (r * s + c) / s = r := by
    rw [add_comm, Int.add_mul_ediv_right _ _ (ne_of_gt hs), Int.ediv_eq_zero_of_lt hc0 hc, zero_add]
  have h2 : (r' * s + c') / s = r' := by
    rw [add_comm, Int.add_mul_ediv_right _ _ (ne_of_gt hs), Int.ediv_eq_zero_of_lt hc0' hc',
      zero_add]
  have hr : r = r' := by rw [← h1, ← h2, h]
  subst hr
  exact ⟨rfl, by linarith⟩

/-- Two cells set to `-∞` at the same element are the same row and token. -/
theorem cell_inj {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {k c k' c' : Nat}
    (h : Mk d init k c) (h' : Mk d init k' c') (he : cell d init k c = cell d init k' c') :
    k = k' ∧ c = c' := by
  have hc : (c : Int) < d.logits_strides := by
    have := hd.vocab_ls; have := h.2.1; exact_mod_cast (by omega : c < d.logits_strides)
  have hc' : (c' : Int) < d.logits_strides := by
    have := hd.vocab_ls; have := h'.2.1; exact_mod_cast (by omega : c' < d.logits_strides)
  obtain ⟨hr, hcc⟩ := rowcol_inj (by positivity) hc (by positivity) hc' he
  exact ⟨rowOf_inj hpre h.1 h'.1 hr, by exact_mod_cast hcc⟩

/-- A cell set to `-∞` is an element of the logits, at a natural index below `len`. -/
theorem cell_bounds {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {k c : Nat}
    (h : Mk d init k c) : 0 ≤ cell d init k c ∧ cell d init k c < d.logits.len := by
  obtain ⟨h0, h1, -⟩ := rowOf_bounds hd hpre h.1
  have hc : (c : Int) < d.logits_strides := by
    have := hd.vocab_ls; have := h.2.1; exact_mod_cast (by omega : c < d.logits_strides)
  have hlen : ((d.logits_rows * d.logits_strides : Nat) : Int) ≤ d.logits.len := by
    exact_mod_cast hd.logits_len
  push_cast at hlen
  unfold cell
  constructor
  · positivity
  · have : rowOf d init k + 1 ≤ d.logits_rows := h1
    nlinarith

/-- An element in `itemElems` or `progElems` is below `len`. -/
theorem progElems_lt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat} :
    ∀ e ∈ progElems d init p, e < d.logits.len := by
  rintro e ⟨k, c, h, -, he⟩
  have := (cell_bounds hd hpre h).2
  rw [← he] at this
  exact_mod_cast this

/-! ## Items -/

/-- A token below `vocab_size` is in one of the row's blocks. -/
theorem blk_lt {d : Inst} {c : Nat} (hc : c < d.vocab_size) : c / 4096 < nb d := by
  unfold nb; omega

/-- The item of a processed row and token is an item of the launch. -/
theorem itemOf_lt {d : Inst} {k c : Nat} (hk : k < d.num_rows) (hc : c < d.vocab_size) :
    itemOf d k c < items d := by
  have hb := blk_lt hc
  unfold itemOf items
  have : k * nb d + nb d ≤ d.num_rows * nb d := by
    rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hk
  unfold nb at *
  omega

/-- `a * n + b = w` with `b < n` gives `a = w / n` and `b = w % n`. -/
theorem divmod_of_eq {a b n w : Nat} (hb : b < n) (h : a * n + b = w) : a = w / n ∧ b = w % n := by
  have hpos : 0 < n := by omega
  subst h
  refine ⟨?_, ?_⟩
  · rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hb, Nat.zero_add]
  · rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hb]

/-- An item of program `p` (`w % S = p`) at or past trip `t` and before trip `t + 1` is trip `t`'s
item `p + t * S`. -/
theorem item_eq_of_mod {w p t S : Nat} (hmod : w % S = p) (hge : p + t * S ≤ w)
    (hlt : w < p + (t + 1) * S) : w = p + t * S := by
  have hdm := Nat.div_add_mod w S
  rw [hmod] at hdm
  have h1 : t * S ≤ (w / S) * S := by rw [Nat.mul_comm (w / S)]; omega
  have h2 : (w / S) * S < (t + 1) * S := by rw [Nat.mul_comm (w / S)]; rw [Nat.add_mul] at hlt ⊢; omega
  have hS : 0 < S := by
    rcases Nat.eq_zero_or_pos S with h | h
    · subst h; simp at hlt hge; omega
    · exact h
  have h3 : t ≤ w / S := Nat.le_of_mul_le_mul_right h1 hS
  have h4 : w / S < t + 1 := Nat.lt_of_mul_lt_mul_right h2
  have h5 : w / S = t := by omega
  rw [h5] at hdm
  rw [Nat.mul_comm] at hdm
  omega

/-- An item determines its row and block. -/
theorem itemOf_eq {d : Inst} {k c w : Nat} (hc : c < d.vocab_size) (h : itemOf d k c = w) :
    k = w / nb d ∧ c / 4096 = w % nb d := by
  exact divmod_of_eq (blk_lt hc) h

/-- Distinct items have no cell in common. -/
theorem itemElems_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {w w' : Nat}
    (hw : w ≠ w') : Disjoint (itemElems d init w) (itemElems d init w') := by
  rw [Set.disjoint_left]
  rintro e ⟨k, c, h, hi, he⟩ ⟨k', c', h', hi', he'⟩
  obtain ⟨rfl, rfl⟩ := cell_inj hd hpre h h' (he.symm.trans he')
  exact hw (hi.symm.trans hi')

/-- Distinct programs have no cell in common. -/
theorem progElems_disjoint {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p q : Nat}
    (hpq : p ≠ q) : Disjoint (progElems d init p) (progElems d init q) := by
  rw [Set.disjoint_left]
  rintro e ⟨k, c, h, hi, he⟩ ⟨k', c', h', hi', he'⟩
  obtain ⟨rfl, rfl⟩ := cell_inj hd hpre h h' (he.symm.trans he')
  exact hpq (hi.symm.trans hi')

end Claims.TokenBitmask
