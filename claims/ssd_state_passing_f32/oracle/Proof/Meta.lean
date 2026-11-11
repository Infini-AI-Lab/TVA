import Claim
import TVA.Proof
namespace Claims.SsdStatePassing
open TVA

/-- The number of requests `1 ≤ r ≤ n` that start at or before token `t`. -/
def cnt (q : Nat → Nat) (n t : Nat) : Nat :=
  ((List.range' 1 n).filter (fun r => decide (q r ≤ t))).length

/-- One more request adds one to the count when it starts at or before the token. -/
theorem cnt_succ (q : Nat → Nat) (n t : Nat) :
    cnt q (n + 1) t = cnt q n t + (if q (n + 1) ≤ t then 1 else 0) := by
  unfold cnt
  rw [List.range'_concat, List.filter_append, List.length_append]
  simp only [one_mul, Nat.add_comm 1 n]
  split_ifs with h <;> simp [h]

/-- The count of requests starting at or before token `t` names the request running at `t`. -/
theorem cnt_spec (q : Nat → Nat) (t : Nat) (h0 : q 0 ≤ t) :
    ∀ n, (∀ r, r < n → q r < q (r + 1)) →
      cnt q n t ≤ n ∧ q (cnt q n t) ≤ t ∧ (cnt q n t < n → t < q (cnt q n t + 1)) := by
  intro n
  induction n with
  | zero => intro _; simp [cnt, h0]
  | succ n ih =>
    intro hq
    obtain ⟨h1, h2, h3⟩ := ih (fun r hr => hq r (by omega))
    rw [cnt_succ]
    by_cases hle : q (n + 1) ≤ t
    · -- all of 1..n start before t
      have hall : cnt q n t = n := by
        by_contra hne
        have hlt : cnt q n t < n := by omega
        have := h3 hlt
        have hmono : q (cnt q n t + 1) ≤ q (n + 1) := by
          have key : ∀ k, k ≤ n → q (k + 1) ≤ q (n + 1) := by
            intro k hk
            induction' h : n - k with j ih2 generalizing k
            · have : k = n := by omega
              subst this; exact le_rfl
            · have := hq (k + 1) (by omega)
              have := ih2 (k + 1) (by omega) (by omega)
              omega
          exact key _ (by omega)
        omega
      rw [ite_eq_left hle, hall]
      exact ⟨le_rfl, hle, fun h => absurd h (by omega)⟩
    · rw [ite_eq_right hle, Nat.add_zero]
      refine ⟨by omega, h2, fun _ => ?_⟩
      by_cases hk : cnt q n t < n
      · exact h3 hk
      · have : cnt q n t = n := by omega
        rw [this]; omega

section Host
variable {d : Inst}

/-- The packed batch's request starts increase up to `nreq`. -/
theorem qsl_mono (hd : d.ok) {a b : Nat} (hab : a ≤ b) (hb : b ≤ d.nreq) : d.qsl a ≤ d.qsl b := by
  induction b, hab using Nat.le_induction with
  | base => exact le_rfl
  | succ b hab ih =>
    have := hd.qsl_lt b (by omega)
    have := ih (by omega)
    omega

/-- The request starts strictly increase up to `nreq`. -/
theorem qsl_strict (hd : d.ok) {a b : Nat} (hab : a < b) (hb : b ≤ d.nreq) : d.qsl a < d.qsl b := by
  have := hd.qsl_lt a (by omega)
  have := qsl_mono hd (show a + 1 ≤ b by omega) hb
  omega

/-- A packed batch has a token. -/
theorem seqlen_pos (hd : d.ok) : 0 < d.seqlen := by
  have := qsl_strict hd (show 0 < d.nreq from hd.nreq_pos) le_rfl
  rw [hd.qsl_zero, hd.qsl_end] at this; exact this

/-- `chunk_size` is positive. -/
theorem chunk_pos (hd : d.ok) : 0 < d.chunk := by
  obtain ⟨k, hk⟩ := hd.chunk_pow2; rw [hk]; positivity

/-- There is a chunk. -/
theorem nchunks_pos (hd : d.ok) : 0 < d.nchunks := by
  have := seqlen_pos hd
  have := chunk_pos hd
  rw [hd.nchunks_eq]
  apply Nat.div_pos <;> omega

/-- `reqOf` is the count of `cnt`. -/
theorem reqOf_eq (t : Nat) : reqOf d t = cnt d.qsl (d.nreq - 1) t := rfl

/-- **The request of token `t`**: `qsl e ≤ t < qsl (e + 1)`. -/
theorem reqOf_spec (hd : d.ok) {t : Nat} (ht : t < d.seqlen) :
    reqOf d t ≤ d.nreq - 1 ∧ d.qsl (reqOf d t) ≤ t ∧ t < d.qsl (reqOf d t + 1) := by
  rw [reqOf_eq]
  obtain ⟨h1, h2, h3⟩ := cnt_spec d.qsl t (by rw [hd.qsl_zero]; omega) (d.nreq - 1)
    (fun r hr => hd.qsl_lt r (by omega))
  refine ⟨h1, h2, ?_⟩
  by_cases hk : cnt d.qsl (d.nreq - 1) t < d.nreq - 1
  · exact h3 hk
  · have : cnt d.qsl (d.nreq - 1) t = d.nreq - 1 := by omega
    rw [this, Nat.sub_add_cancel hd.nreq_pos, hd.qsl_end]; exact ht

/-- The request running at a token does not decrease with the token. -/
theorem reqOf_mono (hd : d.ok) {t t' : Nat} (h : t ≤ t') (ht' : t' < d.seqlen) :
    reqOf d t ≤ reqOf d t' := by
  obtain ⟨a1, a2, a3⟩ := reqOf_spec hd (show t < d.seqlen by omega)
  obtain ⟨b1, b2, b3⟩ := reqOf_spec hd ht'
  by_contra hlt
  have := qsl_mono hd (show reqOf d t' + 1 ≤ reqOf d t by omega) (by omega)
  omega


/-! ### The host's offsets -/

/-- Whether request `r` starts inside a chunk (not at a multiple of `chunk_size`): `1` if so. -/
def nal (d : Inst) (r : Nat) : Nat := if 0 < d.qsl r % d.chunk then 1 else 0

/-- The number of requests `1 ≤ r ≤ n` that start inside a chunk: the host's insertions so far. -/
def pc (d : Inst) (n : Nat) : Nat := ∑ r ∈ Finset.range n, nal d (r + 1)

/-- No insertions before the first request. -/
theorem pc_zero : pc d 0 = 0 := by simp [pc]

/-- One more request adds its insertion. -/
theorem pc_succ (n : Nat) : pc d (n + 1) = pc d n + nal d (n + 1) := by
  simp [pc, Finset.sum_range_succ]

/-- Insertions do not decrease with the requests counted. -/
theorem pc_mono {a b : Nat} (h : a ≤ b) : pc d a ≤ pc d b := by
  induction b, h using Nat.le_induction with
  | base => exact le_rfl
  | succ b _ ih => rw [pc_succ]; omega



/-- On the requests `a < r ≤ b` that all have `nal = v`, `pc` grows by `v` per request. -/
theorem pc_const {a b v : Nat} (hab : a ≤ b) (h : ∀ r, a < r → r ≤ b → nal d r = v) :
    pc d b = pc d a + (b - a) * v := by
  induction b, hab using Nat.le_induction with
  | base => simp
  | succ b hab ih =>
    rw [pc_succ, ih (fun r h1 h2 => h r h1 (by omega)), h (b + 1) (by omega) le_rfl]
    rw [show b + 1 - a = (b - a) + 1 by omega]; ring

/-- The list length of a filter over `1..n` is a sum. -/
theorem length_filter_range' (p : Nat → Bool) (n : Nat) :
    ((List.range' 1 n).filter p).length = ∑ r ∈ Finset.range n, if p (r + 1) then 1 else 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range'_concat, List.filter_append, List.length_append, ih, Finset.sum_range_succ]
    simp only [one_mul, Nat.add_comm 1 n]
    cases hp : p (n + 1) <;> simp [hp]

/-- The host's `N` is the number of physical chunks plus the insertions. -/
theorem hostN_eq (hd : d.ok) : hostN d = d.nchunks + pc d (d.nreq - 1) := by
  unfold hostN pc nal laterReqs
  rw [length_filter_range', ← hd.nchunks_eq]
  congr 1
  refine Finset.sum_congr rfl fun r _ => ?_
  simp only [decide_eq_true_eq, Nat.pos_iff_ne_zero]

/-- The host's loop after the requests `1..n`. -/
def fs (d : Inst) (n : Nat) : Nat × (Nat → Nat) :=
  (List.range' 1 n).foldl (offStep d) (0, fun _ => 0)

/-- The host's loop after `n + 1` requests is one more step. -/
theorem fs_succ (n : Nat) : fs d (n + 1) = offStep d (fs d n) (n + 1) := by
  unfold fs
  rw [List.range'_concat, List.foldl_append]
  simp [Nat.add_comm 1 n]

/-- `chunk_offsets` is the table after all later requests. -/
theorem hostOff_eq : hostOff d = (fs d (d.nreq - 1)).2 := rfl

/-- The host's insertion counter after `n` requests is `pc`. -/
theorem fs_fst (n : Nat) : (fs d n).1 = pc d n := by
  induction n with
  | zero => simp [fs, pc_zero]
  | succ n ih => rw [fs_succ, pc_succ]; simp [offStep, ih, nal]

/-- The slot of request `r` in `chunk_offsets`: its chunk plus the insertions up to it. -/
def idx (d : Inst) (r : Nat) : Nat := d.qsl r / d.chunk + pc d r

/-- The slots of consecutive requests increase. -/
theorem idx_succ_lt (hd : d.ok) {r : Nat} (hr : r + 1 ≤ d.nreq) : idx d r < idx d (r + 1) := by
  have hc := chunk_pos hd
  have hlt := hd.qsl_lt r (by omega)
  have hdiv : d.qsl r / d.chunk ≤ d.qsl (r + 1) / d.chunk := Nat.div_le_div_right hlt.le
  rw [idx, idx, pc_succ]
  by_cases heq : d.qsl r / d.chunk = d.qsl (r + 1) / d.chunk
  · have h1 : 0 < d.qsl (r + 1) % d.chunk := by
      by_contra h0
      have h0' : d.qsl (r + 1) % d.chunk = 0 := by omega
      have e1 := Nat.div_add_mod (d.qsl (r + 1)) d.chunk
      have e3 := Nat.mul_div_le (d.qsl r) d.chunk
      rw [← heq] at e1
      rw [h0'] at e1
      omega
    have : nal d (r + 1) = 1 := by simp [nal, h1]
    omega
  · omega

/-- The slots of requests increase. -/
theorem idx_lt (hd : d.ok) {r r' : Nat} (h : r < r') (hr' : r' ≤ d.nreq) : idx d r < idx d r' := by
  induction r', h using Nat.le_induction with
  | base => exact idx_succ_lt hd hr'
  | succ r' hr ih => exact (ih (by omega)).trans (idx_succ_lt hd hr')

/-- **The host's table**: after the requests `1..n`, the slot of request `r ≤ n` holds `qsl r % chunk`. -/
theorem fs_snd (hd : d.ok) : ∀ n, n + 1 ≤ d.nreq → ∀ r, 1 ≤ r → r ≤ n →
    (fs d n).2 (idx d r) = d.qsl r % d.chunk := by
  intro n
  induction n with
  | zero => intro _ r h1 h2; omega
  | succ n ih =>
    intro hn r h1 h2
    rw [fs_succ]
    by_cases hr : r = n + 1
    · subst hr
      simp [offStep, fs_fst, idx, pc_succ, nal]
    · have hne : idx d r ≠ idx d (n + 1) := (idx_lt hd (show r < n + 1 by omega) (by omega)).ne
      have := ih (by omega) r h1 (by omega)
      have hI : idx d (n + 1) = d.qsl (n + 1) / d.chunk +
          (pc d n + if d.qsl (n + 1) % d.chunk > 0 then 1 else 0) := by
        simp [idx, pc_succ, nal]
      simp only [offStep, fs_fst, Function.update_apply]
      rw [ite_eq_right (fun h => hne (h.trans hI.symm))]
      exact this

/-- The host's table holds, at the slot of request `r`, its start modulo `chunk_size`. -/
theorem hostOff_idx (hd : d.ok) {r : Nat} (h1 : 1 ≤ r) (h2 : r < d.nreq) :
    hostOff d (idx d r) = d.qsl r % d.chunk := by
  rw [hostOff_eq]
  exact fs_snd hd (d.nreq - 1) (by have := hd.nreq_pos; omega) r h1 (by omega)


/-! ### The chunks -/

theorem chunk_start_lt (hd : d.ok) {c : Nat} (hc : c < d.nchunks) : c * d.chunk < d.seqlen := by
  have h := chunk_pos hd
  have h1 := (Nat.le_div_iff_mul_le h).mp (show c + 1 ≤ (d.seqlen + d.chunk - 1) / d.chunk by
    rw [← hd.nchunks_eq]; omega)
  rw [Nat.succ_mul] at h1
  omega

/-- The last token of chunk `c`, in terms of the chunk's bounds. -/
theorem endTok_facts (hd : d.ok) {c : Nat} (hc : c < d.nchunks) :
    c * d.chunk ≤ endTok d c ∧ endTok d c < (c + 1) * d.chunk ∧ endTok d c < d.seqlen := by
  have h := chunk_pos hd
  have h1 := chunk_start_lt hd hc
  unfold endTok
  rw [Nat.succ_mul]
  refine ⟨?_, ?_, ?_⟩ <;> omega

/-- The request running at the end of chunk `c`. -/
theorem eReq_spec (hd : d.ok) {c : Nat} (hc : c < d.nchunks) :
    eReq d c < d.nreq ∧ d.qsl (eReq d c) ≤ endTok d c ∧ endTok d c < d.qsl (eReq d c + 1) := by
  obtain ⟨a, b, c'⟩ := reqOf_spec hd (endTok_facts hd hc).2.2
  have := hd.nreq_pos
  exact ⟨by unfold eReq; omega, b, c'⟩

/-- The request running at the end of the previous chunk (`0` before the first chunk). -/
def prevReq (d : Inst) (c : Nat) : Nat := if c = 0 then 0 else eReq d (c - 1)

/-- The request running at the start of chunk `c`. -/
def sReq (d : Inst) (c : Nat) : Nat := reqOf d (c * d.chunk)

/-- The last token of the chunk before `c` is the one before chunk `c`'s first. -/
theorem endTok_pred (hd : d.ok) {c : Nat} (hc : c < d.nchunks) (h0 : c ≠ 0) :
    endTok d (c - 1) + 1 = c * d.chunk := by
  have := chunk_start_lt hd hc
  unfold endTok
  have e : c - 1 + 1 = c := by omega
  rw [e]
  have : 0 < c * d.chunk := Nat.mul_pos (by omega) (chunk_pos hd)
  rw [Nat.min_eq_left (by omega)]
  omega

/-- Request `0` runs at token `0`. -/
theorem sReq_zero (hd : d.ok) : sReq d 0 = 0 := by
  unfold sReq
  obtain ⟨a, b, c⟩ := reqOf_spec hd (show (0 : Nat) * d.chunk < d.seqlen by
    have := seqlen_pos hd; omega)
  by_contra hne
  have := qsl_strict hd (show 0 < reqOf d (0 * d.chunk) by omega) (by omega)
  rw [hd.qsl_zero] at this
  omega

/-- The request at a chunk's first token: `qsl e ≤ c·chunk_size < qsl (e + 1)`. -/
theorem sReq_spec (hd : d.ok) {c : Nat} (hc : c < d.nchunks) :
    sReq d c ≤ d.nreq - 1 ∧ d.qsl (sReq d c) ≤ c * d.chunk ∧ c * d.chunk < d.qsl (sReq d c + 1) :=
  reqOf_spec hd (chunk_start_lt hd hc)

/-- The request at the chunk's start is at most the one at its end. -/
theorem sReq_le_eReq (hd : d.ok) {c : Nat} (hc : c < d.nchunks) : sReq d c ≤ eReq d c :=
  reqOf_mono hd (endTok_facts hd hc).1 (endTok_facts hd hc).2.2

/-- The request at the end of the previous chunk is at most the one at the chunk's start. -/
theorem prevReq_le_sReq (hd : d.ok) {c : Nat} (hc : c < d.nchunks) : prevReq d c ≤ sReq d c := by
  unfold prevReq
  split_ifs with h0
  · omega
  · unfold eReq sReq
    refine reqOf_mono hd ?_ (chunk_start_lt hd hc)
    have := endTok_pred hd hc h0
    omega

/-- Requests that start exactly at the chunk's first token start at a multiple of `chunk_size`. -/
theorem nal_aligned (hd : d.ok) {c : Nat} (hc : c < d.nchunks) {r : Nat}
    (h1 : prevReq d c < r) (h2 : r ≤ sReq d c) : nal d r = 0 := by
  by_cases h0 : c = 0
  · subst h0
    have := sReq_zero hd
    unfold prevReq at h1
    simp at h1
    omega
  · have hp : prevReq d c = eReq d (c - 1) := by simp [prevReq, h0]
    rw [hp] at h1
    have hcm : c - 1 < d.nchunks := by omega
    obtain ⟨-, -, hlt⟩ := eReq_spec hd hcm
    have hend := endTok_pred hd hc h0
    obtain ⟨hs1, hs2, hs3⟩ := sReq_spec hd hc
    have hq1 : d.qsl (eReq d (c - 1) + 1) ≤ d.qsl r := qsl_mono hd h1 (by omega)
    have hq2 : d.qsl r ≤ d.qsl (sReq d c) := qsl_mono hd h2 (by omega)
    have heq : d.qsl r = c * d.chunk := by omega
    unfold nal
    rw [heq, Nat.mul_mod_left]
    simp

/-- Requests that start strictly inside chunk `c` start inside a chunk. -/
theorem nal_inside (hd : d.ok) {c : Nat} (hc : c < d.nchunks) {r : Nat}
    (h1 : sReq d c < r) (h2 : r ≤ eReq d c) : nal d r = 1 := by
  obtain ⟨-, -, hs3⟩ := sReq_spec hd hc
  obtain ⟨he1, he2, -⟩ := eReq_spec hd hc
  obtain ⟨-, hend, -⟩ := endTok_facts hd hc
  have hcc : (c + 1) * d.chunk = c * d.chunk + d.chunk := by ring
  have hq1 : d.qsl (sReq d c + 1) ≤ d.qsl r := qsl_mono hd h1 (by omega)
  have hq2 : d.qsl r ≤ d.qsl (eReq d c) := qsl_mono hd h2 (by omega)
  have hm : d.qsl r % d.chunk = d.qsl r - c * d.chunk := by
    have hdiv : d.qsl r / d.chunk = c := Nat.div_eq_of_lt_le (by omega) (by omega)
    have := Nat.div_add_mod (d.qsl r) d.chunk
    rw [hdiv, Nat.mul_comm] at this
    omega
  unfold nal
  rw [hm]
  simp
  omega

/-- **The insertions in chunk `c`**: between the request running before it and the one running at
its end, the host inserts one logical chunk per request that starts strictly inside. -/
theorem pc_chunk (hd : d.ok) {c : Nat} (hc : c < d.nchunks) :
    pc d (eReq d c) = pc d (prevReq d c) + (eReq d c - sReq d c) := by
  have h1 := prevReq_le_sReq hd hc
  have h2 := sReq_le_eReq hd hc
  have a := pc_const (v := 0) h1 (fun r hr1 hr2 => nal_aligned hd hc hr1 hr2)
  have b := pc_const (v := 1) h2 (fun r hr1 hr2 => nal_inside hd hc hr1 hr2)
  omega

/-- A chunk where a new request ends: the request starts in the chunk, at a known offset. -/
theorem reset_facts (hd : d.ok) {c : Nat} (hc : c < d.nchunks) (hne : prevReq d c ≠ eReq d c) :
    isReset d c = true ∧ 1 ≤ eReq d c ∧ d.qsl (eReq d c) / d.chunk = c ∧
      offR d c = d.qsl (eReq d c) % d.chunk ∧ c * d.chunk ≤ d.qsl (eReq d c) := by
  have hch := chunk_pos hd
  obtain ⟨he1, he2, -⟩ := eReq_spec hd hc
  obtain ⟨-, hend, -⟩ := endTok_facts hd hc
  have hcc : (c + 1) * d.chunk = c * d.chunk + d.chunk := by ring
  have hge : c * d.chunk ≤ d.qsl (eReq d c) := by
    by_cases h0 : c = 0
    · subst h0; simp
    · have hp : prevReq d c = eReq d (c - 1) := by simp [prevReq, h0]
      rw [hp] at hne
      have hcm : c - 1 < d.nchunks := by omega
      obtain ⟨-, -, hlt⟩ := eReq_spec hd hcm
      have hend' := endTok_pred hd hc h0
      have hle : eReq d (c - 1) ≤ eReq d c := by
        unfold eReq
        exact reqOf_mono hd (by have := endTok_facts hd hc; have := endTok_facts hd hcm; omega)
          (endTok_facts hd hc).2.2
      have := qsl_mono hd (show eReq d (c - 1) + 1 ≤ eReq d c by omega) (by omega)
      omega
  have hdiv : d.qsl (eReq d c) / d.chunk = c := Nat.div_eq_of_lt_le (by omega) (by omega)
  have hm : d.qsl (eReq d c) % d.chunk = d.qsl (eReq d c) - c * d.chunk := by
    have := Nat.div_add_mod (d.qsl (eReq d c)) d.chunk
    rw [hdiv, Nat.mul_comm] at this
    omega
  refine ⟨?_, ?_, hdiv, ?_, hge⟩
  · unfold isReset; simpa using hge
  · by_contra h
    have : eReq d c = 0 := by omega
    have : prevReq d c ≤ eReq d c := le_trans (prevReq_le_sReq hd hc) (sReq_le_eReq hd hc)
    omega
  · unfold offR; omega

/-- A chunk where the request does not change does not reset, except the first chunk of a batch
whose first request ends there (which starts at token `0`: no offset). -/
theorem no_sub_of_eq (hd : d.ok) {c : Nat} (hc : c < d.nchunks) (heq : prevReq d c = eReq d c) :
    hasSub d c = false := by
  unfold hasSub
  by_cases h0 : c = 0
  · subst h0
    have : eReq d 0 = 0 := by unfold prevReq at heq; simpa using heq.symm
    have : offR d 0 = 0 := by unfold offR; rw [this, hd.qsl_zero]; simp
    simp [this]
  · have hp : prevReq d c = eReq d (c - 1) := by simp [prevReq, h0]
    rw [hp] at heq
    have hcm : c - 1 < d.nchunks := by omega
    obtain ⟨-, hq, -⟩ := eReq_spec hd hcm
    have hend' := endTok_pred hd hc h0
    have : ¬ (c * d.chunk ≤ d.qsl (eReq d c)) := by rw [← heq]; omega
    unfold isReset
    simp [this]

/-- A chunk where the request does not change is a reset only for the first chunk of the batch. -/
theorem eq_reset_zero (hd : d.ok) {c : Nat} (hc : c < d.nchunks) (heq : prevReq d c = eReq d c)
    (hr : isReset d c = true) : c = 0 := by
  by_contra h0
  have hp : prevReq d c = eReq d (c - 1) := by simp [prevReq, h0]
  rw [hp] at heq
  have hcm : c - 1 < d.nchunks := by omega
  obtain ⟨-, hq, -⟩ := eReq_spec hd hcm
  have hend' := endTok_pred hd hc h0
  have : ¬ (c * d.chunk ≤ d.qsl (eReq d c)) := by rw [← heq]; omega
  unfold isReset at hr
  simp at hr
  exact this hr

/-- The logical chunk of a reset chunk: the host's slot of the request ending there holds its offset,
and lies inside `chunk_offsets`. -/
theorem logical_slot (hd : d.ok) {c : Nat} (hc : c < d.nchunks) (hne : prevReq d c ≠ eReq d c) :
    hostOff d (c + pc d (eReq d c)) = offR d c ∧ c + pc d (eReq d c) < hostN d := by
  obtain ⟨-, h1, hdiv, hoff, -⟩ := reset_facts hd hc hne
  obtain ⟨he1, -, -⟩ := eReq_spec hd hc
  refine ⟨?_, ?_⟩
  · have := hostOff_idx hd h1 he1
    rw [idx, hdiv] at this
    rw [this, hoff]
  · rw [hostN_eq hd]
    have := pc_mono (d := d) (show eReq d c ≤ d.nreq - 1 by omega)
    omega

/-- The offset of a request's first token in its chunk is below `chunk_size`. -/
theorem offR_lt (hd : d.ok) {c : Nat} (hc : c < d.nchunks) (hne : prevReq d c ≠ eReq d c) :
    offR d c < d.chunk := by
  obtain ⟨-, -, -, hoff, -⟩ := reset_facts hd hc hne
  rw [hoff]; exact Nat.mod_lt _ (chunk_pos hd)

/-- Without a reset the carried state is the reference's: `xRef` and `tolIn` are `sRef` and `tol`. -/
theorem xRef_of_eq (hd : d.ok) {m₀ : Mem} {h m c : Nat} (hc : c < d.nchunks)
    (heq : prevReq d c = eReq d c) :
    xRef m₀ d h m c = sRef m₀ d h m c ∧ tolIn m₀ d h m c = tol m₀ d h m c := by
  by_cases hr : isReset d c = true
  · have h0 := eq_reset_zero hd hc heq hr
    subst h0
    have he : eReq d 0 = 0 := by unfold prevReq at heq; simpa using heq.symm
    simp [xRef, tolIn, hr, sRef, tol, he]
  · simp [xRef, tolIn, hr]

end Host
end Claims.SsdStatePassing
