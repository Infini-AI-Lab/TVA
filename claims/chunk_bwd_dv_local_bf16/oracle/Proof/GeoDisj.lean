import Claim
import TVA.Proof
import Proof.Geo
import Proof.GeoBound

/-!
# The footprints of the programs are disjoint

The element `(tok, hv, e)` of `dv` determines the token, the value head and the column; the token
determines the sequence (the sequences' token intervals are disjoint), then the chunk, and then
the program `(r, bh)`. So distinct programs write distinct elements of `dv`. A program reads only
the input buffers and the tables, which share no byte with the fresh `dv`.
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The element index determines the token, the value head and the column. -/
theorem oIdx_inj {d : Inst} {t t' h h' e e' : Nat} (hh : h < d.HV) (hh' : h' < d.HV)
    (he : e < d.V) (he' : e' < d.V) (hx : oIdx d t h e = oIdx d t' h' e') :
    t = t' ∧ h = h' ∧ e = e' := by
  have hV : 0 < d.V := by omega
  have hH : 0 < d.HV := by omega
  unfold oIdx at hx
  have h1 : ((t * d.HV + h) * d.V + e) % d.V = e := by
    rw [Nat.mul_comm, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt he]
  have h2 : ((t' * d.HV + h') * d.V + e') % d.V = e' := by
    rw [Nat.mul_comm, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt he']
  have hee : e = e' := by rw [← h1, ← h2, hx]
  subst hee
  have h3 : (t * d.HV + h) = (t' * d.HV + h') := Nat.eq_of_mul_eq_mul_right hV (by omega)
  have h4 : (t * d.HV + h) % d.HV = h := by
    rw [Nat.mul_comm, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hh]
  have h5 : (t' * d.HV + h') % d.HV = h' := by
    rw [Nat.mul_comm, Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt hh']
  have hhh : h = h' := by rw [← h4, ← h5, h3]
  subst hhh
  refine ⟨Nat.eq_of_mul_eq_mul_right hH (by omega), rfl, rfl⟩

/-- The sequence of a program is determined by its lane's flat token. -/
theorem nOf_eq_of_pos_eq {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {r bh r' bh' l l' : Nat} (hr : r < nChunks d) (hbh : bh < nRows d) (hr' : r' < nChunks d)
    (hbh' : bh' < nRows d) (hei : en d r bh l = true) (hej : en d r' bh' l' = true)
    (h : pos d r bh l = pos d r' bh' l') : nOf d r bh = nOf d r' bh' := by
  have hni : nOf d r bh < nSeq d := nOf_lt hd hT hr hbh
  have hnj : nOf d r' bh' < nSeq d := nOf_lt hd hT hr' hbh'
  have hli := lo_le_pos d r bh l
  have hlj := lo_le_pos d r' bh' l'
  have hhi : pos d r bh l < hi d (nOf d r bh) := pos_lt_hi hT hni hei
  have hhj : pos d r' bh' l' < hi d (nOf d r' bh') := pos_lt_hi hT hnj hej
  cases hvv : d.varlen
  · simp only [lo_false hvv, hi_false hvv, nOf_false hvv] at hli hhi hlj hhj ⊢
    by_contra hne
    rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
    · have h1 : (bOf d bh + 1) * d.T ≤ bOf d bh' * d.T :=
        Nat.mul_le_mul_right _ (by omega)
      have h2 : (bOf d bh + 1) * d.T = bOf d bh * d.T + d.T := by ring
      omega
    · have h1 : (bOf d bh' + 1) * d.T ≤ bOf d bh * d.T :=
        Nat.mul_le_mul_right _ (by omega)
      have h2 : (bOf d bh' + 1) * d.T = bOf d bh' * d.T + d.T := by ring
      omega
  · by_contra hne
    have hNi : nOf d r bh < d.N := by simpa [nSeq_true hvv] using hni
    have hNj : nOf d r' bh' < d.N := by simpa [nSeq_true hvv] using hnj
    rw [hi_true hvv (nOf d r bh)] at hhi
    rw [hi_true hvv (nOf d r' bh')] at hhj
    rw [lo_true hvv (nOf d r bh)] at hli
    rw [lo_true hvv (nOf d r' bh')] at hlj
    rcases Nat.lt_or_gt_of_ne hne with hlt | hgt
    · have hm : nOf d r bh + 1 ≤ nOf d r' bh' := by omega
      have hmono := cuAt_mono_le (hT hvv) hm (by omega : nOf d r' bh' ≤ d.N)
      have hle : (cuAt d (nOf d r bh + 1)).toNat ≤ (cuAt d (nOf d r' bh')).toNat :=
        Int.toNat_le_toNat hmono
      omega
    · have hm : nOf d r' bh' + 1 ≤ nOf d r bh := by omega
      have hmono := cuAt_mono_le (hT hvv) hm (by omega : nOf d r bh ≤ d.N)
      have hle : (cuAt d (nOf d r' bh' + 1)).toNat ≤ (cuAt d (nOf d r bh)).toNat :=
        Int.toNat_le_toNat hmono
      omega

/-- The chunk of a program is determined by its lane's flat token and its sequence. -/
theorem chkOf_eq_of_pos_eq {d : Inst} (hd : d.ok) {r bh r' bh' l l' : Nat}
    (hn : nOf d r bh = nOf d r' bh') (h : pos d r bh l = pos d r' bh' l') (hl : l < d.BT)
    (hl' : l' < d.BT) : chkOf d r bh = chkOf d r' bh' := by
  have hBT := bt_pos hd
  have h1 : chkOf d r bh * d.BT + l = chkOf d r' bh' * d.BT + l' := by
    unfold pos baseOf at h
    rw [hn] at h
    omega
  have h2 : (chkOf d r bh * d.BT + l) / d.BT = chkOf d r bh := by
    rw [Nat.mul_comm, Nat.mul_add_div hBT, Nat.div_eq_of_lt hl, Nat.add_zero]
  have h3 : (chkOf d r' bh' * d.BT + l') / d.BT = chkOf d r' bh' := by
    rw [Nat.mul_comm, Nat.mul_add_div hBT, Nat.div_eq_of_lt hl', Nat.add_zero]
  rw [h1, h3] at h2
  exact h2.symm

/-- **Two enabled lanes touching the same element of `dv` belong to the same program.** -/
theorem same_prog {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {r bh r' bh' l l' e e' : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    (hr' : r' < nChunks d) (hbh' : bh' < nRows d) (hei : en d r bh l = true)
    (hej : en d r' bh' l' = true) (hl : l < d.BT) (hl' : l' < d.BT) (he : e < d.V)
    (he' : e' < d.V) (h : wEl d r bh l e = wEl d r' bh' l' e') : r = r' ∧ bh = bh' := by
  obtain ⟨hpos, hhv, -⟩ := oIdx_inj (hvOf_lt hd bh) (hvOf_lt hd bh') he he' h
  have hn : nOf d r bh = nOf d r' bh' := nOf_eq_of_pos_eq hd hT hr hbh hr' hbh' hei hej hpos
  have hc : chkOf d r bh = chkOf d r' bh' := chkOf_eq_of_pos_eq hd hn hpos hl hl'
  cases hvv : d.varlen
  · rw [chkOf_false hvv, chkOf_false hvv] at hc
    rw [nOf_false hvv, nOf_false hvv] at hn
    refine ⟨hc, ?_⟩
    have := Nat.div_add_mod bh d.HV
    have := Nat.div_add_mod bh' d.HV
    unfold bOf at hn
    unfold hvOf at hhv
    rw [← Nat.div_add_mod bh d.HV, ← Nat.div_add_mod bh' d.HV, hn, hhv]
  · have hTab := hT hvv
    have hri : r < d.NT := row_lt_NT hvv hr
    have hrj : r' < d.NT := row_lt_NT hvv hr'
    obtain ⟨-, -, -, hseqInfo, hinj, -⟩ := hTab
    have hni : 0 ≤ seqAt d r := (hseqInfo r hri).1
    have hnj : 0 ≤ seqAt d r' := (hseqInfo r' hrj).1
    have hci : 0 ≤ chkAt d r := (hseqInfo r hri).2.2.1
    have hcj : 0 ≤ chkAt d r' := (hseqInfo r' hrj).2.2.1
    rw [nOf_true hvv, nOf_true hvv] at hn
    rw [chkOf_true hvv, chkOf_true hvv] at hc
    have hseqeq : seqAt d r = seqAt d r' := by
      have hcast : ((seqAt d r).toNat : Int) = ((seqAt d r').toNat : Int) := by rw [hn]
      rwa [Int.toNat_of_nonneg hni, Int.toNat_of_nonneg hnj] at hcast
    have hchkeq : chkAt d r = chkAt d r' := by
      have hcast : ((chkAt d r).toNat : Int) = ((chkAt d r').toNat : Int) := by rw [hc]
      rwa [Int.toNat_of_nonneg hci, Int.toNat_of_nonneg hcj] at hcast
    have hrr : r = r' := by
      by_contra hne
      exact hinj r r' hri hrj hne (by rw [hseqeq, hchkeq])
    refine ⟨hrr, ?_⟩
    have hb : bOf d bh = 0 := bOf_eq_zero hd hvv hbh
    have hb' : bOf d bh' = 0 := bOf_eq_zero hd hvv hbh'
    unfold bOf at hb hb'
    unfold hvOf at hhv
    rw [← Nat.div_add_mod bh d.HV, ← Nat.div_add_mod bh' d.HV, hb, hb', hhv]

/-- Distinct programs write disjoint bytes. -/
theorem W_disjoint {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {r bh r' bh' : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    (hr' : r' < nChunks d) (hbh' : bh' < nRows d) (hne : r ≠ r' ∨ bh ≠ bh') :
    Disjoint (W d r bh) (W d r' bh') := by
  refine Buf.elems_disjoint d.dv ?_
  rw [Set.disjoint_left]
  rintro x ⟨l, e, hl, hen, he, rfl⟩ ⟨l', e', hl', hen', he', h⟩
  obtain ⟨h1, h2⟩ := same_prog hd hT hr hbh hr' hbh' hen hen' hl hl' he he' h
  rcases hne with hne | hne
  · exact hne h1
  · exact hne h2

/-- A program's reads are disjoint from `dv`'s bytes: `dv` is a fresh allocation. -/
theorem R_disjoint_dv {d : Inst} (hd : d.ok) (r bh : Nat) : Disjoint (R d r bh) d.dv.bytes := by
  unfold R
  refine Disjoint.union_left (Disjoint.union_left (Disjoint.union_left
    (Disjoint.union_left hd.dv_q.symm hd.dv_k.symm) hd.dv_dO.symm) ?_) ?_
  · split
    · rename_i hg; exact (hd.dv_g hg).symm
    · exact Set.empty_disjoint _
  · split
    · rename_i hv; exact Disjoint.union_left (hd.dv_cu hv).symm (hd.dv_ci hv).symm
    · exact Set.empty_disjoint _

/-- No program writes a byte any program reads. -/
theorem W_disjoint_R {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {r bh r' bh' : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d) : Disjoint (W d r bh) (R d r' bh') :=
  (Disjoint.mono_left (W_subset_dvBytes hd hT hr hbh) (R_disjoint_dv hd r' bh').symm)

/-- A program's writes are disjoint from what any other program reads or writes. -/
theorem W_disjoint_all {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d)
    {r bh r' bh' : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    (hr' : r' < nChunks d) (hbh' : bh' < nRows d) (hne : r ≠ r' ∨ bh ≠ bh') :
    Disjoint (W d r bh) (R d r' bh' ∪ W d r' bh') :=
  Disjoint.union_right (W_disjoint_R hd hT hr hbh) (W_disjoint hd hT hr hbh hr' hbh' hne)

end Claims.ChunkBwdDvLocal
