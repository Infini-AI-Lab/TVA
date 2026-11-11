import Proof.Common
import Proof.Meta

/-!
# From the programs' local results to `Post`
-/

namespace Claims.RotaryVarlen

open TVA

/-! ## Arithmetic and bounds -/

private theorem varlen_false {d : Inst} (h : d.isVarlen ≠ true) : d.isVarlen = false := by
  cases h' : d.isVarlen <;> simp_all

/-- A sequence's length is at most the launch's tokens. -/
private theorem lenAt_le_T {d : Inst} (_hd : d.ok) {init : Mem} (hpre : Pre d init) {n : Nat}
    (hn : n < nSeq d) : lenAt d init n ≤ d.T := by
  by_cases hv : d.isVarlen = true
  · have hs := hpre.seqs hv
    have hnN : n < d.N := by simpa [nSeq, hv] using hn
    have h1 : cuN d init n ≤ cuN d init (n + 1) := hs.cu_mono n hnN
    have h2 : cuN d init (n + 1) ≤ cuN d init d.N := cuN_mono_le hs (by omega) le_rfl
    have h3 : cuN d init d.N = d.T := hs.cu_last
    have h4 : lenAt d init n = seqLen d init n := by simp [lenAt, hv]
    rw [h4]
    simp only [seqLen]
    omega
  · have hv2 : d.isVarlen = false := varlen_false hv
    simp [lenAt, hv2]

/-- Token `l` of sequence `n` sits at a table position inside the tables. -/
private theorem pos_lt_TR {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {n l : Nat}
    (hn : n < nSeq d) (hl : l < lenAt d init n) :
    (l : Int) + offAt d init n < (d.TR : Int) := by
  by_cases ho : d.offT = true
  · obtain ⟨-, hoff⟩ := hpre.offs_vals ho
    obtain ⟨v, hv, -, hle⟩ := hoff n hn
    have hoffn : offAt d init n = v := by simp [offAt, ho, hv]
    rw [hoffn]
    have hli : (l : Int) < (lenAt d init n : Int) := by exact_mod_cast hl
    omega
  · have ho2 : d.offT = false := by cases h : d.offT <;> simp_all
    have hle := hd.off_tr ho2
    have hlen := lenAt_le_T hd hpre hn
    have hoffn : offAt d init n = (d.off : Int) := by simp [offAt, ho2]
    rw [hoffn]
    have hli : (l : Int) < (lenAt d init n : Int) := by exact_mod_cast hl
    have hle' : (d.off : Int) + (d.T : Int) ≤ (d.TR : Int) := by exact_mod_cast hle
    omega

private theorem div_lt_g0 {d : Inst} (hd : d.ok) (hv : d.isVarlen = false) {l : Nat}
    (hl : l < d.T) : l / d.BT < g0 d := by
  have hb := bt_pos hd
  have hg : g0 d = (d.T + d.BT - 1) / d.BT := by simp [g0, hv]
  rw [hg, Nat.div_lt_iff_lt_mul hb]
  have hm : (d.T + d.BT - 1) % d.BT < d.BT := Nat.mod_lt _ hb
  have hdm : d.BT * ((d.T + d.BT - 1) / d.BT) + (d.T + d.BT - 1) % d.BT = d.T + d.BT - 1 :=
    Nat.div_add_mod _ _
  have hmul : (d.T + d.BT - 1) / d.BT * d.BT = d.BT * ((d.T + d.BT - 1) / d.BT) :=
    Nat.mul_comm _ _
  have hTpos : 0 < d.T := lt_of_le_of_lt (Nat.zero_le _) hl
  omega

/-! ## The program that works on a token -/

/-- Padded path: for token `l` of batch entry `n` and head `h`, a program whose lane `l % BT` is
that token. -/
private theorem padded_program {d : Inst} (hd : d.ok) (hv : d.isVarlen = false)
    {init : Mem} (hpre : Pre d init) {n l h : Nat}
    (hn : n < nSeq d) (hl : l < lenAt d init n) (hh : h < d.H)
    (hp : 0 ≤ (l : Int) + offAt d init n) :
    ∃ i : Fin (launch d).size,
      laneOn d init i (l % d.BT) ∧
      tokI d init i (l % d.BT) = bosAt d init n + l ∧
      ax2 d i = h ∧
      posI d init i (l % d.BT) = (l : Int) + offAt d init n := by
  have hnB : n < d.B := by simpa [nSeq, hv] using hn
  have hlT : l < d.T := by simpa [lenAt, hv] using hl
  have hq : l / d.BT < g0 d := div_lt_g0 hd hv hlT
  have hgpos : 0 < g0 d := lt_of_le_of_lt (Nat.zero_le _) hq
  have hBpos : 0 < d.B := by omega
  set i := (h * d.B + n) * g0 d + l / d.BT with hi_def
  have hsum : h * d.B + n + 1 ≤ d.H * d.B := by
    have hb1 : n + 1 ≤ d.B := by omega
    have hb2 : h * d.B + d.B = (h + 1) * d.B := by rw [Nat.add_mul, Nat.one_mul]
    have hb3 : (h + 1) * d.B ≤ d.H * d.B := Nat.mul_le_mul_right _ (by omega)
    have hb4 : h * d.B + n + 1 ≤ h * d.B + d.B := by omega
    omega
  have hai : i < (launch d).size := by
    have h1 : i < (h * d.B + n + 1) * g0 d := by
      have e : (h * d.B + n + 1) * g0 d = (h * d.B + n) * g0 d + g0 d := by
        rw [Nat.add_mul, Nat.one_mul]
      rw [e]
      omega
    have h2 : (h * d.B + n + 1) * g0 d ≤ (d.H * d.B) * g0 d := Nat.mul_le_mul_right _ hsum
    have h3 : (launch d).size = g0 d * d.B * d.H := launch_size d
    have h4 : (d.H * d.B) * g0 d = g0 d * d.B * d.H := by ring
    omega
  have hax0 : ax0 d i = l / d.BT := by
    have e : i = l / d.BT + g0 d * (h * d.B + n) := by
      rw [hi_def, Nat.add_comm]
      congr 1
      ring
    simp only [ax0]
    rw [e, Nat.add_mul_mod_self_left (l / d.BT) (g0 d) (h * d.B + n), Nat.mod_eq_of_lt hq]
  have hdiv : i / g0 d = h * d.B + n := by
    have e : i = l / d.BT + g0 d * (h * d.B + n) := by
      rw [hi_def, Nat.add_comm]
      congr 1
      ring
    rw [e, Nat.add_mul_div_left (l / d.BT) (h * d.B + n) hgpos, Nat.div_eq_of_lt hq,
      Nat.zero_add]
  have hax1 : ax1 d i = n := by
    simp only [ax1]
    rw [hdiv]
    have e : h * d.B + n = n + d.B * h := by
      rw [Nat.mul_comm d.B h]
      ring
    rw [e, Nat.add_mul_mod_self_left n d.B h, Nat.mod_eq_of_lt hnB]
  have hax2 : ax2 d i = h := by
    have e : i = (n * g0 d + l / d.BT) + (g0 d * d.B) * h := by
      rw [hi_def]
      ring
    have hlt : n * g0 d + l / d.BT < g0 d * d.B := by
      have h1 : n * g0 d + l / d.BT < n * g0 d + g0 d := by omega
      have h2 : n * g0 d + g0 d = (n + 1) * g0 d := by rw [Nat.add_mul, Nat.one_mul]
      have h3 : (n + 1) * g0 d ≤ d.B * g0 d := Nat.mul_le_mul_right _ (by omega)
      have h4 : d.B * g0 d = g0 d * d.B := Nat.mul_comm _ _
      omega
    simp only [ax2]
    rw [e, Nat.add_mul_div_left (n * g0 d + l / d.BT) h (Nat.mul_pos hgpos hBpos),
      Nat.div_eq_of_lt hlt, Nat.zero_add]
  have hchunk : chunkI d init i = l / d.BT := by rw [chunkI_false hv, hax0]
  have hseq : seqI d init i = n := by rw [seqI_false hv, hax1]
  have hloc : locI d init i (l % d.BT) = l := by
    simp only [locI, hchunk]
    have h := Nat.div_add_mod l d.BT
    have hm : (l / d.BT) * d.BT = d.BT * (l / d.BT) := Nat.mul_comm _ _
    omega
  have hlenI : lenI d init i = d.T := by simp [lenI, lenAt, hv]
  have hbosI : bosI d init i = n * d.T := by simp [bosI, bosAt, hv, hseq]
  have hoffI : offI d init i = offAt d init n := by simp [offI, hseq]
  have htok : tokI d init i (l % d.BT) = bosAt d init n + l := by
    have hba : bosAt d init n = n * d.T := by simp [bosAt, hv]
    simp only [tokI, hbosI, hloc, hba]
  have hpos : posI d init i (l % d.BT) = (l : Int) + offAt d init n := by
    simp only [posI, hloc, hoffI]
  have hlane : laneOn d init i (l % d.BT) := by
    refine ⟨?_, ?_, ?_⟩
    · rw [hloc, hlenI]
      exact_mod_cast hlT
    · rw [hpos]
      exact hp
    · rw [hpos]
      exact pos_lt_TR hd hpre hn hl
  exact ⟨⟨i, hai⟩, hlane, htok, hax2, hpos⟩

/-- Packed path: for token `l` of sequence `n` and head `h`, a program whose lane `l % BT` is
that token. -/
private theorem packed_program {d : Inst} (hd : d.ok) (hv : d.isVarlen = true)
    {init : Mem} (hpre : Pre d init) {n l h : Nat}
    (hn : n < nSeq d) (hl : l < lenAt d init n) (hh : h < d.H)
    (hp : 0 ≤ (l : Int) + offAt d init n) :
    ∃ i : Fin (launch d).size,
      laneOn d init i (l % d.BT) ∧
      tokI d init i (l % d.BT) = bosAt d init n + l ∧
      ax2 d i = h ∧
      posI d init i (l % d.BT) = (l : Int) + offAt d init n := by
  have hs := hpre.seqs hv
  have hnN : n < d.N := by simpa [nSeq, hv] using hn
  have hlS : l < seqLen d init n := by simpa [lenAt, hv] using hl
  obtain ⟨i0, hi0, hseq0, hchunk0⟩ :=
    hs.ci_all n (l / d.BT) hnN (lt_of_le_of_lt (Nat.div_mul_le_self l d.BT) hlS)
  have hB1 : d.B = 1 := hd.varlen_b hv
  have hg0 : g0 d = d.NT := by simp [g0, hv]
  have hi0g : i0 < g0 d := by rw [hg0]; exact hi0
  have hgpos : 0 < g0 d := lt_of_le_of_lt (Nat.zero_le _) hi0g
  have hBpos : 0 < d.B := by omega
  set i := h * (g0 d * d.B) + i0 with hi_def
  have hai : i < (launch d).size := by
    have h1 : i < (h + 1) * (g0 d * d.B) := by
      have e : (h + 1) * (g0 d * d.B) = h * (g0 d * d.B) + g0 d * d.B := by
        rw [Nat.add_mul, Nat.one_mul]
      rw [e]
      have hlt : i0 < g0 d * d.B := by
        have hgb : g0 d * d.B = g0 d := by rw [hB1, Nat.mul_one]
        omega
      omega
    have h2 : (h + 1) * (g0 d * d.B) ≤ d.H * (g0 d * d.B) := Nat.mul_le_mul_right _ (by omega)
    have h3 : (launch d).size = g0 d * d.B * d.H := launch_size d
    have h4 : d.H * (g0 d * d.B) = g0 d * d.B * d.H := by ring
    omega
  have hax0 : ax0 d i = i0 := by
    have e : i = i0 + g0 d * (h * d.B) := by
      rw [hi_def]
      ring
    simp only [ax0]
    rw [e, Nat.add_mul_mod_self_left i0 (g0 d) (h * d.B), Nat.mod_eq_of_lt hi0g]
  have hdiv : i / g0 d = h * d.B := by
    have e : i = i0 + g0 d * (h * d.B) := by
      rw [hi_def]
      ring
    rw [e, Nat.add_mul_div_left i0 (h * d.B) hgpos, Nat.div_eq_of_lt hi0g, Nat.zero_add]
  have hax1 : ax1 d i = 0 := by
    simp only [ax1]
    rw [hdiv, hB1, Nat.mul_one, Nat.mod_one]
  have hax2 : ax2 d i = h := by
    have e : i = i0 + (g0 d * d.B) * h := by
      rw [hi_def]
      ring
    have hlt : i0 < g0 d * d.B := by
      have hgb : g0 d * d.B = g0 d := by rw [hB1, Nat.mul_one]
      omega
    simp only [ax2]
    rw [e, Nat.add_mul_div_left i0 h (Nat.mul_pos hgpos hBpos), Nat.div_eq_of_lt hlt,
      Nat.zero_add]
  have hseq : seqI d init i = n := by
    simp only [seqI, hv, ite_true, hax0]
    exact hseq0
  have hchunk : chunkI d init i = l / d.BT := by
    simp only [chunkI, hv, ite_true, hax0]
    exact hchunk0
  have hloc : locI d init i (l % d.BT) = l := by
    simp only [locI, hchunk]
    have h := Nat.div_add_mod l d.BT
    have hm : (l / d.BT) * d.BT = d.BT * (l / d.BT) := Nat.mul_comm _ _
    omega
  have hlenI : lenI d init i = seqLen d init n := by simp [lenI, lenAt, hv, hseq]
  have hbosI : bosI d init i = cuN d init n := by simp [bosI, bosAt, hv, hseq]
  have hoffI : offI d init i = offAt d init n := by simp [offI, hseq]
  have htok : tokI d init i (l % d.BT) = bosAt d init n + l := by
    have hba : bosAt d init n = cuN d init n := by simp [bosAt, hv]
    simp only [tokI, hbosI, hloc, hba]
  have hpos : posI d init i (l % d.BT) = (l : Int) + offAt d init n := by
    simp only [posI, hloc, hoffI]
  have hlane : laneOn d init i (l % d.BT) := by
    refine ⟨?_, ?_, ?_⟩
    · rw [hloc, hlenI]
      exact_mod_cast hlS
    · rw [hpos]
      exact hp
    · rw [hpos]
      exact pos_lt_TR hd hpre hn hl
  exact ⟨⟨i, hai⟩, hlane, htok, hax2, hpos⟩

/-- Every token, head, and channel is handled by a program at a lane of the right chunk. -/
private theorem exists_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {n l h : Nat} (hn : n < nSeq d) (hl : l < lenAt d init n) (hh : h < d.H)
    (hp : 0 ≤ (l : Int) + offAt d init n) :
    ∃ i : Fin (launch d).size,
      laneOn d init i (l % d.BT) ∧
      tokI d init i (l % d.BT) = bosAt d init n + l ∧
      ax2 d i = h ∧
      posI d init i (l % d.BT) = (l : Int) + offAt d init n := by
  by_cases hv : d.isVarlen = true
  · exact packed_program hd hv hpre hn hl hh hp
  · exact padded_program hd (varlen_false hv) hpre hn hl hh hp

/-! ## Injectivity of `tokIdx` -/

private theorem tokIdx_inj {d : Inst} {t h c t' h' c' : Nat}
    (hc : c < d.D) (hc' : c' < d.D) (hh : h < d.H) (hh' : h' < d.H)
    (he : tokIdx d t h c = tokIdx d t' h' c') : t = t' ∧ h = h' ∧ c = c' := by
  simp only [tokIdx] at he
  have hD : 0 < d.D := by omega
  have hH : 0 < d.H := by omega
  have hc_eq : c = c' := by
    have e1 : ((t * d.H + h) * d.D + c) % d.D = c := by
      rw [Nat.add_comm, Nat.mul_comm (t * d.H + h) d.D, Nat.add_mul_mod_self_left,
        Nat.mod_eq_of_lt hc]
    have e2 : ((t' * d.H + h') * d.D + c') % d.D = c' := by
      rw [Nat.add_comm, Nat.mul_comm (t' * d.H + h') d.D, Nat.add_mul_mod_self_left,
        Nat.mod_eq_of_lt hc']
    rw [← e1, he, e2]
  have hmul : t * d.H + h = t' * d.H + h' := by
    rw [hc_eq] at he
    exact Nat.mul_right_cancel hD (Nat.add_right_cancel he)
  have hh_eq : h = h' := by
    have e1 : (t * d.H + h) % d.H = h := by
      rw [Nat.add_comm, Nat.mul_comm t d.H, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hh]
    have e2 : (t' * d.H + h') % d.H = h' := by
      rw [Nat.add_comm, Nat.mul_comm t' d.H, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hh']
    rw [← e1, hmul, e2]
  have ht : t = t' := by
    rw [hh_eq] at hmul
    exact Nat.mul_right_cancel hH (Nat.add_right_cancel hmul)
  exact ⟨ht, hh_eq, hc_eq⟩

/-! ## The negative-position case -/

/-- A lane whose flat token is token `l` of sequence `n` has that token's position. -/
private theorem lane_data {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {n l : Nat} (hn : n < nSeq d) (hl : l < lenAt d init n)
    {i : Nat} (hi : i < (launch d).size) {l' : Nat}
    (hlane : laneOn d init i l') (htok : tokI d init i l' = bosAt d init n + l) :
    seqI d init i = n ∧ locI d init i l' = l := by
  by_cases hv : d.isVarlen = true
  · have hs := hpre.seqs hv
    have hnN : n < d.N := by simpa [nSeq, hv] using hn
    have hlS : l < seqLen d init n := by simpa [lenAt, hv] using hl
    have hsN : seqI d init i < d.N := by
      have := seqI_lt hd hpre hi
      simpa [nSeq, hv] using this
    have hlocS : locI d init i l' < seqLen d init (seqI d init i) := by
      have h1 := hlane.1
      have hlen : lenI d init i = seqLen d init (seqI d init i) := by
        simp [lenI, lenAt, hv]
      rw [hlen] at h1
      exact Int.ofNat_lt.mp h1
    have htok' : cuN d init (seqI d init i) + locI d init i l' = cuN d init n + l := by
      simpa [tokI, bosI, bosAt, hv] using htok
    have hmono_s : cuN d init (seqI d init i) ≤ cuN d init (seqI d init i + 1) :=
      hs.cu_mono _ hsN
    have hmono_n : cuN d init n ≤ cuN d init (n + 1) := hs.cu_mono n hnN
    have hseq : seqI d init i = n := by
      rcases Nat.lt_trichotomy (seqI d init i) n with h | h | h
      · exfalso
        have h1 : cuN d init (seqI d init i + 1) ≤ cuN d init n :=
          cuN_mono_le hs (by omega) (by omega)
        have h2 : seqLen d init (seqI d init i) =
            cuN d init (seqI d init i + 1) - cuN d init (seqI d init i) := rfl
        omega
      · exact h
      · exfalso
        have h1 : cuN d init (n + 1) ≤ cuN d init (seqI d init i) :=
          cuN_mono_le hs (by omega) (by omega)
        have h2 : seqLen d init n = cuN d init (n + 1) - cuN d init n := rfl
        omega
    refine ⟨hseq, ?_⟩
    rw [hseq] at htok'
    omega
  · have hv2 : d.isVarlen = false := varlen_false hv
    have hnB : n < d.B := by simpa [nSeq, hv2] using hn
    have hlT : l < d.T := by simpa [lenAt, hv2] using hl
    have hlocT : locI d init i l' < d.T := by
      have h1 := hlane.1
      have hlen : lenI d init i = d.T := by simp [lenI, lenAt, hv2]
      rw [hlen] at h1
      exact Int.ofNat_lt.mp h1
    have htok' : seqI d init i * d.T + locI d init i l' = n * d.T + l := by
      simpa [tokI, bosI, bosAt, hv2, seqI_false hv2] using htok
    have hseq : seqI d init i = n := by
      rcases Nat.lt_trichotomy (seqI d init i) n with h | h | h
      · exfalso
        have h1 : (seqI d init i + 1) * d.T ≤ n * d.T := Nat.mul_le_mul_right _ (by omega)
        have h2 : (seqI d init i + 1) * d.T = seqI d init i * d.T + d.T := by
          rw [Nat.add_mul, Nat.one_mul]
        omega
      · exact h
      · exfalso
        have h1 : (n + 1) * d.T ≤ seqI d init i * d.T := Nat.mul_le_mul_right _ (by omega)
        have h2 : (n + 1) * d.T = n * d.T + d.T := by rw [Nat.add_mul, Nat.one_mul]
        omega
    refine ⟨hseq, ?_⟩
    rw [hseq] at htok'
    omega

/-- A token at a negative position is on no lane of any program. -/
private theorem no_lane_neg {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {n l : Nat} (hn : n < nSeq d) (hl : l < lenAt d init n)
    (hp : (l : Int) + offAt d init n < 0)
    {i : Nat} (hi : i < (launch d).size) {l' : Nat}
    (hlane : laneOn d init i l') (htok : tokI d init i l' = bosAt d init n + l) : False := by
  obtain ⟨hseq, hloc⟩ := lane_data hd hpre hn hl hi hlane htok
  obtain ⟨-, hpos0, -⟩ := hlane
  have hpos : posI d init i l' = (l : Int) + offAt d init n := by
    simp only [posI, hloc, offI, hseq]
  omega

/-! ## Transfer of `CellOk` -/

/-- `CellOk` transfers to a memory that agrees on the element's bytes. -/
private theorem cellOk_transfer {d : Inst} {init m fin : Mem} {t h c pos : Nat}
    (hcell : CellOk d init m t h c pos)
    (hy : ∀ x ∈ d.y.elems {tokIdx d t h c}, fin x = m x) :
    CellOk d init fin t h c pos := by
  obtain ⟨a, b, cs, sn, v, ha, hb, hcs, hsn, hyv, hvfin, htol⟩ := hcell
  refine ⟨a, b, cs, sn, v, ha, hb, hcs, hsn, ?_, hvfin, htol⟩
  show fin.read .bf16 (d.y.ptr (tokIdx d t h c : Int)).addr = some v
  rw [Mem.read_congr (m₁ := fin) (m₂ := m)
    (fun x hx => hy x (Buf.span_sub_elems d.y (Set.mem_singleton _) hx))]
  exact hyv

/-- The bytes of the element at `k` are written by program `i`. -/
private theorem mem_W {d : Inst} {init : Mem} {i l c : Nat} {x : Int}
    (hl : l < d.BT) (hc : c < 2 * d.R) (hlane : laneOn d init i l) :
    x ∈ d.y.elems {tokIdx d (tokI d init i l) (ax2 d i) c} → x ∈ W d init i := by
  intro hx
  show x ∈ d.y.elems {k | ∃ l c, l < d.BT ∧ c < 2 * d.R ∧ laneOn d init i l ∧
    k = tokIdx d (tokI d init i l) (ax2 d i) c}
  exact Buf.elems_mono d.y (fun k hk => by
    rw [Set.mem_singleton_iff] at hk
    subst hk
    exact ⟨l, c, hl, hc, hlane, rfl⟩) hx

/-- Every byte a program writes is a rotated channel of a token and head. -/
private theorem W_subset_outSet {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size) : W d init i ⊆ d.y.elems (outSet d) := by
  intro x hx
  obtain ⟨k, hk, hxspan⟩ := hx
  obtain ⟨l, c, hl, hc, hlane, rfl⟩ := hk
  exact ⟨tokIdx d (tokI d init i l) (ax2 d i) c,
    ⟨tokI d init i l, ax2 d i, c, tokI_lt hd hpre hi hlane, ax2_lt hi, hc, rfl⟩, hxspan⟩

/-- The programs' local results give `Post`. -/
theorem post_of_local {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {fin : Mem}
    (hloc : ∀ i : Fin (launch d).size, ∃ m, Q d init i m ∧ ∀ x ∈ R d init i ∪ W d init i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d init i) → fin x = init x) :
    Post d init fin := by
  refine ⟨?_, ?_⟩
  · intro n l h c hn hl hh hc
    refine ⟨?_, ?_⟩
    · intro hp
      obtain ⟨i, hlane, htok, hax2, hpos⟩ := exists_program hd hpre hn hl hh hp
      obtain ⟨m, hQ, hag⟩ := hloc i
      have hbt : l % d.BT < d.BT := Nat.mod_lt _ (bt_pos hd)
      have hcell : CellOk d init m (bosAt d init n + l) h c
          ((l : Int) + offAt d init n).toNat := by
        have h0 := hQ (l % d.BT) c hbt hc hlane
        rwa [htok, hax2, hpos] at h0
      refine cellOk_transfer hcell ?_
      intro x hx
      refine hag x (Or.inr ?_)
      have hx' : x ∈ d.y.elems {tokIdx d (tokI d init i (l % d.BT)) (ax2 d i) c} := by
        rw [← htok, ← hax2] at hx
        exact hx
      exact mem_W hbt hc hlane hx'
    · intro hp z hz
      refine hfr z (fun i => ?_)
      intro hzi
      obtain ⟨k', hk', hz'⟩ := hzi
      obtain ⟨l', c', hl', hc', hlane', rfl⟩ := hk'
      obtain ⟨k, hk, hzk⟩ := hz
      rw [Set.mem_singleton_iff] at hk
      subst hk
      have hkk : tokIdx d (tokI d init i l') (ax2 d i) c' =
          tokIdx d (bosAt d init n + l) h c := by
        by_contra hne
        have hne' : ((tokIdx d (tokI d init i l') (ax2 d i) c' : Nat) : Int) ≠
            ((tokIdx d (bosAt d init n + l) h c : Nat) : Int) := by
          intro hcon
          exact hne (Int.ofNat_inj.mp hcon)
        exact Set.disjoint_left.mp (Buf.span_disjoint d.y hne') hz' hzk
      have hsame := tokIdx_inj (d := d) (t := tokI d init i l') (h := ax2 d i) (c := c')
        (t' := bosAt d init n + l) (h' := h) (c' := c)
        (lt_of_lt_of_le hc' hd.r_d) (lt_of_lt_of_le hc hd.r_d) (ax2_lt i.isLt) hh hkk
      exact no_lane_neg hd hpre hn hl hp i.isLt hlane' hsame.1
  · intro z hz
    exact hfr z (fun i => fun h => hz (W_subset_outSet hd hpre i.isLt h))

end Claims.RotaryVarlen
