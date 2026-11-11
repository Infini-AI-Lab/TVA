import Proof.Defs
import Proof.Grp

/-!
# `lora_shrink_bf16`: which program owns which output element

The grouped order of the program ids (lines 351-361) and the adapter groups: every output element
`(s, t, j)` of a token with an adapter is added to by exactly `SPLIT_K` programs, one per part of
`K`.
-/

namespace Claims.LoraShrink

open TVA

/-- The block shapes of the launch are Triton block shapes. -/
theorem launch_blocks_ok (d : Inst) : ∀ s ∈ (launch d).blocks, s.TritonOk := by
  intro s hs
  have h : bk d = 256 ∨ bk d = 32 := by unfold bk; split_ifs <;> simp
  simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
  have p2 : ∀ k, k = 32 ∨ k = 16 ∨ k = 256 → ∃ e, k = 2 ^ e := by
    rintro k (rfl | rfl | rfl)
    · exact ⟨5, by norm_num⟩
    · exact ⟨4, by norm_num⟩
    · exact ⟨8, by norm_num⟩
  have hb : ∃ e, bk d = 2 ^ e := p2 _ (by rcases h with h | h <;> simp [h])
  refine ⟨?_, ?_⟩
  · intro k hk
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hk <;>
      first
      | (rcases hk with rfl | rfl <;> first | exact hb | exact p2 _ (by simp))

  · rcases h with h | h <;>
      rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp [h, List.prod]

/-- The grid is the product of its three axes. -/
theorem launch_size (d : Inst) : (launch d).size = grid0 d * d.S * d.G := rfl

/-- The launch is valid on a covered instance. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  refine ⟨hd.grid0_lt, ?_, ?_, ?_, rfl, rfl, rfl, launch_blocks_ok d⟩
  · exact hd.S_lt
  · exact hd.G_lt
  · rw [launch_size]; exact hd.grid_size

/-- The program context of program `i` in terms of the decode of `i`. -/
theorem ctx_pid (d : Inst) (i : Nat) :
    ((launch d).ctx i).pid 0 = ax0 d i ∧ ((launch d).ctx i).pid 1 = slice d i ∧
      ((launch d).ctx i).pid 2 = grpOf d i := ⟨rfl, rfl, rfl⟩

/-- A program number below the size has its axes in range. -/
theorem decode_lt (d : Inst) (hd : d.ok) {i : Nat} (hi : i < (launch d).size) :
    ax0 d i < grid0 d ∧ slice d i < d.S ∧ grpOf d i < d.G := by
  rw [launch_size] at hi
  have hS := hd.S_pos
  have h0 : 0 < grid0 d := by
    rcases Nat.eq_zero_or_pos (grid0 d) with h | h
    · rw [h] at hi; simp at hi
    · exact h
  refine ⟨Nat.mod_lt _ h0, Nat.mod_lt _ hS, ?_⟩
  unfold grpOf
  exact Nat.div_lt_of_lt_mul hi

/-- The tile of an axis-0 program is in range. -/
theorem tile_lt (d : Inst) {x : Nat} (hx : x < grid0 d) : pidM d x < cm d ∧ pidN d x < cn d := by
  have h : pidMN d x < cm d * cn d := by
    unfold pidMN
    refine Nat.div_lt_of_lt_mul ?_
    have : grid0 d = sk d * (cm d * cn d) := by unfold grid0 cm cn; ring
    omega
  exact ⟨gM_lt h, gN_lt h⟩

/-- The program number is its three axes recombined. -/
theorem decode_eq (d : Inst) (i : Nat) : i = ax0 d i + grid0 d * (slice d i + d.S * grpOf d i) := by
  unfold ax0 slice grpOf
  rw [← Nat.div_div_eq_div_mul]
  have h1 := Nat.mod_add_div i (grid0 d)
  have h2 := Nat.mod_add_div (i / grid0 d) d.S
  rw [h2]; exact h1.symm

/-- The axes of a recombined program number. -/
theorem decode_of_eq (d : Inst) {x s g : Nat} (hx : x < grid0 d) (hs : s < d.S) :
    ax0 d (x + grid0 d * (s + d.S * g)) = x ∧ slice d (x + grid0 d * (s + d.S * g)) = s ∧
      grpOf d (x + grid0 d * (s + d.S * g)) = g := by
  have h0 : 0 < grid0 d := by omega
  have hS : 0 < d.S := by omega
  have hq : (x + grid0 d * (s + d.S * g)) / grid0 d = s + d.S * g := by
    rw [Nat.add_mul_div_left _ _ h0, Nat.div_eq_of_lt hx, zero_add]
  refine ⟨?_, ?_, ?_⟩
  · unfold ax0; rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hx]
  · unfold slice; rw [hq, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hs]
  · unfold grpOf; rw [← Nat.div_div_eq_div_mul, hq, Nat.add_mul_div_left _ _ hS,
      Nat.div_eq_of_lt hs, zero_add]

/-- What an active program knows: its group has an adapter, and the positions of its rows lie in
the group's range of the sorted token indices. -/
theorem active_facts (d : Inst) (hd : d.ok) {i : Nat} (h : Active d i) :
    grpOf d i < d.n ∧ idAt d (grpOf d i) = d.groupId (grpOf d i) ∧ 0 ≤ d.groupId (grpOf d i) ∧
      ∀ m, m < mlen d i → d.groupStart (grpOf d i) ≤ startAt d (grpOf d i) + tileM d i * 32 + m ∧
        startAt d (grpOf d i) + tileM d i * 32 + m < d.groupStart (grpOf d i) + d.groupCnt (grpOf d i) := by
  obtain ⟨h1, h2⟩ := h
  have hg : grpOf d i < d.n := by
    by_contra hc
    apply h1
    simp [idAt, hc]
  have hid : idAt d (grpOf d i) = d.groupId (grpOf d i) := by simp [idAt, hg]
  have hnn : 0 ≤ d.groupId (grpOf d i) := by
    obtain ⟨t0, ht0, hat⟩ := hd.grp_nonempty _ hg
    have := (hd.ad_range t0 ht0).1
    rw [hid] at h1
    omega
  refine ⟨hg, hid, hnn, fun m hm => ?_⟩
  have hs : startAt d (grpOf d i) = d.groupStart (grpOf d i) := by simp [startAt, hg.le]
  have hc : cntAt d (grpOf d i) = d.groupCnt (grpOf d i) := by simp [cntAt, hg]
  rw [hs]
  unfold mlen at hm
  rw [hc] at hm h2
  have := Nat.lt_min.mp hm
  omega

/-- A program that targets `(s, t, j)` is in a group whose adapter is `t`'s, and the element is in
the output. -/
theorem tgt_range (d : Inst) (hd : d.ok) {i s t j : Nat} (hi : i < (launch d).size)
    (h : Tgt d i s t j) : s < d.S ∧ t < d.M ∧ j < d.R ∧ 0 ≤ d.adapter t := by
  obtain ⟨hact, hs, ⟨m, hm, ht⟩, hj1, hj2⟩ := h
  obtain ⟨hx, hsl, hg⟩ := decode_lt d hd hi
  obtain ⟨hgn, hid, hnn, hpos⟩ := active_facts d hd hact
  obtain ⟨hp1, hp2⟩ := hpos m hm
  unfold tokOf at ht
  obtain ⟨hpM, htM, hadj⟩ := hd.tid_range _ hgn _ hp1 hp2
  have htile := (tile_lt d hx).2
  have hR := hd.R_blk
  have hcn : 16 * cn d = d.R := by unfold cn; omega
  refine ⟨by rw [hs]; exact hsl, by rw [ht]; exact htM, ?_, ?_⟩
  · unfold tileN at hj2; nlinarith
  · rw [ht, hadj]; exact hnn

/-- **Every output element of a token with an adapter is added to by exactly one program per part of
`K`**: for each part `p < SPLIT_K` there is exactly one program `f p` that targets `(s, t, j)`
(`Tgt`), and it accumulates part `p`. -/
theorem tgt_unique (d : Inst) (hd : d.ok) {s t j : Nat} (hs : s < d.S) (ht : t < d.M)
    (hj : j < d.R) (hadapt : 0 ≤ d.adapter t) :
    ∃ f : Nat → Nat, (∀ p, p < sk d → f p < (launch d).size ∧ Tgt d (f p) s t j ∧ part d (f p) = p) ∧
      ∀ i, i < (launch d).size → Tgt d i s t j → ∃ p, p < sk d ∧ i = f p := by
  have hsk : 0 < sk d := by unfold sk; split_ifs <;> norm_num
  obtain ⟨g, hgn, hgt⟩ := hd.grp_cover t ht
  obtain ⟨pos, hp1, hp2, hpt⟩ := hd.tid_cover t ht g hgn hgt
  obtain ⟨hposM, -, -⟩ := hd.tid_range g hgn pos hp1 hp2
  have hR := hd.R_blk
  have hcn : 16 * cn d = d.R := by unfold cn; omega
  have hcnt : d.groupCnt g ≤ d.M := by
    rw [hd.cnt_def g hgn]
    calc _ ≤ (Finset.range d.M).card := Finset.card_filter_le _ _
      _ = d.M := Finset.card_range _
  have hsg : startAt d g = d.groupStart g := by simp [startAt, hgn.le]
  have hcg : cntAt d g = d.groupCnt g := by simp [cntAt, hgn]
  have hidg : idAt d g = d.groupId g := by simp [idAt, hgn]
  set a := (pos - d.groupStart g) / 32 with ha
  set m := (pos - d.groupStart g) % 32 with hm
  set b := j / 16 with hb
  have hdm := Nat.div_add_mod (pos - d.groupStart g) 32
  have hacm : a < cm d := by unfold cm; omega
  have hbcn : b < cn d := by omega
  have hGn : g < d.G := lt_of_lt_of_le hgn hd.n_le_G
  have hcmcn : cm d * cn d * sk d = grid0 d := by unfold grid0 cm cn; ring
  have hslot := (g_dec_enc (cm := cm d) (cn := cn d) hacm hbcn)
  have hen := gEnc_lt (cm := cm d) (cn := cn d) hacm hbcn
  refine ⟨fun p => p + sk d * gEnc (cm d) (cn d) a b + grid0 d * (s + d.S * g), ?_, ?_⟩
  · intro p hp
    beta_reduce
    have hx : p + sk d * gEnc (cm d) (cn d) a b < grid0 d := by
      rw [← hcmcn]
      nlinarith
    obtain ⟨e0, e1, e2⟩ := decode_of_eq d (g := g) hx hs
    have hpart : part d (p + sk d * gEnc (cm d) (cn d) a b + grid0 d * (s + d.S * g)) = p := by
      unfold part pidSk; rw [e0, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hp]
    have hmn : pidMN d (p + sk d * gEnc (cm d) (cn d) a b) = gEnc (cm d) (cn d) a b := by
      unfold pidMN
      rw [Nat.add_mul_div_left _ _ (by omega : 0 < sk d), Nat.div_eq_of_lt hp, zero_add]
    have hM : tileM d (p + sk d * gEnc (cm d) (cn d) a b + grid0 d * (s + d.S * g)) = a := by
      unfold tileM pidM; rw [e0]
      have : pidM d (p + sk d * gEnc (cm d) (cn d) a b) = gM (cm d) (cn d) (gEnc (cm d) (cn d) a b) := by
        show gM (cm d) (cn d) (pidMN d _) = _; rw [hmn]
      rw [← this] at hslot; exact hslot.1
    have hN : tileN d (p + sk d * gEnc (cm d) (cn d) a b + grid0 d * (s + d.S * g)) = b := by
      unfold tileN pidN; rw [e0]
      have : pidN d (p + sk d * gEnc (cm d) (cn d) a b) = gN (cm d) (cn d) (gEnc (cm d) (cn d) a b) := by
        show gN (cm d) (cn d) (pidMN d _) = _; rw [hmn]
      rw [← this] at hslot; exact hslot.2
    have hsz : p + sk d * gEnc (cm d) (cn d) a b + grid0 d * (s + d.S * g) < (launch d).size := by
      rw [launch_size]
      have h2 : d.S * (g + 1) ≤ d.S * d.G := Nat.mul_le_mul_left _ hGn
      have : s + d.S * g + 1 ≤ d.S * d.G := by rw [Nat.mul_succ] at h2; omega
      have h3 : grid0 d * (s + d.S * g + 1) ≤ grid0 d * (d.S * d.G) := Nat.mul_le_mul_left _ this
      rw [Nat.mul_succ] at h3
      have h4 : grid0 d * (d.S * d.G) = grid0 d * d.S * d.G := by ring
      omega
    generalize p + sk d * gEnc (cm d) (cn d) a b + grid0 d * (s + d.S * g) = X at e1 e2 hpart hM hN hsz
    refine ⟨hsz, ⟨⟨?_, ?_⟩, ?_, ⟨m, ?_, ?_⟩, ?_, ?_⟩, hpart⟩
    · rw [e2, hidg]; omega
    · rw [hM, e2, hcg]; omega
    · exact e1.symm
    · unfold mlen; rw [hM, e2, hcg]
      exact Nat.lt_min.mpr ⟨by omega, by omega⟩
    · unfold tokOf; rw [hM, e2, hsg]
      have : d.groupStart g + a * 32 + m = pos := by omega
      rw [this, hpt]
    · rw [hN]; omega
    · rw [hN]; omega
  · intro i hi hT
    obtain ⟨hx, hsl, hgl⟩ := decode_lt d hd hi
    have hT' := hT
    obtain ⟨hact, hs', ⟨m', hm', ht'⟩, hj1, hj2⟩ := hT
    obtain ⟨hgn', hid', hnn', hpos'⟩ := active_facts d hd hact
    obtain ⟨hp1', hp2'⟩ := hpos' m' hm'
    have hs'' : startAt d (grpOf d i) = d.groupStart (grpOf d i) := by simp [startAt, hgn'.le]
    unfold tokOf at ht'
    obtain ⟨-, -, hadj'⟩ := hd.tid_range _ hgn' _ hp1' hp2'
    have hgg : grpOf d i = g := by
      refine hd.grp_inj _ _ hgn' hgn ?_
      rw [← hadj', ← ht', hgt]
    rw [hgg] at hp1' hp2' ht' hs''
    have hpp : startAt d g + tileM d i * 32 + m' = pos := by
      refine hd.tid_inj _ _ (hd.tid_range g hgn _ hp1' hp2').1 hposM ?_
      rw [← ht', hpt]
    have hm32 : m' < 32 := by
      have := Nat.lt_min.mp (show m' < Nat.min 32 _ from hm'); omega
    have hA : tileM d i = a := by
      rw [ha]; omega
    have hB : tileN d i = b := by rw [hb]; omega
    have hxt := tile_lt d hx
    have hmn : gEnc (cm d) (cn d) (pidM d (ax0 d i)) (pidN d (ax0 d i)) = pidMN d (ax0 d i) :=
      g_enc_dec (by unfold pidMN; exact Nat.div_lt_of_lt_mul (by
        have : grid0 d = sk d * (cm d * cn d) := by unfold grid0 cm cn; ring
        omega))
    have hAa : pidM d (ax0 d i) = a := hA
    have hBb : pidN d (ax0 d i) = b := hB
    rw [hAa, hBb] at hmn
    refine ⟨part d i, Nat.mod_lt _ (by omega), ?_⟩
    have hdec := decode_eq d i
    have hax : ax0 d i = part d i + sk d * gEnc (cm d) (cn d) a b := by
      rw [hmn]; unfold part pidSk pidMN
      exact (Nat.mod_add_div _ _).symm
    conv_lhs => rw [hdec]
    rw [hax, ← hs', hgg]

end Claims.LoraShrink
