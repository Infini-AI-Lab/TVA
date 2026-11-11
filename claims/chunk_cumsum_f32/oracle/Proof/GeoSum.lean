import Claim
import TVA.Proof
import Proof.Geo
import Proof.GeoBound

/-!
# The leaves of a lane's scan, as the claim's reference sum

Lane `l` of program `i` scans the lanes `0 … l` (forward) or `l … BT-1` (reverse) of its chunk.
The enabled ones are exactly the positions of `rng`, so the sum of the values the scan combines is
`refSum`, and the sum of their magnitudes is `absSum`. The number of leaves is `depth + 1`.
-/

namespace Claims.ChunkCumsum

open TVA

/-- The leaf lane `j` contributes to the scan: the gate value at its position, or `+0` when masked. -/
def sLeaf (d : Inst) (i h j : Nat) : F32 :=
  if en d i j then sAt d (pos d i j) h else FloatV.ofPat 0

/-- A lane's flat position is its sequence's start plus its offset in the sequence. -/
theorem pos_eq_base {d : Inst} {i l : Nat} :
    pos d i l = lo d (nOf d i) + (chkOf d i * d.BT + l) := by
  unfold pos baseOf
  rw [Nat.add_assoc]

/-- `BT` is positive. -/
theorem BT_pos {d : Inst} (hd : d.ok) : 0 < d.BT := by
  obtain ⟨e, he⟩ := hd.bt_pow2
  rw [he]
  exact pow_pos (by norm_num) e

/-- The start of a lane's chunk is the sequence start plus `chunk·BT`. -/
theorem chunkStart_pos {d : Inst} (hd : d.ok) {i l : Nat} (hl : l < d.BT) :
    chunkStart d (nOf d i) (pos d i l) = lo d (nOf d i) + chkOf d i * d.BT := by
  unfold chunkStart
  rw [pos_eq_base, Nat.add_sub_cancel_left]
  congr 1
  have hdiv : (chkOf d i * d.BT + l) / d.BT = chkOf d i := by
    rw [Nat.mul_comm (chkOf d i) d.BT, Nat.mul_add_div (BT_pos hd),
      Nat.div_eq_of_lt hl, Nat.add_zero]
  rw [hdiv]

/-- The same, with the program's base. -/
theorem chunkStart_eq_base {d : Inst} (hd : d.ok) {i l : Nat} (hl : l < d.BT) :
    chunkStart d (nOf d i) (pos d i l) = baseOf d i + chkOf d i * d.BT := by
  rw [chunkStart_pos hd hl]; rfl

/-- The chunk starts at or before the lane. -/
theorem chunkStart_le_pos {d : Inst} (hd : d.ok) {i l : Nat} (hl : l < d.BT) :
    chunkStart d (nOf d i) (pos d i l) ≤ pos d i l := by
  rw [chunkStart_pos hd hl, pos_eq_base]
  omega

/-- A lane is `l` positions past its chunk's start. -/
theorem pos_sub_chunkStart {d : Inst} (hd : d.ok) {i l : Nat} (hl : l < d.BT) :
    pos d i l - chunkStart d (nOf d i) (pos d i l) = l := by
  rw [chunkStart_pos hd hl, pos_eq_base]
  omega

/-- A lane's position is its chunk start plus `l`. -/
theorem pos_eq_chunkStart_add {d : Inst} (hd : d.ok) {i l j : Nat} (hl : l < d.BT) :
    pos d i j = chunkStart d (nOf d i) (pos d i l) + j := by
  rw [chunkStart_eq_base hd hl, pos_eq_base]
  unfold baseOf
  omega

/-- The depth of a lane's scan: `l` forward, `BT - 1 - l` reverse. -/
theorem depth_eq {d : Inst} (hd : d.ok) {i l : Nat} (hl : l < d.BT) :
    depth d (nOf d i) (pos d i l) = if d.reverse then d.BT - 1 - l else l := by
  unfold depth
  have h1 := pos_sub_chunkStart (i := i) hd hl
  have h2 := chunkStart_le_pos (i := i) hd hl
  cases hrv : d.reverse <;> simp only [ite_true, ite_false, Bool.false_eq_true] <;> omega

/-- Forward, the positions summed for `p` run from the chunk start to `p`. -/
theorem rng_false {d : Inst} (h : d.reverse = false) (n p : Nat) :
    rng d n p = Finset.Icc (chunkStart d n p) p := by
  simp [rng, h]

/-- Reverse, the positions summed for `p` run from `p` to the chunk end. -/
theorem rng_true {d : Inst} (h : d.reverse = true) (n p : Nat) :
    rng d n p = Finset.Ico p (min (chunkStart d n p + d.BT) (hi d n)) := by
  simp [rng, h]

/-- The enabled lanes of a forward chunk are exactly the positions of `rng`. -/
theorem mem_rng_fwd {d : Inst} (hT : d.varlen = true → TablesOk d) (hd : d.ok) {i l j : Nat}
    (hn : nOf d i < nSeq d) (hl : l < d.BT) (hj : j ≤ l) (he : en d i l = true)
    (hr : d.reverse = false) :
    pos d i j ∈ rng d (nOf d i) (pos d i l) ↔ en d i j = true := by
  rw [rng_false hr, Finset.mem_Icc]
  have hbl := baseOf_add_len hT i hn
  have hcb := chunkStart_eq_base (i := i) hd hl
  have hbase : baseOf d i = lo d (nOf d i) := rfl
  have hlen : lenOf d i = hi d (nOf d i) - lo d (nOf d i) := rfl
  have hpj := pos_eq_base (d := d) (i := i) (l := j)
  have hpl := pos_eq_base (d := d) (i := i) (l := l)
  have hhi := pos_lt_hi hT hn he
  constructor
  · rintro ⟨-, h2⟩
    rw [en_iff]
    omega
  · intro hlt
    rw [en_iff] at hlt
    constructor
    · omega
    · omega

/-- The enabled lanes of a reverse chunk are exactly the positions of `rng`. -/
theorem mem_rng_rev {d : Inst} (hT : d.varlen = true → TablesOk d) (hd : d.ok) {i l j : Nat}
    (hn : nOf d i < nSeq d) (hl : l < d.BT) (hj : l ≤ j) (hj' : j < d.BT)
    (hr : d.reverse = true) :
    pos d i j ∈ rng d (nOf d i) (pos d i l) ↔ en d i j = true := by
  rw [rng_true hr, Finset.mem_Ico, Nat.lt_min]
  have hbl := baseOf_add_len hT i hn
  have hpos := pos_eq_chunkStart_add (i := i) (j := j) hd hl
  have hpl := pos_eq_chunkStart_add (d := d) (i := i) (l := l) (j := l) hd hl
  have hcb := chunkStart_eq_base (i := i) hd hl
  have hbase : baseOf d i = lo d (nOf d i) := rfl
  have hlen : lenOf d i = hi d (nOf d i) - lo d (nOf d i) := rfl
  constructor
  · rintro ⟨-, h2, h3⟩
    rw [en_iff]
    omega
  · intro hlt
    rw [en_iff] at hlt
    refine ⟨?_, ?_, ?_⟩
    · rw [hpl, hpos]; omega
    · rw [hpos]; omega
    · exact pos_lt_hi hT hn (by rw [en_iff]; exact hlt)

/-- A lane of a lane's leaf list is enabled iff its position is in range. -/
theorem leaves_mem_rng {d : Inst} (hT : d.varlen = true → TablesOk d) (hd : d.ok) {i l j : Nat}
    (hn : nOf d i < nSeq d) (hl : l < d.BT)
    (hj : j ∈ leavesOf d.reverse d.BT l) (hen : en d i j = true) :
    pos d i j ∈ rng d (nOf d i) (pos d i l) := by
  unfold leavesOf at hj
  have hpos : pos d i j = chunkStart d (nOf d i) (pos d i l) + j := pos_eq_chunkStart_add hd hl
  have hpl : pos d i l = chunkStart d (nOf d i) (pos d i l) + l :=
    pos_eq_chunkStart_add (d := d) (i := i) (l := l) (j := l) hd hl
  by_cases hb : d.reverse
  · have hrv : d.reverse = true := hb
    rw [hrv] at hj
    simp only [ite_true] at hj
    rw [rng_true hrv, Finset.mem_Ico, Nat.lt_min]
    rw [List.mem_range'] at hj
    obtain ⟨x, hx, rfl⟩ := hj
    constructor
    · rw [hpl, hpos]
      omega
    · constructor
      · rw [hpos]
        omega
      · exact pos_lt_hi hT hn hen
  · have hrv : d.reverse = false := by simpa using hb
    rw [hrv] at hj
    simp only [Bool.false_eq_true, ite_false] at hj
    rw [rng_false hrv, Finset.mem_Icc]
    rw [List.mem_range] at hj
    constructor
    · rw [chunkStart_eq_base (i := i) hd hl, pos_eq_base]
      have hbase : baseOf d i = lo d (nOf d i) := rfl
      omega
    · calc pos d i j = chunkStart d (nOf d i) (pos d i l) + j := hpos
        _ ≤ chunkStart d (nOf d i) (pos d i l) + l := by omega
        _ = pos d i l := hpl.symm

/-- Every leaf's value is finite when the element is in range. -/
theorem leaves_finite {d : Inst} (hT : d.varlen = true → TablesOk d) (hd : d.ok) {i h l j : Nat}
    (hn : nOf d i < nSeq d) (hl : l < d.BT) (hj : j ∈ leavesOf d.reverse d.BT l)
    (hin : InRange d (nOf d i) (pos d i l) h) : (sLeaf d i h j).isFinite = true := by
  unfold sLeaf
  by_cases hc : en d i j
  · simp only [hc, ite_true]
    exact hin.1 (pos d i j) (leaves_mem_rng hT hd hn hl hj hc)
  · simp only [hc]
    exact ofPat_zero_isFinite

/-- The list of leaves has no repeats. -/
theorem leaves_nodup (rev : Bool) (BT l : Nat) : (leavesOf rev BT l).Nodup := by
  unfold leavesOf
  by_cases hr : rev
  · rw [ite_eq_left hr]
    exact List.nodup_range' 1
  · rw [ite_eq_right hr]
    exact List.nodup_range

/-- The number of leaves of a lane is its depth plus one. -/
theorem leaves_length {d : Inst} (hd : d.ok) {i l : Nat} (hl : l < d.BT) :
    (leavesOf d.reverse d.BT l).length = depth d (nOf d i) (pos d i l) + 1 := by
  rw [depth_eq hd hl]
  unfold leavesOf
  by_cases hr : d.reverse = true
  · simp only [hr, ite_true, List.length_range']
    omega
  · have hf : d.reverse = false := by
      cases h : d.reverse <;> simp_all
    simp only [hf, Bool.false_eq_true, ite_false, List.length_range]

/-- The depth bound of the claim. -/
theorem leaves_length_le {d : Inst} (hd : d.ok) (rev : Bool) {l : Nat} (hl : l < d.BT) :
    (leavesOf rev d.BT l).length - 1 ≤ 2 ^ 20 := by
  have hbt := hd.bt_le
  unfold leavesOf
  by_cases hr : rev
  · rw [ite_eq_left hr, List.length_range']
    omega
  · rw [ite_eq_right hr, List.length_range]
    omega

/-! ## The two sums -/

/-- Generic sum: the leaves' `F`-images are the reference sum's `F`-images. -/
theorem leaves_map_sum {d : Inst} (hT : d.varlen = true → TablesOk d) (hd : d.ok) {i h : Nat}
    (hn : nOf d i < nSeq d) {l : Nat} (hl : l < d.BT) (he : en d i l = true)
    (F : ℚ → ℚ) (hF0 : F 0 = 0) :
    ((leavesOf d.reverse d.BT l).map (fun j => F ((sLeaf d i h j).toRat))).sum =
      ∑ q ∈ rng d (nOf d i) (pos d i l), F ((sAt d q h).toRat) := by
  have hposl : pos d i l = chunkStart d (nOf d i) (pos d i l) + l :=
    pos_eq_chunkStart_add (d := d) (i := i) (l := l) (j := l) hd hl
  have hpos : ∀ j, pos d i j = chunkStart d (nOf d i) (pos d i l) + j :=
    fun j => pos_eq_chunkStart_add (i := i) (j := j) hd hl
  cases hrv : d.reverse
  · -- forward
    have hleaves : leavesOf false d.BT l = List.range (l + 1) := by
      simp [leavesOf]
    rw [hleaves, rng_false hrv]
    rw [← List.sum_toFinset (fun j => F ((sLeaf d i h j).toRat)) List.nodup_range,
      List.toFinset_range]
    have hcongr : ∀ j ∈ Finset.range (l + 1),
        F ((sLeaf d i h j).toRat) =
          F ((sAt d (chunkStart d (nOf d i) (pos d i l) + j) h).toRat) := by
      intro j hj
      rw [Finset.mem_range] at hj
      have hjle : j ≤ l := by omega
      have henj : en d i j = true := by
        rw [en_iff] at he ⊢
        omega
      rw [sLeaf, ite_eq_left henj, hpos j]
    rw [Finset.sum_congr rfl hcongr]
    apply Finset.sum_bij (fun x _ => chunkStart d (nOf d i) (pos d i l) + x)
    · intro x hx
      rw [Finset.mem_range] at hx
      rw [Finset.mem_Icc]
      constructor
      · omega
      · have hxl : x ≤ l := by omega
        omega
    · intro a _ b _ hab
      omega
    · intro b hb
      rw [Finset.mem_Icc] at hb
      refine ⟨b - chunkStart d (nOf d i) (pos d i l), ?_, ?_⟩
      · rw [Finset.mem_range]; omega
      · omega
    · intro x _; rfl
  · -- reverse
    have hleaves : leavesOf true d.BT l = List.range' l (d.BT - l) := by
      simp [leavesOf]
    rw [hleaves, List.range'_eq_map_range, rng_true hrv]
    have hmap : (List.map (fun j => F ((sLeaf d i h j).toRat))
        (List.map (fun x => l + x) (List.range (d.BT - l)))) =
        List.map (fun x => F ((sLeaf d i h (l + x)).toRat)) (List.range (d.BT - l)) := by
      rw [List.map_map]; rfl
    rw [hmap]
    rw [← List.sum_toFinset (fun x => F ((sLeaf d i h (l + x)).toRat)) List.nodup_range,
      List.toFinset_range]
    have htofilter : ∑ x ∈ Finset.range (d.BT - l), F ((sLeaf d i h (l + x)).toRat) =
        ∑ x ∈ (Finset.range (d.BT - l)).filter (fun x => en d i (l + x) = true),
          F ((sAt d (pos d i (l + x)) h).toRat) := by
      rw [Finset.sum_filter]
      apply Finset.sum_congr rfl
      intro x _
      by_cases hc : en d i (l + x) = true
      · simp [sLeaf, hc]
      · simp [sLeaf, hc, hF0]
    rw [htofilter]
    apply Finset.sum_bij (fun x _ => pos d i (l + x))
    · intro x hx
      rw [Finset.mem_filter, Finset.mem_range] at hx
      obtain ⟨hxr, hxe⟩ := hx
      rw [Finset.mem_Ico, Nat.lt_min]
      have hxlt : l + x < d.BT := by omega
      constructor
      · calc pos d i l = chunkStart d (nOf d i) (pos d i l) + l := hposl
          _ ≤ chunkStart d (nOf d i) (pos d i l) + (l + x) := by omega
          _ = pos d i (l + x) := (hpos (l + x)).symm
      · constructor
        · rw [hpos]
          omega
        · exact pos_lt_hi hT hn hxe
    · intro a _ b _ hab
      rw [hpos (l + a), hpos (l + b)] at hab
      omega
    · intro b hb
      rw [Finset.mem_Ico, Nat.lt_min] at hb
      obtain ⟨hb1, hb2, hb3⟩ := hb
      have hcl : chunkStart d (nOf d i) (pos d i l) + l ≤ b := by rw [← hposl]; exact hb1
      refine ⟨b - (chunkStart d (nOf d i) (pos d i l) + l), ?_, ?_⟩
      · rw [Finset.mem_filter, Finset.mem_range]
        have hxsum : chunkStart d (nOf d i) (pos d i l) + l +
            (b - (chunkStart d (nOf d i) (pos d i l) + l)) = b := by omega
        constructor
        · omega
        · rw [en_iff]
          have hbl := baseOf_add_len hT i hn
          have hbase : baseOf d i = lo d (nOf d i) := rfl
          have hlen : lenOf d i = hi d (nOf d i) - lo d (nOf d i) := rfl
          have hpj := hpos (l + (b - (chunkStart d (nOf d i) (pos d i l) + l)))
          have hpb := pos_eq_base (d := d) (i := i)
            (l := l + (b - (chunkStart d (nOf d i) (pos d i l) + l)))
          have hcb := chunkStart_eq_base (i := i) hd hl
          omega
      · have hxsum : chunkStart d (nOf d i) (pos d i l) + l +
            (b - (chunkStart d (nOf d i) (pos d i l) + l)) = b := by omega
        rw [hpos]
        omega
    · intro x _; rfl

/-- The sum of the leaves' values is the claim's reference sum. -/
theorem leaves_sum {d : Inst} (hT : d.varlen = true → TablesOk d) (hd : d.ok) {i h : Nat}
    (hn : nOf d i < nSeq d) {l : Nat} (hl : l < d.BT) (he : en d i l = true) :
    ((leavesOf d.reverse d.BT l).map (fun j => (sLeaf d i h j).toRat)).sum =
      refSum d (nOf d i) (pos d i l) h := by
  rw [refSum]
  exact leaves_map_sum hT hd hn hl he (fun x => x) rfl

/-- The sum of the leaves' magnitudes is the claim's absolute sum. -/
theorem leaves_absSum {d : Inst} (hT : d.varlen = true → TablesOk d) (hd : d.ok) {i h : Nat}
    (hn : nOf d i < nSeq d) {l : Nat} (hl : l < d.BT) (he : en d i l = true) :
    ((leavesOf d.reverse d.BT l).map (fun j => |(sLeaf d i h j).toRat|)).sum =
      absSum d (nOf d i) (pos d i l) h := by
  rw [absSum]
  exact leaves_map_sum hT hd hn hl he (fun x => |x|) (by simp)

/-- The unit tensor's lane access. -/
theorem get_set0 {α : Type} {BT : Nat} (x : Tensor [BT] α) {l : Nat} (hl : l < BT) {j : Nat}
    (hj : j < BT) : x.get (Idx.set (⟨l, hl⟩, ()) 0 j) = x.get (⟨j, hj⟩, ()) := by
  simp [Idx.set, hj]

end Claims.ChunkCumsum
