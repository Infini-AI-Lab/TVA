import Proof.Basic

/-!
# `prepare_pos_seq_lens_i32`: the last program

Program `N` runs `for i in tl.range(N, M, 1024)` and stores `0` to `seq_lens[i + l]` for the
lanes `i + l < M`. Trip `t` writes `seq_lens[N + 1024 t .. N + 1024 t + 1024) ∩ [0, M)`.
-/

namespace Claims.PreparePosSeqLens

open TVA

/-- The zeroing loop of the last program, its first bound `lo` (the `int32` `num_reqs`). -/
def padProg (d : Inst) (lo : I32) : Prog Unit :=
  tl.forRange lo (IntV.ofNat d.M : I32) (IntV.ofNat 1024 : I32) () fun i _ => do
    let block := i + tl.arange 0 1024
    let mask := block.ltS (IntV.ofNat d.M : I32)
    tl.store (d.seq_lens.ptr 0 + block) (tl.full [1024] (IntV.ofInt 0 : I32)) (some mask)

/-- The elements of `seq_lens` trip `t` writes. -/
def padTrip (d : Inst) (t : Nat) : Set Nat :=
  {j | d.N + t * 1024 ≤ j ∧ j < d.N + t * 1024 + 1024 ∧ j < d.M}

/-- The invariant after `t` trips: the elements below `N + 1024 t` of `seq_lens[N .. M)` hold `0`,
and `seq_lens` is still valid. -/
def padInv (d : Inst) (t : Nat) (m : Mem) : Prop :=
  (∀ r : Nat, d.N ≤ r → r < d.M → r < d.N + t * 1024 → ∃ v : I32,
    m.read .i32 (d.seq_lens.ptr (r : Int)).addr = some v ∧ v.toInt = 0) ∧ d.seq_lens.Valid m

/-- One trip of the zeroing loop. -/
theorem safe_padTrip {d : Inst} (hd : d.ok) (E : Env) (t : Nat)
    (ht : d.N + t * 1024 < d.M) (i : I32) (hi : i.toInt = ((d.N + t * 1024 : Nat) : Int))
    (m : Mem) (hI : padInv d t m) :
    Safe E ∅ (d.seq_lens.elems (padTrip d t)) (fun _ m' => padInv d (t + 1) m') m
      (do
        let block := i + tl.arange 0 1024
        let mask := block.ltS (IntV.ofNat d.M : I32)
        tl.store (d.seq_lens.ptr 0 + block) (tl.full [1024] (IntV.ofInt 0 : I32)) (some mask)) := by
  have hM := hd.M_le
  obtain ⟨hQ, hV⟩ := hI
  set r0 : Nat := d.N + t * 1024 with hr0
  have hlane : ∀ l : Idx [1024 - 0], ((i + tl.arange 0 1024).get l).toInt = ((r0 + l.1 : Nat) : Int) := by
    rintro ⟨l, ⟨⟩⟩
    exact IntV.I32.toInt_sadd_arange hi (by omega) l
  let g : Idx [1024 - 0] → Nat := fun l => r0 + l.1.val
  let en : Idx [1024 - 0] → Bool := fun l => decide (r0 + l.1.val < d.M)
  have hMv : (IntV.ofNat d.M : I32).toInt = (d.M : Int) := IntV.I32.toInt_ofNat (by omega)
  have hmask : ∀ l, ((i + tl.arange 0 1024).ltS (IntV.ofNat d.M : I32)).get l = en l := by
    intro l
    simp only [Tensor.get_ltS, IntV.lt_eq, hlane l, hMv, en]
    norm_cast
  have hptr : ∀ l, en l = true →
      (d.seq_lens.ptr 0 + (i + tl.arange 0 1024)).get l = d.seq_lens.ptr (g l : Int) := by
    intro l _
    rw [Tensor.get_ptrAdd, Buf.ptr_offset, hlane l, zero_add]
  have hin : ∀ l, en l = true → g l < d.seq_lens.len := by
    intro l hl
    simp only [en, decide_eq_true_eq] at hl
    have := hd.sl_M
    show r0 + l.1.val < _
    omega
  have himg : Buf.img g en = padTrip d t := by
    ext j
    constructor
    · rintro ⟨l, hl, rfl⟩
      simp only [en, decide_eq_true_eq] at hl
      have := l.1.isLt
      exact ⟨by simp only [g]; omega, by simp only [g]; omega, hl⟩
    · rintro ⟨h1, h2, h3⟩
      refine ⟨(⟨j - r0, by omega⟩, ()), ?_, ?_⟩
      · simp only [en, decide_eq_true_eq]; omega
      · simp only [g]; omega
  refine safe_last ?_
  refine safe_store_buf d.seq_lens (Buf.addrs_of_get d.seq_lens hmask hptr) hin ?_ hV
    (by rw [himg]) fun m' hm' hfr hpres => safe_ret ?_
  · intro a b _ _ h
    rcases a with ⟨a, ⟨⟩⟩; rcases b with ⟨b, ⟨⟩⟩
    dsimp only [g] at h
    exact Prod.ext (Fin.ext (by dsimp only; omega)) rfl
  · refine ⟨fun r hr1 hr2 hr3 => ?_, Buf.valid_of_agree hV fun x hx => hpres x (hV.2.2.2 x hx)⟩
    by_cases hlt : r < r0
    · obtain ⟨v, hv, hv0⟩ := hQ r hr1 hr2 hlt
      refine ⟨v, ?_, hv0⟩
      rw [← hv]
      refine Mem.read_congr fun x hx => hfr x ?_
      rw [himg]
      rintro ⟨j, ⟨hj1, -, -⟩, hxj⟩
      have hne : (r : Int) ≠ (j : Int) := by omega
      exact Set.disjoint_left.mp (d.seq_lens.span_disjoint hne) hx hxj
    · have hl : r - r0 < 1024 - 0 := by omega
      let l : Idx [1024 - 0] := (⟨r - r0, hl⟩, ())
      have hgl : g l = r := by simp only [g, l]; omega
      have hel : en l = true := by simp only [en, l, decide_eq_true_eq]; omega
      refine ⟨(tl.full [1024] (IntV.ofInt 0 : I32)).get l, ?_, ?_⟩
      · have := hm' l hel
        rwa [hgl] at this
      · simp only [tl.full, Tensor.get_full]
        exact toInt_0

/-- **The last program is safe**: it reads nothing, writes `seq_lens[N .. M)` and leaves them
`0`. -/
theorem safe_pad {d : Inst} (hd : d.ok) (E : Env) (m : Mem) (hV : d.seq_lens.Valid m)
    (lo : I32) (hlo : lo.toInt = (d.N : Int)) :
    Safe E ∅ (d.seq_lens.elems (padElems d)) (fun _ m' => QPad d m') m (padProg d lo) := by
  have hM := hd.M_le
  have hN := hd.grid_lt
  have hMv : (IntV.ofNat d.M : I32).toInt = (d.M : Int) := IntV.I32.toInt_ofNat (by omega)
  have hrange : tl.RangeOk lo (IntV.ofNat d.M : I32) (IntV.ofNat 1024 : I32) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [toInt_1024]; norm_num
    · rw [hMv, toInt_1024, maxInt32]; push_cast; omega
    · rw [hlo, maxInt32]; push_cast; omega
    · rw [toInt_1024, maxInt32]; norm_num
  set n := tl.tripCount lo.toInt (IntV.ofNat d.M : I32).toInt (IntV.ofNat 1024 : I32).toInt
    with hn
  have hnv : n = (((d.M : Int) - d.N + 1023) / 1024).toNat := by
    rw [hn, hlo, hMv, toInt_1024]; simp only [tl.tripCount]; congr 2; ring
  -- a trip below `n` starts below `M`; after `n` trips, `N + 1024 n ≥ M`
  have htrip : ∀ t, t < n → d.N + t * 1024 < d.M := by
    intro t ht
    rw [hnv] at ht
    omega
  have hend : d.M ≤ d.N + n * 1024 := by
    rw [hnv]; omega
  unfold padProg
  refine safe_last ?_
  refine Safe.mono (Q := fun _ m' => QPad d m')
    (W := iterW (fun t => d.seq_lens.elems (padTrip d t)) n ∪ ∅) ?_ (fun _ _ h => h) ?_
  · refine safe_forRange hrange (fun t _ m => padInv d t m)
      (fun t => d.seq_lens.elems (padTrip d t)) ?_ (Set.disjoint_empty _) ?_ ?_ ?_
    · intro t u htu _
      refine Buf.elems_disjoint d.seq_lens ?_
      rw [Set.disjoint_left]
      rintro j ⟨h1, h2, -⟩ ⟨h3, -, -⟩
      omega
    · refine ⟨fun r hr1 _ hr3 => by omega, hV⟩
    · intro t ht _ m₁ hI
      have ht' := htrip t ht
      have hi : (IntV.ofInt (lo.toInt + (t : Int) * (IntV.ofNat 1024 : I32).toInt) : I32).toInt
          = ((d.N + t * 1024 : Nat) : Int) := by
        rw [hlo, toInt_1024]
        have h := IntV.I32.toInt_ofInt (x := (d.N : Int) + (t : Int) * 1024) ⟨by omega, by omega⟩
        rw [h]; push_cast; ring
      exact safe_padTrip hd E t ht' _ hi m₁ hI
    · intro _ m₁ hI
      exact safe_ret fun r hr1 hr2 => hI.1 r hr1 hr2 (by omega)
  · rintro z (⟨t, -, hz⟩ | hz)
    · refine Buf.elems_mono d.seq_lens ?_ hz
      rintro j ⟨h1, -, h3⟩
      exact ⟨by omega, h3⟩
    · exact absurd hz (Set.notMem_empty z)

end Claims.PreparePosSeqLens
