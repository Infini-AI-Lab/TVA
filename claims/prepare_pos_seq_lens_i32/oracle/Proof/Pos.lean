import Proof.Basic

/-!
# `prepare_pos_seq_lens_i32`: the positions loop of a request

Request `r`'s program runs `for i in tl.range(0, query_len, 1024)` and stores
`num_computed_tokens + i + l` (converted to `int64`) to `pos[start + i + l]` for the lanes
`i + l < query_len`. With `start = S`, `query_len = Q` and `num_computed_tokens = C` (natural
values), trip `t` writes `pos[S + 1024 t .. S + 1024 t + 1024) ∩ [S, S + Q)`.
-/

namespace Claims.PreparePosSeqLens

open TVA

/-- The positions loop of a request (source lines 83-87). -/
def posProg (d : Inst) (nc start ql : I32) : Prog Unit :=
  tl.forRange (IntV.ofInt 0 : I32) ql (IntV.ofNat 1024 : I32) () fun i _ => do
    let block := i + tl.arange 0 1024
    let mask := block.ltS ql
    let pos := nc + block
    tl.store (d.pos.ptr 0 + start + block) (← tl.convert .i32 .i64 pos) (some mask)

/-- The elements of `pos` trip `t` writes. -/
def posTrip (S Q t : Nat) : Set Nat :=
  {k | S + t * 1024 ≤ k ∧ k < S + t * 1024 + 1024 ∧ k < S + Q}

/-- The invariant after `t` trips: the positions below `1024 t` are written, nothing outside
`pos` changed since `m0`, and `pos` is still valid. -/
def posInv (d : Inst) (S Q C : Nat) (m0 : Mem) (t : Nat) (m : Mem) : Prop :=
  (∀ j : Nat, j < Q → j < t * 1024 → ∃ v : I64,
      m.read .i64 (d.pos.ptr ((S + j : Nat) : Int)).addr = some v ∧ v.toInt = ((C + j : Nat) : Int)) ∧
    (∀ z ∉ d.pos.bytes, m z = m0 z) ∧ d.pos.Valid m

/-- One trip of the positions loop. -/
theorem safe_posTrip {d : Inst} (E : Env) {R : Set Int} (S Q C : Nat)
    (hSQ : S + Q ≤ d.pos.len) (hQ : Q ≤ 2 ^ 22) (hC : C ≤ 2 ^ 22)
    (nc start ql : I32) (hnc : nc.toInt = C) (hst : start.toInt = S) (hql : ql.toInt = Q)
    (m0 : Mem) (t : Nat) (ht : t * 1024 < Q) (i : I32) (hi : i.toInt = ((t * 1024 : Nat) : Int))
    (m : Mem) (hI : posInv d S Q C m0 t m) :
    Safe E R (d.pos.elems (posTrip S Q t)) (fun _ m' => posInv d S Q C m0 (t + 1) m') m
      (do
        let block := i + tl.arange 0 1024
        let mask := block.ltS ql
        let pos := nc + block
        tl.store (d.pos.ptr 0 + start + block) (← tl.convert .i32 .i64 pos) (some mask)) := by
  obtain ⟨hQm, hfr0, hV⟩ := hI
  obtain ⟨r0, hr0⟩ : ∃ r0 : Nat, r0 = t * 1024 := ⟨_, rfl⟩
  rw [← hr0] at hi ht hQm
  have hlane : ∀ l : Idx [1024 - 0],
      ((i + tl.arange 0 1024).get l).toInt = ((r0 + l.1 : Nat) : Int) := by
    rintro ⟨l, ⟨⟩⟩
    exact IntV.I32.toInt_sadd_arange hi (by omega) l
  have hposl : ∀ l : Idx [1024 - 0],
      ((nc + (i + tl.arange 0 1024)).get l).toInt = ((C + (r0 + l.1) : Nat) : Int) := by
    intro l
    rw [Tensor.get_sadd]
    exact IntV.I32.add_nat hnc (hlane l) (by have := l.1.isLt; omega)
  let g : Idx [1024 - 0] → Nat := fun l => S + (r0 + l.1.val)
  let en : Idx [1024 - 0] → Bool := fun l => decide (r0 + l.1.val < Q)
  have hmask : ∀ l, ((i + tl.arange 0 1024).ltS ql).get l = en l := by
    intro l
    simp only [Tensor.get_ltS, IntV.lt_eq, hlane l, hql, en]
    norm_cast
  have hptr : ∀ l, en l = true →
      (d.pos.ptr 0 + start + (i + tl.arange 0 1024)).get l = d.pos.ptr (g l : Int) := by
    intro l _
    rw [Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset, Buf.ptr_offset, hlane l, hst, zero_add]
    simp only [g]; push_cast; ring_nf
  have hin : ∀ l, en l = true → g l < d.pos.len := by
    intro l hl
    simp only [en, decide_eq_true_eq] at hl
    show S + (r0 + l.1.val) < _
    omega
  have himg : Buf.img g en = posTrip S Q t := by
    ext j
    constructor
    · rintro ⟨l, hl, rfl⟩
      simp only [en, decide_eq_true_eq] at hl
      have := l.1.isLt
      exact ⟨by simp only [g]; omega, by simp only [g]; omega, by simp only [g]; omega⟩
    · rintro ⟨h1, h2, h3⟩
      refine ⟨(⟨j - S - r0, by omega⟩, ()), ?_, ?_⟩
      · simp only [en, decide_eq_true_eq]
        omega
      · simp only [g]; omega
  have hsub : ∀ k ∈ posTrip S Q t, k < d.pos.len := by
    rintro k ⟨-, -, h3⟩; omega
  refine safe_convert (fun l => ConvRel.exists_i32_i64 _) fun vals hvals => ?_
  refine safe_last ?_
  refine safe_store_buf d.pos (Buf.addrs_of_get d.pos hmask hptr) hin ?_ hV
    (by rw [himg]) fun m' hm' hfr hpres => safe_ret ?_
  · intro a b _ _ h
    rcases a with ⟨a, ⟨⟩⟩; rcases b with ⟨b, ⟨⟩⟩
    dsimp only [g] at h
    exact Prod.ext (Fin.ext (by dsimp only; omega)) rfl
  · rw [himg] at hfr
    refine ⟨fun j hj1 hj2 => ?_, fun z hz => ?_,
      Buf.valid_of_agree hV fun x hx => hpres x (hV.2.2.2 x hx)⟩
    · by_cases hlt : j < r0
      · obtain ⟨v, hv, hv0⟩ := hQm j hj1 hlt
        refine ⟨v, ?_, hv0⟩
        rw [← hv]
        refine Mem.read_congr fun x hx => hfr x ?_
        rintro ⟨k, ⟨hk1, -, -⟩, hxk⟩
        have hne : ((S + j : Nat) : Int) ≠ (k : Int) := by omega
        exact Set.disjoint_left.mp (d.pos.span_disjoint hne) hx hxk
      · have hl : j - r0 < 1024 - 0 := by omega
        let l : Idx [1024 - 0] := (⟨j - r0, hl⟩, ())
        have hgl : g l = S + j := by simp only [g, l]; omega
        have hel : en l = true := by simp only [en, l, decide_eq_true_eq]; omega
        refine ⟨vals.get l, ?_, ?_⟩
        · have := hm' l hel
          rwa [hgl] at this
        · rw [ConvRel.toInt_i32_i64 (hvals l)]
          have := hposl l
          simp only [l] at this
          rw [this]
          congr 1
          omega
    · have hz' : z ∉ d.pos.elems (posTrip S Q t) := fun h =>
        hz (Buf.elems_sub_bytes d.pos hsub h)
      rw [hfr z hz']
      exact hfr0 z hz

/-- **The positions loop is safe**: it reads nothing, writes `pos[S .. S + Q)`, leaves
`pos[S + j] = C + j` for every `j < Q`, and changes nothing outside `pos`. -/
theorem safe_posProg {d : Inst} (E : Env) {R : Set Int} (S Q C : Nat)
    (hSQ : S + Q ≤ d.pos.len) (hQ : Q ≤ 2 ^ 22) (hC : C ≤ 2 ^ 22)
    (nc start ql : I32) (hnc : nc.toInt = C) (hst : start.toInt = S) (hql : ql.toInt = Q)
    (m0 : Mem) (hV : d.pos.Valid m0) :
    Safe E R (d.pos.elems {k | S ≤ k ∧ k < S + Q})
      (fun _ m => (∀ j : Nat, j < Q → ∃ v : I64,
          m.read .i64 (d.pos.ptr ((S + j : Nat) : Int)).addr = some v ∧
          v.toInt = ((C + j : Nat) : Int)) ∧ ∀ z ∉ d.pos.bytes, m z = m0 z)
      m0 (posProg d nc start ql) := by
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) ql (IntV.ofNat 1024 : I32) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [toInt_1024]; norm_num
    · rw [hql, toInt_1024, maxInt32]; push_cast; omega
    · rw [toInt_0, maxInt32]; norm_num
    · rw [toInt_1024, maxInt32]; norm_num
  set n := tl.tripCount (IntV.ofInt 0 : I32).toInt ql.toInt (IntV.ofNat 1024 : I32).toInt
    with hn
  have hnv : n = (((Q : Int) + 1023) / 1024).toNat := by
    have e : (Q : Int) - 0 + 1024 - 1 = Q + 1023 := by ring
    simp only [hn, tl.tripCount, toInt_0, hql, toInt_1024, e]
  have htrip : ∀ t, t < n → t * 1024 < Q := by
    intro t ht
    rw [hnv] at ht
    omega
  have hend : Q ≤ n * 1024 := by
    rw [hnv]; omega
  unfold posProg
  refine safe_last ?_
  refine Safe.mono (W := iterW (fun t => d.pos.elems (posTrip S Q t)) n ∪ ∅) ?_
    (fun _ _ h => h) ?_
  · refine safe_forRange hrange (fun t _ m => posInv d S Q C m0 t m)
      (fun t => d.pos.elems (posTrip S Q t)) ?_ (Set.disjoint_empty _) ?_ ?_ ?_
    · intro t u htu _
      refine Buf.elems_disjoint d.pos ?_
      rw [Set.disjoint_left]
      rintro j ⟨h1, h2, -⟩ ⟨h3, -, -⟩
      omega
    · exact ⟨fun j _ hj => by omega, fun _ _ => rfl, hV⟩
    · intro t ht _ m₁ hI
      have ht' := htrip t ht
      have hi : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofNat 1024 : I32).toInt)
          : I32).toInt = ((t * 1024 : Nat) : Int) := by
        rw [toInt_0, toInt_1024]
        have h := IntV.I32.toInt_ofInt (x := (0 : Int) + (t : Int) * 1024) ⟨by omega, by omega⟩
        rw [h]; push_cast; ring
      exact safe_posTrip E S Q C hSQ hQ hC nc start ql hnc hst hql m0 t ht' _ hi m₁ hI
    · intro _ m₁ hI
      exact safe_ret ⟨fun j hj => hI.1 j hj (by omega), hI.2.1⟩
  · rintro z (⟨t, -, hz⟩ | hz)
    · refine Buf.elems_mono d.pos ?_ hz
      rintro j ⟨h1, -, h3⟩
      exact ⟨by omega, h3⟩
    · exact absurd hz (Set.notMem_empty z)

end Claims.PreparePosSeqLens
