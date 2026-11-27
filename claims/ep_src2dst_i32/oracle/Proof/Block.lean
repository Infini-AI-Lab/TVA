import Proof.Search

/-!
# `ep_src2dst_i32`: one program

Lines 53–59 of program `p`: the block load of `reorder_ids` (`safe_load_block`), the packed rows,
and the store to the elements `reorder_ids[j]` (distinct: `Pre`); then the whole program, the
search first (`safe_searchIte`).
-/

namespace Claims.EpSrc2Dst

open TVA Block

variable {E : Env}

/-- **Lines 53–59** of program `p`, with `first_nonnegative = f` found by the search. -/
theorem safe_rest {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat}
    (hp : p < G d) (c : Ctx) (hc : c.pid 0 = p) (f : I32) (hf : Found d init f) :
    Safe E (R d p) (W d init p) (fun _ m => Q d init p m) init (restProg d c f) := by
  obtain ⟨hlt, hfit⟩ := prog_bounds hd hp
  obtain ⟨k, hfk, hkn, hk⟩ := hf
  have hn := hd.n_lt
  obtain ⟨hvs, hvr, hvo, hpfx, hrng, hinj⟩ := hpre
  have hnn : nneg d init = k := nneg_eq hkn hk
  simp only [restProg, tl.program_id, hc]
  refine safe_load_block d.reorder hfit hn hd.reorder_len hvr (fun z hz => Or.inr hz)
    (other := none) fun v hv _ => ?_
  -- the lanes of `src_id`: lane `l` holds `reorder_ids[p * 512 + l]`
  let g : Idx [512 - 0] → Nat := fun i => (rid d init (p * 512 + i.1.val)).toNat
  let en : Idx [512 - 0] → Bool := fun i => decide (p * 512 + i.1.val < d.n)
  have hen : ∀ i : Idx [512 - 0], en i = true ↔ p * 512 + i.1.val < d.n := fun i => by
    simp only [en, decide_eq_true_eq]
  have hvi : ∀ i : Idx [512 - 0], en i = true → (v.get i).toInt = rid d init (p * 512 + i.1.val) := by
    rintro ⟨l, ⟨⟩⟩ he
    exact (rid_of_read (hv l ((hen _).mp he))).symm
  have hg : ∀ i : Idx [512 - 0], en i = true → ((g i : Nat) : Int) = rid d init (p * 512 + i.1.val) :=
    fun i he => Int.toNat_of_nonneg (hrng _ ((hen i).mp he)).1
  have hm : ∀ i : Idx [512 - 0], (Block.mask p 512 d.n).get i = en i := by
    rintro ⟨l, ⟨⟩⟩
    exact mask_lane hfit hn l
  have hA : d.src2dst.Addrs (tl.addrs (d.src2dst.ptr 0 + v) (some (Block.mask p 512 d.n))) g en := by
    refine Buf.addrs_of_get d.src2dst hm fun i he => ?_
    rw [Tensor.get_ptrAdd, Buf.ptr_offset, zero_add, hvi i he, hg i he]
  have hin : ∀ i, en i = true → g i < d.src2dst.len := by
    intro i he
    have h1 := hg i he
    have h2 := (hrng _ ((hen i).mp he)).2
    omega
  have hinj' : ∀ i j, en i = true → en j = true → g i = g j → i = j := by
    intro i j hi hj h
    have h1 := hg _ hi
    have h2 := hg _ hj
    have e := hinj _ _ ((hen i).mp hi) ((hen j).mp hj) (by rw [← h1, ← h2, h])
    obtain ⟨⟨i, hi'⟩, ⟨⟩⟩ := i
    obtain ⟨⟨j, hj'⟩, ⟨⟩⟩ := j
    simp only at e
    have : i = j := by omega
    subst this
    rfl
  have himg : Buf.img g en ⊆ outE d init p := by
    intro x hx
    obtain ⟨i, he, hx⟩ := hx
    have hi1 : i.1.val < 512 := i.1.isLt
    have he' := (hen i).mp he
    refine ⟨p * 512 + i.1.val, ⟨Nat.le_add_right _ _, Nat.add_lt_add_left hi1 _, he'⟩, ?_⟩
    rw [← hx, hg i he]
  refine safe_last ?_
  refine safe_store_buf d.src2dst hA hin hinj' hvo (d.src2dst.elems_mono himg)
    fun m' hm' _ _ => safe_ret ?_
  intro j ⟨hj1, hj2, hj3⟩
  have hjB : j - p * 512 < 512 - 0 := by omega
  let i : Idx [512 - 0] := (⟨j - p * 512, hjB⟩, ())
  have hji : p * 512 + i.1.val = j := by simp only [i]; omega
  have hei : en i = true := (hen i).mpr (by rw [hji]; exact hj3)
  have h1 := hm' i hei
  rw [hg i hei, hji] at h1
  rw [h1]
  congr 1
  -- the packed row of lane `i`
  have hoff : ((IntV.ofNat p * (IntV.ofNat 512 : I32) + tl.arange 0 512).get i).toInt = (j : Int) := by
    have := lane_toInt hfit i.1
    rw [← hji]
    exact this
  simp only [tl.where, Tensor.get_where, Tensor.get_le, Tensor.get_full, Tensor.get_sub,
    IntV.le_eq, hoff, hfk, pval, hnn]
  have hs := hk j hj3
  by_cases hjk : k ≤ j
  · rw [ite_eq_left (by simpa using hjk), ite_eq_right (by rw [hs]; omega)]
    have e : (IntV.ofNat p * (IntV.ofNat 512 : I32) + tl.arange 0 512).get i - f =
        (IntV.ofInt ((j : Int) - (k : Int)) : I32) := by
      apply IntV.ext_toInt
      rw [IntV.toInt_sub hoff hfk (IntV.I32.sfits (by omega) (by omega)),
        IntV.I32.toInt_ofInt ⟨by omega, by omega⟩]
    exact e
  · rw [ite_eq_right (by simpa using hjk), ite_eq_left (by rw [hs]; omega)]

/-- **One program** of the grid, `program_id(0) = p`, is safe from `init`: it reads the sorted ids
and its block of `reorder_ids`, writes the elements `reorder_ids[j]` of its block, and ends with
`Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {p : Nat}
    (hp : p < G d) (c : Ctx) (hc : c.pid 0 = p) :
    Safe E (R d p) (W d init p) (fun _ m => Q d init p m) init ((model d).body c) := by
  obtain ⟨hlt, -⟩ := prog_bounds hd hp
  rw [body_eq]
  have hR : d.sorted.elems {j | j < d.n} ⊆ R d p := fun z hz => Or.inl hz
  obtain ⟨v, hv⟩ := Buf.read_some d.sorted hpre.1 (i := 0) (by have := hd.sorted_len; omega)
  refine safe_load1 (v0 := v) (by simpa using hv) ?_ ?_
  · exact (d.sorted.span_sub_elems (show (0 : Nat) ∈ {j | j < d.n} by
      simp only [Set.mem_ofPred_eq]; omega)).trans hR |> fun h => by simpa using h
  simp only [Prog.bind_eq]
  refine Safe.bind_ro ((safe_searchIte hd hpre (by omega) hR v (sid_of_read hv)).mono ?_ le_rfl)
  rintro f m' ⟨rfl, hf⟩
  exact safe_rest hd hpre hp c hc f hf

end Claims.EpSrc2Dst
