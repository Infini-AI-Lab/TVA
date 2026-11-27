import Proof.Pos
import Proof.Pad

/-!
# `prepare_pos_seq_lens_i32`: a request's program

Program `r < N` loads `idx_mapping[r]`, the slot's `num_computed_tokens`, `query_start_loc[r]`
and `query_start_loc[r + 1]`, stores `seq_lens[r]`, then runs the positions loop
(`Proof/Pos.lean`).
-/

namespace Claims.PreparePosSeqLens

open TVA

/-- The program of a request (source lines 73-87), its `program_id` `req_id`. -/
def reqProg (d : Inst) (req_id : I32) : Prog Unit := do
  let req_state_idx ← tl.load1 (d.idx_mapping.ptr 0 + req_id)
  let num_computed_tokens ← tl.load1 (d.num_computed_tokens.ptr 0 + req_state_idx)
  let start ← tl.load1 (d.query_start_loc.ptr 0 + req_id)
  let end_ ← tl.load1 (d.query_start_loc.ptr 0 + req_id + (1 : I32))
  let query_len := end_ - start
  let seq_len := num_computed_tokens + query_len
  tl.store1 (d.seq_lens.ptr 0 + req_id) seq_len
  posProg d num_computed_tokens start query_len

/-- The model's program, by cases on the branch of line 66. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      if (tl.program_id c 0).eq (tl.num_programs c 0 - (1 : I32)) then
        padProg d (tl.num_programs c 0 - (1 : I32))
      else reqProg d (tl.program_id c 0) := rfl

/-- `query_start_loc` is nondecreasing on `[0, N]`. -/
theorem qsl_le {d : Inst} {init : Mem} (hpre : Pre d init) :
    ∀ a b, a ≤ b → b ≤ d.N → qsl d init a ≤ qsl d init b := by
  intro a b hab hb
  induction b with
  | zero => obtain rfl : a = 0 := by omega
            exact le_refl _
  | succ b ih =>
    rcases Nat.eq_or_lt_of_le hab with h | h
    · rw [h]
    · exact le_trans (ih (by omega) (by omega)) (hpre.2.2.2.2.2.2.1 b (by omega))

/-- The value read at element `i` of an `int32` buffer is `rd32`. -/
theorem rd32_of_read {b : Buf .i32} {m : Mem} {i : Int} {v : I32}
    (h : m.read .i32 (b.ptr i).addr = some v) : rd32 b m i = v.toInt := by
  simp only [rd32, h]

/-- The value read at element `i` of an `int64` buffer is `rd64`. -/
theorem rd64_of_read {b : Buf .i64} {m : Mem} {i : Int} {v : I64}
    (h : m.read .i64 (b.ptr i).addr = some v) : rd64 b m i = v.toInt := by
  simp only [rd64, h]

/-- **A request's program is safe**: it reads its entries of the inputs, writes `seq_lens[r]`
and its tokens of `pos`, and leaves `QReq`. -/
theorem safe_req {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) (r : Nat)
    (hr : r < d.N) :
    Safe E (R d init r) (W d init r) (fun _ m => Q d init r m) init (reqProg d (IntV.ofNat r)) := by
  obtain ⟨hVp, hVs, hVi, hVq, hVc, hq0, -, hqN, hq22, hslot, hctx⟩ := id hpre
  have hN := hd.grid_lt
  have hr31 : r < 2 ^ 31 := by omega
  have hr1 : r + 1 < 2 ^ 31 := by omega
  simp only [R, W, Q, hr, ↓reduceIte]
  -- the bounds of the request's window
  have hqa : 0 ≤ qsl d init r := le_trans hq0 (qsl_le hpre 0 r (by omega) (by omega))
  have hqb : qsl d init r ≤ qsl d init (r + 1) := qsl_le hpre r (r + 1) (by omega) (by omega)
  have hqc : qsl d init (r + 1) ≤ qsl d init d.N := qsl_le hpre (r + 1) d.N (by omega) (by omega)
  obtain ⟨hs0, hs1⟩ := hslot r hr
  obtain ⟨hc0, hc1⟩ := hctx r hr
  -- the slot, the computed tokens and the window as natural numbers
  obtain ⟨sl, hsl⟩ : ∃ sl : Nat, (sl : Int) = slot d init r := ⟨_, Int.toNat_of_nonneg hs0⟩
  obtain ⟨S, hS⟩ : ∃ S : Nat, (S : Int) = qsl d init r := ⟨_, Int.toNat_of_nonneg hqa⟩
  obtain ⟨Qn, hQn⟩ : ∃ Q : Nat, (Q : Int) = qlen d init r :=
    ⟨_, Int.toNat_of_nonneg (by simp only [qlen]; omega)⟩
  obtain ⟨C, hC⟩ : ∃ C : Nat, (C : Int) = computed d init r := ⟨_, Int.toNat_of_nonneg hc0⟩
  have hqlen : qlen d init r = qsl d init (r + 1) - qsl d init r := rfl
  have hslen : sl < d.num_computed_tokens.len := by omega
  -- line 73
  have hil : r < d.idx_mapping.len := by have := hd.idx_len; omega
  obtain ⟨v0, hv0⟩ := Buf.read_some d.idx_mapping hVi hil
  refine safe_load1 (v0 := v0) (by rw [ptr_zero_add _ r hr31]; exact hv0) ?_ ?_
  · intro x hx
    rw [ptr_zero_add _ r hr31] at hx
    exact Or.inl (Or.inl (Buf.span_sub_elems _ (Set.mem_singleton r) hx))
  have hv0v : v0.toInt = (sl : Int) := by rw [hsl, slot, rd64_of_read hv0]
  -- line 74
  obtain ⟨v1, hv1⟩ := Buf.read_some d.num_computed_tokens hVc hslen
  have hp1 : (d.num_computed_tokens.ptr 0 + v0).addr = (d.num_computed_tokens.ptr (sl : Int)).addr := by
    rw [ptr_zero_addI, hv0v]
  refine safe_load1 (v0 := v1) (by rw [hp1]; exact hv1) ?_ ?_
  · intro x hx
    rw [hp1] at hx
    exact Or.inr (Buf.span_sub_elems (S := {j : Nat | (j : Int) = slot d init r}) _ hsl hx)
  have hv1v : v1.toInt = (C : Int) := by
    rw [hC, computed, ← hsl, rd32_of_read hv1]
  -- line 76
  have hql : r + 1 < d.query_start_loc.len := by have := hd.qsl_len; omega
  obtain ⟨v2, hv2⟩ := Buf.read_some d.query_start_loc hVq (i := r) (by omega)
  refine safe_load1 (v0 := v2) (by rw [ptr_zero_add _ r hr31]; exact hv2) ?_ ?_
  · intro x hx
    rw [ptr_zero_add _ r hr31] at hx
    exact Or.inl (Or.inr (Buf.span_sub_elems _ (Set.mem_insert r _) hx))
  have hv2v : v2.toInt = (S : Int) := by rw [hS, qsl, rd32_of_read hv2]
  -- line 77
  obtain ⟨v3, hv3⟩ := Buf.read_some d.query_start_loc hVq hql
  have hp3 : (d.query_start_loc.ptr 0 + (IntV.ofNat r : I32) + (1 : I32)).addr =
      (d.query_start_loc.ptr ((r + 1 : Nat) : Int)).addr := by
    rw [Ptr.add_int, Ptr.add_int, toInt_1, IntV.I32.toInt_ofNat hr31, Buf.ptr_offset,
      Buf.ptr_offset, zero_add]
    push_cast; rfl
  refine safe_load1 (v0 := v3) (by rw [hp3]; exact hv3) ?_ ?_
  · intro x hx
    rw [hp3] at hx
    exact Or.inl (Or.inr (Buf.span_sub_elems (S := {r, r + 1}) _ (by simp) hx))
  have hv3v : v3.toInt = ((S + Qn : Nat) : Int) := by
    have h3 : qsl d init (r + 1) = v3.toInt := rd32_of_read hv3
    push_cast
    omega
  -- line 78
  have hqv : (v3 - v2).toInt = (Qn : Int) := by
    have := IntV.I32.sub_nat hv3v hv2v (by omega) (by omega)
    rw [this, Nat.add_sub_cancel_left]
  -- line 80
  have hsv : (v1 + (v3 - v2)).toInt = ((C + Qn : Nat) : Int) :=
    IntV.I32.add_nat hv1v hqv (by omega)
  -- line 81
  have hpS : (d.seq_lens.ptr 0 + (IntV.ofNat r : I32)).addr = (d.seq_lens.ptr (r : Int)).addr :=
    ptr_zero_add _ r hr31
  have hsr : r < d.seq_lens.len := by have := hd.sl_N; omega
  refine safe_store1 (by rw [hpS]; exact Buf.writable _ hVs hsr) ?_ fun m' hm' hfr' => ?_
  · intro x hx
    rw [hpS] at hx
    exact Or.inl (Buf.span_sub_elems _ (Set.mem_singleton r) hx)
  rw [hpS] at hm' hfr'
  -- the positions loop
  have hSQ : S + Qn ≤ d.pos.len := by omega
  have hspan_pos : Disjoint (Mem.span .i32 (d.seq_lens.ptr (r : Int)).addr) d.pos.bytes :=
    Disjoint.mono_left (d.seq_lens.span_sub hsr) hd.disj.1.symm
  have hVp' : d.pos.Valid m' := Buf.valid_of_agree hVp fun x hx => by
    rw [hfr' x (Set.disjoint_right.mp hspan_pos hx)]; exact hVp.2.2.2 x hx
  have hwin : posElems d init r = {k | S ≤ k ∧ k < S + Qn} := by
    ext k
    simp only [posElems, Set.mem_ofPred_eq]
    rw [hqlen] at hQn
    omega
  refine Safe.mono (safe_posProg E S Qn C hSQ (by omega) (by omega) v1 v2 (v3 - v2) hv1v hv2v hqv m'
    hVp') ?_ ?_
  · rintro _ m ⟨hpos, hfr⟩
    refine ⟨⟨v1 + (v3 - v2), ?_, ?_⟩, fun j hj => ?_⟩
    · rw [← hm']
      exact Mem.read_congr fun x hx => hfr x (Set.disjoint_left.mp hspan_pos hx)
    · rw [hsv, ← hC, ← hQn]; push_cast; ring
    · have hjQ : j < Qn := by rw [← hQn] at hj; exact_mod_cast hj
      obtain ⟨v, hv, hvv⟩ := hpos j hjQ
      refine ⟨v, ?_, ?_⟩
      · rw [← hS]; push_cast at hv; exact hv
      · rw [hvv, ← hC]; push_cast; ring
  · intro z hz
    rw [← hwin] at hz
    refine ⟨Or.inr hz, fun hzs => ?_⟩
    have hzb : z ∈ d.pos.bytes := by
      refine Buf.elems_sub_bytes d.pos ?_ hz
      intro k hk
      rw [hwin] at hk
      exact lt_of_lt_of_le hk.2 hSQ
    rw [hpS] at hzs
    exact Set.disjoint_left.mp hspan_pos hzs hzb

end Claims.PreparePosSeqLens
