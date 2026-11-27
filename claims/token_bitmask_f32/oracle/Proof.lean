import Proof.Basic
import Proof.Run

/-!
# `token_bitmask_f32`: `apply_token_bitmask_inplace_kernel`

Program `p` serves the work items `p, p + NUM_SMS, …` below `num_rows * cdiv(vocab_size, 4096)`;
item `w` is row `w / nb`, block `w % nb`, and sets to `-∞` the logits of its block whose bit is
`0`. The proof has four steps:

1. `Proof.Tail`: the block of one item (the load of its words, the shifts, the masked store);
2. `Proof.Trip`: one trip of the loop (the row, the block offset, the index);
3. `Proof.Run`: the loop (`safe_program`), whose trips serve exactly the program's items;
4. `main`: by `correct_of_safe`. Programs write the cells of distinct items (`progElems_disjoint`:
   a cell determines its row and token, `cell_inj`), only in the logits, and read only the bitmask
   and the indices; every cell of `Post` is written by the program `item % NUM_SMS`.
-/

namespace Claims.TokenBitmask

open TVA

/-- A program's writes are logits bytes. -/
theorem W_sub_bytes {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (p : Nat) :
    W d init p ⊆ d.logits.bytes :=
  Buf.elems_sub_bytes d.logits (progElems_lt hd hpre)

/-- A program writes nothing any program reads. -/
theorem W_disj_R {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (p : Nat) :
    Disjoint (W d init p) (R d) := by
  refine Disjoint.mono_left (W_sub_bytes hd hpre p) ?_
  refine Disjoint.union_right hd.disj_bitmask ?_
  cases h : d.hasIdx
  · simp only [Bool.false_eq_true, ↓reduceIte]
    exact Set.disjoint_empty _
  · simp only [↓reduceIte]
    exact hd.disj_idx h

/-- **The claim.** Each program is safe (`safe_program`), the footprints are disjoint, and the
programs' cells give `Post`. -/
theorem main : claim.Holds := by
  intro P _ d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun _ => R d) (fun i => W d init (i : Nat))
    (fun i m => Q d init (i : Nat) m) (launch_valid hd) (fun i => safe_program hd hpre _ i)
    (fun i j hij => Disjoint.union_right (W_disj_R hd hpre i)
      (Buf.elems_disjoint d.logits (progElems_disjoint hd hpre (fun h => hij (Fin.ext h)))))
    (fun i => (W_disj_R hd hpre i).symm) ?_
  intro fin hloc hfr
  refine ⟨?_, ?_⟩
  · intro k hk c hc hb
    have hM : Mk d init k c := ⟨hk, hc, hb⟩
    have hS : 0 < d.NUM_SMS := hd.sms_pos.resolve_right (by
      have := itemOf_lt hk hc; omega)
    have hpS : itemOf d k c % d.NUM_SMS < d.NUM_SMS := Nat.mod_lt _ hS
    have hpT : itemOf d k c % d.NUM_SMS < items d :=
      lt_of_le_of_lt (Nat.mod_le _ _) (itemOf_lt hk hc)
    have hpg : itemOf d k c % d.NUM_SMS < d.grid :=
      lt_of_lt_of_le (lt_min hpS hpT) hd.grid_cover
    obtain ⟨m, hQ, hag⟩ := hloc ⟨itemOf d k c % d.NUM_SMS, lt_of_lt_of_eq hpg (launch_size d).symm⟩
    obtain ⟨h0, -⟩ := cell_bounds hd hpre hM
    have hmem : (cell d init k c).toNat ∈ progElems d init (itemOf d k c % d.NUM_SMS) :=
      ⟨k, c, hM, rfl, Int.toNat_of_nonneg h0⟩
    have hsub := Buf.span_sub_elems d.logits hmem
    rw [Int.toNat_of_nonneg h0] at hsub
    have hread : fin.read .f32 (d.logits.ptr (cell d init k c)).addr =
        m.read .f32 (d.logits.ptr (cell d init k c)).addr :=
      Mem.read_congr fun x hx => hag x (Or.inr (hsub hx))
    rw [show rowOf d init k * (d.logits_strides : Int) + (c : Int) = cell d init k c from rfl, hread]
    exact hQ k c hM rfl
  · intro z hz
    refine hfr z fun i hzi => hz (Buf.elems_mono d.logits ?_ hzi)
    rintro e ⟨k, c, hM, -, he⟩
    exact ⟨k, hM.1, c, hM.2.1, hM.2.2, he⟩

end Claims.TokenBitmask
