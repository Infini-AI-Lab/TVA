import Claim
import TVA.Proof
import Proof.Walk

/-!
# `gelu_and_mul_sparse_bf16`: the proof

The model (`_gelu_and_mul_sparse_kernel`), the conditions (`Pre`) and the specification (`Post`)
are in `Claim.lean`. The route is `correct_of_safe`: one program (one row) of the grid at a time.

1. Footprints (`Proof.Geom`): program `t` reads row `t` of `x` (its `gate` and `up`, `R`) and
   writes row `t` of `output` (`W`); rows of distinct programs are disjoint, and `output` shares
   no byte with `x` (`hdisj`, `hRW`). The index arithmetic (`int64` row offsets, `int32` lanes),
   the mask and the pointer equalities give the `Buf.Addrs` facts of the three accesses.
2. The row's statistics (`Proof.Row*`, `Proof.Stats`): the sum along a tree of depth
   `log₂ BLOCK_SIZE` and the division by `d` (the mean), the centered lanes, the fusable sum of
   their squares and the division (the variance), `tl.sqrt`, and the `bfloat16` roundings of
   the mean, the standard deviation, its product with the multiplier and the cutoff: the
   kernel's cutoff is within `7/8·cutTol − 2^-131` of `cRef` (`stats_chain`). The variance stays
   in `tl.sqrt`'s domain because the row is not nearly constant (`RowOk`).
3. One lane (`Proof.LaneS`, `Proof.LaneG`, `Proof.LaneG2`, `Proof.LaneO`, `Proof.LaneAll`): the
   kernel's sparse gate against `max (g − cutoff) 0`; the GELU through `tl.sigmoid` (`tl.exp`
   and the division) against `geluRef` (`Proof.Gelu`: `geluRef x = x·σ(2k(x + a x³))`, which is
   `2`-Lipschitz on `[0, ∞)`); the product with `up` and the roundings; then from the kernel's
   cutoff to the exact one (`lane_comb`), and the exact zero below the band (`lane_ok`).
4. The walk (`Proof.Walk`, `safe_program`): each program is `Safe`, one `tl.*` operation at a
   time, and leaves row `t` of `output` as `Post` asks (`Qp`).
5. `main`: the launch is valid, the footprints are disjoint, and `correct_of_safe` composes the
   programs; a byte outside `output`'s written elements is in no program's footprint.
-/

namespace Claims.GeluAndMulSparse

open TVA

/-- The programs' results, together with the untouched bytes, give `Post`. -/
theorem hpost {d : Inst} (hd : d.ok) {init fin : Mem}
    (hloc : ∀ i : Fin (launch d).size,
      ∃ m, Qp d init i.val m ∧ ∀ x ∈ R d i ∪ W d i, fin x = m x)
    (hfr : ∀ x, (∀ i : Fin (launch d).size, x ∉ W d i) → fin x = init x) :
    Post d init fin := by
  refine ⟨fun t ht j hj => ?_, fun z hz => hfr z (not_mem_W hz)⟩
  have ht' : t < (launch d).size := by rw [launch_size]; exact ht
  obtain ⟨m, hQ, hagree⟩ := hloc ⟨t, ht'⟩
  obtain ⟨c, hc, hrest⟩ := hQ j hj
  refine ⟨c, ?_, hrest⟩
  rw [Mem.read_congr (m₂ := m) (fun x hx => hagree x (Or.inr (span_out_W hd ht' hj hx)))]
  exact hc

/-- **The claim.** -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (R d) (W d) (fun i m => Qp d init i.val m) (launch_valid hd)
    (fun i => safe_program hd hpre hP i) (fun i j hij => hdisj hd i j hij) (fun i => hRW hd i)
    (fun fin hloc hfr => hpost hd hloc hfr)

end Claims.GeluAndMulSparse
