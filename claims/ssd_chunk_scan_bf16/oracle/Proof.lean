import Claim
import Proof.Main

/-!
# `ssd_chunk_scan_bf16`: the proof of the claim

Program `(pid0, k, h)` of the grid owns one `64 × 64` tile of `out`: rows of the `k`-th logical chunk
(the part of a physical chunk that belongs to one request), channels of head `h`. It only reads the
inputs, so the proof is the independent-programs route (`correct_of_safe`):

1. `Proof.Defs`, `Proof.Geom`: the program index, the logical chunks of the launch (the host's
   metadata) and a tile's footprints `Wt`, post-condition `Qt`;
2. `Proof.NumState`, `Proof.NumLoop`, `Proof.NumFinal`: the numeric facts behind `tol`: the scale
   of the entering state, the state term, one weight, the accumulation loop, the epilogue;
3. `Proof.Core` and `Proof.Phase*`: the kernel's core in three phases (state term, loop,
   epilogue), each walked line by line, and `Proof.CoreSafe` joining them;
4. `Proof.Noinit`, `Proof.Init`: the set-up of each variant (coordinates, pointers, metadata loads,
   the two runtime `if`s), ending in `core_safe`;
5. `Proof.Index`, `Proof.Footprint`, `Proof.Post`, `Proof.Main`: distinct programs write disjoint
   tiles, each element of `out` is in its owner's tile, the launch is valid.
-/

namespace Claims.SsdChunkScan

open TVA

/-- **The claim.** Each program is safe on its own tile (`init_tile`, `noinit_tile`); the tiles are
disjoint and no program reads `out`; the programs' results give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  have dev : Dev ((launch d).env P) :=
    ⟨hP.mathTotal _ _, hP.dotTotal _, hP.expF32, hP.dotBF16, rfl⟩
  refine correct_of_safe (fun _ => R d) (fun i => W d i) (fun i m => Q d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · have hi : (i : Nat) < gx d * d.nmeta * d.H := i.isLt
    by_cases hN : d.hasInit = true
    · exact init_tile hd hpre dev hN hi rfl rfl rfl rfl
    · exact noinit_tile hd hpre dev (by simpa using hN) hi rfl rfl rfl rfl
  · intro i j hij
    have hi : (i : Nat) < gx d * d.nmeta * d.H := i.isLt
    have hj : (j : Nat) < gx d * d.nmeta * d.H := j.isLt
    refine Disjoint.union_right ?_ (W_disjoint hd hi hj (fun h' => hij (Fin.ext h')))
    exact disjoint_compl_right.mono_left (W_sub_bytes hd hi)
  · intro i
    exact (disjoint_compl_left.mono_right (W_sub_bytes hd i.isLt))
  · intro fin hloc hfr
    refine ⟨fun t h p ht hh hp => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
    · obtain ⟨k, pm, pn, ii, jj, hk, hpm, hpn, hii, hjj, hri, hcj, htk, hpk⟩ := owner hd p ht hp
      have hi := progOf_lt hd hk hh hpm hpn
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨progOf d k pm pn h, hi⟩
      have hkk := kOf_progOf hd hk hpm hpn (h := h)
      have hpp := pmOf_progOf hd hpm hpn (k := k) (h := h)
      have hnn := pnOf_progOf hd hpm hpn (k := k) (h := h)
      have hhh := hOf_progOf hd hk hpm hpn (h := h)
      unfold Q at hQ
      rw [hkk, hpp, hnn, hhh] at hQ
      obtain ⟨v, hv, hvals⟩ := hQ ii jj hii hjj hri hcj
      rw [htk, hpk] at hv hvals
      refine ⟨v, ?_, hvals⟩
      unfold rdB at hv ⊢
      rw [Mem.read_congr (fun x hx => hagree x (Or.inr ?_))]
      · exact hv
      · -- the element's bytes lie in the tile
        have hmem : outIdx d t h p ∈ {x | Tile2.ElemAt (lc d k * d.cs * (d.H * d.P) + h * d.P)
            (d.H * d.P) 1 (fun ii : Fin BM => rowOf d k pm ii) (fun jj : Fin BN => colOf pn jj)
            (lend d k) d.P x} := by
          refine ⟨⟨ii, hii⟩, ⟨jj, hjj⟩, hri, hcj, ?_⟩
          rw [← htk, ← hpk]; unfold outIdx tokOf; simp only; ring
        have hW : W d (progOf d k pm pn h) = d.out.elems {x | Tile2.ElemAt
            (lc d k * d.cs * (d.H * d.P) + h * d.P) (d.H * d.P) 1
            (fun ii : Fin BM => rowOf d k pm ii) (fun jj : Fin BN => colOf pn jj) (lend d k) d.P x} := by
          unfold W Wt Buf.tileAt; rw [hkk, hpp, hnn, hhh]
        rw [hW]
        exact d.out.span_sub_elems hmem hx
    · have hi : (i : Nat) < gx d * d.nmeta * d.H := i.isLt
      exact W_sub hd hi hzi

end Claims.SsdChunkScan
