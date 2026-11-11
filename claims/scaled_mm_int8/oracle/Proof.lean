import Proof.Body
import Proof.Dots
import Proof.Loop
import Proof.Masks
import Proof.Zero

/-!
# `scaled_mm_int8`

Proof of `main : claim.Holds`.

1. `Proof.Basic`, `Proof.Foot`: the launch (one program per output tile), and the footprints of
   program `p`: it reads its tiles of `A`, `B`, the scales and the bias (`Rfp`), writes its tile of
   `C` (`Wfp`), and leaves `OutSpec` in every element of that tile (`Qfp`).
2. `Proof.Dots`, `Proof.Zero`, `Proof.Masks`, `Proof.Loop`: the reduction loop. The `int32`
   accumulator never wraps (`Device.DotI8` applies), so after `t` trips it is the exact partial
   dot product; the invariant holds at trip `0` and each trip (`loop_safe`) keeps it.
3. `Proof.Value`, `Proof.Tail`, `Proof.Body`: the epilogue, a `float32` chain (convert, two scale
   products, optional bias) and one rounding to the output dtype, stays within `tol` and
   `outTol` of the real value; `safe_program` walks the whole body of one program.
4. Here, `main`, by `correct_of_safe`: each program is safe, a program's writes are disjoint from
   every other program's reads and writes, and the tiles cover `C`, so `Post` holds.
-/

namespace Claims.ScaledMmInt8

open TVA

/-- **The claim.** Each program is safe on its own tile (`safe_program`); the tiles' footprints
are disjoint, and together they give `Post`. -/
theorem main : claim.Holds := by
  intro P hP d hd init hpre
  show Correct ((launch d).env P) (model d) (launch d) init (Post d init)
  refine correct_of_safe (fun i => Rfp d i) (fun i => Wfp d i) (fun i m => Qfp d init i m)
    (launch_valid hd) (fun i => ?_) ?_ ?_ ?_
  · -- each program is safe
    exact safe_program P d hd init hpre hP.dotI8 (hP.dotTotal _) ((launch d).ctx i) i
      (launch_pid d i) (lt_of_lt_of_eq i.isLt (launch_size d))
  · -- no program writes another program's read or write footprint
    intro i j hij
    have hij' : (i : Nat) ≠ (j : Nat) := fun h => hij (Fin.ext h)
    simp only [Rfp, Wfp]
    refine Disjoint.union_right (Disjoint.union_right (Disjoint.union_right
      (Disjoint.union_right (Disjoint.union_right ?_ ?_) ?_) ?_) ?_) ?_
    · exact hd.c_a.mono (d.c.elems_mono (cTile_sub d i)) (d.a.elems_mono (aTile_sub d j))
    · exact hd.c_b.mono (d.c.elems_mono (cTile_sub d i)) (d.b.elems_mono (bTile_sub d j))
    · exact hd.c_sa.mono (d.c.elems_mono (cTile_sub d i)) (d.sa.elems_mono (saTile_sub hd j))
    · exact hd.c_sb.mono (d.c.elems_mono (cTile_sub d i)) (d.sb.elems_mono (sbTile_sub hd j))
    · by_cases hb : d.hasBias = true
      · exact (hd.c_bias hb).mono (d.c.elems_mono (cTile_sub d i))
          (d.bias.elems_mono (biasTile_sub d j))
      · have hbt : biasTile d j = ∅ := by unfold biasTile; rw [ite_eq_right hb]
        have hbe : d.bias.elems (biasTile d j) = ∅ := by rw [hbt]; simp [Buf.elems]
        rw [hbe]
        simp
    · exact Buf.elems_disjoint d.c (cTile_disjoint hd hij')
  · -- no program reads what it writes
    intro i
    simp only [Rfp, Wfp]
    refine Disjoint.union_left (Disjoint.union_left (Disjoint.union_left
      (Disjoint.union_left ?_ ?_) ?_) ?_) ?_
    · exact (hd.c_a.mono (d.c.elems_mono (cTile_sub d i)) (d.a.elems_mono (aTile_sub d i))).symm
    · exact (hd.c_b.mono (d.c.elems_mono (cTile_sub d i)) (d.b.elems_mono (bTile_sub d i))).symm
    · exact (hd.c_sa.mono (d.c.elems_mono (cTile_sub d i)) (d.sa.elems_mono (saTile_sub hd i))).symm
    · exact (hd.c_sb.mono (d.c.elems_mono (cTile_sub d i)) (d.sb.elems_mono (sbTile_sub hd i))).symm
    · by_cases hb : d.hasBias = true
      · exact ((hd.c_bias hb).mono (d.c.elems_mono (cTile_sub d i))
          (d.bias.elems_mono (biasTile_sub d i))).symm
      · have hbt : biasTile d i = ∅ := by unfold biasTile; rw [ite_eq_right hb]
        have hbe : d.bias.elems (biasTile d i) = ∅ := by rw [hbt]; simp [Buf.elems]
        rw [hbe]
        simp
  · -- the programs' results give `Post`
    intro fin hloc hfr
    refine ⟨fun i j hi hj => ?_, fun z hz => hfr z fun i hzi => hz ?_⟩
    · obtain ⟨hp, hpm, hpn⟩ := cover hd hi hj
      obtain ⟨m, hQ, hagree⟩ := hloc ⟨i / d.BM * gridN d + j / d.BN, (launch_size d).symm ▸ hp⟩
      have hi' : i % d.BM < d.BM := Nat.mod_lt _ (BM_pos hd)
      have hj' : j % d.BN < d.BN := Nat.mod_lt _ (BN_pos hd)
      have hpmi : pm d (i / d.BM * gridN d + j / d.BN) * d.BM + i % d.BM = i := by
        rw [hpm, Nat.mul_comm]
        exact Nat.div_add_mod i d.BM
      have hpni : pn d (i / d.BM * gridN d + j / d.BN) * d.BN + j % d.BN = j := by
        rw [hpn, Nat.mul_comm]
        exact Nat.div_add_mod j d.BN
      obtain ⟨c, hread, hLane⟩ := hQ (i % d.BM) (j % d.BN) hi' hj'
        (by rw [hpmi]; exact hi) (by rw [hpni]; exact hj)
      refine ⟨c, ?_, ?_⟩
      · have haddr : i * d.stride_cm + j * d.stride_cn =
            (pm d (i / d.BM * gridN d + j / d.BN) * d.BM + i % d.BM) * d.stride_cm +
              (pn d (i / d.BM * gridN d + j / d.BN) * d.BN + j % d.BN) * d.stride_cn := by
          rw [hpmi, hpni]
        rw [haddr]
        refine (Mem.read_congr fun x hx => hagree x (Or.inr ?_)).trans hread
        exact ⟨_, ⟨i % d.BM, j % d.BN, hi', hj', by rw [hpmi]; exact hi,
          by rw [hpni]; exact hj, rfl⟩, hx⟩
      · rw [hpmi, hpni] at hLane
        exact hLane
    · exact d.c.elems_mono (cTile_sub d i) hzi

end Claims.ScaledMmInt8
