import Claim
import TVA.Proof
import Proof.Defs
import Proof.Prog
import Proof.Merge
import Proof.Copy

/-!
# One program is safe

Program `(t, h)` reads the two lse scalars and the two `bfloat16` rows it needs, writes its row of
`output` (and its element of `output_lse`), and leaves the merged or copied values there.
-/

namespace Claims.MergeAttnStates

open TVA

/-- **One program.** Program `i` of the grid, run alone from `init`, is safe with the read
footprint `Rf`, the write footprint `Wf`, and leaves `Qm` in memory. -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (Rf d (pidT d i) (pidH d i)) (Wf d (pidT d i) (pidH d i))
      (fun _ m => Qm d init (pidT d i) (pidH d i) m) init ((model d).body ((launch d).ctx i)) := by
  rw [body_split]
  dsimp only
  by_cases h : pidT d i < d.ptwc
  · have hq : (fun (_ : Unit) m => Qm d init (pidT d i) (pidH d i) m)
        = (fun (_ : Unit) m => MergeOk d init m (pidT d i) (pidH d i)) := by
      funext _ m
      simp only [Qm, ite_eq_left h]
    rw [hq, prefix_mask_eq d hd i]
    simp only [decide_eq_true h, Bool.not_true, Bool.false_eq_true, ite_false]
    exact safe_merge hP hd hpre i h
  · have hq : (fun (_ : Unit) m => Qm d init (pidT d i) (pidH d i) m)
        = (fun (_ : Unit) m => CopyOk d init m (pidT d i) (pidH d i)) := by
      funext _ m
      simp only [Qm, ite_eq_right h]
    rw [hq, prefix_mask_eq d hd i]
    simp only [decide_eq_false h, Bool.not_false, ite_true]
    exact safe_copy (d := d) hd hpre ((launch d).env P) i h

end Claims.MergeAttnStates
