import Proof.Spec
import Proof.ExtGlue
import Proof.RowFacts
import Proof.TileQ
import Proof.ExtMask
import Proof.ExtTiles

/-!
# One trip of the new-token loop

`ext_trip_safe` walks `extBody` for one trip: the masks (tree mask, causal, window), the tile skip,
the loads of `k` and `v`, the score `tl.dot`, the online-softmax update and `tl.dot_add`; the
collected relations (`ExtData`) carry the invariant (`ext_trip`).
-/

namespace Claims.ExtendAttn

open TVA

/-- **One new-token trip** keeps the invariant and writes nothing. The loop started at the new token
`extStart d i`; this is its trip `k`, at `sn = extStart d i + 64 k`, with the block's end `cbe`. -/
theorem ext_trip_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {i : Nat} (hi : i < (launch d).size) {s : Sc d} (hs : ScOk d init i s)
    (hm : mOf d i * d.BM < extOf d init (bOf d i)) {cm : Option (I64 × Option I64)}
    (hcm : CmOk d init i cm) {q : Tensor [d.BM, d.BD] BF16} (hq : QFacts d init i q) {k : Nat}
    (hk : extStart d i + BN * k < blockEnd d init i) {cbe : QI d.cm}
    (hcbe : cbe.toInt = (blockEnd d init i : Int)) {sn : QI d.cm}
    (hsn : sn.toInt = ((extStart d i + BN * k : Nat) : Int)) {st : LoopSt d} {m : Mem}
    (hI : LoopInvE d init i (extStart d i) k st m) :
    Safe ((launch d).env P) (Rf d) ∅ (fun st' m' => LoopInvE d init i (extStart d i) (k + 1) st' m') m
      (extBody d s cm (maskM d s) q cbe sn st) := by
  obtain ⟨acc, deno, e_max⟩ := st
  have _ := hm
  have hbody : extBody d s cm (maskM d s) q cbe sn (acc, deno, e_max) =
      extBodyG d s cm (maskM d s) cbe sn
        (fun mn fm => extRest d s q sn mn acc deno e_max fm) := by
    unfold extBody extBodyG extRest extTripP
    rfl
  have hm0 : m = init := hI.1
  rw [hbody, hm0]
  refine extBodyG_safe hd hpre hi hs hcm hcbe hsn hk ?_
  intro fm hfm
  exact ext_rest_safe hP hd hpre hi hs hq hk hcbe hsn acc deno e_max ⟨rfl, hI.2⟩ fm hfm

end Claims.ExtendAttn
