import Claim
import TVA.Proof
import Proof.Body
import Proof.Dots
import Proof.Lanes

/-!
# One trip of the loop

`trip_safe`: the body of the loop over `K` (`loopBody`, lines 136–158) on trip `t` keeps the
invariants `OInv`, `AInv` and leaves memory as it is.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
variable {d : Inst} {g : Geo} {r : Nat} {init : Mem}

/-- **One trip** of the loop. -/
theorem trip_safe (hP : Device E.P) (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    {q k v h o : Ptr .bf16} {o_t : Tensor [BT] I64} {T : I64} {o_v : Tensor [d.bk] I64}
    (hl : TailLanes d g q k v h o o_t T o_v) {t : Nat} (ht : t < nK d) {i_k : I32}
    (hik : i_k.toInt = (t : Int)) {bo : Tensor [BT, d.bk] F32} {bA : Tensor [BT, BT] F32}
    (hO : OInv d g r t bo) (hA : AInv d g t bA) :
    Safe E (Rs d) ∅ (fun st m' => m' = init ∧ OInv d g r (t + 1) st.1 ∧ AInv d g (t + 1) st.2)
      init (loopBody d q k h o_t (o_t.ltS T) o_v i_k (bo, bA)) := by
  simp only [loopBody]
  refine safe_load_q hd hpre hg hl rfl ht (ok_get hd ht hik) fun vq hvq => ?_
  refine safe_load_k hd hpre hg hl rfl (ok_get hd ht hik) fun vk hvk => ?_
  by_cases hs : d.svf = true
  · simp only [hs, ↓reduceIte]
    refine safe_load_h_svf hd hpre hg hl rfl hs (ok_get hd ht hik) fun vh hvh => ?_
    refine dotO_safe hP hd ht hO hvq hvh fun bo' hO' => ?_
    exact dotA_safe hP hd ht hA hvq hvk fun bA' hA' => safe_ret ⟨rfl, hO', hA'⟩
  · simp only [hs]
    refine safe_load_h hd hpre hg hl rfl (by simpa using hs) (ok_get hd ht hik) fun vh hvh => ?_
    refine dotO_safe hP hd ht hO hvq hvh fun bo' hO' => ?_
    exact dotA_safe hP hd ht hA hvq hvk fun bA' hA' => safe_ret ⟨rfl, hO', hA'⟩

end Claims.ChunkFwdO
