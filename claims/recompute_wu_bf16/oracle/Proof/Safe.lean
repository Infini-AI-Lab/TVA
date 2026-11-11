import Claim
import TVA.Proof
import Proof.Body

/-!
# Each program is safe

`safe_program`: program `i` run alone from `init` is safe, reads the inputs, writes `Wf`, and
leaves `Qf`. One walk serves both dtypes of `beta`: `body32_eq`, `body16_eq` write each model as
the head and `tailProg`, and `tail_safe` walks the tail.
-/

namespace Claims.RecomputeWU

open TVA

/-- The post-condition of the tail gives `Qf`: every cell of every row. -/
theorem Qf_of_WInv {d : Inst} {init : Mem} {i : Nat} {m : Mem}
    (hI : WInv d init (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i)
      ((d.K + 63) / 64) m) : Qf d init i m := by
  intro r hr hrow
  have hrE : rowEn (itOf d init i) (lenOf d init i) r = true := by simp [rowEn, hrow]
  have hr' : r < 64 := hr
  refine ⟨fun c hc => ?_, fun c hc => ?_⟩
  · have := hI.2.2.1 ⟨r, hr'⟩ hrE c hc
    exact this
  · have := hI.2.2.2 ⟨r, hr'⟩ hrE c (lt_min hc (by omega))
    exact this

/-- `Wf` is the union of the `u` and `w` elements of the program's chunk. -/
theorem Wf_eq (d : Inst) (init : Mem) (i : Nat) :
    Wf d init i = d.u.elems (UE d (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i)) ∪
      d.w.elems (WE d (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i)) := rfl

/-- **Program `i` is safe.** -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (Rall d) (Wf d init i) (fun _ m' => Qf d init i m') init
      ((model d).body ((launch d).ctx i)) := by
  have hi : (i : Nat) < g0 d * (d.B * d.HV) := by
    have := Nat.lt_of_lt_of_eq i.isLt (launch_size_eq d); omega
  obtain ⟨hl, hs⟩ := geo hd hpre hi
  obtain ⟨-, -, hh, -⟩ := coords hd hi
  have hag0 : Ag d init init := fun _ _ => rfl
  have hpres0 : Pres d init := ⟨hpre.uv, hpre.wv⟩
  have hTlt := hd.t_lt
  have hlen : lenOf d init i ≤ nTok d := by omega
  have hnT : nTok d ≤ 2 ^ 35 := by
    unfold nTok
    calc d.B * d.T ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul hd.b_batch hd.t_ctx
      _ = 2 ^ 35 := by norm_num
  by_cases hbb : d.bbf = true
  · have hm : model d = recompute_w_u_fwd_kernel_b16 (sD := d.idx) (d.k.ptr 0) (d.v.ptr 0)
        (d.betab.ptr 0) (d.w.ptr 0) (d.u.ptr 0) (d.A.ptr 0) (d.g.ptr 0) (d.cu.ptr 0) (d.ci.ptr 0)
        (IntV.ofNat d.T) d.H d.HV d.K d.V 64 64 64 d.isVarlen := by
      unfold model; simp [hbb]
    rw [hm, body16_eq]
    refine body_of_pre hd hpre i ?_
    intro i_h i_t bos T_n h1 h2 h3 h4
    have ho := oT_lane h2 (by omega)
    have hmt := mT_lane ho h4
    exact (tail_safe hP hd hpre hs hl hh h3 h1 ho hmt (β := .bf16)
      (loadβ := loadBeta16 (d.betab.ptr 0) bos i_h (oT i_t) (mT (oT i_t) T_n) d.HV)
      (vbK := vbStep16) (kbK := kbStep16)
      (Bl := BLane16 d init (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i))
      (fun W Q m k hag hk => loadBeta16_safe hd hpre hbb hs hl hh h3 h1 ho hmt hag hk)
      (fun b_b hBl t ht W Q m b_v k hV hk => vbStep16_safe hpre hbb hs hh hBl hV hk)
      (fun b_b l_g b_g hBl hGl hEl t ht W Q m b_k k hK hk =>
        kbStep16_safe hd hpre hbb hs hh hBl hGl hEl hK hk) hag0 hpres0).mono
      (fun _ m' h => Qf_of_WInv h) (by rw [Wf_eq])
  · have hbb' : d.bbf = false := by simpa using hbb
    have hm : model d = recompute_w_u_fwd_kernel_b32 (sD := d.idx) (d.k.ptr 0) (d.v.ptr 0)
        (d.beta.ptr 0) (d.w.ptr 0) (d.u.ptr 0) (d.A.ptr 0) (d.g.ptr 0) (d.cu.ptr 0) (d.ci.ptr 0)
        (IntV.ofNat d.T) d.H d.HV d.K d.V 64 64 64 d.isVarlen := by
      unfold model; simp [hbb']
    rw [hm, body32_eq]
    refine body_of_pre hd hpre i ?_
    intro i_h i_t bos T_n h1 h2 h3 h4
    have ho := oT_lane h2 (by omega)
    have hmt := mT_lane ho h4
    exact (tail_safe hP hd hpre hs hl hh h3 h1 ho hmt (β := .f32)
      (loadβ := loadBeta32 (d.beta.ptr 0) bos i_h (oT i_t) (mT (oT i_t) T_n) d.HV)
      (vbK := vbStep32) (kbK := kbStep32)
      (Bl := BLane32 d init (s0Of d init i) (lenOf d init i) (itOf d init i) (hOf d i))
      (fun W Q m k hag hk => loadBeta32_safe hd hpre hbb' hs hl hh h3 h1 ho hmt hag hk)
      (fun b_b hBl t ht W Q m b_v k hV hk => vbStep32_safe hpre hbb' hs hh hBl hV hk)
      (fun b_b l_g b_g hBl hGl hEl t ht W Q m b_k k hK hk =>
        kbStep32_safe hd hpre hbb' hs hh hBl hGl hEl hK hk) hag0 hpres0).mono
      (fun _ m' h => Qf_of_WInv h) (by rw [Wf_eq])

end Claims.RecomputeWU
