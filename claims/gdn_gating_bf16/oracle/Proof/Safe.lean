import Claim
import TVA.Proof
import Proof.Basic
import Proof.Idx
import Proof.Spec
import Proof.FloatStep
import Proof.LaneG
import Proof.LaneBeta

/-!
# The body of the kernel is safe, and it leaves `Q`

`body_eq` transcribes the model's `do` block with the instance's buffers in place of the kernel's
pointer arguments.  `safe_body` then walks that block: four masked loads, the `float32` `x`, the
softplus branch, `-exp(A_log) * softplus`, the store of `g`, the sigmoid branch, the store of
`beta_output`.  Each line uses its rule; the values the lines compute are passed to `lane_g` and
`lane_beta` of one lane at the end.
-/

set_option maxHeartbeats 1000000

namespace Claims.GdnGating

open TVA

/-- The model's body, with the instance's buffers. -/
theorem body_eq (d : Inst) (ctx : Ctx) :
    (model d).body ctx = (do
      let i_b := tl.program_id ctx 0
      let i_s := tl.program_id ctx 1
      let i_d := tl.program_id ctx 2
      let head_off := i_d * (IntV.ofNat 8 : I32) + tl.arange 0 8
      let off := i_b * (IntV.ofNat 1 : I32) * (IntV.ofNat d.H : I32) +
        i_s * (IntV.ofNat d.H : I32) + head_off
      let mask := head_off.ltS (IntV.ofNat d.H : I32)
      let blk_A_log ← tl.load (d.A_log.ptr 0 + head_off) (some mask)
      let blk_a ← tl.load (d.a.ptr 0 + i_b * (IntV.ofNat d.stride_a) + head_off) (some mask)
      let blk_b ← tl.load (d.b.ptr 0 + i_b * (IntV.ofNat d.stride_b) + head_off) (some mask)
      let blk_bias ← tl.load (d.dt_bias.ptr 0 + head_off) (some mask)
      let xa ← tl.cvt (g := .binary32) blk_a
      let xb ← tl.cvt (g := .binary32) blk_bias
      let x ← tl.add xa xb
      let t1 ← tl.mul (bc oneF32) x
      let cond := t1.fle (bc twentyF32)
      let t2 ← tl.mul (bc oneF32) x
      let e ← tl.exp t2
      let s ← tl.add (bc oneF32) e
      let l ← tl.log s
      let l2 ← tl.mul (bc oneF32) l
      let softplus_x := tl.where cond l2 x
      let eA ← tl.exp blk_A_log
      let neA ← tl.neg eA
      let blk_g ← tl.mul neA softplus_x
      tl.store (d.g.ptr 0 + off) blk_g (some mask)
      let blk_b32 ← tl.cvt (g := .binary32) blk_b
      let nb ← tl.neg blk_b32
      let eb ← tl.exp nb
      let den ← tl.add (bc oneF32) eb
      let blk_beta_output ← tl.div (bc oneF32) den
      let beta16 ← tl.cvt (g := .bfloat16) blk_beta_output
      let beta32 ← tl.cvt (g := .binary32) beta16
      tl.store (d.bout.ptr 0 + off) beta32 (some mask)) := rfl

/-- **One program's body is safe.**  Run alone from `init`, it reads its four tiles, writes its
tiles of `g` and `beta_output`, and leaves `Q`. -/
theorem safe_body {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (hexp : Device.ExpF32 E.P) (hlog : Device.LogF32 E.P) (hfull : Device.DivFull E.P)
    (hetot : Device.MathTotal E.P .exp .binary32) (hltot : Device.MathTotal E.P .log .binary32)
    (hdtot : Device.DivTotal E.P .binary32) (hninf : Device.ExpF32NegInf E.P) (i : Fin (launch d).size) :
    Safe E (Rset d i) (Wset d i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  obtain ⟨hAlv, halv, hblv, hbiaslv, hglv, hboutlv, hIn⟩ := hpre
  rw [body_eq]
  -- the four masked loads
  refine safe_load_buf d.A_log (addrs_A_log hd i)
    (fun p hp => gH_len_A hd i p.1 (hidx_of_en hp)) hAlv
    (fun x hx => Or.inl (Or.inl (Or.inl hx))) ?_
  intro vA hvA _
  refine safe_load_buf d.a (addrs_a hd i)
    (fun p hp => gR_len_a hd i p.1 (hidx_of_en hp)) halv
    (fun x hx => Or.inl (Or.inl (Or.inr hx))) ?_
  intro va hva _
  refine safe_load_buf d.b (addrs_b hd i)
    (fun p hp => gR_len_b hd i p.1 (hidx_of_en hp)) hblv
    (fun x hx => Or.inl (Or.inr hx)) ?_
  intro vb hvb _
  refine safe_load_buf d.dt_bias (addrs_bias hd i)
    (fun p hp => gH_len_bias hd i p.1 (hidx_of_en hp)) hbiaslv
    (fun x hx => Or.inr hx) ?_
  intro vdt hvdt _
  -- the widening conversions and `x = a + dt_bias`
  refine safe_cvt (g := .binary32)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro xa hxa
  refine safe_cvt (g := .binary32)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro xb hxb
  refine safe_add ?_
  intro x hx
  -- `t1`, `cond`, `t2`
  refine safe_mul ?_
  intro t1 ht1
  refine safe_mul ?_
  intro t2 ht2
  -- the softplus branch
  refine safe_math (fn := .exp) (fun p => hetot (t2.get p)) ?_
  intro e he
  refine safe_add ?_
  intro s hsn
  refine safe_math (fn := .log) (fun p => hltot (s.get p)) ?_
  intro logv hlogram
  refine safe_mul ?_
  intro l2 hl2
  -- `-exp(A_log) * softplus_x`, and the store of `g`
  refine safe_math (fn := .exp) (fun p => hetot (vA.get p)) ?_
  intro eA heA
  refine safe_neg ?_
  intro neA hneA
  refine safe_mul ?_
  intro gg hgg
  refine safe_store_buf d.g (addrs_g hd i)
    (fun p hp => gO_len_g hd i p.1 (hidx_of_en hp)) ?_ hglv
    (fun x hx => Or.inl hx) ?_
  · intro p q hp hq hpq
    obtain ⟨lp, up⟩ := p
    obtain ⟨lq, uq⟩ := q
    obtain rfl : lp = lq := gO_inj i lp lq hpq
    exact Prod.ext rfl (by cases up; cases uq; rfl)
  intro m1 hg hfr1 hpres1
  -- the sigmoid branch: `blk_b32`, `nb`, `eb`, `den`, `q`, `beta16`, `beta32`
  refine safe_cvt (g := .binary32)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro b32 hb32
  refine safe_neg ?_
  intro nb hnb
  refine safe_math (fn := .exp) (fun p => hetot (nb.get p)) ?_
  intro eb heb
  refine safe_add ?_
  intro den hden
  refine safe_div (fun p => hdtot oneF32 (den.get p)) ?_
  intro q hq
  refine safe_cvt (g := .bfloat16)
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩ ?_
  intro b16 hb16
  refine safe_cvt (g := .binary32)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro r hr
  -- the store of `beta_output`, and `Q`
  refine safe_last ?_
  refine safe_store_buf d.bout (addrs_bout hd i)
    (fun p hp => gO_len_bout hd i p.1 (hidx_of_en hp)) ?_ ?_ ?_ ?_
  · intro p q hp hq hpq
    obtain ⟨lp, up⟩ := p
    obtain ⟨lq, uq⟩ := q
    obtain rfl : lp = lq := gO_inj i lp lq hpq
    exact Prod.ext rfl (by cases up; cases uq; rfl)
  · exact Buf.valid_of_agree hboutlv (fun x hx => hpres1 x (hboutlv.2.2.2 x hx))
  · intro x hx
    exact ⟨Or.inr hx, fun hgx => Set.disjoint_left.mp (disj_g_bout hd i i) hgx hx⟩
  · intro m2 hbout hfr2 hpres2
    refine safe_ret ?_
    refine ⟨?_, ?_⟩
    · -- the element of `g`
      intro l hen
      obtain ⟨A, hArd, hAfin, hroomA⟩ := hIn.1 (hidx d i.val l.val) (hidx_of_en hen)
      obtain ⟨av, bv, havd, hbvd, havfin, hbvfin, hxub⟩ :=
        hIn.2.1 (tok d i.val) (tok_lt i) (hidx d i.val l.val) (hidx_of_en hen)
      have hAeq : vA.get (lane l) = A :=
        Option.some.inj ((hvA (lane l) hen).symm.trans
          (show init.read .f32 (d.A_log.ptr (gH d i.val (lane l))).addr = some A from hArd))
      have haeq : va.get (lane l) = av :=
        Option.some.inj ((hva (lane l) hen).symm.trans
          (show init.read .bf16 (d.a.ptr (gR d i.val d.stride_a (lane l))).addr = some av
            from havd))
      have hbeq : vdt.get (lane l) = bv :=
        Option.some.inj ((hvdt (lane l) hen).symm.trans
          (show init.read .bf16 (d.dt_bias.ptr (gH d i.val (lane l))).addr = some bv from hbvd))
      have hgread : m2.read .f32 (d.g.ptr (gO d i.val (lane l))).addr = some (gg.get (lane l)) := by
        have hspan : ∀ x ∈ Mem.span .f32 (d.g.ptr (gO d i.val (lane l))).addr, m2 x = m1 x := by
          intro y hy
          refine hfr2 y ?_
          intro hyb
          have hk : gO d i.val (lane l) ∈ oSet d i := ⟨lane l, hen, rfl⟩
          exact Set.disjoint_left.mp (disj_g_bout hd i i)
            (Buf.span_sub_elems d.g hk hy) hyb
        rw [Mem.read_congr hspan]
        exact hg (lane l) hen
      refine ⟨A, hArd, av, havd, bv, hbvd, gg.get (lane l), hgread, ?_⟩
      exact lane_g (P := E.P) (A := A) (av := av) (bv := bv)
        (xa := xa.get (lane l)) (xb := xb.get (lane l)) (x := x.get (lane l))
        (t1 := t1.get (lane l)) (t2 := t2.get (lane l)) (e := e.get (lane l))
        (s := s.get (lane l)) (l := logv.get (lane l)) (l2 := l2.get (lane l))
        (sp := (tl.where (t1.fle (bc twentyF32)) l2 x).get (lane l))
        (eA := eA.get (lane l)) (neA := neA.get (lane l)) (gg := gg.get (lane l))
        (cond := (t1.fle (bc twentyF32)).get (lane l))
        hexp hlog hninf hAfin hroomA havfin hbvfin (by simpa using hxub)
        (by rw [← haeq]; exact hxa (lane l)) (by rw [← hbeq]; exact hxb (lane l))
        (hx (lane l)) (ht1 (lane l)) rfl (ht2 (lane l)) (he (lane l)) (hsn (lane l))
        (hlogram (lane l)) (hl2 (lane l)) rfl (by rw [← hAeq]; exact heA (lane l))
        (hneA (lane l)) (hgg (lane l))
    · -- the element of `beta_output`
      intro l hen
      obtain ⟨bb, hbbd, hbbfin, hroomb⟩ :=
        hIn.2.2 (tok d i.val) (tok_lt i) (hidx d i.val l.val) (hidx_of_en hen)
      have hbb_eq : vb.get (lane l) = bb :=
        Option.some.inj ((hvb (lane l) hen).symm.trans
          (show init.read .bf16 (d.b.ptr (gR d i.val d.stride_b (lane l))).addr = some bb
            from hbbd))
      have hres := lane_beta (P := E.P) (bv := bb) (b32 := b32.get (lane l))
        (nb := nb.get (lane l)) (eb := eb.get (lane l)) (den := den.get (lane l))
        (q := q.get (lane l)) (b16 := b16.get (lane l)) (r := r.get (lane l))
        hexp hfull hbbfin hroomb
        (by rw [← hbb_eq]; exact hb32 (lane l)) (hnb (lane l)) (heb (lane l))
        (hden (lane l)) (hq (lane l)) (hb16 (lane l)) (hr (lane l))
      exact ⟨bb, hbbd, r.get (lane l), hbout (lane l) hen, hres.1, hres.2⟩

end Claims.GdnGating
