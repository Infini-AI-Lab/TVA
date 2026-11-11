import Claim
import TVA.Proof
import Proof.Setup
import Proof.Logit
import Proof.CapBound
import Proof.Topsum
import Proof.WeightDen
import Proof.Weight
import Proof.Ids

/-!
# The last step: `TokenPost` from the walk's state

`safe_program` ends with the last store; what is left is to read the four output elements back
out of the final memory and to compare them with the reference.  This module assembles that:
the capped logits `C` are the computed `cap_bound`s (one per expert, via `ids` for the two
indices), the first weight is `den_qinv_bound` (`1/den` against `1/D`), and the second is
`weight_second`.
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}


/-- One element's span in one buffer avoids the elements of another buffer. -/
theorem span_not_elems {d₁ d₂ : DType} {b : Buf d₁} {c : Buf d₂}
    (hbc : Disjoint b.bytes c.bytes) {j : Nat} (hj : j < b.len) {S : Set Nat}
    (hS : ∀ k ∈ S, k < c.len) :
    ∀ x ∈ Mem.span d₁ (b.ptr (j : Int)).addr, x ∉ c.elems S := by
  have hT : ∀ k ∈ ({j} : Set Nat), k < b.len := by
    intro k hk; rw [Set.mem_singleton_iff] at hk; subst hk; exact hj
  have hsub : Mem.span d₁ (b.ptr (j : Int)).addr ⊆ b.elems {j} :=
    Buf.span_sub_elems b (S := {j}) (j := j) rfl
  exact fun x hx hx' =>
    Set.disjoint_left.mp (Buf.elems_disjoint_of b c hbc hT hS) (hsub hx) hx'

/-- One element's span in a buffer avoids another set of that buffer's elements. -/
theorem span_not_elems_same {d₁ : DType} {b : Buf d₁} {j : Nat} (_hj : j < b.len)
    {S : Set Nat} (hjS : j ∉ S) :
    ∀ x ∈ Mem.span d₁ (b.ptr (j : Int)).addr, x ∉ b.elems S := by
  have hdisj : Disjoint ({j} : Set Nat) S := by
    rw [Set.disjoint_left]
    intro a ha hb
    rw [Set.mem_singleton_iff] at ha
    exact hjS (ha ▸ hb)
  have hsub : Mem.span d₁ (b.ptr (j : Int)).addr ⊆ b.elems {j} :=
    Buf.span_sub_elems b (S := {j}) (j := j) rfl
  exact fun x hx hx' =>
    Set.disjoint_left.mp (Buf.elems_disjoint b hdisj) (hsub hx) hx'

/-- `wC` at an index below `8` is the computed capped logit there. -/
theorem wC_eq {capped : Tensor [8 - 0] F32} {e : Nat} (he : e < 8 - 0) :
    wC capped e = (capped.get1 ⟨e, he⟩).toReal := by
  unfold wC
  rw [dite_eq_left he]

set_option maxHeartbeats 4000000 in
/-- **The last step.** From the state the walk reaches after the final store, the postcondition
`TokenPost` holds. -/
theorem token_post_final {d : Inst} (hd : d.ok) {P : Platform}
    (hexp : Device.MathTotal P MathFn.exp Format.binary32)
    (hexpacc : Device.ExpF32 P)
    (hdiv : Device.DivTotal P Format.binary32)
    (hdivfull : Device.DivFull P)
    (hsum : Device.SumDepth P)
    {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size)
    (hbs : (i : Nat) < d.bs) (hl2 : 2 * (i : Nat) + 1 < 2 * d.bs)
    {wv : Tensor [8 - 0, d.BLOCK] DType.f32.Val}
    (hwrd : ∀ (j : Idx [8 - 0, d.BLOCK]), decide (↑j.2.1 < d.hidden) = true →
      init.read DType.f32 (d.w.ptr ↑(gW d j)).addr = some (wv.get j))
    (hwpad : ∀ (j : Idx [8 - 0, d.BLOCK]), decide (↑j.2.1 < d.hidden) = false →
      ∀ (o : Tensor [8 - 0, d.BLOCK] DType.f32.Val),
        some (bc (c32 0)) = some o → wv.get j = o.get j)
    {xv : Tensor [d.BLOCK - 0] DType.bf16.Val}
    (hxrd : ∀ (j : Idx [d.BLOCK - 0]), decide (↑j.1 < d.hidden) = true →
      init.read DType.bf16 (d.x.ptr ↑(gX d (↑i) j)).addr = some (xv.get j))
    (hxpad : ∀ (j : Idx [d.BLOCK - 0]), decide (↑j.1 < d.hidden) = false →
      ∀ (o : Tensor [d.BLOCK - 0] DType.bf16.Val),
        some (bc (FloatV.litBF16 (FloatV.ofPat 0))) = some o → xv.get j = o.get j)
    {x32 : Tensor [d.BLOCK] F32}
    (hx32 : ∀ (j : Idx [d.BLOCK]),
      (FloatV.cvtWith RoundMode.rn (xv.get j)).NaNEq (x32.get j))
    {prod : Tensor [8 - 0, d.BLOCK] F32}
    (hprod : ∀ (j : Idx [8 - 0, d.BLOCK]),
      ((wv.get j).mul ((x32.row 8).get j)).NaNEq (prod.get j))
    {logits : Tensor [8 - 0] F32}
    (hlogits : ∀ (e : Fin (8 - 0)),
      ∃ t, Tree.Covers d.BLOCK t ∧
        t.depth ≤ tl.sumBound ((launch d).env P).P d.BLOCK ∧
          Tree.Eval (AddRel ((launch d).env P).fusion)
            (fun j => if h : j < d.BLOCK then prod.get2 e ⟨j, h⟩ else default) t
            (logits.get1 e))
    {lsc : Tensor [8 - 0] F32}
    (hls : ∀ (j : Idx [8 - 0]),
      ((launch d).env P).P.div Format.binary32 (logits.get j)
        ((bc (c32 4629137466983448576)).get j) (lsc.get j))
    {doubled : Tensor [8 - 0] F32}
    (hdoubled : ∀ (j : Idx [8 - 0]),
      (((bc (c32 4611686018427387904)).get j).mul (lsc.get j)).NaNEq (doubled.get j))
    {exped : Tensor [8 - 0] F32}
    (hexped : ∀ (j : Idx [8 - 0]),
      ((launch d).env P).P.math MathFn.exp Format.binary32
        (doubled.get j) (exped.get j))
    {top : Tensor [8 - 0] F32}
    (htop : ∀ (j : Idx [8 - 0]),
      ((exped.get j).sub ((bc (c32 4607182418800017408)).get j)).NaNEq (top.get j))
    {bottom : Tensor [8 - 0] F32}
    (hbottom : ∀ (j : Idx [8 - 0]),
      ((exped.get j).add ((bc (c32 4607182418800017408)).get j)).NaNEq (bottom.get j))
    {q : Tensor [8 - 0] F32}
    (hq : ∀ (j : Idx [8 - 0]),
      ((launch d).env P).P.div Format.binary32 (top.get j) (bottom.get j)
        (q.get j))
    {capped : Tensor [8 - 0] F32}
    (hcapped : ∀ (j : Idx [8 - 0]),
      ((q.get j).mul ((bc (c32 4629137466983448576)).get j)).NaNEq (capped.get j))
    {pair1 : F32 × I32}
    (hpair1 : tl.ArgRes true true (8 - 0)
      (fun j => if h : j < 8 - 0 then capped.get1 ⟨j, h⟩ else default) pair1)
    {m1 : Mem}
    (hrd1 : ∀ (i_1 : Idx []), true = true →
      m1.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
        some ((Tensor.scalar pair1.2).get i_1))
    (_hfr1 : ∀ x ∉ d.outi.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat)) (fun _ => true)),
      m1 x = init x)
    {top1v : F32}
    (htop1v : ∃ tr, Tree.Covers (8 - 0) tr ∧
      Tree.Eval (FloatV.MaxRel true)
        (fun j => if h : j < 8 - 0 then capped.get1 ⟨j, h⟩ else default) tr top1v)
    {diffs : Tensor [8 - 0] F32}
    (hdiffs : ∀ (j : Idx [8 - 0]),
      AddRel ((launch d).env P).fusion (capped.get j) ((bc top1v).get j).neg (diffs.get j))
    {exps : Tensor [8 - 0] F32}
    (hexps : ∀ (j : Idx [8 - 0]),
      ((launch d).env P).P.math MathFn.exp Format.binary32 (diffs.get j)
        (exps.get j))
    {den : F32}
    (hden : ∃ tr, Tree.Covers (8 - 0) tr ∧ tr.depth ≤ tl.sumBound ((launch d).env P).P (8 - 0) ∧
      Tree.Eval (AddRel false)
        (fun j => if h : j < 8 - 0 then exps.get1 ⟨j, h⟩ else default) tr den)
    {qinv : Tensor [] F32}
    (hinv : ∀ (j : Idx []),
      ((launch d).env P).P.div Format.binary32
        ((Tensor.scalar (c32 4607182418800017408)).get j) ((Tensor.scalar den).get j)
        (qinv.get j))
    {m2 : Mem}
    (hrd2 : ∀ (i_1 : Idx []), true = true →
      m2.read DType.f32 (d.outw.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
        some ((Tensor.scalar qinv.item).get i_1))
    (hfr2 : ∀ x ∉ d.outw.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat)) (fun _ => true)),
      m2 x = m1 x)
    {pair2 : F32 × I32}
    (hpair2 : tl.ArgRes true true (8 - 0)
      (fun j => if h : j < 8 - 0 then
        (tl.where (Tensor.map (fun k => !k.eq pair1.2) (tl.arange 0 8)) capped
          (bc (c32 18442240474082181120))).get1 ⟨j, h⟩ else default) pair2)
    {m3 : Mem}
    (hrd3 : ∀ (i_1 : Idx []), true = true →
      m3.read DType.i32 (d.outi.ptr ((2 * (i : Nat) + 1 : Nat) : Int)).addr =
        some ((Tensor.scalar pair2.2).get i_1))
    (hfr3 : ∀ x ∉ d.outi.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat) + 1) (fun _ => true)),
      m3 x = m2 x)
    {onehot32 : Tensor [8] DType.f32.Val}
    (honehot : ∀ (j : Idx [8]),
      ConvRel ((launch d).env P).P DType.i1 DType.f32
        ((Tensor.map (fun k => k.eq pair2.2) (tl.arange 0 8)).get j) (onehot32.get j))
    {prod2 : Tensor [8 - 0] F32}
    (hprod2 : ∀ (j : Idx [8 - 0]), ((capped.get j).mul (onehot32.get j)).NaNEq (prod2.get j))
    {top2v : F32}
    (htop2v : ∃ tr, Tree.Covers (8 - 0) tr ∧
      tr.depth ≤ tl.sumBound ((launch d).env P).P (8 - 0) ∧
      Tree.Eval (AddRel ((launch d).env P).fusion)
        (fun j => if h : j < 8 - 0 then prod2.get1 ⟨j, h⟩ else default) tr top2v)
    {wdiff : Tensor [] F32}
    (hwdiff : ∀ (j : Idx []),
      (((Tensor.scalar top2v).get j).sub ((Tensor.scalar top1v).get j)).NaNEq (wdiff.get j))
    {w2e : Tensor [] F32}
    (hw2e : ∀ (j : Idx []),
      ((launch d).env P).P.math MathFn.exp Format.binary32 (wdiff.get j)
        (w2e.get j))
    {w1v : Tensor [] F32}
    (hw1v : ∀ (j : Idx []),
      ((w2e.get j).mul ((Tensor.scalar qinv.item).get j)).NaNEq (w1v.get j))
    {m4 : Mem}
    (hrd4 : ∀ (i_1 : Idx []), true = true →
      m4.read DType.f32 (d.outw.ptr ((2 * (i : Nat) + 1 : Nat) : Int)).addr =
        some ((Tensor.scalar w1v.item).get i_1))
    (hfr4 : ∀ x ∉ d.outw.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat) + 1) (fun _ => true)),
      m4 x = m3 x) :
    TokenPost d init m4 (i : Nat) := by
  -- the per-expert bounds on the computed capped logits
  have hcapE : ∀ e : Fin (8 - 0),
      ((capped.get1 e).isFinite = true ∧
       |(capped.get1 e).toReal - capRef (mu init d (i : Nat) e)| ≤
         capEps (absS init d (i : Nat) e) ∧
       |(capped.get1 e).toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ))) := by
    intro e
    exact cap_bound hd hexp hexpacc hdiv hdivfull (i : Nat) e
      (hpre.2.2.2.2.2.2 _ _ hbs e.isLt).2
      (hpre.2.2.2.2.2.2 _ _ hbs e.isLt).1
      (logit_bound hd hpre hsum i hwrd hwpad hxrd hxpad hx32 hprod hlogits e)
      hls hdoubled hexped htop hbottom hq hcapped
  have hfin : ∀ e : Fin (8 - 0), (capped.get1 e).isFinite = true := fun e => (hcapE e).1
  have hsmall : ∀ e : Fin (8 - 0), |(capped.get1 e).toReal| ≤ 30 * (1 + 2 ^ (-20 : ℤ)) :=
    fun e => (hcapE e).2.2
  -- the two indices
  obtain ⟨i0, i1, hk1, hne, hk2, hle0, hlt0, hle1, hlt1, htop1fin, htop1val⟩ :=
    ids hfin hpair1 htop1v hpair2
  -- the second maximum
  obtain ⟨htop2fin, htop2val⟩ :=
    top_sum_bound (d := d) (P := P) hsum hfin hsmall hk2 honehot hprod2 htop2v
  -- the denominator, `1/den` against `1/D`
  obtain ⟨-, hdenne, hqfin, hqD⟩ :=
    den_qinv_bound (d := d) (P := P) (t := (i : Nat)) hbs hfin hsmall hsum hexpacc hdivfull
      i0.isLt hle0 htop1fin htop1val hdiffs hexps hden hinv
  -- the numerator
  obtain ⟨hw1fin, hw1bound⟩ :=
    weight_second (d := d) (P := P) (t := (i : Nat)) hbs hfin hsmall hsum hexpacc hdivfull
      i0.isLt hle0 htop1fin htop1val hdiffs hexps hden hinv i1.isLt
      (fun h => hne (Fin.ext h)) (fun e he => hle1 e (fun h => he (congrArg Fin.val h)))
      htop2fin htop2val hwdiff hw2e hw1v
  -- the reference denominator is positive and at least one
  have hDpos : (0 : ℝ) < wD capped i0.val := by
    unfold wD
    exact Finset.sum_pos (fun k _ => Real.exp_pos _)
      ⟨i0.val, Finset.mem_range.mpr i0.isLt⟩
  have hD1 : (1 : ℝ) ≤ wD capped i0.val := by
    have h := Finset.single_le_sum
      (s := Finset.range 8) (f := fun k => Real.exp (wC capped k - wC capped i0.val))
      (fun k _ => (Real.exp_pos _).le) (Finset.mem_range.mpr i0.isLt)
    have h0 : Real.exp (wC capped i0.val - wC capped i0.val) = 1 := by
      rw [sub_self, Real.exp_zero]
    unfold wD
    rw [← h0]
    exact h
  have hsoft : softRef (wC capped) i0.val i0.val = 1 / wD capped i0.val := by
    have : Real.exp (wC capped i0.val - wC capped i0.val) = 1 := by
      rw [sub_self, Real.exp_zero]
    unfold softRef wD
    rw [this, one_div]
  have hwRel0 : (3 : ℝ) / 200000 ≤ wRel0 := by
    unfold wRel0
    rw [zpow_neg, show (14 : ℤ) = ((14 : ℕ) : ℤ) by norm_num, zpow_natCast]
    norm_num
  have hwFloor : (0 : ℝ) < wFloor := by
    unfold wFloor
    rw [zpow_neg, show (130 : ℤ) = ((130 : ℕ) : ℤ) by norm_num, zpow_natCast]
    positivity
  have hb0 : |qinv.item.toReal - softRef (wC capped) i0.val i0.val| ≤
      wRel0 * softRef (wC capped) i0.val i0.val + wFloor := by
    rw [hsoft]
    have hkey : qinv.item.toReal - 1 / wD capped i0.val
        = (qinv.item.toReal * wD capped i0.val - 1) / wD capped i0.val := by
      field_simp
    rw [hkey, abs_div, abs_of_pos hDpos]
    calc |qinv.item.toReal * wD capped i0.val - 1| / wD capped i0.val
        ≤ ((3 : ℝ) / 200000) / wD capped i0.val :=
          div_le_div_of_nonneg_right hqD hDpos.le
      _ ≤ wRel0 / wD capped i0.val := by
          exact div_le_div_of_nonneg_right hwRel0 hDpos.le
      _ = wRel0 * (1 / wD capped i0.val) := by ring
      _ ≤ wRel0 * (1 / wD capped i0.val) + wFloor := by linarith
  -- the four output reads
  have hoi2 : 2 * (i : Nat) < d.outi.len := by
    have := hd.outi_len; omega
  have hoi2' : 2 * (i : Nat) + 1 < d.outi.len := Nat.lt_of_lt_of_le hl2 hd.outi_len
  have how2 : 2 * (i : Nat) < d.outw.len := by
    have := hd.outw_len; omega
  have how2' : 2 * (i : Nat) + 1 < d.outw.len := Nat.lt_of_lt_of_le hl2 hd.outw_len
  have hnotOI0 : ∀ x ∈ Mem.span DType.i32
      (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr,
      x ∉ d.outw.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat)) (fun _ => true)) :=
    span_not_elems (b := d.outi) (c := d.outw) hd.outw_outi.symm hoi2
      (img0_lt how2')
  have hnotOI1 : ∀ x ∈ Mem.span DType.i32
      (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr,
      x ∉ d.outi.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat) + 1) (fun _ => true)) :=
    span_not_elems_same (b := d.outi) hoi2 (by rintro ⟨u, -, h⟩; beta_reduce at h; omega)
  have hnotOI2 : ∀ x ∈ Mem.span DType.i32
      (d.outi.ptr ((2 * (i : Nat) + 1 : Nat) : Int)).addr,
      x ∉ d.outw.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat) + 1) (fun _ => true)) :=
    span_not_elems (b := d.outi) (c := d.outw) hd.outw_outi.symm hoi2'
      (img1_lt how2')
  have hnotOI3 : ∀ x ∈ Mem.span DType.i32
      (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr,
      x ∉ d.outw.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat) + 1) (fun _ => true)) :=
    span_not_elems (b := d.outi) (c := d.outw) hd.outw_outi.symm hoi2
      (img1_lt how2')
  have hnotOW0 : ∀ x ∈ Mem.span DType.f32
      (d.outw.ptr ((2 * (i : Nat) : Nat) : Int)).addr,
      x ∉ d.outi.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat) + 1) (fun _ => true)) :=
    span_not_elems (b := d.outw) (c := d.outi) hd.outw_outi how2
      (img1_lt hoi2')
  have hnotOW1 : ∀ x ∈ Mem.span DType.f32
      (d.outw.ptr ((2 * (i : Nat) : Nat) : Int)).addr,
      x ∉ d.outw.elems (Buf.img (fun (_ : Idx []) => 2 * (i : Nat) + 1) (fun _ => true)) :=
    span_not_elems_same (b := d.outw) how2 (by rintro ⟨u, -, h⟩; beta_reduce at h; omega)
  have hA0 : m1.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
      some (IntV.ofNat i0.val) := by
    have h := hrd1 () rfl
    have h' : (Tensor.scalar pair1.2).get (() : Idx []) = pair1.2 := get_scalar_eq pair1.2 ()
    rw [h'] at h
    rwa [← hk1] at h
  have hA1 : m4.read DType.i32 (d.outi.ptr ((2 * (i : Nat) + 1 : Nat) : Int)).addr =
      some (IntV.ofNat i1.val) := by
    have h := hrd3 () rfl
    have h' : (Tensor.scalar pair2.2).get (() : Idx []) = pair2.2 := get_scalar_eq pair2.2 ()
    rw [h'] at h
    rw [← hk2] at h
    exact (Mem.read_congr (fun x hx => hfr4 x (hnotOI2 x hx))).trans h
  have hA2 : m4.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
      some (IntV.ofNat i0.val) := by
    have h12 : m2.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
        m1.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr :=
      Mem.read_congr (fun x hx => hfr2 x (hnotOI0 x hx))
    have h23 : m3.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
        m2.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr :=
      Mem.read_congr (fun x hx => hfr3 x (hnotOI1 x hx))
    have h34 : m4.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
        m3.read DType.i32 (d.outi.ptr ((2 * (i : Nat) : Nat) : Int)).addr :=
      Mem.read_congr (fun x hx => hfr4 x (hnotOI3 x hx))
    exact (h34.trans h23).trans (h12.trans hA0)
  have hB0 : m4.read DType.f32 (d.outw.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
      some qinv.item := by
    have h := hrd2 () rfl
    have h' : (Tensor.scalar qinv.item).get (() : Idx []) = qinv.item :=
      get_scalar_eq qinv.item ()
    rw [h'] at h
    have h23 : m3.read DType.f32 (d.outw.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
        m2.read DType.f32 (d.outw.ptr ((2 * (i : Nat) : Nat) : Int)).addr :=
      Mem.read_congr (fun x hx => hfr3 x (hnotOW0 x hx))
    have h34 : m4.read DType.f32 (d.outw.ptr ((2 * (i : Nat) : Nat) : Int)).addr =
        m3.read DType.f32 (d.outw.ptr ((2 * (i : Nat) : Nat) : Int)).addr :=
      Mem.read_congr (fun x hx => hfr4 x (hnotOW1 x hx))
    exact (h34.trans h23).trans h
  have hB1 : m4.read DType.f32 (d.outw.ptr ((2 * (i : Nat) + 1 : Nat) : Int)).addr =
      some w1v.item := by
    have h := hrd4 () rfl
    have h' : (Tensor.scalar w1v.item).get (() : Idx []) = w1v.item :=
      get_scalar_eq w1v.item ()
    rw [h'] at h
    exact h
  refine ⟨wC capped, i0.val, i1.val, ?_, i0.isLt, ?_, ?_, i1.isLt, ?_, ?_, ?_, hA2, hA1,
    qinv.item, w1v.item, hB0, hB1, hqfin, hw1fin, hb0, hw1bound⟩
  · intro e he
    have he8 : e < 8 - 0 := by omega
    rw [wC_eq he8]
    exact (hcapE ⟨e, he8⟩).2.1
  · intro e he
    have he8 : e < 8 - 0 := by omega
    rw [wC_eq he8, wC_eq i0.isLt]
    exact hle0 ⟨e, he8⟩
  · intro e he
    have he8 : e < 8 - 0 := by omega
    have hi08 : i0.val < 8 - 0 := i0.isLt
    rw [wC_eq he8, wC_eq hi08]
    exact hlt0 ⟨e, he8⟩ he
  · intro h
    exact hne (Fin.ext h)
  · intro e he hne0
    have he8 : e < 8 - 0 := by omega
    rw [wC_eq he8, wC_eq i1.isLt]
    exact hle1 ⟨e, he8⟩ (fun h => hne0 (congrArg Fin.val h))
  · intro e he hne0
    have he8 : e < 8 - 0 := by omega
    have hi18 : i1.val < 8 - 0 := i1.isLt
    rw [wC_eq he8, wC_eq hi18]
    exact hlt1 ⟨e, he8⟩ he (fun h => hne0 (congrArg Fin.val h))

end Claims.MoeRouter
