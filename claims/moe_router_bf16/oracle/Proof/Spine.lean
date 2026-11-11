import Claim
import TVA.Proof
import Proof.Setup
import Proof.Addrs
import Proof.Final

/-!
# `moe_router_bf16`: one program's body, and its `Safe` walk

`body_eq` unfolds the model's body; `safe_program` walks it, load by load and operation by
operation, to the postcondition `TokenPost`.
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}

/-- Program `p`'s body, spelled out (the model's body is its `rfl` unfold). -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let pid := tl.program_id c 0
        let offsets := tl.arange 0 d.BLOCK
        let mask := offsets.ltS (IntV.ofNat d.hidden : I32)
        let expert_offsets := (tl.arange 0 8).col d.BLOCK
        let router_mask := mask.row 8
        let w_router ← tl.load
          (d.w.ptr 0 + expert_offsets * (IntV.ofNat d.hidden : I32) + offsets.row 8)
          (some router_mask) (some (bc (c32 0)))
        let x ← tl.load (d.x.ptr 0 + pid * (IntV.ofNat d.hidden : I32) + offsets) (some mask)
          (some (bc (FloatV.litBF16 (FloatV.ofPat 0))))
        let x32 : Tensor [d.BLOCK] F32 ← tl.cvt x
        let prod ← tl.mul w_router (x32.row 8)
        let logits ← tl.sum1c prod
        let logits_scaled ← tl.div logits (bc (c32 0x403E000000000000))
        let doubled ← tl.mul (bc (c32 0x4000000000000000)) logits_scaled
        let exped ← tl.exp doubled
        let top ← tl.sub exped (bc (c32 0x3FF0000000000000))
        let bottom ← tl.add exped (bc (c32 0x3FF0000000000000))
        let logits_softcapped ← tl.mul (← tl.div top bottom) (bc (c32 0x403E000000000000))
        let top1 := (← tl.argmaxAll true true logits_softcapped).2
        tl.store1 (d.outi.ptr 0 + (pid * (IntV.ofNat 2 : I32) + (IntV.ofInt 0 : I32))) top1
        let top1_v ← tl.maxAll true logits_softcapped
        let diffs ← tl.subc logits_softcapped (bc top1_v)
        let exps ← tl.exp diffs
        let den ← tl.sum exps
        let invsumexp : F32 := (← tl.div (Tensor.scalar (c32 0x3FF0000000000000)) (Tensor.scalar den)).item
        tl.store1 (d.outw.ptr 0 + (pid * (IntV.ofNat 2 : I32) + (IntV.ofInt 0 : I32))) invsumexp
        let masked := tl.where ((tl.arange 0 8).map (fun j => !(IntV.eq j top1))) logits_softcapped
          (bc (c32 0xFFF0000000000000))
        let top2 := (← tl.argmaxAll true true masked).2
        tl.store1 (d.outi.ptr 0 + (pid * (IntV.ofNat 2 : I32) + (IntV.ofInt 1 : I32))) top2
        let onehot32 : Tensor [8] F32 ← tl.convert .i1 .f32 ((tl.arange 0 8).map (fun j => IntV.eq j top2))
        let prod2 ← tl.mul logits_softcapped onehot32
        let top2_v ← tl.sumc prod2
        let wdiff ← tl.sub (Tensor.scalar top2_v) (Tensor.scalar top1_v)
        let w2e ← tl.exp wdiff
        let w1 : F32 := (← tl.mul w2e (Tensor.scalar invsumexp)).item
        tl.store1 (d.outw.ptr 0 + (pid * (IntV.ofNat 2 : I32) + (IntV.ofInt 1 : I32))) w1) := rfl

/-- The address of the store of `outi[t, 0]` (or `outw[t, 0]`): `buf.ptr (2 t)`. -/
theorem ptr0 {dt : DType} (b : Buf dt) (p : Nat) (hp : p < 2 ^ 30) :
    b.ptr 0 + ((IntV.ofNat p : I32) * IntV.ofNat 2 + (IntV.ofInt 0 : I32))
      = b.ptr (↑(2 * p)) := by
  have h2 : (IntV.ofNat 2 : I32).toInt = 2 := IntV.I32.toInt_ofNat (by norm_num)
  have hpv : (IntV.ofNat p : I32).toInt = (p : Int) := IntV.I32.toInt_ofNat (by omega)
  have hmul : ((IntV.ofNat p : I32) * (IntV.ofNat 2 : I32)).toInt = (p : Int) * 2 := by
    exact IntV.toInt_mul hpv h2 (by constructor <;> push_cast <;> omega)
  have h0 : (IntV.ofInt 0 : I32).toInt = (0 : Int) :=
    IntV.I32.toInt_ofInt (by constructor <;> norm_num)
  have hadd : ((IntV.ofNat p : I32) * IntV.ofNat 2 + (IntV.ofInt 0 : I32)).toInt
      = (p : Int) * 2 + 0 := by
    exact IntV.toInt_add hmul h0 (by constructor <;> push_cast <;> omega)
  show (b.ptr 0).offset (((IntV.ofNat p : I32) * IntV.ofNat 2 + (IntV.ofInt 0 : I32))).toInt
      = b.ptr (↑(2 * p))
  rw [Buf.ptr_offset, hadd]
  congr 1
  push_cast
  ring

/-- The address of the store of `outi[t, 1]` (or `outw[t, 1]`): `buf.ptr (2 t + 1)`. -/
theorem ptr1 {dt : DType} (b : Buf dt) (p : Nat) (hp : p < 2 ^ 30) :
    b.ptr 0 + ((IntV.ofNat p : I32) * IntV.ofNat 2 + (IntV.ofInt 1 : I32))
      = b.ptr (↑(2 * p + 1)) := by
  have h2 : (IntV.ofNat 2 : I32).toInt = 2 := IntV.I32.toInt_ofNat (by norm_num)
  have hpv : (IntV.ofNat p : I32).toInt = (p : Int) := IntV.I32.toInt_ofNat (by omega)
  have hmul : ((IntV.ofNat p : I32) * (IntV.ofNat 2 : I32)).toInt = (p : Int) * 2 := by
    exact IntV.toInt_mul hpv h2 (by constructor <;> push_cast <;> omega)
  have h0 : (IntV.ofInt 1 : I32).toInt = (1 : Int) :=
    IntV.I32.toInt_ofInt (by constructor <;> norm_num)
  have hadd : ((IntV.ofNat p : I32) * IntV.ofNat 2 + (IntV.ofInt 1 : I32)).toInt
      = (p : Int) * 2 + 1 := by
    exact IntV.toInt_add hmul h0 (by constructor <;> push_cast <;> omega)
  show (b.ptr 0).offset (((IntV.ofNat p : I32) * IntV.ofNat 2 + (IntV.ofInt 1 : I32))).toInt
      = b.ptr (↑(2 * p + 1))
  rw [Buf.ptr_offset, hadd]
  congr 1
  push_cast
  ring

/-- A scalar store writes one element: the element `2p` of the output. -/
theorem img_two (p : Nat) :
    Buf.img (fun (_ : Idx []) => 2 * p) (fun _ => true) ⊆ two p := by
  rintro j ⟨u, _, rfl⟩
  exact Or.inl rfl

/-- A scalar store writes one element: the element `2p + 1` of the output. -/
theorem img_two1 (p : Nat) :
    Buf.img (fun (_ : Idx []) => 2 * p + 1) (fun _ => true) ⊆ two p := by
  rintro j ⟨u, _, rfl⟩
  exact Or.inr rfl

/-- The two scalar-store element sets of one program are disjoint. -/
theorem img_disj (p : Nat) :
    Disjoint (Buf.img (fun (_ : Idx []) => 2 * p + 1) (fun _ => true))
      (Buf.img (fun (_ : Idx []) => 2 * p) (fun _ => true)) := by
  rw [Set.disjoint_left]
  rintro j ⟨u, -, h⟩ ⟨v, -, h'⟩
  beta_reduce at h h'
  omega

/-- A program that ends in a store, with no continuation, is the store bound to `ret`. -/
theorem safe_last {E : Env} {R W : Set Int} {Q : Unit → Mem → Prop} {m : Mem} {p : Prog Unit}
    (h : Safe E R W Q m (p >>= Prog.ret)) : Safe E R W Q m p := by
  rwa [Prog.bind_eq, Prog.bind_ret] at h

set_option maxHeartbeats 4000000 in
/-- **One program.** Program `i` of the grid, run alone from `init`, is safe, reads its row of `x` and all of `w`, writes elements `2i` and `2i+1` of the two outputs, and leaves `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {P : Platform}
    (hexp : Device.MathTotal P .exp .binary32)
    (hexpacc : Device.ExpF32 P)
    (hdiv : Device.DivTotal P .binary32)
    (hdivfull : Device.DivFull P)
    (hsum : Device.SumDepth P)
    {init : Mem} (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (R d init i) (W d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hpv : ((launch d).ctx i).pid 0 = (i : Nat) := launch_pid d i
  have hbs : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  have hpidlt : (i : Nat) < 2 ^ 31 := by have := hd.bs_le; omega
  have hp30 : (i : Nat) < 2 ^ 30 := by have := hd.bs_le; omega
  have hl2 : 2 * (i : Nat) + 1 < 2 * d.bs := by omega
  have hBLOCK : 0 < d.BLOCK := lt_of_lt_of_le hd.hidden_pos hd.hidden_le
  have hx_len := hd.x_len
  have hw_len := hd.w_len
  have hvx := hpre.1
  have hvw := hpre.2.1
  have hvo := hpre.2.2.1
  have hvi := hpre.2.2.2.1
  rw [body_eq]
  simp only [tl.program_id, hpv]
  -- the `w` tile load
  refine safe_load_buf d.w (w_addrs hd) ?_ hpre.2.1 ?_ ?_
  · intro j he
    have h1 : j.1.val < 8 := j.1.isLt
    have h2 : j.2.1.val < d.hidden := of_decide_eq_true he
    have hA : j.1.val * d.hidden + j.2.1.val < (j.1.val + 1) * d.hidden := by
      rw [Nat.add_mul, Nat.one_mul]; exact Nat.add_lt_add_left h2 _
    have hB : (j.1.val + 1) * d.hidden ≤ 8 * d.hidden := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt h1)
    show j.1.val * d.hidden + j.2.1.val < d.w.len
    omega
  · refine Set.Subset.trans (Buf.elems_mono d.w ?_) Set.subset_union_right
    rintro j ⟨j, he, rfl⟩
    have h1 : j.1.val < 8 := j.1.isLt
    have h2 : j.2.1.val < d.hidden := of_decide_eq_true he
    have hA : j.1.val * d.hidden + j.2.1.val < (j.1.val + 1) * d.hidden := by
      rw [Nat.add_mul, Nat.one_mul]; exact Nat.add_lt_add_left h2 _
    have hB : (j.1.val + 1) * d.hidden ≤ 8 * d.hidden := Nat.mul_le_mul_right _ (Nat.succ_le_of_lt h1)
    exact lt_of_lt_of_le hA hB
  · intro wv hwrd hwpad
    -- the `x` row load
    refine safe_load_buf (other := some (bc (FloatV.litBF16 (FloatV.ofPat 0)))) d.x
      (x_addrs hd (IntV.ofNat (i : Nat)) (i : Nat) ?_ hbs) ?_ hpre.1 ?_ ?_
    · rw [IntV.I32.toInt_ofNat hpidlt]
    · intro j he
      have h1 : j.1.val < d.hidden := of_decide_eq_true he
      have hA : (i : Nat) * d.hidden + j.1.val < ((i : Nat) + 1) * d.hidden := by
        rw [Nat.add_mul, Nat.one_mul]; exact Nat.add_lt_add_left h1 _
      have hB : ((i : Nat) + 1) * d.hidden ≤ d.bs * d.hidden :=
        Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hbs)
      show (i : Nat) * d.hidden + j.1.val < d.x.len
      omega
    · refine Set.Subset.trans (Buf.elems_mono d.x ?_) Set.subset_union_left
      rintro j ⟨j, he, rfl⟩
      have h1 : j.1.val < d.hidden := of_decide_eq_true he
      have hA : (i : Nat) * d.hidden + j.1.val < ((i : Nat) + 1) * d.hidden := by
        rw [Nat.add_mul, Nat.one_mul]; exact Nat.add_lt_add_left h1 _
      have hB : ((i : Nat) + 1) * d.hidden ≤ d.bs * d.hidden :=
        Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hbs)
      have hlt := lt_of_lt_of_le hA hB
      show (i : Nat) * d.hidden ≤ gX d (i : Nat) j ∧ gX d (i : Nat) j < (i : Nat) * d.hidden + d.hidden
      constructor
      · show (i : Nat) * d.hidden ≤ (i : Nat) * d.hidden + j.1.val
        omega
      · show (i : Nat) * d.hidden + j.1.val < (i : Nat) * d.hidden + d.hidden
        omega
    · intro xv hxrd hxpad
      -- the conversion
      refine safe_cvt (f := .bfloat16) (g := .binary32) (rnd := .rn) ?_ ?_
      · exact ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩
      · intro x32 hx32
        -- the products
        refine safe_mul ?_
        intro prod hprod
        -- the row sums
        refine safe_sum1c hBLOCK ?_
        intro logits hlogits
        -- `/ 30`
        refine safe_div (fun j => hdiv _ _) ?_
        intro lsc hls
        -- `* 2`
        refine safe_mul ?_
        intro doubled hdoubled
        -- `exp`
        refine safe_math (fn := .exp) (fun j => hexp _) ?_
        intro exped hexped
        -- `- 1`
        refine safe_sub ?_
        intro top htop
        -- `+ 1`
        refine safe_add ?_
        intro bottom hbottom
        -- `/`
        refine safe_div (fun j => hdiv _ _) ?_
        intro q hq
        -- `* 30`
        refine safe_mul ?_
        intro capped hcapped
        -- the first argmax
        refine safe_argmaxAll (hn := by decide) true true ?_
        intro pair1 hpair1
        -- store `topk_ids[t, 0]`
        rw [store1_eq]
        refine safe_store_buf (g := fun _ => 2 * (i : Nat)) (en := fun _ => true) d.outi ?_ ?_
          (fun (i j : Idx []) _ _ _ => by cases i; cases j; rfl) hvi ?_ ?_
        · -- the address
          refine Buf.addrs_of_get_all d.outi (fun _ => ?_)
          rw [get_scalar_eq]
          exact ptr0 d.outi (i : Nat) hp30
        · intro j _; have := hd.outi_len; omega
        · exact Set.Subset.trans (Buf.elems_mono d.outi (img_two (i : Nat)))
            Set.subset_union_right
        · intro m1 hrd1 hfr1 hpr1
          have hv1 : d.outi.Valid m1 :=
            Buf.valid_of_agree hvi (fun x hx => hpr1 x (hvi.2.2.2 x hx))
          have hvW1 : d.outw.Valid m1 :=
            Buf.valid_of_agree hvo (fun x hx => hpr1 x (hvo.2.2.2 x hx))
          -- the max
          refine safe_maxAll (hn := by decide) true ?_
          intro top1v htop1v
          -- `capped - max`
          refine safe_subc ?_
          intro diffs hdiffs
          -- `exp`
          refine safe_math (fn := .exp) (fun j => hexp _) ?_
          intro exps hexps
          -- `sum`
          refine safe_sum (by decide) ?_
          intro den hden
          -- `1 / den`
          refine safe_div (fun j => hdiv _ _) ?_
          intro qinv hinv
          -- store `topk_weights[t, 0]`
          rw [store1_eq]
          refine safe_store_buf (g := fun _ => 2 * (i : Nat)) (en := fun _ => true) d.outw ?_ ?_
            (fun (i j : Idx []) _ _ _ => by cases i; cases j; rfl) hvW1 ?_ ?_
          · refine Buf.addrs_of_get_all d.outw (fun _ => ?_)
            rw [get_scalar_eq]
            exact ptr0 d.outw (i : Nat) hp30
          · intro j _; have := hd.outw_len; omega
          · intro x hx
            refine ⟨Or.inl (Buf.elems_mono d.outw (img_two (i : Nat)) hx), ?_⟩
            exact fun hx' => Set.disjoint_left.mp
              (Buf.elems_disjoint_of d.outw d.outi hd.outw_outi
                (img0_lt (Nat.lt_of_lt_of_le hl2 hd.outw_len))
                (img0_lt (Nat.lt_of_lt_of_le hl2 hd.outi_len))) hx hx'
          · intro m2 hrd2 hfr2 hpr2
            have hvI2 : d.outi.Valid m2 :=
              Buf.valid_of_agree hv1 (fun x hx => hpr2 x (hv1.2.2.2 x hx))
            have hv2 : d.outw.Valid m2 :=
              Buf.valid_of_agree hvW1 (fun x hx => hpr2 x (hvW1.2.2.2 x hx))
            -- the second argmax (the `tl.where` is pure)
            refine safe_argmaxAll (hn := by decide) true true ?_
            intro pair2 hpair2
            -- store `topk_ids[t, 1]`
            rw [store1_eq]
            refine safe_store_buf (g := fun _ => 2 * (i : Nat) + 1) (en := fun _ => true) d.outi ?_
              ?_ (fun (i j : Idx []) _ _ _ => by cases i; cases j; rfl) hvI2 ?_ ?_
            · refine Buf.addrs_of_get_all d.outi (fun _ => ?_)
              rw [get_scalar_eq]
              exact ptr1 d.outi (i : Nat) hp30
            · intro j _; have := hd.outi_len; omega
            · intro x hx
              refine ⟨⟨Or.inr (Buf.elems_mono d.outi (img_two1 (i : Nat)) hx), ?_⟩, ?_⟩
              · exact fun h => Set.disjoint_left.mp
                  (Buf.elems_disjoint d.outi (img_disj (i : Nat))) hx h
              · exact fun h => Set.disjoint_left.mp
                  (Buf.elems_disjoint_of d.outw d.outi hd.outw_outi
                    (img0_lt (Nat.lt_of_lt_of_le hl2 hd.outw_len))
                    (img1_lt (Nat.lt_of_lt_of_le hl2 hd.outi_len))) h hx
            · intro m3 hrd3 hfr3 hpr3
              have hv3 : d.outi.Valid m3 :=
                Buf.valid_of_agree hvI2 (fun x hx => hpr3 x (hvI2.2.2.2 x hx))
              have hvO3 : d.outw.Valid m3 :=
                Buf.valid_of_agree hv2 (fun x hx => hpr3 x (hv2.2.2.2 x hx))
              -- the one-hot conversion
              refine safe_convert ?_ ?_
              · intro j
                exact ⟨FloatV.ofRat (((if ((tl.arange 0 8).map (fun k => IntV.eq k pair2.2)).get j
                  then (1 : Int) else 0) : Rat)), Or.inl (by push_cast; split <;> rfl)⟩
              · intro onehot32 honehot
                -- `capped * onehot`
                refine safe_mul ?_
                intro prod2 hprod2
                -- `sumc`
                refine safe_sumc (by decide) ?_
                intro top2v htop2v
                -- `top2 - top1`
                refine safe_sub ?_
                intro wdiff hwdiff
                -- `exp`
                refine safe_math (fn := .exp) (fun j => hexp _) ?_
                intro w2e hw2e
                -- `* invsumexp`
                refine safe_mul ?_
                intro w1v hw1v
                -- store `topk_weights[t, 1]`
                rw [store1_eq]
                refine safe_last ?_
                refine safe_store_buf (g := fun _ => 2 * (i : Nat) + 1) (en := fun _ => true)
                  (k := Prog.ret) d.outw ?_ ?_
                  (fun (i j : Idx []) _ _ _ => by cases i; cases j; rfl) hvO3 ?_ ?_
                · refine Buf.addrs_of_get_all d.outw (fun _ => ?_)
                  rw [get_scalar_eq]
                  exact ptr1 d.outw (i : Nat) hp30
                · intro j _; have := hd.outw_len; omega
                · intro x hx
                  refine ⟨⟨⟨Or.inl (Buf.elems_mono d.outw (img_two1 (i : Nat)) hx), ?_⟩, ?_⟩, ?_⟩
                  · exact fun h => Set.disjoint_left.mp
                      (Buf.elems_disjoint_of d.outw d.outi hd.outw_outi
                        (img1_lt (Nat.lt_of_lt_of_le hl2 hd.outw_len))
                        (img0_lt (Nat.lt_of_lt_of_le hl2 hd.outi_len))) hx h
                  · exact fun h => Set.disjoint_left.mp
                      (Buf.elems_disjoint d.outw (img_disj (i : Nat))) hx h
                  · exact fun h => Set.disjoint_left.mp
                      (Buf.elems_disjoint_of d.outw d.outi hd.outw_outi
                        (img1_lt (Nat.lt_of_lt_of_le hl2 hd.outw_len))
                        (img1_lt (Nat.lt_of_lt_of_le hl2 hd.outi_len))) hx h
                · intro m4 hrd4 hfr4 hpr4
                  exact safe_ret (token_post_final hd hexp hexpacc hdiv hdivfull hsum hpre i
                    hbs hl2 hwrd hwpad hxrd hxpad hx32 hprod hlogits hls hdoubled hexped htop
                    hbottom hq hcapped hpair1 hrd1 hfr1 htop1v hdiffs hexps hden hinv hrd2 hfr2
                    hpair2 hrd3 hfr3 honehot hprod2 htop2v hwdiff hw2e hw1v hrd4 hfr4)

end Claims.MoeRouter
