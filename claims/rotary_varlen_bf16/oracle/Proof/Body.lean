import Proof.Basic

/-!
# The instantiated program

`model d`'s body at program `i`'s context, with the grid indices of `Proof/Basic` written out.
-/

namespace Claims.RotaryVarlen

open TVA

/-- The kernel body at program `i`, unfolded: the prologue (metadata), the early return on an empty chunk, the offset line, and one of the two tails (non-interleaved or interleaved). -/
theorem body_eq (d : Inst) (i : Nat) :
    (model d).body ((launch d).ctx i) =
      (do
        let i_t0 : I64 := IntV.cast (IntV.ofNat (ax0 d i) : I32)
        let i_b : I64 := IntV.cast (IntV.ofNat (ax1 d i) : I32)
        let i_h : I32 := IntV.ofNat (ax2 d i)
        let st ← (if d.isVarlen then (do
            let v_n ← tl.load1 (d.ci.ptr 0 + i_t0 * (IntV.ofNat 2 : I64))
            let i_n : I32 := (← tl.convert d.idx .i32 (Tensor.scalar v_n)).item
            let v_t ← tl.load1 (d.ci.ptr 0 + i_t0 * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))
            let i_t : I64 := (← tl.convert d.idx .i64 (Tensor.scalar v_t)).item
            let v_b ← tl.load1 (d.cu.ptr 0 + i_n)
            let bos : I64 := (← tl.convert d.idx .i64 (Tensor.scalar v_b)).item
            let v_e ← tl.load1 (d.cu.ptr 0 + i_n + (IntV.ofNat 1 : I32))
            let eos : I64 := (← tl.convert d.idx .i64 (Tensor.scalar v_e)).item
            let T_n : I64 := eos - bos
            let x_h : Ptr .bf16 := d.x.ptr 0 + bos * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + i_h * (IntV.ofNat d.D : I32)
            let y_h : Ptr .bf16 := d.y.ptr 0 + bos * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + i_h * (IntV.ofNat d.D : I32)
            pure (i_t, x_h, y_h, T_n, i_n))
          else (do
            let x_h : Ptr .bf16 := d.x.ptr 0 + i_b * (IntV.cast (IntV.ofNat d.T : I32) : I64) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + i_h * (IntV.ofNat d.D : I32)
            let y_h : Ptr .bf16 := d.y.ptr 0 + i_b * (IntV.cast (IntV.ofNat d.T : I32) : I64) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + i_h * (IntV.ofNat d.D : I32)
            pure (i_t0, x_h, y_h, (IntV.cast (IntV.ofNat d.T : I32) : I64), (IntV.ofNat 0 : I32))) :
              Prog (I64 × Ptr .bf16 × Ptr .bf16 × I64 × I32))
        let (i_t, x_h, y_h, T_n, i_n) := st
        if IntV.le T_n (i_t * (IntV.ofNat d.BT : I64)) then
          return ()
        let o_t : Tensor [d.BT] I64 := i_t * (IntV.ofNat d.BT : I64) + ((tl.arange 0 d.BT).castInt : Tensor [d.BT] I64)
        let o_cs : Tensor [d.BT] I64 ← (if !d.offT then
            pure (o_t + (IntV.cast (IntV.ofNat d.off : I32) : I64))
          else (do
            let v_o ← tl.load1 (if d.isVarlen then d.offs.ptr 0 + i_n else d.offs.ptr 0 + i_b)
            let off_n : I64 := (← tl.convert d.offD .i64 (Tensor.scalar v_o)).item
            pure (o_t + off_n)) : Prog (Tensor [d.BT] I64))
        let m_t : Tensor [d.BT] Bool :=
          (((o_t.ltS (IntV.ofInt 0 : I64)).not.and (o_t.ltS T_n)).and
            (o_cs.ltS (IntV.ofInt 0 : I64)).not).and (o_cs.ltS (IntV.ofNat d.TR : I64))
        if !d.interleaved then (do
          let o_r : Tensor [d.BD / 2] I32 := tl.arange 0 (d.BD / 2)
          let p_x : Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
            x_h + o_t.col (d.BD / 2) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + o_r.row d.BT
          let p_cos : Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
            d.cos.ptr 0 + (o_cs.col (d.BD / 2) * (IntV.ofNat d.R : I64) + ((o_r.row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64))
          let p_sin : Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
            d.sin.ptr 0 + (o_cs.col (d.BD / 2) * (IntV.ofNat d.R : I64) + ((o_r.row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64))
          let mask : Tensor [d.BT, d.BD / 2] Bool :=
            (m_t.col (d.BD / 2)).and ((o_r.ltS (IntV.ofNat d.R : I32)).row d.BT)
          let one_bf : Tensor [d.BT, d.BD / 2] BF16 ←
            tl.convert .f32 .bf16 (tl.full [d.BT, d.BD / 2] (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
          let l_cos ← tl.load p_cos (some mask) (some one_bf)
          let b_cos : Tensor [d.BT, d.BD / 2] F32 ← tl.cvt l_cos
          let zero_bf : Tensor [d.BT, d.BD / 2] BF16 ←
            tl.convert .f32 .bf16 (tl.full [d.BT, d.BD / 2] (FloatV.litF32 (FloatV.ofPat 0)))
          let l_sin ← tl.load p_sin (some mask) (some zero_bf)
          let b_sin : Tensor [d.BT, d.BD / 2] F32 ← tl.cvt l_sin
          let l_x0 ← tl.load p_x (some mask) (some zero_bf)
          let b_x0 : Tensor [d.BT, d.BD / 2] F32 ← tl.cvt l_x0
          let l_x1 ← tl.load (p_x + (bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32)) (some mask) (some zero_bf)
          let b_x1 : Tensor [d.BT, d.BD / 2] F32 ← tl.cvt l_x1
          let b_o0 ← tl.subc (← tl.mul b_x0 b_cos) (← tl.mul b_x1 b_sin)
          let b_o1 ← tl.addc (← tl.mul b_x0 b_sin) (← tl.mul b_x1 b_cos)
          let p_y : Tensor [d.BT, d.BD / 2] (Ptr .bf16) :=
            y_h + (o_t.col (d.BD / 2) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) +
              ((o_r.row d.BT).castInt : Tensor [d.BT, d.BD / 2] I64))
          let s_o0 : Tensor [d.BT, d.BD / 2] BF16 ← tl.cvt b_o0
          tl.store p_y s_o0 (some mask)
          let s_o1 : Tensor [d.BT, d.BD / 2] BF16 ← tl.cvt b_o1
          tl.store (p_y + (bc (IntV.ofNat d.R : I32) : Tensor [d.BT, d.BD / 2] I32)) s_o1 (some mask))
        else (do
          let o_d : Tensor [d.BD] I32 := tl.arange 0 d.BD
          let r_swap ← tl.irem (o_d + (IntV.ofNat 1 : I32)) (bc (IntV.ofNat 2 : I32))
          let o_d_swap : Tensor [d.BD] I32 := o_d + r_swap * (IntV.ofNat 2 : I32) - bc (IntV.ofNat 1 : I32)
          let o_d_repeat ← tl.idiv (tl.arange 0 d.BD) (bc (IntV.ofNat 2 : I32))
          let p_x0 : Tensor [d.BT, d.BD] (Ptr .bf16) :=
            x_h + o_t.col d.BD * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + o_d.row d.BT
          let p_x1 : Tensor [d.BT, d.BD] (Ptr .bf16) :=
            x_h + o_t.col d.BD * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + o_d_swap.row d.BT
          let p_cos : Tensor [d.BT, d.BD] (Ptr .bf16) :=
            d.cos.ptr 0 + (o_cs.col d.BD * (IntV.ofNat d.R : I64) + ((o_d_repeat.row d.BT).castInt : Tensor [d.BT, d.BD] I64))
          let p_sin : Tensor [d.BT, d.BD] (Ptr .bf16) :=
            d.sin.ptr 0 + (o_cs.col d.BD * (IntV.ofNat d.R : I64) + ((o_d_repeat.row d.BT).castInt : Tensor [d.BT, d.BD] I64))
          let mask : Tensor [d.BT, d.BD] Bool :=
            (m_t.col d.BD).and ((o_d_repeat.ltS (IntV.ofNat d.R : I32)).row d.BT)
          let one_bf : Tensor [d.BT, d.BD] BF16 ←
            tl.convert .f32 .bf16 (tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000)))
          let l_cos ← tl.load p_cos (some mask) (some one_bf)
          let b_cos : Tensor [d.BT, d.BD] F32 ← tl.cvt l_cos
          let zero_bf : Tensor [d.BT, d.BD] BF16 ←
            tl.convert .f32 .bf16 (tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0)))
          let l_sin ← tl.load p_sin (some mask) (some zero_bf)
          let b_sin : Tensor [d.BT, d.BD] F32 ← tl.cvt l_sin
          let l_x0 ← tl.load p_x0 (some mask) (some zero_bf)
          let b_x0 : Tensor [d.BT, d.BD] F32 ← tl.cvt l_x0
          let l_x1 ← tl.load p_x1 (some mask) (some zero_bf)
          let b_x1 : Tensor [d.BT, d.BD] F32 ← tl.cvt l_x1
          let b_o0 ← tl.mul b_x0 b_cos
          let b_o1 ← tl.mul b_x1 b_sin
          let r_par ← tl.irem o_d (bc (IntV.ofNat 2 : I32))
          let even : Tensor [d.BT, d.BD] Bool := (Tensor.eq r_par (bc (IntV.ofInt 0 : I32))).row d.BT
          let b_dif ← tl.subc b_o0 b_o1
          let b_sum ← tl.addc b_o0 b_o1
          let b_y : Tensor [d.BT, d.BD] F32 := tl.where even b_dif b_sum
          let p_y : Tensor [d.BT, d.BD] (Ptr .bf16) :=
            y_h + (o_t.col d.BD * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) +
              ((o_d.row d.BT).castInt : Tensor [d.BT, d.BD] I64))
          let s_y : Tensor [d.BT, d.BD] BF16 ← tl.cvt b_y
          tl.store p_y s_y (some mask))) := rfl

end Claims.RotaryVarlen
