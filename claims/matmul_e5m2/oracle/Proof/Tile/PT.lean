import Proof.Tile.Lane

/-!
# The pointer tensors of the `K`-loop

The `tl.forRange` loop of the kernel threads a state `(aPtr, bPtr, acc)`: each trip advances the
two pointer tensors by one step of `64` rows (columns) of `a` (`b`).  This file names the pointer
tensors after `t` trips (`aPtrT`, `bPtrT`), states the lane value each one has, and records the
two definitional equations (initial value, one-step successor) that the `Safe` chain of the loop
needs.
-/

namespace Claims.MatmulE5m2

open TVA

/-- The `a` pointer tensor after `t` trips of the `K`-loop. -/
def aPtrT (d : Inst) (p : Nat) : Nat → Tensor [128, 64] (Ptr .f8e5)
  | 0 =>
    d.a.ptr 0 + ((offsAMi d p).col 64 * (IntV.ofNat d.stride_am : I32) +
      (tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32))
  | t + 1 => aPtrT d p t + bc (s := [128, 64]) ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_ak : I32))

/-- The `b` pointer tensor after `t` trips of the `K`-loop. -/
def bPtrT (d : Inst) (p : Nat) : Nat → Tensor [64, 256] (Ptr .f8e5)
  | 0 =>
    d.b.ptr 0 + ((tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
      (offsBNi d p).row 64 * (IntV.ofNat d.stride_bn : I32))
  | t + 1 => bPtrT d p t + bc (s := [64, 256]) ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_bk : I32))

/-- The initial `a` pointer tensor is the model's. -/
theorem aPtrT_zero (d : Inst) (p : Nat) :
    aPtrT d p 0 =
      d.a.ptr 0 + ((offsAMi d p).col 64 * (IntV.ofNat d.stride_am : I32) +
        (tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32)) := rfl

/-- One trip advances the `a` pointer tensor by `64` columns of `a`. -/
theorem aPtrT_succ (d : Inst) (p t : Nat) :
    aPtrT d p (t + 1) =
      aPtrT d p t + bc (s := [128, 64]) ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_ak : I32)) := rfl

/-- The initial `b` pointer tensor is the model's. -/
theorem bPtrT_zero (d : Inst) (p : Nat) :
    bPtrT d p 0 =
      d.b.ptr 0 + ((tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
        (offsBNi d p).row 64 * (IntV.ofNat d.stride_bn : I32)) := rfl

/-- One trip advances the `b` pointer tensor by `64` rows of `b`. -/
theorem bPtrT_succ (d : Inst) (p t : Nat) :
    bPtrT d p (t + 1) =
      bPtrT d p t + bc (s := [64, 256]) ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_bk : I32)) := rfl

/-- Lane `(x, e)` of the `a` pointer tensor after `t` trips points at `a (om, 64·t + e)`. -/
theorem aPtrT_get (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (t : Nat)
    (x : Fin 128) (e : Fin 64) :
    (aPtrT d p t).get2 x e = d.a.ptr (aOff d p x.val (64 * t + e.val)) := by
  induction t with
  | zero =>
    rw [show (64 * 0 + e.val) = e.val from by omega]
    exact aPtrs_eq d hd hp (x, e, ())
  | succ t ih =>
    have h1 : (aPtrT d p (t + 1)).get2 x e =
        ((aPtrT d p t).get2 x e).offset
          (((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_ak : I32)).toInt) := by
      rw [aPtrT_succ]
      rfl
    rw [h1, ih, stepAK_toInt d hd, Buf.ptr_offset]
    rw [show ((aOff d p x.val (64 * t + e.val) : Nat) : ℤ) +
          64 * (d.stride_ak : ℤ) =
        ((aOff d p x.val (64 * (t + 1) + e.val) : Nat) : ℤ) from by
      unfold aOff
      push_cast
      ring]

/-- Lane `(e, y)` of the `b` pointer tensor after `t` trips points at `b (64·t + e, on)`. -/
theorem bPtrT_get (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (t : Nat)
    (e : Fin 64) (y : Fin 256) :
    (bPtrT d p t).get2 e y = d.b.ptr (bOff d p y.val (64 * t + e.val)) := by
  induction t with
  | zero =>
    rw [show (64 * 0 + e.val) = e.val from by omega]
    exact bPtrs_eq d hd hp (e, y, ())
  | succ t ih =>
    have h1 : (bPtrT d p (t + 1)).get2 e y =
        ((bPtrT d p t).get2 e y).offset
          (((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_bk : I32)).toInt) := by
      rw [bPtrT_succ]
      rfl
    rw [h1, ih, stepBK_toInt d hd, Buf.ptr_offset]
    rw [show ((bOff d p y.val (64 * t + e.val) : Nat) : ℤ) +
          64 * (d.stride_bk : ℤ) =
        ((bOff d p y.val (64 * (t + 1) + e.val) : Nat) : ℤ) from by
      unfold bOff
      push_cast
      ring]

end Claims.MatmulE5m2
