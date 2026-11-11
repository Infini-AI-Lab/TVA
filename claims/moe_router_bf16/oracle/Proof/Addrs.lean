import Claim
import TVA.Proof
import Proof.Setup

/-!
# `moe_router_bf16`: the two loads' address patterns

The element a lane of each load reads, and the proof that the load's pointer tensor and mask
address exactly those elements (under the mask).
-/

namespace Claims.MoeRouter

open TVA

open Block

variable {d : Inst}

/-- The element a lane of the `w_router` tile reads: expert `e`, hidden position `h`. -/
def gW (d : Inst) (i : Idx [8 - 0, d.BLOCK]) : Nat := i.1.val * d.hidden + i.2.1.val

/-- The element a lane of the input row reads. -/
def gX (d : Inst) (p : Nat) (i : Idx [d.BLOCK]) : Nat := p * d.hidden + i.1.val

/-- `hidden_dim` fits in `int32`. -/
theorem hidden31 (hd : d.ok) : d.hidden < 2 ^ 31 := by
  have := hd.hidden_le17; omega

/-- `BLOCK_SIZE` fits in `int32`. -/
theorem block31 (hd : d.ok) : d.BLOCK < 2 ^ 31 := by
  have := hd.block_le; omega

/-- Every lane of the `w_router` tile addresses its element of `w`, under the router mask. -/
theorem w_addrs (hd : d.ok) :
    d.w.Addrs
      (tl.addrs (d.w.ptr 0 + (tl.arange 0 8).col d.BLOCK * (IntV.ofNat d.hidden : I32)
          + (tl.arange 0 d.BLOCK).row 8)
        (some (((tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.hidden : I32)).row 8)))
      (gW d) (fun i : Idx [8 - 0, d.BLOCK] => decide (i.2.1.val < d.hidden)) := by
  have hh := hidden31 hd
  have hb31 := block31 hd
  refine Buf.addrs_of_get d.w (fun i => ?_) (fun i he => ?_)
  · obtain ⟨e, h, ⟨⟩⟩ := i
    have hB' : (IntV.ofNat d.hidden : I32).toInt = (d.hidden : Int) := IntV.I32.toInt_ofNat hh
    have hC : (IntV.ofInt (0 + (h : Nat)) : I32).toInt = (h.val : Int) := by
      rw [IntV.I32.toInt_ofInt (by constructor <;> omega)]
      ring
    have s0 : (((tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.hidden : I32)).row 8).get (e, h, ())
        = IntV.lt ((tl.arange 0 d.BLOCK).get (h, ())) (IntV.ofNat d.hidden : I32) := rfl
    rw [s0]
    have s1 : (tl.arange 0 d.BLOCK).get (h, ()) = IntV.ofInt (0 + (h : Nat)) := rfl
    rw [s1]
    rw [IntV.lt_eq, hC, hB']
    simp only [decide_eq_decide]; norm_cast
  · obtain ⟨e, h, ⟨⟩⟩ := i
    have he8 : e.val < 8 := by have := e.isLt; omega
    have hB' : (IntV.ofNat d.hidden : I32).toInt = (d.hidden : Int) := IntV.I32.toInt_ofNat hh
    have hA : (IntV.ofInt (0 + (e : Nat)) : I32).toInt = (e.val : Int) := by
      rw [IntV.I32.toInt_ofInt (by constructor <;> omega)]
      push_cast; ring
    have hC : (IntV.ofInt (0 + (h : Nat)) : I32).toInt = (h.val : Int) := by
      rw [IntV.I32.toInt_ofInt (by constructor <;> omega)]
      ring
    have hmul : (IntV.ofInt (0 + (e : Nat)) * (IntV.ofNat d.hidden : I32) : I32).toInt
        = (e.val : Int) * (d.hidden : Int) := by
      have hs : IntV.SFits 32 ((e.val : Int) * (d.hidden : Int)) := by
        have h0e : (0 : Int) ≤ (e.val : Int) := Int.natCast_nonneg _
        have h0h : (0 : Int) ≤ (d.hidden : Int) := Int.natCast_nonneg _
        have h1 : (e.val : Int) < 8 := by exact_mod_cast he8
        have h2 : (d.hidden : Int) ≤ 2 ^ 17 := by exact_mod_cast hd.hidden_le17
        have hbig : (e.val : Int) * (d.hidden : Int) < 2 ^ 20 := by nlinarith
        have hlow : -(2 : Int) ^ 20 ≤ (e.val : Int) * (d.hidden : Int) := by
          have := mul_nonneg h0e h0h; omega
        constructor <;> omega
      exact IntV.toInt_mul hA hB' hs
    have s0 : (d.w.ptr 0 + (tl.arange 0 8).col d.BLOCK * (IntV.ofNat d.hidden : I32)
        + (tl.arange 0 d.BLOCK).row 8).get (e, h, ())
        = ((d.w.ptr 0 + (tl.arange 0 8).col d.BLOCK * (IntV.ofNat d.hidden : I32)).get (e, h, ())).offset
            (((tl.arange 0 d.BLOCK).row 8).get (e, h, ())).toInt := rfl
    rw [s0]
    have s1 : ((tl.arange 0 d.BLOCK).row 8).get (e, h, ()) = IntV.ofInt (0 + (h : Nat)) := rfl
    rw [s1]
    have s2 : (d.w.ptr 0 + (tl.arange 0 8).col d.BLOCK * (IntV.ofNat d.hidden : I32)).get (e, h, ())
        = (d.w.ptr 0).offset (((tl.arange 0 8).col d.BLOCK * (IntV.ofNat d.hidden : I32)).get (e, h, ())).toInt := rfl
    rw [s2]
    have s3 : ((tl.arange 0 8).col d.BLOCK * (IntV.ofNat d.hidden : I32)).get (e, h, ())
        = (tl.arange 0 8).get (e, ()) * (IntV.ofNat d.hidden : I32) := rfl
    rw [s3]
    have s4 : (tl.arange 0 8).get (e, ()) = IntV.ofInt (0 + (e : Nat)) := rfl
    rw [s4]
    rw [hmul, hC, Ptr.offset_offset, Buf.ptr_offset]
    show d.w.ptr (0 + ((e.val : Int) * (d.hidden : Int) + (h.val : Int)))
        = d.w.ptr ↑(e.val * d.hidden + h.val)
    have hcast : ((e.val * d.hidden + h.val : Nat) : Int)
        = (e.val : Int) * (d.hidden : Int) + (h.val : Int) := by push_cast; ring
    rw [hcast, Int.zero_add]

/-- The element a lane of the input row reads: `x[pid, h]` under `mask`. -/
theorem x_addrs (hd : d.ok) (pid : I32) (p : Nat) (hp : pid.toInt = (p : Int))
    (hb : p < d.bs) :
    d.x.Addrs
      (tl.addrs ((d.x.ptr 0 + pid * (IntV.ofNat d.hidden : I32)) + tl.arange 0 d.BLOCK)
        (some ((tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.hidden : I32))))
      (gX d p) (fun i : Idx [d.BLOCK] => decide (i.1.val < d.hidden)) := by
  have hh := hidden31 hd
  refine Buf.addrs_of_get d.x (fun i => ?_) (fun i he => ?_)
  · obtain ⟨h, ⟨⟩⟩ := i
    have hb31 : d.BLOCK < 2 ^ 31 := block31 hd
    have hB' : (IntV.ofNat d.hidden : I32).toInt = (d.hidden : Int) := IntV.I32.toInt_ofNat hh
    have hC : (IntV.ofInt (0 + (h : Nat)) : I32).toInt = (h.val : Int) := by
      rw [IntV.I32.toInt_ofInt (by constructor <;> omega)]
      ring
    have s0 : ((tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.hidden : I32)).get (h, ())
        = IntV.lt ((tl.arange 0 d.BLOCK).get (h, ())) (IntV.ofNat d.hidden : I32) := rfl
    rw [s0]
    have s1 : (tl.arange 0 d.BLOCK).get (h, ()) = IntV.ofInt (0 + (h : Nat)) := rfl
    rw [s1]
    rw [IntV.lt_eq, hC, hB']
    simp only [decide_eq_decide]; norm_cast
  · obtain ⟨h, ⟨⟩⟩ := i
    have hb31 : d.BLOCK < 2 ^ 31 := block31 hd
    have hB' : (IntV.ofNat d.hidden : I32).toInt = (d.hidden : Int) := IntV.I32.toInt_ofNat hh
    have hC : (IntV.ofInt (0 + (h : Nat)) : I32).toInt = (h.val : Int) := by
      rw [IntV.I32.toInt_ofInt (by constructor <;> omega)]
      ring
    have hmul : (pid * (IntV.ofNat d.hidden : I32)).toInt = (p : Int) * (d.hidden : Int) := by
      have hs : IntV.SFits 32 ((p : Int) * (d.hidden : Int)) := by
        have h0p : (0 : Int) ≤ (p : Int) := Int.natCast_nonneg _
        have h0h : (0 : Int) ≤ (d.hidden : Int) := Int.natCast_nonneg _
        have hpl : p ≤ d.bs - 1 := by omega
        have hbig : (p : Int) * (d.hidden : Int) < 2 ^ 31 := by
          have h1 : p * d.hidden ≤ (d.bs - 1) * d.hidden := Nat.mul_le_mul_right _ hpl
          have h2 : p * d.hidden < 2 ^ 31 := lt_of_le_of_lt h1 hd.x_off
          exact_mod_cast h2
        have hlow : -(2 : Int) ^ 31 ≤ (p : Int) * (d.hidden : Int) := by
          have := mul_nonneg h0p h0h; omega
        constructor <;> omega
      exact IntV.toInt_mul hp hB' hs
    have s0 : ((d.x.ptr 0 + pid * (IntV.ofNat d.hidden : I32)) + tl.arange 0 d.BLOCK).get (h, ())
        = (d.x.ptr 0 + pid * (IntV.ofNat d.hidden : I32)).offset
            ((tl.arange 0 d.BLOCK).get (h, ())).toInt := rfl
    rw [s0]
    have s1 : (d.x.ptr 0 + pid * (IntV.ofNat d.hidden : I32))
        = (d.x.ptr 0).offset (pid * (IntV.ofNat d.hidden : I32)).toInt := rfl
    rw [s1]
    have s2 : (tl.arange 0 d.BLOCK).get (h, ()) = IntV.ofInt (0 + (h : Nat)) := rfl
    rw [s2]
    rw [hmul, hC, Ptr.offset_offset, Buf.ptr_offset]
    show d.x.ptr (0 + ((p : Int) * (d.hidden : Int) + (h.val : Int)))
        = d.x.ptr (((p * d.hidden + h.val : Nat)) : Int)
    have hcast : ((p * d.hidden + h.val : Nat) : Int)
        = (p : Int) * (d.hidden : Int) + (h.val : Int) := by push_cast; ring
    rw [hcast, Int.zero_add]

end Claims.MoeRouter
