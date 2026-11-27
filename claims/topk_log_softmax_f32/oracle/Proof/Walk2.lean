import Proof.Walk1

/-!
# The sum loop

One trip of the sum loop (`safe_sumTrip`) reads block `t` of the row, takes the platform's `exp` of
each lane's rounded `x − mv`, zeroes the lanes past `vocab_size`, sums the block along a tree of
depth at most `10` and adds it to `se`: it keeps `SumInv` (`TVA.Proof` rules, then `sumInv_step`).
-/
namespace Claims.TopkLogSoftmax

open TVA

/-- A float reduction over `1024` positions has depth at most `10`. -/
lemma sumBound_le {P : Platform} (hsum : Device.SumDepth P) : tl.sumBound P (1024 - 0) ≤ 10 := by
  have h1 : Nat.clog 2 (1024 - 0) = 10 := by
    rw [show (1024 - 0 : Nat) = 2 ^ 10 by norm_num, Nat.clog_pow 2 10 (by norm_num)]
  unfold tl.sumBound
  rw [h1]
  exact max_le (h1 ▸ hsum (1024 - 0)) le_rfl

set_option maxHeartbeats 800000 in
/-- **One trip of the sum loop** reads block `t` of row `p` and keeps `SumInv` (on a `RowOk`
row). -/
lemma safe_sumTrip {d : Inst} (hd : d.ok) {E : Env} (htot : Device.MathTotal E.P .exp .binary32)
    (hsum : Device.SumDepth E.P) {init : Mem} (hv : d.logits.Valid init)
    {p t : Nat} (hp : p < d.batch) (ht : t * 1024 < d.vocab) {c : I32}
    (hc : c.toInt = ((t * 1024 : Nat) : Int)) {mv se : F32}
    (hI : RowOk (xsOf d init p) d.vocab → SumInv E.P d.vocab (xsOf d init p) mv t se)
    {Rp : Set Int} (hR : d.logits.elems (rowIn d p) ⊆ Rp) :
    Safe E Rp ∅ (fun st m₂ => m₂ = init ∧
        (RowOk (xsOf d init p) d.vocab → SumInv E.P d.vocab (xsOf d init p) mv (t + 1) st)) init
      (sumTrip (d.logits.ptr ((p * d.logits_stride : Nat) : Int)) (IntV.ofNat d.vocab) mv c se) := by
  have hV19 := hd.vocab_le
  have hfit : t * 1024 + 1024 ≤ 2 ^ 31 := by omega
  have hV31 : d.vocab < 2 ^ 31 := by omega
  unfold sumTrip
  refine safe_load_buf d.logits (blk_addrs d.logits (p * d.logits_stride) hc hfit hV31)
    ?_ hv ?_ fun v hv1 hv2 => ?_
  · intro i hi
    have := hd.logits_len p hp
    have := of_decide_eq_true hi
    omega
  · exact (Buf.elems_mono d.logits blk_img).trans hR
  refine safe_sub fun s hs => ?_
  refine safe_math (fn := .exp) (fun i => htot (s.get i)) fun e he => ?_
  refine safe_sum (by norm_num) fun b hb => ?_
  refine safe_add fun r hr => ?_
  refine safe_ret ⟨rfl, fun hrow => ?_⟩
  obtain ⟨tr, hcov, hdep, hev⟩ := hb
  have hbcget : ∀ i : Idx [1024 - 0], (bc mv : Tensor [1024 - 0] F32).get i = mv := fun _ => rfl
  simp only [hbcget] at hs
  refine sumInv_step (hI hrow) hcov (hdep.trans (sumBound_le hsum)) hev ?_ (Or.inl (hr ()))
  intro i hi
  have hi' : i < 1024 - 0 := by simpa using hi
  have hmask : ((c + tl.arange 0 1024).ltS (IntV.ofNat d.vocab : I32)).get1 ⟨i, hi'⟩ =
      decide (t * 1024 + i < d.vocab) := blk_mask hc hfit hV31 (⟨i, hi'⟩, ())
  have hw : (tl.where ((c + tl.arange 0 1024).ltS (IntV.ofNat d.vocab : I32)) e (bc zeroC)).get1
      ⟨i, hi'⟩ = if ((c + tl.arange 0 1024).ltS (IntV.ofNat d.vocab : I32)).get1 ⟨i, hi'⟩ then
        e.get1 ⟨i, hi'⟩ else zeroC := rfl
  rw [hmask] at hw
  unfold LaneE
  simp only [dif_pos hi']
  by_cases hk : t * 1024 + i < d.vocab
  · rw [if_pos hk]
    rw [if_pos (by simpa using hk)] at hw
    refine ⟨s.get1 ⟨i, hi'⟩, ?_, ?_⟩
    · have h1 := hv1 (⟨i, hi'⟩, ()) (decide_eq_true hk)
      have hx : v.get1 ⟨i, hi'⟩ = xsOf d init p (t * 1024 + i) :=
        (xsOf_of_read (d := d) (r := p) (k := t * 1024 + i) h1).symm
      have := hs (⟨i, hi'⟩, ())
      change ((v.get1 ⟨i, hi'⟩).sub mv).NaNEq (s.get1 ⟨i, hi'⟩) at this
      rwa [hx] at this
    · rw [hw]
      exact he (⟨i, hi'⟩, ())
  · rw [if_neg hk]
    rw [if_neg (by simpa using hk)] at hw
    exact hw

end Claims.TopkLogSoftmax
