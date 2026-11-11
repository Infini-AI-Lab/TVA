import Claim
import TVA.Proof
import Proof.Inv
import Proof.Trip

/-!
# The end of a trip

`safe_tail` walks the lines every trip ends with, with or without initial states: `tl.exp` of the
decay, the scale (`0` where the request changes without initial states), `tl.mul_add`, and the
store into the next slot. It proves the stored values within the claim's tolerance.
-/
namespace Claims.SsdStatePassing

open TVA

/-- The lanes of a block mask: lane `k` of `offs < n` is enabled when `pm·1024 + k < n`. -/
theorem mask_lane' {offs_m : Tensor [BL - 0] I32} {pm n : Nat} (hn : n < 2 ^ 31)
    (hoff : ∀ k : Fin BL, (offs_m.get1 k).toInt = ((pm * BL + k : Nat) : Int)) (k : Fin BL) :
    (offs_m.ltS (IntV.ofNat n : I32)).get1 k = decide (pm * BL + k < n) := by
  rw [Tensor.get1_ltS, IntV.lt_eq, hoff k, IntV.I32.toInt_ofNat hn]
  simp only [Nat.cast_lt]

/-- The lanes of an access to buffer elements: lane `k` addresses element `g k` when enabled. -/
theorem addrs_lane {dt : DType} (b : Buf dt) {pm n : Nat} {ptrs : Tensor [BL - 0] (Ptr dt)}
    {mask : Tensor [BL - 0] Bool} {g : Fin BL → Nat}
    (hm : ∀ k : Fin BL, mask.get1 k = decide (pm * BL + k < n))
    (hp : ∀ k : Fin BL, ptrs.get1 k = b.ptr (g k)) :
    b.Addrs (tl.addrs ptrs (some mask)) (fun i : Idx [BL - 0] => g i.1)
      (fun i : Idx [BL - 0] => decide (pm * BL + (i.1 : Nat) < n)) :=
  Buf.addrs_of_get b (fun i => hm i.1) (fun i _ => hp i.1)

/-- The elements the enabled lanes of a block access address through `idxf`. -/
def laneSet (idxf : Nat → Nat) (pm n : Nat) : Set Nat :=
  {j | ∃ k : Fin BL, pm * BL + k < n ∧ j = idxf (pm * BL + k)}

/-- **A store of a block of lanes** into the elements `idxf (pm·1024 + k)` of a buffer: the
enabled lanes hold their values afterwards and nothing else changed. -/
theorem safe_store_lanes {α : Type} {E : Env} {R W : Set Int} {Q : α → Mem → Prop} {m : Mem}
    (b : Buf .f32) (idxf : Nat → Nat) {pm n : Nat} {ptrs : Tensor [BL - 0] (Ptr .f32)}
    {mask : Tensor [BL - 0] Bool} {vals : Tensor [BL - 0] F32}
    (hinj : ∀ k k' : Fin BL, pm * BL + k < n → pm * BL + k' < n →
      idxf (pm * BL + k) = idxf (pm * BL + k') → k = k')
    (hin : ∀ k : Fin BL, pm * BL + k < n → idxf (pm * BL + k) < b.len)
    (hm : ∀ k : Fin BL, mask.get1 k = decide (pm * BL + k < n))
    (hp : ∀ k : Fin BL, ptrs.get1 k = b.ptr (idxf (pm * BL + k)))
    (hv : b.Valid m) (hW : b.elems (laneSet idxf pm n) ⊆ W) {k : Unit → Prog α}
    (hk : ∀ m' : Mem,
      (∀ k : Fin BL, pm * BL + k < n →
        m'.read .f32 (b.ptr (idxf (pm * BL + k) : Int)).addr = some (vals.get1 k)) →
      (∀ x, x ∉ b.elems (laneSet idxf pm n) → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems (laneSet idxf pm n)) Q m' (k ())) :
    Safe E R W Q m (tl.store ptrs vals (some mask) >>= k) := by
  have himg : Buf.img (fun i : Idx [BL - 0] => idxf (pm * BL + (i.1 : Nat)))
      (fun i : Idx [BL - 0] => decide (pm * BL + (i.1 : Nat) < n)) = laneSet idxf pm n := by
    ext j
    constructor
    · rintro ⟨i, hi, rfl⟩
      exact ⟨i.1, of_decide_eq_true hi, rfl⟩
    · rintro ⟨k, hk, rfl⟩
      exact ⟨(k, ()), decide_eq_true hk, rfl⟩
  have hA := addrs_lane b hm hp
  refine safe_store_buf b hA (fun i hi => hin i.1 (of_decide_eq_true hi)) ?_ hv (by rw [himg]; exact hW)
    (fun m' hr hf hpres => ?_)
  · intro i j hi hj e
    have := hinj i.1 j.1 (of_decide_eq_true hi) (of_decide_eq_true hj) e
    exact Prod.ext this rfl
  · rw [himg] at hf ⊢
    refine hk m' (fun k hk => ?_) hf hpres
    exact hr (k, ()) (decide_eq_true hk)

/-- What a trip leaves after its store into slot `t + 1`: the lanes' new values are within the
tolerance, they are in the slot, and nothing else changed. -/
def TailPost (d : Inst) (init : Mem) (pm h t : Nat) (m₁ : Mem) (r : Tensor [BL] F32) (m' : Mem) :
    Prop :=
  (∀ k : Fin BL, pm * BL + k < d.dim → OutO init d h (pm * BL + k) (t + 1) (r.get1 k)) ∧
  (∀ k : Fin BL, pm * BL + k < d.dim →
    slotRead d h (t + 1) (pm * BL + k) m' = some (r.get1 k)) ∧
  (∀ x, x ∉ slotW d h pm (t + 1) → m' x = m₁ x) ∧ (∀ x, m₁ x ≠ none → m' x ≠ none)


/-- **The end of a trip**: `tl.exp`, the scale, `tl.mul_add` and the store of the new state into
slot `t + 1` (`out` chunk `t + 1`, or `final_states` after the last chunk). With `sm` the scale
is `tl.exp(dA_cs)`, else `0`. The premises say what the state, the new contribution and the decay
are; the rest is the post-condition `TailPost` (for `g` of the result). -/
theorem safe_tail {α : Type} {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    {pm h t : Nat} (hh : h < d.nheads) (ht : t < d.nchunks)
    {m₁ : Mem} (hvo : d.out.Valid m₁) (hvf : d.finalst.Valid m₁)
    {offs_m : Tensor [BL - 0] I32} {out_ptrs final_ptrs : Tensor [BL - 0] (Ptr .f32)} {c : I32} (hc : c.toInt = t)
    (hoff : ∀ k : Fin BL, (offs_m.get1 k).toInt = ((pm * BL + k : Nat) : Int))
    (hop : ∀ k : Fin BL, out_ptrs.get1 k =
      d.out.ptr (stIdx d (t + 1) h (pm * BL + k) : Int))
    (hfp : ∀ k : Fin BL, final_ptrs.get1 k =
      d.finalst.ptr (finIdx d h (pm * BL + k) : Int))
    (sm : Bool) (dA_cs : F32) (states new_states : Tensor [BL] F32)
    (hS : ∀ k : Fin BL, pm * BL + k < d.dim → Vals init d → (states.get1 k).isFinite = true ∧
      (sm = true → |(states.get1 k).toReal - xRef init d h (pm * BL + k) t| ≤
        tolIn init d h (pm * BL + k) t))
    (hZ : sm = false → ∀ k : Fin BL, pm * BL + k < d.dim →
      xRef init d h (pm * BL + k) t = 0 ∧ tolIn init d h (pm * BL + k) t = 0)
    (hN : ∀ k : Fin BL, pm * BL + k < d.dim → Vals init d →
      (new_states.get1 k).isFinite = true ∧
        (new_states.get1 k).toReal = newV init d t h (pm * BL + k))
    (hA : sm = true → Vals init d → dA_cs.isFinite = true ∧
      |dA_cs.toReal - dRef init d h t| ≤ dErr init d h t)
    (g : Tensor [BL] F32 → α) :
    Safe ((launch d).env P) (Rin d) (slotW d h pm (t + 1))
      (fun x m' => ∃ r, x = g r ∧ TailPost d init pm h t m₁ r m') m₁
      (tl.exp (Tensor.scalar dA_cs) >>= fun e =>
        tl.mul_add (bc (tl.where (Tensor.scalar sm) e
          (Tensor.scalar (FloatV.litF32 (FloatV.ofPat 0)))).item) states new_states >>= fun r =>
        if IntV.lt c (IntV.ofNat d.nchunks - IntV.ofInt 1) = true then
          tl.store out_ptrs r (some (offs_m.ltS (IntV.ofNat d.dim))) >>= fun _ => pure (g r)
        else tl.store final_ptrs r (some (offs_m.ltS (IntV.ofNat d.dim))) >>= fun _ => pure (g r)) := by
  have hch := chunk_pos hd
  have hn31 : d.nchunks < 2 ^ 31 := by
    have h1 := hd.tok_lt
    have : d.nchunks ≤ d.nchunks * d.chunk := Nat.le_mul_of_pos_right _ hch
    omega
  have hdim31 : d.dim < 2 ^ 31 := by have := hd.dim_lt; omega
  have hnn : (IntV.ofNat d.nchunks : I32).toInt = d.nchunks := IntV.I32.toInt_ofNat hn31
  have h1v : (IntV.ofInt 1 : I32).toInt = ((1 : Nat) : Int) := by
    simpa using IntV.toInt_ofInt_one (w := 32) (by norm_num)
  have hn1 : (IntV.ofNat d.nchunks - IntV.ofInt 1 : I32).toInt = ((d.nchunks - 1 : Nat) : Int) :=
    IntV.I32.sub_nat hnn h1v (by have := nchunks_pos hd; omega) hn31
  have hcond : (IntV.lt c (IntV.ofNat d.nchunks - IntV.ofInt 1) = true) ↔ t + 1 < d.nchunks := by
    rw [IntV.lt_eq, decide_eq_true_eq, hc, hn1]
    omega
  have hmask := mask_lane' (offs_m := offs_m) (pm := pm) hdim31 hoff
  have hlane_mem : ∀ k : Fin BL, pm * BL + k < d.dim → pm * BL + k ∈ lanes d pm := by
    intro k hk; exact ⟨by omega, by have := k.isLt; omega, hk⟩
  have hsub : ∀ idxf : Nat → Nat, ∀ S : Set Nat, (∀ m ∈ lanes d pm, idxf m ∈ S) →
      laneSet idxf pm d.dim ⊆ S := by
    rintro idxf S hS j ⟨k, hk, rfl⟩
    exact hS _ (hlane_mem k hk)
  refine safe_math (fn := .exp) (fun _ => hP.mathTotal _ _ _) fun e he => ?_
  refine safe_mul_add fun r hr => ?_
  -- the values of the new state
  have hlane : ∀ k : Fin BL, pm * BL + k < d.dim → OutO init d h (pm * BL + k) (t + 1) (r.get1 k) := by
    intro k hk hV
    obtain ⟨hsf, hse⟩ := hS k hk hV
    obtain ⟨hnf, hnn'⟩ := hN k hk hV
    have hmul := hr (k, ())
    have hm : pm * BL + k < d.dim := hk
    cases sm with
    | true =>
      obtain ⟨hdf, hde⟩ := hA rfl hV
      have he' : P.math .exp .binary32 dA_cs (e.get ()) := he ()
      have hscale : (tl.where (Tensor.scalar true) e
          (Tensor.scalar (FloatV.litF32 (FloatV.ofPat 0)))).item = e.get () := rfl
      rw [hscale] at hmul
      obtain ⟨hef, hprod, hmag⟩ := prod_exp hP.expF32 hV hh hm ht hdf hde he' (hse rfl)
      exact chunk_final hV hh hm ht hef hsf hnf hnn' hprod hmag hmul
    | false =>
      obtain ⟨hx0, ht0⟩ := hZ rfl k hk
      have hscale : (tl.where (Tensor.scalar false) e
          (Tensor.scalar (FloatV.litF32 (FloatV.ofPat 0)))).item = FloatV.ofPat 0 := by
        simp [tl.where, Tensor.where_, Tensor.scalar, Tensor.item]
      rw [hscale] at hmul
      have hz : (FloatV.ofPat 0 : F32).toReal = 0 := by simp
      have hzf : (FloatV.ofPat 0 : F32).isFinite = true := FloatV.isFinite_ofPat_zero Format.binary32_regular
      refine chunk_final hV hh hm ht hzf hsf hnf hnn' ?_ ?_ hmul
      · rw [hz, hx0, ht0]; simp
      · rw [hz, hx0, ht0]; simp
  by_cases hlt : t + 1 < d.nchunks
  · rw [ite_eq_left (hcond.mpr hlt)]
    refine safe_store_lanes d.out (fun m => stIdx d (t + 1) h m) (n := d.dim) ?_ ?_ hmask hop hvo ?_
      (fun m' hread hframe hpres => safe_ret ?_)
    · intro k k' hk hk' e
      have := (stIdx_inj hh hh hk hk' e).2.2
      exact Fin.ext (by omega)
    · intro k hk
      exact lt_of_lt_of_le (stIdx_lt hlt hh hk) hd.out_len
    · rw [slotW_out hlt]
      exact d.out.elems_mono (hsub _ _ fun m hm => ⟨m, hm, rfl⟩)
    · have hW : d.out.elems (laneSet (fun m => stIdx d (t + 1) h m) pm d.dim) ⊆
          slotW d h pm (t + 1) := by
        rw [slotW_out hlt]
        exact d.out.elems_mono (hsub _ _ fun m hm => ⟨m, hm, rfl⟩)
      refine ⟨r, rfl, hlane, fun k hk => ?_, fun x hx => hframe x (fun hxs => hx (hW hxs)), hpres⟩
      unfold slotRead
      rw [ite_eq_left hlt]
      exact hread k hk
  · have hlt' : t + 1 = d.nchunks := by omega
    rw [ite_eq_right (fun h' => hlt (hcond.mp h'))]
    refine safe_store_lanes d.finalst (fun m => finIdx d h m) (n := d.dim) ?_ ?_ hmask hfp hvf ?_
      (fun m' hread hframe hpres => safe_ret ?_)
    · intro k k' hk hk' e
      have := (finIdx_inj hk hk' e).2
      exact Fin.ext (by omega)
    · intro k hk
      exact lt_of_lt_of_le (finIdx_lt hh hk) hd.fin_len
    · rw [hlt', slotW_fin]
      exact d.finalst.elems_mono (hsub _ _ fun m hm => ⟨m, hm, rfl⟩)
    · have hW : d.finalst.elems (laneSet (fun m => finIdx d h m) pm d.dim) ⊆
          slotW d h pm (t + 1) := by
        rw [hlt', slotW_fin]
        exact d.finalst.elems_mono (hsub _ _ fun m hm => ⟨m, hm, rfl⟩)
      refine ⟨r, rfl, hlane, fun k hk => ?_, fun x hx => hframe x (fun hxs => hx (hW hxs)), hpres⟩
      unfold slotRead
      rw [ite_eq_right hlt]
      exact hread k hk

end Claims.SsdStatePassing
