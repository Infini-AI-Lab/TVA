import AttnFwdE5m2.Row

/-!
# `attn_fwd_e5m2`: the tiles a program loads and stores

* `safe_load_rows`: `desc.load([O, 0])` of `BM` rows through `descOf d b BM` (lines 72, 221);
* `safe_store_rows`: `desc.store([O, 0], ·)` of `BM` rows (line 245);
* `safe_load_v`: `desc_v.load([0, O])` of a `[HEAD_DIM, 64]` block through `descV d` (line 98): the
  lanes inside the descriptor hold `v`'s elements, the others the padding `+0`.
-/

namespace Kernels.AttnFwdE5m2

open TVA TVA.Math.Float

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **A block of `BM` rows at row `O`** through `descOf d b BM`: every lane is inside, and lane
`(r, e)` holds element `(O + r)·HEAD_DIM + e` of `b`. -/
theorem safe_load_rows {d : Inst} (hd : d.ok) (b : Buf .f8e5) (hb : b.base % 16 = 0)
    (hlen : rows d * d.HEAD_DIM ≤ b.len) (hv : b.Valid m)
    (hR : b.elems (below (rows d * d.HEAD_DIM)) ⊆ R) {BM : Nat} {o : I32} {O : Nat}
    (ho : o.toInt = O) (hO : O + BM ≤ rows d) {k : Tensor [BM, d.HEAD_DIM] F8E5 → Prog α}
    (hk : ∀ v : Tensor [BM, d.HEAD_DIM] F8E5, (∀ (r : Fin BM) (e : Fin d.HEAD_DIM),
      m.read .f8e5 (b.ptr (((O + r) * d.HEAD_DIM + e : ℕ) : ℤ)).addr = some (v.get2 r e)) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.desc_load (bs := [BM, d.HEAD_DIM]) (descOf d b BM) [o, IntV.ofInt 0] >>= k) := by
  have hl := fun (r : Fin BM) (e : Fin d.HEAD_DIM) => lane2 d b BM ho le_rfl r e
  have hidx : ∀ (r : Fin BM) (e : Fin d.HEAD_DIM), (O + r) * d.HEAD_DIM + e < rows d * d.HEAD_DIM := by
    intro r e
    have h1 : O + r + 1 ≤ rows d := by omega
    calc (O + r) * d.HEAD_DIM + e < (O + r) * d.HEAD_DIM + d.HEAD_DIM := by omega
      _ = (O + r + 1) * d.HEAD_DIM := by ring
      _ ≤ rows d * d.HEAD_DIM := Nat.mul_le_mul_right _ h1
  refine safe_desc_load b rfl (access_descOf hd hb o) ?_ hv ?_ fun v hv1 _ => hk v fun r e => ?_
  · rintro ⟨r, e, ⟨⟩⟩ _
    rw [(hl r e).2]; exact lt_of_lt_of_le (hidx r e) hlen
  · refine Set.Subset.trans (b.elems_mono ?_) hR
    rintro _ ⟨⟨r, e, ⟨⟩⟩, -, rfl⟩
    simp only [below, Set.mem_ofPred_eq]
    rw [(hl r e).2]; exact hidx r e
  · have h := hv1 (r, e, ()) (by rw [(hl r e).1]; simp; omega)
    rw [(hl r e).2] at h
    exact h

/-- **A block of `BM` rows stored at row `O`** through `descOf d b BM`: afterwards element
`(O + r)·HEAD_DIM + e` of `b` holds lane `(r, e)`, and nothing else changed. -/
theorem safe_store_rows {d : Inst} (hd : d.ok) (b : Buf .f8e5) (hb : b.base % 16 = 0)
    (hlen : rows d * d.HEAD_DIM ≤ b.len) (hv : b.Valid m) {BM : Nat} {o : I32} {O : Nat}
    (ho : o.toInt = O) (hO : O + BM ≤ rows d) {vals : Tensor [BM, d.HEAD_DIM] F8E5}
    (hW : b.elems {x | ∃ r < BM, ∃ e < d.HEAD_DIM, x = (O + r) * d.HEAD_DIM + e} ⊆ W)
    {k : Unit → Prog α}
    (hk : ∀ m', (∀ (r : Fin BM) (e : Fin d.HEAD_DIM),
        m'.read .f8e5 (b.ptr (((O + r) * d.HEAD_DIM + e : ℕ) : ℤ)).addr = some (vals.get2 r e)) →
      (∀ x, x ∉ b.elems {x | ∃ r < BM, ∃ e < d.HEAD_DIM, x = (O + r) * d.HEAD_DIM + e} → m' x = m x) →
      (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R (W \ b.elems {x | ∃ r < BM, ∃ e < d.HEAD_DIM, x = (O + r) * d.HEAD_DIM + e}) Q m' (k ())) :
    Safe E R W Q m (tl.desc_store (descOf d b BM) [o, IntV.ofInt 0] vals >>= k) := by
  have hl := fun (r : Fin BM) (e : Fin d.HEAD_DIM) => lane2 d b BM ho le_rfl r e
  have hidx : ∀ (r : Fin BM) (e : Fin d.HEAD_DIM), (O + r) * d.HEAD_DIM + e < rows d * d.HEAD_DIM := by
    intro r e
    have h1 : O + r + 1 ≤ rows d := by omega
    calc (O + r) * d.HEAD_DIM + e < (O + r) * d.HEAD_DIM + d.HEAD_DIM := by omega
      _ = (O + r + 1) * d.HEAD_DIM := by ring
      _ ≤ rows d * d.HEAD_DIM := Nat.mul_le_mul_right _ h1
  have himg : Buf.img (fun i : Idx [BM, d.HEAD_DIM] => ((descOf d b BM).elemAt
      (TensorDesc.coord (([o, (IntV.ofInt 0 : I32)] : List I32).map IntV.toInt) i)).toNat)
      (fun i => (descOf d b BM).inbAt (TensorDesc.coord (([o, (IntV.ofInt 0 : I32)] : List I32).map IntV.toInt) i)) =
      {x | ∃ r < BM, ∃ e < d.HEAD_DIM, x = (O + r) * d.HEAD_DIM + e} := by
    ext x
    constructor
    · rintro ⟨⟨r, e, ⟨⟩⟩, -, rfl⟩
      exact ⟨r, r.2, e, e.2, (hl r e).2⟩
    · rintro ⟨r, hr, e, he, rfl⟩
      let i : Idx [BM, d.HEAD_DIM] := ((⟨r, hr⟩ : Fin BM), ((⟨e, he⟩ : Fin d.HEAD_DIM), (() : Unit)))
      refine ⟨i, ?_, ?_⟩
      · show (descOf d b BM).inbAt _ = true
        rw [(hl ⟨r, hr⟩ ⟨e, he⟩).1]; simp; omega
      · exact (hl ⟨r, hr⟩ ⟨e, he⟩).2
  refine safe_desc_store b rfl (access_descOf hd hb o) ?_ ?_ hv (by rw [himg]; exact hW) fun m' h1 h2 h3 => ?_
  · rintro ⟨r, e, ⟨⟩⟩ _
    rw [(hl r e).2]; exact lt_of_lt_of_le (hidx r e) hlen
  · rintro ⟨r, e, ⟨⟩⟩ ⟨r', e', ⟨⟩⟩ _ _ h
    rw [(hl r e).2, (hl r' e').2] at h
    have hr : r.val = r'.val := by
      by_contra hne
      rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
      · have : (O + r + 1) * d.HEAD_DIM ≤ (O + r') * d.HEAD_DIM := Nat.mul_le_mul_right _ (by omega)
        have := e.2; have := e'.2; nlinarith
      · have : (O + r' + 1) * d.HEAD_DIM ≤ (O + r) * d.HEAD_DIM := Nat.mul_le_mul_right _ (by omega)
        have := e.2; have := e'.2; nlinarith
    have he : e.val = e'.val := by rw [hr] at h; omega
    rw [Fin.ext hr, Fin.ext he]
  · rw [himg] at h2 ⊢
    refine hk m' (fun r e => ?_) h2 h3
    have h := h1 (r, e, ()) (by rw [(hl r e).1]; simp; omega)
    rw [(hl r e).2] at h
    exact h

/-- **A `[HEAD_DIM, 64]` block of `v` at column `O`** through `descV d` (line 98): lane `(e, jj)` is
inside when `O + jj < rows` and then holds element `e·N_CTX + O + jj` of `v`; otherwise it holds the
zero bit pattern. -/
theorem safe_load_v {d : Inst} (hd : d.ok) (hv : d.v.Valid m)
    (hR : d.v.elems (below (rows d * d.HEAD_DIM)) ⊆ R) {o : I32} {O : Nat}
    (ho : o.toInt = O) (ho16 : O % 16 = 0) {k : Tensor [d.HEAD_DIM, 64] F8E5 → Prog α}
    (hk : ∀ vb : Tensor [d.HEAD_DIM, 64] F8E5,
      (∀ (e : Fin d.HEAD_DIM) (jj : Fin 64), O + jj < rows d →
        m.read .f8e5 (d.v.ptr ((e * d.N_CTX + (O + jj) : ℕ) : ℤ)).addr = some (vb.get2 e jj)) →
      (∀ (e : Fin d.HEAD_DIM) (jj : Fin 64), rows d ≤ O + jj → vb.get2 e jj = FloatV.ofPat 0) →
      Safe E R W Q m (k vb)) :
    Safe E R W Q m (tl.desc_load (bs := [d.HEAD_DIM, 64]) (descV d) [IntV.ofInt 0, o] >>= k) := by
  have hl := fun (e : Fin d.HEAD_DIM) (jj : Fin 64) => laneV d ho e jj
  have hN : d.N_CTX ≤ rows d := by
    unfold rows; have := hd.zh_pos
    exact Nat.le_mul_of_pos_left _ (Nat.mul_pos this.1 this.2)
  have hidx : ∀ (e : Fin d.HEAD_DIM) (jj : Fin 64), O + jj < rows d →
      e * d.N_CTX + (O + jj) < rows d * d.HEAD_DIM := by
    intro e jj h
    have h1 : e.val + 1 ≤ d.HEAD_DIM := e.2
    have h2 : e * d.N_CTX ≤ e * rows d := Nat.mul_le_mul_left _ hN
    calc e * d.N_CTX + (O + jj) < e * rows d + rows d := by omega
      _ = (e.val + 1) * rows d := by ring
      _ ≤ d.HEAD_DIM * rows d := Nat.mul_le_mul_right _ h1
      _ = rows d * d.HEAD_DIM := by ring
  have hacc := access_descV hd o (by rw [ho]; omega)
  refine safe_desc_load d.v rfl hacc ?_ hv ?_ fun vb hv1 hv2 => hk vb (fun e jj hin => ?_) fun e jj hout => ?_
  · rintro ⟨e, jj, ⟨⟩⟩ hin
    rw [(hl e jj).1] at hin
    rw [(hl e jj).2]; exact lt_of_lt_of_le (hidx e jj (of_decide_eq_true hin)) hd.lens.2.2.1
  · refine Set.Subset.trans (d.v.elems_mono ?_) hR
    rintro _ ⟨⟨e, jj, ⟨⟩⟩, hin, rfl⟩
    simp only [below, Set.mem_ofPred_eq]
    dsimp only at hin
    rw [(hl e jj).1] at hin
    rw [(hl e jj).2]; exact hidx e jj (of_decide_eq_true hin)
  · have h := hv1 (e, jj, ()) (by rw [(hl e jj).1]; simp; omega)
    rw [(hl e jj).2] at h
    exact h
  · have h := hv2 (e, jj, ()) (by rw [(hl e jj).1]; simp; omega)
    exact h

end Kernels.AttnFwdE5m2
