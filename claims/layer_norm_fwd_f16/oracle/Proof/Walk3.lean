import Proof.OutLane

/-!
# Lines 86–95: the output loop
-/

namespace Claims.LayerNormFwdF16

open TVA

variable {d : Inst}

/-- What the output loop leaves, from memory `m`: only row `t` of `y` changed, and its elements are
finite and within `yTol` of the exact output. -/
def YDone (d : Inst) (init m m' : Mem) (t : Nat) : Prop :=
  (∀ z, z ∉ d.y.elems (rowEls d t) → m' z = m z) ∧
    ∀ j, j < d.N → ∃ c : F16, m'.read .f16 (d.y.ptr (idx d t j : Int)).addr = some c ∧
      c.isFinite = true ∧ |c.toReal - yRef d init t j| ≤ yTol d init t j

set_option maxHeartbeats 1600000 in
/-- **The output loop** (lines 86–95): row `t` of `y` receives the `float16` values
`(x − mean) · rstd · w + b`, each within `yTol` of the exact one. -/
theorem safe_out (hd : d.ok) {init : Mem} (hpre : Pre d init) {t : Nat} (ht : t < d.M) {c : Ctx}
    (hc : c.pid 0 = t) {m : Mem} (hm : MemOk d init m) {E : Env} {W : Set ℤ}
    (hW : d.y.elems (rowEls d t) ⊆ W) (mq rs : Tensor [] F32) (hmq : mq.item.isFinite = true)
    (hmq' : |mq.item.toReal - (rowP d hd init hpre t ht).mu| ≤
      11 / 20 * (rowP d hd init hpre t ht).Delta)
    (hrs : RstdOK (rowP d hd init hpre t ht) rs.item) :
    Safe E (R d t) W (fun _ m' => YDone d init m m' t) m
      (tl.forRange (IntV.ofInt 0 : I32) (IntV.ofNat d.N) (IntV.ofNat d.BLOCK) ()
        fun off _ =>
          (tl.load (d.w.ptr 0 + (off + tl.arange 0 d.BLOCK))
              (some ((off + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N)))).bind fun w =>
            (tl.load (d.b.ptr 0 + (off + tl.arange 0 d.BLOCK))
                (some ((off + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N)))).bind fun b =>
              (other16 d.BLOCK).bind fun o =>
                (tl.load (d.x.ptr 0 + tl.program_id c 0 * (IntV.ofNat d.N : I32) +
                      (off + tl.arange 0 d.BLOCK))
                    (some ((off + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N))) (some o)).bind
                  fun l => (tl.cvt l).bind fun x =>
                    (tl.subc x (bc mq.item)).bind fun dd =>
                      (tl.mul dd (bc rs.item)).bind fun xh =>
                        (tl.cvt w).bind fun w32 => (tl.cvt b).bind fun b32 =>
                          (tl.mul_add xh w32 b32).bind fun y => (tl.cvt y).bind fun y16 =>
                            tl.store (d.y.ptr 0 + tl.program_id c 0 * (IntV.ofNat d.N : I32) +
                                (off + tl.arange 0 d.BLOCK)) y16
                              (some ((off + tl.arange 0 d.BLOCK).ltS (IntV.ofNat d.N)))) := by
  have hN := hd.n_ge
  have hB := hd.block_le
  have hB31 : d.BLOCK ≤ 2 ^ 31 := by omega
  have hN31 : d.N < 2 ^ 31 := by have := hd.n_le; omega
  rw [forRange_one (by omega) hd.n_le_block (by omega)]
  refine safe_load_buf d.w (g := fun i : Idx [d.BLOCK - 0] => i.1.val)
    (en := fun i : Idx [d.BLOCK - 0] => decide (i.1.val < d.N))
    (Buf.addrs_of_get d.w (fun i => cols_mask hB31 hN31 i) fun i _ => vec_ptr hB31 d.w i)
    (fun i hi => ?_) (hm.valid hpre.2.1) ?_ fun wt hwt _ => ?_
  · exact lt_of_lt_of_le (by simpa using hi) hd.w_len
  · rintro _ ⟨_, ⟨i, hi, rfl⟩, hz⟩
    exact Or.inl (Or.inr ⟨i.1.val, show i.1.val < d.N by simpa using hi, hz⟩)
  refine safe_load_buf d.b (g := fun i : Idx [d.BLOCK - 0] => i.1.val)
    (en := fun i : Idx [d.BLOCK - 0] => decide (i.1.val < d.N))
    (Buf.addrs_of_get d.b (fun i => cols_mask hB31 hN31 i) fun i _ => vec_ptr hB31 d.b i)
    (fun i hi => ?_) (hm.valid hpre.2.2.1) ?_ fun bt hbt _ => ?_
  · exact lt_of_lt_of_le (by simpa using hi) hd.b_len
  · rintro _ ⟨_, ⟨i, hi, rfl⟩, hz⟩
    exact Or.inr ⟨i.1.val, show i.1.val < d.N by simpa using hi, hz⟩
  refine safe_loadX hd hpre ht hc hm fun a ha => ?_
  refine safe_subc fun dd hdd => ?_
  refine safe_mul fun xh hxh => ?_
  refine safe_cvt ⟨Format.binary16_std, Format.binary32_std, Or.inl rfl⟩ fun w32 hw32 => ?_
  refine safe_cvt ⟨Format.binary16_std, Format.binary32_std, Or.inl rfl⟩ fun b32 hb32 => ?_
  refine safe_mul_add fun y hy => ?_
  refine safe_cvt ⟨Format.binary32_std, Format.binary16_std, Or.inl rfl⟩ fun y16 hy16 => ?_
  refine safe_last ?_
  have hinj : ∀ i j : Idx [d.BLOCK - 0], decide (i.1.val < d.N) = true →
      decide (j.1.val < d.N) = true → idx d t i.1.val = idx d t j.1.val → i = j := by
    rintro ⟨i, ⟨⟩⟩ ⟨j, ⟨⟩⟩ _ _ h
    exact Prod.ext (Fin.ext (by simp only [idx] at h; dsimp only; omega)) rfl
  have himg : Buf.img (fun i : Idx [d.BLOCK - 0] => idx d t i.1.val)
      (fun i : Idx [d.BLOCK - 0] => decide (i.1.val < d.N)) ⊆ rowEls d t := by
    rintro _ ⟨i, hi, rfl⟩
    exact ⟨i.1.val, by simpa using hi, rfl⟩
  refine safe_store_buf d.y (g := fun i : Idx [d.BLOCK - 0] => idx d t i.1.val)
    (en := fun i : Idx [d.BLOCK - 0] => decide (i.1.val < d.N))
    (Buf.addrs_of_get d.y (fun i => cols_mask hB31 hN31 i) fun i _ => row_ptr hd ht hc d.y i)
    (fun i hi => lt_of_lt_of_le (idx_lt ht (by simpa using hi)) hd.y_len) hinj
    (hm.valid hpre.2.2.2.1) ((d.y.elems_mono himg).trans hW) fun m' hm1 hm2 _ => safe_ret ?_
  refine ⟨fun z hz => hm2 z fun hzi => hz (d.y.elems_mono himg hzi), fun j hj => ?_⟩
  have hjB : j < d.BLOCK := lt_of_lt_of_le hj hd.n_le_block
  set i : Idx [d.BLOCK - 0] := ((⟨j, hjB⟩ : Fin d.BLOCK), ()) with hi
  have hen : decide (i.1.val < d.N) = true := by simpa [hi] using hj
  refine ⟨y16.get i, hm1 i hen, ?_⟩
  have hwv : wt.get i = wv d init j := by
    have h1 := hwt i hen
    rw [hm.read_w (k := j) hj] at h1
    exact (rval_of_read h1).symm
  have hbv : bt.get i = bv d init j := by
    have h1 := hbt i hen
    rw [hm.read_b (k := j) hj] at h1
    exact (rval_of_read h1).symm
  obtain ⟨hfinw, hfinb⟩ := hpre.2.2.2.2.2.2.2.1 j hj
  have hroom := hpre.2.2.2.2.2.2.2.2 t ht j hj
  have hxv : (a.get i).toReal = (rowP d hd init hpre t ht).x j := by
    rw [(ha i).2]; exact ite_eq_left hj
  have key := out_lane (rowP d hd init hpre t ht) hj hmq hmq' hrs (ha i).1 hxv (hdd i) (hxh i)
    (hwv ▸ hfinw) (hbv ▸ hfinb) (hw32 i) (hb32 i) (hy i) (hy16 i) (hwv ▸ hbv ▸ hroom)
  rw [hwv, hbv] at key
  exact key

end Claims.LayerNormFwdF16
