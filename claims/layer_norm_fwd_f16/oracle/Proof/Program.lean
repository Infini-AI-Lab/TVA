import Proof.Walk3
import Proof.Stores

/-!
# One program of the launch

`safe_program`: program `t` (row `t`) is `Safe` from the initial memory, reads inside `R d t`, writes
inside `W d t`, and ends with `RowDone`: the three phases of `Walk1`, `Walk2` and `Walk3` and the two
stores between them.
-/

namespace Claims.LayerNormFwdF16

open TVA

variable {d : Inst}

/-- The pointer `Mean + row` is element `t`. -/
theorem buf_ptr_pid {dy : DType} (B : Buf dy) {c : Ctx} {t : Nat} (ht31 : t < 2 ^ 31)
    (hc : c.pid 0 = t) : B.ptr 0 + tl.program_id c 0 = B.ptr (t : Int) := by
  have hpid : (tl.program_id c 0).toInt = (t : Int) := by
    rw [tl.program_id, hc]; exact IntV.I32.toInt_ofNat ht31
  rw [Ptr.add_int, hpid, Buf.ptr_offset]; simp

/-- The bytes of `mean[t]` are among the elements `{t}` of `mean`. -/
theorem mean_span_sub {t : Nat} (_ht : t < d.M) :
    Mem.span .f32 (d.mean.ptr (t : Int)).addr ⊆ d.mean.elems {t} :=
  d.mean.span_sub_elems rfl

/-- The bytes of `rstd[t]` are among the elements `{t}` of `rstd`. -/
theorem rstd_span_sub {t : Nat} :
    Mem.span .f32 (d.rstd.ptr (t : Int)).addr ⊆ d.rstd.elems {t} :=
  d.rstd.span_sub_elems rfl

/-- `mean[t]`'s element is among the elements of the `[M]` tensor. -/
theorem mean_els_sub {t : Nat} (ht : t < d.M) : d.mean.elems {t} ⊆ d.mean.elems (rowSet d) :=
  d.mean.elems_mono (by intro k hk; rw [Set.mem_singleton_iff.mp hk]; exact ht)

/-- Element `t` of `rstd` is among the elements of the `[M]` tensor. -/
theorem rstd_els_sub {t : Nat} (ht : t < d.M) : d.rstd.elems {t} ⊆ d.rstd.elems (rowSet d) :=
  d.rstd.elems_mono (by intro k hk; rw [Set.mem_singleton_iff.mp hk]; exact ht)

/-- Row `t` of `y` is among the elements of the `[M, N]` tensor. -/
theorem y_row_sub {t : Nat} (ht : t < d.M) : d.y.elems (rowEls d t) ⊆ d.y.elems (matSet d) :=
  d.y.elems_mono (by rintro _ ⟨j, hj, rfl⟩; exact idx_lt ht hj)

/-- Row `t` of `x` is among the elements of the `[M, N]` tensor. -/
theorem x_row_sub {t : Nat} (ht : t < d.M) : d.x.elems (rowEls d t) ⊆ d.x.elems (matSet d) :=
  d.x.elems_mono (by rintro _ ⟨j, hj, rfl⟩; exact idx_lt ht hj)


set_option maxHeartbeats 1600000 in
/-- **One program of the launch**: row `t` is processed from the initial memory, reads inside `R d t`
and writes inside `W d t`, and ends with `RowDone`. -/
theorem safe_program (hd : d.ok) {init : Mem} (hpre : Pre d init) {t : Nat} (ht : t < d.M)
    (E : Env) (hP : Device E.P) (c : Ctx) (hc : c.pid 0 = t) :
    Safe E (R d t) (W d t) (fun _ m => RowDone d init m t) init ((model d).body c) := by
  have ht31 : t < 2 ^ 31 := by
    have := hd.rows_lt; have := hd.n_ge
    have h2 : t * d.N ≤ (d.M - 1) * d.N := Nat.mul_le_mul_right _ (by omega)
    nlinarith
  have hm0 : MemOk d init init := ⟨fun _ _ => rfl, fun _ h => h⟩
  have hout_in : Disjoint (outBytes d) (inBytes d) := hd.out_in
  have hmean_in : Disjoint (d.mean.elems {t}) (inBytes d) :=
    hout_in.mono_left (fun z hz => Or.inl (Or.inr (mean_els_sub ht hz)))
  have hrstd_in : Disjoint (d.rstd.elems {t}) (inBytes d) :=
    hout_in.mono_left (fun z hz => Or.inr (rstd_els_sub ht hz))
  have hy_mean : Disjoint (d.y.elems (rowEls d t)) (d.mean.elems {t}) :=
    hd.y_mean.mono (y_row_sub ht) (mean_els_sub ht)
  have hy_rstd : Disjoint (d.y.elems (rowEls d t)) (d.rstd.elems {t}) :=
    hd.y_rstd.mono (y_row_sub ht) (rstd_els_sub ht)
  have hmean_rstd : Disjoint (d.mean.elems {t}) (d.rstd.elems {t}) :=
    hd.mean_rstd.mono (mean_els_sub ht) (rstd_els_sub ht)
  simp only [model, layer_norm_fwd_fused, Prog.bind_eq]
  refine safe_mean hd hpre ht hc hm0 hP fun mq hmqf hmq => ?_
  refine safe_rstd hd hpre ht hc hm0 hP mq hmqf hmq fun rs hrs => ?_
  rw [buf_ptr_pid d.mean ht31 hc, buf_ptr_pid d.rstd ht31 hc]
  have hspan_mean : Mem.span .f32 (d.mean.ptr (t : Int)).addr ⊆ W d t :=
    fun z hz => Or.inl (Or.inr (mean_span_sub ht hz))
  have hspan_rstd : Mem.span .f32 (d.rstd.ptr (t : Int)).addr ⊆ W d t :=
    fun z hz => Or.inr (rstd_span_sub hz)
  refine safe_store1 (d.mean.writable hpre.2.2.2.2.1 (lt_of_lt_of_le ht hd.mean_len))
    hspan_mean fun m1 hr1 hf1 => ?_
  have hm1 : MemOk d init m1 := hm0.store1 hr1 hf1 (hmean_in.mono_left (mean_span_sub ht))
  refine safe_store1 (d.rstd.writable (hm1.valid hpre.2.2.2.2.2.1) (lt_of_lt_of_le ht hd.rstd_len))
    (fun z hz => ⟨hspan_rstd hz, fun hs => Set.disjoint_left.mp hmean_rstd (mean_span_sub ht hs)
      (rstd_span_sub hz)⟩) fun m2 hr2 hf2 => ?_
  have hm2 : MemOk d init m2 := hm1.store1 hr2 hf2 (hrstd_in.mono_left (rstd_span_sub))
  have hW2 : d.y.elems (rowEls d t) ⊆
      (W d t \ Mem.span .f32 (d.mean.ptr (t : Int)).addr) \ Mem.span .f32 (d.rstd.ptr (t : Int)).addr :=
    fun z hz => ⟨⟨Or.inl (Or.inl hz), fun hs => Set.disjoint_left.mp hy_mean hz (mean_span_sub ht hs)⟩,
      fun hs => Set.disjoint_left.mp hy_rstd hz (rstd_span_sub hs)⟩
  refine Safe.mono (safe_out hd hpre ht hc hm2 hW2 mq rs hmqf hmq hrs) (fun _ m3 hY => ?_) le_rfl
  refine ⟨⟨mq.item, ?_, hmqf, ?_⟩, ⟨rs.item, ?_, hrs.1, hrs.2.2.1⟩, hY.2⟩
  · rw [← hr1]
    refine Mem.read_congr fun z hz => ?_
    have hzm := mean_span_sub ht hz
    rw [hY.1 z fun hy => Set.disjoint_left.mp hy_mean hy hzm,
      hf2 z fun hs => Set.disjoint_left.mp hmean_rstd hzm (rstd_span_sub hs)]
  · have hΔ := (rowP d hd init hpre t ht).Delta_nonneg
    show |mq.item.toReal - (rowP d hd init hpre t ht).mu| ≤ (rowP d hd init hpre t ht).Delta
    linarith
  · rw [← hr2]
    refine Mem.read_congr fun z hz => ?_
    exact hY.1 z fun hy => Set.disjoint_left.mp hy_rstd hy (rstd_span_sub hz)

end Claims.LayerNormFwdF16
