import Proof.Disj
import Proof.SafeAux

/-!
# `kv_cache_write_bf16`: the model's program is `Safe` in `Mode.fp8`

The program loads its token's slot, returns on a padding slot, and otherwise writes the lane's
element of the key and value caches with the saturating `float8e4nv` rounding of the element it
loaded divided by the scale.
-/

namespace Claims.KvCacheWrite

open TVA

/-- A scale read from a buffer's first element is its `scaleAt`. -/
theorem scaleAt_read {m : Mem} {b : Buf .f32} {v : F32}
    (h : m.read .f32 (b.ptr 0).addr = some v) : scaleAt m b = v := by
  simp only [scaleAt, h]

/-- The formats and rounding of the widening conversion `bfloat16 → float32`. -/
theorem cvt_ok_bf16_f32 :
    Format.Std .bfloat16 ∧ Format.Std .binary32 ∧
      tl.RoundOk .bfloat16 .binary32 .rn :=
  ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩

/-- The formats and rounding of the narrowing conversion `float32 → float8e4nv`. -/
theorem cvt8_ok_f32_e4m3 :
    (Format.binary32 = .binary16 ∨ Format.binary32 = .bfloat16 ∨
        Format.binary32 = .binary32) ∧
      Format.F8 .e4m3 ∧ tl.RoundOk .binary32 .e4m3 .rn :=
  ⟨Or.inr (Or.inr rfl), Or.inr rfl, Or.inl rfl⟩

/-- The cache element stored in `Mode.fp8`: the saturating fp8 conversion of a `float32` within
`DivFull`'s bound of `x / s`, where the `float32` is the widening conversion of `x`. -/
theorem stored_fp8 {P : Platform} {x : BF16} {s xf q : F32} {v : F8E4}
    (hdivf : Device.DivFull P) (hcvt : Device.CvtSatF32E4m3 P)
    (hnan : FloatV.NaNEq (FloatV.cvtWith .rn x) xf)
    (hdiv : P.div .binary32 xf s q) (hcv : P.cvt .binary32 .e4m3 .rn q v) :
    Stored .fp8 s x v := by
  intro hxfin hsf hs0 hbound
  have hw : (FloatV.cvtWith .rn x : F32).isFinite = true ∧
      (FloatV.cvtWith .rn x : F32).toRat = x.toRat :=
    FloatV.toRat_cvt_of_wider (f := .bfloat16) (g := .binary32)
      (by decide) (by decide) Format.wider_bfloat16_binary32 .rn hxfin
  have hnan0 : (FloatV.cvtWith .rn x : F32).isNaN = false := (FloatV.isFinite_iff.mp hw.1).1
  have hxf : xf = FloatV.cvtWith .rn x := FloatV.NaNEq.eq hnan hnan0
  have hxw : xf.toRat = x.toRat := by rw [hxf, hw.2]
  have hxff : xf.isFinite = true := hxf ▸ hw.1
  obtain ⟨hqfin, hqerr⟩ := hdivf xf s q hdiv hxff hsf hs0 (by rw [hxw]; exact hbound)
  have hsat := FloatV.cvtSat_rn_err (g := .e4m3) (hg := by decide) (x := q) hqfin
  have hsatnan : (FloatV.cvtSat .rn q : F8E4).isNaN = false :=
    (FloatV.isFinite_iff.mp hsat.1).1
  exact ⟨q, hqfin, by simpa only [hxw] using hqerr, FloatV.NaNEq.eq (hcvt q v hcv) hsatnan⟩

/-- In `Mode.fp8`, program `i` is `Safe`: it divides each loaded value by its scale and stores the
saturating `float8e4nv` rounding. -/
theorem safe_fp8 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (hdivT : Device.DivTotal E.P .binary32)
    (hdivF : Device.DivFull E.P)
    (hcvtT : Device.CvtTotal E.P .binary32 .e4m3)
    (hcvtS : Device.CvtSatF32E4m3 E.P)
    (i : Fin (launch d).size) (hm : d.mode = .fp8) :
    Safe E (R d) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m) init
      ((model d).body ((launch d).ctx i)) := by
  rcases d with ⟨m, nt, nh, hs, bs, nb, ts, kst, vst, bst, pst, hsd, key, value, kc, vc, sm,
    ksc, vsc⟩
  cases m
  · cases hm
  · let dd : Inst :=
      ⟨.fp8, nt, nh, hs, bs, nb, ts, kst, vst, bst, pst, hsd, key, value, kc, vc, sm, ksc, vsc⟩
    have hd' : dd.ok := hd
    have hpre' : Pre dd init := hpre
    change Safe E (R dd) (W dd init (i : Nat)) (fun _ m => Q dd init (i : Nat) m) init
      ((model dd).body ((launch dd).ctx i))
    rw [body_eq]
    dsimp only
    rw [slot_ptr_eq dd hd' i]
    rw [pid0_eq dd i, pid1_eq dd i]
    have hp : (i : Nat) < dd.num_tokens * numTiles dd := by
      simpa [launch_size] using i.isLt
    have htk : tk dd (i : Nat) < dd.slot_mapping.len :=
      Nat.lt_of_lt_of_le (tk_lt hp) hd'.slot_len
    have hslall : ∀ t, t < dd.num_tokens → slotAt dd init t < (numSlots dd : Int) :=
      hpre'.2.2.2.2.2.1
    have hsl : slotAt dd init (tk dd (i : Nat)) < (numSlots dd : Int) := slot_lt hslall hp
    obtain ⟨v0, hv0⟩ := dd.slot_mapping.read_some hpre'.2.2.2.2.1 htk
    refine safe_load1 (v0 := v0) hv0 ?_ ?_
    · exact (dd.slot_mapping.span_sub htk).trans (R_sub_slot dd)
    · refine safe_convert (d := .i64) (d' := .i64) (x := Tensor.scalar v0) ?_ ?_
      · intro u
        exact ⟨IntV.ofInt v0.toInt, rfl⟩
      · intro r hr
        have h1 : r.item.toInt = v0.toInt := by
          have h := conv_i64_i64_toInt (P := E.P) (hr ())
          simpa only [Tensor.item, Tensor.scalar] using h
        have hslot : slotAt dd init (tk dd (i : Nat)) = r.item.toInt := by
          simp only [slotAt, hv0]
          exact h1.symm
        by_cases hneg : IntV.lt r.item (IntV.ofInt 0 : I64) = true
        · simp only [hneg, ite_true]
          refine safe_ret ?_
          intro l hen
          exfalso
          have hlt : r.item.toInt < 0 := lt_zero_of_true hneg
          have hs0 : 0 ≤ slotAt dd init (tk dd (i : Nat)) := en_slot hen
          rw [hslot] at hs0
          omega
        · simp only [hneg]
          have hsl0 : 0 ≤ slotAt dd init (tk dd (i : Nat)) := by
            rw [hslot]
            exact le_zero_of_not_true hneg
          have hx0 : 0 ≤ r.item.toInt := by
            rw [← hslot]
            exact hsl0
          have hbs : 0 < dd.block_size := block_size_pos' hsl0 hsl
          have hy0 : 0 < (IntV.ofNat dd.block_size : I64).toInt := by
            rw [I64_toInt_ofNat hd'.block_size_lt]
            exact_mod_cast hbs
          refine safe_idiv1 (x := r.item) (y := (IntV.ofNat dd.block_size : I64)) hx0 hy0 ?_
          intro r1 hdiv1
          have hblk : r1.item.toInt = ((slotAt dd init (tk dd (i : Nat))).toNat /
              dd.block_size : Nat) := by
            rw [hdiv1, I64_toInt_ofNat hd'.block_size_lt, ← hslot]
            exact int_div_eq_nat_div hsl0
          refine safe_irem1 (x := r.item) (y := (IntV.ofNat dd.block_size : I64)) hx0 hy0 ?_
          intro r2 hmod1
          have hoff : r2.item.toInt = ((slotAt dd init (tk dd (i : Nat))).toNat %
              dd.block_size : Nat) := by
            rw [hmod1, I64_toInt_ofNat hd'.block_size_lt, ← hslot]
            exact int_mod_eq_nat_mod hsl0
          have htk2 : tk dd (i : Nat) < 2 ^ 31 :=
            Nat.lt_of_lt_of_le (tk_lt hp) hd'.tokens_lt.le
          refine safe_convert (d := .i32) (d' := .i64)
            (x := Tensor.scalar (IntV.ofNat (tk dd (i : Nat)) : I32)) ?_ ?_
          · intro u
            exact ⟨IntV.ofInt ((IntV.ofNat (tk dd (i : Nat)) : I32)).toInt, rfl⟩
          · intro tk1 htk1
            have hst : tk1.item.toInt = (tk dd (i : Nat) : Int) := by
              have h := conv_i32_i64_toInt (P := E.P) (htk1 ())
              simpa only [Tensor.item, Tensor.scalar, I32_toInt_ofNat htk2] using h
            refine safe_convert (d := .i32) (d' := .i64)
              (x := Tensor.scalar (IntV.ofNat (tk dd (i : Nat)) : I32)) ?_ ?_
            · intro u
              exact ⟨IntV.ofInt ((IntV.ofNat (tk dd (i : Nat)) : I32)).toInt, rfl⟩
            · intro tv htv
              have hstv : tv.item.toInt = (tk dd (i : Nat) : Int) := by
                have h := conv_i32_i64_toInt (P := E.P) (htv ())
                simpa only [Tensor.item, Tensor.scalar, I32_toInt_ofNat htk2] using h
              have hhs : 0 < dd.head_size := head_pos hd' hp
              have hdsz : (IntV.ofNat dd.head_size : I32).toInt = (dd.head_size : Int) :=
                I32_toInt_ofNat hd'.hs_lt
              have hfull : ∀ l : Idx [dd.TILE_SIZE],
                  ((tl.full [dd.TILE_SIZE] (IntV.ofNat dd.head_size : I32)).get l).toInt =
                    (dd.head_size : Int) := by
                intro l
                show (IntV.ofNat dd.head_size : I32).toInt = (dd.head_size : Int)
                exact hdsz
              refine safe_idiv (x := tpos dd (i : Nat))
                (y := tl.full [dd.TILE_SIZE] (IntV.ofNat dd.head_size : I32)) ?_ ?_
              · intro l
                have hxl : 0 ≤ ((tpos dd (i : Nat)).get l).toInt := by
                  rw [tpos_toInt hd' hp l]
                  positivity
                have hyl : 0 < ((tl.full [dd.TILE_SIZE]
                    (IntV.ofNat dd.head_size : I32)).get l).toInt := by
                  rw [hfull l]
                  exact_mod_cast hhs
                obtain ⟨q, hq, -⟩ := IntV.div?_of_nonneg hxl hyl
                exact ⟨q, hq⟩
              · intro curh hcurh
                have hcurH : ∀ l, (curh.get l).toInt = (pos dd (i : Nat) l / dd.head_size : Nat) := by
                  intro l
                  have hxl : 0 ≤ ((tpos dd (i : Nat)).get l).toInt := by
                    rw [tpos_toInt hd' hp l]
                    positivity
                  have hyl : 0 < ((tl.full [dd.TILE_SIZE]
                      (IntV.ofNat dd.head_size : I32)).get l).toInt := by
                    rw [hfull l]
                    exact_mod_cast hhs
                  obtain ⟨q, hq, hqv⟩ := IntV.div?_of_nonneg hxl hyl
                  have he : curh.get l = q := Option.some.inj ((hcurh l).symm.trans hq)
                  rw [he, hqv, tpos_toInt hd' hp l, hfull l]
                  simp
                refine safe_irem (x := tpos dd (i : Nat))
                  (y := tl.full [dd.TILE_SIZE] (IntV.ofNat dd.head_size : I32)) ?_ ?_
                · intro l
                  have hxl : 0 ≤ ((tpos dd (i : Nat)).get l).toInt := by
                    rw [tpos_toInt hd' hp l]
                    positivity
                  have hyl : 0 < ((tl.full [dd.TILE_SIZE]
                      (IntV.ofNat dd.head_size : I32)).get l).toInt := by
                    rw [hfull l]
                    exact_mod_cast hhs
                  obtain ⟨q, hq, -⟩ := IntV.rem?_of_nonneg hxl hyl
                  exact ⟨q, hq⟩
                · intro curd hcurd
                  have hcurD : ∀ l, (curd.get l).toInt =
                      (pos dd (i : Nat) l % dd.head_size : Nat) := by
                    intro l
                    have hxl : 0 ≤ ((tpos dd (i : Nat)).get l).toInt := by
                      rw [tpos_toInt hd' hp l]
                      positivity
                    have hyl : 0 < ((tl.full [dd.TILE_SIZE]
                        (IntV.ofNat dd.head_size : I32)).get l).toInt := by
                      rw [hfull l]
                      exact_mod_cast hhs
                    obtain ⟨q, hq, hqv⟩ := IntV.rem?_of_nonneg hxl hyl
                    have he : curd.get l = q := Option.some.inj ((hcurd l).symm.trans hq)
                    rw [he, hqv, tpos_toInt hd' hp l, hfull l]
                    simp
                  refine safe_convert (d := .i32) (d' := .i64) (x := curh) ?_ ?_
                  · intro l
                    exact ⟨IntV.ofInt (curh.get l).toInt, rfl⟩
                  · intro head64 hhead
                    have hhead' : ∀ l, (head64.get l).toInt =
                        (pos dd (i : Nat) l / dd.head_size : Nat) := by
                      intro l
                      exact (conv_i32_i64_toInt (P := E.P) (hhead l)).trans (hcurH l)
                    refine safe_convert (d := .i32) (d' := .i64) (x := curd) ?_ ?_
                    · intro l
                      exact ⟨IntV.ofInt (curd.get l).toInt, rfl⟩
                    · intro dim64 hdim
                      have hdim' : ∀ l, (dim64.get l).toInt =
                          (pos dd (i : Nat) l % dd.head_size : Nat) := by
                        intro l
                        exact (conv_i32_i64_toInt (P := E.P) (hdim l)).trans (hcurD l)
                      refine safe_load_buf (b := dd.key) (g := giK dd (i : Nat))
                        (en := en dd init (i : Nat)) (key_addrs (d := dd) hd' hp hsl0 hst) ?_ ?_ ?_ ?_
                      · intro l hen
                        exact giK_lt (d := dd) hd' hp hen
                      · exact hpre'.1
                      · refine (Buf.elems_sub_bytes dd.key
                          (S := Buf.img (giK dd (i : Nat)) (en dd init (i : Nat))) ?_).trans
                          (R_sub_key dd)
                        intro j hj
                        obtain ⟨l, hen, rfl⟩ := hj
                        exact giK_lt (d := dd) hd' hp hen
                      · intro keyv hkeyv hnoth
                        refine safe_cvt (f := .bfloat16) (g := .binary32) (x := keyv) ?_ ?_
                        · exact cvt_ok_bf16_f32
                        · intro keyf32 hkeyf
                          have hklen : 0 < dd.k_scale.len := (hd'.scales rfl).1
                          have hvk : dd.k_scale.Valid init := (hpre'.2.2.2.2.2.2.2 rfl).1
                          obtain ⟨sk, hsk⟩ := dd.k_scale.read_some hvk hklen
                          have hskv : scaleAt init dd.k_scale = sk := scaleAt_read hsk
                          refine safe_load1 (d := .f32) (p := dd.k_scale.ptr 0) (v0 := sk)
                            hsk ?_ ?_
                          · exact (dd.k_scale.span_sub hklen).trans (R_sub_kscale rfl)
                          · refine safe_div (x := keyf32) ?_ ?_
                            · intro l
                              exact hdivT (keyf32.get l) sk
                            · intro keyq hkeyq
                              refine safe_load_buf (b := dd.value) (g := giV dd (i : Nat))
                                (en := en dd init (i : Nat))
                                (value_addrs (d := dd) hd' hp hsl0 hstv) ?_ ?_ ?_ ?_
                              · intro l hen
                                exact giV_lt (d := dd) hd' hp hen
                              · exact hpre'.2.1
                              · refine (Buf.elems_sub_bytes dd.value
                                  (S := Buf.img (giV dd (i : Nat))
                                    (en dd init (i : Nat))) ?_).trans
                                  (R_sub_value dd)
                                intro j hj
                                obtain ⟨l, hen, rfl⟩ := hj
                                exact giV_lt (d := dd) hd' hp hen
                              · intro valv hvalv hnothv
                                refine safe_cvt (f := .bfloat16) (g := .binary32) (x := valv) ?_ ?_
                                · exact cvt_ok_bf16_f32
                                · intro valf32 hvalf
                                  have hvlen : 0 < dd.v_scale.len := (hd'.scales rfl).2.1
                                  have hvv : dd.v_scale.Valid init :=
                                    (hpre'.2.2.2.2.2.2.2 rfl).2
                                  obtain ⟨sv, hsv⟩ := dd.v_scale.read_some hvv hvlen
                                  have hsvv : scaleAt init dd.v_scale = sv := scaleAt_read hsv
                                  refine safe_load1 (d := .f32) (p := dd.v_scale.ptr 0)
                                    (v0 := sv) hsv ?_ ?_
                                  · exact (dd.v_scale.span_sub hvlen).trans (R_sub_vscale rfl)
                                  · refine safe_div (x := valf32) ?_ ?_
                                    · intro l
                                      exact hdivT (valf32.get l) sv
                                    · intro valq hvalq
                                      refine safe_cvt8 (f := .binary32) (g := .e4m3)
                                        (x := keyq) ?_ ?_ ?_
                                      · exact cvt8_ok_f32_e4m3
                                      · intro l
                                        exact hcvtT (keyq.get l)
                                      · intro key8 hkey8
                                        have himg : Buf.img (gi dd init (i : Nat))
                                            (en dd init (i : Nat)) ⊆ dstSet dd init :=
                                          img_subset_dstSet hd' hp
                                        have hdisj : Disjoint
                                            (dd.key_cache.elems (Buf.img (gi dd init (i : Nat))
                                              (en dd init (i : Nat))))
                                            (dd.value_cache.elems (Buf.img (gi dd init (i : Nat))
                                              (en dd init (i : Nat)))) :=
                                          kc_elems_disjoint_vc hd' hslall himg himg
                                        refine safe_store_buf (b := dd.key_cache)
                                          (g := gi dd init (i : Nat))
                                          (en := en dd init (i : Nat))
                                          (kc_addrs (d := dd) hd' hp hsl0 hsl hblk hoff hhead'
                                            hdim') ?_ ?_ ?_ ?_ ?_
                                        · intro l hen
                                          exact gi_lt_kc (d := dd) hd' hp hsl hen
                                        · intro l1 l2 h1 h2 hg
                                          exact gi_inj (d := dd) hd' hp hsl h1 h2 hg
                                        · exact hpre'.2.2.1
                                        · exact Set.subset_union_left
                                        · intro m1 hstore1 hframe1 hpres1
                                          refine safe_cvt8 (f := .binary32) (g := .e4m3)
                                            (x := valq) ?_ ?_ ?_
                                          · exact cvt8_ok_f32_e4m3
                                          · intro l
                                            exact hcvtT (valq.get l)
                                          · intro val8 hval8
                                            refine safe_store_buf (b := dd.value_cache)
                                              (g := gi dd init (i : Nat))
                                              (en := en dd init (i : Nat))
                                              (vc_addrs (d := dd) hd' hp hsl0 hsl hblk hoff hhead'
                                                hdim') ?_ ?_ ?_ ?_ ?_
                                            · intro l hen
                                              exact gi_lt_vc (d := dd) hd' hp hsl hen
                                            · intro l1 l2 h1 h2 hg
                                              exact gi_inj (d := dd) hd' hp hsl h1 h2 hg
                                            · refine Buf.valid_of_agree (b := dd.value_cache)
                                                hpre'.2.2.2.1 (fun x hx => ?_)
                                              exact hpres1 x (hpre'.2.2.2.1.2.2.2 x hx)
                                            · intro x hx
                                              refine ⟨Set.subset_union_right hx, ?_⟩
                                              intro hxk
                                              exact Set.disjoint_left.mp hdisj hxk hx
                                            · intro m2 hstore2 hframe2 hpres2
                                              refine safe_ret ?_
                                              intro l hen
                                              have hj : gi dd init (i : Nat) l ∈
                                                  Buf.img (gi dd init (i : Nat))
                                                    (en dd init (i : Nat)) := ⟨l, hen, rfl⟩
                                              have hdisj1 : Disjoint
                                                  (dd.key_cache.elems {gi dd init (i : Nat) l})
                                                  (dd.value_cache.elems
                                                    (Buf.img (gi dd init (i : Nat))
                                                      (en dd init (i : Nat)))) :=
                                                kc_elems_disjoint_vc hd' hslall
                                                  (fun j hj1 => by
                                                    rw [Set.mem_singleton_iff] at hj1
                                                    subst hj1
                                                    exact gi_mem_dstSet hd' hp hen) himg
                                              have hread : m2.read dd.mode.cd
                                                  (dd.key_cache.ptr
                                                    (gi dd init (i : Nat) l)).addr =
                                                  m1.read dd.mode.cd
                                                    (dd.key_cache.ptr
                                                      (gi dd init (i : Nat) l)).addr := by
                                                refine Mem.read_congr (fun x hx => ?_)
                                                have hxs : x ∈ dd.key_cache.elems
                                                    {gi dd init (i : Nat) l} :=
                                                  Buf.span_sub_elems dd.key_cache
                                                    (S := {gi dd init (i : Nat) l})
                                                    (by rw [Set.mem_singleton_iff]) hx
                                                exact hframe2 x (fun hv =>
                                                  Set.disjoint_left.mp hdisj1 hxs hv)
                                              refine ⟨⟨keyv.get l, key8.get l, ?_, ?_, ?_⟩,
                                                ⟨valv.get l, val8.get l, ?_, ?_, ?_⟩⟩
                                              · simpa only [giK] using hkeyv l hen
                                              · rw [hread]
                                                exact hstore1 l hen
                                              · rw [hskv]
                                                exact stored_fp8 (P := E.P) hdivF hcvtS
                                                  (hkeyf l) (hkeyq l) (hkey8 l)
                                              · simpa only [giV] using hvalv l hen
                                              · exact hstore2 l hen
                                              · rw [hsvv]
                                                exact stored_fp8 (P := E.P) hdivF hcvtS
                                                  (hvalf l) (hvalq l) (hval8 l)
  · cases hm

end Claims.KvCacheWrite
