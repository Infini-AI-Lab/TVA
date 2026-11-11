import Proof.Out

/-!
# One program of the grid

`safe_program`: program `(s, h)` is safe from `init`, reads inside `R d init s h`, writes inside
`W d s h`, and leaves `Q d init s h`.
-/
namespace Claims.PagedAttention

open TVA

/-- **One program** `(s, h)` of the launch: the setup (`kv_head`, the sequence length, the query
row), the loop over pages (`safe_trip` with the invariant `Inv`), the division, the rounding and
the store. -/
theorem safe_program {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) {s h : Nat} (hs : s < d.num_seqs) (hh : h < d.num_heads) (c : Ctx)
    (hc0 : c.pid 0 = s) (hc1 : c.pid 1 = h) :
    Safe ((launch d).env P) (R d init s h) (W d s h) (fun _ m => Q d init s h m) init
      ((model d).body c) := by
  obtain ⟨hvq, hvk, hvv, hvo, hvb, hvs, hsok⟩ := hpre
  have hsok_s := hsok s hs
  obtain ⟨n, hn_read, hn0, hn_le, hn22, hbt⟩ := hsok s hs
  have hseq : (tl.program_id c 0).toInt = s := by
    rw [tl.program_id, hc0]; exact IntV.I32.toInt_ofNat (by have := hd.seqs_lt; omega)
  have hhead : (tl.program_id c 1).toInt = h := by
    rw [tl.program_id, hc1]; exact IntV.I32.toInt_ofNat (by have := hd.heads_lt; omega)
  have hqpk : 0 < qpk d := by
    unfold qpk
    obtain ⟨k, hk⟩ := hd.kv_dvd
    have := hd.kv_pos
    have hH : 0 < d.num_heads := by omega
    rw [hk]
    rw [Nat.mul_div_cancel_left _ this]
    rcases Nat.eq_zero_or_pos k with h0 | h0
    · rw [h0] at hk; omega
    · exact h0
  have hqpk_val : (IntV.ofNat (qpk d) : I32).toInt = qpk d := IntV.I32.toInt_ofNat (by
    have : qpk d ≤ d.num_heads := Nat.div_le_self _ _
    have := hd.heads_lt; omega)
  have hRq : d.q.elems (qSet d s h) ⊆ R d init s h := fun x hx => Or.inl (Or.inl (Or.inl (Or.inl hx)))
  have hRsl : d.seq_lens.elems {s} ⊆ R d init s h := fun x hx => Or.inl (Or.inl (Or.inl (Or.inr hx)))
  have hRbt : d.block_tables.elems (btSet d init s) ⊆ R d init s h := fun x hx => Or.inl (Or.inl (Or.inr hx))
  have hRk : d.k_cache.elems (tokSet d init s h) ⊆ R d init s h := fun x hx => Or.inl (Or.inr hx)
  have hRv : d.v_cache.elems (tokSet d init s h) ⊆ R d init s h := fun x hx => Or.inr hx
  rw [body_eq]
  -- kv_head
  refine safe_idiv1 (x := tl.program_id c 1) (y := (IntV.ofNat (qpk d) : I32)) (by omega) ?_ fun kvt hkv => ?_
  · rw [hqpk_val]; exact_mod_cast hqpk
  · have hkvt : kvt.item.toInt = h / qpk d := by rw [hkv, hhead, hqpk_val]
    -- seq_lens[s]
    have hS31 : s < 2 ^ 31 := by have := hd.seqs_lt; omega
    have hptr_sl : d.seq_lens.ptr 0 + tl.program_id c 0 = d.seq_lens.ptr (s : Int) := by
      rw [Ptr.add_int, hseq, Buf.ptr_offset]; simp
    rw [hptr_sl]
    refine safe_load1_buf d.seq_lens (i := s) (by have := hd.sl_len; omega) hvs ?_ fun sl hsl => ?_
    · intro x hx
      exact hRsl ⟨s, rfl, hx⟩
    · have hsl_eq : sl = n := by
        rw [hsl] at hn_read; exact Option.some.inj hn_read
      subst hsl_eq
      dsimp only
      -- the query row
      have hqs : (tl.program_id c 0 * (IntV.ofNat d.stride_qs : I32)).toInt = (s * d.stride_qs : ℕ) :=
        IntV.I32.mul_nat hseq (IntV.I32.toInt_ofNat (by have := hd.strides_lt; omega)) (by
          have h1 := hd.prods_lt.1
          have h2 : s * d.stride_qs ≤ (d.num_seqs - 1) * d.stride_qs :=
            Nat.mul_le_mul_right _ (by omega)
          omega)
      have hqh : (tl.program_id c 1 * (IntV.ofNat d.stride_qh : I32)).toInt = (h * d.stride_qh : ℕ) :=
        IntV.I32.mul_nat hhead (IntV.I32.toInt_ofNat (by have := hd.strides_lt; omega)) (by
          have h1 := hd.prods_lt.2.1
          have h2 : h * d.stride_qh ≤ (d.num_heads - 1) * d.stride_qh :=
            Nat.mul_le_mul_right _ (by omega)
          omega)
      have hD31 : d.HEAD_DIM ≤ 2 ^ 31 := by
        obtain ⟨a, ha, he⟩ := hd.head_dim
        rw [he]; calc 2 ^ a ≤ 2 ^ 8 := Nat.pow_le_pow_right (by norm_num) ha
          _ ≤ 2 ^ 31 := by norm_num
      refine safe_load_buf d.q (g := fun i : Idx [d.HEAD_DIM - 0] => qIdx d s h i.1.val)
        (en := fun _ => true) (Buf.addrs_of_get_all d.q fun i => ?_) ?_ hvq ?_ fun qraw hqraw _ => ?_
      · rcases i with ⟨i, ⟨⟩⟩
        have hlane : ((tl.arange 0 d.HEAD_DIM).get (i, ())).toInt = (i : Nat) :=
          IntV.I32.toInt_tl_arange hD31 i
        simp only [Ptr.add_int, Buf.ptr_offset, hqs, hqh]
        refine (Tensor.get_ptrAdd _ _ _).trans ?_
        rw [hlane, Buf.ptr_offset]
        congr 1
        unfold qIdx
        push_cast
        ring
      · intro i _
        exact hd.q_len s h i.1.val hs hh i.1.isLt
      · refine Set.Subset.trans (d.q.elems_mono ?_) hRq
        rintro _ ⟨i, -, rfl⟩
        exact ⟨i.1.val, i.1.isLt, rfl⟩
      · refine safe_cvt ⟨Format.binary16_std, Format.binary32_std, Or.inl rfl⟩ fun q hq => ?_
        have hQF : SeqVals d init s h → ∀ (e : Nat) (he : e < d.HEAD_DIM),
            (q.get1 ⟨e, he⟩).isFinite = true ∧
              (q.get1 ⟨e, he⟩).toReal = (val16 init d.q (qIdx d s h e) : ℝ) := by
          intro hSV e he
          obtain ⟨x, hx, hxf⟩ := hSV.2.1 e he
          have h1 := hqraw (⟨e, he⟩, ()) rfl
          have hxe : qraw.get (⟨e, he⟩, ()) = x := by
            rw [hx] at h1; exact (Option.some.inj h1).symm
          obtain ⟨hf, hv⟩ := Num.cvt16_exact (x := qraw.get (⟨e, he⟩, ())) (by rw [hxe]; exact hxf)
            RoundMode.rn (hq (⟨e, he⟩, ()))
          refine ⟨hf, ?_⟩
          show FloatV.toReal (q.get (⟨e, he⟩, ())) = _
          rw [hv, hxe, val16_eq hx]; rfl
        have hP0 : 0 < d.PAGE_SIZE := by obtain ⟨b, hb, he⟩ := hd.page_size; rw [he]; positivity
        have hP8 : d.PAGE_SIZE ≤ 2 ^ 8 := by
          obtain ⟨b', hb', he⟩ := hd.page_size
          rw [he]; exact Nat.pow_le_pow_right (by norm_num) hb'
        have hnl : seqLen d init s = sl.toInt.toNat := by unfold seqLen val32; rw [hn_read]
        have hn_nat : (sl.toInt.toNat : Int) = sl.toInt := Int.toNat_of_nonneg hn0
        have hxn : (sl + (IntV.ofNat (d.PAGE_SIZE - 1) : I32)).toInt =
            ((sl.toInt.toNat + (d.PAGE_SIZE - 1) : Nat) : Int) :=
          IntV.I32.add_nat (x := sl.toInt.toNat) (y := d.PAGE_SIZE - 1) hn_nat.symm
            (IntV.I32.toInt_ofNat (by omega)) (by omega)
        have hyn : (IntV.ofNat d.PAGE_SIZE : I32).toInt = d.PAGE_SIZE :=
          IntV.I32.toInt_ofNat (by omega)
        refine safe_idiv1 (x := sl + (IntV.ofNat (d.PAGE_SIZE - 1) : I32))
          (y := (IntV.ofNat d.PAGE_SIZE : I32)) (by rw [hxn]; positivity) (by rw [hyn]; exact_mod_cast hP0)
          fun nb hnb => ?_
        have hnb_val : nb.item.toInt = (nblocks d init s : Int) := by
          rw [hnb, hxn, hyn]
          unfold nblocks
          rw [hnl]
          have : (sl.toInt.toNat + (d.PAGE_SIZE - 1) : Nat) = sl.toInt.toNat + d.PAGE_SIZE - 1 := by omega
          rw [this]
          exact (Int.natCast_div _ _).symm
        obtain ⟨hT2, hnT, hn22'⟩ := nblocks_le hd hsok_s
        have hsl' : sl.toInt = seqLen d init s := by rw [hnl, hn_nat]
        have hRange : tl.RangeOk (IntV.ofInt 0 : I32) nb.item (IntV.ofInt 1 : I32) := by
          have h0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.toInt_ofInt_zero
          have h1 : (IntV.ofInt 1 : I32).toInt = 1 := IntV.toInt_ofInt_one (by norm_num)
          unfold tl.RangeOk
          rw [h0, h1, hnb_val]
          simp only [IntV.maxInt, ite_true]
          norm_num
          omega
        have hcount : tl.tripCount (IntV.ofInt 0 : I32).toInt nb.item.toInt (IntV.ofInt 1 : I32).toInt =
            nblocks d init s := by
          rw [IntV.toInt_ofInt_zero, IntV.toInt_ofInt_one (by norm_num), hnb_val]
          unfold tl.tripCount
          simp
        have hiter : iterW (fun _ => (∅ : Set Int)) (tl.tripCount (IntV.ofInt 0 : I32).toInt nb.item.toInt
            (IntV.ofInt 1 : I32).toInt) = ∅ := iterW_empty _
        refine (safe_forRange (σ := Tensor [d.HEAD_DIM] F32 × F32 × F32) hRange
          (fun t st m => Inv d init s h t st m) (fun _ => ∅) (Wk := W d s h)
          (fun t u _ _ => Set.disjoint_empty _) (by rw [hiter]; exact Set.empty_disjoint _)
          ⟨rfl, fun hSV X hX0 hX1 hXk => ⟨fun _ => rfl, fun h' => absurd h' (lt_irrefl 0)⟩⟩
          ?_ ?_).mono (fun _ _ h => h) ?_
        · intro t ht st m₁ hI
          have ht' : t < nblocks d init s := by rw [hcount] at ht; exact ht
          refine safe_trip hd hP hvk hvv hvb hs hh hsok_s ht' hseq hkvt hsl' hQF (j := IntV.ofInt (0 + t * 1)) ?_ hI
          have hT31 : t < 2 ^ 31 := by omega
          rw [IntV.I32.toInt_ofInt (x := ((0 : Int) + t * 1)) ⟨by omega, by omega⟩]; ring
        · -- the epilogue
          intro st m₁ hI
          obtain ⟨acc, l_i, m_i⟩ := st
          obtain ⟨hm1, hnum⟩ := hI
          have hm1' : init = m₁ := hm1.symm
          subst hm1'
          dsimp only
          refine safe_div (fun i => hP.divTotal _ _ _) fun qd hqd => ?_
          refine safe_cvt ⟨Format.binary32_std, Format.binary16_std, Or.inl rfl⟩ fun o16 ho16 => ?_
          refine safe_last ?_
          have hos : (tl.program_id c 0 * (IntV.ofNat d.stride_os : I32)).toInt = (s * d.stride_os : ℕ) :=
            IntV.I32.mul_nat hseq (IntV.I32.toInt_ofNat hd.strides_lt.2.2.2.2.2.2.1) (by
              have h1 := hd.prods_lt.2.2.1
              have h2 : s * d.stride_os ≤ (d.num_seqs - 1) * d.stride_os :=
                Nat.mul_le_mul_right _ (by omega)
              omega)
          have hoh : (tl.program_id c 1 * (IntV.ofNat d.stride_oh : I32)).toInt = (h * d.stride_oh : ℕ) :=
            IntV.I32.mul_nat hhead (IntV.I32.toInt_ofNat hd.strides_lt.2.2.2.2.2.2.2) (by
              have h1 := hd.prods_lt.2.2.2.1
              have h2 : h * d.stride_oh ≤ (d.num_heads - 1) * d.stride_oh :=
                Nat.mul_le_mul_right _ (by omega)
              omega)
          refine safe_store_buf d.out (g := fun i : Idx [d.HEAD_DIM - 0] => outIdx d s h i.1.val)
            (en := fun _ => true) (Buf.addrs_of_get_all d.out fun i => ?_) ?_ ?_ hvo ?_ fun m' hrd hfr hpres => ?_
          · rcases i with ⟨i, ⟨⟩⟩
            have hlane : ((tl.arange 0 d.HEAD_DIM).get (i, ())).toInt = (i : Nat) :=
              IntV.I32.toInt_tl_arange hD31 i
            simp only [Ptr.add_int, Buf.ptr_offset, hos, hoh]
            refine (Tensor.get_ptrAdd _ _ _).trans ?_
            rw [hlane, Buf.ptr_offset]
            congr 1
            unfold outIdx
            push_cast
            ring
          · intro i _
            exact hd.out_len s h i.1.val hs hh i.1.isLt
          · intro i j _ _ hij
            have := hd.out_inj s h i.1.val s h j.1.val hs hh i.1.isLt hs hh j.1.isLt hij
            rcases i with ⟨i, ⟨⟩⟩; rcases j with ⟨j, ⟨⟩⟩
            exact Prod.ext (Fin.ext this.2.2) rfl
          · refine d.out.elems_mono ?_
            rintro _ ⟨i, -, rfl⟩
            exact ⟨i.1.val, i.1.isLt, rfl⟩
          · refine safe_ret ?_
            intro e he
            refine ⟨o16.get1 ⟨e, he⟩, hrd (⟨e, he⟩, ()) rfl, fun hSV => ?_⟩
            have hnum' := hnum hSV
            rw [hcount] at hnum'
            have key : ∀ X : ℚ, X ≤ 2 ^ 10 → (∀ k, k < seqLen d init s → sigma d * scoreAbs d init s h k ≤ X) →
                (o16.get1 ⟨e, he⟩).isFinite = true ∧
                  |(o16.get1 ⟨e, he⟩).toReal - attnRef d init s h e| ≤
                    (tolA (nblocks d init s) X : ℝ) * (attnAbs d init s h e + |attnRef d init s h e|) +
                      1 / 2 ^ 11 * |attnRef d init s h e| + 1 / 2 ^ 24 := by
              intro X hXle hXk
              refine out_lane hd hP hsok_s hSV hXle hXk (acc := acc) (l_i := l_i) (m_i := m_i)
                (fun Y hY0 hY1 hYk => (hnum' Y hY0 hY1 hYk).2 (nblocks_pos hd hSV.1)) he
                (hqd (⟨e, he⟩, ())) (ho16 (⟨e, he⟩, ()))
            refine ⟨(key (2 ^ 10) le_rfl hSV.2.2.2).1, fun X hXle hXk => (key X hXle hXk).2⟩
        · intro x hx
          rcases hx with ⟨t, ht, hx⟩ | hx
          · exact absurd hx (Set.notMem_empty _)
          · exact hx
end Claims.PagedAttention
