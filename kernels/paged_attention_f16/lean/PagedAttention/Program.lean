import PagedAttention.Facts

/-!
# `paged_attention_f16`: one program of the launch

* `pS`, `pH`: program `i`'s sequence and query head;
* `Rd`, `Wr`: every program's read footprint (the inputs) and program `i`'s write footprint (its
  row of `out`);
* `OutO`, `Qp`: what `Post` asks of an element of `out`, and of a program's row;
* `safe_program`: a whole program (lines 42–72).
-/

namespace Kernels.PagedAttention

open TVA TVA.Math.Float

/-- Program `i`'s sequence (`tl.program_id(0)`). -/
def pS (d : Inst) (i : ℕ) : ℕ := i % d.num_seqs

/-- Program `i`'s query head (`tl.program_id(1)`). -/
def pH (d : Inst) (i : ℕ) : ℕ := i / d.num_seqs

/-- Every program's read footprint: the inputs' bytes. -/
def Rd (d : Inst) : Set Int :=
  d.q.bytes ∪ d.k_cache.bytes ∪ d.v_cache.bytes ∪ d.block_tables.bytes ∪ d.seq_lens.bytes

/-- The elements of `out` program `(s, h)` writes (line 72). -/
def outRow (d : Inst) (s h : ℕ) : Set ℕ := {x | ∃ e, e < d.HEAD_DIM ∧ x = outIdx d s h e}

/-- Program `i`'s write footprint. -/
def Wr (d : Inst) (i : ℕ) : Set Int := d.out.elems (outRow d (pS d i) (pH d i))

/-- What `Post` asks of element `(s, h, e)` of `out` holding `c`. -/
def OutO (d : Inst) (init : Mem) (s h e : ℕ) (c : F16) : Prop :=
  SeqVals d init s h → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
    (∀ t, t < seqLen d init s → sigma d * scoreAbs d init s h t ≤ X) →
    |c.toReal - attnRef d init s h e| ≤
      (tolO d.HEAD_DIM d.PAGE_SIZE (nblocks d init s) X : ℝ) * attnAbs d init s h e +
        (tolF d.HEAD_DIM d.PAGE_SIZE (nblocks d init s) X : ℝ)

/-- What program `i` leaves in memory `m`: its row of `out` as `Post` asks. -/
def Qp (d : Inst) (init : Mem) (i : ℕ) (m : Mem) : Prop :=
  ∀ e : Fin d.HEAD_DIM, ∃ c : F16,
    m.read .f16 (d.out.ptr (outIdx d (pS d i) (pH d i) e : ℤ)).addr = some c ∧
      OutO d init (pS d i) (pH d i) e c

theorem launch_size (d : Inst) : (launch d).size = d.num_seqs * d.num_heads := by
  simp [Launch.size, launch]

theorem pid0 (d : Inst) (i : ℕ) : ((launch d).ctx i).pid 0 = pS d i := by
  simp [Launch.ctx, launch, pS]

theorem pid1 (d : Inst) {i : ℕ} (hi : i < (launch d).size) : ((launch d).ctx i).pid 1 = pH d i := by
  rw [launch_size] at hi
  simp only [Launch.ctx, launch]
  show i / d.num_seqs % d.num_heads = pH d i
  exact Nat.mod_eq_of_lt (Nat.div_lt_of_lt_mul hi)

theorem pS_lt {d : Inst} (hd : d.ok) (i : ℕ) : pS d i < d.num_seqs := Nat.mod_lt _ hd.seqs_pos

theorem pH_lt {d : Inst} {i : ℕ} (hi : i < (launch d).size) : pH d i < d.num_heads := by
  rw [launch_size] at hi; exact Nat.div_lt_of_lt_mul hi

theorem arange_val {n : ℕ} (hn : n ≤ 2 ^ 30) (e : Fin n) :
    ((tl.arange 0 n).get1 e).toInt = ((e : ℕ) : ℤ) := by
  show (IntV.ofInt (((0 : ℕ) : ℤ) + ((e : ℕ) : ℤ)) : I32).toInt = _
  rw [IntV.I32.toInt_ofInt (by have := e.2; constructor <;> push_cast <;> omega)]; push_cast; ring

theorem rd_q {d : Inst} : d.q.bytes ⊆ Rd d := fun _ hx => Or.inl (Or.inl (Or.inl (Or.inl hx)))
theorem rd_k {d : Inst} : d.k_cache.bytes ⊆ Rd d := fun _ hx => Or.inl (Or.inl (Or.inl (Or.inr hx)))
theorem rd_v {d : Inst} : d.v_cache.bytes ⊆ Rd d := fun _ hx => Or.inl (Or.inl (Or.inr hx))
theorem rd_bt {d : Inst} : d.block_tables.bytes ⊆ Rd d := fun _ hx => Or.inl (Or.inr hx)
theorem rd_sl {d : Inst} : d.seq_lens.bytes ⊆ Rd d := fun _ hx => Or.inr hx

/-! ## The values a program reads -/

theorem val16_of {init : Mem} {b : Buf .f16} {i : ℕ} {x : F16}
    (h : init.read .f16 (b.ptr (i : ℤ)).addr = some x) : val16 init b i = x.toRat := by
  unfold val16; rw [h]

theorem fin_of {init : Mem} {b : Buf .f16} {i : ℕ} {x : F16}
    (h : init.read .f16 (b.ptr (i : ℤ)).addr = some x) (hf : fin16 init b i) : x.isFinite = true := by
  obtain ⟨y, hy, hyf⟩ := hf
  rw [h] at hy; cases hy; exact hyf

/-- Token `P·t + r`'s element: page `block_tables[s, t]`, row `r`. -/
theorem tokElem_eq {d : Inst} {init : Mem} {s h t r e g : ℕ} {b : I32} (hP0 : 0 < d.PAGE_SIZE)
    (hr : r < d.PAGE_SIZE) (hb : init.read .i32 (d.block_tables.ptr (btIdx d s t : ℤ)).addr = some b)
    (hg : h / qpk d = g) : tokElem d init s h (d.PAGE_SIZE * t + r) e = kvIdx d b.toInt.toNat r g e := by
  unfold tokElem pageOf val32
  have h1 : (d.PAGE_SIZE * t + r) / d.PAGE_SIZE = t := by
    rw [show d.PAGE_SIZE * t + r = r + d.PAGE_SIZE * t by ring, Nat.add_mul_div_left _ _ hP0,
      Nat.div_eq_of_lt hr, zero_add]
  have h2 : (d.PAGE_SIZE * t + r) % d.PAGE_SIZE = r := by
    rw [show d.PAGE_SIZE * t + r = r + d.PAGE_SIZE * t by ring, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hr]
  rw [h1, h2, hb, hg]

/-- **One lane's score** (line 61) for a token `tok` of the sequence: within `Es` of
`scale·score`. -/
theorem lane_score {d : Inst} (hd : d.ok) {init : Mem} {s h : ℕ} {X : ℚ}
    (hRok : (rowCtx d init s h X).Ok) {tok : ℕ} (hXk : sigma d * scoreAbs d init s h tok ≤ X)
    {fusion : Bool} {kf qf pr : Fin d.HEAD_DIM → F32} {srow sc : F32}
    (hk : ∀ e : Fin d.HEAD_DIM, (kf e).isFinite = true ∧
      (kf e).toRat = val16 init d.k_cache (tokElem d init s h tok e))
    (hq : ∀ e : Fin d.HEAD_DIM, (qf e).isFinite = true ∧ (qf e).toRat = val16 init d.q (qIdx d s h e))
    (hkb : ∀ e, |(kf e).toReal| ≤ 65504) (hqb : ∀ e, |(qf e).toReal| ≤ 65504)
    (hpr : ∀ e, FloatV.NaNEq (FloatV.mul (kf e) (qf e)) (pr e))
    (hsum : ∃ tr : Tree, tr.Covers d.HEAD_DIM ∧ tr.Eval (AddRel fusion) (laneSeq pr) srow)
    (hsc : FloatV.NaNEq (FloatV.mul srow (FloatV.hostF32 d.scale)) sc) :
    sc.isFinite = true ∧
      |sc.toReal - (rowCtx d init s h X).c * (rowCtx d init s h X).sv tok| ≤ (rowCtx d init s h X).Es := by
  have hsS : ∑ e, (kf e).toReal * (qf e).toReal = (score d init s h tok : ℝ) := by
    unfold score; push_cast
    rw [← Fin.sum_univ_eq_sum_range (fun e => (val16 init d.q (qIdx d s h e) : ℝ) *
      (val16 init d.k_cache (tokElem d init s h tok e) : ℝ))]
    refine Finset.sum_congr rfl fun e _ => ?_
    rw [toReal_def, toReal_def, (hk e).2, (hq e).2]; ring
  have hsA : ∑ e, |(kf e).toReal * (qf e).toReal| = (scoreAbs d init s h tok : ℝ) := by
    unfold scoreAbs; push_cast
    rw [← Fin.sum_univ_eq_sum_range (fun e => |(val16 init d.q (qIdx d s h e) : ℝ) *
      (val16 init d.k_cache (tokElem d init s h tok e) : ℝ)|)]
    refine Finset.sum_congr rfl fun e _ => ?_
    rw [toReal_def, toReal_def, (hk e).2, (hq e).2, mul_comm]
  obtain ⟨hc32f, hc32⟩ := scale_facts hd
  have hXr : (rowCtx d init s h X).c * (scoreAbs d init s h tok : ℝ) ≤ (rowCtx d init s h X).Y := by
    show (sigma d : ℝ) * _ ≤ (X : ℝ); exact_mod_cast hXk
  have hB : (rowCtx d init s h X).c * ∑ e, |(kf e).toReal * (qf e).toReal| ≤ 2 ^ 10 := by
    rw [hsA]; exact hXr.trans hRok.Y_le
  obtain ⟨hf, he⟩ := score_lane (headDim_le hd) hRok.c_lo hRok.c_hi (fun e => (hk e).1) (fun e => (hq e).1)
    hkb hqb hpr hsum hc32f hc32 hsc hB
  refine ⟨hf, ?_⟩
  rw [hsS, hsA] at he
  have hr0 := hRok.rhoD_nonneg
  have hcoef : (1 + 1 / 2 ^ 24) * (1 + 3 / 2 ^ 24) ^ (d.HEAD_DIM - 1) - 1 + 3 / 2 ^ 24 =
      (rowCtx d init s h X).rhoD + 3 / 2 ^ 24 := by
    show _ = (rhoD d.HEAD_DIM : ℝ) + 3 / 2 ^ 24
    rw [rhoD_cast]
  rw [hcoef] at he
  have := mul_le_mul_of_nonneg_left hXr (by linarith : (0 : ℝ) ≤ (rowCtx d init s h X).rhoD + 3 / 2 ^ 24)
  show _ ≤ ((rowCtx d init s h X).rhoD + 3 / 2 ^ 24) * (rowCtx d init s h X).Y + 1 / 2 ^ 70
  exact he.trans (by linarith)

/-! ## The end of a row (lines 71–72) -/

theorem nblocks_cover {d : Inst} (hd : d.ok) (init : Mem) (s : ℕ) :
    seqLen d init s ≤ d.PAGE_SIZE * nblocks d init s := by
  have hP0 := pageSize_pos hd
  unfold nblocks
  have h1 := Nat.div_add_mod (seqLen d init s + d.PAGE_SIZE - 1) d.PAGE_SIZE
  have h2 := Nat.mod_lt (seqLen d init s + d.PAGE_SIZE - 1) hP0
  omega

/-- **The end of a row** (lines 71–72): from the row's state after the loop, `acc / l_i` rounded to
`float16` is as `Post` asks. -/
theorem row_out {d : Inst} (hd : d.ok) {init : Mem} {P : Platform} (hdivF : Rules.Num.DivFull P)
    {s h : ℕ} (hT : nblocks d init s ≤ 2 ^ 16)
    {acc : Tensor [d.HEAD_DIM] F32} {l m : F32} (hst : StInv d init s h (nblocks d init s) acc l m)
    {o' : Fin d.HEAD_DIM → F32} (ho' : ∀ e, P.div .binary32 (acc.get1 e) l (o' e))
    {o16 : Fin d.HEAD_DIM → F16} (ho16 : ∀ e, FloatV.NaNEq (FloatV.cvt (o' e)) (o16 e))
    (e : Fin d.HEAD_DIM) : OutO d init s h e (o16 e) := by
  intro hH
  have hn1 := hH.1
  have hnb := nblocks_cover hd init s
  have key : ∀ X : ℚ, X ≤ 2 ^ 10 → (∀ t, t < seqLen d init s → sigma d * scoreAbs d init s h t ≤ X) →
      (o16 e).isFinite = true ∧ |(o16 e).toReal - attnRef d init s h e| ≤
        (tolO d.HEAD_DIM d.PAGE_SIZE (nblocks d init s) X : ℝ) * attnAbs d init s h e +
          (tolF d.HEAD_DIM d.PAGE_SIZE (nblocks d init s) X : ℝ) := by
    intro X hX hXk
    obtain ⟨hRok, -⟩ := rowCtx_ok hd init hn1 hX hXk
    rcases hst hH X hX hXk with ⟨h0, -⟩ | ⟨-, hM, hL, hA⟩
    · rw [h0] at hnb; omega
    · obtain ⟨hf, hb⟩ := final_o (D := d.HEAD_DIM) (X := X) hdivF hRok rfl rfl hT hnb
        (veOf_le hH e e.2) hM hL (hA e) (ho' e) (ho16 e)
      refine ⟨hf, ?_⟩
      have hve : ∀ j ∈ Finset.range (seqLen d init s),
          veOf d init s h e j = (val16 init d.v_cache (tokElem d init s h j e) : ℝ) := fun j hj => by
        unfold veOf; rw [ite_eq_left (Finset.mem_range.mp hj)]
      have S1 : (∑ j ∈ Finset.range (seqLen d init s), weight d init s h j * veOf d init s h e j) =
          ∑ j ∈ Finset.range (seqLen d init s),
            weight d init s h j * (val16 init d.v_cache (tokElem d init s h j e) : ℝ) :=
        Finset.sum_congr rfl fun j hj => by rw [hve j hj]
      have S2 : (∑ j ∈ Finset.range (seqLen d init s), weight d init s h j * |veOf d init s h e j|) =
          ∑ j ∈ Finset.range (seqLen d init s),
            weight d init s h j * |(val16 init d.v_cache (tokElem d init s h j e) : ℝ)| :=
        Finset.sum_congr rfl fun j hj => by rw [hve j hj]
      unfold attnRef attnAbs
      rw [← S1, ← S2]
      exact hb
  exact ⟨(key _ le_rfl hH.2.2.2).1, fun X hX hXk => (key X hX hXk).2⟩

/-! ## Addresses -/

/-- Line 59: the `int64` offset of lane `(r, e)` of page `bN`, key/value head `g`. -/
theorem kv_ptr {d : Inst} (b : Buf .f16) (hlen : b.len ≤ 2 ^ 63) {pg sk c2 : I64}
    {c1 c3 : Tensor [d.PAGE_SIZE, d.HEAD_DIM] I64} {bN g : ℕ}
    (hpg : pg.toInt = bN) (hsk : sk.toInt = d.stride_kp)
    (hc1 : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM), (c1.get (r, e, ())).toInt = ((r * d.stride_kt : ℕ) : ℤ))
    (hc2 : c2.toInt = ((g * d.stride_kh : ℕ) : ℤ))
    (hc3 : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM), (c3.get (r, e, ())).toInt = ((e : ℕ) : ℤ))
    (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM) (hlt : kvIdx d bN r g e < b.len) :
    (b.ptr 0 + (pg * sk + c1 + c2 + c3)).get (r, e, ()) = b.ptr (kvIdx d bN r g e : ℤ) := by
  unfold kvIdx at hlt
  have hv1 : (pg * sk).toInt = ((bN * d.stride_kp : ℕ) : ℤ) := by
    rw [i64_mul hpg hsk (by positivity) (by
      have : bN * d.stride_kp < 2 ^ 63 := by omega
      exact_mod_cast this)]; push_cast; ring
  have hv2 : (pg * sk + c1.get (r, e, ())).toInt = ((bN * d.stride_kp + r * d.stride_kt : ℕ) : ℤ) := by
    rw [i64_add hv1 (hc1 r e) (by positivity) (by
      have : bN * d.stride_kp + r * d.stride_kt < 2 ^ 63 := by omega
      exact_mod_cast this)]; push_cast; ring
  have hv3 : (pg * sk + c1.get (r, e, ()) + c2).toInt =
      ((bN * d.stride_kp + r * d.stride_kt + g * d.stride_kh : ℕ) : ℤ) := by
    rw [i64_add hv2 hc2 (by positivity) (by
      have : bN * d.stride_kp + r * d.stride_kt + g * d.stride_kh < 2 ^ 63 := by omega
      exact_mod_cast this)]; push_cast; ring
  have hv4 : (pg * sk + c1.get (r, e, ()) + c2 + c3.get (r, e, ())).toInt =
      ((bN * d.stride_kp + r * d.stride_kt + g * d.stride_kh + e : ℕ) : ℤ) := by
    rw [i64_add hv3 (hc3 r e) (by positivity) (by
      have : bN * d.stride_kp + r * d.stride_kt + g * d.stride_kh + e < 2 ^ 63 := by omega
      exact_mod_cast this)]; push_cast; ring
  show (b.ptr 0).offset (pg * sk + c1.get (r, e, ()) + c2 + c3.get (r, e, ())).toInt = _
  rw [hv4, Buf.ptr_offset]
  congr 1; unfold kvIdx; push_cast; ring

/-- Line 58: lane `r` of the mask is `P·t + r < seq_len`. -/
theorem mask_eq {d : Inst} {j n : I32} {t N : ℕ} (hj : j.toInt = t) (hn : n.toInt = N)
    (hP : d.PAGE_SIZE ≤ 256) (htP : t * d.PAGE_SIZE + d.PAGE_SIZE < 2 ^ 31) (r : Fin d.PAGE_SIZE) :
    ((j * IntV.ofNat d.PAGE_SIZE + tl.arange 0 d.PAGE_SIZE).ltS n).get1 r =
      decide (d.PAGE_SIZE * t + r < N) := by
  show IntV.lt (j * IntV.ofNat d.PAGE_SIZE + (tl.arange 0 d.PAGE_SIZE).get1 r) n = _
  have hr := r.2
  rw [i32_lt (IntV.I32.add_nat (IntV.I32.mul_nat hj (i32_nat (by omega)) (by omega)) (arange_val (by omega) r)
    (by omega)) hn, show d.PAGE_SIZE * t = t * d.PAGE_SIZE from mul_comm _ _]
  exact decide_eq_decide.mpr Nat.cast_lt

theorem len_le {b : Buf .f16} {m : Mem} (hv : b.Valid m) : b.len ≤ 2 ^ 63 := by
  obtain ⟨h0, h1, -⟩ := hv
  have : (b.len : ℤ) * 2 ≤ 2 ^ 64 := by
    have e : DType.bytes .f16 = 2 := rfl
    rw [e] at h1; push_cast at h1; linarith
  have : (b.len : ℤ) ≤ 2 ^ 63 := by linarith
  exact_mod_cast this

theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hexp : Rules.Num.ExpF32 P) (hni : Rules.Num.ExpF32NegInf P)
    (hext : Rules.Num.MathTotal P .exp .binary32) (hdivF : Rules.Num.DivFull P)
    (hdivT : Rules.Num.DivTotal P .binary32) {i : ℕ} (hi : i < (launch d).size) :
    Safe ((launch d).env P) (Rd d) (Wr d i) (fun _ m => Qp d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  simp only [model, paged_attention_kernel, tl.program_id, pid0, pid1 d hi]
  have hs := pS_lt hd i
  have hh := pH_lt hi
  generalize hsd : pS d i = s at hs ⊢
  generalize hhd : pH d i = h at hh ⊢
  have hSH := hd.grid_lt
  have hS1 := hd.seqs_pos
  have hH1 := hd.heads_pos
  have hH16 := hd.heads_lt
  have hS31 : d.num_seqs < 2 ^ 31 := lt_of_le_of_lt (Nat.le_mul_of_pos_right _ hH1) hSH
  have hD := headDim_le hd
  have hP := pageSize_le hd
  have hP0 := pageSize_pos hd
  have hD0 := headDim_pos hd
  obtain ⟨hsqs, hsqh, hskp, hskt, hskh, hsbs, hsos, hsoh⟩ := hd.strides_lt
  obtain ⟨hpqs, hpqh, hpos, hpoh, hpbs, hpkt, hpkh⟩ := hd.prods_lt
  have hsv : (IntV.ofNat s : I32).toInt = s := i32_nat (by omega)
  have hhv : (IntV.ofNat h : I32).toInt = h := i32_nat (by omega)
  have hle_s : ∀ x : ℕ, s * x ≤ (d.num_seqs - 1) * x := fun x => Nat.mul_le_mul_right _ (by omega)
  have hle_h : ∀ x : ℕ, h * x ≤ (d.num_heads - 1) * x := fun x => Nat.mul_le_mul_right _ (by omega)
  -- line 44: `head // QUERIES_PER_KV`
  obtain ⟨cq, hcq⟩ := hd.kv_dvd
  have hKV := hd.kv_pos
  have hqpk : qpk d = cq := by unfold qpk; rw [hcq]; exact Nat.mul_div_cancel_left _ hKV
  have hcq0 : 0 < cq := by
    rcases Nat.eq_zero_or_pos cq with h0 | h0
    · rw [h0, mul_zero] at hcq; omega
    · exact h0
  have hqv : (IntV.ofNat (qpk d) : I32).toInt = qpk d := i32_nat (by
    rw [hqpk]; have : cq ≤ d.num_heads := by rw [hcq]; exact Nat.le_mul_of_pos_left _ hKV
    omega)
  refine safe_idiv1 (by rw [hhv]; positivity) (by rw [hqv, hqpk]; exact_mod_cast hcq0) fun kvh hkvh => ?_
  have hg : kvh.item.toInt = ((h / qpk d : ℕ) : ℤ) := by rw [hkvh, hhv, hqv]; push_cast; rfl
  have hglt : h / qpk d < d.num_kv_heads := by
    rw [hqpk]; apply Nat.div_lt_of_lt_mul; rw [mul_comm, ← hcq]; exact hh
  generalize hgd : h / qpk d = g at hg hglt
  -- line 45: `seq_lens[seq]`
  obtain ⟨n, hn_read, hn0, hn1, hbt⟩ := hpre.2.2.2.2.2.2 s hs
  have hslp : d.seq_lens.ptr 0 + (IntV.ofNat s : I32) = d.seq_lens.ptr (s : ℤ) := by
    rw [Ptr.add_int, Buf.ptr_offset, hsv]; simp
  rw [hslp]
  refine safe_load1 hn_read ((d.seq_lens.span_sub (by have := hd.sl_len; omega)).trans rd_sl) ?_
  have hseq : seqLen d init s = n.toInt.toNat := by unfold seqLen val32; rw [hn_read]
  have hmp := hd.max_pages_le
  have hn24 : n.toInt ≤ 2 ^ 24 := by
    have : ((d.max_pages * d.PAGE_SIZE : ℕ) : ℤ) ≤ 2 ^ 24 := by
      have := Nat.mul_le_mul hmp hP; push_cast; exact_mod_cast this.trans (by norm_num)
    omega
  -- line 49: `q`
  refine safe_load_buf d.q (g := fun i : Idx [d.HEAD_DIM] => qIdx d s h i.1) (en := fun _ : Idx [d.HEAD_DIM] => true)
    (Buf.addrs_of_get_all d.q fun ⟨e, ⟨⟩⟩ => ?_) (fun ⟨e, ⟨⟩⟩ _ => hd.q_len s h e hs hh e.2) hpre.1
    ((d.q.elems_sub_bytes ?_).trans rd_q) fun q16 hq16 _ => ?_
  · show (((d.q.ptr 0).offset ((IntV.ofNat s : I32) * IntV.ofNat d.stride_qs).toInt).offset
      ((IntV.ofNat h : I32) * IntV.ofNat d.stride_qh).toInt).offset ((tl.arange 0 d.HEAD_DIM).get1 e).toInt = _
    rw [IntV.I32.mul_nat hsv (i32_nat hsqs) (lt_of_le_of_lt (hle_s _) hpqs),
      IntV.I32.mul_nat hhv (i32_nat hsqh) (lt_of_le_of_lt (hle_h _) hpqh), arange_val (by omega) e,
      Buf.ptr_offset, Buf.ptr_offset, Buf.ptr_offset]
    congr 1; unfold qIdx; push_cast; ring
  · rintro _ ⟨⟨e, ⟨⟩⟩, -, rfl⟩; exact hd.q_len s h e hs hh e.2
  refine safe_cvt f16_f32_ok fun q32 hq32 => ?_
  -- line 54: the loop bound `cdiv(seq_len, PAGE_SIZE)`
  have hnP : (n + IntV.ofNat (d.PAGE_SIZE - 1) : I32).toInt = n.toInt + ((d.PAGE_SIZE - 1 : ℕ) : ℤ) :=
    i32_add rfl (i32_nat (by omega)) (by omega) (by omega)
  refine safe_idiv1 (by rw [hnP]; omega) (by rw [i32_nat (by omega)]; exact_mod_cast hP0) fun rT hrT => ?_
  have e1 : ((n.toInt.toNat + d.PAGE_SIZE - 1 : ℕ) : ℤ) = n.toInt + ((d.PAGE_SIZE - 1 : ℕ) : ℤ) := by omega
  have hTv : rT.item.toInt = (nblocks d init s : ℤ) := by
    rw [hrT, hnP, i32_nat (by omega)]; unfold nblocks; rw [hseq, Int.natCast_div, e1]
  have hTmp : nblocks d init s ≤ d.max_pages := by
    unfold nblocks; rw [hseq]
    have : n.toInt.toNat ≤ d.max_pages * d.PAGE_SIZE := by omega
    have key := (Nat.div_lt_iff_lt_mul hP0 (x := n.toInt.toNat + d.PAGE_SIZE - 1) (y := d.max_pages + 1)).mpr
      (by rw [Nat.add_mul, one_mul]; omega)
    omega
  have hTlt : ∀ t, t < nblocks d init s → d.PAGE_SIZE * t < n.toInt.toNat := by
    intro t ht
    unfold nblocks at ht; rw [hseq] at ht
    have key : ¬ (n.toInt.toNat + d.PAGE_SIZE - 1 < (t + 1) * d.PAGE_SIZE) := fun hlt =>
      absurd ((Nat.div_lt_iff_lt_mul hP0).mpr hlt) (by omega)
    rw [Nat.add_mul, one_mul] at key
    have e2 : d.PAGE_SIZE * t = t * d.PAGE_SIZE := by ring
    omega
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) rT.item (IntV.ofInt 1) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [i32_zero, i32_one, hTv]; norm_num; omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt rT.item.toInt (IntV.ofInt 1 : I32).toInt =
      nblocks d init s := by
    rw [i32_zero, i32_one, hTv]; simp [tl.tripCount]
  refine Safe.mono (safe_forRange (Wk := Wr d i) hrange
    (fun t st m₁ => m₁ = init ∧ StInv d init s h t st.1 st.2.1 st.2.2)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, ?_⟩ ?_ ?_) (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])
  · intro hH X hX hXk
    left
    refine ⟨rfl, ninf_lit, ?_, ?_, fun e => ⟨zero32.1, ?_⟩⟩
    · rw [f32_zero_lit]; exact zero32.1
    · rw [f32_zero_lit, toReal_def, zero32.2]; simp
    · rw [toReal_def]; exact (congrArg (fun x : ℚ => (x : ℝ)) zero32.2).trans Rat.cast_zero
  · rw [htc]
    rintro t ht ⟨acc, l, m⟩ m₁ ⟨hm₁, hinv⟩
    subst m₁
    simp only at hinv ⊢
    have htmp : t < d.max_pages := lt_of_lt_of_le ht hTmp
    have hjv : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + ↑t * (IntV.ofInt 1 : I32).toInt) : I32).toInt = t := by
      rw [i32_zero, i32_one, zero_add, mul_one]; exact IntV.I32.toInt_ofInt (by constructor <;> omega)
    generalize (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + ↑t * (IntV.ofInt 1 : I32).toInt) : I32) = j at hjv ⊢
    have hPt := hTlt t ht
    have hnN : n.toInt = ((seqLen d init s : ℕ) : ℤ) := by rw [hseq]; omega
    have hPt' : d.PAGE_SIZE * t < seqLen d init s := by rw [hseq]; exact hPt
    have htP : t * d.PAGE_SIZE + d.PAGE_SIZE < 2 ^ 31 := by
      have h1 : (t + 1) * d.PAGE_SIZE ≤ d.max_pages * d.PAGE_SIZE := Nat.mul_le_mul_right _ htmp
      rw [Nat.add_mul, one_mul] at h1
      have h2 : d.max_pages * d.PAGE_SIZE ≤ 2 ^ 16 * 256 := Nat.mul_le_mul hmp hP
      omega
    -- line 57: `block_tables[seq, j]`, widened to `int64`
    obtain ⟨b, hb_read, hb0, hb1⟩ := hbt t (by
      have h1 : ((t * d.PAGE_SIZE : ℕ) : ℤ) < (n.toInt.toNat : ℤ) := by
        exact_mod_cast (show t * d.PAGE_SIZE < n.toInt.toNat by rw [mul_comm]; exact hPt)
      rwa [Int.toNat_of_nonneg hn0] at h1)
    have hbtp : d.block_tables.ptr 0 + (IntV.ofNat s : I32) * IntV.ofNat d.stride_bs + j =
        d.block_tables.ptr (btIdx d s t : ℤ) := by
      rw [Ptr.add_int, Ptr.add_int, IntV.I32.mul_nat hsv (i32_nat hsbs) (lt_of_le_of_lt (hle_s _) hpbs), hjv,
        Buf.ptr_offset, Buf.ptr_offset]
      congr 1; unfold btIdx; push_cast; ring
    rw [hbtp]
    refine safe_load1 hb_read ((d.block_tables.span_sub (hd.bt_len s t hs htmp)).trans rd_bt) ?_
    refine safe_convert (fun _ => conv64_ex P _) fun pg hpg => ?_
    have hbN : b.toInt.toNat < d.num_pages := by omega
    have hpgv : pg.item.toInt = (b.toInt.toNat : ℤ) := (conv64 (hpg ())).trans (Int.toNat_of_nonneg hb0).symm
    -- line 59: the offsets
    refine safe_convert (fun _ => conv64_ex P _) fun sk hsk => ?_
    have hskv : sk.item.toInt = d.stride_kp := (conv64 (hsk ())).trans (i32_nat hskp)
    refine safe_convert (fun _ => conv64_ex P _) fun c1 hc1 => ?_
    have hc1v : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM),
        (c1.get (r, e, ())).toInt = ((r * d.stride_kt : ℕ) : ℤ) := fun r e =>
      (conv64 (hc1 (r, e, ()))).trans (IntV.I32.mul_nat (arange_val (by omega) r) (i32_nat hskt)
        (lt_of_le_of_lt (Nat.mul_le_mul_right _ (by have := r.2; omega)) hpkt))
    refine safe_convert (fun _ => conv64_ex P _) fun c2 hc2 => ?_
    have hc2v : c2.item.toInt = ((g * d.stride_kh : ℕ) : ℤ) :=
      (conv64 (hc2 ())).trans (IntV.I32.mul_nat hg (i32_nat hskh)
        (lt_of_le_of_lt (Nat.mul_le_mul_right _ (by omega)) hpkh))
    refine safe_convert (fun _ => conv64_ex P _) fun c3 hc3 => ?_
    have hc3v : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM), (c3.get (r, e, ())).toInt = ((e : ℕ) : ℤ) :=
      fun r e => (conv64 (hc3 (r, e, ()))).trans (arange_val (by omega) e)
    have hkvlt : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM),
        kvIdx d b.toInt.toNat r g e < d.k_cache.len ∧ kvIdx d b.toInt.toNat r g e < d.v_cache.len :=
      fun r e => hd.kv_len _ r g e hbN r.2 hglt e.2
    have hmask := mask_eq (d := d) hjv hnN hP htP
    -- line 60: `k`
    refine safe_convert (fun _ => zero_conv_ex P) fun zk hzk => ?_
    refine safe_load_buf d.k_cache (g := fun i : Idx [d.PAGE_SIZE, d.HEAD_DIM] => kvIdx d b.toInt.toNat i.1 g i.2.1)
      (en := fun i : Idx [d.PAGE_SIZE, d.HEAD_DIM] => decide (d.PAGE_SIZE * t + i.1 < seqLen d init s))
      (Buf.addrs_of_get d.k_cache (fun ⟨r, e, ⟨⟩⟩ => hmask r)
        (fun ⟨r, e, ⟨⟩⟩ _ => kv_ptr d.k_cache (len_le hpre.2.1) hpgv hskv hc1v hc2v hc3v r e (hkvlt r e).1))
      (fun ⟨r, e, ⟨⟩⟩ _ => (hkvlt r e).1) hpre.2.1 ((d.k_cache.elems_sub_bytes ?_).trans rd_k)
      fun k16 hk1 hk2 => ?_
    · rintro _ ⟨⟨r, e, ⟨⟩⟩, -, rfl⟩; exact (hkvlt r e).1
    refine safe_cvt f16_f32_ok fun k32 hk32 => ?_
    -- line 61
    refine safe_mul fun pr hpr => ?_
    refine safe_sum1c hD0 fun srow hsrow => ?_
    refine safe_mul fun sc hsc => ?_
    -- line 62
    have hwget : ∀ r : Fin d.PAGE_SIZE,
        (tl.where ((j * IntV.ofNat d.PAGE_SIZE + tl.arange 0 d.PAGE_SIZE).ltS n) sc
          (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000 : F64)))).get1 r =
        if d.PAGE_SIZE * t + r < seqLen d init s then sc.get1 r else FloatV.negInf := fun r => by
      show (if ((j * IntV.ofNat d.PAGE_SIZE + tl.arange 0 d.PAGE_SIZE).ltS n).get1 r = true then _ else _) = _
      rw [hmask r]
      by_cases hr : d.PAGE_SIZE * t + r < seqLen d init s
      · rw [decide_eq_true hr, if_pos rfl, if_pos hr]; rfl
      · rw [decide_eq_false hr, if_neg (by simp), if_neg hr]; exact ninf_lit
    generalize (tl.where ((j * IntV.ofNat d.PAGE_SIZE + tl.arange 0 d.PAGE_SIZE).ltS n) sc
      (bc (FloatV.litF32 (FloatV.ofPat 0xFFF0000000000000 : F64)))) = sw at hwget ⊢
    -- line 63
    refine safe_maxAll hP0 true fun mx hmx => ?_
    refine safe_maximum fun m2 hm2 => ?_
    -- line 64
    refine safe_sub fun dm hdm => ?_
    refine safe_math (fun _ => hext _) fun al hal => ?_
    -- line 65
    refine safe_sub fun x hx => ?_
    refine safe_math (fun _ => hext _) fun p hp => ?_
    -- line 66
    refine safe_sum hP0 fun lsum hlsum => ?_
    refine safe_mul_add fun l2 hl2 => ?_
    -- line 67: `v`
    refine safe_convert (fun _ => zero_conv_ex P) fun zv hzv => ?_
    refine safe_load_buf d.v_cache (g := fun i : Idx [d.PAGE_SIZE, d.HEAD_DIM] => kvIdx d b.toInt.toNat i.1 g i.2.1)
      (en := fun i : Idx [d.PAGE_SIZE, d.HEAD_DIM] => decide (d.PAGE_SIZE * t + i.1 < seqLen d init s))
      (Buf.addrs_of_get d.v_cache (fun ⟨r, e, ⟨⟩⟩ => hmask r)
        (fun ⟨r, e, ⟨⟩⟩ _ => kv_ptr d.v_cache (len_le hpre.2.2.1) hpgv hskv hc1v hc2v hc3v r e (hkvlt r e).2))
      (fun ⟨r, e, ⟨⟩⟩ _ => (hkvlt r e).2) hpre.2.2.1 ((d.v_cache.elems_sub_bytes ?_).trans rd_v)
      fun v16 hv1 hv2 => ?_
    · rintro _ ⟨⟨r, e, ⟨⟩⟩, -, rfl⟩; exact (hkvlt r e).2
    refine safe_cvt f16_f32_ok fun v32 hv32 => ?_
    -- line 68
    refine safe_mul fun pv hpv => ?_
    refine safe_sum0c hP0 fun vsum hvsum => ?_
    refine safe_mul_add fun acc2 hacc2 => ?_
    refine safe_ret ⟨rfl, ?_⟩
    -- the row's state after the page
    intro hH X hX hXk
    obtain ⟨hRok, -⟩ := rowCtx_ok hd init hH.1 hX hXk
    have hkr : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM), d.PAGE_SIZE * t + r < seqLen d init s →
        init.read .f16 (d.k_cache.ptr (tokElem d init s h (d.PAGE_SIZE * t + r) e : ℤ)).addr = some (k16.get (r, e, ())) := by
      intro r e hr; rw [tokElem_eq hP0 r.2 hb_read hgd]; exact hk1 (r, e, ()) (decide_eq_true hr)
    have hvr : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM), d.PAGE_SIZE * t + r < seqLen d init s →
        init.read .f16 (d.v_cache.ptr (tokElem d init s h (d.PAGE_SIZE * t + r) e : ℤ)).addr = some (v16.get (r, e, ())) := by
      intro r e hr; rw [tokElem_eq hP0 r.2 hb_read hgd]; exact hv1 (r, e, ()) (decide_eq_true hr)
    have hk' : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM), d.PAGE_SIZE * t + r < seqLen d init s →
        (k32.get2 r e).isFinite = true ∧
          (k32.get2 r e).toRat = val16 init d.k_cache (tokElem d init s h (d.PAGE_SIZE * t + r) e) ∧
          |(k32.get2 r e).toReal| ≤ 65504 := by
      intro r e hr
      have hf := fin_of (hkr r e hr) (hH.2.2.1 _ e hr e.2).1
      obtain ⟨h1, h2⟩ : (k32.get2 r e).isFinite = true ∧ (k32.get2 r e).toRat = (k16.get (r, e, ())).toRat :=
        widen hf (hk32 (r, e, ()))
      refine ⟨h1, by rw [h2, val16_of (hkr r e hr)], ?_⟩
      rw [toReal_def, h2, ← toReal_def]; exact f16_abs_le hf
    have hq' : ∀ e : Fin d.HEAD_DIM, (q32.get1 e).isFinite = true ∧
        (q32.get1 e).toRat = val16 init d.q (qIdx d s h e) ∧ |(q32.get1 e).toReal| ≤ 65504 := by
      intro e
      have hrd : init.read .f16 (d.q.ptr (qIdx d s h e : ℤ)).addr = some (q16.get (e, ())) := hq16 (e, ()) rfl
      have hf := fin_of hrd (hH.2.1 e e.2)
      obtain ⟨h1, h2⟩ : (q32.get1 e).isFinite = true ∧ (q32.get1 e).toRat = (q16.get (e, ())).toRat :=
        widen hf (hq32 (e, ()))
      refine ⟨h1, by rw [h2, val16_of hrd], ?_⟩
      rw [toReal_def, h2, ← toReal_def]; exact f16_abs_le hf
    have hS : ∀ r : Fin d.PAGE_SIZE, (d.PAGE_SIZE * t + r < seqLen d init s → (sw.get1 r).isFinite = true ∧
        |(sw.get1 r).toReal - (rowCtx d init s h X).c * (rowCtx d init s h X).sv (d.PAGE_SIZE * t + r)| ≤
          (rowCtx d init s h X).Es) ∧
        (¬ d.PAGE_SIZE * t + r < seqLen d init s → sw.get1 r = FloatV.negInf) := fun r => by
      refine ⟨fun hr => ?_, fun hr => ?_⟩
      · rw [hwget r, if_pos hr]
        exact lane_score hd hRok (hXk _ hr) (kf := fun e => k32.get2 r e) (qf := fun e => q32.get1 e)
          (pr := fun e => pr.get2 r e) (srow := srow.get1 r) (fun e => ⟨(hk' r e hr).1, (hk' r e hr).2.1⟩)
          (fun e => ⟨(hq' e).1, (hq' e).2.1⟩) (fun e => (hk' r e hr).2.2) (fun e => (hq' e).2.2)
          (fun e => hpr (r, e, ())) (hsrow r) (hsc (r, ()))
      · rw [hwget r, if_neg hr]
    have hvt : ∀ (r : Fin d.PAGE_SIZE) (e : Fin d.HEAD_DIM), (v32.get2 r e).isFinite = true ∧
        (d.PAGE_SIZE * t + r < seqLen d init s → (v32.get2 r e).toReal = veOf d init s h e (d.PAGE_SIZE * t + r)) ∧
        (¬ d.PAGE_SIZE * t + r < seqLen d init s → (v32.get2 r e).toReal = 0) := fun r e => by
      by_cases hr : d.PAGE_SIZE * t + r < seqLen d init s
      · have hf := fin_of (hvr r e hr) (hH.2.2.1 _ e hr e.2).2.1
        obtain ⟨h1, h2⟩ : (v32.get2 r e).isFinite = true ∧ (v32.get2 r e).toRat = (v16.get (r, e, ())).toRat :=
          widen hf (hv32 (r, e, ()))
        refine ⟨h1, fun _ => ?_, fun h' => absurd hr h'⟩
        unfold veOf; rw [ite_eq_left hr, toReal_def, h2, val16_of (hvr r e hr)]
      · have hz : v16.get (r, e, ()) = zv.get (r, e, ()) := hv2 (r, e, ()) (decide_eq_false hr) zv rfl
        have hzf := zero_conv (hzv (r, e, ()))
        rw [← hz] at hzf
        obtain ⟨h1, h2⟩ : (v32.get2 r e).isFinite = true ∧ (v32.get2 r e).toRat = (v16.get (r, e, ())).toRat :=
          widen hzf.1 (hv32 (r, e, ()))
        refine ⟨h1, fun h' => absurd h' hr, fun _ => ?_⟩
        rw [toReal_def, h2, hzf.2]; simp
    exact row_step (R := rowCtx d init s h X) (D := d.HEAD_DIM) (sc := fun r => sw.get1 r)
      (x := fun r => x.get1 r) (p := fun r => p.get1 r) (m' := m2.item) (dm := dm.item) (al := al.item)
      (l' := l2.item) (vt := fun r e => v32.get2 r e) (pv := fun r e => pv.get2 r e)
      (vsum := fun e => vsum.get1 e) (acc := fun e => acc.get1 e) (acc' := fun e => acc2.get1 e)
      hexp hni hRok (by omega) hPt' (fun e j => veOf_le hH e e.2 j) hS hmx (hm2 ()) (hdm ()) (hal ())
      (fun r => hx (r, ())) (fun r => hp (r, ())) hlsum (hl2 ()) hvt (fun r e => hpv (r, e, ())) hvsum
      (fun e => hacc2 (e, ())) (hinv hH X hX hXk)
  · rw [htc]
    rintro ⟨acc, l, m⟩ m₁ ⟨hm₁, hinv⟩
    subst m₁
    simp only at hinv ⊢
    -- line 71
    refine safe_div (fun _ => hdivT _ _) fun o' ho' => ?_
    -- line 72
    refine safe_cvt f32_f16_ok fun o16 ho16 => ?_
    have houtp : ∀ e : Fin d.HEAD_DIM, (d.out.ptr 0 + (IntV.ofNat s : I32) * IntV.ofNat d.stride_os +
        (IntV.ofNat h : I32) * IntV.ofNat d.stride_oh + tl.arange 0 d.HEAD_DIM).get (e, ()) =
        d.out.ptr (outIdx d s h e : ℤ) := fun e => by
      show (((d.out.ptr 0).offset ((IntV.ofNat s : I32) * IntV.ofNat d.stride_os).toInt).offset
        ((IntV.ofNat h : I32) * IntV.ofNat d.stride_oh).toInt).offset ((tl.arange 0 d.HEAD_DIM).get1 e).toInt = _
      rw [IntV.I32.mul_nat hsv (i32_nat hsos) (lt_of_le_of_lt (hle_s _) hpos),
        IntV.I32.mul_nat hhv (i32_nat hsoh) (lt_of_le_of_lt (hle_h _) hpoh), arange_val (by omega) e,
        Buf.ptr_offset, Buf.ptr_offset, Buf.ptr_offset]
      congr 1; unfold outIdx; push_cast; ring
    refine safe_last (safe_store_buf d.out (g := fun i : Idx [d.HEAD_DIM] => outIdx d s h i.1)
      (en := fun _ : Idx [d.HEAD_DIM] => true) (Buf.addrs_of_get_all d.out fun ⟨e, ⟨⟩⟩ => houtp e)
      (fun ⟨e, ⟨⟩⟩ _ => hd.out_len s h e hs hh e.2) (fun ⟨e, ⟨⟩⟩ ⟨e', ⟨⟩⟩ _ _ hee => ?_) hpre.2.2.2.1 ?_
      fun m' hm' _ _ => safe_ret ?_)
    · obtain ⟨-, -, he⟩ := hd.out_inj s h e s h e' hs hh e.2 hs hh e'.2 hee
      have : e = e' := Fin.ext he
      subst this; rfl
    · unfold Wr; rw [hsd, hhd]
      exact d.out.elems_mono fun _ ⟨⟨e, ⟨⟩⟩, _, he⟩ => ⟨e, e.2, he.symm⟩
    · unfold Qp; rw [hsd, hhd]
      intro e
      exact ⟨o16.get1 e, hm' (e, ()) rfl, row_out hd hdivF (by omega) hinv (fun e => ho' (e, ()))
        (fun e => ho16 (e, ())) e⟩

end Kernels.PagedAttention
