import Proof.Tile
import Proof.Link
import Proof.Post
import Proof.Loop

/-!
# One program, walked line by line
-/

namespace Claims.AttnFwd

open TVA

noncomputable section

/-- Everything the key loop knows about its inputs: the `q` tile (lane by lane the elements of `q`),
the scale `qk_scale` (a positive `float32` within `3·u32` of `sm_scale / ln 2`), the offsets and the
lane values of `offs_m`, `offs_n`. -/
structure InCtx (d : Inst) (init : Mem) (bh sm : Nat) (q : Tensor [128, d.HEAD_DIM] F16) (c : F32)
    (offset_y start_m N_CTX : I32) (offs_m : Tensor [128] I32) (offs_n : Tensor [64] I32) : Prop where
  hq : ∀ (r : Fin 128) (e : Fin d.HEAD_DIM),
    init.read .f16 (d.q.ptr ((idx d bh (sm * 128 + r) e : Nat) : Int)).addr = some (q.get2 r e)
  hc : c.isFinite = true ∧ 0 < c.toReal ∧
    |c.toReal - (sigma d : ℝ) / Real.log 2| ≤ 31 / 10 * u32 * ((sigma d : ℝ) / Real.log 2)
  hoff : offset_y.toInt = ((bh * d.N_CTX : Nat) : Int)
  hstart : start_m.toInt = (sm : Int)
  hN : N_CTX.toInt = (d.N_CTX : Int)
  hoffm : ∀ r : Fin 128, (offs_m.get1 r).toInt = ((sm * 128 + r : Nat) : Int)
  hoffn : ∀ j : Fin 64, (offs_n.get1 j).toInt = ((j : Nat) : Int)

/-- **The unmasked key loop** over all the keys (`STAGE = 3`, non-causal). -/
theorem safe_inner3 {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d)
    {q : Tensor [128, d.HEAD_DIM] F16} {c : F32} {offset_y start_m N_CTX : I32}
    {offs_m : Tensor [128] I32} {offs_n : Tensor [64] I32}
    (hin : InCtx d init bh sm q c offset_y start_m N_CTX offs_m offs_n)
    (acc : Tensor [128, d.HEAD_DIM] F32) (l_i m_i : Tensor [128] F32)
    (hst : StateOK d init bh sm 0 m_i l_i acc) (hcau : d.causal = false) :
    Safe ((launch d).env P) (Rset d) ∅
      (fun r m' => m' = init ∧ StateOK d init bh sm (d.N_CTX / 64) r.2.2 r.2.1 r.1) init
      (attn_fwd_inner (BLOCK_M := 128) (HEAD_DIM := d.HEAD_DIM) (BLOCK_N := 64) acc l_i m_i q
        (hostDesc d d.k 64) (hostDesc d d.v 64) offset_y start_m c 3 offs_m offs_n N_CTX) := by
  unfold attn_fwd_inner
  simp (config := {decide := true}) only [ite_false, Prog.bind_eq, Prog.pure_eq, Prog.ret_bind]
  have hN := hd.n_mul
  have hNle := hd.n_le
  have hbN := row_bound' hd hbh
  have hrl := hd.rows_lt
  have hnum : d.N_CTX / 64 * 64 = d.N_CTX := by omega
  refine safe_keyloop (P := P) (init := init) (sm := sm) (t0 := 0) (T := d.N_CTX / 64) (t1 := d.N_CTX / 64) (by omega) (by omega) (by omega)
    (lo := IntV.ofInt 0) (hi := N_CTX) (offset_y := offset_y) (by simpa using IntV.toInt_ofInt_zero)
    (by rw [hin.hN]; omega) (by rw [hin.hoff]) hst
    (body := fun _ x => plainTrip d q (hostDesc d d.k 64) (hostDesc d d.v 64) c x) ?_
  · intro t ht _ st _ ho ho2 hs
    have := safe_plain_trip hd hP hpre hbh hsm hin.hq hin.hc (0 + t) (by omega) (by omega) ?_ st ho ho2 hs
    · simpa using this
    · intro r j
      simp only [keys, hcau, Finset.mem_filter, Finset.mem_range, true_or, and_true]
      have := j.isLt
      omega

/-- **The unmasked part of the causal key loop** (`STAGE = 1`): the keys below the query tile. -/
theorem safe_inner1 {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d)
    {q : Tensor [128, d.HEAD_DIM] F16} {c : F32} {offset_y start_m N_CTX : I32}
    {offs_m : Tensor [128] I32} {offs_n : Tensor [64] I32}
    (hin : InCtx d init bh sm q c offset_y start_m N_CTX offs_m offs_n)
    (acc : Tensor [128, d.HEAD_DIM] F32) (l_i m_i : Tensor [128] F32)
    (hst : StateOK d init bh sm 0 m_i l_i acc) :
    Safe ((launch d).env P) (Rset d) ∅
      (fun r m' => m' = init ∧ StateOK d init bh sm (2 * sm) r.2.2 r.2.1 r.1) init
      (attn_fwd_inner (BLOCK_M := 128) (HEAD_DIM := d.HEAD_DIM) (BLOCK_N := 64) acc l_i m_i q
        (hostDesc d d.k 64) (hostDesc d d.v 64) offset_y start_m c 1 offs_m offs_n N_CTX) := by
  unfold attn_fwd_inner
  simp (config := {decide := true}) only [ite_false, ite_true, Prog.bind_eq, Prog.pure_eq, Prog.ret_bind]
  have hN := hd.n_mul
  have hNle := hd.n_le
  have hbN := row_bound' hd hbh
  have hrl := hd.rows_lt
  have hG := G_mul hd
  have h128 : (IntV.ofNat 128 : I32).toInt = 128 := IntV.I32.toInt_ofNat (by norm_num)
  have hsm128 : sm * 128 + 128 ≤ d.N_CTX := by omega
  have hmul : (start_m * IntV.ofNat 128 : I32).toInt = ((sm * 128 : Nat) : Int) :=
    IntV.I32.mul_nat hin.hstart h128 (by omega)
  refine safe_keyloop (P := P) (init := init) (sm := sm) (t0 := 0) (T := 2 * sm) (t1 := 2 * sm)
    (by omega) (by omega) (by omega)
    (lo := IntV.ofInt 0) (hi := start_m * IntV.ofNat 128) (offset_y := offset_y)
    (by simpa using IntV.toInt_ofInt_zero) (by rw [hmul]; push_cast; ring) (by rw [hin.hoff]) hst
    (body := fun _ x => plainTrip d q (hostDesc d d.k 64) (hostDesc d d.v 64) c x) ?_
  intro t ht _ st _ ho ho2 hs
  have := safe_plain_trip hd hP hpre hbh hsm hin.hq hin.hc (0 + t) (by omega) (by omega) ?_ st ho ho2 hs
  · simpa using this
  · intro r j
    simp only [keys, Finset.mem_filter, Finset.mem_range]
    have := j.isLt
    have := r.isLt
    exact ⟨by omega, Or.inr (by omega)⟩

/-- **The masked part of the causal key loop** (`STAGE = 2`): the two blocks of the query tile. -/
theorem safe_inner2 {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d)
    {q : Tensor [128, d.HEAD_DIM] F16} {c : F32} {offset_y start_m N_CTX : I32}
    {offs_m : Tensor [128] I32} {offs_n : Tensor [64] I32}
    (hin : InCtx d init bh sm q c offset_y start_m N_CTX offs_m offs_n)
    (acc : Tensor [128, d.HEAD_DIM] F32) (l_i m_i : Tensor [128] F32)
    (hst : StateOK d init bh sm (2 * sm) m_i l_i acc) (hc : d.causal = true) :
    Safe ((launch d).env P) (Rset d) ∅
      (fun r m' => m' = init ∧ StateOK d init bh sm (2 * sm + 2) r.2.2 r.2.1 r.1) init
      (attn_fwd_inner (BLOCK_M := 128) (HEAD_DIM := d.HEAD_DIM) (BLOCK_N := 64) acc l_i m_i q
        (hostDesc d d.k 64) (hostDesc d d.v 64) offset_y start_m c 2 offs_m offs_n N_CTX) := by
  unfold attn_fwd_inner
  simp (config := {decide := true}) only [ite_false, ite_true, Prog.bind_eq, Prog.pure_eq, Prog.ret_bind]
  have hN := hd.n_mul
  have hNle := hd.n_le
  have hbN := row_bound' hd hbh
  have hrl := hd.rows_lt
  have hG := G_mul hd
  have h128 : (IntV.ofNat 128 : I32).toInt = 128 := IntV.I32.toInt_ofNat (by norm_num)
  have hsm128 : sm * 128 + 128 ≤ d.N_CTX := by omega
  have hmul : (start_m * IntV.ofNat 128 : I32).toInt = ((sm * 128 : Nat) : Int) :=
    IntV.I32.mul_nat hin.hstart h128 (by omega)
  have h1 : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have hsum : (start_m + IntV.ofInt 1 : I32).toInt = ((sm + 1 : Nat) : Int) :=
    IntV.I32.add_nat (y := 1) hin.hstart (by rw [h1]; rfl) (by omega)
  have hfit : (sm + 1) * 128 < 2 ^ 31 := by
    have : (sm + 1) * 128 = sm * 128 + 128 := by ring
    omega
  have hhi : ((start_m + IntV.ofInt 1) * IntV.ofNat 128 : I32).toInt = ((64 * (2 * sm + 2) : Nat) : Int) := by
    rw [IntV.I32.mul_nat hsum h128 hfit]; push_cast; ring
  have hdvd : ((128 : Nat) : Int) ∣ HintVal.val (start_m * IntV.ofNat 128 : I32) := by
    show ((128 : Nat) : Int) ∣ (start_m * IntV.ofNat 128 : I32).toInt
    rw [hmul]; exact ⟨sm, by push_cast; ring⟩
  refine safe_multiple_of1 (v := 128) hdvd ?_
  refine safe_keyloop (P := P) (init := init) (sm := sm) (t0 := 2 * sm) (T := 2) (t1 := 2 * sm + 2)
    rfl (by omega) (by omega)
    (lo := start_m * IntV.ofNat 128) (hi := (start_m + IntV.ofInt 1) * IntV.ofNat 128) (offset_y := offset_y)
    (by rw [hmul]; push_cast; ring) hhi (by rw [hin.hoff]) hst
    (body := fun start_n x => maskedTrip d q (hostDesc d d.k 64) (hostDesc d d.v 64) c offs_m offs_n start_n x) ?_
  intro t ht start_n st hsn ho ho2 hs
  exact safe_masked_trip hd hP hpre hbh hsm hc hin.hq hin.hc hin.hoffm hin.hoffn (2 * sm + t)
    (by omega) (by omega) (by omega) hsn st ho ho2 hs

/-- The epilogue of the kernel (source lines 241–245). -/
def epilogue (d : Inst) (c : Ctx) (desc_o : TensorDesc .f16) (qo : I32)
    (x : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32) : Prog Unit := do
  let w ← tl.log2 x.2.1
  let m_i ← tl.add x.2.2 w
  let acc ← tl.div x.1 (x.2.1.col d.HEAD_DIM)
  tl.store (d.M.ptr 0 + tl.program_id c 1 * IntV.ofNat d.N_CTX +
    (tl.program_id c 0 * IntV.ofNat 128 + tl.arange 0 128)) m_i
  let o16 : Tensor [128, d.HEAD_DIM] F16 ← tl.cvt acc
  tl.desc_store desc_o [qo, IntV.ofInt 0] o16

/-- **The epilogue** after `T` blocks of keys writes the tile's `M` and `o`. -/
theorem safe_epilogue {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) {bh sm : Nat} (hbh : bh < d.Z * d.H) (hsm : sm < G d) (T : Nat)
    (hT2 : 2 ≤ T) (hT1 : T ≤ 1024) (hTN : 64 * T ≤ d.N_CTX)
    (hTk : ∀ r : Fin 128, keys d (sm * 128 + r) ⊆ Finset.range (T * 64))
    {c : Ctx} (hc0 : c.pid 0 = sm) (hc1 : c.pid 1 = bh) {qo : I32}
    (hqo : qo.toInt = ((bh * d.N_CTX + sm * 128 : Nat) : Int))
    (x : Tensor [128, d.HEAD_DIM] F32 × Tensor [128] F32 × Tensor [128] F32)
    (hst : StateOK d init bh sm T x.2.2 x.2.1 x.1) :
    Safe ((launch d).env P) (Rset d) (Wset d bh sm) (fun _ m => Qpost d init bh sm m) init
      (epilogue d c (hostDesc d d.o 128) qo x) := by
  unfold epilogue
  refine safe_math (fn := .log2) (fun i => hP.mathTotal _ _ _) fun w hw => ?_
  refine safe_add fun mi hmi => ?_
  refine safe_div (fun i => hP.divTotal _ _ _) fun qh hqh => ?_
  have hN := hd.n_mul
  have hrowsLt := hd.rows_lt
  have hNle := hd.n_le
  have hrowb := row_bound' hd hbh
  have hG128 := G_mul hd
  have hsm128' : sm * 128 + 128 ≤ d.N_CTX := by omega
  have hN32 : (IntV.ofNat d.N_CTX : I32).toInt = d.N_CTX := IntV.I32.toInt_ofNat (by omega)
  have h128 : (IntV.ofNat 128 : I32).toInt = 128 := IntV.I32.toInt_ofNat (by norm_num)
  have hpid1 : (tl.program_id c 1).toInt = bh := by
    rw [tl.program_id, hc1]; exact IntV.I32.toInt_ofNat (by have := hd.zh_lt; omega)
  have hpid0 : (tl.program_id c 0).toInt = sm := by
    rw [tl.program_id, hc0]; exact IntV.I32.toInt_ofNat (by omega)
  have h1 : (tl.program_id c 1 * IntV.ofNat d.N_CTX : I32).toInt = ((bh * d.N_CTX : Nat) : Int) :=
    IntV.I32.mul_nat hpid1 hN32 (by omega)
  have hsm128 : (tl.program_id c 0 * IntV.ofNat 128 : I32).toInt = ((sm * 128 : Nat) : Int) :=
    IntV.I32.mul_nat (x := sm) (y := 128) hpid0 h128 (by omega)
  have h2 : ∀ i : Idx [128], ((tl.program_id c 0 * IntV.ofNat 128 + tl.arange 0 128 : Tensor [128] I32).get i).toInt
      = ((sm * 128 + i.1.val : Nat) : Int) := by
    rintro ⟨i, ⟨⟩⟩
    exact IntV.I32.toInt_sadd_arange (n := 128) hsm128 (by omega) i
  have hptr : ∀ i : Idx [128],
      (d.M.ptr 0 + tl.program_id c 1 * IntV.ofNat d.N_CTX +
        (tl.program_id c 0 * IntV.ofNat 128 + tl.arange 0 128)).get i =
      d.M.ptr (((fun i : Idx [128] => bh * d.N_CTX + (sm * 128 + i.1.val)) i : Nat) : Int) := by
    intro i
    show ((d.M.ptr 0).offset (tl.program_id c 1 * IntV.ofNat d.N_CTX).toInt).offset _ = _
    rw [h2 i, Buf.ptr_offset, Buf.ptr_offset, h1]
    push_cast; ring_nf
  have hMlen := hd.lens.2.2.2.2
  have himgM : Buf.img (fun i : Idx [128] => bh * d.N_CTX + (sm * 128 + i.1.val)) (fun _ => true) ⊆ mTile d bh sm := by
    rintro _ ⟨i, _, rfl⟩
    exact ⟨i.1.val, i.1.isLt, rfl⟩
  refine safe_store_buf d.M (Buf.addrs_of_get_all d.M hptr) (fun i _ => ?_) (fun i j _ _ hij => ?_)
    hpre.2.2.2.2 ((d.M.elems_mono himgM).trans Set.subset_union_right) fun m' hm' hfr hpres => ?_
  · have := row_bound hd hbh (sm_row_lt hd hsm i.1.isLt)
    show bh * d.N_CTX + (sm * 128 + i.1.val) < d.M.len
    unfold rows at this hMlen; omega
  · rcases i with ⟨i, ⟨⟩⟩; rcases j with ⟨j, ⟨⟩⟩
    have hij' : bh * d.N_CTX + (sm * 128 + (i : Nat)) = bh * d.N_CTX + (sm * 128 + (j : Nat)) := hij
    exact Prod.ext (Fin.ext (by omega)) rfl
  refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩ fun o16 ho16 => ?_
  refine safe_last ?_
  have hqo0 : 0 ≤ qo.toInt := by rw [hqo]; positivity
  have hqoN : qo.toInt.toNat = bh * d.N_CTX + sm * 128 := by rw [hqo, Int.toNat_natCast]
  have hvo : d.o.Valid m' := Buf.valid_of_agree hpre.2.2.2.1 fun x hx => hpres x (hpre.2.2.2.1.2.2.2 x hx)
  have hdisjOM := hd.disj.2.2.2.1
  have himgM' : d.M.elems (Buf.img (fun i : Idx [128] => bh * d.N_CTX + (sm * 128 + i.1.val)) (fun _ => true))
      ⊆ d.M.elems (mTile d bh sm) := d.M.elems_mono himgM
  refine safe_tile_store hd (b := d.o) (by simpa [Buf.ptr] using hd.align.2.2.2) hd.lens.2.2.2.1 hvo 128 qo hqo0
    (by rw [hqo]; push_cast; omega) (oTile d bh sm) (fun r e => ?_) ?_ fun m'' hm'' hfr'' => safe_ret ?_
  · rw [hqoN]
    exact ⟨r, r.isLt, e, e.isLt, by unfold idx; ring⟩
  · intro x hx
    refine ⟨Or.inl hx, fun hxM => ?_⟩
    exact Set.disjoint_left.mp (hdisjOM.mono (d.o.elems_mono (oTile_sub hd hbh hsm))
      (himgM'.trans (d.M.elems_mono (mTile_sub hd hbh hsm)))) hx hxM
  · intro r hr
    let rF : Fin 128 := ⟨r, hr⟩
    refine ⟨fun e he => ?_, ?_⟩
    · let eF : Fin d.HEAD_DIM := ⟨e, he⟩
      refine ⟨o16.get2 rF eF, ?_, fun h => ?_⟩
      · have := hm'' rF eF
        have hidx : ((qo.toInt.toNat + (rF : Nat)) * d.HEAD_DIM + (eF : Nat)) = idx d bh (sm * 128 + r) e := by
          rw [hqoN]; unfold idx; show (bh * d.N_CTX + sm * 128 + r) * d.HEAD_DIM + e = _; ring
        rw [hidx] at this
        exact this
      · exact (row_O hP hd hsm hT2 hT1 hTN hTk hst rF eF (qh := qh.get2 rF eF) (o := o16.get2 rF eF)
          (hqh (rF, eF, ())) (ho16 (rF, eF, ())) h).2 |> fun h2 => by
            exact ⟨(row_O hP hd hsm hT2 hT1 hTN hTk hst rF eF (hqh (rF, eF, ())) (ho16 (rF, eF, ())) h).1, h2⟩
    · refine ⟨mi.get1 rF, ?_, fun h => ?_⟩
      · have hmem : bh * d.N_CTX + (sm * 128 + r) ∈ mTile d bh sm := ⟨r, hr, rfl⟩
        have hrd := hm' (rF, ()) rfl
        rw [Mem.read_congr (m₂ := m') (fun y hy => hfr'' y (fun hyo => ?_))]
        · exact hrd
        · exact Set.disjoint_left.mp (hdisjOM.mono (d.o.elems_mono (oTile_sub hd hbh hsm))
            (d.M.elems_mono (mTile_sub hd hbh hsm))) hyo (d.M.span_sub_elems hmem hy)
      · exact row_M hP hd hsm hT2 hT1 hTN hTk hst rF (hw (rF, ())) (hmi (rF, ())) h


/-- The kernel's start state: `m_i = −∞`, `l_i = 1`, `acc = 0`. -/
theorem state_init {d : Inst} (bh sm : Nat) {m_i l_i : Tensor [128] F32}
    (hm : ∀ i, FloatV.NaNEq (FloatV.sub ((tl.zeros [128] DType.f32).get i)
      ((bc (FloatV.ofPat 9218868437227405312).litF32 : Tensor [128] F32).get i)) (m_i.get i))
    (hl : ∀ i, FloatV.NaNEq (FloatV.add ((tl.zeros [128] DType.f32).get i)
      ((bc (FloatV.ofPat 4607182418800017408).litF32 : Tensor [128] F32).get i)) (l_i.get i)) :
    ∀ init, StateOK d init bh sm 0 m_i l_i (tl.zeros [128, d.HEAD_DIM] DType.f32) := by
  intro init h r
  left
  have hm' := hm (r, ())
  have hl' := hl (r, ())
  have e1 : FloatV.sub ((tl.zeros [128] DType.f32).get (r, ()))
      ((bc (FloatV.ofPat 9218868437227405312).litF32 : Tensor [128] F32).get (r, ())) = FloatV.negInf := by
    change FloatV.sub (DType.f32.ofBits 0) (FloatV.ofPat 9218868437227405312).litF32 = _
    decide +kernel
  have e2 : FloatV.add ((tl.zeros [128] DType.f32).get (r, ()))
      ((bc (FloatV.ofPat 4607182418800017408).litF32 : Tensor [128] F32).get (r, ())) =
      FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) := by
    change FloatV.add (DType.f32.ofBits 0) (FloatV.ofPat 4607182418800017408).litF32 = _
    decide +kernel
  rw [e1] at hm'
  rw [e2] at hl'
  have n1 : (FloatV.negInf : F32).isNaN = false := by decide +kernel
  have n2 : (FloatV.litF32 (FloatV.ofPat 0x3FF0000000000000) : F32).isNaN = false := by decide +kernel
  refine ⟨rfl, hm'.eq n1, hl'.eq n2, fun e => ?_⟩
  change DType.f32.ofBits 0 = _
  decide +kernel

set_option maxHeartbeats 800000 in
/-- The algebra of the scale: `σ̂ = σ(1+t₁)`, `κ = k₁(1+t₂)`, `c = σ̂κ` rounded. -/
theorem scale_alg {σ σ' κ κ0 k1 c : ℝ} (hσ : 1 / 2 ^ 64 ≤ σ) (hs : |σ' - σ| ≤ σ * (u32 + 1 / 2 ^ 86))
    (hκ : |κ - κ0| ≤ κ0 * (u32 + 1 / 2 ^ 100)) (hk : |κ0 - k1| ≤ k1 * (1 / 10 ^ 9))
    (hk1 : 144 / 100 ≤ k1) (_hk2 : k1 ≤ 145 / 100) (hc : |c - σ' * κ| ≤ u32 * |σ' * κ| + 1 / 2 ^ 150) :
    |c - σ * k1| ≤ 31 / 10 * u32 * (σ * k1) ∧ 0 < c := by
  have hσ0 : 0 < σ := lt_of_lt_of_le (by positivity) hσ
  have hk0 : 0 < k1 := by linarith
  have hu : u32 = 1 / 2 ^ 24 := rfl
  -- relative errors
  obtain ⟨t1, ht1, e1⟩ : ∃ t1, |t1| ≤ u32 + 1 / 2 ^ 86 ∧ σ' = σ * (1 + t1) := by
    refine ⟨(σ' - σ) / σ, ?_, by field_simp; ring⟩
    rw [abs_div, abs_of_pos hσ0, div_le_iff₀ hσ0]; linarith
  obtain ⟨t2, ht2, e2⟩ : ∃ t2, |t2| ≤ (u32 + 1 / 2 ^ 100) * (1 + 1 / 10 ^ 9) + 1 / 10 ^ 9 ∧
      κ = k1 * (1 + t2) := by
    refine ⟨(κ - k1) / k1, ?_, by field_simp; ring⟩
    rw [abs_div, abs_of_pos hk0, div_le_iff₀ hk0]
    have h1 : |κ - k1| ≤ |κ - κ0| + |κ0 - k1| := by
      have := abs_add_le (κ - κ0) (κ0 - k1); rwa [sub_add_sub_cancel] at this
    have h2 : κ0 ≤ k1 * (1 + 1 / 10 ^ 9) := by
      have := (abs_le.mp hk).2; nlinarith
    have h3 : κ0 * (u32 + 1 / 2 ^ 100) ≤ k1 * (1 + 1 / 10 ^ 9) * (u32 + 1 / 2 ^ 100) :=
      mul_le_mul_of_nonneg_right h2 (by rw [hu]; positivity)
    nlinarith
  have hA : σ' * κ = σ * k1 * ((1 + t1) * (1 + t2)) := by rw [e1, e2]; ring
  set c0 := σ * k1 with hc0
  have hc00 : 0 < c0 := by positivity
  have ht1' := abs_le.mp ht1
  have ht2' := abs_le.mp ht2
  have hu0 : 0 < u32 := by rw [hu]; positivity
  have hp : |(1 + t1) * (1 + t2) - 1| ≤ 202 / 100 * u32 := by
    have : (1 + t1) * (1 + t2) - 1 = t1 + t2 + t1 * t2 := by ring
    rw [this]
    calc |t1 + t2 + t1 * t2| ≤ |t1| + |t2| + |t1| * |t2| := by
          refine (abs_add_le _ _).trans ?_
          rw [abs_mul]
          linarith [abs_add_le t1 t2]
      _ ≤ (u32 + 1 / 2 ^ 86) + ((u32 + 1 / 2 ^ 100) * (1 + 1 / 10 ^ 9) + 1 / 10 ^ 9) +
          (u32 + 1 / 2 ^ 86) * ((u32 + 1 / 2 ^ 100) * (1 + 1 / 10 ^ 9) + 1 / 10 ^ 9) := by
          have := mul_le_mul ht1 ht2 (abs_nonneg _) (by rw [hu]; norm_num)
          linarith
      _ ≤ 202 / 100 * u32 := by rw [hu]; norm_num
  have hpos : 0 < (1 + t1) * (1 + t2) := by
    have a1 : u32 + 1 / 2 ^ 86 < 1 := by rw [hu]; norm_num
    have a2 : (u32 + 1 / 2 ^ 100) * (1 + 1 / 10 ^ 9) + 1 / 10 ^ 9 < 1 := by rw [hu]; norm_num
    apply mul_pos <;> linarith [ht1'.1, ht2'.1]
  have hAabs : |σ' * κ| = c0 * ((1 + t1) * (1 + t2)) := by rw [hA, abs_of_pos (by positivity)]
  have hq := abs_le.mp hp
  have hP1 : (1 + t1) * (1 + t2) ≤ 1 + 202 / 100 * u32 := by linarith [hq.2]
  have hP2 : 1 - 202 / 100 * u32 ≤ (1 + t1) * (1 + t2) := by linarith [hq.1]
  have hAle : |σ' * κ| ≤ c0 * (1 + 202 / 100 * u32) := by
    rw [hAabs]; exact mul_le_mul_of_nonneg_left hP1 hc00.le
  have hAge : c0 * (1 - 202 / 100 * u32) ≤ |σ' * κ| := by
    rw [hAabs]; exact mul_le_mul_of_nonneg_left hP2 hc00.le
  have hdev : |σ' * κ - c0| ≤ 202 / 100 * u32 * c0 := by
    rw [hA]
    have : c0 * ((1 + t1) * (1 + t2)) - c0 = c0 * ((1 + t1) * (1 + t2) - 1) := by ring
    rw [this, abs_mul, abs_of_pos hc00]
    have := mul_le_mul_of_nonneg_left hp hc00.le
    linarith
  have hlow : (1 : ℝ) / 2 ^ 150 ≤ c0 * (1 / 2 ^ 86) := by
    have h1 : (1 / 2 ^ 64 : ℝ) * (144 / 100) ≤ c0 :=
      mul_le_mul hσ hk1 (by norm_num) hσ0.le
    have h2 : (1 : ℝ) / 2 ^ 150 ≤ (1 / 2 ^ 64 : ℝ) * (144 / 100) * (1 / 2 ^ 86) := by norm_num
    have h3 := mul_le_mul_of_nonneg_right h1 (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 86)
    linarith
  have hnum : u32 * (c0 * (1 + 202 / 100 * u32)) + c0 * (1 / 2 ^ 86) ≤ 108 / 100 * u32 * c0 := by
    rw [hu]
    nlinarith [hc00]
  have hu1 := mul_le_mul_of_nonneg_left hAle hu0.le
  have hcA := abs_le.mp hc
  have hdv := abs_le.mp hdev
  constructor
  · rw [abs_le]
    constructor <;> linarith [hcA.1, hcA.2, hdv.1, hdv.2, hnum, hlow, hu1]
  · have hAp : 0 < σ' * κ := by rw [hA]; positivity
    have hAeq : |σ' * κ| = σ' * κ := abs_of_pos hAp
    rw [hAeq] at hcA hAge
    have hn1 : (9 : ℝ) / 10 ≤ (1 - u32) * (1 - 202 / 100 * u32) := by rw [hu]; norm_num
    have h1 : (1 - u32) * (c0 * (1 - 202 / 100 * u32)) ≤ (1 - u32) * (σ' * κ) :=
      mul_le_mul_of_nonneg_left hAge (by rw [hu]; norm_num)
    have h2 : 9 / 10 * c0 ≤ (1 - u32) * (c0 * (1 - 202 / 100 * u32)) := by
      have e : (1 - u32) * (c0 * (1 - 202 / 100 * u32)) = c0 * ((1 - u32) * (1 - 202 / 100 * u32)) := by
        ring
      rw [e]
      have := mul_le_mul_of_nonneg_left hn1 hc00.le
      linarith
    have h3 : (1 : ℝ) / 2 ^ 86 < 9 / 10 := by norm_num
    have h4 : c0 * (1 / 2 ^ 86) < c0 * (9 / 10) := mul_lt_mul_of_pos_left h3 hc00
    have h5 : (1 - u32) * (σ' * κ) = σ' * κ - u32 * (σ' * κ) := by ring
    linarith [hcA.1, h1, h2, h4, hlow, h5]

/-- The scale `qk_scale = sm_scale · 1.44269504` (both `float32`, one product). -/
theorem scale_fact {d : Inst} (hd : d.ok) {r : Tensor [] F32}
    (hr : ∀ i, FloatV.NaNEq (FloatV.mul ((Tensor.scalar (FloatV.hostF32 d.sm_scale)).get i)
      ((Tensor.scalar (FloatV.ofPat 4609176140017200175).litF32).get i)) (r.get i)) :
    r.item.isFinite = true ∧ 0 < r.item.toReal ∧
      |r.item.toReal - (sigma d : ℝ) / Real.log 2| ≤ 31 / 10 * u32 * ((sigma d : ℝ) / Real.log 2) := by
  obtain ⟨hfin, hlo, hhi⟩ := hd.scale
  have hreg : Format.binary32.Regular := Format.Std.regular Format.binary32_std
  have hσ : (d.sm_scale.toReal) = (sigma d : ℝ) := rfl
  have hσlo : (1 : ℝ) / 2 ^ 64 ≤ (sigma d : ℝ) := by
    have : ((1 / 2 ^ 64 : ℚ) : ℝ) ≤ (sigma d : ℝ) := by exact_mod_cast hlo
    simpa using this
  have hσhi : (sigma d : ℝ) ≤ 2 ^ 64 := by
    have : (sigma d : ℝ) ≤ ((2 ^ 64 : ℚ) : ℝ) := by exact_mod_cast hhi
    simpa using this
  have hnan : d.sm_scale.isNaN = false := (FloatV.isFinite_iff.mp hfin).1
  have e1 : FloatV.hostF32 d.sm_scale = FloatV.cvt d.sm_scale := by
    unfold FloatV.hostF32; simp [hnan]
  -- the host's float32 of sm_scale
  have hσpos : 0 < (sigma d : ℝ) := by
    have : (0 : ℝ) < 1 / 2 ^ 64 := by norm_num
    linarith
  obtain ⟨hs1, hs2⟩ := FloatV.cvt_err_real (g := Format.binary32) hreg hfin (by
    rw [hσ, Format.binary32_maxVal_real, abs_of_pos hσpos]; nlinarith)
  rw [Format.binary32_u_real, Format.binary32_eta_real, hσ, abs_of_pos hσpos] at hs2
  -- the constant 1.44269504
  have hx1 : (FloatV.ofPat 4609176140017200175 : F64).isFinite = true := by decide +kernel
  have hx0 : (FloatV.ofPat 4609176140017200175 : F64).isZero = false := by decide +kernel
  have hxv : (FloatV.ofPat 4609176140017200175 : F64).toReal = 6497320844553263 / 4503599627370496 := by
    unfold FloatV.toReal
    have : (FloatV.ofPat 4609176140017200175 : F64).toRat = 6497320844553263 / 4503599627370496 := by
      decide +kernel
    rw [this]; norm_num
  have e2 : (FloatV.ofPat 4609176140017200175).litF32 = FloatV.cvt (FloatV.ofPat 4609176140017200175 : F64) := by
    unfold FloatV.litF32 FloatV.hostF32
    have hn : (FloatV.ofPat 4609176140017200175 : F64).isNaN = false := by decide +kernel
    simp [hx0, hn]
  obtain ⟨hk1', hk2⟩ := FloatV.cvt_err_real (g := Format.binary32) hreg hx1 (by
    rw [hxv, Format.binary32_maxVal_real]; norm_num)
  rw [Format.binary32_u_real, Format.binary32_eta_real, hxv] at hk2
  have hη : (2 : ℝ) ^ (-150 : ℤ) ≤ (sigma d : ℝ) * (1 / 2 ^ 86) := by
    have : (2 : ℝ) ^ (-150 : ℤ) = (1 / 2 ^ 64) * (1 / 2 ^ 86) := by norm_num
    rw [this]; exact mul_le_mul_of_nonneg_right hσlo (by norm_num)
  have hs' : |(FloatV.hostF32 d.sm_scale).toReal - (sigma d : ℝ)| ≤ (sigma d : ℝ) * (u32 + 1 / 2 ^ 86) := by
    rw [e1]
    refine hs2.trans ?_
    have : (2 : ℝ) ^ (-24 : ℤ) = u32 := by unfold u32; norm_num
    rw [this]; nlinarith
  -- κ against `1/ln 2`
  have hlog := Real.log_two_near_10
  have hlogpos : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hP : |(6497320844553263 / 4503599627370496 : ℝ) * Real.log 2 - 1| ≤ 1 / 10 ^ 9 := by
    have e : (6497320844553263 / 4503599627370496 : ℝ) * Real.log 2 - 1 =
        (6497320844553263 / 4503599627370496) * (Real.log 2 - 287209 / 414355) +
          ((6497320844553263 / 4503599627370496) * (287209 / 414355) - 1) := by ring
    rw [e]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 6497320844553263 / 4503599627370496)]
    have h1 := mul_le_mul_of_nonneg_left hlog (by norm_num : (0 : ℝ) ≤ 6497320844553263 / 4503599627370496)
    have h2 : |(6497320844553263 / 4503599627370496 : ℝ) * (287209 / 414355) - 1| ≤ 7 / 10 ^ 10 := by
      rw [abs_le]; constructor <;> norm_num
    have h3 : (6497320844553263 / 4503599627370496 : ℝ) * (1 / 10 ^ 10) ≤ 2 / 10 ^ 10 := by norm_num
    linarith
  have hkk : |(6497320844553263 / 4503599627370496 : ℝ) - 1 / Real.log 2| ≤
      1 / Real.log 2 * (1 / 10 ^ 9) := by
    have e : (6497320844553263 / 4503599627370496 : ℝ) - 1 / Real.log 2 =
        ((6497320844553263 / 4503599627370496 : ℝ) * Real.log 2 - 1) / Real.log 2 := by
      field_simp
    rw [e, abs_div, abs_of_pos hlogpos, div_le_iff₀ hlogpos]
    have : 1 / Real.log 2 * (1 / 10 ^ 9) * Real.log 2 = 1 / 10 ^ 9 := by field_simp
    rw [this]; exact hP
  have hlk1 : (144 : ℝ) / 100 ≤ 1 / Real.log 2 := by
    rw [le_div_iff₀ hlogpos]; linarith [Real.log_two_lt_d9]
  have hlk2 : 1 / Real.log 2 ≤ (145 : ℝ) / 100 := by
    rw [div_le_iff₀ hlogpos]; linarith [Real.log_two_gt_d9]
  -- κ
  have hκ0 : (6497320844553263 / 4503599627370496 : ℝ) ≥ 144 / 100 := by norm_num
  have hκη : (2 : ℝ) ^ (-150 : ℤ) ≤ (6497320844553263 / 4503599627370496 : ℝ) * (1 / 2 ^ 100) := by
    norm_num
  have hκ' : |(FloatV.ofPat 4609176140017200175).litF32.toReal - 6497320844553263 / 4503599627370496| ≤
      (6497320844553263 / 4503599627370496 : ℝ) * (u32 + 1 / 2 ^ 100) := by
    rw [e2]
    refine hk2.trans ?_
    have : (2 : ℝ) ^ (-24 : ℤ) = u32 := by unfold u32; norm_num
    rw [abs_of_pos (by norm_num : (0 : ℝ) < 6497320844553263 / 4503599627370496), this]
    nlinarith
  -- the product
  have hfs : (FloatV.hostF32 d.sm_scale).isFinite = true := by rw [e1]; exact hs1
  have hfk : ((FloatV.ofPat 4609176140017200175 : F64).litF32 : F32).isFinite = true := by rw [e2]; exact hk1'
  have hu : u32 = 1 / 2 ^ 24 := rfl
  have hκ0' : (FloatV.ofPat 4609176140017200175).litF32.toReal ≤ 3 / 2 := by
    have := (abs_le.mp hκ').2
    rw [hu] at this
    norm_num at this
    linarith
  have hκ0'' : 0 < (FloatV.ofPat 4609176140017200175).litF32.toReal := by
    have := (abs_le.mp hκ').1
    rw [hu] at this
    norm_num at this
    linarith
  have hσ0' : (FloatV.hostF32 d.sm_scale).toReal ≤ 2 ^ 65 := by
    have := (abs_le.mp hs').2
    rw [hu] at this
    have h2 := mul_le_mul_of_nonneg_right hσhi (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 24 + 1 / 2 ^ 86)
    norm_num at this h2 ⊢
    linarith
  have hσ0'' : 0 < (FloatV.hostF32 d.sm_scale).toReal := by
    have := (abs_le.mp hs').1
    rw [hu] at this
    have h2 := mul_le_mul_of_nonneg_right hσlo (by norm_num : (0 : ℝ) ≤ 1 / 2 ^ 24 + 1 / 2 ^ 86)
    norm_num at this h2 ⊢
    nlinarith
  have hmaxp : |(FloatV.hostF32 d.sm_scale).toReal * (FloatV.ofPat 4609176140017200175).litF32.toReal| ≤
      (Format.binary32.maxVal : ℝ) := by
    rw [abs_of_pos (mul_pos hσ0'' hκ0''), Format.binary32_maxVal_real]
    have := mul_le_mul hσ0' hκ0' hκ0''.le (by norm_num)
    norm_num at this ⊢
    linarith
  obtain ⟨hmf, hme⟩ := FloatV.mul_err_real hreg hfs hfk hmaxp
  have hrr := (hr ()).eq_of_isFinite hmf
  have hritem : r.item = FloatV.mul (FloatV.hostF32 d.sm_scale) (FloatV.ofPat 4609176140017200175).litF32 := hrr
  rw [Format.binary32_u_real, Format.binary32_eta_real] at hme
  have hmain := scale_alg (σ := (sigma d : ℝ)) (σ' := (FloatV.hostF32 d.sm_scale).toReal)
    (κ := (FloatV.ofPat 4609176140017200175).litF32.toReal) (κ0 := 6497320844553263 / 4503599627370496)
    (k1 := 1 / Real.log 2) (c := r.item.toReal) hσlo hs' hκ' hkk hlk1 hlk2 (by
      rw [hritem]
      have h24 : (2 : ℝ) ^ (-24 : ℤ) = u32 := by unfold u32; norm_num
      have h150 : (2 : ℝ) ^ (-150 : ℤ) = 1 / 2 ^ 150 := by norm_num
      rw [h24, h150] at hme
      linarith [mul_comm |(FloatV.hostF32 d.sm_scale).toReal * (FloatV.ofPat 4609176140017200175).litF32.toReal| u32])
  refine ⟨by rw [hritem]; exact hmf, hmain.2, ?_⟩
  have e : (sigma d : ℝ) / Real.log 2 = (sigma d : ℝ) * (1 / Real.log 2) := by ring
  rw [e]; exact hmain.1

/-- Program id 0 of the context at launch index `i`. -/
theorem ctx_pid0 (d : Inst) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 0 = (i : Nat) % G d := rfl

/-- Program id 1 of the context at launch index `i`, for valid instances. -/
theorem ctx_pid1 (d : Inst) (hd : d.ok) (i : Fin (launch d).size) :
    ((launch d).ctx i).pid 1 = (i : Nat) / G d := by
  have hG := G_pos hd
  have hsize := launch_size d
  have : (i : Nat) / G d < d.Z * d.H := (Nat.div_lt_iff_lt_mul hG).mpr (by
    have := lt_of_lt_of_eq i.isLt hsize; rwa [Nat.mul_comm] at this)
  show (i : Nat) / G d % (d.Z * d.H) = _
  exact Nat.mod_eq_of_lt this

/-- **One program** of the launch is safe from the initial memory: it reads only `q`, `k`, `v`,
writes only its tile of `o` and `M`, and leaves `Qpost` in it. The prologue computes the offsets,
the scale and the `q` tile; the key loops (`safe_inner3`, or `safe_inner1` then `safe_inner2` when
causal) carry the state of the rows; the epilogue stores. -/
theorem safe_program {d : Inst} (hd : d.ok) {P : Platform} (hP : Device P) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (Rset d) (Wset d (i / G d) (i % G d))
      (fun _ m => Qpost d init (i / G d) (i % G d) m) init ((model d).body ((launch d).ctx i)) := by
  have hG := G_pos hd
  have hsize := launch_size d
  have hbh' : (i : Nat) / G d < d.Z * d.H := (Nat.div_lt_iff_lt_mul hG).mpr (by
    have := lt_of_lt_of_eq i.isLt hsize; rwa [Nat.mul_comm] at this)
  have hsm' : (i : Nat) % G d < G d := Nat.mod_lt _ hG
  generalize hsmd : (i : Nat) % G d = sm at hsm' ⊢
  generalize hbhd : (i : Nat) / G d = bh at hbh' ⊢
  have hc0 : ((launch d).ctx i).pid 0 = sm := by rw [ctx_pid0, hsmd]
  have hc1 : ((launch d).ctx i).pid 1 = bh := by rw [ctx_pid1 d hd, hbhd]
  generalize ((launch d).ctx i) = c at hc0 hc1 ⊢
  simp only [model, attn_fwd, maybe_make_tensor_desc]
  have hN := hd.n_mul
  have hrows := hd.rows_lt
  have hzh := hd.zh_pos
  have hNle := hd.n_le
  have hH32 : (IntV.ofNat d.H : I32).toInt = d.H := IntV.I32.toInt_ofNat (by
    have : d.H ≤ d.Z * d.H := Nat.le_mul_of_pos_left _ hzh.1
    have := hd.zh_lt
    omega)
  have hN32 : (IntV.ofNat d.N_CTX : I32).toInt = d.N_CTX := IntV.I32.toInt_ofNat (by omega)
  have hpid1 : (tl.program_id c 1).toInt = bh := by
    rw [tl.program_id, hc1]; exact IntV.I32.toInt_ofNat (by have := hd.zh_lt; omega)
  have hpid0 : (tl.program_id c 0).toInt = sm := by
    rw [tl.program_id, hc0]; exact IntV.I32.toInt_ofNat (by have := G_mul hd; omega)
  have hHpos : 0 < d.H := hzh.2
  refine safe_idiv1 (x := tl.program_id c 1) (y := IntV.ofNat d.H) (by rw [hpid1]; positivity)
    (by rw [hH32]; exact_mod_cast hHpos) fun off_z hz => ?_
  refine safe_irem1 (x := tl.program_id c 1) (y := IntV.ofNat d.H) (by rw [hpid1]; positivity)
    (by rw [hH32]; exact_mod_cast hHpos) fun off_h hh => ?_
  rw [hpid1, hH32] at hz hh
  have hz' : off_z.item.toInt = ((bh / d.H : Nat) : Int) := by rw [hz]; push_cast; rfl
  have hh' : off_h.item.toInt = ((bh % d.H : Nat) : Int) := by rw [hh]; push_cast; rfl
  have hoff : (off_z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) + off_h.item * IntV.ofNat d.N_CTX).toInt
      = ((bh * d.N_CTX : Nat) : Int) := by
    have hbhN : bh * d.N_CTX < rows d := by
      have := row_bound hd hbh' hN.1; omega
    have hNH : d.N_CTX * d.H < 2 ^ 31 := by
      have : d.N_CTX * d.H ≤ rows d := by
        unfold rows
        calc d.N_CTX * d.H = 1 * d.H * d.N_CTX := by ring
          _ ≤ d.Z * d.H * d.N_CTX := Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hzh.1)
      omega
    have h1 : (IntV.ofNat d.N_CTX * IntV.ofNat d.H : I32).toInt = ((d.N_CTX * d.H : Nat) : Int) :=
      IntV.I32.mul_nat hN32 hH32 hNH
    have hdiv := Nat.div_add_mod bh d.H
    have hlt : bh % d.H < d.H := Nat.mod_lt _ hHpos
    have h2 : (off_z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H)).toInt =
        ((bh / d.H * (d.N_CTX * d.H) : Nat) : Int) :=
      IntV.I32.mul_nat hz' h1 (by
        have : bh / d.H * (d.N_CTX * d.H) ≤ bh * d.N_CTX := by
          calc bh / d.H * (d.N_CTX * d.H) = (d.H * (bh / d.H)) * d.N_CTX := by ring
            _ ≤ bh * d.N_CTX := Nat.mul_le_mul_right _ (by omega)
        omega)
    have h3 : (off_h.item * IntV.ofNat d.N_CTX).toInt = ((bh % d.H * d.N_CTX : Nat) : Int) :=
      IntV.I32.mul_nat hh' hN32 (by
        have : bh % d.H * d.N_CTX ≤ bh * d.N_CTX := Nat.mul_le_mul_right _ (Nat.mod_le bh d.H)
        omega)
    have h4 := IntV.I32.add_nat h2 h3 (by
      have : bh / d.H * (d.N_CTX * d.H) + bh % d.H * d.N_CTX = bh * d.N_CTX := by
        calc _ = (d.H * (bh / d.H) + bh % d.H) * d.N_CTX := by ring
          _ = bh * d.N_CTX := by rw [hdiv]
      omega)
    rw [h4]
    congr 1
    calc _ = (d.H * (bh / d.H) + bh % d.H) * d.N_CTX := by ring
      _ = bh * d.N_CTX := by rw [hdiv]
  have h128 : (IntV.ofNat 128 : I32).toInt = 128 := IntV.I32.toInt_ofNat (by norm_num)
  have hsm128 : (tl.program_id c 0 * IntV.ofNat 128).toInt = ((sm * 128 : Nat) : Int) :=
    IntV.I32.mul_nat (x := sm) (y := 128) hpid0 h128 (by have := G_mul hd; omega)
  have hqo : (off_z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) + off_h.item * IntV.ofNat d.N_CTX +
      tl.program_id c 0 * IntV.ofNat 128).toInt = ((bh * d.N_CTX + sm * 128 : Nat) : Int) :=
    IntV.I32.add_nat hoff hsm128 (by
      have := row_bound' hd hbh'
      have := G_mul hd
      have : sm * 128 + 128 ≤ d.N_CTX := by omega
      omega)
  simp only [Prog.bind_eq, Prog.ret_bind, Prog.pure_eq]
  refine safe_sub fun m_i hm => ?_
  refine safe_add fun l_i hl => ?_
  refine safe_mul fun cs hcs => ?_
  refine safe_tile_load hd (b := d.q) (by have := hd.align.1; exact (by simpa [Buf.ptr] using this)) hd.lens.1
    hpre.1 128 _ (by rw [hqo]; positivity) (by
      rw [hqo]
      have := G_mul hd
      have := row_bound' hd hbh'
      have : sm * 128 + 128 ≤ d.N_CTX := by omega
      push_cast; omega) (Set.subset_union_left.trans Set.subset_union_left)
    fun q hq => ?_
  have hin : InCtx d init bh sm q cs.item
      (off_z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) + off_h.item * IntV.ofNat d.N_CTX)
      (tl.program_id c 0) (IntV.ofNat d.N_CTX) (tl.program_id c 0 * IntV.ofNat 128 + tl.arange 0 128)
      (tl.arange 0 64) := by
    refine ⟨fun r e => ?_, scale_fact hd hcs, hoff, hpid0, hN32, fun r => ?_, fun j => ?_⟩
    · have := hq r e
      rw [hqo, Int.toNat_natCast] at this
      have e1 : idx d bh (sm * 128 + (r : Nat)) e = (bh * d.N_CTX + sm * 128 + (r : Nat)) * d.HEAD_DIM + (e : Nat) := by
        unfold idx; ring
      rw [e1]; exact this
    · exact IntV.I32.toInt_sadd_arange (n := 128) hsm128 (by have := G_mul hd; omega) r
    · show ((tl.arange 0 64).get1 j).toInt = _
      rw [tl.arange, Tensor.get1_arange]
      rw [IntV.I32.toInt_ofInt (by have := j.isLt; constructor <;> omega)]
      simp
  have hst0 := state_init (d := d) bh sm hm hl
  have hrow := row_bound' hd hbh'
  have hG128 := G_mul hd
  cases hcaus : d.causal
  · simp (config := {decide := true}) only [ite_false]
    refine Safe.bind_ro ((safe_inner3 hd hP hpre hbh' hsm' hin _ _ _ (hst0 init) hcaus).mono (fun r m' hr => ?_) le_rfl)
    obtain ⟨rfl, hr⟩ := hr
    exact safe_epilogue hd hP hpre hbh' hsm' (d.N_CTX / 64) (by omega) (by omega) (by omega) (by
      intro r k hk
      simp only [keys, Finset.mem_filter, Finset.mem_range, hcaus] at hk ⊢
      omega) hc0 hc1 hqo r hr
  · simp (config := {decide := true}) only [ite_true]
    refine Safe.bind_ro ((safe_inner1 hd hP hpre hbh' hsm' hin _ _ _ (hst0 init)).mono (fun r m' hr => ?_) le_rfl)
    obtain ⟨rfl, hr⟩ := hr
    refine Safe.bind_ro ((safe_inner2 hd hP hpre hbh' hsm' hin _ _ _ hr hcaus).mono (fun r' m' hr' => ?_) le_rfl)
    obtain ⟨rfl, hr'⟩ := hr'
    exact safe_epilogue hd hP hpre hbh' hsm' (2 * sm + 2) (by omega) (by omega) (by omega) (by
      intro r k hk
      have hrr := r.isLt
      simp [keys, hcaus] at hk ⊢
      omega) hc0 hc1 hqo r' hr'

end

end Claims.AttnFwd
