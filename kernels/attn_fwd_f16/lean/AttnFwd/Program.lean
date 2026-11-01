import AttnFwd.Inner

/-!
# `attn_fwd_f16`: one program of the launch

* `row0`: program `i`'s query tile starts at row `128·i` of the `[Z·H·N_CTX, HEAD_DIM]` view
  (`off_hz·N_CTX + 128·start_m`);
* `oTile`, `mTile`, `Wr`: the elements of `o` and `M` program `i` writes;
* `OutO`, `OutM`, `Qp`: what `Post` asks of an element, and of a program's tile;
* `row_out`: the end of a row (lines 241–245) from its state after the loop (`final_o`, `final_M`);
* `safe_epi`: lines 241–245 of a program;
* `safe_program`: a whole program (lines 189–245).
-/

namespace Kernels.AttnFwd

open TVA TVA.Math.Float

/-! ## The tiles -/

theorem n_eq {d : Inst} (hd : d.ok) : d.N_CTX = 128 * g0 d := by
  rw [g0_eq hd]; have := hd.n_mul; omega

/-- **Program `i`'s first row** in the `[Z·H·N_CTX, HEAD_DIM]` view: `off_hz·N_CTX + 128·start_m`
is `128·i`. -/
theorem row0 {d : Inst} (hd : d.ok) (i : ℕ) : pbh d i * d.N_CTX + 128 * pm d i = 128 * i := by
  have h2 := Nat.div_add_mod i (g0 d)
  unfold pbh pm
  rw [n_eq hd]
  calc i / g0 d * (128 * g0 d) + 128 * (i % g0 d) = 128 * (g0 d * (i / g0 d) + i % g0 d) := by ring
    _ = 128 * i := by rw [h2]

theorem tile_le {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) : 128 * i + 128 ≤ rows d := by
  rw [launch_size] at hi
  have h1 := n_eq hd
  have : i + 1 ≤ g0 d * (d.Z * d.H) := hi
  calc 128 * i + 128 = 128 * (i + 1) := by ring
    _ ≤ 128 * (g0 d * (d.Z * d.H)) := Nat.mul_le_mul_left _ this
    _ = rows d := by unfold rows; rw [h1]; ring

/-- The elements of `o` program `i` writes (line 245): rows `128·i …` of the view. -/
def oTile (d : Inst) (i : ℕ) : Set ℕ :=
  {x | ∃ r < 128, ∃ e < d.HEAD_DIM, x = (128 * i + r) * d.HEAD_DIM + e}

/-- The elements of `M` program `i` writes (line 244). -/
def mTile (i : ℕ) : Set ℕ := {x | ∃ r < 128, x = 128 * i + r}

/-- Program `i`'s write footprint. -/
def Wr (d : Inst) (i : ℕ) : Set Int := d.o.elems (oTile d i) ∪ d.M.elems (mTile i)

theorem oTile_sub {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) :
    oTile d i ⊆ below (rows d * d.HEAD_DIM) := by
  rintro _ ⟨r, hr, e, he, rfl⟩
  have h1 := tile_le hd hi
  show (128 * i + r) * d.HEAD_DIM + e < rows d * d.HEAD_DIM
  calc (128 * i + r) * d.HEAD_DIM + e < (128 * i + r) * d.HEAD_DIM + d.HEAD_DIM := by omega
    _ = (128 * i + r + 1) * d.HEAD_DIM := by ring
    _ ≤ rows d * d.HEAD_DIM := Nat.mul_le_mul_right _ (by omega)

theorem mTile_sub {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) :
    mTile i ⊆ below (rows d) := by
  rintro _ ⟨r, hr, rfl⟩
  have h1 := tile_le hd hi
  show 128 * i + r < rows d
  omega

theorem Wr_sub {d : Inst} (hd : d.ok) {i : ℕ} (hi : i < (launch d).size) : Wr d i ⊆ outBytes d :=
  Set.union_subset_union (d.o.elems_mono (oTile_sub hd hi)) (d.M.elems_mono (mTile_sub hd hi))

theorem row_inj {D a b e e' : ℕ} (h : a * D + e = b * D + e') (he : e < D) (he' : e' < D) : a = b := by
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
  · have : (a + 1) * D ≤ b * D := Nat.mul_le_mul_right _ hlt
    nlinarith
  · have : (b + 1) * D ≤ a * D := Nat.mul_le_mul_right _ hlt
    nlinarith

theorem Wr_disj {d : Inst} (hd : d.ok) {i j : ℕ} (hi : i < (launch d).size) (hj : j < (launch d).size)
    (hij : i ≠ j) : Disjoint (Wr d i) (Wr d j) := by
  have hoo : Disjoint (d.o.elems (oTile d i)) (d.o.elems (oTile d j)) := by
    refine d.o.elems_disjoint (Set.disjoint_left.mpr ?_)
    rintro _ ⟨r, hr, e, he, rfl⟩ ⟨r', hr', e', he', h⟩
    have := row_inj h he he'
    omega
  have hmm : Disjoint (d.M.elems (mTile i)) (d.M.elems (mTile j)) := by
    refine d.M.elems_disjoint (Set.disjoint_left.mpr ?_)
    rintro _ ⟨r, hr, rfl⟩ ⟨r', hr', h⟩
    omega
  have hoM := hd.disj.2.2.2.1
  refine Disjoint.union_left (Disjoint.union_right hoo ?_) (Disjoint.union_right ?_ hmm)
  · exact hoM.mono (d.o.elems_mono (oTile_sub hd hi)) (d.M.elems_mono (mTile_sub hd hj))
  · exact hoM.symm.mono (d.M.elems_mono (mTile_sub hd hi)) (d.o.elems_mono (oTile_sub hd hj))

/-! ## What a program leaves -/

/-- What `Post` asks of element `(bh, n, e)` of `o` holding `c`. -/
def OutO (d : Inst) (init : Mem) (bh n e : ℕ) (c : F16) : Prop :=
  HeadOk d init bh → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
    (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) →
    |c.toReal - attnRef d init bh n e| ≤
      (tolO d.HEAD_DIM d.N_CTX X : ℝ) * attnAbs d init bh n e +
        (tolS d.HEAD_DIM d.N_CTX X : ℝ) * (vAbsSum d init bh n e : ℝ) + (tolF d.HEAD_DIM d.N_CTX X : ℝ)

/-- What `Post` asks of element `(bh, n)` of `M` holding `c`. -/
def OutM (d : Inst) (init : Mem) (bh n : ℕ) (c : F32) : Prop :=
  HeadOk d init bh → c.isFinite = true ∧ ∀ X : ℚ, X ≤ 2 ^ 10 →
    (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) →
    |c.toReal - lseRef d init bh n| ≤ (tolM d.HEAD_DIM d.N_CTX X : ℝ) + (u32 : ℝ) * |lseRef d init bh n|

/-- What program `(sm, bh)` leaves in memory `m`: its rows of `o` and `M` as `Post` asks. -/
def Qp (d : Inst) (init : Mem) (sm bh : ℕ) (m : Mem) : Prop :=
  ∀ r : Fin 128,
    (∀ e : Fin d.HEAD_DIM, ∃ c : F16,
      m.read .f16 (d.o.ptr ((idx d bh (128 * sm + r) e : ℕ) : ℤ)).addr = some c ∧
        OutO d init bh (128 * sm + r) e c) ∧
    ∃ c : F32, m.read .f32 (d.M.ptr ((bh * d.N_CTX + (128 * sm + r) : ℕ) : ℤ)).addr = some c ∧
      OutM d init bh (128 * sm + r) c

theorem keys_lt {d : Inst} {n j : ℕ} (hj : j ∈ keys d n) : j < d.N_CTX := by
  unfold keys at hj
  simp only [Finset.mem_filter, Finset.mem_range] at hj
  exact hj.1

/-- **The end of a row** (lines 241–245): from the row's state after `T` blocks that hold all its
keys, `acc / l_i` rounded to `float16` and `m_i + log2(l_i)` are as `Post` asks. -/
theorem row_out {d : Inst} (hd : d.ok) {init : Mem} {P : Platform} (hdivF : Rules.Num.DivFull P)
    (hlog : Rules.Num.Log2F32 P) {c : F32} (hc : CFacts d c) {bh n : ℕ} (hn : n < d.N_CTX)
    {T : ℕ} (hT0 : 0 < T) (hT : T ≤ nblocks d.N_CTX) (hTk : ∀ j, alOf d n j = true → j < 64 * T)
    {acc : Fin d.HEAD_DIM → F32} {l m : F32}
    (hst : HeadOk d init bh → ∀ X : ℚ, X ≤ 2 ^ 10 →
      (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) →
      RowSt (rowCtx d init c.toReal bh n X) (veOf d init bh) T m l acc)
    {lg Mv : F32} (hlg : P.math .log2 .binary32 l lg) (hMv : FloatV.NaNEq (FloatV.add m lg) Mv)
    {o' : Fin d.HEAD_DIM → F32} (ho' : ∀ e, P.div .binary32 (acc e) l (o' e))
    {o16 : Fin d.HEAD_DIM → F16} (ho16 : ∀ e, FloatV.NaNEq (FloatV.cvt (o' e)) (o16 e)) :
    (∀ e : Fin d.HEAD_DIM, OutO d init bh n e (o16 e)) ∧ OutM d init bh n Mv := by
  have hD := headDim_le hd
  have hN := hd.n_le
  have hne : (keys d n).Nonempty := ⟨0, mem_keys.mpr (by unfold alOf; simp; omega)⟩
  have hσ : (0 : ℝ) ≤ (sigma d : ℝ) := by exact_mod_cast (sigma_pos hd).le
  have hθ : ∀ X : ℚ, (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) → ∀ j ∈ keys d n,
      |Real.log 2 * ((rowCtx d init c.toReal bh n X).c * (rowCtx d init c.toReal bh n X).sv j) -
        (sigma d : ℝ) * (rowCtx d init c.toReal bh n X).sv j| ≤ X / 2 ^ 22 := by
    intro X hXk j hj
    show |Real.log 2 * (c.toReal * (score d init bh n j : ℝ)) - (sigma d : ℝ) * (score d init bh n j : ℝ)| ≤
      X / 2 ^ 22
    have e1 : Real.log 2 * (c.toReal * (score d init bh n j : ℝ)) - (sigma d : ℝ) * (score d init bh n j : ℝ) =
        (c.toReal * Real.log 2 - sigma d) * score d init bh n j := by ring
    have h3 : (sigma d : ℝ) * (scoreMag d init bh n j : ℝ) ≤ X := by exact_mod_cast hXk j hj
    rw [e1, abs_mul]
    calc |c.toReal * Real.log 2 - (sigma d : ℝ)| * |(score d init bh n j : ℝ)|
        ≤ (sigma d : ℝ) / 2 ^ 22 * (scoreMag d init bh n j : ℝ) :=
          mul_le_mul hc.2.2.2.2 (score_le d init bh n j) (abs_nonneg _) (by positivity)
      _ = (sigma d : ℝ) * (scoreMag d init bh n j : ℝ) / 2 ^ 22 := by ring
      _ ≤ X / 2 ^ 22 := div_le_div_of_nonneg_right h3 (by positivity)
  have hfin : HeadOk d init bh → ∀ X : ℚ, X ≤ 2 ^ 10 →
      (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) →
      (rowCtx d init c.toReal bh n X).Ok ∧ 0 ≤ X ∧ MInv (rowCtx d init c.toReal bh n X) T m ∧
        LInv (rowCtx d init c.toReal bh n X) T m l ∧
        ∀ e, AInv (rowCtx d init c.toReal bh n X) (veOf d init bh e) T m (acc e) := by
    intro hH X hX hXk
    obtain ⟨hRok, hX0⟩ := rowCtx_ok hd init hc.2.1 hc.2.2.1 hc.2.2.2.1 bh hn hX hXk
    rcases hst hH X hX hXk with ⟨h0, -⟩ | ⟨-, hM, hL, hA⟩
    · omega
    · exact ⟨hRok, hX0, hM, hL, hA⟩
  have hve : ∀ e : Fin d.HEAD_DIM, ∀ j ∈ keys d n,
      veOf d init bh e j = (val16 init d.v (idx d bh j e) : ℝ) := fun e j hj => by
    unfold veOf; rw [ite_eq_left (keys_lt hj)]
  have hOo : ∀ e : Fin d.HEAD_DIM, HeadOk d init bh → ∀ X : ℚ, X ≤ 2 ^ 10 →
      (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) →
      (o16 e).isFinite = true ∧ |(o16 e).toReal - attnRef d init bh n e| ≤
        (tolO d.HEAD_DIM d.N_CTX X : ℝ) * attnAbs d init bh n e +
          (tolS d.HEAD_DIM d.N_CTX X : ℝ) * (vAbsSum d init bh n e : ℝ) + (tolF d.HEAD_DIM d.N_CTX X : ℝ) := by
    intro e hH X hX hXk
    obtain ⟨hRok, hX0, hM, hL, hA⟩ := hfin hH X hX hXk
    obtain ⟨hf, hb⟩ := final_o (D := d.HEAD_DIM) (N := d.N_CTX) hdivF hRok hD hX0 hX hN rfl rfl hT
      (psum_keys hTk) hne (hθ X hXk) (veOf_le hH e) hM hL (hA e) (ho' e) (ho16 e)
    refine ⟨hf, ?_⟩
    have S1 : (∑ j ∈ keys d n, weight d init bh n j * veOf d init bh e j) =
        ∑ j ∈ keys d n, weight d init bh n j * (val16 init d.v (idx d bh j e) : ℝ) :=
      Finset.sum_congr rfl fun j hj => by rw [hve e j hj]
    have S2 : (∑ j ∈ keys d n, weight d init bh n j * |veOf d init bh e j|) =
        ∑ j ∈ keys d n, weight d init bh n j * |(val16 init d.v (idx d bh j e) : ℝ)| :=
      Finset.sum_congr rfl fun j hj => by rw [hve e j hj]
    have S3 : (∑ j ∈ keys d n, |veOf d init bh e j|) = (vAbsSum d init bh n e : ℝ) := by
      unfold vAbsSum; push_cast
      exact Finset.sum_congr rfl fun j hj => by rw [hve e j hj]
    unfold attnRef attnAbs
    rw [← S1, ← S2, ← S3]
    exact hb
  have hOM : HeadOk d init bh → ∀ X : ℚ, X ≤ 2 ^ 10 →
      (∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ X) →
      Mv.isFinite = true ∧ |Mv.toReal - lseRef d init bh n| ≤
        (tolM d.HEAD_DIM d.N_CTX X : ℝ) + (u32 : ℝ) * |lseRef d init bh n| := by
    intro hH X hX hXk
    obtain ⟨hRok, hX0, hM, hL, -⟩ := hfin hH X hX hXk
    exact final_M (D := d.HEAD_DIM) (N := d.N_CTX) hlog hRok hD hX0 hX hN rfl rfl hT (psum_keys hTk) hne
      (hθ X hXk) hM hL hlg hMv
  have hall : HeadOk d init bh → ∀ j ∈ keys d n, sigma d * scoreMag d init bh n j ≤ 2 ^ 10 :=
    fun hH j hj => hH.2.2 n j hn (keys_lt hj)
  refine ⟨fun e hH => ⟨(hOo e hH _ le_rfl (hall hH)).1, fun X hX hXk => (hOo e hH X hX hXk).2⟩,
    fun hH => ⟨(hOM hH _ le_rfl (hall hH)).1, fun X hX hXk => (hOM hH X hX hXk).2⟩⟩

/-! ## The end of a program (lines 241–245) -/

/-- **Lines 241–245** of program `i` (tile `(sm, bh)`), from every row's state after `T` blocks
that hold all its keys: the stores of `M` and `o` leave `Qp`. -/
theorem safe_epi {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hdivF : Rules.Num.DivFull P) (hdivT : Rules.Num.DivTotal P .binary32) (hlog : Rules.Num.Log2F32 P)
    (hlt : Rules.Num.MathTotal P .log2 .binary32) {i sm bh : ℕ} (hi : i < (launch d).size)
    (hsm : 128 * sm + 128 ≤ d.N_CTX) (hrow : bh * d.N_CTX + 128 * sm = 128 * i)
    {c : F32} (hc : CFacts d c) {T : ℕ} (hT0 : 0 < T) (hT : T ≤ nblocks d.N_CTX)
    (hTk : ∀ (r : Fin 128) (j : ℕ), alOf d (128 * sm + r) j = true → j < 64 * T)
    {acc : Tensor [128, d.HEAD_DIM] F32} {l m : Tensor [128] F32} (hinv : StInv d init c sm bh T acc l m)
    {mp : Tensor [128] (Ptr .f32)}
    (hmp : ∀ r : Fin 128, mp.get (r, ()) = d.M.ptr ((bh * d.N_CTX + (128 * sm + r) : ℕ) : ℤ))
    {qo : I32} (hqo : qo.toInt = ((128 * i : ℕ) : ℤ)) :
    Safe ((launch d).env P) (Rd d) (Wr d i) (fun _ m' => Qp d init sm bh m') init (do
      let lg ← tl.log2 l
      let mv ← tl.add m lg
      let o' ← tl.div acc (l.col d.HEAD_DIM)
      tl.store mp mv
      let o16 : Tensor [128, d.HEAD_DIM] F16 ← tl.cvt o'
      tl.desc_store (descOf d d.o 128) [qo, IntV.ofInt 0] o16) := by
  have hO := tile_le hd hi
  have hMl := hd.lens.2.2.2.2
  have hoM := hd.disj.2.2.2.1
  -- line 241
  refine safe_math (fun _ => hlt _) fun lg hlg => ?_
  refine safe_add fun mv hmv => ?_
  -- line 242
  refine safe_div (fun _ => hdivT _ _) fun o' ho' => ?_
  -- line 244
  have himg : Buf.img (fun ri : Idx [128] => bh * d.N_CTX + (128 * sm + ri.1)) (fun _ => true) ⊆ mTile i := by
    rintro _ ⟨⟨r, ⟨⟩⟩, -, rfl⟩
    exact ⟨r, r.2, by simp only; omega⟩
  have hMsub : d.M.elems (Buf.img (fun ri : Idx [128] => bh * d.N_CTX + (128 * sm + ri.1)) (fun _ => true)) ⊆
      d.M.elems (below (rows d)) := d.M.elems_mono (himg.trans (mTile_sub hd hi))
  refine safe_store_buf d.M (g := fun ri : Idx [128] => bh * d.N_CTX + (128 * sm + ri.1)) (en := fun _ => true)
    (Buf.addrs_of_get_all d.M fun ⟨r, ⟨⟩⟩ => hmp r) ?_ ?_ hpre.2.2.2.2
    (fun x hx => Or.inr (d.M.elems_mono himg hx)) fun m1 hm1 hfr1 hpr1 => ?_
  · rintro ⟨r, ⟨⟩⟩ -
    show bh * d.N_CTX + (128 * sm + r) < d.M.len
    have := r.2; omega
  · rintro ⟨a, ⟨⟩⟩ ⟨b, ⟨⟩⟩ - - h
    have : a = b := Fin.ext (by simp only at h; omega)
    rw [this]
  -- line 245
  refine safe_cvt f32_f16_ok fun o16 ho16 => ?_
  rw [← bind_pure (tl.desc_store (descOf d d.o 128) [qo, IntV.ofInt 0] o16)]
  have hov : d.o.Valid m1 := Buf.valid_of_agree hpre.2.2.2.1 fun x hx => hpr1 x (hpre.2.2.2.1.2.2.2 x hx)
  refine safe_store_rows hd d.o hd.align.2.2.2 hd.lens.2.2.2.1 hov hqo hO ?_ fun m2 hm2 hfr2 _ => ?_
  · intro x hx
    refine ⟨Or.inl hx, fun hxM => ?_⟩
    exact Set.disjoint_left.mp hoM (d.o.elems_mono (oTile_sub hd hi) hx) (hMsub hxM)
  refine safe_ret fun r => ?_
  have hn : 128 * sm + r < d.N_CTX := by have := r.2; omega
  obtain ⟨hOo, hOM⟩ := row_out hd hdivF hlog hc hn hT0 hT (hTk r) (hinv r) (hlg (r, ())) (hmv (r, ()))
    (o' := fun e => o'.get2 r e) (fun e => ho' (r, e, ())) (o16 := fun e => o16.get2 r e)
    (fun e => ho16 (r, e, ()))
  refine ⟨fun e => ⟨o16.get2 r e, ?_, hOo e⟩, mv.get (r, ()), ?_, hOM⟩
  · have hidx : idx d bh (128 * sm + r) e = (128 * i + r) * d.HEAD_DIM + e := by
      unfold idx; rw [← Nat.add_assoc, hrow]
    rw [hidx]; exact hm2 r e
  · refine (Mem.read_congr fun x hx => hfr2 x fun hxo => ?_).trans (hm1 (r, ()) rfl)
    have hmem : bh * d.N_CTX + (128 * sm + r) ∈ below (rows d) := by
      show _ < rows d; have := r.2; omega
    exact Set.disjoint_left.mp hoM (d.o.elems_mono (oTile_sub hd hi) hxo) (d.M.span_sub_elems hmem hx)

/-! ## A program -/

theorem offs_m_val {sm : ℕ} (hsm : 128 * sm + 128 < 2 ^ 31) (r : Fin 128) :
    (((IntV.ofNat sm : I32) * (IntV.ofNat 128 : I32) + tl.arange 0 128).get1 r).toInt =
      ((128 * sm + r : ℕ) : ℤ) := by
  show ((IntV.ofNat sm : I32) * IntV.ofNat 128 + IntV.ofInt (((0 : ℕ) : ℤ) + ((r : ℕ) : ℤ))).toInt = _
  have h1 : (IntV.ofInt (((0 : ℕ) : ℤ) + ((r : ℕ) : ℤ)) : I32).toInt = ((r : ℕ) : ℤ) := by
    rw [IntV.I32.toInt_ofInt (by have := r.2; constructor <;> push_cast <;> omega)]; push_cast; ring
  rw [IntV.I32.add_nat (IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)) h1
    (by have := r.2; omega)]
  congr 1; ring

theorem offs_n_val (jj : Fin 64) : ((tl.arange 0 64).get1 jj).toInt = ((jj : ℕ) : ℤ) := by
  show (IntV.ofInt (((0 : ℕ) : ℤ) + ((jj : ℕ) : ℤ)) : I32).toInt = _
  rw [IntV.I32.toInt_ofInt (by have := jj.2; constructor <;> push_cast <;> omega)]; push_cast; ring

/-- **One program** (lines 189–245): program `i` reads only `q`, `k`, `v`, writes only its tile of
`o` and `M`, and leaves them as `Post` asks (`Qp`). -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hdot : Rules.Num.DotF16 P) (hdtot : Rules.Num.DotTotal P cfg16) (hexp : Rules.Num.Exp2F32 P)
    (hni : Rules.Num.Exp2F32NegInf P) (hext : Rules.Num.MathTotal P .exp2 .binary32)
    (hlog : Rules.Num.Log2F32 P) (hlt : Rules.Num.MathTotal P .log2 .binary32)
    (hdivF : Rules.Num.DivFull P) (hdivT : Rules.Num.DivTotal P .binary32)
    {i : ℕ} (hi : i < (launch d).size) :
    Safe ((launch d).env P) (Rd d) (Wr d i) (fun _ m => Qp d init (pm d i) (pbh d i) m) init
      ((model d).body ((launch d).ctx i)) := by
  have hsm := pm_row hd i
  have hbh := pbh_lt hd hi
  have hrow := row0 hd i
  simp only [model, attn_fwd]
  generalize hS : (if d.causal = true then 3 else 1) = STAGE
  simp only [tl.program_id, pid0, pid1 hd hi]
  generalize pm d i = sm at hsm hrow ⊢
  generalize pbh d i = bh at hbh hrow ⊢
  have hrows := hd.rows_lt
  have hzh := hd.zh_lt
  have hN := hd.n_le
  have hNm := hd.n_mul
  have hZH := hd.zh_pos
  have hO := tile_le hd hi
  have hH1 : d.H ≤ d.Z * d.H := Nat.le_mul_of_pos_left _ hZH.1
  have hbhN : bh * d.N_CTX + d.N_CTX ≤ rows d := by
    unfold rows
    calc bh * d.N_CTX + d.N_CTX = (bh + 1) * d.N_CTX := by ring
      _ ≤ (d.Z * d.H) * d.N_CTX := Nat.mul_le_mul_right _ hbh
  have hNH : d.N_CTX * d.H ≤ rows d := by
    unfold rows
    calc d.N_CTX * d.H = 1 * d.H * d.N_CTX := by ring
      _ ≤ d.Z * d.H * d.N_CTX := Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hZH.1)
  -- lines 191–192
  refine safe_idiv1 (by rw [i32_nat (by omega)]; positivity) (by rw [i32_nat (by omega)]; exact_mod_cast hZH.2)
    fun z hz => ?_
  refine safe_irem1 (by rw [i32_nat (by omega)]; positivity) (by rw [i32_nat (by omega)]; exact_mod_cast hZH.2)
    fun h hh => ?_
  rw [i32_nat (by omega), i32_nat (by omega)] at hz hh
  -- lines 195–206
  refine safe_maybe_desc hd d.q 128 ?_
  refine safe_maybe_desc hd d.v 64 ?_
  refine safe_maybe_desc hd d.k 64 ?_
  refine safe_maybe_desc hd d.o 128 ?_
  -- lines 214–215
  refine safe_sub fun mi hmi => ?_
  refine safe_add fun li hli => ?_
  -- line 219
  refine safe_mul fun cs hcs => ?_
  have hc : CFacts d cs.item := qk_scale_facts hd (hcs ())
  -- lines 208–209
  have hdm := Nat.div_add_mod bh d.H
  have e1 : bh / d.H * (d.N_CTX * d.H) + bh % d.H * d.N_CTX = bh * d.N_CTX := by
    calc bh / d.H * (d.N_CTX * d.H) + bh % d.H * d.N_CTX = (d.H * (bh / d.H) + bh % d.H) * d.N_CTX := by ring
      _ = bh * d.N_CTX := by rw [hdm]
  have hzv : z.item.toInt = ((bh / d.H : ℕ) : ℤ) := by rw [hz]; push_cast; rfl
  have hhv : h.item.toInt = ((bh % d.H : ℕ) : ℤ) := by rw [hh]; push_cast; rfl
  have hNHv : (IntV.ofNat d.N_CTX * IntV.ofNat d.H : I32).toInt = ((d.N_CTX * d.H : ℕ) : ℤ) :=
    IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega)) (by omega)
  have hle1 : bh / d.H * (d.N_CTX * d.H) ≤ bh * d.N_CTX := e1 ▸ Nat.le_add_right _ _
  have hle2 : bh % d.H * d.N_CTX ≤ bh * d.N_CTX := e1 ▸ Nat.le_add_left _ _
  have hoy : (z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) + h.item * IntV.ofNat d.N_CTX : I32).toInt =
      ((bh * d.N_CTX : ℕ) : ℤ) := by
    rw [IntV.I32.add_nat (IntV.I32.mul_nat hzv hNHv (by omega)) (IntV.I32.mul_nat hhv (i32_nat (by omega)) (by omega))
      (by omega), e1]
  have hqo : (z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) + h.item * IntV.ofNat d.N_CTX +
      IntV.ofNat sm * IntV.ofNat 128 : I32).toInt = ((128 * i : ℕ) : ℤ) := by
    rw [IntV.I32.add_nat hoy (IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)) (by omega)]
    congr 1; omega
  -- line 221
  refine safe_load_rows hd d.q hd.align.1 hd.lens.1 hpre.1 (fun x hx => Or.inl (Or.inl hx)) hqo hO
    fun q hq => ?_
  have hQ : QReads d init sm bh q := fun r e => by
    have hidx : idx d bh (128 * sm + r) e = (128 * i + r) * d.HEAD_DIM + e := by
      unfold idx; rw [← Nat.add_assoc, hrow]
    rw [hidx]; exact hq r e
  have hcall : CallOk d init sm bh cs.item q (z.item * (IntV.ofNat d.N_CTX * IntV.ofNat d.H) +
      h.item * IntV.ofNat d.N_CTX) (IntV.ofNat sm) (IntV.ofNat d.N_CTX)
      ((IntV.ofNat sm : I32) * IntV.ofNat 128 + tl.arange 0 128) (tl.arange 0 64) :=
    { sm_le := hsm, bh_lt := hbh, c_ok := hc, q_ok := hQ, oy := hoy, start := i32_nat (by omega),
      n := i32_nat (by omega), om := offs_m_val (by omega), on := offs_n_val }
  have hinv0 : StInv d init cs.item sm bh 0 (tl.zeros [128, d.HEAD_DIM] .f32) li mi := by
    intro r _ X _ _
    obtain ⟨hl1, hl2⟩ := l_init_eq (hli (r, ()))
    have hl2' : (li.get1 r).toReal = 1 := hl2
    exact Or.inl ⟨rfl, m_init_eq (hmi (r, ())), hl1, by rw [hl2']; norm_num,
      fun e => ⟨zero32.1, by rw [toReal_def]; exact (congrArg (fun x : ℚ => (x : ℝ)) zero32.2).trans Rat.cast_zero⟩⟩
  have hmp : ∀ r : Fin 128, (d.M.ptr 0 + (IntV.ofNat bh : I32) * IntV.ofNat d.N_CTX +
      ((IntV.ofNat sm : I32) * IntV.ofNat 128 + tl.arange 0 128)).get (r, ()) =
      d.M.ptr ((bh * d.N_CTX + (128 * sm + r) : ℕ) : ℤ) := fun r => by
    have h1 := offs_m_val (sm := sm) (by omega) r
    have h2 : ((IntV.ofNat bh : I32) * IntV.ofNat d.N_CTX).toInt = ((bh * d.N_CTX : ℕ) : ℤ) :=
      IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by omega)) (by omega)
    show ((d.M.ptr 0).offset ((IntV.ofNat bh : I32) * IntV.ofNat d.N_CTX).toInt).offset
      (((IntV.ofNat sm : I32) * IntV.ofNat 128 + tl.arange 0 128).get1 r).toInt = _
    rw [h1, h2, Buf.ptr_offset, Buf.ptr_offset]
    congr 1; push_cast; ring
  -- lines 225–239
  by_cases hcz : d.causal = true
  · rw [ite_eq_left hcz] at hS
    subst hS
    rw [ite_eq_left (show (3 : ℕ) &&& 1 ≠ 0 by decide)]
    refine safe_inner1 hd hpre hdot hdtot hexp hni hext subset_rfl hcall hinv0 fun acc l m hst => ?_
    rw [ite_eq_left (show (3 : ℕ) &&& 2 ≠ 0 by decide)]
    refine safe_inner2 hd hcz hpre hdot hdtot hexp hni hext subset_rfl hcall hst fun acc l m hst => ?_
    refine safe_epi hd hpre hdivF hdivT hlog hlt hi hsm hrow hc (by omega) (by unfold nblocks; omega)
      (fun r j hj => ?_) hst hmp hqo
    unfold alOf at hj
    simp only [hcz, Bool.true_eq_false, false_or, decide_eq_true_eq] at hj
    have := r.2; omega
  · rw [ite_eq_right hcz] at hS
    subst hS
    rw [ite_eq_left (show (1 : ℕ) &&& 1 ≠ 0 by decide)]
    have hcf : d.causal = false := by simpa using hcz
    refine safe_inner3 hd hcf hpre hdot hdtot hexp hni hext subset_rfl hcall hinv0 fun acc l m hst => ?_
    rw [ite_eq_right (show ¬((1 : ℕ) &&& 2 ≠ 0) by decide)]
    simp only [pure_bind]
    refine safe_epi hd hpre hdivF hdivT hlog hlt hi hsm hrow hc (by omega) (by unfold nblocks; omega)
      (fun r j hj => ?_) hst hmp hqo
    unfold alOf at hj
    simp only [decide_eq_true_eq] at hj
    omega

end Kernels.AttnFwd
