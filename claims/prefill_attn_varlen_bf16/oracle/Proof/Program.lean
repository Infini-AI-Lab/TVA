import Proof.Setup
import Proof.Floats
import Proof.NumEnd
import Proof.NumStep
import Proof.Access
import Proof.LoopStep

section
/-!
# The epilogue: the output tile against the exact attention

`row_post`: the accumulator of a valid row after its last trip, converted to `bfloat16`, is within
the claim's tolerance of the exact attention (`Num.final_bound` and `cvt_store`). `post_safe_proof`
walks `postProg`: the pointers, the conversion and the masked store of the tile.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-! ## The facts about a valid row -/

/-- A sequence of the batch is at most `max_input_len ≤ 2^17` long. -/
private theorem len_le {d : Inst} (hd : d.ok) {init : Mem} (hs : SeqOk d init) {b : Nat}
    (hb : b < d.batch) : lenOf d init b ≤ 2 ^ 17 := by
  obtain ⟨s, n, h1, h2, h3, h4, h5, h6⟩ := hs.1 b hb
  have := hd.maxLen_le
  simp only [lenOf, val32, h2]
  omega

/-- The exact score of a pair is at most its magnitude `scoreMag`. -/
private theorem abs_score_le {d : Inst} (hd : d.ok) (m : Mem) (b h p j : Nat) :
    |score d m b h p j| ≤ scoreMag d m b h p j := by
  have hsig : 0 ≤ sig d := by
    have := hd.scale.2.1
    have h0 : (0 : ℝ) ≤ (d.sm_scale.toRat : ℝ) := by
      exact_mod_cast (le_trans (by positivity) this : (0 : ℚ) ≤ d.sm_scale.toRat)
    exact h0
  unfold score scoreMag
  rw [abs_mul, abs_of_nonneg hsig]
  refine mul_le_mul_of_nonneg_left ?_ hsig
  refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun e _ => ?_)
  rw [← Rat.cast_mul, ← Rat.cast_abs]
  exact_mod_cast Device.abs_mul_le_liftMag .bf16 .bf16 (qv d m b h p e) (kv d m b h j e)

/-- The score bound `X` of a valid row: nonnegative, at most `2^12`, above every attended score. -/
private theorem Xrow_facts {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b h p : Nat}
    (hb : b < d.batch) (hh : h < d.H) (hp : p < lenOf d init b) :
    lastKey d init b p < lenOf d init b ∧ 0 ≤ Xrow d init b h p ∧ Xrow d init b h p ≤ 4096 ∧
      ∀ j, j ≤ lastKey d init b p → |score d init b h p j| ≤ Xrow d init b h p := by
  have hL : lastKey d init b p < lenOf d init b := by
    unfold lastKey; split_ifs <;> omega
  have hj : ∀ j, j ≤ lastKey d init b p → |score d init b h p j| ≤ Xrow d init b h p := fun j hj =>
    (abs_score_le hd init b h p j).trans
      (Finset.le_sup' (fun j => scoreMag d init b h p j) (Finset.mem_range.2 (by omega)))
  refine ⟨hL, (abs_nonneg _).trans (hj 0 (Nat.zero_le _)), ?_, hj⟩
  have hX : Xrow d init b h p ≤ 2 ^ 12 := by
    unfold Xrow
    refine Finset.sup'_le _ _ fun j hjm => ?_
    exact (hpre.2.2.2.2.2.2.2 b hb).2.2 h p j hh hp (by have := Finset.mem_range.1 hjm; omega)
  linarith

/-- **The output of a valid row.** After the row's last trip (`trips (lastKey ..)` of them) the
accumulator lane `a e`, converted to `bfloat16` (`c`), is finite and within the claim's tolerance
of the exact attention. -/
theorem row_post {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b h p : Nat}
    (hb : b < d.batch) (hh : h < d.H) (hp : p < lenOf d init b) {m l : F32}
    {a : Fin d.BD → F32}
    (hS : RowState (rowSpec d init b h p) (scoreErr d.BD (Xrow d init b h p))
      (rowY d.BD (Xrow d init b h p)) (rowV d init b h p) (trips (lastKey d init b p)) m l a)
    (e : Fin d.BD) (he : e.val < d.Lk) {c : BF16}
    (hc : FloatV.NaNEq (FloatV.cvtWith .rn (a e)) c) :
    c.isFinite = true ∧
      |c.toReal - attnRef d init b h p e.val| ≤
        tolRel d.BD (trips (lastKey d init b p)) (Xrow d init b h p) * attnAbs d init b h p e.val +
          tolAbs (vSum d init b h p e.val) := by
  obtain ⟨hL, hX0, hX4, hsc⟩ := Xrow_facts hd hpre hb hh hp
  have hlen := len_le hd hpre.2.2.2.2.2.2.1 hb
  have hT : trips (lastKey d init b p) = lastKey d init b p / 128 + 1 := rfl
  have hT1 : trips (lastKey d init b p) ≤ 1024 := by omega
  have hD1 : 1 ≤ d.BD := le_trans hd.lk_pos hd.lk_le
  have hD2 := hd.bd_le
  have hvv : ∀ j, j ≤ lastKey d init b p → |((vv d init b h j e.val : ℚ) : ℝ)| ≤ 2 ^ 120 := by
    intro j hj
    have := ((hpre.2.2.2.2.2.2.2 b hb).2.1 j (kvh d h) e.val (by omega) (kvh_lt hd hh) he).2.2
    exact_mod_cast this
  have hV0 : 0 ≤ vSum d init b h p e.val := Finset.sum_nonneg fun j _ => abs_nonneg _
  have hVu : vSum d init b h p e.val ≤ 2 ^ 138 := by
    refine (Finset.sum_le_card_nsmul _ _ ((2 : ℝ) ^ 120) fun j hj => hvv j
      (by have := Finset.mem_range.1 hj; omega)).trans ?_
    rw [Finset.card_range, nsmul_eq_mul]
    have : ((lastKey d init b p + 1 : ℕ) : ℝ) ≤ 2 ^ 17 := by exact_mod_cast (by omega)
    calc ((lastKey d init b p + 1 : ℕ) : ℝ) * 2 ^ 120 ≤ 2 ^ 17 * 2 ^ 120 := by gcongr
      _ ≤ 2 ^ 138 := by norm_num
  rcases hS with ⟨h0, -⟩ | ⟨ht0, sh, inv, -, -, hall⟩
  · omega
  obtain ⟨hfin, hclose⟩ := hall e
  have hvveq : (rowSpec d init b h p).vv e = fun j => ((vv d init b h j e.val : ℚ) : ℝ) := by
    funext j; simp [rowSpec, he]
  have hVe : rowV d init b h p e = vSum d init b h p e.val := by simp [rowV, he]
  rw [hvveq, hVe] at hclose
  have hδ0 : 0 ≤ scoreErr d.BD (Xrow d init b h p) := by unfold scoreErr; positivity
  have hδ1 : scoreErr d.BD (Xrow d init b h p) ≤ 1 := by
    have hD : (d.BD : ℝ) ≤ 256 := by exact_mod_cast hD2
    have h1 : ((d.BD : ℝ) + 8) * Xrow d init b h p ≤ 264 * 4096 := by
      nlinarith [mul_nonneg (show (0 : ℝ) ≤ 256 - d.BD by linarith) hX0]
    unfold scoreErr
    have c22 : (2 : ℝ) ^ (-22 : ℤ) = 1 / 4194304 := by norm_num
    have c40 : (2 : ℝ) ^ (-40 : ℤ) ≤ 1e-11 := by norm_num
    rw [mul_assoc, c22]
    nlinarith
  have hY : ∀ j, j ≤ lastKey d init b p →
      |(rowSpec d init b h p).sc j| + scoreErr d.BD (Xrow d init b h p) ≤
        rowY d.BD (Xrow d init b h p) := fun j hj => by
    have := hsc j hj
    simp only [rowSpec, rowY]
    linarith
  have hY5 : rowY d.BD (Xrow d init b h p) ≤ 5000 := by unfold rowY; linarith
  have hacc := Num.acc_le hδ0 hY hY5 ht0 hT1 inv (fun j hj => hvv j hj) hV0 le_rfl hVu hclose
  obtain ⟨hcf, hce⟩ := cvt_store hfin hacc.1 hc
  refine ⟨hcf, ?_⟩
  have := Num.final_bound hT hT1 hD1 hD2 hX0 hX4 hsc rfl rfl inv hvv rfl hclose hce
  exact this

/-! ## The epilogue -/

/-- The loop's trips of a program are the trips of each of its valid rows' last key. -/
private theorem nTrips_eq {d : Inst} {init : Mem} {i r : Nat} (hv : validRow d init i r)
    (hr : r < BLOCK) : nTrips d init i = trips (lastKey d init (bOf d i) (mOf d i * BLOCK + r)) := by
  unfold validRow at hv
  unfold nTrips trips lastKey
  simp only [BLOCK] at hv hr ⊢
  have h1 : mOf d i * 128 < lenOf d init (bOf d i) := by omega
  by_cases hc : d.causal = true <;> simp [h1, hc] <;> omega

/-- **The epilogue** after the last trip stores the output tile and ends with `Q`. -/
theorem post_safe_proof {P : Platform} (_hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) {cb ch sm kvh len st : I32}
    (hs : Scal d init i cb ch sm kvh len st)
    (st3 : Tensor [BLOCK] F32 × Tensor [BLOCK] F32 × Tensor [BLOCK, d.BD] F32) (m : Mem)
    (hI : LoopInv d init i (nTrips d init i) st3 m) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m' => Q d init i m') m
      (postProgD d cb ch sm kvh len st st3.2.2) := by
  obtain ⟨hmi, hrows⟩ := hI
  rw [hmi]
  have hsize : (i : Nat) < d.batch * d.H * nBlocks d := i.isLt
  have hbatch : 0 < d.batch := by
    rcases Nat.eq_zero_or_pos d.batch with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hH : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hb : bOf d i < d.batch := Nat.mod_lt _ hbatch
  have hh : hOf d i < d.H := Nat.mod_lt _ hH
  have hmN := mOf_lt d i
  have hSL := seq_le hpre hb
  unfold postProgD postProg
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun o16 hcv => ?_
  show Safe _ _ _ _ _ (tl.store (ptrsO d ch sm st) o16 (some (maskQ d len sm)) >>=
    fun _ => Prog.ret ())
  refine safe_store_tileAt d.o (ρ := rowBlk (mOf d i)) (κ := lane d.BD)
    (e := startOf d init (bOf d i) * d.sot + hOf d i * d.soh) (s0 := d.sot) (s1 := 1)
    (M := lenOf d init (bOf d i)) (N := d.Lk) (k := fun _ => Prog.ret ())
    (o_ptrs hd hs.st hs.ch hs.sm hSL hh hmN) (o_mask hd hs.len hs.sm hmN) (o_len hd hSL hh)
    (o_inj hd hh) o_rows_inj o_cols_inj hpre.2.2.2.1 (o_Wf d init i) fun m' hrd hout hpres => ?_
  refine safe_ret ?_
  intro r e hr he hv
  have heB : e < d.BD := lt_of_lt_of_le he hd.lk_le
  have hrow := hrows ⟨r, hr⟩ hv
  rw [nTrips_eq hv hr] at hrow
  obtain ⟨hfin, hbound⟩ := row_post hd hpre hb hh hv hrow ⟨e, heB⟩ he (hcv (⟨r, hr⟩, ⟨e, heB⟩, ()))
  refine ⟨o16.get2 ⟨r, hr⟩ ⟨e, heB⟩, ?_, hfin, hbound⟩
  have hidx : oIdx d (startOf d init (bOf d i) + (mOf d i * BLOCK + r)) (hOf d i) e =
      startOf d init (bOf d i) * d.sot + hOf d i * d.soh + rowBlk (mOf d i) ⟨r, hr⟩ * d.sot +
        lane d.BD ⟨e, heB⟩ * 1 := by
    simp only [oIdx, rowBlk, lane]; ring
  rw [rdB, hidx]
  exact hrd ⟨r, hr⟩ ⟨e, heB⟩ hv he

end Claims.PrefillAttnVarlen

end

section
/-!
# The body after the scalar steps

`afterQ_safe` walks everything after the load of `q`: the initial state, the loop (by the
invariant `LoopInv`, one trip per `trip_safe`) and the epilogue (`post_safe`). `rest_safe` adds the
load of `q`.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-- **The loop bound.** For a block `SM` of a sequence of length `LEN`, the bound
`block_mask * end_n` has the value `0` when the block starts at or beyond `LEN`, and `end_n`
(`LEN`, or `min((SM + 1) * 128, LEN)` when causal) otherwise. -/
theorem hi_facts {sm len : I32} {SM LEN : Nat} (hsm : sm.toInt = SM) (hlen : len.toInt = LEN)
    (hLEN : LEN < 2 ^ 31) (hSM : SM < 2 ^ 11) (causal : Bool) :
    ((tl.where ((Tensor.scalar (IntV.ofNat BLOCK * sm)).lt (Tensor.scalar len))
        (bc (IntV.ofInt 1 : I32)) (bc (IntV.ofInt 0 : I32)) : Tensor [] I32) *
      Tensor.scalar (if (!causal) = true then len
        else (tl.iminimum (Tensor.scalar ((sm + IntV.ofInt 1) * IntV.ofNat BLOCK))
          (Tensor.scalar len)).item)).item.toInt =
    if SM * BLOCK < LEN then (if causal then min ((SM + 1) * BLOCK) LEN else LEN : Nat) else 0 := by
  have hB : (IntV.ofNat BLOCK : I32).toInt = BLOCK := IntV.I32.toInt_ofNat (by decide)
  have hs128 : (IntV.ofNat BLOCK * sm).toInt = ((BLOCK * SM : Nat) : Int) :=
    IntV.I32.mul_nat hB hsm (by simp [BLOCK]; omega)
  have h1 : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)
  have hsm1 : (sm + IntV.ofInt 1).toInt = ((SM + 1 : Nat) : Int) :=
    IntV.I32.add_nat hsm (a := sm) (b := IntV.ofInt 1) (y := 1) (by simpa using h1) (by omega)
  have hs2 : ((sm + IntV.ofInt 1) * IntV.ofNat BLOCK).toInt = (((SM + 1) * BLOCK : Nat) : Int) :=
    IntV.I32.mul_nat hsm1 hB (by simp [BLOCK]; omega)
  have hmin : (IntV.min ((sm + IntV.ofInt 1) * IntV.ofNat BLOCK) len).toInt =
      (min ((SM + 1) * BLOCK) LEN : Nat) := by
    rw [IntV.toInt_min, hs2, hlen]; push_cast; rfl
  have hmul : ∀ x y : Tensor [] I32, (x * y).item = x.item * y.item := fun _ _ => rfl
  have hw : (tl.where ((Tensor.scalar (IntV.ofNat BLOCK * sm)).lt (Tensor.scalar len))
        (bc (IntV.ofInt 1 : I32)) (bc (IntV.ofInt 0 : I32)) : Tensor [] I32).item =
      if (IntV.ofNat BLOCK * sm).lt len then IntV.ofInt 1 else IntV.ofInt 0 := rfl
  have he : (Tensor.scalar (if (!causal) = true then len
        else (tl.iminimum (Tensor.scalar ((sm + IntV.ofInt 1) * IntV.ofNat BLOCK))
          (Tensor.scalar len)).item) : Tensor [] I32).item =
      if (!causal) = true then len else IntV.min ((sm + IntV.ofInt 1) * IntV.ofNat BLOCK) len := rfl
  have h0 : (IntV.ofInt 0 : I32).toInt = ((0 : Nat) : Int) := by
    rw [IntV.toInt_ofInt_zero]; rfl
  have h1' : (IntV.ofInt 1 : I32).toInt = ((1 : Nat) : Int) := by simpa using h1
  have hE : (if (!causal) = true then len
        else IntV.min ((sm + IntV.ofInt 1) * IntV.ofNat BLOCK) len).toInt
      = ((if causal then min ((SM + 1) * BLOCK) LEN else LEN : Nat) : Int) := by
    cases causal
    · simpa using hlen
    · simpa using hmin
  have hEl : (if causal then min ((SM + 1) * BLOCK) LEN else LEN : Nat) ≤ LEN := by
    cases causal <;> simp
  rw [hmul, hw, he, IntV.lt_eq, hs128, hlen]
  by_cases hlt : SM * BLOCK < LEN
  · have hlt' : ((BLOCK * SM : Nat) : Int) < (LEN : Int) := by
      have : BLOCK * SM < LEN := by rw [Nat.mul_comm]; exact hlt
      exact_mod_cast this
    simp only [hlt, hlt', decide_true, ite_true]
    have := IntV.I32.mul_nat h1' hE (by omega)
    simpa using this
  · have hlt' : ¬ ((BLOCK * SM : Nat) : Int) < (LEN : Int) := by
      have : ¬ BLOCK * SM < LEN := by rw [Nat.mul_comm]; exact hlt
      exact_mod_cast this
    simp only [hlt, hlt', decide_false, Bool.false_eq_true, ite_false]
    have := IntV.I32.mul_nat h0 hE (by omega)
    simpa using this

/-- The sequence's length is at most `max_input_len`. -/
theorem lenOf_le {d : Inst} {init : Mem} (hpre : Pre d init) {b : Nat} (hb : b < d.batch) :
    lenOf d init b ≤ d.maxLen := by
  obtain ⟨-, -, -, -, -, -, ⟨hseq, -⟩, -⟩ := id hpre
  obtain ⟨s, n, -, hn, -, hn0, hnm, -⟩ := hseq b hb
  have : lenOf d init b = n.toInt.toNat := by unfold lenOf val32; rw [hn]
  omega

/-- The loop bound `block_mask * end_n` of a program, and its trip count `nTrips`. -/
theorem loop_bound {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) {cb ch sm kvh len st : I32}
    (hs : Scal d init i cb ch sm kvh len st) :
    let hi := ((tl.where ((Tensor.scalar (IntV.ofNat BLOCK * sm)).lt (Tensor.scalar len))
        (bc (IntV.ofInt 1 : I32)) (bc (IntV.ofInt 0 : I32)) : Tensor [] I32) *
      Tensor.scalar (if (!d.causal) = true then len
        else (tl.iminimum (Tensor.scalar ((sm + IntV.ofInt 1) * IntV.ofNat BLOCK))
          (Tensor.scalar len)).item)).item
    tl.RangeOk (IntV.ofInt 0 : I32) hi (IntV.ofNat BLOCK : I32) ∧
      tl.tripCount (IntV.ofInt 0 : I32).toInt hi.toInt (IntV.ofNat BLOCK : I32).toInt =
        nTrips d init i := by
  intro hi
  have hsize : (i : Nat) < d.batch * d.H * nBlocks d := i.isLt
  have hbatch : 0 < d.batch := by
    rcases Nat.eq_zero_or_pos d.batch with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hH : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hb : bOf d i < d.batch := Nat.mod_lt _ hbatch
  have hm : mOf d i < nBlocks d := by
    unfold mOf
    rw [Nat.div_lt_iff_lt_mul (Nat.mul_pos hbatch hH)]
    rw [Nat.mul_comm (nBlocks d)]; exact hsize
  have hnB := nBlocks_le hd
  have hlm := lenOf_le hpre hb
  have hmax := hd.maxLen_le
  have hf := hi_facts (SM := mOf d i) (LEN := lenOf d init (bOf d i)) (by rw [hs.sm])
    (by rw [hs.len]) (by omega) (by omega) d.causal
  have hB : (IntV.ofNat BLOCK : I32).toInt = BLOCK := IntV.I32.toInt_ofNat (by decide)
  have h0 : (IntV.ofInt 0 : I32).toInt = 0 := IntV.toInt_ofInt_zero
  have hbound : hi.toInt ≤ d.maxLen := by
    rw [hf]; split_ifs <;> simp <;> omega
  have hmI : IntV.maxInt true 32 = 2 ^ 31 - 1 := by simp [IntV.maxInt]
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
  · rw [hB]; simp [BLOCK]
  · rw [hB, hmI]; simp [BLOCK]; omega
  · rw [h0, hmI]; norm_num
  · rw [hB, hmI]; simp [BLOCK]
  · rw [h0, hB, hf]
    unfold nTrips tl.tripCount
    simp only [BLOCK]
    by_cases h1 : mOf d i * 128 < lenOf d init (bOf d i)
    · simp only [h1, ↓reduceIte]
      split_ifs <;> push_cast <;> omega
    · simp only [h1, ↓reduceIte]
      omega

/-- **Everything after the load of `q`** is safe: the initial state satisfies the invariant, each
trip keeps it (`trip_safe`) and the epilogue (`post_safe`) ends with `Q`. -/
theorem afterQ_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) {cb ch sm kvh len st : I32}
    (hs : Scal d init i cb ch sm kvh len st) {q : Tensor [BLOCK, d.BD] BF16}
    (hq : QFacts d init i q) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      (afterQD d cb ch sm kvh len st q) := by
  obtain ⟨hrange, htrip⟩ := loop_bound hd hpre i hs
  unfold afterQD afterQ
  refine safe_sub (fun r hr => ?_)
  have hr0 : ∀ j, r.get j = FloatV.negInf := by
    intro j
    have h := hr j
    have e : FloatV.sub (FloatV.ofPat 0 : F32) pinf32 = FloatV.negInf := by decide
    rw [tl.zeros_get_f32] at h
    simp only [Tensor.get_full] at h
    rw [e] at h
    rcases h with h | ⟨h, -⟩
    · exact h.symm
    · exact absurd h (by decide)
  have key := safe_forRange (E := (launch d).env P) (R := Rf d) (Q := fun _ m => Q d init i m)
    (m := init) (init := ((r, tl.zeros [BLOCK] .f32, tl.zeros [BLOCK, d.BD] .f32) :
      Tensor [BLOCK] F32 × Tensor [BLOCK] F32 × Tensor [BLOCK, d.BD] F32))
    (body := fun start_n st3 => loopBodyD d cb ch sm kvh len st q start_n st3)
    (k := fun st3 => postProgD d cb ch sm kvh len st st3.2.2) hrange
    (fun t st3 m => LoopInv d init i t st3 m) (fun _ => ∅) (Wk := Wf d init i)
    (fun _ _ _ _ => by simp) (by simp)
    ⟨rfl, fun row _ => Or.inl ⟨rfl, hr0 _, rfl, fun _ => rfl⟩⟩
    (fun t ht st3 m hI => by
      rw [htrip] at ht
      have hv : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofNat BLOCK : I32).toInt)
          : I32) = IntV.ofInt (0 + (t : Int) * 128) := by
        rw [IntV.toInt_ofInt_zero, IntV.I32.toInt_ofNat (by decide)]; rfl
      rw [hv]
      exact trip_safe_proof hP hd hpre i hs hq ht st3 m hI)
    (fun st3 m hI => by
      rw [htrip] at hI
      exact post_safe_proof hP hd hpre i hs st3 m hI)
  simpa using key

/-- The step lemma for the load of `q`, which belongs to the access proofs. -/
theorem rest_load_q {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) (cb ch sm kvh len st : I32)
    (hs : Scal d init i cb ch sm kvh len st)
    (h : ∀ q : Tensor [BLOCK, d.BD] BF16, QFacts d init i q →
      Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
        (afterQD d cb ch sm kvh len st q)) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      (restProgD d cb ch sm kvh len st) := by
  have _ := hP
  have hsize : (i : Nat) < d.batch * d.H * nBlocks d := i.isLt
  have hbatch : 0 < d.batch := by
    rcases Nat.eq_zero_or_pos d.batch with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hH : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hb : bOf d i < d.batch := Nat.mod_lt _ hbatch
  have hh : hOf d i < d.H := Nat.mod_lt _ hH
  have hm : mOf d i < nBlocks d := mOf_lt d i
  have hSL := seq_le hpre hb
  have hptr := q_ptrs hd hs.st hs.ch hs.sm hSL hh hm
  have hmask := q_mask hd hs.len hs.sm hm
  have hlen := q_len hd hSL hh
  have hR := q_Rf (mN := mOf d i) hd hSL hh
  unfold restProgD restProg
  refine convert_zero_safe fun o ho => ?_
  refine safe_load_tileAt (b := d.q) (ρ := rowBlk (mOf d i)) (κ := lane d.BD)
    (M := lenOf d init (bOf d i)) (N := d.Lk) (mk := some (maskQ d len sm))
    (fun r e hr he => hptr r e hr he) hmask hlen hpre.1 hR fun v hv0 hv1 => ?_
  refine h v fun r e => ?_
  by_cases hc : mOf d i * BLOCK + r < lenOf d init (bOf d i) ∧ e.val < d.Lk
  · simp only [hc, and_self, ite_true]
    obtain ⟨x, hx, hxf⟩ := (hpre.2.2.2.2.2.2.2 (bOf d i) hb).1 (mOf d i * BLOCK + r) (hOf d i)
      e.val hc.1 hh hc.2
    have hrd := hv0 r e (by simpa [rowBlk] using hc.1) hc.2
    have hidx : d.q.ptr ((startOf d init (bOf d i) * d.sqt + hOf d i * d.sqh +
        rowBlk (mOf d i) r * d.sqt + lane d.BD e * 1 : Nat) : Int) =
        d.q.ptr ((qIdx d (startOf d init (bOf d i) + (mOf d i * BLOCK + r)) (hOf d i) e.val : Nat) : Int) := by
      congr 2
      simp only [qIdx, rowBlk, lane]
      ring
    rw [hidx] at hrd
    have hxv : v.get2 r e = x := by
      have := hx
      unfold rdB at this
      rw [hrd] at this
      exact Option.some.inj this
    rw [hxv, DType.val?_bf16]
    refine ⟨hxf, ?_⟩
    simp [qv, valB, hx]
  · simp only [hc, ite_false]
    have hn : ¬ (rowBlk (mOf d i) r < lenOf d init (bOf d i) ∧ lane d.BD e < d.Lk) := by
      simpa [rowBlk, lane] using hc
    rw [hv1 r e hn o rfl, show o.get2 r e = o.get (r, e, ()) from rfl, ho (r, e, ())]
    exact DType.val?_zero_bf16

/-- The body after the five scalar steps is safe for scalars with the right values. -/
theorem rest_safe {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) (cb ch sm kvh len st : I32)
    (hcb : cb.toInt = bOf d i) (hch : ch.toInt = hOf d i) (hsm : sm.toInt = mOf d i)
    (hkv : kvh.toInt = (hOf d i : Int) / kvg d)
    (hlen : len.toInt = lenOf d init (bOf d i)) (hst : st.toInt = startOf d init (bOf d i)) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      (restProg (d.q.ptr 0) (d.k.ptr 0) (d.v.ptr 0) (d.o.ptr 0) (FloatV.hostF32 d.sm_scale)
        (d.b_start_loc.ptr 0) (d.b_seq_len.ptr 0)
        (IntV.ofNat d.sqt) (IntV.ofNat d.sqh) (IntV.ofNat d.skt) (IntV.ofNat d.skh)
        (IntV.ofNat d.svt) (IntV.ofNat d.svh) (IntV.ofNat d.sot) (IntV.ofNat d.soh)
        (kvg d) BLOCK d.BD BLOCK d.causal d.Lk cb ch sm kvh len st) :=
  rest_load_q hP hd hpre i cb ch sm kvh len st ⟨hcb, hch, hsm, hkv, hlen, hst⟩
    fun _ hq => afterQ_safe hP hd hpre i ⟨hcb, hch, hsm, hkv, hlen, hst⟩ hq

end Claims.PrefillAttnVarlen

end

section
/-!
# One program is safe

The scalar steps of the body (the program ids, the key/value head, the sequence's length and
start), then `rest_safe` for the rest.
-/

namespace Claims.PrefillAttnVarlen

open TVA

/-- **One program** of the launch is safe from `init`: it reads inputs only, writes its tile of
`o`, and ends with `Q`. -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe ((launch d).env P) (Rf d) (Wf d init i) (fun _ m => Q d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hsize : (i : Nat) < d.batch * d.H * nBlocks d := i.isLt
  have hbatch : 0 < d.batch := by
    rcases Nat.eq_zero_or_pos d.batch with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hH : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · rw [h] at hsize; simp at hsize
    · exact h
  have hb : bOf d i < d.batch := Nat.mod_lt _ hbatch
  have hh : hOf d i < d.H := Nat.mod_lt _ hH
  have hm : mOf d i < nBlocks d := by
    unfold mOf
    rw [Nat.div_lt_iff_lt_mul (Nat.mul_pos hbatch hH)]
    rw [Nat.mul_comm (nBlocks d)]; exact hsize
  have hnB := nBlocks_le hd
  have hbat31 := hd.batch_lt
  have hH31 := hd.H_lt
  have p0 : (tl.program_id ((launch d).ctx i) 0).toInt = (bOf d i : Int) := by
    rw [tl.program_id, launch_pid0]; exact IntV.I32.toInt_ofNat (by omega)
  have p1 : (tl.program_id ((launch d).ctx i) 1).toInt = (hOf d i : Int) := by
    rw [tl.program_id, launch_pid1]; exact IntV.I32.toInt_ofNat (by omega)
  have p2 : (tl.program_id ((launch d).ctx i) 2).toInt = (mOf d i : Int) := by
    rw [tl.program_id, launch_pid2]; exact IntV.I32.toInt_ofNat (by omega)
  have hkvg : 0 < kvg d := by
    obtain ⟨c, hc⟩ := hd.kv_dvd
    have := hd.kv_pos
    unfold kvg
    rw [hc, Nat.mul_div_cancel_left _ this]
    rcases Nat.eq_zero_or_pos c with h | h
    · rw [h] at hc; omega
    · exact h
  have hkvg31 : kvg d < 2 ^ 31 := lt_of_le_of_lt (Nat.div_le_self _ _) (by have := hd.H_lt; omega)
  unfold model
  rw [body_eq]
  refine safe_idiv1 (by rw [p1]; positivity) (by
    rw [show (IntV.ofNat (kvg d) : I32).toInt = (kvg d : Int) from IntV.I32.toInt_ofNat hkvg31]
    exact_mod_cast hkvg) fun r hr => ?_
  have hptr : ∀ b : Buf .i32, b.ptr 0 + tl.program_id ((launch d).ctx i) 0 = b.ptr (bOf d i : Int) := by
    intro b
    show (b.ptr 0).offset _ = _
    rw [Buf.ptr_offset, p0, zero_add]
  obtain ⟨-, -, -, -, hvs, hvl, ⟨hseq, -⟩, -⟩ := id hpre
  obtain ⟨s, n, hs, hn, hs0, hn0, -, -⟩ := hseq (bOf d i) hb
  rw [hptr]
  refine safe_load1_buf d.b_seq_len (i := bOf d i) (by have := hd.bsq_len; omega) hvl
    (fun x hx => ?_) fun vlen hvlen => ?_
  · exact Or.inr (d.b_seq_len.span_sub (by have := hd.bsq_len; omega) hx)
  rw [hptr]
  refine safe_load1_buf d.b_start_loc (i := bOf d i) (by have := hd.bsl_len; omega) hvs
    (fun x hx => ?_) fun vst hvst => ?_
  · exact Or.inl (Or.inr (d.b_start_loc.span_sub (by have := hd.bsl_len; omega) hx))
  have e1 : vlen = n := by rw [hvlen] at hn; exact Option.some.inj hn
  have e2 : vst = s := by rw [hvst] at hs; exact Option.some.inj hs
  subst e1 e2
  have hlen : lenOf d init (bOf d i) = vlen.toInt.toNat := by unfold lenOf val32; rw [hvlen]
  have hst : startOf d init (bOf d i) = vst.toInt.toNat := by unfold startOf val32; rw [hvst]
  refine rest_safe hP hd hpre i _ _ _ _ _ _ p0 p1 p2 ?_ ?_ ?_
  · rw [hr, p1, IntV.I32.toInt_ofNat hkvg31]
  · rw [hlen]; exact (Int.toNat_of_nonneg hn0).symm
  · rw [hst]; exact (Int.toNat_of_nonneg hs0).symm

end Claims.PrefillAttnVarlen

end
