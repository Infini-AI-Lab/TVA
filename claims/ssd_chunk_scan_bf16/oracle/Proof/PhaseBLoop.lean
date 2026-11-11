import Proof.PhaseBStep

/-!
# Phase B: the loop

The loop of line 397 as a whole: the invariant holds before the first trip (`BInv` at `0`, from
`APost`), each trip keeps it (`bstep_safe`), and after the last trip `loop_final` gives `BPost`.
-/

namespace Claims.SsdChunkScan

open TVA

/-- **Phase B is safe**: the loop leaves `BPost`. -/
theorem phaseB_safe {w : Nat} {E : Env} {d : Inst} {init : Mem} {k pm pn h : Nat}
    (S : Std d init E k pm pn h) {α : Type} {Q : α → Mem → Prop} {Wf : Set Int}
    {cb_ptr : Ptr .f32} {x_ptr : Ptr .bf16} {dt_ptr dA_ptr : Ptr .f32} {offs_m : Tensor [BM] I32}
    {offs_n : Tensor [BN] I32} {c_off : I32}
    {widen : {n : Nat} → Tensor [n] I32 → Tensor [n] (IntV true w)}
    {csw limit kmaxarg : IntV true w} {dA_cs_m : Tensor [BM] F32} {acc₀ : Tensor [BM, BN] F32}
    (hw : 32 ≤ w)
    (hcb : cb_ptr = d.cb.ptr (cbIdx d (lc d k) (grp d h) 0 0 : Int))
    (hx : x_ptr = d.x.ptr (xIdx d (lc d k * d.cs) h 0 : Int))
    (hdt : dt_ptr = d.dt.ptr (dtIdx d h (lc d k) 0 : Int))
    (hdA : dA_ptr = d.dA.ptr (dtIdx d h (lc d k) 0 : Int))
    (hom : ∀ ii : Fin BM, (offs_m.get1 ii).toInt = (rowOf d k pm ii : Int))
    (hon : ∀ jj : Fin BN, (offs_n.get1 jj).toInt = (colOf pn jj : Int))
    (hco : c_off.toInt = (lo d k : Int))
    (hwiden : ∀ {n : Nat} (x : Tensor [n] I32) (i : Fin n),
      ((widen x).get1 i).toInt = (x.get1 i).toInt)
    (hcsw : csw.toInt = (d.cs : Int)) (hlim : limit.toInt = (lend d k : Int))
    (hkm : kmaxarg.toInt = ((BM * (pm + 1) : Nat) : Int))
    (hdAm : ∀ ii : Fin BM, RowOk d k pm ii → ∃ v : F32,
      init.read .f32 (d.dA.ptr (dtIdx d h (lc d k) (rowOf d k pm ii) : Int)).addr = some v ∧
        dA_cs_m.get1 ii = v)
    (hA : APost d init k pm pn h acc₀) {kont : Tensor [BM, BN] F32 → Prog α}
    (hkont : ∀ acc, BPost d init k pm pn h acc → Safe E (R d) Wf Q init (kont acc)) :
    Safe E (R d) Wf Q init
      (phaseB d cb_ptr x_ptr dt_ptr dA_ptr offs_m offs_n c_off widen csw limit kmaxarg dA_cs_m
        acc₀ kont) := by
  have S' := S
  obtain ⟨hd, hpre, dev, hk, hpm, hpn, hh⟩ := S
  unfold phaseB
  have hlend := lend_le hd hk
  have hcsp := cs_pos hd
  have hcs14 := hd.cs_le
  have hlo := lo_lt hd hk
  have hpow : 2 ^ 31 ≤ 2 ^ (w - 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hok : ∀ l : Fin BK, ((tl.arange 0 BK + c_off : Tensor [BK] I32).get1 l).toInt =
      ((lo d k + l : Nat) : Int) :=
    fun l => by
      have h1 := IntV.I32.toInt_tl_arange (n := BK) (by norm_num [BK]) l
      have h2 := l.isLt
      rw [Tensor.get1_adds, show lo d k + (l : Nat) = (l : Nat) + lo d k by ring]
      exact IntV.add_nat h1 hco (by norm_num [BK] at *; omega)
  have C : BCtx d init k pm pn h offs_m offs_n (tl.arange 0 BK + c_off) widen csw limit dA_cs_m :=
    ⟨hw, hom, hon, hok, hwiden, hcsw, hlim, hdAm⟩
  have hKm : ((tl.iminimum (Tensor.scalar kmaxarg) (Tensor.scalar limit)).item).toInt =
      ((min (BM * (pm + 1)) (lend d k) : Nat) : Int) := by
    show (IntV.min kmaxarg limit).toInt = _
    rw [IntV.toInt_min, hkm, hlim]; push_cast; rfl
  have hlo0 : (IntV.ofInt 0 : IntV true w).toInt = 0 := IntV.toInt_ofInt_zero
  have hst : (IntV.ofNat BK : IntV true w).toInt = (BK : Int) :=
    IntV.toInt_ofNat (lt_of_lt_of_le (by norm_num [BK]) hpow)
  have hrange : tl.RangeOk (IntV.ofInt 0 : IntV true w)
      ((tl.iminimum (Tensor.scalar kmaxarg) (Tensor.scalar limit)).item) (IntV.ofNat BK) := by
    have hpowI : (2 : Int) ^ 31 ≤ 2 ^ (w - 1) := by exact_mod_cast hpow
    unfold tl.RangeOk IntV.maxInt
    simp only [ite_true]
    rw [hKm, hst, hlo0]
    refine ⟨by norm_num [BK], ?_, ?_, ?_⟩ <;> (try simp only [BK]) <;> push_cast <;> omega
  have hnt : tl.tripCount (IntV.ofInt 0 : IntV true w).toInt
      ((tl.iminimum (Tensor.scalar kmaxarg) (Tensor.scalar limit)).item).toInt
      (IntV.ofNat BK : IntV true w).toInt = (min (BM * (pm + 1)) (lend d k) + 31) / 32 := by
    rw [hlo0, hKm, hst]; unfold tl.tripCount; simp only [BK]; omega
  refine Safe.mono (safe_forRange hrange (fun n st m => BInv d init k pm pn h n st m)
    (fun _ => (∅ : Set Int)) (Wk := Wf) ?_ ?_ ?_ ?_ ?_) (fun _ _ hq => hq) (by simp)
  · intro _ _ _ _; exact Set.disjoint_empty _
  · simp
  · -- before the first trip
    have hcsv : (IntV.ofNat d.cs : I32).toInt = (d.cs : Int) := IntV.I32.toInt_ofNat (by omega)
    have hone : (IntV.ofNat 1 : I32).toInt = ((1 : Nat) : Int) :=
      IntV.I32.toInt_ofNat (by norm_num)
    have hxTv : (IntV.ofNat d.xT : I32).toInt = (d.xT : Int) := by
      refine IntV.I32.toInt_ofNat ?_
      have := hd.x_off; have : d.xT ≤ (2 * d.cs + 64) * d.xT := Nat.le_mul_of_pos_left _ (by omega)
      omega
    have hok' : ∀ l : Fin BK, ((tl.arange 0 BK + c_off : Tensor [BK] I32).get1 l).toInt =
        ((lo d k + BK * 0 + l : Nat) : Int) := fun l => by simpa using hok l
    have hcbP := Tile2.ptr_lane_on (d := .f32) d.cb hcb (rows := offs_m)
      (cols := (tl.arange 0 BK + c_off : Tensor [BK] I32)) (a0 := (IntV.ofNat d.cs : I32))
      (a1 := (IntV.ofNat 1 : I32)) (ρ := fun i : Fin BM => rowOf d k pm i)
      (κ := fun l : Fin BK => lo d k + BK * 0 + l) (M := d.cs) (N := lo d k + BK) hom hok' hcsv hone
      (by have : (d.cs - 1) * d.cs ≤ 2 ^ 14 * 2 ^ 14 := Nat.mul_le_mul (by omega) hcs14
          simp only [BK] at *; omega)
    have hxP := Tile2.ptr_lane_on (d := .bf16) d.x hx (rows := (tl.arange 0 BK + c_off : Tensor [BK] I32))
      (cols := offs_n) (a0 := (IntV.ofNat d.xT : I32)) (a1 := (IntV.ofNat 1 : I32))
      (ρ := fun l : Fin BK => lo d k + BK * 0 + l) (κ := fun jj : Fin BN => colOf pn jj)
      (M := lo d k + BK) (N := d.P) hok' hon hxTv hone
      (by have h1 : (lo d k + BK) * d.xT ≤ (2 * d.cs + 64) * d.xT :=
            Nat.mul_le_mul_right _ (by simp only [BK]; omega)
          have e : (lo d k + BK - 1) * d.xT + d.xT = (lo d k + BK) * d.xT := by
            rw [← Nat.succ_mul]; congr 1
          have := hd.x_off
          have h3 : d.P ≤ d.H * d.P := Nat.le_mul_of_pos_left _ hd.H_pos
          have := hd.xT_ge
          have := hd.P_pos
          omega)
    have hlt1 : ∀ l : Fin BK, (lo d k + l) * 1 < 2 ^ (32 - 1) := by
      intro l; have := l.isLt; simp only [BK] at *; omega
    have hm : ∀ l : Fin BK, (((tl.arange 0 BK + c_off : Tensor [BK] I32) *
        (IntV.ofNat 1 : I32)).get (l, ())).toInt = (((lo d k + l) * 1 : Nat) : Int) := fun l => by
      have := Tensor.get1_muls (tl.arange 0 BK + c_off : Tensor [BK] I32) (IntV.ofNat 1 : I32) l
      rw [show ((tl.arange 0 BK + c_off : Tensor [BK] I32) * (IntV.ofNat 1 : I32)).get (l, ()) =
        ((tl.arange 0 BK + c_off : Tensor [BK] I32) * (IntV.ofNat 1 : I32)).get1 l from rfl, this]
      exact IntV.mul_nat (hok l) hone (hlt1 l)
    refine ⟨rfl, ?_, ?_, ?_, ?_, ?_⟩
    · intro ii l hrow
      exact hcbP ii l hrow (by simp only [BK]; omega)
    · intro l jj hj
      exact hxP l jj (by simp only [BK]; omega) hj
    · intro l
      show dt_ptr.offset _ = _
      rw [hdt, Buf.ptr_offset, hm l]; congr 1; push_cast; ring
    · intro l
      show dA_ptr.offset _ = _
      rw [hdA, Buf.ptr_offset, hm l]; congr 1; push_cast; ring
    · intro hV ii jj hri hcj
      obtain ⟨hfin, h1, h2⟩ := hA hV ii jj hri hcj
      exact loop_init (lTok_tile hd hk hri) hfin h1 h2
  · intro n hn st m1 hI
    rw [hnt] at hn
    have hkk : (IntV.ofInt ((IntV.ofInt 0 : IntV true w).toInt + (n : Int) *
        (IntV.ofNat BK : IntV true w).toInt) : IntV true w).toInt = ((BK * n : Nat) : Int) := by
      rw [hlo0, hst]
      have e : (0 : Int) + (n : Int) * (BK : Int) = ((BK * n : Nat) : Int) := by push_cast; ring
      rw [e]
      exact IntV.toInt_ofInt (by omega) (IntV.sfits_natCast (lt_of_lt_of_le
        (by simp only [BK] at *; omega) hpow))
    exact bstep_safe S' C (by simp only [BK] at *; omega) hkk hI
  · intro st m1 hI
    obtain ⟨acc, cbp, xp, dtp, dAp⟩ := st
    obtain ⟨rfl, -, -, -, -, hacc⟩ := hI
    refine hkont acc ?_
    intro hV ii jj hri hcj
    have ht' : tokOf d k pm ii < d.T := (tok_of_row hd hk ⟨by unfold rowOf; omega, hri⟩).1
    refine loop_final hd hV ht' hh hcj (lTok_tile hd hk hri) ?_ ?_ (hacc hV ii jj hri hcj)
    · rw [hnt]
      unfold nIt
      show _ ≤ (d.cs + 32 - 1) / 32
      exact Nat.div_le_div_right (by omega)
    · rw [hnt]
      have hri' : lo d k + BM * pm + ii < lend d k := hri
      unfold tokOf rowOf
      have := ii.isLt
      simp only [BK, BM] at *
      omega

end Claims.SsdChunkScan
