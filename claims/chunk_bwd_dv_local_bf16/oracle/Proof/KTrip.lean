import Claim
import TVA.Proof
import Proof.Setup
import Proof.Spec
import Proof.ValsLane
import Proof.LoadLanes
import Proof.Prologue

/-!
# The two loads of one trip of the `K` loop

Trip `t` loads `b_k` (rows: the chunk's tokens; columns `t·BK + kk`) from `k` and `b_q`
(its transpose) from `q`. `safe_kload_k`, `safe_kload_q` give the values: at every enabled token
lane `l` and column `kk` the entry is finite, the exact `k`/`q` element when the column is below
`K` and `+0` otherwise (`KvalK`, `KvalQ`).
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The column offsets `i_k·BK + tl.arange(0, BK)` of a trip of the `K` loop. -/
def okOf (d : Inst) (i_k : I32) : Tensor [d.BK] I32 :=
  i_k * (IntV.ofNat d.BK : I32) + tl.arange 0 d.BK

/-- Lane `kk` of `i·B + tl.arange(0, B)` for the loop variable `i` of value `t`. -/
theorem ofs_lane {i : I32} {t B : Nat} (hi : i.toInt = (t : Int)) (hB : B < 2 ^ 31)
    (hfit : t * B + B < 2 ^ 31) (kk : Fin B) :
    ((i * (IntV.ofNat B : I32) + tl.arange 0 B).get1 kk).toInt = ((t * B + kk.val : Nat) : Int) := by
  have hm : (i * (IntV.ofNat B : I32)).toInt = ((t * B : Nat) : Int) :=
    IntV.I32.mul_nat hi (IntV.I32.toInt_ofNat hB) (by omega)
  rw [Tensor.get1_sadd]
  exact IntV.I32.add_nat hm (IntV.I32.toInt_tl_arange (by omega) kk) (by have := kk.isLt; omega)

/-- What `b_k` holds after trip `t`'s load: at an enabled token lane, column `kk` is finite and
equals `k[pos l, hk, t·BK + kk]` when that column is below `K`, and `0` otherwise. -/
def KvalK (d : Inst) (init : Mem) (r bh t : Nat) (v : Tensor [d.BT, d.BK] BF16) : Prop :=
  ∀ l : Fin d.BT, en d r bh l = true → ∀ kk : Fin d.BK,
    (v.get2 l kk).isFinite = true ∧
      (v.get2 l kk).toRat = if t * d.BK + kk.val < d.K then
        valB init d.k (qkIdx d (pos d r bh l) (hk d (hvOf d bh)) (t * d.BK + kk.val)) else 0

/-- What `b_q` holds after trip `t`'s load: the transpose of `KvalK`, for `q`. -/
def KvalQ (d : Inst) (init : Mem) (r bh t : Nat) (v : Tensor [d.BK, d.BT] BF16) : Prop :=
  ∀ l : Fin d.BT, en d r bh l = true → ∀ kk : Fin d.BK,
    (v.get2 kk l).isFinite = true ∧
      (v.get2 kk l).toRat = if t * d.BK + kk.val < d.K then
        valB init d.q (qkIdx d (pos d r bh l) (hk d (hvOf d bh)) (t * d.BK + kk.val)) else 0

/-! ### Lanes of the pieces -/

/-- Column lane `kk` of trip `t` of the `K` loop is `t·BK + kk`. -/
theorem okOf_lane {d : Inst} (hd : d.ok) {t : Nat} (ht : t * d.BK < d.K) {i_k : I32}
    (hik : i_k.toInt = (t : Int)) (kk : Fin d.BK) :
    ((okOf d i_k).get1 kk).toInt = ((t * d.BK + kk.val : Nat) : Int) := by
  have := hd.K_le; have := hd.bk_le
  exact ofs_lane hik (by omega) (by omega) kk

/-- The column mask `o_k < K`. -/
theorem okMask_lane {d : Inst} (hd : d.ok) {t : Nat} (ht : t * d.BK < d.K) {i_k : I32}
    (hik : i_k.toInt = (t : Int)) (kk : Fin d.BK) :
    (((okOf d i_k).ltS (IntV.ofNat d.K : I32)).get1 kk) = decide (t * d.BK + kk.val < d.K) := by
  show IntV.lt ((okOf d i_k).get1 kk) (IntV.ofNat d.K : I32) = _
  have := hd.K_le
  rw [IntV.lt_eq, okOf_lane hd ht hik kk, IntV.I32.toInt_ofNat (by omega)]
  simp only [Nat.cast_lt]

/-- The row offsets `o_t · (H·K)` at an enabled lane. -/
theorem otHK_lane {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T) {l : Fin d.BT} (hl : en d r bh l = true) :
    (o_t.get1 l * (IntV.ofNat (d.H * d.K) : I64)).toInt =
      (((chkOf d r bh * d.BT + l.val) * (d.H * d.K) : Nat) : Int) := by
  have h1 := qk_tile_lt hd hT hr hbh
  have hlen : d.B * d.T * d.H * d.K ≤ d.k.len := hd.k_len
  have hv := buf_len_le hpre.2.1
  simp only [DType.bytes] at hv
  have hi' : chkOf d r bh * d.BT + l.val ≤ lenOf d r bh - 1 := by
    have := (en_iff r bh l.val).mp hl; omega
  have hmul : (chkOf d r bh * d.BT + l.val) * (d.H * d.K) ≤ (lenOf d r bh - 1) * (d.H * d.K) :=
    Nat.mul_le_mul_right _ hi'
  have hHK : d.H * d.K < 2 ^ 63 := by
    have hH : d.H < 2 ^ 31 := by
      have := hd.hvv_lt
      have h' : d.HV ≤ d.HV * d.V := Nat.le_mul_of_pos_right _ hd.V_pos
      have := Nat.le_of_dvd hd.HV_pos hd.hv_dvd
      omega
    have hK : d.K ≤ 2 ^ 8 := hd.K_le
    calc d.H * d.K ≤ 2 ^ 31 * 2 ^ 8 := Nat.mul_le_mul hH.le hK
      _ < 2 ^ 63 := by norm_num
  exact IntV.I64.mul_nat (S.ot l) (IntV.I64.toInt_ofNat hHK) (by omega)

/-- **The load of `b_k`** (line 134): trip `t` of the `K` loop reads the tile of `k` of the chunk's
tokens and columns `t·BK + kk`, masked to the sequence and to `K`; masked-off lanes hold `+0`. -/
theorem safe_kload_k {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r bh : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T)
    {t : Nat} (ht : t * d.BK < d.K) {i_k : I32} (hik : i_k.toInt = (t : Int))
    {E : Env} {α : Type} {Wk : Set Int} {Q : α → Mem → Prop}
    {cont : Tensor [d.BT, d.BK] BF16 → Prog α}
    (hcont : ∀ v, KvalK d init r bh t v → Safe E (R d r bh) Wk Q init (cont v)) :
    Safe E (R d r bh) Wk Q init
      (tl.convert .f32 .bf16 (tl.full [d.BT, d.BK] zf32) >>= fun z =>
        tl.load (k + (o_t * (IntV.ofNat (d.H * d.K) : I64)).col d.BK + (okOf d i_k).row d.BT)
          (some (((o_t.ltS T).col d.BK).and
            (((okOf d i_k).ltS (IntV.ofNat d.K : I32)).row d.BT))) (some z) >>= cont) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hV : ValsOk d init := hpre.2.2.2.2.2.2
  refine safe_conv_zero fun z hz => ?_
  have hx : ∀ (i : Fin d.BT) (j : Fin d.BK), chkOf d r bh * d.BT + i.val < lenOf d r bh →
      (((o_t * (IntV.ofNat (d.H * d.K) : I64)).col d.BK).get2 i j).toInt =
        (((chkOf d r bh * d.BT + i.val) * (d.H * d.K) : Nat) : Int) :=
    fun i j hi => otHK_lane hd hT hpre hr hbh S ((en_iff r bh i.val).mpr hi)
  have hy : ∀ (i : Fin d.BT) (j : Fin d.BK), t * d.BK + j.val < d.K →
      (((okOf d i_k).row d.BT).get2 i j).toInt = (((t * d.BK + j.val) * 1 : Nat) : Int) := by
    intro i j hj
    rw [mul_one]; exact okOf_lane hd ht hik j
  have hp := Tile2.ptr_lane_seq_of_on d.k S.k_eq
    (ρ := fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (κ := fun j : Fin d.BK => t * d.BK + j.val)
    (M := lenOf d r bh) (N := d.K) hx hy
  have hm : Tile2.MaskAt (some (((o_t.ltS T).col d.BK).and
      (((okOf d i_k).ltS (IntV.ofNat d.K : I32)).row d.BT)))
      (fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (fun j : Fin d.BK => t * d.BK + j.val)
      (lenOf d r bh) d.K := by
    refine Tile2.maskAt_of_get fun i j => ?_
    show ((o_t.ltS T).get1 i && ((okOf d i_k).ltS (IntV.ofNat d.K : I32)).get1 j) = _
    rw [mt_get S i, okMask_lane hd ht hik j]
    rfl
  have hkt := qk_tile_lt hd hT hr hbh
  have hlen : d.B * d.T * d.H * d.K ≤ d.k.len := hd.k_len
  have hlen' : lenOf d r bh = 0 ∨ d.K = 0 ∨
      qOff d r bh + (lenOf d r bh - 1) * (d.H * d.K) + (d.K - 1) * 1 < d.k.len :=
    Or.inr (Or.inr (by omega))
  refine safe_load_tileAt d.k hp hm hlen' hpre.2.1 ?_ ?_
  · intro x hx'
    have := Buf.tileAt_sub_bytes d.k hlen' hx'
    unfold R
    simp only [Set.mem_union]
    tauto
  · intro v hv1 hv2
    refine hcont v fun l hl kk => ?_
    by_cases hkk : t * d.BK + kk.val < d.K
    · have hread := hv1 l kk ((en_iff r bh l.val).mp hl) hkk
      have hidx : qOff d r bh + (chkOf d r bh * d.BT + l.val) * (d.H * d.K) + (t * d.BK + kk.val) * 1 =
          qkIdx d (pos d r bh l) (hk d (hvOf d bh)) (t * d.BK + kk.val) := by
        unfold qOff qkIdx pos baseOf; ring
      rw [hidx] at hread
      obtain ⟨hqf, hkf⟩ := (vals_tok hd hT hV hr hbh hl).1 (hk d (hvOf d bh)) (t * d.BK + kk.val)
        (hk_lt hd bh) hkk
      obtain ⟨xv, hxv, hxfin⟩ := hkf
      unfold rdB at hxv
      have hvx : v.get2 l kk = xv := Option.some.inj (hread.symm.trans hxv)
      rw [ite_eq_left hkk]
      refine ⟨by rw [hvx]; exact hxfin, ?_⟩
      unfold valB rdB
      rw [hxv]
      show (v.get2 l kk).toRat = xv.toRat
      rw [hvx]
    · have hz0 : v.get2 l kk = FloatV.ofPat 0 := by
        have := hv2 l kk (fun h => hkk h.2) z rfl
        rw [this]; exact hz (l, kk, ())
      rw [ite_eq_right hkk, hz0]
      exact ⟨(FloatV.isFinite_ofPat_zero Format.bfloat16_regular), FloatV.toRat_ofPat_zero⟩

/-- **The load of `b_q`** (line 135): the same tile of `q`, transposed. -/
theorem safe_kload_q {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r bh : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T)
    {t : Nat} (ht : t * d.BK < d.K) {i_k : I32} (hik : i_k.toInt = (t : Int))
    {E : Env} {α : Type} {Wk : Set Int} {Q : α → Mem → Prop}
    {cont : Tensor [d.BK, d.BT] BF16 → Prog α}
    (hcont : ∀ v, KvalQ d init r bh t v → Safe E (R d r bh) Wk Q init (cont v)) :
    Safe E (R d r bh) Wk Q init
      (tl.convert .f32 .bf16 (tl.full [d.BK, d.BT] zf32) >>= fun z =>
        tl.load (q + (okOf d i_k).col d.BT + (o_t * (IntV.ofNat (d.H * d.K) : I64)).row d.BK)
          (some ((((okOf d i_k).ltS (IntV.ofNat d.K : I32)).col d.BT).and
            ((o_t.ltS T).row d.BK))) (some z) >>= cont) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hV : ValsOk d init := hpre.2.2.2.2.2.2
  refine safe_conv_zero fun z hz => ?_
  have hx : ∀ (i : Fin d.BK) (j : Fin d.BT), t * d.BK + i.val < d.K →
      (((okOf d i_k).col d.BT).get2 i j).toInt = (((t * d.BK + i.val) * 1 : Nat) : Int) := by
    intro i j hi
    rw [mul_one]; exact okOf_lane hd ht hik i
  have hy : ∀ (i : Fin d.BK) (j : Fin d.BT), chkOf d r bh * d.BT + j.val < lenOf d r bh →
      (((o_t * (IntV.ofNat (d.H * d.K) : I64)).row d.BK).get2 i j).toInt =
        (((chkOf d r bh * d.BT + j.val) * (d.H * d.K) : Nat) : Int) :=
    fun i j hj => otHK_lane hd hT hpre hr hbh S ((en_iff r bh j.val).mpr hj)
  have hp := Tile2.ptr_lane_seq_of_on d.q S.q_eq
    (ρ := fun i : Fin d.BK => t * d.BK + i.val)
    (κ := fun j : Fin d.BT => chkOf d r bh * d.BT + j.val)
    (M := d.K) (N := lenOf d r bh) hx hy
  have hm : Tile2.MaskAt (some ((((okOf d i_k).ltS (IntV.ofNat d.K : I32)).col d.BT).and
      ((o_t.ltS T).row d.BK)))
      (fun i : Fin d.BK => t * d.BK + i.val)
      (fun j : Fin d.BT => chkOf d r bh * d.BT + j.val) d.K (lenOf d r bh) := by
    refine Tile2.maskAt_of_get fun i j => ?_
    show (((okOf d i_k).ltS (IntV.ofNat d.K : I32)).get1 i && (o_t.ltS T).get1 j) = _
    rw [mt_get S j, okMask_lane hd ht hik i]
    rfl
  have hkt := qk_tile_lt hd hT hr hbh
  have hlen : d.B * d.T * d.H * d.K ≤ d.q.len := hd.q_len
  have hlen' : d.K = 0 ∨ lenOf d r bh = 0 ∨
      qOff d r bh + (d.K - 1) * 1 + (lenOf d r bh - 1) * (d.H * d.K) < d.q.len :=
    Or.inr (Or.inr (by omega))
  refine safe_load_tileAt d.q hp hm hlen' hpre.1 ?_ ?_
  · intro x hx'
    have := Buf.tileAt_sub_bytes d.q hlen' hx'
    unfold R
    simp only [Set.mem_union]
    tauto
  · intro v hv1 hv2
    refine hcont v fun l hl kk => ?_
    by_cases hkk : t * d.BK + kk.val < d.K
    · have hread := hv1 kk l hkk ((en_iff r bh l.val).mp hl)
      have hidx : qOff d r bh + (t * d.BK + kk.val) * 1 + (chkOf d r bh * d.BT + l.val) * (d.H * d.K) =
          qkIdx d (pos d r bh l) (hk d (hvOf d bh)) (t * d.BK + kk.val) := by
        unfold qOff qkIdx pos baseOf; ring
      rw [hidx] at hread
      obtain ⟨hqf, hkf⟩ := (vals_tok hd hT hV hr hbh hl).1 (hk d (hvOf d bh)) (t * d.BK + kk.val)
        (hk_lt hd bh) hkk
      obtain ⟨xv, hxv, hxfin⟩ := hqf
      unfold rdB at hxv
      have hvx : v.get2 kk l = xv := Option.some.inj (hread.symm.trans hxv)
      rw [ite_eq_left hkk]
      refine ⟨by rw [hvx]; exact hxfin, ?_⟩
      unfold valB rdB
      rw [hxv]
      show (v.get2 kk l).toRat = xv.toRat
      rw [hvx]
    · have hz0 : v.get2 kk l = FloatV.ofPat 0 := by
        have := hv2 kk l (fun h => hkk h.1) z rfl
        rw [this]; exact hz (kk, l, ())
      rw [ite_eq_right hkk, hz0]
      exact ⟨(FloatV.isFinite_ofPat_zero Format.bfloat16_regular), FloatV.toRat_ofPat_zero⟩

end Claims.ChunkBwdDvLocal
