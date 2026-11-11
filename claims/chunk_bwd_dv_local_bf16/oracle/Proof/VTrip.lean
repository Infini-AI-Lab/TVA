import Claim
import TVA.Proof
import Proof.Setup
import Proof.Spec
import Proof.ValsLane
import Proof.LoadLanes
import Proof.Prologue
import Proof.VNum
import Proof.KTrip

/-!
# One trip of the `V` loop

Trip `t` loads the tile of `do` (`safe_vload_do`), multiplies it by the rounded matrix, rounds
the product to `bfloat16` and stores it into the tile of `dv` of the same rows and the columns
`t·BV .. t·BV + BV - 1` below `V` (`safe_vBody`). The memory the trip starts from agrees with `init`
on everything the program reads and holds the earlier trips' columns (`QCols`).
-/

namespace Claims.ChunkBwdDvLocal

open TVA

/-- The column offsets `i_v·BV + tl.arange(0, BV)` of a trip of the `V` loop. -/
def ovOf (d : Inst) (i_v : I32) : Tensor [d.BV] I32 :=
  i_v * (IntV.ofNat d.BV : I32) + tl.arange 0 d.BV

/-- Memory `m` agrees with `init` on everything program `(r, bh)` reads. -/
def Agree (d : Inst) (init : Mem) (r bh : Nat) (m : Mem) : Prop := ∀ x ∈ R d r bh, m x = init x

/-- Memory `m` holds, in the first `n` columns, the finite `bfloat16` within the tolerance of
`ref` at every enabled lane of `dv`. -/
def QCols (d : Inst) (init : Mem) (r bh n : Nat) (m : Mem) : Prop :=
  ∀ l, l < d.BT → en d r bh l = true → ∀ e, e < n → e < d.V →
    ∃ c : BF16, rdB m d.dv (wEl d r bh l e) = some c ∧ c.isFinite = true ∧
      |c.toReal - ref d init (nOf d r bh) (pos d r bh l) (hvOf d bh) e| ≤
        tol d init (nOf d r bh) (pos d r bh l) (hvOf d bh) e

/-- `Q` is `QCols` for all columns. -/
theorem Q_of_QCols {d : Inst} {init : Mem} {r bh : Nat} {m : Mem} {n : Nat} (hn : d.V ≤ n)
    (h : QCols d init r bh n m) : Q d init r bh m :=
  fun l hl he e hev => h l hl he e (lt_of_lt_of_le hev hn) hev

/-- Column lane `j` of trip `t` of the `V` loop is `t·BV + j`. -/
theorem ovOf_lane {d : Inst} (hd : d.ok) {t : Nat} (ht : t * d.BV < d.V) {i_v : I32}
    (hiv : i_v.toInt = (t : Int)) (j : Fin d.BV) :
    ((ovOf d i_v).get1 j).toInt = ((t * d.BV + j.val : Nat) : Int) := by
  have := hd.V_le; have := hd.bv_le
  exact ofs_lane hiv (by omega) (by omega) j

/-- The column mask `o_v < V`. -/
theorem ovMask_lane {d : Inst} (hd : d.ok) {t : Nat} (ht : t * d.BV < d.V) {i_v : I32}
    (hiv : i_v.toInt = (t : Int)) (j : Fin d.BV) :
    (((ovOf d i_v).ltS (IntV.ofNat d.V : I32)).get1 j) = decide (t * d.BV + j.val < d.V) := by
  show IntV.lt ((ovOf d i_v).get1 j) (IntV.ofNat d.V : I32) = _
  have := hd.V_le
  rw [IntV.lt_eq, ovOf_lane hd ht hiv j, IntV.I32.toInt_ofNat (by omega)]
  simp only [Nat.cast_lt]

/-- What `b_do` holds after trip `t`'s load (`DoVal`), or the row offsets `o_t · (HV·V)` at an
enabled lane. -/
theorem otHV_lane {d : Inst} (hd : d.ok) (hT : d.varlen = true → TablesOk d) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T) {l : Fin d.BT} (hl : en d r bh l = true) :
    (o_t.get1 l * (IntV.ofNat (d.HV * d.V) : I64)).toInt =
      (((chkOf d r bh * d.BT + l.val) * (d.HV * d.V) : Nat) : Int) := by
  have h1 := o_tile_lt hd hT hr hbh
  have hlen : d.B * d.T * d.HV * d.V ≤ d.dO.len := hd.dO_len
  have hv := buf_len_le hpre.2.2.1
  simp only [DType.bytes] at hv
  have hi' : chkOf d r bh * d.BT + l.val ≤ lenOf d r bh - 1 := by
    have := (en_iff r bh l.val).mp hl; omega
  have hmul : (chkOf d r bh * d.BT + l.val) * (d.HV * d.V) ≤ (lenOf d r bh - 1) * (d.HV * d.V) :=
    Nat.mul_le_mul_right _ hi'
  have hHV : d.HV * d.V < 2 ^ 63 := by have := hd.hvv_lt; omega
  exact IntV.I64.mul_nat (S.ot l) (IntV.I64.toInt_ofNat hHV) (by omega)

/-- **The load of `b_do`** (line 147): trip `t` of the `V` loop reads the tile of `do` of the
chunk's tokens and the columns `t·BV + j` below `V`; masked-off lanes hold `+0`. -/
theorem safe_vload_do {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {r bh : Nat}
    (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T)
    {t : Nat} (ht : t * d.BV < d.V) {i_v : I32} (hiv : i_v.toInt = (t : Int))
    {m : Mem} (hag : Agree d init r bh m)
    {E : Env} {α : Type} {Wk : Set Int} {Q : α → Mem → Prop}
    {cont : Tensor [d.BT, d.BV] BF16 → Prog α}
    (hcont : ∀ v, DoVal d init r bh t v → Safe E (R d r bh) Wk Q m (cont v)) :
    Safe E (R d r bh) Wk Q m
      (tl.convert .f32 .bf16 (tl.full [d.BT, d.BV] zf32) >>= fun z =>
        tl.load (do_ + (o_t * (IntV.ofNat (d.HV * d.V) : I64)).col d.BV + (ovOf d i_v).row d.BT)
          (some (((o_t.ltS T).col d.BV).and
            (((ovOf d i_v).ltS (IntV.ofNat d.V : I32)).row d.BT))) (some z) >>= cont) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hV : ValsOk d init := hpre.2.2.2.2.2.2
  refine safe_conv_zero fun z hz => ?_
  have hx : ∀ (i : Fin d.BT) (j : Fin d.BV), chkOf d r bh * d.BT + i.val < lenOf d r bh →
      (((o_t * (IntV.ofNat (d.HV * d.V) : I64)).col d.BV).get2 i j).toInt =
        (((chkOf d r bh * d.BT + i.val) * (d.HV * d.V) : Nat) : Int) :=
    fun i j hi => otHV_lane hd hT hpre hr hbh S ((en_iff r bh i.val).mpr hi)
  have hy : ∀ (i : Fin d.BT) (j : Fin d.BV), t * d.BV + j.val < d.V →
      (((ovOf d i_v).row d.BT).get2 i j).toInt = (((t * d.BV + j.val) * 1 : Nat) : Int) := by
    intro i j hj
    rw [mul_one]; exact ovOf_lane hd ht hiv j
  have hp := Tile2.ptr_lane_seq_of_on d.dO S.do_eq
    (ρ := fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (κ := fun j : Fin d.BV => t * d.BV + j.val)
    (M := lenOf d r bh) (N := d.V) hx hy
  have hm : Tile2.MaskAt (some (((o_t.ltS T).col d.BV).and
      (((ovOf d i_v).ltS (IntV.ofNat d.V : I32)).row d.BT)))
      (fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (fun j : Fin d.BV => t * d.BV + j.val)
      (lenOf d r bh) d.V := by
    refine Tile2.maskAt_of_get fun i j => ?_
    show ((o_t.ltS T).get1 i && ((ovOf d i_v).ltS (IntV.ofNat d.V : I32)).get1 j) = _
    rw [mt_get S i, ovMask_lane hd ht hiv j]
    rfl
  have hkt := o_tile_lt hd hT hr hbh
  have hlen : d.B * d.T * d.HV * d.V ≤ d.dO.len := hd.dO_len
  have hlen' : lenOf d r bh = 0 ∨ d.V = 0 ∨
      oOff d r bh + (lenOf d r bh - 1) * (d.HV * d.V) + (d.V - 1) * 1 < d.dO.len :=
    Or.inr (Or.inr (by omega))
  have hRdO : d.dO.bytes ⊆ R d r bh := by
    intro x hx'; unfold R; simp only [Set.mem_union]; tauto
  have hdOv : d.dO.Valid m :=
    Buf.valid_of_agree hpre.2.2.1 fun x hx' => by
      rw [hag x (hRdO hx')]; exact hpre.2.2.1.2.2.2 x hx'
  refine safe_load_tileAt d.dO hp hm hlen' hdOv ?_ ?_
  · intro x hx'
    exact hRdO (Buf.tileAt_sub_bytes d.dO hlen' hx')
  · intro v hv1 hv2
    refine hcont v fun l kk => ?_
    by_cases hl : en d r bh l = true
    · by_cases hkk : t * d.BV + kk.val < d.V
      · have hread := hv1 l kk ((en_iff r bh l.val).mp hl) hkk
        have hidx : oOff d r bh + (chkOf d r bh * d.BT + l.val) * (d.HV * d.V) + (t * d.BV + kk.val) * 1 =
            oIdx d (pos d r bh l) (hvOf d bh) (t * d.BV + kk.val) := by
          unfold oOff oIdx pos baseOf; ring
        rw [hidx] at hread
        have hread' : init.read .bf16 (d.dO.ptr (oIdx d (pos d r bh l) (hvOf d bh) (t * d.BV + kk.val) : Nat)).addr =
            some (v.get2 l kk) := by
          rw [← hread]
          symm
          refine Mem.read_congr fun x hx' => (hag x (hRdO ?_))
          exact Buf.span_sub d.dO (by
            have := oIdx_lt (d := d) (tok := pos d r bh l) (hv := hvOf d bh) (e := t * d.BV + kk.val)
              (pos_lt hd hT hr hbh hl) (hvOf_lt hd bh) hkk
            omega) hx'
        obtain ⟨-, hdo, -⟩ := vals_tok hd hT hV hr hbh hl
        obtain ⟨⟨xv, hxv, hxfin⟩, -⟩ := hdo (hvOf d bh) (t * d.BV + kk.val) (hvOf_lt hd bh) hkk
        unfold rdB at hxv
        have hvx : v.get2 l kk = xv := Option.some.inj (hread'.symm.trans hxv)
        rw [ite_eq_left ⟨hl, hkk⟩]
        refine ⟨by rw [hvx]; exact hxfin, ?_⟩
        unfold valB rdB
        rw [hxv]
        show (v.get2 l kk).toRat = xv.toRat
        rw [hvx]
      · have hz0 : v.get2 l kk = FloatV.ofPat 0 := by
          have := hv2 l kk (fun h => hkk h.2) z rfl
          rw [this]; exact hz (l, kk, ())
        rw [ite_eq_right (fun h => hkk h.2), hz0]
        exact ⟨(FloatV.isFinite_ofPat_zero Format.bfloat16_regular), FloatV.toRat_ofPat_zero⟩
    · have hz0 : v.get2 l kk = FloatV.ofPat 0 := by
        have := hv2 l kk (fun h => hl ((en_iff r bh l.val).mpr h.1)) z rfl
        rw [this]; exact hz (l, kk, ())
      rw [ite_eq_right (fun h => hl h.1), hz0]
      exact ⟨(FloatV.isFinite_ofPat_zero Format.bfloat16_regular), FloatV.toRat_ofPat_zero⟩

/-- The elements of `dv` that trip `t` of the `V` loop writes: the enabled lanes in the columns
`[t·BV, (t+1)·BV)` below `V`. -/
def tileSet (d : Inst) (r bh t : Nat) : Set Nat :=
  {x | ∃ l e, l < d.BT ∧ en d r bh l = true ∧ e < d.V ∧ t * d.BV ≤ e ∧ e < (t + 1) * d.BV ∧
    x = wEl d r bh l e}

/-- Trip `t`'s elements are among the program's. -/
theorem tileSet_sub (d : Inst) (r bh t : Nat) : tileSet d r bh t ⊆ wSet d r bh := by
  rintro _ ⟨l, e, hl, hen, hev, -, -, hx⟩
  exact ⟨l, e, hl, hen, hev, hx⟩

/-- The columns of different trips are different elements. -/
theorem tileSet_disjoint {d : Inst} (hd : d.ok) (r bh : Nat) {t u : Nat} (htu : t < u) :
    Disjoint (tileSet d r bh t) (tileSet d r bh u) := by
  rw [Set.disjoint_left]
  rintro x ⟨l, e, hl, hen, hev, h1, h2, rfl⟩ ⟨l', e', hl', hen', hev', h3, h4, h⟩
  obtain ⟨-, -, he⟩ := oIdx_inj (hvOf_lt hd bh) (hvOf_lt hd bh) hev hev' (by unfold wEl at h; exact h)
  have : (t + 1) * d.BV ≤ u * d.BV := Nat.mul_le_mul_right _ htu
  omega

set_option maxHeartbeats 1600000 in
/-- **One trip of the `V` loop** is safe: it reads within `R`, writes the elements `tileSet t` of
`dv`, and leaves `Agree`, the validity of `dv` and `QCols` for one more block of columns. -/
theorem safe_vBody {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) {r bh : Nat} (hr : r < nChunks d) (hbh : bh < nRows d)
    {q k do_ dv : Ptr .bf16} {gb : Ptr .f32} {o_t : Tensor [d.BT] I64} {T : I64}
    (S : Setup d r bh q k do_ dv gb o_t T) {A : Tensor [d.BT, d.BT] BF16}
    (hA : AOk d init r bh A) {t : Nat} (ht : t * d.BV < d.V) {i_v : I32}
    (hiv : i_v.toInt = (t : Int)) {m : Mem} (hag : Agree d init r bh m) (hdv : d.dv.Valid m)
    (hQ : QCols d init r bh (t * d.BV) m) :
    Safe ((launch d).env P) (R d r bh) (d.dv.elems (tileSet d r bh t))
      (fun _ m' => Agree d init r bh m' ∧ d.dv.Valid m' ∧ QCols d init r bh ((t + 1) * d.BV) m') m
      (vBody d do_ dv o_t (o_t.ltS T) A i_v ()) := by
  have hT : d.varlen = true → TablesOk d := tables_of_pre hpre
  have hV : ValsOk d init := hpre.2.2.2.2.2.2
  unfold vBody
  refine safe_vload_do hd hpre hr hbh S ht hiv hag fun bdo hbdo => ?_
  refine safe_dot (hP.dotTotal cfgBF16 A bdo _) fun X hX => ?_
  refine safe_cvt ⟨Format.binary32_std, Format.bfloat16_std, Or.inl rfl⟩ fun o16 ho16 => ?_
  refine safe_last ?_
  -- the store's tile
  have hx : ∀ (i : Fin d.BT) (j : Fin d.BV), chkOf d r bh * d.BT + i.val < lenOf d r bh →
      (((o_t * (IntV.ofNat (d.HV * d.V) : I64)).col d.BV).get2 i j).toInt =
        (((chkOf d r bh * d.BT + i.val) * (d.HV * d.V) : Nat) : Int) :=
    fun i j hi => otHV_lane hd hT hpre hr hbh S ((en_iff r bh i.val).mpr hi)
  have hy : ∀ (i : Fin d.BT) (j : Fin d.BV), t * d.BV + j.val < d.V →
      (((ovOf d i_v).row d.BT).get2 i j).toInt = (((t * d.BV + j.val) * 1 : Nat) : Int) := by
    intro i j hj
    rw [mul_one]; exact ovOf_lane hd ht hiv j
  have hp := Tile2.ptr_lane_seq_of_on d.dv S.dv_eq
    (ρ := fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (κ := fun j : Fin d.BV => t * d.BV + j.val)
    (M := lenOf d r bh) (N := d.V) hx hy
  have hm : Tile2.MaskAt (some (((o_t.ltS T).col d.BV).and
      (((ovOf d i_v).ltS (IntV.ofNat d.V : I32)).row d.BT)))
      (fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (fun j : Fin d.BV => t * d.BV + j.val)
      (lenOf d r bh) d.V := by
    refine Tile2.maskAt_of_get fun i j => ?_
    show ((o_t.ltS T).get1 i && ((ovOf d i_v).ltS (IntV.ofNat d.V : I32)).get1 j) = _
    rw [mt_get S i, ovMask_lane hd ht hiv j]
    rfl
  have hkt := o_tile_lt hd hT hr hbh
  have hlen : d.B * d.T * d.HV * d.V ≤ d.dv.len := hd.dv_len
  have hlen' : lenOf d r bh = 0 ∨ d.V = 0 ∨
      oOff d r bh + (lenOf d r bh - 1) * (d.HV * d.V) + (d.V - 1) * 1 < d.dv.len :=
    Or.inr (Or.inr (by omega))
  have hinj : Tile2.Inj (d.HV * d.V) 1 (lenOf d r bh) d.V :=
    Tile2.inj_rowMajor1 (Nat.le_mul_of_pos_left _ hd.HV_pos)
  have hW : d.dv.tileAt (oOff d r bh) (d.HV * d.V) 1
      (fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (fun j : Fin d.BV => t * d.BV + j.val)
      (lenOf d r bh) d.V ⊆ d.dv.elems (tileSet d r bh t) := by
    refine Buf.elems_mono d.dv ?_
    rintro x ⟨i, j, hi, hj, rfl⟩
    refine ⟨i.val, t * d.BV + j.val, i.isLt, (en_iff r bh i.val).mpr hi, hj, by omega, by
      have := j.isLt; have h2 : (t + 1) * d.BV = t * d.BV + d.BV := by ring
      omega, ?_⟩
    unfold wEl oIdx oOff pos baseOf
    ring
  refine safe_store_tileAt d.dv hp hm hlen' hinj
    (fun i i' _ _ h => Fin.ext (by omega)) (fun j j' _ _ h => Fin.ext (by omega)) hdv hW ?_
  intro m' hread hframe hpres
  refine safe_ret ⟨?_, ?_, ?_⟩
  · -- the inputs are untouched
    intro x hx'
    have hnot : x ∉ d.dv.tileAt (oOff d r bh) (d.HV * d.V) 1
        (fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (fun j : Fin d.BV => t * d.BV + j.val)
        (lenOf d r bh) d.V := by
      intro hin
      have := Buf.tileAt_sub_bytes d.dv hlen' hin
      exact Set.disjoint_left.mp (R_disjoint_dv hd r bh) hx' this
    rw [hframe x hnot]; exact hag x hx'
  · exact Buf.valid_of_agree hdv fun x hx' => hpres x (hdv.2.2.2 x hx')
  · intro l hl hen e he hev
    by_cases hold : e < t * d.BV
    · -- an earlier column
      obtain ⟨c, hc, hfin, herr⟩ := hQ l hl hen e hold hev
      refine ⟨c, ?_, hfin, herr⟩
      rw [← hc]
      unfold rdB
      refine Mem.read_congr fun z hz => hframe z ?_
      intro hin
      have hsp := Buf.span_sub_elems d.dv (S := {wEl d r bh l e}) (j := wEl d r bh l e) rfl
      have hdisj : Disjoint (d.dv.elems {wEl d r bh l e}) (d.dv.tileAt (oOff d r bh) (d.HV * d.V) 1
          (fun i : Fin d.BT => chkOf d r bh * d.BT + i.val) (fun j : Fin d.BV => t * d.BV + j.val)
          (lenOf d r bh) d.V) := by
        refine Buf.elems_disjoint d.dv (Set.disjoint_left.mpr ?_)
        rintro y (rfl : y = wEl d r bh l e) ⟨i, j, hi, hj, h⟩
        have h' : oIdx d (pos d r bh l) (hvOf d bh) e =
            oIdx d (pos d r bh i.val) (hvOf d bh) (t * d.BV + j.val) := by
          unfold wEl at h
          rw [h]; unfold oIdx oOff pos baseOf; ring
        obtain ⟨-, -, he'⟩ := oIdx_inj (hvOf_lt hd bh) (hvOf_lt hd bh) hev hj h'
        have he'' : e = t * d.BV + j.val := he'
        omega
      exact Set.disjoint_left.mp hdisj (hsp hz) hin
    · -- a column of this block
      have hj : e - t * d.BV < d.BV := by
        have h2 : (t + 1) * d.BV = t * d.BV + d.BV := by ring
        omega
      let j : Fin d.BV := ⟨e - t * d.BV, hj⟩
      let lf : Fin d.BT := ⟨l, hl⟩
      have hidx : oOff d r bh + (chkOf d r bh * d.BT + l) * (d.HV * d.V) + (t * d.BV + (e - t * d.BV)) * 1 =
          wEl d r bh l e := by
        unfold wEl oIdx oOff pos baseOf
        have : t * d.BV + (e - t * d.BV) = e := by omega
        rw [this]; ring
      have hread' : m'.read .bf16 (d.dv.ptr ((oOff d r bh + (chkOf d r bh * d.BT + l) * (d.HV * d.V) +
          (t * d.BV + (e - t * d.BV)) * 1 : Nat) : Int)).addr = some (o16.get2 lf j) :=
        hread lf j ((en_iff r bh l).mp hen) (by show t * d.BV + (e - t * d.BV) < d.V; omega)
      rw [hidx] at hread'
      have hjV : t * d.BV + j.val < d.V := by show t * d.BV + (e - t * d.BV) < d.V; omega
      obtain ⟨hfin, herr⟩ := vlane hP hd hpre hr hbh hA hbdo hX (a := lf) hen hjV (ho16 (lf, j, ()))
      have hee : t * d.BV + j.val = e := by show t * d.BV + (e - t * d.BV) = e; omega
      rw [hee] at herr
      exact ⟨o16.get2 lf j, hread', hfin, herr⟩

end Claims.ChunkBwdDvLocal
