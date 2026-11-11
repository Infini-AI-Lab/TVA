import Claim
import TVA.Proof
import Proof.Epi
import Proof.LoopSafe
import Proof.Gate

/-!
# The tail of the kernel

`safe_tail2`: lines 129–181 from the offset pointers and the scalars: the token and column offsets,
the loop over `K`, the gates and the epilogue.
-/

open TVA
namespace Claims.ChunkFwdO

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}
variable {d : Inst} {g : Geo} {r : Nat} {init : Mem}

/-- `tl.arange(0, n).to(tl.int64)` (lines 132, 134): lane `i` has value `i`. -/
theorem safe_conv_arange64 {n : Nat} (hn : n ≤ 2 ^ 31) {k : Tensor [n - 0] I64 → Prog α}
    (hk : ∀ x : Tensor [n - 0] I64, (∀ i : Fin (n - 0), (x.get1 i).toInt = ((i : Nat) : Int)) →
      Safe E R W Q m (k x)) :
    Safe E R W Q m (tl.convert .i32 .i64 (tl.arange 0 n) >>= k) :=
  safe_convert (fun i => ⟨IntV.cast ((tl.arange 0 n).get i), ConvRel.iff_cast_i32_i64.2 rfl⟩)
    fun x hx => hk x (Tile2.arange64_toInt hx hn)

/-- **Lines 129–181.** -/
theorem safe_tail2 (hP : Device E.P) (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    (hm : m = init) {q k v h o : Ptr .bf16} {i_v i_h i_t bos T : I64}
    (hq : q = d.q.ptr (((g.bos * d.H + qh d g.ih) * d.K : Nat) : Int))
    (hk : k = d.k.ptr (((g.bos * d.H + qh d g.ih) * d.K : Nat) : Int))
    (hv : v = d.v.ptr (((g.bos * d.HV + g.ih) * d.V : Nat) : Int))
    (ho : o = d.o.ptr (((g.bos * d.HV + g.ih) * d.V : Nat) : Int))
    (hh : h = d.h.ptr (((g.itg * d.HV + g.ih) * d.K * d.V : Nat) : Int))
    (hiv : i_v.toInt = (g.iv : Int)) (hih : i_h.toInt = (g.ih : Int))
    (hit : i_t.toInt = (g.it : Int)) (hbos : bos.toInt = (g.bos : Int))
    (hT : T.toInt = (g.Tn : Int)) :
    Safe E (Rs d) (Wg d g) (fun _ m' => Qg d g r m') m (tail2 d q k v h o i_v i_h i_t bos T) := by
  subst hm
  rw [tail2_eq]
  simp only [tail2s, Prog.bind_eq]
  have hBT : BT ≤ 2 ^ 31 := by norm_num [BT]
  have hbk : d.bk ≤ 2 ^ 31 := by have := bk_le hd; omega
  refine safe_conv_arange64 hBT fun xr hxr => ?_
  refine safe_conv_arange64 hbk fun xv hxv => ?_
  have hBTv : (IntV.ofNat BT : I64).toInt = (BT : Int) := IntV.toInt_ofNat (w := 64) (by norm_num [BT])
  have hbkv : (IntV.ofNat d.bk : I64).toInt = (d.bk : Int) :=
    IntV.toInt_ofNat (w := 64) (by have := bk_le hd; omega)
  have hits : (i_t * (IntV.ofNat BT : I64)).toInt = ((g.it * BT : Nat) : Int) := by
    refine IntV.mul_nat hit hBTv ?_
    have h1 := hg.start_lt; have h2 := hg.end_le; have h3 := BT_le hd
    simp only [BT] at *
    omega
  have hvs : (i_v * (IntV.ofNat d.bk : I64)).toInt = ((g.iv * d.bk : Nat) : Int) := by
    refine IntV.mul_nat hiv hbkv ?_
    have h1 := hg.iv_lt; have h2 := hd.V_le
    have : g.iv * d.bk < d.V := h1
    omega
  have hot : ∀ i : Fin BT, ((bc (i_t * (IntV.ofNat BT : I64)) + xr).get1 i).toInt =
      ((g.it * BT + i : Nat) : Int) := by
    intro i
    show ((i_t * (IntV.ofNat BT : I64)) + xr.get1 i).toInt = _
    have h1 := hg.start_lt; have h2 := hg.end_le; have h3 := BT_le hd
    have h4 := i.isLt
    refine IntV.add_nat hits (by simpa using hxr i) ?_
    simp only [BT] at *
    omega
  have hov : ∀ c : Fin d.bk, ((bc (i_v * (IntV.ofNat d.bk : I64)) + xv).get1 c).toInt =
      ((g.iv * d.bk + c : Nat) : Int) := by
    intro c
    show ((i_v * (IntV.ofNat d.bk : I64)) + xv.get1 c).toInt = _
    have h1 : g.iv * d.bk < d.V := hg.iv_lt
    have h2 := hd.V_le
    have h4 := c.isLt
    refine IntV.add_nat hvs (by simpa using hxv c) ?_
    have := bk_le hd
    omega
  have hl : TailLanes d g q k v h o (bc (i_t * (IntV.ofNat BT : I64)) + xr) T
      (bc (i_v * (IntV.ofNat d.bk : I64)) + xv) := ⟨hq, hk, hv, ho, hh, hot, hT, hov⟩
  refine loop_safe hP hd hpre hg hl fun st hO hA => ?_
  obtain ⟨bo, bA⟩ := st
  refine safe_gateG hP hd hpre hg hl rfl hih hbos hO hA fun bo' bA' hgt => ?_
  exact safe_gateGamma hP hd hpre hg rfl hih hgt fun bo'' bA'' hOG hAG =>
    safe_epilogue hP hd hpre hg hl rfl hOG hAG

/-- A pointer to element `0` plus an `int64` offset of value `n` is the pointer to element `n`. -/
theorem ptr_add_val {dt : DType} (b : Buf dt) {sg : Bool} {w : Nat} {X : IntV sg w} {n : Nat}
    (hX : X.toInt = (n : Int)) :
    b.ptr 0 + X = b.ptr (n : Int) := by
  show (b.ptr 0).offset X.toInt = _
  rw [Buf.ptr_offset, hX]; simp

/-- The same with a second offset added after the first. -/
theorem ptr_add_val2 {dt : DType} (b : Buf dt) {sg : Bool} {w : Nat} {X : IntV sg w}
    {sg' : Bool} {w' : Nat} {Y : IntV sg' w'} {n m : Nat} (hX : X.toInt = (n : Int))
    (hY : Y.toInt = (m : Int)) : b.ptr 0 + X + Y = b.ptr ((n + m : Nat) : Int) := by
  show ((b.ptr 0).offset X.toInt).offset Y.toInt = _
  rw [Buf.ptr_offset, Buf.ptr_offset, hX, hY]; simp

/-- The value of an `int64` constant. -/
theorem ofNat64 {n : Nat} (hn : n < 2 ^ 63) : (IntV.ofNat n : I64).toInt = (n : Int) :=
  IntV.toInt_ofNat (w := 64) hn

/-- The value of an `int64` built from an integer. -/
theorem ofInt64 {n : Nat} (hn : n < 2 ^ 63) : (IntV.ofInt (n : Int) : I64).toInt = (n : Int) :=
  IntV.toInt_ofInt (w := 64) (by norm_num) (sfits_nat (by norm_num) (by exact_mod_cast hn))

/-- **Lines 123–181**: the five offset pointers, then `tail2`. -/
theorem safe_tail1 (hP : Device E.P) (hd : d.ok) (hpre : Pre d init) (hg : Geo.Ok d g r)
    (hm : m = init) {i_v i_h i_tg i_t bos T : I64}
    (hiv : i_v.toInt = (g.iv : Int)) (hih : i_h.toInt = (g.ih : Int))
    (hitg : i_tg.toInt = (g.itg : Int)) (hit : i_t.toInt = (g.it : Int))
    (hbos : bos.toInt = (g.bos : Int)) (hT : T.toInt = (g.Tn : Int)) :
    Safe E (Rs d) (Wg d g) (fun _ m' => Qg d g r m') m (tail1 d i_v i_h i_tg i_t bos T) := by
  have hH := H_lt16 hd
  have hHV := HV_lt16 hd
  have hBT := BT_le hd
  have hend := hg.end_le
  have hKle := hd.K_le
  have hVle := hd.V_le
  have hqh := qh_lt hd hg.ih_lt
  have hdv : 0 < d.HV / d.H := by
    obtain ⟨c, hc⟩ := hd.hv_dvd
    have := hd.H_pos
    have : d.HV / d.H = c := by rw [hc]; exact Nat.mul_div_cancel_left c this
    rw [this]
    rcases Nat.eq_zero_or_pos c with h | h
    · have := hd.HV_pos; rw [hc, h] at this; simp at this
    · exact h
  have hdv63 : d.HV / d.H < 2 ^ 63 := lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)
  have hbH : (bos * (IntV.ofNat d.H : I64)).toInt = ((g.bos * d.H : Nat) : Int) :=
    IntV.mul_nat hbos (ofNat64 (by omega)) (by
      have : g.bos * d.H ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) (by omega)
      omega)
  have hqoff : ∀ q0 : Nat, q0 < d.H →
      ((bos * (IntV.ofNat d.H : I64) + (IntV.ofInt (q0 : Int) : I64)) * (IntV.ofNat d.K : I64)).toInt =
        (((g.bos * d.H + q0) * d.K : Nat) : Int) := by
    intro q0 hq0
    have h1 : g.bos * d.H ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) (by omega)
    have h2 : (g.bos * d.H + q0) * d.K ≤ (2 ^ 35 * 2 ^ 16 + 2 ^ 16) * 256 :=
      Nat.mul_le_mul (by omega) (by omega)
    refine IntV.mul_nat (IntV.add_nat hbH (ofInt64 (by omega)) (by omega)) (ofNat64 (by omega)) ?_
    omega
  simp only [tail1, Prog.bind_eq]
  refine safe_idiv_val (w := 64) g.ih (d.HV / d.H) (by norm_num) hih (ofNat64 hdv63) hdv
    (by have : g.ih / (d.HV / d.H) ≤ g.ih := Nat.div_le_self _ _
        have := hg.ih_lt; omega) ?_
  refine safe_idiv_val (w := 64) g.ih (d.HV / d.H) (by norm_num) hih (ofNat64 hdv63) hdv
    (by have : g.ih / (d.HV / d.H) ≤ g.ih := Nat.div_le_self _ _
        have := hg.ih_lt; omega) ?_
  have hbHV : (bos * (IntV.ofNat d.HV : I64)).toInt = ((g.bos * d.HV : Nat) : Int) :=
    IntV.mul_nat hbos (ofNat64 (by omega)) (by
      have : g.bos * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) (by omega)
      omega)
  have hvoff : ((bos * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.V : I64)).toInt =
      (((g.bos * d.HV + g.ih) * d.V : Nat) : Int) := by
    have h1 : g.bos * d.HV ≤ 2 ^ 35 * 2 ^ 16 := Nat.mul_le_mul (by omega) (by omega)
    have h2 : (g.bos * d.HV + g.ih) * d.V ≤ (2 ^ 35 * 2 ^ 16 + 2 ^ 16) * 512 :=
      Nat.mul_le_mul (by have := hg.ih_lt; omega) (by omega)
    refine IntV.mul_nat (IntV.add_nat hbHV hih (by have := hg.ih_lt; omega)) (ofNat64 (by omega)) ?_
    omega
  have hhoff : ((i_tg * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.K : I64) *
      (IntV.ofNat d.V : I64)).toInt = (((g.itg * d.HV + g.ih) * d.K * d.V : Nat) : Int) := by
    have hrow := state_row_lt hg
    have hlen := hd.h_len
    have hval := hpre.h_valid.2.1
    have hb2 : (d.h.len : Int) * 2 ≤ 2 ^ 64 := by
      have := hpre.h_valid.1
      have h3 : d.h.base + (d.h.len : Int) * 2 ≤ 2 ^ 64 := hval
      omega
    have hlt : (g.itg * d.HV + g.ih) * d.K * d.V < d.h.len := by
      have h1 : (g.itg * d.HV + g.ih + 1) * d.K * d.V ≤ nRows d * d.HV * d.K * d.V :=
        Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hrow)
      have h2 : (g.itg * d.HV + g.ih + 1) * d.K * d.V =
          (g.itg * d.HV + g.ih) * d.K * d.V + d.K * d.V := by ring
      have := Nat.mul_pos hd.K_pos hd.V_pos
      omega
    have hlt63 : (g.itg * d.HV + g.ih) * d.K * d.V < 2 ^ 63 := by
      have : (d.h.len : Int) ≤ 2 ^ 63 := by omega
      have : d.h.len ≤ 2 ^ 63 := by exact_mod_cast this
      omega
    have hm1 : (g.itg * d.HV + g.ih) * d.K ≤ (g.itg * d.HV + g.ih) * d.K * d.V := by
      exact Nat.le_mul_of_pos_right _ hd.V_pos
    have hbi : (i_tg * (IntV.ofNat d.HV : I64)).toInt = ((g.itg * d.HV : Nat) : Int) :=
      IntV.mul_nat hitg (ofNat64 (by omega)) (by
        have : g.itg * d.HV ≤ (g.itg * d.HV + g.ih) * d.K * d.V := by
          calc g.itg * d.HV ≤ (g.itg * d.HV + g.ih) := Nat.le_add_right _ _
            _ ≤ (g.itg * d.HV + g.ih) * d.K := Nat.le_mul_of_pos_right _ hd.K_pos
            _ ≤ _ := hm1
        omega)
    have hbs : (i_tg * (IntV.ofNat d.HV : I64) + i_h).toInt = ((g.itg * d.HV + g.ih : Nat) : Int) :=
      IntV.add_nat hbi hih (by
        have : g.itg * d.HV + g.ih ≤ (g.itg * d.HV + g.ih) * d.K * d.V := by
          calc _ ≤ (g.itg * d.HV + g.ih) * d.K := Nat.le_mul_of_pos_right _ hd.K_pos
            _ ≤ _ := hm1
        omega)
    have hbk : ((i_tg * (IntV.ofNat d.HV : I64) + i_h) * (IntV.ofNat d.K : I64)).toInt =
        (((g.itg * d.HV + g.ih) * d.K : Nat) : Int) :=
      IntV.mul_nat hbs (ofNat64 (by omega)) (by omega)
    exact IntV.mul_nat hbk (ofNat64 (by omega)) hlt63
  exact safe_tail2 hP hd hpre hg hm (ptr_add_val d.q (hqoff _ hqh)) (ptr_add_val d.k (hqoff _ hqh))
    (ptr_add_val d.v hvoff) (ptr_add_val d.o hvoff) (ptr_add_val d.h hhoff) hiv hih hit hbos hT

end Claims.ChunkFwdO
