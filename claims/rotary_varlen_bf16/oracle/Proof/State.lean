import Proof.Basic
import Proof.Meta

/-!
# The prologue of one program

The state a program's context gives its body (`i_t`, `x_h`, `y_h`, `T_n`, `i_n`), and that the
prologue computes it without writing memory.
-/

namespace Claims.RotaryVarlen

open TVA

private theorem convRel_idx32 (P : Platform) (dt : DType) (h : dt = .i32 ∨ dt = .i64)
    (x : dt.Val) : ConvRel P dt .i32 x (IntV.ofInt (idxVal dt x)) := by
  rcases h with h | h <;> subst h <;> simp [ConvRel, ofIntRel, idxVal]

private theorem convRel_idx64 (P : Platform) (dt : DType) (h : dt = .i32 ∨ dt = .i64)
    (x : dt.Val) : ConvRel P dt .i64 x (IntV.ofInt (idxVal dt x)) := by
  rcases h with h | h <;> subst h <;> simp [ConvRel, ofIntRel, idxVal]

private theorem convRel_i32_eq {P : Platform} {dt : DType} (h : dt = .i32 ∨ dt = .i64)
    {x : dt.Val} {r : I32} (hr : ConvRel P dt .i32 x r) : r = IntV.ofInt (idxVal dt x) := by
  rcases h with h | h <;> subst h <;> simpa [ConvRel, ofIntRel, idxVal] using hr

private theorem convRel_i64_eq {P : Platform} {dt : DType} (h : dt = .i32 ∨ dt = .i64)
    {x : dt.Val} {r : I64} (hr : ConvRel P dt .i64 x r) : r = IntV.ofInt (idxVal dt x) := by
  rcases h with h | h <;> subst h <;> simpa [ConvRel, ofIntRel, idxVal] using hr

private theorem convRel_item_i32 {P : Platform} {dt : DType} (h : dt = .i32 ∨ dt = .i64)
    {raw : dt.Val} {r : Tensor [] I32} (hr : ConvRel P dt .i32 raw (r.get ())) :
    r.item = (IntV.ofInt (idxVal dt raw) : I32) := by
  have h1 : r.get () = (IntV.ofInt (idxVal dt raw) : I32) := convRel_i32_eq h hr
  simpa [Tensor.item] using h1

private theorem convRel_item_i64 {P : Platform} {dt : DType} (h : dt = .i32 ∨ dt = .i64)
    {raw : dt.Val} {r : Tensor [] I64} (hr : ConvRel P dt .i64 raw (r.get ())) :
    r.item = (IntV.ofInt (idxVal dt raw) : I64) := by
  have h1 : r.get () = (IntV.ofInt (idxVal dt raw) : I64) := convRel_i64_eq h hr
  simpa [Tensor.item] using h1

private theorem exists_read_of_idxAt {dt : DType} {b : Buf dt} {m : Mem} {j : Nat} {v : Int}
    (h : idxAt b m j = some v) :
    ∃ raw : dt.Val, m.read dt (b.ptr (j : Int)).addr = some raw ∧ idxVal dt raw = v := by
  simpa only [idxAt, Option.map_eq_some_iff, Option.some.injEq] using h

private theorem sfits64_idxVal {dt : DType} (h : dt = .i32 ∨ dt = .i64) (x : dt.Val) :
    IntV.SFits 64 (idxVal dt x) := by
  rcases h with h | h
  · subst h
    have hx := IntV.sfits_toInt x
    simp only [idxVal, IntV.SFits] at hx ⊢
    constructor <;> omega
  · subst h
    simpa only [idxVal, IntV.SFits] using IntV.sfits_toInt x

private theorem subset_R_ci {d : Inst} {init : Mem} {i : Nat} : d.ci.bytes ⊆ R d init i := by
  intro z hz; simp only [R, Set.mem_union]; tauto

private theorem subset_R_cu {d : Inst} {init : Mem} {i : Nat} : d.cu.bytes ⊆ R d init i := by
  intro z hz; simp only [R, Set.mem_union]; tauto

private theorem ptr_off1 {dt : DType} {sg : Bool} {w : Nat} (b : Buf dt) (k : IntV sg w) (t : Int)
    (h : k.toInt = t) : (b.ptr 0 + k).addr = (b.ptr t).addr := by
  have hidx : (b.ptr 0 + k).addr = b.base + k.toInt * ↑dt.bytes := by
    simp only [Buf.ptr_addr, Ptr.add_int, Ptr.offset_addr]; ring
  rw [hidx, h]
  simp only [Buf.ptr_addr]

private theorem ptr_off2 {dt : DType} {sg : Bool} {w : Nat} {sg' : Bool} {w' : Nat}
    (b : Buf dt) (k : IntV sg w) (k' : IntV sg' w')
    (t : Int) (h : k.toInt + k'.toInt = t) : (b.ptr 0 + k + k').addr = (b.ptr t).addr := by
  have hidx : (b.ptr 0 + k + k').addr = b.base + (k.toInt + k'.toInt) * ↑dt.bytes := by
    simp only [Buf.ptr_addr, Ptr.add_int, Ptr.offset_addr]; ring
  rw [hidx, h]
  simp only [Buf.ptr_addr]

private theorem i32_ofInt_val {z : Int} (h0 : 0 ≤ z) (hlt : z < 2 ^ 31) :
    ((IntV.ofInt z : I32)).toInt = z :=
  IntV.I32.toInt_ofInt ⟨by omega, hlt⟩

private theorem i64_ofInt_val {z : Int} (h0 : 0 ≤ z) (hlt : z < 2 ^ 31) :
    ((IntV.ofInt z : I64)).toInt = z := by
  refine IntV.toInt_ofInt (w := 64) (by norm_num) ?_
  simp only [IntV.SFits]; constructor <;> omega

private theorem i64_ofNat_val {n : Nat} (h : n < 2 ^ 31) :
    ((IntV.ofNat n : I64)).toInt = (n : Int) := by
  refine ofNat_toInt (w := 64)  ?_
  have h' : n < 2 ^ 63 := lt_trans h (by norm_num)
  exact_mod_cast h'

private theorem i32_ofNat_val {n : Nat} (h : n < 2 ^ 31) :
    ((IntV.ofNat n : I32)).toInt = (n : Int) := IntV.I32.toInt_ofNat h

private theorem i64_of_cast32 {n : Nat} (h : n < 2 ^ 31) :
    ((IntV.cast (IntV.ofNat n : I32) : I64)).toInt = (n : Int) := by
  rw [IntV.toInt_cast (a := (IntV.ofNat n : I32)) (w' := 64) (by norm_num)]
  exact IntV.I32.toInt_ofNat h

private theorem toInt_mul_of {w : Nat} {a b : IntV true w} {x y : Int}
    (ha : a.toInt = x) (hb : b.toInt = y) (_hw : 0 < w) (hsf : IntV.SFits w (x * y)) :
    (a * b).toInt = x * y := by
  have h : IntV.SFits w (a.toInt * b.toInt) := by rw [ha, hb]; exact hsf
  rw [IntV.toInt_mul rfl rfl h, ha, hb]

private theorem i64_sub_val {a b : Int} (hb0 : 0 ≤ b) (hba : b ≤ a) (ha : a < 2 ^ 31) :
    ((IntV.ofInt a : I64) - (IntV.ofInt b : I64)).toInt = a - b := by
  have ha0 : (0:Int) ≤ a := le_trans hb0 hba
  have ha64 : (IntV.ofInt a : I64).toInt = a := i64_ofInt_val ha0 ha
  have hb64 : (IntV.ofInt b : I64).toInt = b := i64_ofInt_val hb0 (by omega)
  have hsf : IntV.SFits 64 ((IntV.ofInt a : I64).toInt - (IntV.ofInt b : I64).toInt) := by
    rw [ha64, hb64]
    simp only [IntV.SFits]; constructor <;> omega
  rw [IntV.toInt_sub rfl rfl hsf, ha64, hb64]

private theorem i64_sub_of {x y : I64} {a b : Int} (hx : x.toInt = a) (hy : y.toInt = b)
    (hsf : IntV.SFits 64 (a - b)) : (x - y).toInt = a - b := by
  have h : IntV.SFits 64 (x.toInt - y.toInt) := by rw [hx, hy]; exact hsf
  rw [IntV.toInt_sub rfl rfl h, hx, hy]

private theorem natMul2_lt {a b : Nat} (ha : a < 2 ^ 31) (hb : b < 2 ^ 16) : a * b < 2 ^ 47 := by
  have h := Nat.mul_lt_mul'' ha hb
  have h3 : (2:Nat)^31 * 2^16 = 2^47 := by norm_num
  omega

private theorem int_mul_lt47 {a b : Nat} (ha : a < 2 ^ 31) (hb : b < 2 ^ 16) :
    (a : Int) * (b : Int) < 2 ^ 47 := by
  have h : ((a * b : Nat) : Int) < (2 ^ 47 : Int) := by
    exact_mod_cast natMul2_lt ha hb
  push_cast at h ⊢
  linarith

private theorem sfits64_of {x : Int} (h0 : 0 ≤ x) (hx : x < 2 ^ 62) : IntV.SFits 64 x := by
  simp only [IntV.SFits]; constructor <;> omega

private theorem sfits32_of {x : Int} (h0 : 0 ≤ x) (hx : x < 2 ^ 31) : IntV.SFits 32 x := by
  simp only [IntV.SFits]; constructor <;> omega

private theorem ptr_varlen (buf : Buf .bf16) (d : Inst) (hd : d.ok)
    {init : Mem} {i : Nat} (hi : i < (launch d).size) {k : I64}
    (hk : k.toInt = (bosI d init i : Int)) (hb : bosI d init i ≤ d.T) :
    (buf.ptr 0 + k * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64)
        + (IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)).addr
      = (buf.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr := by
  have hbI0 : (0:Int) ≤ (bosI d init i : Int) := by positivity
  have hH0 : (0:Int) ≤ (d.H : Int) := by positivity
  have hHI : (d.H : Int) < 2 ^ 16 := by exact_mod_cast hd.grid_h
  have hD0 : (0:Int) ≤ (d.D : Int) := by positivity
  have hDI : (d.D : Int) ≤ 256 := by exact_mod_cast hd.d_le
  have hax0 : (0:Int) ≤ (ax2 d i : Int) := by positivity
  have haxI : (ax2 d i : Int) < 2 ^ 16 := by
    have h := lt_trans (ax2_lt hi) hd.grid_h
    exact_mod_cast h
  have hHlt : d.H < 2 ^ 31 := lt_trans hd.grid_h (by norm_num)
  have hDlt : d.D < 2 ^ 31 := by have h := hd.d_le; omega
  have hax2lt : ax2 d i < 2 ^ 31 := by
    have h := lt_trans (ax2_lt hi) hd.grid_h
    omega
  have hH64 : (IntV.ofNat d.H : I64).toInt = (d.H : Int) := i64_ofNat_val hHlt
  have hD64 : (IntV.ofNat d.D : I64).toInt = (d.D : Int) := i64_ofNat_val hDlt
  have hD32 : (IntV.ofNat d.D : I32).toInt = (d.D : Int) := i32_ofNat_val hDlt
  have hax232 : (IntV.ofNat (ax2 d i) : I32).toInt = (ax2 d i : Int) :=
    i32_ofNat_val hax2lt
  have hm1 : (bosI d init i : Int) * (d.H : Int) < 2 ^ 47 :=
    int_mul_lt47 (lt_of_le_of_lt hb hd.t_lt) hd.grid_h
  have hp1 : (k * (IntV.ofNat d.H : I64)).toInt = (bosI d init i : Int) * (d.H : Int) :=
    toInt_mul_of hk hH64 (by norm_num) (sfits64_of (by positivity) (by linarith))
  have hp2 : (k * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64)).toInt
      = (bosI d init i : Int) * (d.H : Int) * (d.D : Int) :=
    toInt_mul_of hp1 hD64 (by norm_num) (sfits64_of (by positivity) (by nlinarith))
  have hp3 : ((IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)).toInt
      = (ax2 d i : Int) * (d.D : Int) :=
    toInt_mul_of hax232 hD32 (by norm_num) (sfits32_of (by positivity) (by nlinarith))
  have hidx : (buf.ptr 0 + k * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64)
        + (IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)).addr
      = buf.base + ((bosI d init i : Int) * (d.H : Int) * (d.D : Int)
          + (ax2 d i : Int) * (d.D : Int)) * ↑(DType.bytes .bf16) := by
    simp only [Buf.ptr_addr, Ptr.add_int, Ptr.offset_addr, hp2, hp3]
    ring
  rw [hidx]
  congr 1
  ring

private theorem ptr_packed (buf : Buf .bf16) (d : Inst) (hd : d.ok)
    {i : Nat} (hi : i < (launch d).size) (hv2 : d.isVarlen = false) {k1 k2 : I64}
    (h1 : k1.toInt = (ax1 d i : Int)) (h2 : k2.toInt = (d.T : Int)) :
    (buf.ptr 0 + k1 * k2 * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64)
        + (IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)).addr
      = (buf.ptr (((ax1 d i * d.T) * d.H + ax2 d i) * d.D)).addr := by
  have hHpos : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h0 | hp
    · rw [launch_size] at hi; rw [h0] at hi; exact absurd hi (Nat.not_lt_zero i)
    · exact hp
  have hDpos : 0 < d.D := by
    have h1' := hd.r_d
    have h2' := hd.r_pos
    omega
  have hBT : (d.B * d.T) * d.H * d.D < 2 ^ 62 := by
    have h := hd.size_lt
    simpa [nTok, hv2] using h
  have hB1 : ax1 d i * d.T ≤ d.B * d.T := Nat.mul_le_mul_right _ (le_of_lt (ax1_lt hi))
  have hstepH : d.B * d.T ≤ (d.B * d.T) * d.H := Nat.le_mul_of_pos_right _ hHpos
  have hstepHD : (d.B * d.T) * d.H ≤ (d.B * d.T) * d.H * d.D :=
    Nat.le_mul_of_pos_right _ hDpos
  have hstepD : d.B * d.T ≤ (d.B * d.T) * d.H * d.D := le_trans hstepH hstepHD
  have hP1n : ax1 d i * d.T < 2 ^ 62 := lt_of_le_of_lt (le_trans hB1 hstepD) hBT
  have hP2n : (ax1 d i * d.T) * d.H < 2 ^ 62 :=
    lt_of_le_of_lt (le_trans (Nat.mul_le_mul_right _ hB1) hstepHD) hBT
  have hP3n : (ax1 d i * d.T) * d.H * d.D < 2 ^ 62 :=
    lt_of_le_of_lt (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hB1)) hBT
  have hP1i : (ax1 d i : Int) * (d.T : Int) < 2 ^ 62 := by exact_mod_cast hP1n
  have hP2i : (ax1 d i : Int) * (d.T : Int) * (d.H : Int) < 2 ^ 62 := by exact_mod_cast hP2n
  have hP3i : (ax1 d i : Int) * (d.T : Int) * (d.H : Int) * (d.D : Int) < 2 ^ 62 := by
    exact_mod_cast hP3n
  have hax2D : (ax2 d i : Int) * (d.D : Int) < 2 ^ 31 := by
    have h1' : ax2 d i < 2 ^ 16 := lt_trans (ax2_lt hi) hd.grid_h
    have h2' : d.D ≤ 256 := hd.d_le
    have hnat : ax2 d i * d.D ≤ 2 ^ 16 * 256 := Nat.mul_le_mul (le_of_lt h1') h2'
    have h24 : (2:Nat) ^ 16 * 256 = 2 ^ 24 := by norm_num
    rw [h24] at hnat
    have hlt : ax2 d i * d.D < 2 ^ 31 := by omega
    exact_mod_cast hlt
  have hHlt : d.H < 2 ^ 31 := lt_trans hd.grid_h (by norm_num)
  have hDlt : d.D < 2 ^ 31 := by have h := hd.d_le; omega
  have hax2lt : ax2 d i < 2 ^ 31 := by
    have h := lt_trans (ax2_lt hi) hd.grid_h
    omega
  have hH64 : (IntV.ofNat d.H : I64).toInt = (d.H : Int) := i64_ofNat_val hHlt
  have hD64 : (IntV.ofNat d.D : I64).toInt = (d.D : Int) := i64_ofNat_val hDlt
  have hD32 : (IntV.ofNat d.D : I32).toInt = (d.D : Int) := i32_ofNat_val hDlt
  have hax232 : (IntV.ofNat (ax2 d i) : I32).toInt = (ax2 d i : Int) :=
    i32_ofNat_val hax2lt
  have hp1 : (k1 * k2).toInt = (ax1 d i : Int) * (d.T : Int) :=
    toInt_mul_of h1 h2 (by norm_num) (sfits64_of (by positivity) hP1i)
  have hp2 : (k1 * k2 * (IntV.ofNat d.H : I64)).toInt
      = (ax1 d i : Int) * (d.T : Int) * (d.H : Int) :=
    toInt_mul_of hp1 hH64 (by norm_num) (sfits64_of (by positivity) hP2i)
  have hp3 : (k1 * k2 * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64)).toInt
      = (ax1 d i : Int) * (d.T : Int) * (d.H : Int) * (d.D : Int) :=
    toInt_mul_of hp2 hD64 (by norm_num) (sfits64_of (by positivity) hP3i)
  have hp4 : ((IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)).toInt
      = (ax2 d i : Int) * (d.D : Int) :=
    toInt_mul_of hax232 hD32 (by norm_num) (sfits32_of (by positivity) hax2D)
  have hidx : (buf.ptr 0 + k1 * k2 * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64)
        + (IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)).addr
      = buf.base + ((ax1 d i : Int) * (d.T : Int) * (d.H : Int) * (d.D : Int)
          + (ax2 d i : Int) * (d.D : Int)) * ↑(DType.bytes .bf16) := by
    simp only [Buf.ptr_addr, Ptr.add_int, Ptr.offset_addr, hp3, hp4]
    ring
  rw [hidx]
  congr 1
  ring

private theorem ci_ptr1 (d : Inst) (hd : d.ok) {i : Nat} (hi : i < (launch d).size) :
    (d.ci.ptr 0 + (IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64) * (IntV.ofNat 2 : I64)).addr
      = (d.ci.ptr ((2 * ax0 d i : Nat) : Int)).addr := by
  have hcast : (IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64).toInt = (ax0 d i : Int) :=
    i64_of_cast32 (lt_trans (ax0_lt hi) hd.grid_t)
  have h2 : (IntV.ofNat 2 : I64).toInt = (2 : Int) := i64_ofNat_val (by norm_num)
  have hax0lt : ax0 d i < 2 ^ 31 := lt_trans (ax0_lt hi) hd.grid_t
  have hs : (ax0 d i : Int) * 2 < 2 ^ 62 := by
    have h : (ax0 d i : Int) < 2 ^ 31 := by exact_mod_cast hax0lt
    linarith
  have hk : ((IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64) * (IntV.ofNat 2 : I64)).toInt
      = ((2 * ax0 d i : Nat) : Int) := by
    rw [toInt_mul_of hcast h2 (by norm_num) (sfits64_of (by positivity) hs)]
    push_cast
    ring
  exact ptr_off1 d.ci _ _ hk

private theorem ci_ptr2 (d : Inst) (hd : d.ok) {i : Nat} (hi : i < (launch d).size) :
    (d.ci.ptr 0 + (IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64) * (IntV.ofNat 2 : I64)
        + (IntV.ofNat 1 : I32)).addr
      = (d.ci.ptr ((2 * ax0 d i + 1 : Nat) : Int)).addr := by
  have hcast : (IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64).toInt = (ax0 d i : Int) :=
    i64_of_cast32 (lt_trans (ax0_lt hi) hd.grid_t)
  have h2 : (IntV.ofNat 2 : I64).toInt = (2 : Int) := i64_ofNat_val (by norm_num)
  have hax0lt : ax0 d i < 2 ^ 31 := lt_trans (ax0_lt hi) hd.grid_t
  have hs : (ax0 d i : Int) * 2 < 2 ^ 62 := by
    have h : (ax0 d i : Int) < 2 ^ 31 := by exact_mod_cast hax0lt
    linarith
  have hk : ((IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64) * (IntV.ofNat 2 : I64)).toInt
      = ((2 * ax0 d i : Nat) : Int) := by
    rw [toInt_mul_of hcast h2 (by norm_num) (sfits64_of (by positivity) hs)]
    push_cast
    ring
  have h1 : (IntV.ofNat 1 : I32).toInt = (1 : Int) := i32_ofNat_val (by norm_num)
  have hsum : ((IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64) * (IntV.ofNat 2 : I64)).toInt
      + (IntV.ofNat 1 : I32).toInt = ((2 * ax0 d i + 1 : Nat) : Int) := by
    rw [hk, h1]; push_cast; ring
  exact ptr_off2 d.ci _ (IntV.ofNat 1 : I32) _ hsum

private theorem cu_ptr1 {dt : DType} (b : Buf dt) {k : I32} {t : Nat} (hk : k.toInt = (t : Int)) :
    (b.ptr 0 + k).addr = (b.ptr (t : Int)).addr := ptr_off1 b k (t : Int) hk

private theorem cu_ptr2 {dt : DType} (b : Buf dt) {k : I32} {t : Nat} (hk : k.toInt = (t : Int)) :
    (b.ptr 0 + k + (IntV.ofNat 1 : I32)).addr = (b.ptr ((t + 1 : Nat) : Int)).addr := by
  have h1 : (IntV.ofNat 1 : I32).toInt = (1 : Int) := i32_ofNat_val (by norm_num)
  refine ptr_off2 b k (IntV.ofNat 1 : I32) ((t + 1 : Nat) : Int) ?_
  rw [hk, h1]; push_cast; ring
/-- What the prologue of `body_eq` establishes: the chunk in `i_t`, the sequence's length in
`T_n`, the sequence in `i_n` (packed batches), and the first element of the program's row of `x`
and of `y` in `x_h` and `y_h`. -/
def StOk (d : Inst) (init : Mem) (i : Nat)
    (st : I64 × Ptr .bf16 × Ptr .bf16 × I64 × I32) : Prop :=
  st.1.toInt = (chunkI d init i : Int) ∧
  st.2.2.2.1.toInt = (lenI d init i : Int) ∧
  (d.isVarlen = true → st.2.2.2.2.toInt = (seqI d init i : Int)) ∧
  st.2.1.addr = (d.x.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr ∧
  st.2.2.1.addr = (d.y.ptr ((bosI d init i * d.H + ax2 d i) * d.D)).addr

/-- The prologue reads only its metadata and writes nothing. -/
theorem safe_prologue {P : Platform} {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {i : Nat} (hi : i < (launch d).size) {W : Set Int} {Q : Mem → Prop}
    {f : (I64 × Ptr .bf16 × Ptr .bf16 × I64 × I32) → Prog Unit}
    (hk : ∀ st, StOk d init i st →
      Safe ((launch d).env P) (R d init i) W (fun _ m => Q m) init (f st)) :
    Safe ((launch d).env P) (R d init i) W (fun _ m => Q m) init
      ((if d.isVarlen then (do
          let v_n ← tl.load1 (d.ci.ptr 0 + (IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64) * (IntV.ofNat 2 : I64))
          let i_n : I32 := (← tl.convert d.idx .i32 (Tensor.scalar v_n)).item
          let v_t ← tl.load1 (d.ci.ptr 0 + (IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64) * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))
          let i_t : I64 := (← tl.convert d.idx .i64 (Tensor.scalar v_t)).item
          let v_b ← tl.load1 (d.cu.ptr 0 + i_n)
          let bos : I64 := (← tl.convert d.idx .i64 (Tensor.scalar v_b)).item
          let v_e ← tl.load1 (d.cu.ptr 0 + i_n + (IntV.ofNat 1 : I32))
          let eos : I64 := (← tl.convert d.idx .i64 (Tensor.scalar v_e)).item
          let T_n : I64 := eos - bos
          let x_h : Ptr .bf16 := d.x.ptr 0 + bos * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + (IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)
          let y_h : Ptr .bf16 := d.y.ptr 0 + bos * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + (IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)
          pure (i_t, x_h, y_h, T_n, i_n))
        else (do
          let x_h : Ptr .bf16 := d.x.ptr 0 + (IntV.cast (IntV.ofNat (ax1 d i) : I32) : I64) * (IntV.cast (IntV.ofNat d.T : I32) : I64) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + (IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)
          let y_h : Ptr .bf16 := d.y.ptr 0 + (IntV.cast (IntV.ofNat (ax1 d i) : I32) : I64) * (IntV.cast (IntV.ofNat d.T : I32) : I64) * (IntV.ofNat d.H : I64) * (IntV.ofNat d.D : I64) + (IntV.ofNat (ax2 d i) : I32) * (IntV.ofNat d.D : I32)
          pure ((IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64), x_h, y_h, (IntV.cast (IntV.ofNat d.T : I32) : I64), (IntV.ofNat 0 : I32))
          ) : Prog (I64 × Ptr .bf16 × Ptr .bf16 × I64 × I32)) >>= f) := by
  by_cases hv : d.isVarlen = true
  · rw [ite_eq_left hv]
    simp only [Prog.bind_eq, Prog.bind_assoc, Prog.pure_eq]
    have hidxok : d.idx = .i32 ∨ d.idx = .i64 := hd.idx_ok
    have hs : SeqOk d init := hpre.seqs hv
    have hg : g0 d = d.NT := by simp [g0, hv]
    have haxNT : ax0 d i < d.NT := by rw [← hg]; exact ax0_lt hi
    have hseqI_lt : seqI d init i < d.N := by simpa [nSeq, hv] using seqI_lt hd hpre hi
    have hseqI_le : seqI d init i ≤ d.N := le_of_lt hseqI_lt
    have hseqI_le1 : seqI d init i + 1 ≤ d.N := by omega
    have hci0 : 2 * ax0 d i < d.ci.len := by have h := hd.ci_len hv; omega
    have hci1 : 2 * ax0 d i + 1 < d.ci.len := by have h := hd.ci_len hv; omega
    have hcu0 : seqI d init i < d.cu.len := by have h := hd.cu_len hv; omega
    have hcu1 : seqI d init i + 1 < d.cu.len := by have h := hd.cu_len hv; omega
    obtain ⟨hchSeqN, hchBT⟩ := hs.ci_ok (ax0 d i) haxNT
    obtain ⟨vn, hvn_some, hvn0⟩ := hs.ci_some (2 * ax0 d i) (by omega)
    obtain ⟨vt, hvt_some, hvt0⟩ := hs.ci_some (2 * ax0 d i + 1) (by omega)
    obtain ⟨raw_n, hread_n, hval_n⟩ := exists_read_of_idxAt hvn_some
    obtain ⟨raw_t, hread_t, hval_t⟩ := exists_read_of_idxAt hvt_some
    have hchSeq : chSeq d init (ax0 d i) = vn.toNat := by
      simp only [chSeq, hvn_some, Option.getD_some]
    have hchChunk : chChunk d init (ax0 d i) = vt.toNat := by
      simp only [chChunk, hvt_some, Option.getD_some]
    have hseqvn : seqI d init i = vn.toNat := by rw [seqI_true hv]; exact hchSeq
    have hchunkvt : chunkI d init i = vt.toNat := by rw [chunkI_true hv]; exact hchChunk
    have hvn_lt : vn < 2 ^ 31 := by
      have h1 : (vn.toNat : Int) < (d.N : Int) := by
        have h1' : vn.toNat < d.N := by rw [← hchSeq]; exact hchSeqN
        exact_mod_cast h1'
      have hN : (d.N : Int) < 2 ^ 31 := by
        have hN' : d.N < 2 ^ 31 := hd.n_lt hv
        exact_mod_cast hN'
      have hEq : (vn.toNat : Int) = vn := Int.toNat_of_nonneg hvn0
      omega
    have hBTpos : 0 < d.BT := by obtain ⟨e, he⟩ := hd.bt_pow2; rw [he]; positivity
    have hseqLen_le : seqLen d init (chSeq d init (ax0 d i)) ≤ d.T := by
      have hmono : cuN d init (chSeq d init (ax0 d i) + 1) ≤ cuN d init d.N :=
        cuN_mono_le hs (by omega) le_rfl
      rw [hs.cu_last] at hmono
      calc seqLen d init (chSeq d init (ax0 d i))
          = cuN d init (chSeq d init (ax0 d i) + 1)
              - cuN d init (chSeq d init (ax0 d i)) := rfl
        _ ≤ cuN d init (chSeq d init (ax0 d i) + 1) := Nat.sub_le _ _
        _ ≤ d.T := hmono
    have hvt_lt : vt < 2 ^ 31 := by
      have h2 : vt.toNat * d.BT < seqLen d init (chSeq d init (ax0 d i)) := by
        rw [← hchunkvt, chunkI_true hv]
        exact hchBT
      have h3 : vt.toNat * d.BT < d.T := lt_of_lt_of_le h2 hseqLen_le
      have h4 : vt.toNat ≤ vt.toNat * d.BT := Nat.le_mul_of_pos_right _ hBTpos
      have h5 : vt.toNat < d.T := lt_of_le_of_lt h4 h3
      have h5' : (vt.toNat : Int) < (d.T : Int) := by exact_mod_cast h5
      have hT : (d.T : Int) < 2 ^ 31 := by exact_mod_cast hd.t_lt
      have hEq : (vt.toNat : Int) = vt := Int.toNat_of_nonneg hvt0
      omega
    obtain ⟨vb, hvb_some, hvb0⟩ := hs.cu_some (seqI d init i) hseqI_le
    obtain ⟨ve, hve_some, hve0⟩ := hs.cu_some (seqI d init i + 1) hseqI_le1
    obtain ⟨raw_b, hread_b, hval_b⟩ := exists_read_of_idxAt hvb_some
    obtain ⟨raw_e, hread_e, hval_e⟩ := exists_read_of_idxAt hve_some
    have hcu_vb : cuN d init (seqI d init i) = vb.toNat := by
      simp only [cuN, hvb_some, Option.getD_some]
    have hcu_ve : cuN d init (seqI d init i + 1) = ve.toNat := by
      simp only [cuN, hve_some, Option.getD_some]
    have hvb_le : (vb : Int) ≤ (d.T : Int) := by
      have hmono : cuN d init (seqI d init i) ≤ cuN d init d.N :=
        cuN_mono_le hs hseqI_le le_rfl
      rw [hs.cu_last] at hmono
      have h1 : vb.toNat ≤ d.T := by rw [← hcu_vb]; exact hmono
      have h1' : (vb.toNat : Int) ≤ (d.T : Int) := by exact_mod_cast h1
      have hEq : (vb.toNat : Int) = vb := Int.toNat_of_nonneg hvb0
      omega
    have hvb_lt : vb < 2 ^ 31 := by
      have hT : (d.T : Int) < 2 ^ 31 := by exact_mod_cast hd.t_lt
      omega
    have hve_lt : ve < 2 ^ 31 := by
      have hmono : cuN d init (seqI d init i + 1) ≤ cuN d init d.N :=
        cuN_mono_le hs hseqI_le1 le_rfl
      rw [hs.cu_last] at hmono
      have h1 : ve.toNat ≤ d.T := by rw [← hcu_ve]; exact hmono
      have h1' : (ve.toNat : Int) ≤ (d.T : Int) := by exact_mod_cast h1
      have hEq : (ve.toNat : Int) = ve := Int.toNat_of_nonneg hve0
      have hT : (d.T : Int) < 2 ^ 31 := by exact_mod_cast hd.t_lt
      omega
    have hvbve_nat : vb.toNat ≤ ve.toNat := by
      rw [← hcu_vb, ← hcu_ve]
      exact hs.cu_mono (seqI d init i) hseqI_lt
    have hbos_eq : bosI d init i = vb.toNat := by
      simp only [bosI, bosAt]
      rw [ite_eq_left hv]
      exact hcu_vb
    have hbos_le : bosI d init i ≤ d.T := by
      have h1 : vb.toNat ≤ d.T := by
        have hEq : (vb.toNat : Int) = vb := Int.toNat_of_nonneg hvb0
        omega
      omega
    have hlen_eq : lenI d init i = ve.toNat - vb.toNat := by
      simp only [lenI, lenAt]
      rw [ite_eq_left hv]
      simp only [seqLen, hcu_ve, hcu_vb]
    have hlenI : (lenI d init i : Int) = ve - vb := by
      rw [hlen_eq, Nat.cast_sub hvbve_nat, Int.toNat_of_nonneg hve0,
        Int.toNat_of_nonneg hvb0]
    have hn_addr : (d.ci.ptr 0 + (IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64)
        * (IntV.ofNat 2 : I64)).addr
        = (d.ci.ptr ((2 * ax0 d i : Nat) : Int)).addr := ci_ptr1 d hd hi
    have ht_addr : (d.ci.ptr 0 + (IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64)
        * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32)).addr
        = (d.ci.ptr ((2 * ax0 d i + 1 : Nat) : Int)).addr := ci_ptr2 d hd hi
    refine safe_load1 (v0 := raw_n) ?_ ?_ ?_
    · rw [hn_addr]; exact hread_n
    · rw [hn_addr]; intro z hz; exact subset_R_ci (Buf.span_sub d.ci hci0 hz)
    · refine safe_convert (d := d.idx) (d' := .i32) (x := Tensor.scalar raw_n) ?_ ?_
      · intro j
        exact ⟨IntV.ofInt (idxVal d.idx raw_n),
          convRel_idx32 P d.idx hidxok raw_n⟩
      · intro r hr
        have hconv := hr ()
        have hitem_n : r.item = (IntV.ofInt (idxVal d.idx raw_n) : I32) :=
          convRel_item_i32 hidxok hconv
        have hkn : r.item.toInt = (seqI d init i : Int) := by
          rw [hitem_n, hval_n, hseqvn, i32_ofInt_val hvn0 hvn_lt,
            Int.toNat_of_nonneg hvn0]
        refine safe_load1 (v0 := raw_t) ?_ ?_ ?_
        · rw [ht_addr]; exact hread_t
        · rw [ht_addr]; intro z hz; exact subset_R_ci (Buf.span_sub d.ci hci1 hz)
        · refine safe_convert (d := d.idx) (d' := .i64) (x := Tensor.scalar raw_t) ?_ ?_
          · intro j
            exact ⟨IntV.ofInt (idxVal d.idx raw_t),
              convRel_idx64 P d.idx hidxok raw_t⟩
          · intro r2 hr2
            have hconv2 := hr2 ()
            have hitem_t : r2.item = (IntV.ofInt (idxVal d.idx raw_t) : I64) :=
              convRel_item_i64 hidxok hconv2
            have hkt : r2.item.toInt = (chunkI d init i : Int) := by
              rw [hitem_t, hval_t, hchunkvt, i64_ofInt_val hvt0 hvt_lt,
                Int.toNat_of_nonneg hvt0]
            refine safe_load1 (v0 := raw_b) ?_ ?_ ?_
            · rw [cu_ptr1 d.cu hkn]; exact hread_b
            · rw [cu_ptr1 d.cu hkn]; intro z hz
              exact subset_R_cu (Buf.span_sub d.cu hcu0 hz)
            · refine safe_convert (d := d.idx) (d' := .i64) (x := Tensor.scalar raw_b) ?_ ?_
              · intro j
                exact ⟨IntV.ofInt (idxVal d.idx raw_b),
                  convRel_idx64 P d.idx hidxok raw_b⟩
              · intro r3 hr3
                have hconv3 := hr3 ()
                have hitem_b : r3.item = (IntV.ofInt (idxVal d.idx raw_b) : I64) :=
                  convRel_item_i64 hidxok hconv3
                have hkb : r3.item.toInt = (bosI d init i : Int) := by
                  rw [hitem_b, hval_b, hbos_eq, i64_ofInt_val hvb0 hvb_lt,
                    Int.toNat_of_nonneg hvb0]
                refine safe_load1 (v0 := raw_e) ?_ ?_ ?_
                · rw [cu_ptr2 d.cu hkn]; exact hread_e
                · rw [cu_ptr2 d.cu hkn]; intro z hz
                  exact subset_R_cu (Buf.span_sub d.cu hcu1 hz)
                · refine safe_convert (d := d.idx) (d' := .i64)
                    (x := Tensor.scalar raw_e) ?_ ?_
                  · intro j
                    exact ⟨IntV.ofInt (idxVal d.idx raw_e),
                      convRel_idx64 P d.idx hidxok raw_e⟩
                  · intro r4 hr4
                    have hconv4 := hr4 ()
                    have hitem_e : r4.item = (IntV.ofInt (idxVal d.idx raw_e) : I64) :=
                      convRel_item_i64 hidxok hconv4
                    have heos : r4.item.toInt = ve := by
                      rw [hitem_e, hval_e, i64_ofInt_val hve0 hve_lt]
                    have hbos : r3.item.toInt = vb := by
                      rw [hkb, hbos_eq, Int.toNat_of_nonneg hvb0]
                    have hsub : (r4.item - r3.item).toInt = ve - vb := by
                      refine i64_sub_of heos hbos ?_
                      exact sfits64_of (by omega) (by omega)
                    apply hk
                    refine ⟨?_, ?_, ?_, ?_, ?_⟩
                    · exact hkt
                    · rw [hsub, hlenI]
                    · intro hc
                      exact hkn
                    · exact ptr_varlen d.x d hd hi hkb hbos_le
                    · exact ptr_varlen d.y d hd hi hkb hbos_le
  · rw [ite_eq_right hv]
    simp only [Prog.bind_eq, Prog.pure_eq, Prog.ret_bind]
    have hv2 : d.isVarlen = false := by cases h : d.isVarlen <;> simp_all
    have hbos : bosI d init i = ax1 d i * d.T := by
      simp [bosI, bosAt, hv2, seqI_false hv2]
    have hl : lenI d init i = d.T := by simp [lenI, lenAt, hv2]
    have h1 : (IntV.cast (IntV.ofNat (ax1 d i) : I32) : I64).toInt = (ax1 d i : Int) :=
      i64_of_cast32 (by have h := ax1_lt hi; have hb := hd.b_le; omega)
    have h2 : (IntV.cast (IntV.ofNat d.T : I32) : I64).toInt = (d.T : Int) :=
      i64_of_cast32 hd.t_lt
    apply hk
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · show ((IntV.cast (IntV.ofNat (ax0 d i) : I32) : I64)).toInt
        = (chunkI d init i : Int)
      rw [i64_of_cast32 (lt_trans (ax0_lt hi) hd.grid_t), chunkI_false hv2]
    · show ((IntV.cast (IntV.ofNat d.T : I32) : I64)).toInt = (lenI d init i : Int)
      rw [i64_of_cast32 hd.t_lt, hl]
    · intro hc
      rw [hv2] at hc
      exact absurd hc (by simp)
    · rw [hbos]; exact ptr_packed d.x d hd hi hv2 h1 h2
    · rw [hbos]; exact ptr_packed d.y d hd hi hv2 h1 h2

end Claims.RotaryVarlen
