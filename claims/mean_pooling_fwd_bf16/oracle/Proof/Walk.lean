import Claim
import TVA.Proof
import Proof.Basic
import Proof.Geo
import Proof.Lanes
import Proof.Value
import Proof.Tail

/-!
# One program, walked line by line

`tail_safe` takes the six `int64` scalars with their values through the offsets, the mask and the
tile load into `rest_safe`; `safe_program` computes those scalars (the program ids, `//` and `%`,
the table reads of a variable-length launch, the `tl.cdiv`s) and ends in `tail_safe`.
-/

namespace Claims.MeanPoolingFwd

open TVA

/-! ## `tl.cdiv` and table reads -/

/-- **`tl.cdiv(x, y)` of scalars** with `x ≥ 0`, `y > 0` and `x + y − 1 < 2^(w−1)`: safe, of
value `(x + y − 1) / y`. -/
theorem safe_cdiv1 {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {w : Nat}
    {x y : IntV true w} {a b : Nat} (hx : x.toInt = a) (hy : y.toInt = b) (hb : 0 < b)
    (h1 : (1 : IntV true w).toInt = 1) (hfit : a + b < 2 ^ (w - 1))
    {k : Tensor [] (IntV true w) → Prog α}
    (hk : ∀ r : Tensor [] (IntV true w), r.item.toInt = (((a + b - 1) / b : Nat) : Int) →
      Safe E R W Q m (k r)) :
    Safe E R W Q m (tl.cdiv (Tensor.scalar x) (Tensor.scalar y) >>= k) := by
  have hy1 : (y - 1).toInt = ((b - 1 : Nat) : Int) :=
    IntV.sub_nat hy (by rw [h1]; rfl) (by omega) (by omega)
  have hs : (x + (y - 1)).toInt = ((a + (b - 1) : Nat) : Int) := IntV.add_nat hx hy1 (by omega)
  unfold tl.cdiv
  refine safe_idiv ?_ ?_
  · intro i
    obtain ⟨q, hq, -⟩ := IntV.div?_of_nonneg (a := x + (y - 1)) (b := y) (by rw [hs]; omega)
      (by rw [hy]; omega)
    exact ⟨q, hq⟩
  · intro r hr
    refine hk r ?_
    obtain ⟨q, hq, hqv⟩ := IntV.div?_of_nonneg (a := x + (y - 1)) (b := y) (by rw [hs]; omega)
      (by rw [hy]; omega)
    have h2 : r.get () = q := by
      have := hr ()
      have h3 : (Tensor.scalar x + (Tensor.scalar y - Tensor.full 1)).get () = x + (y - 1) := rfl
      rw [h3, show (Tensor.scalar y : Tensor [] (IntV true w)).get () = y from rfl, hq] at this
      exact (Option.some.inj this).symm
    show (r.get ()).toInt = _
    rw [h2, hqv, hs, hy]
    rw [show a + b - 1 = a + (b - 1) by omega]
    push_cast
    rfl

/-- A pointer advanced by an integer of value `i` is the buffer's element `i`. -/
theorem addr_ptr_add {dt : DType} (b : Buf dt) {sg : Bool} {w : Nat} {k : IntV sg w} {i : Int}
    (h : k.toInt = i) : (b.ptr 0 + k).addr = (b.ptr i).addr := by
  rw [Ptr.add_int, h]
  simp [Ptr.offset, Buf.ptr]

/-- A pointer advanced by two integers whose values sum to `i` is the buffer's element `i`. -/
theorem addr_ptr_add2 {dt : DType} (b : Buf dt) {sg : Bool} {w : Nat} {k : IntV sg w}
    {sg' : Bool} {w' : Nat} {k' : IntV sg' w'} {i : Int} (h : k.toInt + k'.toInt = i) :
    ((b.ptr 0 + k) + k').addr = (b.ptr i).addr := by
  rw [Ptr.add_int, Ptr.add_int, Ptr.offset_offset, h]
  simp [Ptr.offset, Buf.ptr]

/-- A scalar load of an `int32`/`int64` table entry, converted to `int64`, is safe; the rest sees
the entry's value. -/
theorem load_conv_i64 {dt : DType} (hh : dt = .i32 ∨ dt = .i64)
    {init : Mem} {b : Buf dt} {v : Nat → dt.Val} (hH : b.Holds init v) {n : Nat} (hn : n < b.len)
    {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
    {p : Ptr dt} (hp : p.addr = (b.ptr n).addr) (hR : Mem.span dt p.addr ⊆ R)
    {rest : Tensor [] I64 → Prog α}
    (hrest : ∀ a : Tensor [] I64, a.item.toInt = entryInt dt (v n) → Safe E R W Q init (rest a)) :
    Safe E R W Q init (tl.load1 p >>= fun x => tl.convert dt .i64 (Tensor.scalar x) >>= rest) := by
  rcases hh with h | h
  · subst h
    have hv0 : init.read .i32 p.addr = some (v n) := by rw [hp]; exact hH n hn
    refine safe_load1 (v0 := v n) hv0 hR ?_
    refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((v n).toInt), rfl⟩
    · intro r hr
      refine hrest r ?_
      have h1 : r.item = IntV.ofInt ((v n).toInt) := hr ()
      rw [h1]
      refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
      · have hs := IntV.sfits_toInt (v n)
        simp only [IntV.SFits] at hs ⊢
        omega
      · rfl
  · subst h
    have hv0 : init.read .i64 p.addr = some (v n) := by rw [hp]; exact hH n hn
    refine safe_load1 (v0 := v n) hv0 hR ?_
    refine safe_convert (d := .i64) (d' := .i64) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((v n).toInt), rfl⟩
    · intro r hr
      refine hrest r ?_
      have h1 : r.item = IntV.ofInt ((v n).toInt) := hr ()
      rw [h1]
      refine (IntV.toInt_ofInt (by norm_num) ?_).trans ?_
      · have hs := IntV.sfits_toInt (v n)
        simp only [IntV.SFits] at hs ⊢
        omega
      · rfl

/-- The same converted to `int32`, when the entry fits in `int32`. -/
theorem load_conv_i32 {dt : DType} (hh : dt = .i32 ∨ dt = .i64)
    {init : Mem} {b : Buf dt} {v : Nat → dt.Val} (hH : b.Holds init v) {n : Nat} (hn : n < b.len)
    {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop}
    {p : Ptr dt} (hp : p.addr = (b.ptr n).addr) (hR : Mem.span dt p.addr ⊆ R)
    (hz : IntV.SFits 32 (entryInt dt (v n)))
    {rest : Tensor [] I32 → Prog α}
    (hrest : ∀ a : Tensor [] I32, a.item.toInt = entryInt dt (v n) → Safe E R W Q init (rest a)) :
    Safe E R W Q init (tl.load1 p >>= fun x => tl.convert dt .i32 (Tensor.scalar x) >>= rest) := by
  rcases hh with h | h
  · subst h
    have hv0 : init.read .i32 p.addr = some (v n) := by rw [hp]; exact hH n hn
    refine safe_load1 (v0 := v n) hv0 hR ?_
    refine safe_convert (d := .i32) (d' := .i32) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((v n).toInt), rfl⟩
    · intro r hr
      refine hrest r ?_
      have h1 : r.item = IntV.ofInt ((v n).toInt) := hr ()
      rw [h1]
      exact (IntV.toInt_ofInt (by norm_num) hz).trans rfl
  · subst h
    have hv0 : init.read .i64 p.addr = some (v n) := by rw [hp]; exact hH n hn
    refine safe_load1 (v0 := v n) hv0 hR ?_
    refine safe_convert (d := .i64) (d' := .i32) ?_ ?_
    · intro j
      exact ⟨IntV.ofInt ((v n).toInt), rfl⟩
    · intro r hr
      refine hrest r ?_
      have h1 : r.item = IntV.ofInt ((v n).toInt) := hr ()
      rw [h1]
      exact (IntV.toInt_ofInt (by norm_num) hz).trans rfl

/-! ## The tail -/

/-- **From line 108 on**: with the scalars `i_d`, `i_h`, `i_tg`, `i_t`, `bos`, `T` of values
`iD`, `hOf`, `outRow`, `chk`, `bos`, `lenS`, the rest of the body is safe and ends with `Q`. -/
theorem tail_safe {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    (hP : Device E.P) (hT : d.varlen = true → TablesOk d) {i : Nat} (hi : i < gx d * gy d)
    {idv ih itg it bs TT : I64}
    (hidv : idv.toInt = (iD d i : Int)) (hih : ih.toInt = (hOf d i : Int))
    (hitg : itg.toInt = (outRow d i : Int)) (hit : it.toInt = (chk d i : Int))
    (hbs : bs.toInt = (bos d i : Int)) (hTT : TT.toInt = (lenS d i : Int)) :
    Safe E (R d i) (W d i) (fun _ m => Q d i m) init
      (do
        let o_t := bc (it * (IntV.ofNat d.BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.BT))
        let o_d := bc (idv * (IntV.ofNat d.BD : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.BD))
        let m_x := ((o_t.col d.BD).ltS TT).and ((o_d.row d.BT).ltS (IntV.ofNat d.D : I64))
        let p_x := d.x.ptr 0 + (bs * (IntV.ofNat d.H : I64) + ih) * (IntV.ofNat d.D : I64)
          + o_t.col d.BD * (IntV.ofNat (d.H * d.D) : I64) + o_d.row d.BT
        let p_o := d.o.ptr 0 + (itg * (IntV.ofNat d.H : I64) + ih) * (IntV.ofNat d.D : I64) + o_d
        let b_x : Tensor [d.BT, d.BD] F32 ← tl.cvt (← tl.load p_x (some m_x)
          (some (← tl.convert .f32 .bf16 (tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0))))))
        let b_s ← tl.sum0 b_x
        let L := (tl.iminimum (Tensor.scalar (IntV.ofNat d.BT : I64))
          (Tensor.scalar (TT - it * (IntV.ofNat d.BT : I64)))).item
        let L32 : Tensor [] F32 ← tl.itof (Tensor.scalar L)
        let b_o ← tl.div b_s (bc L32.item)
        let out : Tensor [d.BD] BF16 ← tl.cvt b_o
        tl.store p_o out (some (o_d.ltS (IntV.ofNat d.D : I64)))) := by
  have hxv : d.x.Valid init := hpre.1
  have hxH : d.x.Holds init d.xv := hpre.2.2.1
  have hHD := BTHD_le hd hpre
  obtain ⟨heX, hrow⟩ := eX_lt hd hT hpre hi
  obtain ⟨hoR, heO⟩ := eO_lt hd hpre hi
  have hbosH := bosH_lt hd hT hpre hi
  have hH16 : d.H < 2 ^ 16 := lt_of_le_of_lt (Nat.le_mul_of_pos_left _ hd.B_pos) hd.BH_lt
  have hH63 : d.H < 2 ^ 63 := by omega
  have hD63 : d.D < 2 ^ 63 := by have := hd.D_le; omega
  have hcl := chk_lt hd hT hi
  have hlen22 := lenS_le hd hT hi
  have hbt := bt_le hd
  have hhH := (bOf_lt hd hi).2
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j; exact ⟨IntV.ofInt ((tl.arange 0 d.BT).get j).toInt, rfl⟩
  intro c1 hc1
  refine safe_convert (d := .i32) (d' := .i64) ?_ ?_
  · intro j; exact ⟨IntV.ofInt ((tl.arange 0 d.BD).get j).toInt, rfl⟩
  intro c2 hc2
  have hr := o_t_toInt hd hT hi hit hc1
  have hc := o_d_toInt hd hi hidv hc2
  refine safe_convert (d := .f32) (d' := .bf16) ?_ ?_
  · intro j
    exact ⟨FloatV.cvt ((tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0))).get j),
      Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), FloatV.NaNEq.refl _⟩⟩
  intro pad hpad
  -- the scalars of the two pointers
  have hS : ((bs * (IntV.ofNat d.H : I64) + ih) * (IntV.ofNat d.D : I64)).toInt = (eX d i : Int) := by
    have h1 : (bs * (IntV.ofNat d.H : I64)).toInt = ((bos d i * d.H : Nat) : Int) :=
      IntV.I64.mul_nat hbs (i64_ofNat hH63) hbosH
    have hle : bos d i * d.H + hOf d i ≤ eX d i := by
      unfold eX; exact Nat.le_mul_of_pos_right _ hd.D_pos
    have h2 : (bs * (IntV.ofNat d.H : I64) + ih).toInt = ((bos d i * d.H + hOf d i : Nat) : Int) :=
      IntV.I64.add_nat h1 hih (by omega)
    exact IntV.I64.mul_nat h2 (i64_ofNat hD63) heX
  have hSo : ((itg * (IntV.ofNat d.H : I64) + ih) * (IntV.ofNat d.D : I64)).toInt = (eO d i : Int) := by
    have h1 : (itg * (IntV.ofNat d.H : I64)).toInt = ((outRow d i * d.H : Nat) : Int) :=
      IntV.I64.mul_nat hitg (i64_ofNat hH63) hoR
    have hle : outRow d i * d.H + hOf d i ≤ eO d i := by
      unfold eO; exact Nat.le_mul_of_pos_right _ hd.D_pos
    have h2 : (itg * (IntV.ofNat d.H : I64) + ih).toInt = ((outRow d i * d.H + hOf d i : Nat) : Int) :=
      IntV.I64.add_nat h1 hih (by omega)
    exact IntV.I64.mul_nat h2 (i64_ofNat hD63) heO
  have hHDc : (IntV.ofNat (d.H * d.D) : I64).toInt = ((d.H * d.D : Nat) : Int) :=
    i64_ofNat (by
      have h1 : d.H * d.D ≤ 2 ^ 16 * 2 ^ 17 := Nat.mul_le_mul hH16.le hd.D_le
      omega)
  -- the load
  refine safe_load_tileAt d.x (e := eX d i) (s0 := d.H * d.D) (s1 := 1) (ρ := rho d i)
    (κ := kap d i) (M := lenS d i) (N := d.D) ?_ ?_ ?_ hxv ?_ ?_
  · exact Tile2.ptr_lane_seq_of_on d.x (ptr_add_eq d.x hS)
      (Tile2.colMul_toInt_on (BN := d.BD) hr hHDc (by simpa using hrow))
      (fun r l _ => Tile2.row_toInt (BM := d.BT) hc r l)
  · exact Tile2.maskAt_and hr hc hTT (i64_ofNat hD63)
  · refine Or.inr (Or.inr ?_)
    have h1 := eX_last hd hT hi
    have h2 := hd.x_len
    have := hd.D_pos
    omega
  · intro z hz; exact Or.inl hz
  intro v hread hother
  have hL : ((tl.iminimum (Tensor.scalar (IntV.ofNat d.BT : I64))
      (Tensor.scalar (TT - it * (IntV.ofNat d.BT : I64)))).item).toInt = (Lof d i : Int) := by
    have h1 : (it * (IntV.ofNat d.BT : I64)).toInt = ((chk d i * d.BT : Nat) : Int) :=
      IntV.I64.mul_nat hit (i64_ofNat (by omega)) (by omega)
    have h2 : (TT - it * (IntV.ofNat d.BT : I64)).toInt = ((lenS d i - chk d i * d.BT : Nat) : Int) :=
      IntV.I64.sub_nat hTT h1 (by omega) (by omega)
    show (IntV.min (IntV.ofNat d.BT : I64) (TT - it * (IntV.ofNat d.BT : I64))).toInt = _
    rw [IntV.toInt_min, h2, i64_ofNat (by omega)]
    unfold Lof
    push_cast
    rfl
  refine rest_safe hd hpre hP hT hi ?_ ?_ hL hc ?_
  · intro r l hrl hkl
    have h1 := hread r l hrl hkl
    have hidx : eX d i + rho d i r * (d.H * d.D) + kap d i l * 1 =
        ((start d (outRow d i) + r) * d.H + hOf d i) * d.D + kap d i l := by
      rw [start_eq hd hi]; unfold eX rho; ring
    rw [hidx] at h1
    have hlt : ((start d (outRow d i) + r) * d.H + hOf d i) * d.D + kap d i l < d.x.len := by
      have h3 := hd.x_len
      have h4 := eX_last hd hT hi
      rw [← hidx]
      have : rho d i r * (d.H * d.D) ≤ (lenS d i - 1) * (d.H * d.D) :=
        Nat.mul_le_mul_right _ (by omega)
      omega
    have h2 := hxH _ hlt
    rw [h1] at h2
    exact (Option.some.inj h2)
  · intro r l hrl
    rw [hother r l (by omega) _ rfl]
    exact pad_zero (hpad (r, l, ()))
  · intro l
    show (d.o.ptr 0 + (itg * (IntV.ofNat d.H : I64) + ih) * (IntV.ofNat d.D : I64)).offset
      ((_ : Tensor [d.BD - 0] I64).get1 l).toInt = _
    rw [hc l, ptr_add_eq d.o hSo, Buf.ptr_offset]
    push_cast
    rfl

end Claims.MeanPoolingFwd
