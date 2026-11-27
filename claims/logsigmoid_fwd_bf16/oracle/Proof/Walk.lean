import Claim
import TVA.Proof
import Proof.Lane

/-!
# One program

Program `p` handles the elements `blk d p = {k | p·B ≤ k < p·B + B, k < T}`: lane `i` is element
`k = p·B + i`, enabled when `k < T`; it reads element `xIdx d k` of `x` and writes element `k` of
`y`. `safe_program` walks the model's body line by line: the `int64` offsets (`offs`, `row`, `col`,
the three runtime integers through `arg64`), the masked load, the float operations, the store, and
`lane` for each enabled lane at the end.
-/

set_option maxHeartbeats 4000000

namespace Claims.LogsigmoidFwd

open TVA

/-- The number of programs, `cdiv(T, B)`. -/
def G (d : Inst) : Nat := (d.T + d.B - 1) / d.B

/-- The elements program `p` owns: its block, cut at `T`. -/
def blk (d : Inst) (p : Nat) : Set Nat := {k | p * d.B ≤ k ∧ k < p * d.B + d.B ∧ k < d.T}

/-- Program `p` reads the elements of `x` of its block. -/
def R (d : Inst) (p : Nat) : Set Int := d.x.elems {j | ∃ k ∈ blk d p, j = xIdx d k}

/-- Program `p` writes its block of `y`. -/
def W (d : Inst) (p : Nat) : Set Int := d.y.elems (blk d p)

/-- After program `p`, each element of its block of `y` is as `Post` says. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ k ∈ blk d p, ∃ v : BF16, init.read .bf16 (d.x.ptr (xIdx d k)).addr = some v ∧
    ∃ c : BF16, m.read .bf16 (d.y.ptr k).addr = some c ∧ c.isFinite = true ∧
      |c.toReal - lsRef v.toReal / d.temperature.toReal| ≤ yTol v.toReal d.temperature.toReal

/-- `B` is positive. -/
theorem B_pos {d : Inst} (hd : d.ok) : 0 < d.B := by
  obtain ⟨e, he⟩ := hd.block_pow2; rw [he]; positivity

/-- A program's block ends within `2^51`. -/
theorem p_bound {d : Inst} (hd : d.ok) {p : Nat} (hp : p < G d) : p * d.B + d.B ≤ 2 ^ 51 := by
  have hg : G d < 2 ^ 31 := hd.grid_lt
  have h1 : p * d.B + d.B = (p + 1) * d.B := by ring
  rw [h1]
  calc (p + 1) * d.B ≤ 2 ^ 31 * 2 ^ 20 := Nat.mul_le_mul (by omega) hd.block_le
    _ = 2 ^ 51 := by norm_num

/-- A program exists only when `T > 0`; its first element is below `T`. -/
theorem p_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < G d) : p * d.B < d.T := by
  have hB := B_pos hd
  have h1 := (Nat.le_div_iff_mul_le hB).mp (Nat.succ_le_of_lt hp)
  rw [Nat.succ_mul] at h1
  omega

/-- `T < 2^51`. -/
theorem T_lt {d : Inst} (hd : d.ok) : d.T < 2 ^ 51 := by
  have hB := B_pos hd
  have hg : (d.T + d.B - 1) / d.B < 2 ^ 31 := hd.grid_lt
  have h2 := (Nat.div_lt_iff_lt_mul hB).mp hg
  have : 2 ^ 31 * d.B ≤ 2 ^ 51 := by
    calc 2 ^ 31 * d.B ≤ 2 ^ 31 * 2 ^ 20 := Nat.mul_le_mul_left _ hd.block_le
      _ = 2 ^ 51 := by norm_num
  omega

/-- A runtime integer argument through `arg64`: whatever its class, the program continues with an
`int64` of its value. -/
theorem safe_arg64 {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem} {n : Nat}
    (hn : n < 2 ^ 63) {k : I64 → Prog α}
    (hk : ∀ v : I64, v.toInt = n → Safe E R W Q m (k v)) :
    Safe E R W Q m (arg64 (intArg n) >>= k) := by
  unfold intArg
  split_ifs with h
  · show Safe E R W Q m (tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat n)) >>= fun r => k r.item)
    refine safe_convert (hex := fun _ => ⟨_, rfl⟩) ?_
    intro r hr
    refine hk _ ?_
    have h1 : r.item = IntV.ofInt (IntV.toInt ((Tensor.scalar (IntV.ofNat n : I32)).get ())) := hr ()
    have h2 : IntV.toInt ((Tensor.scalar (IntV.ofNat n : I32)).get ()) = (n : Int) :=
      IntV.I32.toInt_ofNat h
    rw [h1, h2]
    exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
  · exact hk _ (IntV.I64.toInt_ofNat hn)

/-- **One program** of the launch, the one with `program_id(0) = p`, is safe from `init`: it reads
the elements of `x` of its block, writes its block of `y`, and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (hexp : Device.ExpF32 E.P) (hlog : Device.LogF32 E.P) (hfull : Device.DivFull E.P)
    (hetot : Device.MathTotal E.P .exp .binary32) (hltot : Device.MathTotal E.P .log .binary32)
    (hdtot : Device.DivTotal E.P .binary32) {p : Nat} (hp : p < G d) (c : Ctx)
    (hc : c.pid 0 = p) :
    Safe E (R d p) (W d p) (fun _ m => Q d init p m) init ((model d).body c) := by
  obtain ⟨hvx, hvy, hin⟩ := hpre
  have hB := B_pos hd
  have hpb := p_bound hd hp
  have hplt := p_lt hd hp
  have hT51 := T_lt hd
  have hB20 := hd.block_le
  have hN : 0 < d.N := by
    rcases Nat.eq_zero_or_pos d.N with h | h
    · simp [Inst.T, h] at hplt
    · exact h
  have hD : 0 < d.D := by
    rcases Nat.eq_zero_or_pos d.D with h | h
    · simp [Inst.T, h] at hplt
    · exact h
  have hDT : d.D ≤ d.T := by
    unfold Inst.T; exact Nat.le_mul_of_pos_left _ hN
  have hp31 : p < 2 ^ 31 := lt_trans hp hd.grid_lt
  have hxlen : d.x.len ≤ 2 ^ 63 := by
    have := hvx.2.1; have := hvx.1
    simp only [DType.bytes] at *
    omega
  have hylen : d.y.len ≤ 2 ^ 63 := by
    have := hvy.2.1; have := hvy.1
    simp only [DType.bytes] at *
    omega
  have hsx63 : d.sx < 2 ^ 63 := hd.sx_lt
  -- the lanes: lane `i` is element `k i = p·B + i`, enabled below `T`
  let k : Fin d.B → Nat := fun i => p * d.B + i.val
  let en : Idx [d.B] → Bool := fun i => decide (k i.1 < d.T)
  let g : Idx [d.B] → Nat := fun i => xIdx d (k i.1)
  let gy : Idx [d.B] → Nat := fun i => k i.1
  have hen : ∀ i : Idx [d.B], en i = true → k i.1 < d.T := fun i h => of_decide_eq_true h
  simp only [model, logsigmoid_fwd_kernel]
  refine safe_convert (hex := fun _ => ⟨_, rfl⟩) ?_
  intro r hr
  refine safe_convert (hex := fun _ => ⟨_, rfl⟩) ?_
  intro o_a hoa
  have hi_n : (r.item).toInt = (p : Int) := by
    have h1 : r.item = IntV.ofInt (IntV.toInt ((Tensor.scalar (tl.program_id c 0)).get ())) := hr ()
    have h2 : IntV.toInt ((Tensor.scalar (tl.program_id c 0)).get ()) = (p : Int) := by
      show (tl.program_id c 0).toInt = _
      rw [tl.program_id, hc]
      exact IntV.I32.toInt_ofNat hp31
    rw [h1, h2]
    exact IntV.toInt_ofInt (by norm_num) ⟨by omega, by omega⟩
  have hstart : (r.item * (IntV.ofNat d.B : I64)).toInt = ((p * d.B : Nat) : Int) :=
    IntV.I64.mul_nat hi_n (IntV.I64.toInt_ofNat (by omega)) (by omega)
  have hoffs : ∀ i : Fin d.B,
      ((r.item * (IntV.ofNat d.B : I64) + o_a).get1 i).toInt = ((k i : Nat) : Int) :=
    fun i => Tile2.start_arange64_toInt (n := d.B) (r0 := p * d.B) hstart hoa (by omega) (by omega) i
  refine safe_arg64 (by omega) ?_
  intro T64 hT64
  have hfullD : ∀ i : Fin d.B, ((tl.full [d.B] (IntV.ofNat d.D : I64)).get1 i).toInt = (d.D : Int) :=
    fun _ => IntV.I64.toInt_ofNat (by omega)
  refine safe_idiv (idiv_ok hoffs hfullD hD) ?_
  intro row hrow
  have hrow' := idiv_toInt hoffs hfullD hD hrow
  refine Tile2.safe_irem_lanes hoffs hfullD hD ?_
  intro col hcol
  refine safe_arg64 hsx63 ?_
  intro sx hsx
  -- the mask, the pointers into `x`
  have hmask : ∀ i : Idx [d.B],
      ((r.item * (IntV.ofNat d.B : I64) + o_a).ltS T64).get i = en i := by
    intro i
    rw [Tensor.get_ltS, IntV.lt_eq, show (r.item * (IntV.ofNat d.B : I64) + o_a).get i =
      (r.item * (IntV.ofNat d.B : I64) + o_a).get1 i.1 from rfl, hoffs i.1, hT64]
    simp only [en, Nat.cast_lt]
  have hkrow : ∀ i : Fin d.B, k i < d.T → k i / d.D < d.N := fun i h =>
    (Nat.div_lt_iff_lt_mul hD).mpr (by simpa [Inst.T] using h)
  have hptrx : ∀ i : Idx [d.B], en i = true →
      (d.x.ptr 0 + row * sx + col).get i = d.x.ptr (g i : Int) := by
    intro i hi
    have hkT := hen _ hi
    have hr1 := hkrow i.1 hkT
    have hxl := hd.x_len _ hr1
    have hmul : ((row.get i * sx)).toInt = ((k i.1 / d.D * d.sx : Nat) : Int) :=
      IntV.I64.mul_nat (hrow' i.1) hsx (by omega)
    rw [Tensor.get_ptrAddT, Tensor.get_ptrAdd, Tensor.get_muls, Buf.ptr_offset, Buf.ptr_offset,
      hmul, show col.get i = col.get1 i.1 from rfl, hcol i.1]
    simp only [g, xIdx]
    push_cast
    rw [zero_add]
  have hblk : ∀ i : Idx [d.B], en i = true → k i.1 ∈ blk d p := by
    intro i hi
    have := i.1.isLt
    exact ⟨Nat.le_add_right _ _, by simp only [k]; omega, hen i hi⟩
  refine safe_load_buf d.x (Buf.addrs_of_get d.x hmask hptrx)
    (fun i hi => xIdx_lt hd.x_len (hen i hi)) hvx ?_ ?_
  · rintro z ⟨j, ⟨i, hi, rfl⟩, hz⟩
    exact ⟨g i, ⟨k i.1, hblk i hi, rfl⟩, hz⟩
  intro bx16 hx16 _
  refine safe_cvt (g := .binary32)
    ⟨Or.inr (Or.inl rfl), Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl⟩ ?_
  intro bx hbx
  refine safe_minimum ?_
  intro bm hbm
  refine safe_abs ?_
  intro ba hba
  refine safe_neg ?_
  intro bna hbna
  refine safe_math (fn := .exp) (fun i => hetot _) ?_
  intro be hbe
  refine safe_add ?_
  intro bz hbz
  refine safe_math (fn := .log) (fun i => hltot _) ?_
  intro bl hbl
  refine safe_sub ?_
  intro bs hbs
  refine safe_div (fun i => hdtot _ _) ?_
  intro by' hby
  refine safe_arg64 (show d.D < 2 ^ 63 by omega) ?_
  intro sy hsy
  refine safe_cvt (g := .bfloat16)
    ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inr (Or.inl rfl), Or.inl rfl⟩ ?_
  intro by16 hby16
  refine safe_last ?_
  -- the store
  have hptry : ∀ i : Idx [d.B], en i = true →
      (d.y.ptr 0 + row * sy + col).get i = d.y.ptr (gy i : Int) := by
    intro i hi
    have hkT := hen _ hi
    have hdm := Nat.div_mul_le_self (k i.1) d.D
    have hmul : ((row.get i * sy)).toInt = ((k i.1 / d.D * d.D : Nat) : Int) :=
      IntV.I64.mul_nat (hrow' i.1) hsy (by omega)
    rw [Tensor.get_ptrAddT, Tensor.get_ptrAdd, Tensor.get_muls, Buf.ptr_offset, Buf.ptr_offset,
      hmul, show col.get i = col.get1 i.1 from rfl, hcol i.1]
    simp only [gy]
    have h1 : k i.1 / d.D * d.D + k i.1 % d.D = k i.1 := by
      rw [mul_comm]; exact Nat.div_add_mod (k i.1) d.D
    congr 1
    rw [zero_add]
    exact_mod_cast h1
  refine safe_store_buf d.y (Buf.addrs_of_get d.y hmask hptry)
    (fun i hi => by have := hen i hi; have := hd.y_len; simp only [gy]; omega) ?_ hvy ?_ ?_
  · intro i j _ _ h
    simp only [gy, k] at h
    exact Prod.ext (Fin.ext (by omega)) rfl
  · rintro z ⟨j, ⟨i, hi, rfl⟩, hz⟩
    exact ⟨gy i, hblk i hi, hz⟩
  intro m' hm' _ _
  refine safe_ret ?_
  -- `Q`: each element of the block
  intro kk hkk
  obtain ⟨hk1, hk2, hk3⟩ := hkk
  have hiB : kk - p * d.B < d.B := by omega
  let i : Idx [d.B] := (⟨kk - p * d.B, hiB⟩, ())
  have hki : k i.1 = kk := by simp only [k, i]; omega
  have hei : en i = true := by simp only [en, decide_eq_true_eq]; rw [hki]; exact hk3
  obtain ⟨v, hvr, hvf, hvroom⟩ := hin kk hk3
  have hxv : bx16.get i = v := by
    have h1 := hx16 i hei
    simp only [g] at h1
    rw [hki] at h1
    exact Option.some.inj (h1.symm.trans hvr)
  refine ⟨v, hvr, by16.get i, ?_, ?_⟩
  · have := hm' i hei
    simp only [gy] at this
    rwa [hki] at this
  · refine lane hexp hlog hfull hvf hd.t_fin hd.t_lo hd.t_hi hvroom ?_ (hbm i) (hba i) (hbna i)
      (hbe i) (hbz i) (hbl i) (hbs i) (hby i) (hby16 i)
    rw [← hxv]; exact hbx i

end Claims.LogsigmoidFwd
