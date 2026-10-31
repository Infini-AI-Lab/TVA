import AttnBwdPre.Err

/-!
# `attn_bwd_pre_f16`: the proof

`main` applies `TVA.meets_independent`. Program `p` of the grid (`p < gx·Z·H`, with
`gx = N_CTX // BLOCK_M`) has `program_id(0) = p mod gx` and `program_id(1) = p / gx`, and its lanes
`i < BLOCK_M` handle the rows `r = p·BLOCK_M + i` (`N_CTX = gx·BLOCK_M`): it reads the elements
`r·HEAD_DIM + k` of `O` and `DO` and writes element `r` of `Delta`. So the programs write disjoint
tiles of `Delta` (`Buf.tile`), and read only `O` and `DO`, which `Delta`'s elements do not meet.
`safe_program` walks the model line by line with the library's rules; `row_bound` (`AttnBwdPre.Err`)
gives each output.
-/

namespace Kernels.AttnBwdPre

open TVA TVA.Math.Float Block

/-! ## The launch -/

/-- The programs along axis 0: `N_CTX // BLOCK_M`. -/
def gx (d : Inst) : Nat := d.N_CTX / d.BLOCK_M

theorem launch_size (d : Inst) : (launch d).size = gx d * (d.Z * d.H) := by
  simp [Launch.size, launch, gx]

theorem block_pos {d : Inst} (hd : d.ok) : 0 < d.BLOCK_M := by
  obtain ⟨e, he⟩ := hd.block_pow2
  rw [he]; exact Nat.two_pow_pos e

theorem head_pos {d : Inst} (hd : d.ok) : 0 < d.HEAD_DIM := by
  obtain ⟨e, he⟩ := hd.head_pow2
  rw [he]; exact Nat.two_pow_pos e

/-- `N_CTX = gx · BLOCK_M` (`BLOCK_M` divides `N_CTX`). -/
theorem ctx_eq {d : Inst} (hd : d.ok) : d.N_CTX = gx d * d.BLOCK_M :=
  (Nat.div_mul_cancel hd.ctx_dvd).symm

theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hB := block_pos hd
  have hD := head_pos hd
  have hBle : d.BLOCK_M ≤ 2 ^ 20 := le_trans (Nat.le_mul_of_pos_right _ hD) hd.block_le
  have hDle : d.HEAD_DIM ≤ 2 ^ 20 := le_trans (Nat.le_mul_of_pos_left _ hB) hd.block_le
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, ?_⟩
  · show d.N_CTX / d.BLOCK_M < 2 ^ 31
    exact lt_of_le_of_lt (Nat.div_le_self _ _) hd.ctx_lt
  · show d.Z * d.H < 2 ^ 16
    exact hd.bh_lt
  · show 1 < 2 ^ 16
    norm_num
  · rw [launch_size]; exact hd.grid_lt
  · intro s hs
    obtain ⟨eB, heB⟩ := hd.block_pow2
    obtain ⟨eD, heD⟩ := hd.head_pow2
    have h1 : (1 : Nat) = 2 ^ 0 := rfl
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl
    · exact ⟨by simpa using ⟨eB, heB⟩, by simpa using hBle⟩
    · exact ⟨by simpa using ⟨eD, heD⟩, by simpa using hDle⟩
    · exact ⟨by simpa using ⟨⟨eB, heB⟩, ⟨0, h1⟩⟩, by simpa using hBle⟩
    · exact ⟨by simpa using ⟨⟨0, h1⟩, ⟨eD, heD⟩⟩, by simpa using hDle⟩
    · exact ⟨by simpa using ⟨⟨eB, heB⟩, ⟨eD, heD⟩⟩, by simpa using hd.block_le⟩

theorem launch_pid0 (d : Inst) (p : Nat) : tl.program_id ((launch d).ctx p) 0 = IntV.ofNat (p % gx d) := by
  simp [tl.program_id, Launch.ctx, launch, gx]

theorem launch_pid1 (d : Inst) {p : Nat} (hp : p < gx d * (d.Z * d.H)) :
    tl.program_id ((launch d).ctx p) 1 = IntV.ofNat (p / gx d) := by
  have hb : p / gx d < d.Z * d.H := Nat.div_lt_of_lt_mul hp
  simp [tl.program_id, Launch.ctx, launch, gx, Nat.mod_eq_of_lt (show p / (d.N_CTX / d.BLOCK_M) < d.Z * d.H from hb)]

/-! ## Footprints -/

/-- Every program's read footprint: the elements of `O` and `DO` the launch reads. -/
def R (d : Inst) : Set Int := d.O.elems (inSet d) ∪ d.DO.elems (inSet d)

/-- Program `p`'s write footprint: its tile of `Delta`, the rows `p·BLOCK_M + i`, `i < BLOCK_M`. -/
def W (d : Inst) (p : Nat) : Set Int := d.Delta.tile p d.BLOCK_M (d.Z * d.H * d.N_CTX)

/-- What program `p` leaves in memory `m`: every row of its tile as `OutSpec` says. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ r, Tile p d.BLOCK_M (d.Z * d.H * d.N_CTX) r → OutSpec d init m r

theorem W_sub (d : Inst) (p : Nat) : W d p ⊆ d.Delta.elems (outSet d) :=
  d.Delta.elems_mono fun _ hj => hj.2.2

theorem W_R {d : Inst} (hd : d.ok) (p : Nat) : Disjoint (W d p) (R d) :=
  Disjoint.union_right ((hd.Delta_O).mono_left (W_sub d p)) ((hd.Delta_DO).mono_left (W_sub d p))

/-! ## One program -/

/-- The `int32` offsets of one program do not wrap, and its rows lie below `Z·H·N_CTX`. -/
theorem prog_bounds {d : Inst} (hd : d.ok) {p : Nat} (hp : p < gx d * (d.Z * d.H)) :
    0 < d.N_CTX ∧ p % gx d * d.BLOCK_M + d.BLOCK_M ≤ d.N_CTX ∧
      p / gx d * d.HEAD_DIM * d.N_CTX < 2 ^ 31 ∧ p / gx d * d.HEAD_DIM < 2 ^ 31 ∧
      p / gx d * d.N_CTX < 2 ^ 31 ∧ p * d.BLOCK_M + d.BLOCK_M ≤ d.Z * d.H * d.N_CTX ∧
      p % gx d < 2 ^ 31 ∧ p / gx d < 2 ^ 31 := by
  have hB := block_pos hd
  have hD := head_pos hd
  have hN := ctx_eq hd
  have hgx : 0 < gx d := by
    rcases Nat.eq_zero_or_pos (gx d) with h | h
    · rw [h, zero_mul] at hp; omega
    · exact h
  have hb : p / gx d < d.Z * d.H := Nat.div_lt_of_lt_mul hp
  have ha : p % gx d < gx d := Nat.mod_lt _ hgx
  have hN0 : 0 < d.N_CTX := by rw [hN]; exact Nat.mul_pos hgx hB
  have hbD : p / gx d * d.HEAD_DIM * d.N_CTX ≤ (d.Z * d.H - 1) * d.HEAD_DIM * d.N_CTX :=
    Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (by omega))
  have hfit := hd.head_fit
  have hgxN : gx d ≤ d.N_CTX := Nat.div_le_self _ _
  have hNlt := hd.ctx_lt
  have hbh := hd.bh_lt
  refine ⟨hN0, ?_, by omega, ?_, ?_, ?_, by omega, by omega⟩
  · rw [hN]
    have : (p % gx d + 1) * d.BLOCK_M ≤ gx d * d.BLOCK_M := Nat.mul_le_mul_right _ ha
    nlinarith
  · have : p / gx d * d.HEAD_DIM ≤ p / gx d * d.HEAD_DIM * d.N_CTX := Nat.le_mul_of_pos_right _ hN0
    omega
  · have : p / gx d * d.N_CTX ≤ p / gx d * d.HEAD_DIM * d.N_CTX := by
      rw [Nat.mul_right_comm]; exact Nat.le_mul_of_pos_right _ hD
    omega
  · have h1 : (p + 1) * d.BLOCK_M ≤ gx d * (d.Z * d.H) * d.BLOCK_M := Nat.mul_le_mul_right _ hp
    have h2 : gx d * (d.Z * d.H) * d.BLOCK_M = d.Z * d.H * d.N_CTX := by rw [hN]; ring
    nlinarith

/-- The element of `O` (and `DO`) that lane `(i, k)` of program `p` reads: `(p·BLOCK_M + i)·HEAD_DIM + k`. -/
def gIn (d : Inst) (p : Nat) : Idx [d.BLOCK_M, d.HEAD_DIM] → Nat :=
  fun x => (p * d.BLOCK_M + x.1) * d.HEAD_DIM + x.2.1

/-- The element of `Delta` that lane `i` of program `p` writes: `p·BLOCK_M + i`. -/
def gOut (d : Inst) (p : Nat) : Idx [d.BLOCK_M] → Nat := fun x => p * d.BLOCK_M + x.1

/-- Lane `z` of `tl.arange(0, n)`, an `int32` below `2^31`. -/
theorem arange_lane {z : Nat} (h : z < 2 ^ 31) :
    (IntV.ofInt (((0 : Nat) : Int) + (z : Int)) : I32).toInt = (z : Int) := by
  rw [IntV.I32.toInt_ofInt] <;> push_cast
  · simp
  · constructor <;> omega

/-- The pointers of lines 258 and 259, lane by lane. -/
theorem ptr_in {d : Inst} (hd : d.ok) {p : Nat} (hp : p < gx d * (d.Z * d.H)) (b : Buf .f16) (x : Idx [d.BLOCK_M, d.HEAD_DIM]) :
    (b.ptr 0 + IntV.ofNat (p / gx d) * (IntV.ofNat d.HEAD_DIM : I32) * (IntV.ofNat d.N_CTX : I32) +
        (IntV.ofNat (p % gx d) * (IntV.ofNat d.BLOCK_M : I32) + tl.arange 0 d.BLOCK_M : Tensor [d.BLOCK_M] I32).col d.HEAD_DIM *
          (IntV.ofNat d.HEAD_DIM : I32) +
        (tl.arange 0 d.HEAD_DIM : Tensor [d.HEAD_DIM] I32).row d.BLOCK_M).get x = b.ptr (gIn d p x : Int) := by
  obtain ⟨hN0, haB, hbDN, hbD, -, -, ha31, hb31⟩ := prog_bounds hd hp
  have hN := ctx_eq hd
  have hNlt := hd.ctx_lt
  have hDle := hd.head_le
  have hrow := hd.row_fit
  have hi := x.1.isLt
  have hk := x.2.1.isLt
  have hpab := Nat.div_add_mod p (gx d)
  have hBv : (IntV.ofNat d.BLOCK_M : I32).toInt = d.BLOCK_M := IntV.I32.ofNat_nat (by omega)
  have hDv : (IntV.ofNat d.HEAD_DIM : I32).toInt = d.HEAD_DIM := IntV.I32.ofNat_nat (by omega)
  have hNv : (IntV.ofNat d.N_CTX : I32).toInt = d.N_CTX := IntV.I32.ofNat_nat hNlt
  have hav : (IntV.ofNat (p % gx d) : I32).toInt = (p % gx d : Nat) := IntV.I32.ofNat_nat ha31
  have hbv : (IntV.ofNat (p / gx d) : I32).toInt = (p / gx d : Nat) := IntV.I32.ofNat_nat hb31
  have hbD' := IntV.I32.mul_nat hbv hDv hbD
  have hbDN' := IntV.I32.mul_nat hbD' hNv hbDN
  have haB' := IntV.I32.mul_nat hav hBv (by omega)
  have hi' := arange_lane (z := x.1) (by omega)
  have hk' := arange_lane (z := x.2.1) (by omega)
  have hlane := IntV.I32.add_nat haB' hi' (by omega)
  have hlD := IntV.I32.mul_nat hlane hDv (show (p % gx d * d.BLOCK_M + x.1) * d.HEAD_DIM < 2 ^ 31 by
    have : (p % gx d * d.BLOCK_M + x.1) * d.HEAD_DIM ≤ (d.N_CTX - 1) * d.HEAD_DIM :=
      Nat.mul_le_mul_right _ (by omega)
    omega)
  show (((b.ptr 0).offset (IntV.ofNat (p / gx d) * (IntV.ofNat d.HEAD_DIM : I32) * (IntV.ofNat d.N_CTX : I32)).toInt).offset
      ((IntV.ofNat (p % gx d) * (IntV.ofNat d.BLOCK_M : I32) + IntV.ofInt (((0 : Nat) : Int) + ((x.1 : Nat) : Int))) *
        (IntV.ofNat d.HEAD_DIM : I32)).toInt).offset
      ((IntV.ofInt (((0 : Nat) : Int) + ((x.2.1 : Nat) : Int)) : I32).toInt) = _
  rw [hbDN', hlD, hk', Buf.ptr_offset, Buf.ptr_offset, Buf.ptr_offset]
  congr 1
  unfold gIn
  have h1 : p * d.BLOCK_M = p / gx d * d.N_CTX + p % gx d * d.BLOCK_M := by
    rw [hN]; conv_lhs => rw [← hpab]
    ring
  have e : (p * d.BLOCK_M + x.1) * d.HEAD_DIM + x.2.1 =
      p / gx d * d.HEAD_DIM * d.N_CTX + (p % gx d * d.BLOCK_M + x.1) * d.HEAD_DIM + x.2.1 := by
    rw [h1]; ring
  rw [e]; push_cast; ring

/-- The pointers of line 262, lane by lane. -/
theorem ptr_out {d : Inst} (hd : d.ok) {p : Nat} (hp : p < gx d * (d.Z * d.H)) (x : Idx [d.BLOCK_M]) :
    (d.Delta.ptr 0 + IntV.ofNat (p / gx d) * (IntV.ofNat d.N_CTX : I32) +
        (IntV.ofNat (p % gx d) * (IntV.ofNat d.BLOCK_M : I32) + tl.arange 0 d.BLOCK_M : Tensor [d.BLOCK_M] I32)).get x =
      d.Delta.ptr (gOut d p x : Int) := by
  obtain ⟨hN0, haB, -, -, hbN, -, ha31, hb31⟩ := prog_bounds hd hp
  have hN := ctx_eq hd
  have hNlt := hd.ctx_lt
  have hi := x.1.isLt
  have hpab := Nat.div_add_mod p (gx d)
  have hBv : (IntV.ofNat d.BLOCK_M : I32).toInt = d.BLOCK_M := IntV.I32.ofNat_nat (by omega)
  have hNv : (IntV.ofNat d.N_CTX : I32).toInt = d.N_CTX := IntV.I32.ofNat_nat hNlt
  have hav : (IntV.ofNat (p % gx d) : I32).toInt = (p % gx d : Nat) := IntV.I32.ofNat_nat ha31
  have hbv : (IntV.ofNat (p / gx d) : I32).toInt = (p / gx d : Nat) := IntV.I32.ofNat_nat hb31
  have hbN' := IntV.I32.mul_nat hbv hNv hbN
  have haB' := IntV.I32.mul_nat hav hBv (by omega)
  have hi' := arange_lane (z := x.1) (by omega)
  have hlane := IntV.I32.add_nat haB' hi' (by omega)
  show ((d.Delta.ptr 0).offset (IntV.ofNat (p / gx d) * (IntV.ofNat d.N_CTX : I32)).toInt).offset
      (IntV.ofNat (p % gx d) * (IntV.ofNat d.BLOCK_M : I32) + IntV.ofInt (((0 : Nat) : Int) + ((x.1 : Nat) : Int))).toInt = _
  rw [hbN', hlane, Buf.ptr_offset, Buf.ptr_offset]
  congr 1
  unfold gOut
  have h1 : p * d.BLOCK_M = p / gx d * d.N_CTX + p % gx d * d.BLOCK_M := by
    rw [hN]; conv_lhs => rw [← hpab]
    ring
  rw [h1]; push_cast; ring

/-- `ConvRel` from `float16` to `float32` has a result: the widening. -/
theorem conv_ex (P : Platform) (x : F16) : ∃ r, ConvRel P .f16 .f32 x r :=
  ⟨FloatV.cvt x, by
    show FloatRel P Format.binary16 Format.binary32 _ _
    unfold FloatRel
    rw [ite_eq_right (by decide)]
    exact Or.inl ⟨by unfold Format.Std; simp, by unfold Format.Std; simp, FloatV.NaNEq.refl _⟩⟩

/-- The widening of a finite `float16` to `float32` (any rounding mode: it is exact), up to NaN
payload, is finite with the same value. -/
theorem widen {x : F16} {r : F32} (hx : x.isFinite = true) (h : FloatV.NaNEq (FloatV.cvtWith .rn x) r) :
    r.isFinite = true ∧ r.toRat = x.toRat := by
  obtain ⟨hf, hv⟩ := FloatV.toRat_cvt_of_wider Format.binary16_regular Format.binary32_regular
    Format.wider_binary16_binary32 .rn hx
  rw [h.eq (FloatV.isFinite_iff.mp hf).1]
  exact ⟨hf, hv⟩

/-- `ConvRel` from `float16` to `float32` of a finite value: finite, the same value. -/
theorem conv_widen {P : Platform} {x : F16} {r : F32} (hx : x.isFinite = true)
    (h : ConvRel P .f16 .f32 x r) : r.isFinite = true ∧ r.toRat = x.toRat := by
  change FloatRel P Format.binary16 Format.binary32 x r at h
  unfold FloatRel at h
  rw [ite_eq_right (by decide)] at h
  rcases h with ⟨-, -, h⟩ | ⟨-, hF8, -⟩ | ⟨hF8, -, -⟩
  · exact widen hx h
  · exact absurd hF8 (by unfold Format.F8; decide)
  · exact absurd hF8 (by unfold Format.F8; decide)

/-- **One program.** Program `p` of the grid, run alone from `init`, is safe: it reads only `O`
and `DO`, writes only its tile of `Delta`, and leaves there what `OutSpec` says. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {p : Nat}
    (hp : p < gx d * (d.Z * d.H)) :
    Safe E (R d) (W d p) (fun _ m => Q d init p m) init ((model d).body ((launch d).ctx p)) := by
  obtain ⟨hvO, hvDO, hvDelta⟩ := hpre
  obtain ⟨-, -, -, -, -, hrows, -, -⟩ := prog_bounds hd hp
  have hD := head_pos hd
  have hin : ∀ x : Idx [d.BLOCK_M, d.HEAD_DIM], gIn d p x < d.Z * d.H * d.N_CTX * d.HEAD_DIM := by
    rintro ⟨i, k, ⟨⟩⟩
    unfold gIn
    have h1 : p * d.BLOCK_M + i + 1 ≤ d.Z * d.H * d.N_CTX := by have := i.isLt; omega
    have h2 : (p * d.BLOCK_M + i + 1) * d.HEAD_DIM ≤ d.Z * d.H * d.N_CTX * d.HEAD_DIM :=
      Nat.mul_le_mul_right _ h1
    have := k.isLt
    nlinarith
  have himg : Buf.img (gIn d p) (fun _ => true) ⊆ inSet d := by
    rintro _ ⟨x, -, rfl⟩; exact hin x
  simp only [model, _attn_bwd_preprocess]
  rw [launch_pid0, launch_pid1 d hp]
  -- line 258: `o = tl.load(O + …)`
  refine safe_load_buf d.O (g := gIn d p) (en := fun _ => true)
    (Buf.addrs_of_get_all d.O fun x => ptr_in hd hp d.O x)
    (fun x _ => lt_of_lt_of_le (hin x) hd.O_len) hvO
    (Set.Subset.trans (d.O.elems_mono himg) Set.subset_union_left) fun vO hO _ => ?_
  -- line 259: `do = tl.load(DO + …).to(tl.float32)`
  refine safe_load_buf d.DO (g := gIn d p) (en := fun _ => true)
    (Buf.addrs_of_get_all d.DO fun x => ptr_in hd hp d.DO x)
    (fun x _ => lt_of_lt_of_le (hin x) hd.DO_len) hvDO
    (Set.Subset.trans (d.DO.elems_mono himg) Set.subset_union_right) fun vDO hDO _ => ?_
  refine safe_cvt ⟨by unfold Format.Std; simp, by unfold Format.Std; simp, Or.inl rfl⟩ fun dd hdd => ?_
  -- line 260: `delta = tl.sum(o * do, axis=1)`
  refine safe_convert (fun _ => conv_ex E.P _) fun o32 ho32 => ?_
  refine safe_mul fun pr hpr => ?_
  refine safe_sum1c hD fun delta hdelta => ?_
  -- line 262: `tl.store(Delta + …, delta)`
  refine safe_last (safe_store_buf d.Delta (g := gOut d p) (en := fun _ => true)
    (Buf.addrs_of_get_all d.Delta fun x => ptr_out hd hp x)
    (fun x _ => by have := x.1.isLt; unfold gOut; have := hd.Delta_len; omega)
    (fun x y _ _ h => by
      obtain ⟨i, ⟨⟩⟩ := x; obtain ⟨j, ⟨⟩⟩ := y
      have : i = j := Fin.ext (by simp only [gOut] at h; omega)
      subst this; rfl)
    hvDelta ?_ fun m' hm' _ _ => safe_ret ?_)
  · rintro _ ⟨r, ⟨x, -, rfl⟩, hx⟩
    have hx1 := x.1.isLt
    have ht : Tile p d.BLOCK_M (d.Z * d.H * d.N_CTX) (gOut d p x) :=
      ⟨by unfold gOut; omega, by unfold gOut; omega, by unfold gOut; omega⟩
    exact d.Delta.span_sub_tile ht hx
  -- what the program leaves
  intro r hr
  obtain ⟨l, rfl⟩ := tile_lane hr
  have hl : (l : Nat) < d.BLOCK_M := by have := l.isLt; omega
  refine ⟨delta.get (⟨l, hl⟩, ()), hm' (⟨l, hl⟩, ()) rfl, fun vo vdo hvo hvdo => ?_⟩
  obtain ⟨t, hcov, ht⟩ := hdelta ⟨l, hl⟩
  -- the lanes of row `l`
  let av : Nat → F32 := fun k => if h : k < d.HEAD_DIM then o32.get (⟨l, hl⟩, ⟨k, h⟩, ()) else default
  let bv : Nat → F32 := fun k => if h : k < d.HEAD_DIM then dd.get (⟨l, hl⟩, ⟨k, h⟩, ()) else default
  have hO' : ∀ k (hk : k < d.HEAD_DIM), vO.get (⟨l, hl⟩, ⟨k, hk⟩, ()) = vo k := fun k hk =>
    Option.some.inj ((hO (⟨l, hl⟩, ⟨k, hk⟩, ()) rfl).symm.trans (hvo k hk).1)
  have hDO' : ∀ k (hk : k < d.HEAD_DIM), vDO.get (⟨l, hl⟩, ⟨k, hk⟩, ()) = vdo k := fun k hk =>
    Option.some.inj ((hDO (⟨l, hl⟩, ⟨k, hk⟩, ()) rfl).symm.trans (hvdo k hk).1)
  refine row_bound hd.head_le vo vdo av bv _ (fun k hk => (hvo k hk).2) (fun k hk => (hvdo k hk).2)
    (fun k hk => ?_) (fun k hk => ?_) (fun k hk => ?_) hcov ht
  · have := conv_widen (by rw [hO' k hk]; exact (hvo k hk).2) (ho32 (⟨l, hl⟩, ⟨k, hk⟩, ()))
    rw [hO' k hk] at this
    simpa only [av, dite_eq_left hk] using this
  · have := widen (by rw [hDO' k hk]; exact (hvdo k hk).2) (hdd (⟨l, hl⟩, ⟨k, hk⟩, ()))
    rw [hDO' k hk] at this
    simpa only [bv, dite_eq_left hk] using this
  · simp only [av, bv, dite_eq_left hk, Tensor.get2]
    exact hpr (⟨l, hl⟩, ⟨k, hk⟩, ())

/-! ## The claim -/

/-- `OutSpec` reads the final memory only at its element of `Delta`. -/
theorem outSpec_of_agree {d : Inst} {init m fin : Mem} {r : Nat} (h : OutSpec d init m r)
    (hag : ∀ x ∈ Mem.span .f32 (d.Delta.ptr ↑r).addr, fin x = m x) : OutSpec d init fin r := by
  obtain ⟨c, hc, hrest⟩ := h
  exact ⟨c, by rw [Mem.read_congr hag]; exact hc, hrest⟩

theorem main : ∀ (K : Claim.Behaviour Inst) (P : Platform), Assumptions K P → Spec K := by
  intro K P hA
  refine meets_independent hA.faithful (fun _ _ => rfl) (fun _ hd => launch_valid hd)
    (fun d _ _ => R d) (fun d _ i => W d i) (fun d init i m => Q d init i m) ?_ ?_ ?_ ?_
  · -- each program is safe
    intro d hd init hpre i
    exact safe_program hd hpre _ (lt_of_lt_of_eq i.isLt (launch_size d))
  · -- no program writes another's footprint
    intro d hd init _ i j hij
    have hij' : (i : Nat) ≠ j := fun h => hij (Fin.ext h)
    exact Disjoint.union_right (W_R hd i) (d.Delta.tile_disjoint hij')
  · -- no program reads what it writes
    intro d hd init _ i
    exact (W_R hd i).symm
  · -- the programs' results give `Post`
    intro d hd init _ fin hloc hfr
    refine ⟨fun bh m hbh hm => ?_, fun z hz => hfr z fun i hzi => hz (W_sub d i hzi)⟩
    have hB := block_pos hd
    have hr : bh * d.N_CTX + m < d.Z * d.H * d.N_CTX := by
      have : (bh + 1) * d.N_CTX ≤ d.Z * d.H * d.N_CTX := Nat.mul_le_mul_right _ hbh
      nlinarith
    have ht : Tile ((bh * d.N_CTX + m) / d.BLOCK_M) d.BLOCK_M (d.Z * d.H * d.N_CTX) (bh * d.N_CTX + m) :=
      tile_div hB hr
    have hp : (bh * d.N_CTX + m) / d.BLOCK_M < (launch d).size := by
      rw [launch_size, Nat.div_lt_iff_lt_mul hB]
      have h2 : gx d * (d.Z * d.H) * d.BLOCK_M = d.Z * d.H * d.N_CTX := by rw [ctx_eq hd]; ring
      rw [h2]; exact hr
    obtain ⟨mm, hQ, hagree⟩ := hloc ⟨_, hp⟩
    exact outSpec_of_agree (hQ _ ht) fun x hx => hagree x (Or.inr (d.Delta.span_sub_tile ht hx))

theorem consistent : Claim.Consistent Assumptions := by
  obtain ⟨K, htot, hf⟩ := exists_total_faithful Platform.any model launch
  exact ⟨K, Platform.any, htot, ⟨fun d _ => hf d⟩⟩

/-- One instance of the source's test (lines 631–647): `Z = 1`, `H = 2`, `N_CTX = 128`,
`HEAD_DIM = 64`, with the host's `BLOCK_M = 128`; `o` and `do` of `1·2·128·64 = 16384` elements and
`delta` of `256`, at the 16-byte-aligned addresses `2^21`, `2^22` and `3·2^21`. -/
def testInst : Inst where
  Z := 1
  H := 2
  N_CTX := 128
  HEAD_DIM := 64
  BLOCK_M := 128
  O := ⟨2097152, 16384⟩
  DO := ⟨4194304, 16384⟩
  Delta := ⟨6291456, 256⟩

/-- A memory in which every byte is present (and `0`). -/
def testMem : Mem := fun _ => some 0

/-- Elements of two buffers whose byte ranges do not meet do not meet. -/
theorem elems_disjoint_of_lt {d₁ d₂ : DType} {b : Buf d₁} {c : Buf d₂} {S T : Set Nat}
    (hS : ∀ j ∈ S, j < b.len) (hT : ∀ j ∈ T, j < c.len)
    (h : b.base + b.len * d₁.bytes ≤ c.base ∨ c.base + c.len * d₂.bytes ≤ b.base) :
    Disjoint (b.elems S) (c.elems T) := by
  refine Buf.elems_disjoint_of b c ?_ hS hT
  rw [Set.disjoint_left]
  intro x hx hy
  simp only [Buf.bytes, Set.mem_ofPred_eq] at hx hy
  omega

theorem satisfiable : Claim.Satisfiable Inst.ok Pre := by
  refine ⟨testInst, testMem, ⟨⟨7, rfl⟩, ⟨6, rfl⟩, by decide, by decide, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, ?_, ?_⟩, ?_, ?_, ?_⟩
  · exact elems_disjoint_of_lt (fun j hj => by simp only [outSet, Set.mem_ofPred_eq, testInst] at hj ⊢; omega)
      (fun j hj => by simp only [inSet, Set.mem_ofPred_eq, testInst] at hj ⊢; omega)
      (by norm_num [testInst, DType.bytes])
  · exact elems_disjoint_of_lt (fun j hj => by simp only [outSet, Set.mem_ofPred_eq, testInst] at hj ⊢; omega)
      (fun j hj => by simp only [inSet, Set.mem_ofPred_eq, testInst] at hj ⊢; omega)
      (by norm_num [testInst, DType.bytes])
  all_goals
    refine ⟨by norm_num [testInst], by norm_num [testInst, DType.bytes], by norm_num [testInst, DType.bytes],
      fun _ _ => by simp [testMem]⟩

end Kernels.AttnBwdPre
