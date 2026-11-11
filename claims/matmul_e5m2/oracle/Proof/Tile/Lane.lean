import Proof.Tile.Val

/-!
# The lane values

The model builds each lane from `offs_am`/`offs_bn` by `int32` arithmetic; this file proves that,
on the instances `Inst.ok` covers, that arithmetic computes the `Nat` offsets of `Val.lean`
(`aOff`, `bOff`, `cOff`) without wrap-around, that the masks are the `Nat` comparisons one
expects, and that the footprints of the loads and the store lie in `R`/`W`.
-/

namespace Claims.MatmulE5m2

open TVA

/-! ## The lanes of the model's expressions -/

/-- `offs_am`, before the reduction. -/
abbrev offsAM (d : Inst) (p : Nat) : Tensor [128] I32 :=
  IntV.ofInt (pidm d p) * (IntV.ofNat 128 : I32) + tl.arange 0 128

/-- `offs_am` after `tl.irem`, lane by lane `% M`. -/
abbrev offsAMi (d : Inst) (p : Nat) : Tensor [128] I32 :=
  Tensor.map2 (fun a b : I32 => IntV.ofInt (a.toInt % b.toInt)) (offsAM d p)
    (bc (IntV.ofNat d.M))

/-- `offs_bn`, before the reduction. -/
abbrev offsBN (d : Inst) (p : Nat) : Tensor [256] I32 :=
  IntV.ofInt (pidn d p) * (IntV.ofNat 256 : I32) + tl.arange 0 256

/-- `offs_bn` after `tl.irem`. -/
abbrev offsBNi (d : Inst) (p : Nat) : Tensor [256] I32 :=
  Tensor.map2 (fun a b : I32 => IntV.ofInt (a.toInt % b.toInt)) (offsBN d p)
    (bc (IntV.ofNat d.N))

/-- `offs_cm`: `pid_m·128 + arange 0 128`. -/
abbrev offsCM (d : Inst) (p : Nat) : Tensor [128] I32 := offsAM d p

/-- `offs_cn`: `pid_n·256 + arange 0 256`. -/
abbrev offsCN (d : Inst) (p : Nat) : Tensor [256] I32 := offsBN d p

/-- The mask of the store. -/
abbrev cMask (d : Inst) (p : Nat) : Tensor [128, 256] Bool :=
  (((offsCM d p).col 256).ltS (IntV.ofNat d.M)).and (((offsCN d p).row 128).ltS (IntV.ofNat d.N))

/-- The enabled lanes of the store. -/
abbrev cEn (d : Inst) (p : Nat) : Idx [128, 256] → Bool := fun i => (cMask d p).get i

/-- The mask of the `a` load. -/
abbrev dotMaskA (d : Inst) (t : Nat) : Tensor [128, 64] Bool :=
  ((tl.arange 0 64).row 128).ltS
    ((IntV.ofNat d.K : I32) - IntV.ofInt ((0 + (t : ℤ) * 1)) * (IntV.ofNat 64 : I32))

/-- The mask of the `b` load. -/
abbrev dotMaskB (d : Inst) (t : Nat) : Tensor [64, 256] Bool :=
  ((tl.arange 0 64).col 256).ltS
    ((IntV.ofNat d.K : I32) - IntV.ofInt ((0 + (t : ℤ) * 1)) * (IntV.ofNat 64 : I32))

/-- The `int32` offsets of the `a` tile load, `offs_am * stride_am + offs_k * stride_ak`. -/
private abbrev aT (d : Inst) (p : Nat) : Tensor [128, 64] I32 :=
  (offsAMi d p).col 64 * (IntV.ofNat d.stride_am : I32) +
    (tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32)

/-- The `int32` offsets of the `b` tile load, `offs_k * stride_bk + offs_bn * stride_bn`. -/
private abbrev bT (d : Inst) (p : Nat) : Tensor [64, 256] I32 :=
  (tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
    (offsBNi d p).row 64 * (IntV.ofNat d.stride_bn : I32)

/-- `64` divides `K` on covered instances. -/
private theorem steps_dvd (d : Inst) (hd : d.ok) : 64 ∣ d.K := by
  have h128 : 128 ∣ d.K := Nat.dvd_of_mod_eq_zero hd.K_dim.2.1
  exact dvd_trans (by norm_num : 64 ∣ 128) h128

/-- Every trip of the `K`-loop is a full step of `64`. -/
private theorem step_lt_K (d : Inst) (hd : d.ok) {t : Nat} (ht : t < steps d.K) :
    64 * t + 64 ≤ d.K := by
  have hsteps : steps d.K = d.K / 64 := steps_eq (steps_dvd d hd)
  have ht' : t < d.K / 64 := by rw [← hsteps]; exact ht
  have h1 : t + 1 ≤ d.K / 64 := Nat.succ_le_of_lt ht'
  have h2 : 64 * (d.K / 64) ≤ d.K := Nat.mul_div_le d.K 64
  have h3 : 64 * (t + 1) ≤ d.K := le_trans (Nat.mul_le_mul_left 64 h1) h2
  omega

/-! ### the values of `offs_am` / `offs_bn` -/

/-- Lane `x` of `offs_am` (before the `% M`) is the global row. -/
private theorem offsAM_val' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (x : Fin 128) :
    ((offsAM d p).get1 x).toInt = (ocm d p x.val : ℤ) := by
  have h1 : ((offsAM d p).get1 x).toInt
      = (IntV.ofInt (pidm d p) * (IntV.ofNat 128 : I32) + IntV.ofInt (0 + (x.val : ℤ))).toInt :=
    rfl
  refine h1.trans ?_
  have hpid : (IntV.ofInt (pidm d p) : I32).toInt = (pidm d p : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (pidm_lim hd hp))
  have h128 : (IntV.ofNat 128 : I32).toInt = (128 : ℤ) := toInt_ofNat' (by norm_num)
  have hocm_lt : ocm d p x.val < 2 ^ 31 := ocm_lt hd hp x.isLt
  have hprod : (pidm d p : ℤ) * 128 ≤ (ocm d p x.val : ℤ) := by
    have h : pidm d p * 128 ≤ ocm d p x.val := by unfold ocm; omega
    exact_mod_cast h
  have hmul : ((IntV.ofInt (pidm d p) * (IntV.ofNat 128 : I32) : I32)).toInt
      = (pidm d p : ℤ) * 128 :=
    mul_toInt hpid h128 (by positivity) (by
      have h2 : (ocm d p x.val : ℤ) < 2 ^ 31 := natCast_lt' hocm_lt
      linarith)
  have hx : (IntV.ofInt (0 + (x.val : ℤ)) : I32).toInt = 0 + (x.val : ℤ) :=
    toInt_ofInt' (by positivity) (by
      have h2 : (x.val : ℤ) < 128 := by exact_mod_cast x.isLt
      linarith)
  rw [add_toInt hmul hx (by positivity) (by
    have h2 : (ocm d p x.val : ℤ) < 2 ^ 31 := natCast_lt' hocm_lt
    have h3 : (pidm d p : ℤ) * 128 + (0 + (x.val : ℤ)) = (ocm d p x.val : ℤ) := by
      unfold ocm; push_cast; ring
    linarith)]
  unfold ocm
  push_cast
  ring

/-- Lane `y` of `offs_bn` (before the `% N`) is the global column. -/
private theorem offsBN_val' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (y : Fin 256) :
    ((offsBN d p).get1 y).toInt = (ocn d p y.val : ℤ) := by
  have h1 : ((offsBN d p).get1 y).toInt
      = (IntV.ofInt (pidn d p) * (IntV.ofNat 256 : I32) + IntV.ofInt (0 + (y.val : ℤ))).toInt :=
    rfl
  refine h1.trans ?_
  have hpid : (IntV.ofInt (pidn d p) : I32).toInt = (pidn d p : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (pidn_lim hd hp))
  have h256 : (IntV.ofNat 256 : I32).toInt = (256 : ℤ) := toInt_ofNat' (by norm_num)
  have hocn_lt : ocn d p y.val < 2 ^ 31 := ocn_lt hd hp y.isLt
  have hprod : (pidn d p : ℤ) * 256 ≤ (ocn d p y.val : ℤ) := by
    have h : pidn d p * 256 ≤ ocn d p y.val := by unfold ocn; omega
    exact_mod_cast h
  have hmul : ((IntV.ofInt (pidn d p) * (IntV.ofNat 256 : I32) : I32)).toInt
      = (pidn d p : ℤ) * 256 :=
    mul_toInt hpid h256 (by positivity) (by
      have h2 : (ocn d p y.val : ℤ) < 2 ^ 31 := natCast_lt' hocn_lt
      linarith)
  have hy : (IntV.ofInt (0 + (y.val : ℤ)) : I32).toInt = 0 + (y.val : ℤ) :=
    toInt_ofInt' (by positivity) (by
      have h2 : (y.val : ℤ) < 256 := by exact_mod_cast y.isLt
      linarith)
  rw [add_toInt hmul hy (by positivity) (by
    have h2 : (ocn d p y.val : ℤ) < 2 ^ 31 := natCast_lt' hocn_lt
    have h3 : (pidn d p : ℤ) * 256 + (0 + (y.val : ℤ)) = (ocn d p y.val : ℤ) := by
      unfold ocn; push_cast; ring
    linarith)]
  unfold ocn
  push_cast
  ring

/-- Lane `x` of `offs_am` after the `% M` is the row (`om`). -/
private theorem offsAM_get' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (x : Fin 128) :
    (offsAMi d p).get1 x = IntV.ofInt (om d p x.val) := by
  have h : (offsAMi d p).get1 x
      = IntV.ofInt (((offsAM d p).get1 x).toInt % (IntV.ofNat d.M : I32).toInt) := rfl
  refine h.trans ?_
  rw [offsAM_val' d hd hp x, toInt_ofNat' (M_lt hd)]
  have harith : ((ocm d p x.val : ℤ) % (d.M : ℤ)) = (om d p x.val : ℤ) := by
    unfold om
    rw [← Int.natCast_emod]
  rw [harith]

/-- Lane `y` of `offs_bn` after the `% N` is the column (`on`). -/
private theorem offsBN_get' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (y : Fin 256) :
    (offsBNi d p).get1 y = IntV.ofInt (on d p y.val) := by
  have h : (offsBNi d p).get1 y
      = IntV.ofInt (((offsBN d p).get1 y).toInt % (IntV.ofNat d.N : I32).toInt) := rfl
  refine h.trans ?_
  rw [offsBN_val' d hd hp y, toInt_ofNat' (N_lt hd)]
  have harith : ((ocn d p y.val : ℤ) % (d.N : ℤ)) = (on d p y.val : ℤ) := by
    unfold on
    rw [← Int.natCast_emod]
  rw [harith]

/-- Lane `x` of `offs_cm` is the global row. -/
private theorem offsCM_get1' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (x : Fin 128) :
    (offsAM d p).get1 x = IntV.ofInt (ocm d p x.val) := by
  have h1 : (offsAM d p).get1 x
      = IntV.ofInt (pidm d p) * (IntV.ofNat 128 : I32) + IntV.ofInt (0 + (x.val : ℤ)) := rfl
  refine h1.trans ?_
  have hval : (IntV.ofInt (pidm d p) * (IntV.ofNat 128 : I32) +
      IntV.ofInt (0 + (x.val : ℤ)) : I32).toInt = (ocm d p x.val : ℤ) := offsAM_val' d hd hp x
  rw [← IntV.ofInt_toInt (IntV.ofInt (pidm d p) * (IntV.ofNat 128 : I32) +
    IntV.ofInt (0 + (x.val : ℤ)))]
  rw [hval]

/-- Lane `y` of `offs_cn` is the global column. -/
private theorem offsCN_get1' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (y : Fin 256) :
    (offsBN d p).get1 y = IntV.ofInt (ocn d p y.val) := by
  have h1 : (offsBN d p).get1 y
      = IntV.ofInt (pidn d p) * (IntV.ofNat 256 : I32) + IntV.ofInt (0 + (y.val : ℤ)) := rfl
  refine h1.trans ?_
  have hval : (IntV.ofInt (pidn d p) * (IntV.ofNat 256 : I32) +
      IntV.ofInt (0 + (y.val : ℤ)) : I32).toInt = (ocn d p y.val : ℤ) := offsBN_val' d hd hp y
  rw [← IntV.ofInt_toInt (IntV.ofInt (pidn d p) * (IntV.ofNat 256 : I32) +
    IntV.ofInt (0 + (y.val : ℤ)))]
  rw [hval]

/-! ### the 2-D lane offsets -/

/-- The `a` tile offsets, as integers. -/
private theorem aOffs_toInt' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d)
    (i : Idx [128, 64]) :
    ((aT d p).get i).toInt = (aOff d p i.1 i.2.1 : ℤ) := by
  have h1 : ((aT d p).get i).toInt = ((aT d p).get (i.1, i.2.1, ())).toInt := rfl
  refine h1.trans ?_
  have h2 : (aT d p).get (i.1, i.2.1, ())
      = (offsAMi d p).get1 i.1 * (IntV.ofNat d.stride_am : I32) +
        IntV.ofInt (0 + (i.2.1.val : ℤ)) * (IntV.ofNat d.stride_ak : I32) := rfl
  refine (congrArg (fun z : I32 => z.toInt) h2).trans ?_
  have hx : ((offsAMi d p).get1 i.1).toInt = (om d p i.1.val : ℤ) := by
    rw [offsAM_get' d hd hp i.1]
    exact toInt_ofInt' (by positivity) (natCast_lt' (lt_trans (om_lt hp i.1.val) (M_lt hd)))
  have hsa : (IntV.ofNat d.stride_am : I32).toInt = (d.stride_am : ℤ) := toInt_ofNat' hd.am.2
  have hka : (IntV.ofNat d.stride_ak : I32).toInt = (d.stride_ak : ℤ) := toInt_ofNat' hd.ak.2
  have he : (IntV.ofInt (0 + (i.2.1.val : ℤ)) : I32).toInt = 0 + (i.2.1.val : ℤ) :=
    toInt_ofInt' (by positivity) (by
      have h2 : (i.2.1.val : ℤ) < 64 := by exact_mod_cast i.2.1.isLt
      linarith)
  have hlim : aOff d p i.1.val i.2.1.val < 2 ^ 31 := aOff_lim hd hp i.2.1.isLt
  have hp1 : (((offsAMi d p).get1 i.1) * (IntV.ofNat d.stride_am : I32)).toInt
      = (om d p i.1.val : ℤ) * (d.stride_am : ℤ) := by
    refine mul_toInt hx hsa (by positivity) ?_
    have hle : om d p i.1.val * d.stride_am ≤ aOff d p i.1.val i.2.1.val := by
      unfold aOff; omega
    have h2 : ((om d p i.1.val * d.stride_am : Nat) : ℤ) < (2 ^ 31 : ℤ) :=
      natCast_lt' (lt_of_le_of_lt hle hlim)
    push_cast at h2 ⊢
    linarith
  have hp2 : ((IntV.ofInt (0 + (i.2.1.val : ℤ)) * (IntV.ofNat d.stride_ak : I32))).toInt
      = (0 + (i.2.1.val : ℤ)) * (d.stride_ak : ℤ) := by
    refine mul_toInt he hka (by positivity) ?_
    have hle : (0 + (i.2.1.val : ℤ)) * (d.stride_ak : ℤ) ≤
        (aOff d p i.1.val i.2.1.val : ℤ) := by
      have h1 : i.2.1.val * d.stride_ak ≤ aOff d p i.1.val i.2.1.val := by
        unfold aOff; omega
      have h2 : ((i.2.1.val * d.stride_ak : Nat) : ℤ) ≤
          (aOff d p i.1.val i.2.1.val : ℤ) := by exact_mod_cast h1
      push_cast at h2 ⊢
      linarith
    have h3 : (aOff d p i.1.val i.2.1.val : ℤ) < 2 ^ 31 := natCast_lt' hlim
    linarith
  rw [add_toInt hp1 hp2 (by positivity) (by
    have h3 : (aOff d p i.1.val i.2.1.val : ℤ) < 2 ^ 31 := natCast_lt' hlim
    have heq : (om d p i.1.val : ℤ) * (d.stride_am : ℤ) +
        (0 + (i.2.1.val : ℤ)) * (d.stride_ak : ℤ) = (aOff d p i.1.val i.2.1.val : ℤ) := by
      unfold aOff; push_cast; ring
    linarith)]
  unfold aOff
  push_cast
  ring

/-- The `b` tile offsets, as integers. -/
private theorem bOffs_toInt' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d)
    (i : Idx [64, 256]) :
    ((bT d p).get i).toInt = (bOff d p i.2.1 i.1 : ℤ) := by
  have h1 : ((bT d p).get i).toInt = ((bT d p).get (i.1, i.2.1, ())).toInt := rfl
  refine h1.trans ?_
  have h2 : (bT d p).get (i.1, i.2.1, ())
      = IntV.ofInt (0 + (i.1.val : ℤ)) * (IntV.ofNat d.stride_bk : I32) +
        (offsBNi d p).get1 i.2.1 * (IntV.ofNat d.stride_bn : I32) := rfl
  refine (congrArg (fun z : I32 => z.toInt) h2).trans ?_
  have hy : ((offsBNi d p).get1 i.2.1).toInt = (on d p i.2.1.val : ℤ) := by
    rw [offsBN_get' d hd hp i.2.1]
    exact toInt_ofInt' (by positivity) (natCast_lt' (lt_trans (on_lt hd p i.2.1.val) (N_lt hd)))
  have hsb : (IntV.ofNat d.stride_bk : I32).toInt = (d.stride_bk : ℤ) := toInt_ofNat' hd.bk.2
  have hnb : (IntV.ofNat d.stride_bn : I32).toInt = (d.stride_bn : ℤ) := toInt_ofNat' hd.bn.2
  have he : (IntV.ofInt (0 + (i.1.val : ℤ)) : I32).toInt = 0 + (i.1.val : ℤ) :=
    toInt_ofInt' (by positivity) (by
      have h2 : (i.1.val : ℤ) < 64 := by exact_mod_cast i.1.isLt
      linarith)
  have hlim : bOff d p i.2.1.val i.1.val < 2 ^ 31 := bOff_lim hd i.1.isLt
  have hp1 : ((IntV.ofInt (0 + (i.1.val : ℤ)) * (IntV.ofNat d.stride_bk : I32))).toInt
      = (0 + (i.1.val : ℤ)) * (d.stride_bk : ℤ) := by
    refine mul_toInt he hsb (by positivity) ?_
    have hle : (0 + (i.1.val : ℤ)) * (d.stride_bk : ℤ) ≤
        (bOff d p i.2.1.val i.1.val : ℤ) := by
      have h1 : i.1.val * d.stride_bk ≤ bOff d p i.2.1.val i.1.val := by
        unfold bOff; omega
      have h2 : ((i.1.val * d.stride_bk : Nat) : ℤ) ≤
          (bOff d p i.2.1.val i.1.val : ℤ) := by exact_mod_cast h1
      push_cast at h2 ⊢
      linarith
    have h3 : (bOff d p i.2.1.val i.1.val : ℤ) < 2 ^ 31 := natCast_lt' hlim
    linarith
  have hp2 : (((offsBNi d p).get1 i.2.1) * (IntV.ofNat d.stride_bn : I32)).toInt
      = (on d p i.2.1.val : ℤ) * (d.stride_bn : ℤ) := by
    refine mul_toInt hy hnb (by positivity) ?_
    have hle : on d p i.2.1.val * d.stride_bn ≤ bOff d p i.2.1.val i.1.val := by
      unfold bOff; omega
    have h2 : ((on d p i.2.1.val * d.stride_bn : Nat) : ℤ) < (2 ^ 31 : ℤ) :=
      natCast_lt' (lt_of_le_of_lt hle hlim)
    push_cast at h2 ⊢
    linarith
  rw [add_toInt hp1 hp2 (by positivity) (by
    have h3 : (bOff d p i.2.1.val i.1.val : ℤ) < 2 ^ 31 := natCast_lt' hlim
    have heq : (0 + (i.1.val : ℤ)) * (d.stride_bk : ℤ) +
        (on d p i.2.1.val : ℤ) * (d.stride_bn : ℤ) = (bOff d p i.2.1.val i.1.val : ℤ) := by
      unfold bOff; push_cast; ring
    linarith)]
  unfold bOff
  push_cast
  ring

/-! ### the pointer lanes -/

/-- The `a` tile pointers are those of the offsets. -/
private theorem aPtrs_eq' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d)
    (i : Idx [128, 64]) :
    ((d.a.ptr 0) + aT d p).get i = d.a.ptr (aOff d p i.1 i.2.1) := by
  have h1 : ((d.a.ptr 0) + aT d p).get i
      = ((d.a.ptr 0) + aT d p).get (i.1, i.2.1, ()) := rfl
  refine h1.trans ?_
  have h2 : ((d.a.ptr 0) + aT d p).get (i.1, i.2.1, ())
      = (d.a.ptr 0).offset (((aT d p).get (i.1, i.2.1, ())).toInt) := rfl
  refine h2.trans ?_
  have h3 : (((aT d p).get (i.1, i.2.1, ())).toInt) = (aOff d p i.1 i.2.1 : ℤ) :=
    aOffs_toInt' d hd hp i
  refine (congrArg (fun k => (d.a.ptr 0).offset k) h3).trans ?_
  simp [Buf.ptr, Ptr.offset]

/-- The `b` tile pointers are those of the offsets. -/
private theorem bPtrs_eq' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d)
    (i : Idx [64, 256]) :
    ((d.b.ptr 0) + bT d p).get i = d.b.ptr (bOff d p i.2.1 i.1) := by
  have h1 : ((d.b.ptr 0) + bT d p).get i
      = ((d.b.ptr 0) + bT d p).get (i.1, i.2.1, ()) := rfl
  refine h1.trans ?_
  have h2 : ((d.b.ptr 0) + bT d p).get (i.1, i.2.1, ())
      = (d.b.ptr 0).offset (((bT d p).get (i.1, i.2.1, ())).toInt) := rfl
  refine h2.trans ?_
  have h3 : (((bT d p).get (i.1, i.2.1, ())).toInt) = (bOff d p i.2.1 i.1 : ℤ) :=
    bOffs_toInt' d hd hp i
  refine (congrArg (fun k => (d.b.ptr 0).offset k) h3).trans ?_
  simp [Buf.ptr, Ptr.offset]

/-! ### the masks -/

/-- A lane of the store mask is enabled exactly when its element is in the matrix. -/
private theorem cMask_get' (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d)
    (i : Idx [128, 256]) :
    cEn d p i = true ↔ ocm d p i.1 < d.M ∧ ocn d p i.2.1 < d.N := by
  have hA : ((offsCM d p).col 256).get (i.1, i.2.1, ()) = IntV.ofInt (ocm d p i.1.val) := by
    have e1 : ((offsCM d p).col 256).get (i.1, i.2.1, ()) = (offsCM d p).get1 i.1 := rfl
    exact e1.trans (offsCM_get1' d hd hp i.1)
  have hB : ((offsCN d p).row 128).get (i.1, i.2.1, ()) = IntV.ofInt (ocn d p i.2.1.val) := by
    have e1 : ((offsCN d p).row 128).get (i.1, i.2.1, ()) = (offsCN d p).get1 i.2.1 := rfl
    exact e1.trans (offsCN_get1' d hd hp i.2.1)
  have h3 : (cMask d p).get (i.1, i.2.1, ())
      = (IntV.lt (((offsCM d p).col 256).get (i.1, i.2.1, ()) : I32) (IntV.ofNat d.M) &&
         IntV.lt (((offsCN d p).row 128).get (i.1, i.2.1, ()) : I32) (IntV.ofNat d.N)) := rfl
  have h4 : (IntV.lt (((offsCM d p).col 256).get (i.1, i.2.1, ()) : I32) (IntV.ofNat d.M) &&
         IntV.lt (((offsCN d p).row 128).get (i.1, i.2.1, ()) : I32) (IntV.ofNat d.N))
      = (IntV.lt (IntV.ofInt (ocm d p i.1.val) : I32) (IntV.ofNat d.M) &&
         IntV.lt (IntV.ofInt (ocn d p i.2.1.val) : I32) (IntV.ofNat d.N)) := by
    rw [hA, hB]
  have hmain : cEn d p i
      = (decide ((ocm d p i.1.val : ℤ) < (d.M : ℤ)) &&
         decide ((ocn d p i.2.1.val : ℤ) < (d.N : ℤ))) := by
    have e0 : cEn d p i = (cMask d p).get (i.1, i.2.1, ()) := rfl
    refine e0.trans ?_
    refine h3.trans ?_
    refine h4.trans ?_
    rw [IntV.lt_eq, IntV.lt_eq]
    have hoM : (IntV.ofInt (ocm d p i.1.val) : I32).toInt = (ocm d p i.1.val : ℤ) :=
      toInt_ofInt' (by positivity) (natCast_lt' (ocm_lt hd hp i.1.isLt))
    have hoN : (IntV.ofInt (ocn d p i.2.1.val) : I32).toInt = (ocn d p i.2.1.val : ℤ) :=
      toInt_ofInt' (by positivity) (natCast_lt' (ocn_lt hd hp i.2.1.isLt))
    have hM : (IntV.ofNat d.M : I32).toInt = (d.M : ℤ) := toInt_ofNat' (M_lt hd)
    have hN : (IntV.ofNat d.N : I32).toInt = (d.N : ℤ) := toInt_ofNat' (N_lt hd)
    rw [hoM, hoN, hM, hN]
  rw [hmain]
  rw [Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨by exact_mod_cast h1, by exact_mod_cast h2⟩
  · rintro ⟨h1, h2⟩
    exact ⟨by exact_mod_cast h1, by exact_mod_cast h2⟩

/-- The bound `K - t·64` of trip `t`'s load masks, as an integer. -/
private theorem dotK_toInt' (d : Inst) (hd : d.ok) {t : Nat} (ht : t < steps d.K) :
    ((IntV.ofNat d.K : I32) - IntV.ofInt (0 + (t : ℤ) * 1) * (IntV.ofNat 64 : I32)).toInt
      = (d.K : ℤ) - (t : ℤ) * 64 := by
  have hKlt : d.K < 2 ^ 31 := by have := hd.K_le; omega
  have hKv : (IntV.ofNat d.K : I32).toInt = (d.K : ℤ) := toInt_ofNat' hKlt
  have hKge : 64 * t + 64 ≤ d.K := step_lt_K d hd ht
  have hK64 : (64 * t : ℕ) ≤ d.K := by omega
  have hcast : ((64 * t : ℕ) : ℤ) = (t : ℤ) * 64 := by push_cast; ring
  have hstep : (IntV.ofInt (0 + (t : ℤ) * 1) : I32).toInt = 0 + (t : ℤ) * 1 := by
    refine toInt_ofInt' (by positivity) ?_
    have h6 : (d.K : ℤ) < 2 ^ 31 := by exact_mod_cast hKlt
    have h4 : ((64 * t : ℕ) : ℤ) ≤ (d.K : ℤ) := by exact_mod_cast hK64
    linarith
  have h64 : (IntV.ofNat 64 : I32).toInt = (64 : ℤ) := toInt_ofNat' (by norm_num)
  have hmul : ((IntV.ofInt (0 + (t : ℤ) * 1) * (IntV.ofNat 64 : I32) : I32)).toInt
      = (0 + (t : ℤ) * 1) * 64 := by
    refine mul_toInt hstep h64 (by positivity) ?_
    have h6 : (d.K : ℤ) < 2 ^ 31 := by exact_mod_cast hKlt
    have h4 : ((64 * t : ℕ) : ℤ) ≤ (d.K : ℤ) := by exact_mod_cast hK64
    linarith
  have h0 : 0 ≤ (d.K : ℤ) - (0 + (t : ℤ) * 1) * 64 := by
    have h4 : ((64 * t : ℕ) : ℤ) ≤ (d.K : ℤ) := by exact_mod_cast hK64
    linarith
  have hlt : (d.K : ℤ) - (0 + (t : ℤ) * 1) * 64 < 2 ^ 31 := by
    have h6 : (d.K : ℤ) < 2 ^ 31 := by exact_mod_cast hKlt
    have hnn : (0 : ℤ) ≤ (t : ℤ) * 1 := by positivity
    linarith
  rw [sub_toInt hKv hmul h0 hlt]
  ring

/-- Every lane of trip `t`'s `a` load mask is enabled (`64 ∣ K`). -/
private theorem dotMaskA_get' (d : Inst) (hd : d.ok) {t : Nat} (ht : t < steps d.K)
    (i : Idx [128, 64]) : (dotMaskA d t).get i = true := by
  have h1 : (dotMaskA d t).get i = (dotMaskA d t).get (i.1, i.2.1, ()) := rfl
  refine h1.trans ?_
  have h2 : (dotMaskA d t).get (i.1, i.2.1, ())
      = IntV.lt (IntV.ofInt (0 + (i.2.1.val : ℤ)))
          ((IntV.ofNat d.K : I32) - IntV.ofInt (0 + (t : ℤ) * 1) * (IntV.ofNat 64 : I32)) := rfl
  refine h2.trans ?_
  rw [IntV.lt_eq, decide_eq_true_iff]
  have ha : (IntV.ofInt (0 + (i.2.1.val : ℤ)) : I32).toInt = 0 + (i.2.1.val : ℤ) :=
    toInt_ofInt' (by positivity) (by
      have h2 : (i.2.1.val : ℤ) < 64 := by exact_mod_cast i.2.1.isLt
      linarith)
  have hb := dotK_toInt' d hd ht
  rw [ha, hb]
  have hi : i.2.1.val < 64 := i.2.1.isLt
  have hKge : 64 * t + 64 ≤ d.K := step_lt_K d hd ht
  omega

/-- Every lane of trip `t`'s `b` load mask is enabled (`64 ∣ K`). -/
private theorem dotMaskB_get' (d : Inst) (hd : d.ok) {t : Nat} (ht : t < steps d.K)
    (i : Idx [64, 256]) : (dotMaskB d t).get i = true := by
  have h1 : (dotMaskB d t).get i = (dotMaskB d t).get (i.1, i.2.1, ()) := rfl
  refine h1.trans ?_
  have h2 : (dotMaskB d t).get (i.1, i.2.1, ())
      = IntV.lt (IntV.ofInt (0 + (i.1.val : ℤ)))
          ((IntV.ofNat d.K : I32) - IntV.ofInt (0 + (t : ℤ) * 1) * (IntV.ofNat 64 : I32)) := rfl
  refine h2.trans ?_
  rw [IntV.lt_eq, decide_eq_true_iff]
  have ha : (IntV.ofInt (0 + (i.1.val : ℤ)) : I32).toInt = 0 + (i.1.val : ℤ) :=
    toInt_ofInt' (by positivity) (by
      have h2 : (i.1.val : ℤ) < 64 := by exact_mod_cast i.1.isLt
      linarith)
  have hb := dotK_toInt' d hd ht
  rw [ha, hb]
  have hi : i.1.val < 64 := i.1.isLt
  have hKge : 64 * t + 64 ≤ d.K := step_lt_K d hd ht
  omega

/-! ### the strides -/

/-- The pointer step `BLOCK_SIZE_K * stride_ak`, as an integer. -/
private theorem stepAK_toInt' (d : Inst) (hd : d.ok) :
    ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_ak : I32)).toInt = 64 * (d.stride_ak : ℤ) := by
  have h64 : (IntV.ofNat 64 : I32).toInt = (64 : ℤ) := toInt_ofNat' (by norm_num)
  have hak : (IntV.ofNat d.stride_ak : I32).toInt = (d.stride_ak : ℤ) := toInt_ofNat' hd.ak.2
  have hlt : (64 : ℤ) * (d.stride_ak : ℤ) < 2 ^ 31 := by
    have h2 : 64 * d.stride_ak < 2 ^ 31 := by have := hd.a_fit; omega
    have h3 : ((64 * d.stride_ak : Nat) : ℤ) < (2 ^ 31 : ℤ) := natCast_lt' h2
    push_cast at h3 ⊢
    linarith
  rw [mul_toInt h64 hak (by positivity) hlt]

/-- The pointer step `BLOCK_SIZE_K * stride_bk`, as an integer. -/
private theorem stepBK_toInt' (d : Inst) (hd : d.ok) :
    ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_bk : I32)).toInt = 64 * (d.stride_bk : ℤ) := by
  have h64 : (IntV.ofNat 64 : I32).toInt = (64 : ℤ) := toInt_ofNat' (by norm_num)
  have hbk : (IntV.ofNat d.stride_bk : I32).toInt = (d.stride_bk : ℤ) := toInt_ofNat' hd.bk.2
  have hlt : (64 : ℤ) * (d.stride_bk : ℤ) < 2 ^ 31 := by
    have h2 : 64 * d.stride_bk < 2 ^ 31 := by have := hd.b_fit; omega
    have h3 : ((64 * d.stride_bk : Nat) : ℤ) < (2 ^ 31 : ℤ) := natCast_lt' h2
    push_cast at h3 ⊢
    linarith
  rw [mul_toInt h64 hbk (by positivity) hlt]

/-- The row part `stride_cm * offs_cm` of the store offsets, as an integer. -/
private theorem cmOff_toInt' (d : Inst) (hd : d.ok) {p x : Nat} (hx : ocm d p x < d.M) :
    ((IntV.ofNat d.stride_cm : I32) * IntV.ofInt (ocm d p x)).toInt
      = (d.stride_cm : ℤ) * (ocm d p x : ℤ) := by
  have hcm : (IntV.ofNat d.stride_cm : I32).toInt = (d.stride_cm : ℤ) := toInt_ofNat' hd.cm.2
  have hocm : (IntV.ofInt (ocm d p x) : I32).toInt = (ocm d p x : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (lt_trans hx (M_lt hd)))
  have hlt : (d.stride_cm : ℤ) * (ocm d p x : ℤ) < 2 ^ 31 := by
    have h1 : ocm d p x ≤ d.M - 1 := by omega
    have h2 : ocm d p x * d.stride_cm ≤ (d.M - 1) * d.stride_cm := Nat.mul_le_mul_right _ h1
    have h4 : ocm d p x * d.stride_cm < 2 ^ 31 := lt_of_le_of_lt h2 hd.cm_fit
    have h5 : ((ocm d p x * d.stride_cm : Nat) : ℤ) < (2 ^ 31 : ℤ) := natCast_lt' h4
    push_cast at h5 ⊢
    linarith
  rw [mul_toInt hcm hocm (by positivity) hlt]

/-- The column part `stride_cn * offs_cn` of the store offsets, as an integer. -/
private theorem cnOff_toInt' (d : Inst) (hd : d.ok) {p y : Nat} (hy : ocn d p y < d.N) :
    ((IntV.ofNat d.stride_cn : I32) * IntV.ofInt (ocn d p y)).toInt
      = (d.stride_cn : ℤ) * (ocn d p y : ℤ) := by
  have hcn : (IntV.ofNat d.stride_cn : I32).toInt = (d.stride_cn : ℤ) := toInt_ofNat' hd.cn.2
  have hocn : (IntV.ofInt (ocn d p y) : I32).toInt = (ocn d p y : ℤ) :=
    toInt_ofInt' (by positivity) (natCast_lt' (lt_trans hy (N_lt hd)))
  have hlt : (d.stride_cn : ℤ) * (ocn d p y : ℤ) < 2 ^ 31 := by
    have h1 : ocn d p y ≤ d.N - 1 := by omega
    have h2 : ocn d p y * d.stride_cn ≤ (d.N - 1) * d.stride_cn := Nat.mul_le_mul_right _ h1
    have h4 : ocn d p y * d.stride_cn < 2 ^ 31 := lt_of_le_of_lt h2 hd.cn_fit
    have h5 : ((ocn d p y * d.stride_cn : Nat) : ℤ) < (2 ^ 31 : ℤ) := natCast_lt' h4
    push_cast at h5 ⊢
    linarith
  rw [mul_toInt hcn hocn (by positivity) hlt]

/-! ### injectivity -/

/-- Distinct enabled lanes of the store write distinct offsets. -/
private theorem cImg_inj' {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) :
    ∀ i j, cEn d p i = true → cEn d p j = true →
      cOff d p i.1 i.2.1 = cOff d p j.1 j.2.1 → i = j := by
  intro i j hi hj hoff
  have hi' := (cMask_get' d hd hp i).mp hi
  have hj' := (cMask_get' d hd hp j).mp hj
  obtain ⟨hx, hy⟩ := cOff_inj hd hi'.1 hi'.2 hj'.1 hj'.2 hoff
  have hxi : i.1 = j.1 := Fin.ext hx
  have hyi : i.2.1 = j.2.1 := Fin.ext hy
  exact (show i = (i.1, i.2.1, ()) from rfl).trans
    ((show (i.1, i.2.1, ()) = (j.1, j.2.1, ()) from by rw [hxi, hyi]).trans
      (show (j.1, j.2.1, ()) = j from rfl))

/-! ### `offs_am` / `offs_bn` -/

/-! ### The pointer lanes -/

/-- The `a` pointer lane is element `aOff` of `a`. -/
theorem aPtrs_eq (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (i : Idx [128, 64]) :
    ((d.a.ptr 0) + ((offsAMi d p).col 64 * (IntV.ofNat d.stride_am : I32) +
      (tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32))).get i
      = d.a.ptr (aOff d p i.1 i.2.1) := by
  exact aPtrs_eq' d hd hp i

/-- The `b` pointer lane is element `bOff` of `b`. -/
theorem bPtrs_eq (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (i : Idx [64, 256]) :
    ((d.b.ptr 0) + ((tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
      (offsBNi d p).row 64 * (IntV.ofNat d.stride_bn : I32))).get i
      = d.b.ptr (bOff d p i.2.1 i.1) := by
  exact bPtrs_eq' d hd hp i

/-- The `c` pointer lane is element `cOff` of `c`, on the lanes the store mask enables.  (The
lane's two `int32` products are added to the pointer one after another, each converted to `ℤ`
exactly, so only the products need not wrap: the mask gives `ocm < M`, `ocn < N`, and `cm_fit`,
`cn_fit` bound them.  `cOff` itself — with `ocm`, `ocn` unreduced — is the element index the model
addresses.) -/
theorem cPtrs_eq (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (i : Idx [128, 256])
    (hi : cEn d p i = true) :
    ((d.c.ptr 0) + (IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256 +
      (IntV.ofNat d.stride_cn : I32) * (offsCN d p).row 128).get i
      = d.c.ptr (cOff d p i.1 i.2.1) := by
  have h1 : ((d.c.ptr 0) + (IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256 +
      (IntV.ofNat d.stride_cn : I32) * (offsCN d p).row 128).get i
      = ((d.c.ptr 0) + (IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256 +
      (IntV.ofNat d.stride_cn : I32) * (offsCN d p).row 128).get (i.1, i.2.1, ()) := rfl
  refine h1.trans ?_
  have e1 : ((d.c.ptr 0) + (IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256 +
      (IntV.ofNat d.stride_cn : I32) * (offsCN d p).row 128).get (i.1, i.2.1, ())
      = (((d.c.ptr 0) + (IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256).get
          (i.1, i.2.1, ())).offset
        ((((IntV.ofNat d.stride_cn : I32) * (offsCN d p).row 128).get
          (i.1, i.2.1, ())).toInt) := rfl
  rw [e1]
  have e2 : ((d.c.ptr 0) + (IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256).get
      (i.1, i.2.1, ())
      = (d.c.ptr 0).offset
        ((((IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256).get
          (i.1, i.2.1, ())).toInt) := rfl
  rw [e2, Ptr.offset_offset]
  have hx := (cMask_get' d hd hp i).mp hi
  have hA : ((offsCM d p).col 256).get (i.1, i.2.1, ()) = IntV.ofInt (ocm d p i.1.val) :=
    (rfl : ((offsCM d p).col 256).get (i.1, i.2.1, ()) = (offsCM d p).get1 i.1).trans
      (offsCM_get1' d hd hp i.1)
  have hB : ((offsCN d p).row 128).get (i.1, i.2.1, ()) = IntV.ofInt (ocn d p i.2.1.val) :=
    (rfl : ((offsCN d p).row 128).get (i.1, i.2.1, ()) = (offsCN d p).get1 i.2.1).trans
      (offsCN_get1' d hd hp i.2.1)
  have hT1 : (((IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256).get
      (i.1, i.2.1, ())).toInt = (d.stride_cm : ℤ) * (ocm d p i.1.val : ℤ) := by
    have e : ((IntV.ofNat d.stride_cm : I32) * (offsCM d p).col 256).get (i.1, i.2.1, ())
        = (IntV.ofNat d.stride_cm : I32) * ((offsCM d p).col 256).get (i.1, i.2.1, ()) := rfl
    rw [e, hA]
    exact cmOff_toInt' d hd hx.1
  have hT2 : (((IntV.ofNat d.stride_cn : I32) * (offsCN d p).row 128).get
      (i.1, i.2.1, ())).toInt = (d.stride_cn : ℤ) * (ocn d p i.2.1.val : ℤ) := by
    have e : ((IntV.ofNat d.stride_cn : I32) * (offsCN d p).row 128).get (i.1, i.2.1, ())
        = (IntV.ofNat d.stride_cn : I32) * ((offsCN d p).row 128).get (i.1, i.2.1, ()) := rfl
    rw [e, hB]
    exact cnOff_toInt' d hd hx.2
  rw [hT1, hT2]
  simp only [Buf.ptr, Ptr.offset]
  have hc : ((cOff d p i.1 i.2.1 : Nat) : ℤ)
      = (d.stride_cm : ℤ) * (ocm d p i.1.val : ℤ) +
        (d.stride_cn : ℤ) * (ocn d p i.2.1.val : ℤ) := by
    unfold cOff
    push_cast
    ring
  rw [hc]
  ring_nf

/-- The `c` mask's lanes. -/
theorem cMask_get (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (i : Idx [128, 256]) :
    cEn d p i = true ↔ ocm d p i.1 < d.M ∧ ocn d p i.2.1 < d.N := by
  exact cMask_get' d hd hp i

/-- The `a` load's mask is everywhere true. -/
theorem dotMaskA_get (d : Inst) (hd : d.ok) {t : Nat} (ht : t < steps d.K)
    (i : Idx [128, 64]) : (dotMaskA d t).get i = true := by
  exact dotMaskA_get' d hd ht i

/-- The `b` load's mask is everywhere true. -/
theorem dotMaskB_get (d : Inst) (hd : d.ok) {t : Nat} (ht : t < steps d.K)
    (i : Idx [64, 256]) : (dotMaskB d t).get i = true := by
  exact dotMaskB_get' d hd ht i

/-- The loop's step `64·stride_ak` fits. -/
theorem stepAK_toInt (d : Inst) (hd : d.ok) :
    ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_ak : I32)).toInt = 64 * (d.stride_ak : ℤ) := by
  exact stepAK_toInt' d hd

/-- The loop's step `64·stride_bk` fits. -/
theorem stepBK_toInt (d : Inst) (hd : d.ok) :
    ((IntV.ofNat 64 : I32) * (IntV.ofNat d.stride_bk : I32)).toInt = 64 * (d.stride_bk : ℤ) := by
  exact stepBK_toInt' d hd

/-! ## The footprints -/

/-- The elements the store touches are the program's block of `c`. -/
theorem cImg_sub {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) (init : Mem) :
    d.c.elems (Buf.img (fun i : Idx [128, 256] => cOff d p i.1 i.2.1) (cEn d p)) ⊆
      W d init p := by
  refine Buf.elems_mono d.c ?_
  rintro e ⟨i, hi, rfl⟩
  have h := (cMask_get d hd hp i).mp hi
  exact ⟨i.1, i.2.1, i.1.isLt, i.2.1.isLt, h.1, h.2, rfl⟩

/-- Distinct enabled lanes of the store are distinct elements. -/
theorem cImg_inj {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) :
    ∀ i j, cEn d p i = true → cEn d p j = true →
      cOff d p i.1 i.2.1 = cOff d p j.1 j.2.1 → i = j := by
  exact cImg_inj' hd hp

/-- Lane value of `offs_am` before the reduction. -/
theorem offsAM_raw_get (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (x : Fin 128) :
    (offsAM d p).get1 x = IntV.ofInt (ocm d p x.val) := offsCM_get1' d hd hp x

/-- Lane value of `offs_bn` before the reduction. -/
theorem offsBN_raw_get (d : Inst) (hd : d.ok) {p : Nat} (hp : p < grid d) (y : Fin 256) :
    (offsBN d p).get1 y = IntV.ofInt (ocn d p y.val) := offsCN_get1' d hd hp y

end Claims.MatmulE5m2
