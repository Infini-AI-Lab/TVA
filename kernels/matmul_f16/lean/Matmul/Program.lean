import Matmul.Basic

/-!
# `matmul_f16`: one program

`safe_program`: program `p` of the grid, run alone from the initial memory, is safe
(`TVA.Safe`): it reads only the matrices `a` and `b` (`R`), writes only its tile of `c` (`W`), each
byte once, and leaves in its tile what `OutSpec` says (`Q`). The proof walks the model line by
line with the library's rules.
-/

namespace Kernels.Matmul

open TVA

/-! ## The program's tile and footprints -/

/-- The tile rows, `cdiv(M, 128)`. -/
def tilesM (d : Inst) : Nat := (d.M + 127) / 128

/-- The tile columns, `cdiv(N, 256)`. -/
def tilesN (d : Inst) : Nat := (d.N + 255) / 256

/-- The number of programs. -/
def grid (d : Inst) : Nat := tilesM d * tilesN d

/-- Program `p`'s tile row, `pid_m`. -/
def pm (d : Inst) (p : Nat) : Nat := pmOf (tilesM d) (tilesN d) p

/-- Program `p`'s tile column, `pid_n`. -/
def pn (d : Inst) (p : Nat) : Nat := pnOf (tilesM d) (tilesN d) p

/-- The row of `a` that lane row `i` of program `p` reads (`offs_am`, line 282). -/
def rowA (d : Inst) (p i : Nat) : Nat := (pm d p * 128 + i) % d.M

/-- The column of `b` that lane column `j` of program `p` reads (`offs_bn`, line 283). -/
def colB (d : Inst) (p j : Nat) : Nat := (pn d p * 256 + j) % d.N

/-- `a_ptrs` at trip `t`. -/
def aPtrs (d : Inst) (p t : Nat) : Tensor [128, 64] (Ptr .f16) :=
  Tensor.of2 fun i k => d.a.ptr ↑(rowA d p i * d.stride_am + (64 * t + k) * d.stride_ak)

/-- `b_ptrs` at trip `t`. -/
def bPtrs (d : Inst) (p t : Nat) : Tensor [64, 256] (Ptr .f16) :=
  Tensor.of2 fun k j => d.b.ptr ↑((64 * t + k) * d.stride_bk + colB d p j * d.stride_bn)

/-- The element of `a` lane `(i, k)` loads at trip `t`, and whether it is enabled. -/
def gA (d : Inst) (p t : Nat) : Idx [128, 64] → Nat :=
  fun x => rowA d p x.1 * d.stride_am + (64 * t + x.2.1) * d.stride_ak

def enA (d : Inst) (t : Nat) : Idx [128, 64] → Bool := fun x => decide (64 * t + x.2.1 < d.K)

/-- The element of `b` lane `(k, j)` loads at trip `t`, and whether it is enabled. -/
def gB (d : Inst) (p t : Nat) : Idx [64, 256] → Nat :=
  fun x => (64 * t + x.1) * d.stride_bk + colB d p x.2.1 * d.stride_bn

def enB (d : Inst) (t : Nat) : Idx [64, 256] → Bool := fun x => decide (64 * t + x.1 < d.K)

/-- The element of `c` lane `(i, j)` stores, and whether it is enabled (`c_mask`, line 315). -/
def cG (d : Inst) (p : Nat) : Idx [128, 256] → Nat :=
  fun x => (pm d p * 128 + x.1) * d.stride_cm + (pn d p * 256 + x.2.1) * d.stride_cn

def cEn (d : Inst) (p : Nat) : Idx [128, 256] → Bool :=
  fun x => decide (pm d p * 128 + x.1 < d.M) && decide (pn d p * 256 + x.2.1 < d.N)

/-- Every program's read footprint: the matrices `a` and `b`. -/
def R (d : Inst) : Set Int := d.a.elems (aIdx d) ∪ d.b.elems (bIdx d)

/-- Program `p`'s write footprint: its tile of `c`. -/
def W (d : Inst) (p : Nat) : Set Int := d.c.elems (Buf.img (cG d p) (cEn d p))

/-- What program `p` leaves in memory `m`: its tile of `c` as `OutSpec` says. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ (i : Fin 128) (j : Fin 256), pm d p * 128 + i < d.M → pn d p * 256 + j < d.N →
    OutSpec d init m (pm d p * 128 + i) (pn d p * 256 + j)

/-- The accumulator after `t` trips: each element of the tile whose row of `a` and column of `b`
are finite is finite and within `DotLoop.bound` of the exact partial sum. -/
def AccInv (d : Inst) (init : Mem) (p t : Nat) (acc : Tensor [128, 256] F32) : Prop :=
  ∀ (i : Fin 128) (j : Fin 256), pm d p * 128 + i < d.M → pn d p * 256 + j < d.N →
    ∀ va vb : Nat → F16,
      (∀ k, k < d.K → init.read .f16 (d.a.ptr ↑((pm d p * 128 + i) * d.stride_am + k * d.stride_ak)).addr =
        some (va k) ∧ (va k).isFinite = true) →
      (∀ k, k < d.K → init.read .f16 (d.b.ptr ↑(k * d.stride_bk + (pn d p * 256 + j) * d.stride_bn)).addr =
        some (vb k) ∧ (vb k).isFinite = true) →
      (acc.get2 i j).isFinite = true ∧
        |(acc.get2 i j).toRat - partS va vb d.K t| ≤
          DotLoop.bound (Rules.Num.relDot16 64) (Rules.Num.absDot16 64) t (partA va vb d.K t)

/-! ## Facts about a program of the grid -/

theorem launch_size (d : Inst) : (launch d).size = grid d := by
  rw [launch, Launch.size_grid1]
  unfold grid tilesM tilesN
  congr 2

theorem launch_pid (d : Inst) {p : Nat} (hp : p < (launch d).size) : ((launch d).ctx p).pid 0 = p := by
  have h := hp
  rw [launch_size] at h
  unfold grid tilesM tilesN at h
  simp only [Launch.ctx, launch, Launch.grid1, Matrix.cons_val_zero]
  rw [show d.M + 128 - 1 = d.M + 127 by omega, show d.N + 256 - 1 = d.N + 255 by omega]
  exact Nat.mod_eq_of_lt h

theorem M_pos {d : Inst} {p : Nat} (hp : p < grid d) : 0 < d.M := by
  have : 0 < tilesM d := by
    rcases Nat.eq_zero_or_pos (tilesM d) with h | h
    · unfold grid at hp; rw [h] at hp; simp at hp
    · exact h
  unfold tilesM at this; omega

theorem N_pos {d : Inst} {p : Nat} (hp : p < grid d) : 0 < d.N := by
  have := n_pos_of hp
  unfold tilesN at this; omega

theorem pm_lt' {d : Inst} {p : Nat} (hp : p < grid d) : pm d p < tilesM d := pm_lt hp

theorem pn_lt' {d : Inst} {p : Nat} (hp : p < grid d) : pn d p < tilesN d := pn_lt hp

/-- The rows of program `p`'s tile stay below `2^31`. -/
theorem row_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) (i : Nat) (hi : i < 128) :
    pm d p * 128 + i < 2 ^ 31 := by
  have h1 := pm_lt' hp
  have h2 := hd.M_le
  unfold tilesM at h1
  omega

theorem col_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < grid d) (j : Nat) (hj : j < 256) :
    pn d p * 256 + j < 2 ^ 31 := by
  have h1 := pn_lt' hp
  have h2 := hd.N_le
  unfold tilesN at h1
  omega

/-- Rows of `c` below `M` and columns below `N` determine the element. -/
theorem c_inj {scm scn N r1 r2 c1 c2 : Nat} (hcn : 0 < scn) (hrows : N * scn ≤ scm) (hc1 : c1 < N)
    (hc2 : c2 < N) (h : r1 * scm + c1 * scn = r2 * scm + c2 * scn) : r1 = r2 ∧ c1 = c2 := by
  have hr : r1 = r2 := by
    rcases lt_trichotomy r1 r2 with hlt | heq | hgt
    · exfalso
      have h1 : (r1 + 1) * scm ≤ r2 * scm := Nat.mul_le_mul_right _ hlt
      have h2 : c1 * scn < N * scn := Nat.mul_lt_mul_of_pos_right hc1 hcn
      nlinarith
    · exact heq
    · exfalso
      have h1 : (r2 + 1) * scm ≤ r1 * scm := Nat.mul_le_mul_right _ hgt
      have h2 : c2 * scn < N * scn := Nat.mul_lt_mul_of_pos_right hc2 hcn
      nlinarith
  subst hr
  exact ⟨rfl, Nat.eq_of_mul_eq_mul_right hcn (by omega)⟩

/-! ## The proof of one program -/

/-- The grid has fewer than `2^31` programs: `cdiv(M, 128) · cdiv(N, 256) ≤ M · N ≤ M · stride_cm`. -/
theorem grid_lt {d : Inst} (hd : d.ok) : grid d < 2 ^ 31 := by
  have h1 : tilesM d ≤ d.M := by unfold tilesM; omega
  have h2 : tilesN d ≤ d.N := by unfold tilesN; omega
  have h3 : d.N ≤ d.stride_cm := le_trans (Nat.le_mul_of_pos_right _ hd.cn.1) hd.c_rows
  calc grid d = tilesM d * tilesN d := rfl
    _ ≤ d.M * d.N := Nat.mul_le_mul h1 h2
    _ ≤ d.M * d.stride_cm := Nat.mul_le_mul_left _ h3
    _ < 2 ^ 31 := hd.cm_fit

/-- Two-term offsets below a bound. -/
theorem lin2 {x X y Y a b B : Nat} (hx : x < X) (hy : y < Y) (ha : 0 < a) (hB : X * a + Y * b ≤ B) :
    x * a + y * b < B := by
  have h1 : x * a < X * a := Nat.mul_lt_mul_of_pos_right hx ha
  have h2 : y * b ≤ Y * b := Nat.mul_le_mul_right _ hy.le
  omega

/-- A product below a bound. -/
theorem mul_lt_of {x X a B : Nat} (hx : x < X) (ha : 0 < a) (hB : X * a ≤ B) : a * x < B := by
  have : x * a < X * a := Nat.mul_lt_mul_of_pos_right hx ha
  rw [Nat.mul_comm]; omega

/-- The accumulator before the loop (line 293): `+0.0`, exact. -/
theorem acc0 (d : Inst) (init : Mem) (p : Nat) : AccInv d init p 0 (tl.zeros [128, 256] .f32) := by
  intro i j _ _ va vb _ _
  have h := DotLoop.val?_f32.mp DotLoop.zero_val
  refine ⟨h.1, ?_⟩
  show |(FloatV.ofPat 0 : F32).toRat - partS va vb d.K 0| ≤ _
  rw [h.2]
  simp [partS, DotLoop.bound]

/-- **One program.** Program `p` of the grid, run alone from `init`, is safe: it reads only `a`
and `b`, writes only its tile of `c`, and leaves there what `OutSpec` says. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {P : Platform}
    (hdot : Rules.Num.DotF16 P)
    (htot : Rules.Num.DotTotal P { a := .f16, b := .f16, acc := .f32, precision := .tf32, maxImpreciseAcc := 0 })
    {p : Nat} (hp : p < grid d) :
    Safe ((launch d).env P) (R d) (W d p) (fun _ m => Q d init p m) init
      ((model d).body ((launch d).ctx p)) := by
  have hpid : ((launch d).ctx p).pid 0 = p := launch_pid d (by rw [launch_size]; exact hp)
  have hM0 := M_pos hp
  have hN0 := N_pos hp
  have hMle := hd.M_le
  have hNle := hd.N_le
  have hn0 : 0 < tilesN d := n_pos_of hp
  have hgrid : grid d < 2 ^ 31 := grid_lt hd
  have hnt : tilesN d < 2 ^ 24 := by unfold tilesN; omega
  have hmt : tilesM d < 2 ^ 24 := by unfold tilesM; omega
  simp only [model, matmul_kernel]
  rw [show tl.program_id ((launch d).ctx p) 0 = IntV.ofNat p by simp [tl.program_id, hpid]]
  -- line 253
  have hx1 : (IntV.ofNat d.M + IntV.ofNat (128 - 1) : I32).toInt = d.M + 127 := by
    rw [i32_add (i32_nat (by omega)) (i32_nat (by norm_num))] <;> push_cast <;> omega
  refine safe_idiv1 (by rw [hx1]; omega) (by rw [i32_nat (by norm_num)]; norm_num) fun r1 hr1 => ?_
  have h1 : r1.item.toInt = tilesM d := by rw [hr1, hx1, i32_nat (by norm_num)]; unfold tilesM; push_cast; omega
  -- line 254
  have hx2 : (IntV.ofNat d.N + IntV.ofNat (256 - 1) : I32).toInt = d.N + 255 := by
    rw [i32_add (i32_nat (by omega)) (i32_nat (by norm_num))] <;> push_cast <;> omega
  refine safe_idiv1 (by rw [hx2]; omega) (by rw [i32_nat (by norm_num)]; norm_num) fun r2 hr2 => ?_
  have h2 : r2.item.toInt = tilesN d := by rw [hr2, hx2, i32_nat (by norm_num)]; unfold tilesN; push_cast; omega
  -- lines 255–256
  have hnig : (IntV.ofNat 8 * r2.item : I32).toInt = ((8 * tilesN d : Nat) : Int) := by
    rw [i32_mul (i32_nat (by norm_num)) h2] <;> push_cast <;> omega
  have hpv : (IntV.ofNat p : I32).toInt = p := i32_nat (by omega)
  refine safe_idiv1 (by rw [hpv]; omega) (by rw [hnig]; omega) fun r3 hr3 => ?_
  have h3 : r3.item.toInt = ((p / (8 * tilesN d) : Nat) : Int) := by rw [hr3, hpv, hnig, Int.natCast_div]
  -- line 259 (inner `%`)
  refine safe_irem1 (by rw [hpv]; omega) (by rw [hnig]; omega) fun r4 hr4 => ?_
  have h4 : r4.item.toInt = ((p % (8 * tilesN d) : Nat) : Int) := by rw [hr4, hpv, hnig, Int.natCast_mod]
  -- line 258
  have hf := fpm_lt hp
  have hgs0 := gs_pos hp
  have hgs8 := gs_le (m := tilesM d) (n := tilesN d) (p := p)
  have hgsd : gsOf (tilesM d) (tilesN d) p = min (tilesM d - p / (8 * tilesN d) * 8) 8 := rfl
  have hpmd : pm d p = p / (8 * tilesN d) * 8 + p % (8 * tilesN d) % gsOf (tilesM d) (tilesN d) p := rfl
  have hpnd : pn d p = p % (8 * tilesN d) / gsOf (tilesM d) (tilesN d) p := rfl
  have hpmlt := pm_lt' hp
  unfold fpmOf at hf
  generalize p / (8 * tilesN d) = g at h3 hf hgsd hpmd
  generalize hrr : p % (8 * tilesN d) = rr at h4 hpmd hpnd
  generalize gsOf (tilesM d) (tilesN d) p = gs at hgs0 hgs8 hgsd hpmd hpnd
  have hm8 : (r3.item * IntV.ofNat 8 : I32).toInt = (g : Int) * 8 := by
    rw [i32_mul h3 (i32_nat (by norm_num))] <;> push_cast <;> omega
  have hgs : (tl.iminimum (Tensor.scalar (r1.item - r3.item * IntV.ofNat 8))
      (Tensor.scalar (IntV.ofNat 8 : I32))).item.toInt = gs := by
    show (IntV.min (r1.item - r3.item * IntV.ofNat 8) (IntV.ofNat 8)).toInt = _
    have hsub : (r1.item - r3.item * IntV.ofNat 8 : I32).toInt = (tilesM d : Int) - g * 8 := by
      rw [i32_sub h1 hm8] <;> omega
    rw [i32_min hsub (i32_nat (by norm_num)), hgsd]
    omega
  refine safe_irem1 (by rw [h4]; omega) (by rw [hgs]; omega) fun r5 hr5 => ?_
  have h5 : r5.item.toInt = ((rr % gs : Nat) : Int) := by rw [hr5, h4, hgs, Int.natCast_mod]
  -- line 260
  refine safe_irem1 (by rw [hpv]; omega) (by rw [hnig]; omega) fun r6 hr6 => ?_
  have h6 : r6.item.toInt = (rr : Int) := by rw [hr6, hpv, hnig, ← Int.natCast_mod, hrr]
  refine safe_idiv1 (by rw [h6]; omega) (by rw [hgs]; omega) fun r7 hr7 => ?_
  have hpn : r7.item.toInt = pn d p := by rw [hr7, h6, hgs, hpnd, Int.natCast_div]
  have hpm : (r3.item * IntV.ofNat 8 + r5.item : I32).toInt = pm d p := by
    rw [hpmd] at hpmlt
    rw [i32_add hm8 h5]
    · rw [hpmd]; push_cast; ring
    · push_cast; omega
    · push_cast; omega
  generalize r3.item * IntV.ofNat 8 + r5.item = pid_m at hpm ⊢
  generalize r7.item = pid_n at hpn ⊢
  clear hr1 hr2 hr3 hr4 hr5 hr6 hr7 h1 h2 h3 h4 h5 h6 hgs hm8 hnig hx1 hx2
  -- lines 266–273
  have hstr : ∀ s, 0 < s ∧ s < 2 ^ 31 → (IntV.ofInt 0 : I32).lt (IntV.ofNat s) = true := fun s hs => by
    rw [i32_lt i32_zero (i32_nat hs.2)]; simp; omega
  refine safe_assume (by rw [i32_le i32_zero hpm]; simp) ?_
  refine safe_assume (by rw [i32_le i32_zero hpn]; simp) ?_
  refine safe_assume (hstr _ hd.am) ?_
  refine safe_assume (hstr _ hd.ak) ?_
  refine safe_assume (hstr _ hd.bn) ?_
  refine safe_assume (hstr _ hd.bk) ?_
  refine safe_assume (hstr _ hd.cm) ?_
  refine safe_assume (hstr _ hd.cn) ?_
  -- line 282
  have hpm128 : (pid_m * (IntV.ofNat 128 : I32)).toInt = ((pm d p * 128 : Nat) : Int) := by
    have := row_lt hd hp 0 (by norm_num)
    rw [i32_mul hpm (i32_nat (by norm_num))]
    · push_cast; ring
    · omega
    · omega
  have hpn256 : (pid_n * (IntV.ofNat 256 : I32)).toInt = ((pn d p * 256 : Nat) : Int) := by
    have := col_lt hd hp 0 (by norm_num)
    rw [i32_mul hpn (i32_nat (by norm_num))]
    · push_cast; ring
    · omega
    · omega
  have har : ∀ (n : Nat) (i : Nat), i < 2 ^ 31 →
      (IntV.ofInt (((0 : Nat) : Int) + (i : Int)) : I32).toInt = i := by
    intro n i hi
    rw [IntV.I32.toInt_ofInt] <;> push_cast
    · simp
    · constructor <;> omega
  have hlaneM : ∀ i : Fin 128, (pid_m * (IntV.ofNat 128 : I32) + IntV.ofInt (((0 : Nat) : Int) + (i : Nat))).toInt
      = ((pm d p * 128 + i : Nat) : Int) := by
    intro i
    have := row_lt hd hp i i.2
    rw [i32_add hpm128 (har 0 i (by omega))] <;> push_cast <;> omega
  have hlaneN : ∀ j : Fin 256, (pid_n * (IntV.ofNat 256 : I32) + IntV.ofInt (((0 : Nat) : Int) + (j : Nat))).toInt
      = ((pn d p * 256 + j : Nat) : Int) := by
    intro j
    have := col_lt hd hp j j.2
    rw [i32_add hpn256 (har 0 j (by omega))] <;> push_cast <;> omega
  have hMv : (IntV.ofNat d.M : I32).toInt = d.M := i32_nat (by omega)
  have hNv : (IntV.ofNat d.N : I32).toInt = d.N := i32_nat (by omega)
  have hsam := i32_nat hd.am.2
  have hsak := i32_nat hd.ak.2
  have hsbk := i32_nat hd.bk.2
  have hsbn := i32_nat hd.bn.2
  have hscm := i32_nat hd.cm.2
  have hscn := i32_nat hd.cn.2
  have hamp := hd.am.1
  have hakp := hd.ak.1
  have hbkp := hd.bk.1
  have hbnp := hd.bn.1
  have hcmp := hd.cm.1
  have hcnp := hd.cn.1
  have hafit := hd.a_fit
  have hbfit := hd.b_fit
  have hsak64 : 64 * d.stride_ak < 2 ^ 31 := by
    have : d.stride_am ≤ d.M * d.stride_am := Nat.le_mul_of_pos_left _ hM0
    omega
  have hsbk64 : 64 * d.stride_bk < 2 ^ 31 := by
    have : d.stride_bn ≤ d.N * d.stride_bn := Nat.le_mul_of_pos_left _ hN0
    omega
  refine safe_irem (fun i => ?_) fun offs_am hoam => ?_
  · obtain ⟨i, ⟨⟩⟩ := i
    obtain ⟨q, hq, -⟩ := IntV.rem?_of_nonneg
      (a := pid_m * IntV.ofNat 128 + IntV.ofInt (((0 : Nat) : Int) + (i : Nat))) (b := (IntV.ofNat d.M : I32))
      (by rw [hlaneM]; omega) (by rw [hMv]; omega)
    exact ⟨q, hq⟩
  have hoffA : ∀ i : Fin 128, (offs_am.get (i, ())).toInt = ((rowA d p i : Nat) : Int) := by
    intro i
    have h : (pid_m * IntV.ofNat 128 + IntV.ofInt (((0 : Nat) : Int) + (i : Nat)) : I32).rem? (IntV.ofNat d.M) =
        some (offs_am.get (i, ())) := hoam (i, ())
    rw [i32_rem (by rw [hlaneM]; omega) (by rw [hMv]; omega) h, hlaneM, hMv]
    unfold rowA; push_cast; rfl
  clear hoam
  -- line 283
  refine safe_irem (fun i => ?_) fun offs_bn hobn => ?_
  · obtain ⟨j, ⟨⟩⟩ := i
    obtain ⟨q, hq, -⟩ := IntV.rem?_of_nonneg
      (a := pid_n * IntV.ofNat 256 + IntV.ofInt (((0 : Nat) : Int) + (j : Nat))) (b := (IntV.ofNat d.N : I32))
      (by rw [hlaneN]; omega) (by rw [hNv]; omega)
    exact ⟨q, hq⟩
  have hoffB : ∀ j : Fin 256, (offs_bn.get (j, ())).toInt = ((colB d p j : Nat) : Int) := by
    intro j
    have h : (pid_n * IntV.ofNat 256 + IntV.ofInt (((0 : Nat) : Int) + (j : Nat)) : I32).rem? (IntV.ofNat d.N) =
        some (offs_bn.get (j, ())) := hobn (j, ())
    rw [i32_rem (by rw [hlaneN]; omega) (by rw [hNv]; omega) h, hlaneN, hNv]
    unfold colB; push_cast; rfl
  clear hobn
  -- lines 285–286
  have haP : (d.a.ptr 0 + (offs_am.col 64 * (IntV.ofNat d.stride_am : I32) +
      (tl.arange 0 64).row 128 * (IntV.ofNat d.stride_ak : I32)) : Tensor [128, 64] (Ptr .f16)) = aPtrs d p 0 := by
    refine Tensor.ext (funext fun ⟨i, k, ⟨⟩⟩ => ?_)
    have hr : rowA d p i < d.M := Nat.mod_lt _ hM0
    show (d.a.ptr 0).offset ((offs_am.get (i, ()) * IntV.ofNat d.stride_am +
      IntV.ofInt (((0 : Nat) : Int) + (k : Nat)) * IntV.ofNat d.stride_ak).toInt) =
      d.a.ptr ↑(rowA d p i * d.stride_am + (64 * 0 + k) * d.stride_ak)
    have hb := lin2 hr k.2 hamp hafit
    rw [IntV.I32.add_nat (IntV.I32.mul_nat (hoffA i) hsam (by omega))
      (IntV.I32.mul_nat (har 0 k (by omega)) hsak (by omega)) hb, Buf.ptr_offset]
    congr 1; push_cast; ring
  have hbP : (d.b.ptr 0 + ((tl.arange 0 64).col 256 * (IntV.ofNat d.stride_bk : I32) +
      offs_bn.row 64 * (IntV.ofNat d.stride_bn : I32)) : Tensor [64, 256] (Ptr .f16)) = bPtrs d p 0 := by
    refine Tensor.ext (funext fun ⟨k, j, ⟨⟩⟩ => ?_)
    have hc : colB d p j < d.N := Nat.mod_lt _ hN0
    show (d.b.ptr 0).offset ((IntV.ofInt (((0 : Nat) : Int) + (k : Nat)) * IntV.ofNat d.stride_bk +
      offs_bn.get (j, ()) * IntV.ofNat d.stride_bn).toInt) =
      d.b.ptr ↑((64 * 0 + k) * d.stride_bk + colB d p j * d.stride_bn)
    have hb := lin2 k.2 hc hbkp hbfit
    rw [IntV.I32.add_nat (IntV.I32.mul_nat (har 0 k (by omega)) hsbk (by omega))
      (IntV.I32.mul_nat (hoffB j) hsbn (by omega)) hb, Buf.ptr_offset]
    congr 1; push_cast; ring
  -- line 294: the loop bound
  have hK : d.K ≤ 2 ^ 16 := by have := hd.K_le; omega
  have hKv : (IntV.ofNat d.K : I32).toInt = d.K := i32_nat (by omega)
  have hK63 : (IntV.ofNat d.K + IntV.ofNat (64 - 1) : I32).toInt = ((d.K + 63 : Nat) : Int) :=
    IntV.I32.add_nat hKv (i32_nat (by norm_num)) (by omega)
  refine safe_idiv1 (by rw [hK63]; omega) (by rw [i32_nat (by norm_num)]; norm_num) fun rK hrK => ?_
  have hT : rK.item.toInt = steps d.K := by
    rw [hrK, hK63, i32_nat (by norm_num)]; unfold steps; push_cast; omega
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) rK.item (IntV.ofInt 1) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [i32_zero, i32_one, hT]; unfold steps; norm_num; omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt rK.item.toInt (IntV.ofInt 1 : I32).toInt = steps d.K := by
    rw [i32_zero, i32_one, hT]; simp [tl.tripCount]
  refine Safe.mono (safe_forRange (Wk := W d p) hrange
    (fun t st m => m = init ∧ st.1 = aPtrs d p t ∧ st.2.1 = bPtrs d p t ∧ AccInv d init p t st.2.2)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, haP, hbP, acc0 d init p⟩ ?_ ?_) (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])
  · rw [htc]
    rintro t ht ⟨sa, sb, sacc⟩ m₁ ⟨rfl, hsa, hsb, hinv⟩
    simp only at hsa hsb hinv
    subst hsa hsb
    have ht64 : 64 * t < d.K := by unfold steps at ht; omega
    have hkt : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + ↑t * (IntV.ofInt 1 : I32).toInt) : I32).toInt = (t : Int) := by
      rw [i32_zero, i32_one, zero_add, mul_one]
      exact IntV.I32.toInt_ofInt (by constructor <;> omega)
    have hrem : (IntV.ofNat d.K - IntV.ofInt ((IntV.ofInt 0 : I32).toInt + ↑t * (IntV.ofInt 1 : I32).toInt) *
        IntV.ofNat 64 : I32).toInt = ((d.K - t * 64 : Nat) : Int) :=
      IntV.I32.sub_nat hKv (IntV.I32.mul_nat hkt (i32_nat (by norm_num)) (by omega)) (by omega) (by omega)
    generalize (IntV.ofNat d.K - IntV.ofInt ((IntV.ofInt 0 : I32).toInt + ↑t * (IntV.ofInt 1 : I32).toInt) *
        IntV.ofNat 64 : I32) = kr at hrem ⊢
    -- line 297
    refine safe_convert (fun _ => zero_conv_ex P) fun za hza => ?_
    refine safe_load_buf d.a (g := gA d p t) (en := enA d t) ?_ ?_ hpre.1 ?_ fun A hA1 hA2 => ?_
    · refine Buf.addrs_of_get d.a (fun ⟨i, k, ⟨⟩⟩ => ?_) fun ⟨i, k, ⟨⟩⟩ _ => rfl
      show IntV.lt (IntV.ofInt (((0 : Nat) : Int) + (k : Nat))) kr = decide (64 * t + k < d.K)
      rw [i32_lt (har 0 k (by omega)) hrem]
      exact decide_eq_decide.mpr (by omega)
    · rintro ⟨i, k, ⟨⟩⟩ he
      simp only [enA, decide_eq_true_eq] at he
      exact hd.a_len _ _ (Nat.mod_lt _ hM0) he
    · refine Set.Subset.trans (d.a.elems_mono ?_) Set.subset_union_left
      rintro _ ⟨⟨i, k, ⟨⟩⟩, he, rfl⟩
      simp only [enA, decide_eq_true_eq] at he
      exact ⟨rowA d p i, 64 * t + k, Nat.mod_lt _ hM0, he, rfl⟩
    -- line 298
    refine safe_convert (fun _ => zero_conv_ex P) fun zb hzb => ?_
    refine safe_load_buf d.b (g := gB d p t) (en := enB d t) ?_ ?_ hpre.2.1 ?_ fun B hB1 hB2 => ?_
    · refine Buf.addrs_of_get d.b (fun ⟨k, j, ⟨⟩⟩ => ?_) fun ⟨k, j, ⟨⟩⟩ _ => rfl
      show IntV.lt (IntV.ofInt (((0 : Nat) : Int) + (k : Nat))) kr = decide (64 * t + k < d.K)
      rw [i32_lt (har 0 k (by omega)) hrem]
      exact decide_eq_decide.mpr (by omega)
    · rintro ⟨k, j, ⟨⟩⟩ he
      simp only [enB, decide_eq_true_eq] at he
      exact hd.b_len _ _ he (Nat.mod_lt _ hN0)
    · refine Set.Subset.trans (d.b.elems_mono ?_) Set.subset_union_right
      rintro _ ⟨⟨k, j, ⟨⟩⟩, he, rfl⟩
      simp only [enB, decide_eq_true_eq] at he
      exact ⟨64 * t + k, colB d p j, he, Nat.mod_lt _ hN0, rfl⟩
    -- line 300
    refine safe_dot (htot A B sacc) fun r hr => ?_
    -- lines 302–303
    refine safe_ret ⟨rfl, ?_, ?_, ?_⟩
    · refine Tensor.ext (funext fun ⟨i, k, ⟨⟩⟩ => ?_)
      show (d.a.ptr ↑(rowA d p i * d.stride_am + (64 * t + k) * d.stride_ak)).offset
        ((IntV.ofNat 64 * IntV.ofNat d.stride_ak : I32).toInt) =
        d.a.ptr ↑(rowA d p i * d.stride_am + (64 * (t + 1) + k) * d.stride_ak)
      rw [IntV.I32.mul_nat (i32_nat (by norm_num)) hsak hsak64, Buf.ptr_offset]
      congr 1; push_cast; ring
    · refine Tensor.ext (funext fun ⟨k, j, ⟨⟩⟩ => ?_)
      show (d.b.ptr ↑((64 * t + k) * d.stride_bk + colB d p j * d.stride_bn)).offset
        ((IntV.ofNat 64 * IntV.ofNat d.stride_bk : I32).toInt) =
        d.b.ptr ↑((64 * (t + 1) + k) * d.stride_bk + colB d p j * d.stride_bn)
      rw [IntV.I32.mul_nat (i32_nat (by norm_num)) hsbk hsbk64, Buf.ptr_offset]
      congr 1; push_cast; ring
    · intro i j hi hj va vb hva hvb
      have hrow : rowA d p i = pm d p * 128 + i := Nat.mod_eq_of_lt hi
      have hcol : colB d p j = pn d p * 256 + j := Nat.mod_eq_of_lt hj
      refine acc_step hdot hr i j hK ht64 (fun e he => (hva e he).2) (fun e he => (hvb e he).2)
        (fun k => ⟨fun h => ?_, fun h => ?_⟩) (fun k => ⟨fun h => ?_, fun h => ?_⟩)
        (hinv i j hi hj va vb hva hvb)
      · have h1 := hA1 (i, k, ()) (by simp only [enA, decide_eq_true_eq]; exact h)
        have h2 := (hva _ h).1
        simp only [gA, hrow] at h1
        exact Option.some.inj (h1.symm.trans h2)
      · have h1 := hA2 (i, k, ()) (by simp only [enA, decide_eq_false_iff_not]; omega) za rfl
        show (A.get (i, k, ())).isFinite = true ∧ (A.get (i, k, ())).toRat = 0
        rw [h1]
        exact zero_conv (hza _)
      · have h1 := hB1 (k, j, ()) (by simp only [enB, decide_eq_true_eq]; exact h)
        have h2 := (hvb _ h).1
        simp only [gB, hcol] at h1
        exact Option.some.inj (h1.symm.trans h2)
      · have h1 := hB2 (k, j, ()) (by simp only [enB, decide_eq_false_iff_not]; omega) zb rfl
        show (B.get (k, j, ())).isFinite = true ∧ (B.get (k, j, ())).toRat = 0
        rw [h1]
        exact zero_conv (hzb _)
  · rw [htc]
    rintro ⟨sa, sb, sacc⟩ m₁ ⟨rfl, -, -, hinv⟩
    simp only at hinv
    -- line 308
    refine safe_cvt ⟨by unfold Math.Float.Format.Std; simp, by unfold Math.Float.Format.Std; simp, Or.inl rfl⟩
      fun c hc => ?_
    -- lines 312–316
    have hcrow : d.N * d.stride_cn < 2 ^ 31 := by have := hd.c_rows; have := hd.cm.2; omega
    refine safe_last (safe_store_buf d.c (g := cG d p) (en := cEn d p) ?_ ?_ ?_ hpre.2.2 subset_rfl
      fun m' hm' _ _ => safe_ret ?_)
    · refine Buf.addrs_of_get d.c (fun ⟨i, j, ⟨⟩⟩ => ?_) fun ⟨i, j, ⟨⟩⟩ he => ?_
      · show (IntV.lt (pid_m * IntV.ofNat 128 + IntV.ofInt (((0 : Nat) : Int) + (i : Nat))) (IntV.ofNat d.M) &&
          IntV.lt (pid_n * IntV.ofNat 256 + IntV.ofInt (((0 : Nat) : Int) + (j : Nat))) (IntV.ofNat d.N)) =
          (decide (pm d p * 128 + i < d.M) && decide (pn d p * 256 + j < d.N))
        rw [i32_lt (hlaneM i) hMv, i32_lt (hlaneN j) hNv]
        simp only [Nat.cast_lt]
      · simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he
        show ((d.c.ptr 0).offset ((IntV.ofNat d.stride_cm *
          (pid_m * IntV.ofNat 128 + IntV.ofInt (((0 : Nat) : Int) + (i : Nat))) : I32).toInt)).offset
          ((IntV.ofNat d.stride_cn * (pid_n * IntV.ofNat 256 + IntV.ofInt (((0 : Nat) : Int) + (j : Nat))) : I32).toInt) =
          d.c.ptr ↑((pm d p * 128 + i) * d.stride_cm + (pn d p * 256 + j) * d.stride_cn)
        rw [IntV.I32.mul_nat hscm (hlaneM i) (mul_lt_of he.1 hcmp hd.cm_fit.le),
          IntV.I32.mul_nat hscn (hlaneN j) (mul_lt_of he.2 hcnp hcrow.le), Buf.ptr_offset, Buf.ptr_offset]
        congr 1; push_cast; ring
    · rintro ⟨i, j, ⟨⟩⟩ he
      simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he
      exact hd.c_len _ _ he.1 he.2
    · rintro ⟨i, j, ⟨⟩⟩ ⟨i', j', ⟨⟩⟩ he he' hg
      simp only [cEn, Bool.and_eq_true, decide_eq_true_eq] at he he'
      dsimp only [cG] at hg
      obtain ⟨h1, h2⟩ := c_inj hcnp hd.c_rows he.2 he'.2 hg
      have h1' : i = i' := Fin.ext (by omega)
      have h2' : j = j' := Fin.ext (by omega)
      subst h1' h2'; rfl
    · intro i j hi hj
      refine ⟨c.get (i, j, ()), hm' (i, j, ()) (by simp only [cEn, hi, hj, decide_true, Bool.and_self]),
        fun va vb hva hvb => ?_⟩
      obtain ⟨hfin, herr⟩ := hinv i j hi hj va vb hva hvb
      obtain ⟨hS, hA⟩ := partS_end va vb (T := steps d.K) (K := d.K) (by unfold steps; omega)
      rw [hS, hA] at herr
      have hcv : FloatV.NaNEq (FloatV.cvt (sacc.get2 i j)) (c.get (i, j, ())) := hc (i, j, ())
      exact ⟨sacc.get2 i j, hfin, herr, hcv, fun hroom => out_f16 hfin herr hcv hroom⟩
end Kernels.Matmul
