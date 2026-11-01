import BlockScaledMxfp8.Basic

/-!
# `block_scaled_matmul_mxfp8`: one program

Program `p` of the grid computes the tile at row of tiles `pm p = p % cdiv(M, 128)` and column
of tiles `pn p = p / cdiv(M, 128)` (lines 204–209): the elements `(pm·128 + i, pn·256 + j)`,
`i < 128`, `j < 256`, inside `c`.

* The tiles and the footprints: `tileIdx`, `R` (the inputs), `W p` (program `p`'s tile), `Q`
  (what program `p` leaves there); `W_disj`, `R_W`.
* `AccInv`: the accumulator after `t` steps of the loop of lines 219–238.
* `safe_program`: program `p`, run alone from the initial memory, is safe (`TVA.Safe`): it reads
  only the inputs, writes only its tile, each byte once, and leaves there what `OutSpec` says.
-/

namespace Kernels.BlockScaledMxfp8

open TVA

/-! ## The tiles -/

/-- `num_pid_m = cdiv(M, 128)` (line 205). -/
def nM (d : Inst) : Nat := (d.M + 127) / 128

/-- `cdiv(N, 256)`: the columns of tiles. -/
def nN (d : Inst) : Nat := (d.N + 255) / 256

/-- The row of tiles of program `p` (`pid_m`, line 206). -/
def pm (d : Inst) (p : Nat) : Nat := p % nM d

/-- The column of tiles of program `p` (`pid_n`, line 207). -/
def pn (d : Inst) (p : Nat) : Nat := p / nM d

/-- The elements of `c` that program `p`'s tile covers: rows `pm·128 + i` and columns
`pn·256 + j < N` (`i < 128`, `j < 256`). -/
def tileIdx (d : Inst) (p : Nat) : Set Nat :=
  {e | ∃ i j, i < 128 ∧ j < 256 ∧ pn d p * 256 + j < d.N ∧ e = (pm d p * 128 + i) * d.N + (pn d p * 256 + j)}

/-- Every program's read footprint: the inputs. -/
def R (d : Inst) : Set Int := inEl d

/-- Program `p`'s write footprint: its tile of `c`. -/
def W (d : Inst) (p : Nat) : Set Int := d.c.elems (tileIdx d p)

/-- What program `p` leaves in memory `m`: every element of its tile as `OutSpec` says. -/
def Q (d : Inst) (init : Mem) (p : Nat) (m : Mem) : Prop :=
  ∀ i j, i < 128 → j < 256 → pn d p * 256 + j < d.N → OutSpec d init m (pm d p * 128 + i) (pn d p * 256 + j)

theorem nM_eq {d : Inst} (hd : d.ok) : nM d = d.M / 128 := by
  have := hd.M_ok; unfold nM; omega

theorem nM_pos {d : Inst} (hd : d.ok) : 0 < nM d := by
  have := hd.M_ok; unfold nM; omega

theorem pm_lt {d : Inst} (hd : d.ok) (p : Nat) : pm d p < nM d := Nat.mod_lt _ (nM_pos hd)

theorem pn_lt {d : Inst} {p : Nat} (hp : p < nM d * nN d) : pn d p < nN d := by
  unfold pn; exact Nat.div_lt_of_lt_mul hp

/-- The rows of program `p`'s tile are inside `a` and `c`. -/
theorem row_lt {d : Inst} (hd : d.ok) (p : Nat) {i : Nat} (hi : i < 128) : pm d p * 128 + i < d.M := by
  have h1 := pm_lt hd p
  rw [nM_eq hd] at h1
  have := hd.M_ok
  omega

/-- The first column of program `p`'s tile is inside `c`. -/
theorem col0_lt {d : Inst} (hd : d.ok) {p : Nat} (hp : p < nM d * nN d) : pn d p * 256 < d.N := by
  have h1 := pn_lt hp
  have := hd.N_ok
  unfold nN at h1
  omega

/-- Elements of a row-major matrix with `N` columns determine their row and column. -/
theorem c_inj {N r1 r2 c1 c2 : Nat} (hc1 : c1 < N) (hc2 : c2 < N) (h : r1 * N + c1 = r2 * N + c2) :
    r1 = r2 ∧ c1 = c2 := by
  have hr : r1 = r2 := by
    rcases lt_trichotomy r1 r2 with hlt | heq | hgt
    · exfalso
      have h1 : (r1 + 1) * N ≤ r2 * N := Nat.mul_le_mul_right _ hlt
      nlinarith
    · exact heq
    · exfalso
      have h1 : (r2 + 1) * N ≤ r1 * N := Nat.mul_le_mul_right _ hgt
      nlinarith
  subst hr
  exact ⟨rfl, by omega⟩

theorem tileIdx_sub {d : Inst} (hd : d.ok) (p : Nat) : tileIdx d p ⊆ {e | e < d.M * d.N} := by
  rintro _ ⟨i, j, hi, hj, hjN, rfl⟩
  have h1 := row_lt hd p hi
  show (pm d p * 128 + i) * d.N + (pn d p * 256 + j) < d.M * d.N
  have h2 : (pm d p * 128 + i + 1) * d.N ≤ d.M * d.N := Nat.mul_le_mul_right _ h1
  nlinarith

theorem W_sub {d : Inst} (hd : d.ok) (p : Nat) : W d p ⊆ cEl d := d.c.elems_mono (tileIdx_sub hd p)

/-- Two programs write different bytes. -/
theorem W_disj (d : Inst) {p q : Nat} (hpq : p ≠ q) : Disjoint (W d p) (W d q) := by
  refine d.c.elems_disjoint (Set.disjoint_left.mpr ?_)
  rintro _ ⟨i, j, hi, hj, hjN, rfl⟩ ⟨i', j', hi', hj', hjN', he⟩
  obtain ⟨hr, hc⟩ := c_inj hjN hjN' he
  have e1 : pm d p = pm d q := by omega
  have e2 : pn d p = pn d q := by omega
  apply hpq
  rw [← Nat.div_add_mod p (nM d), ← Nat.div_add_mod q (nM d)]
  exact congrArg₂ (· + ·) (congrArg (nM d * ·) e2) e1

theorem R_W {d : Inst} (hd : d.ok) (p : Nat) : Disjoint (R d) (W d p) :=
  (hd.c_in.mono_left (W_sub hd p)).symm

/-! ## The accumulator -/

/-- The accumulator of program `p` after `t` steps of the loop of line 219: each element of the
tile inside `c` whose row of `a` and row of `b` (with their scales) satisfy `OutSpec`'s hypotheses
is finite and within `tol` of the exact partial sum. -/
def AccInv (d : Inst) (init : Mem) (p t : Nat) (acc : Tensor [128, 256] F32) : Prop :=
  ∀ (i : Fin 128) (j : Fin 256), pn d p * 256 + j < d.N →
    ∀ (xa xb : Nat → F8E4) (sa sb : Nat → U8),
      (∀ k, k < d.K → init.read .f8e4nv (d.a.ptr ↑((pm d p * 128 + i) * d.K + k)).addr = some (xa k) ∧
        init.read .u8 (d.a_scale.ptr ↑(sIdx d.K (pm d p * 128 + i) k)).addr = some (sa k)) →
      (∀ k, k < d.K → init.read .f8e4nv (d.b.ptr ↑((pn d p * 256 + j) * d.K + k)).addr = some (xb k) ∧
        init.read .u8 (d.b_scale.ptr ↑(sIdx d.K (pn d p * 256 + j) k)).addr = some (sb k)) →
      (∀ k, k < d.K → InDom (xa k) (sa k) ∧ InDom (xb k) (sb k)) →
      dotAbs (prodV xa xb sa sb) d.K + tol (prodV xa xb sa sb) (d.K / 128) ≤ 2 ^ 127 →
      (acc.get2 i j).isFinite = true ∧
        |(acc.get2 i j).toRat - partSum (prodV xa xb sa sb) t| ≤ tol (prodV xa xb sa sb) t

theorem f32_zero : (FloatV.ofPat 0 : F32).isFinite = true ∧ (FloatV.ofPat 0 : F32).toRat = 0 :=
  DotLoop.val?_f32.mp DotLoop.zero_val

theorem acc0 (d : Inst) (init : Mem) (p : Nat) : AccInv d init p 0 (tl.zeros [128, 256] .f32) := by
  intro i j _ xa xb sa sb _ _ _ _
  refine ⟨f32_zero.1, ?_⟩
  show |(FloatV.ofPat 0 : F32).toRat - _| ≤ _
  rw [f32_zero.2]
  simp [partSum, tol]

/-! ## Where the scale lanes lie -/

/-- The scale of element `(pm·128 + i, 128 t + k)` of `a` (`i, k < 128`): entry `q` of the
`512` bytes at `(pm·(K/128) + t)·512`, `q = 16·(i % 32) + 4·(i / 32) + k / 32`. -/
theorem sIdx_a (K pm t i k : Nat) (hi : i < 128) (hk : k < 128) :
    sIdx K (pm * 128 + i) (128 * t + k) = (pm * (K / 128) + t) * 512 + (16 * (i % 32) + 4 * (i / 32) + k / 32) := by
  unfold sIdx
  have e1 : (pm * 128 + i) / 128 = pm := by omega
  have e2 : (128 * t + k) / 128 = t := by omega
  have e3 : (pm * 128 + i) % 32 = i % 32 := by omega
  have e4 : (pm * 128 + i) % 128 / 32 = i / 32 := by omega
  have e5 : (128 * t + k) % 128 / 32 = k / 32 := by omega
  rw [e1, e2, e3, e4, e5]
  ring

/-- The scale of element `(pn·256 + j, 128 t + k)` of `b` (`j < 256`, `k < 128`): entry `q` of
the `512` bytes at `((2 pn + j / 128)·(K/128) + t)·512`, `q = 16·(j % 32) + 4·((j % 128) / 32) + k / 32`. -/
theorem sIdx_b (K pn t j k : Nat) (hk : k < 128) :
    sIdx K (pn * 256 + j) (128 * t + k) =
      ((2 * pn + j / 128) * (K / 128) + t) * 512 + (16 * (j % 32) + 4 * (j % 128 / 32) + k / 32) := by
  unfold sIdx
  have e1 : (pn * 256 + j) / 128 = 2 * pn + j / 128 := by omega
  have e2 : (128 * t + k) / 128 = t := by omega
  have e3 : (pn * 256 + j) % 32 = j % 32 := by omega
  have e4 : (pn * 256 + j) % 128 / 32 = j % 128 / 32 := by omega
  have e5 : (128 * t + k) % 128 / 32 = k / 32 := by omega
  rw [e1, e2, e3, e4, e5]
  ring

/-! ## The launch -/

theorem launch_size (d : Inst) : (launch d).size = nM d * nN d := by
  rw [launch, Launch.size_grid1]; rfl

theorem launch_pid (d : Inst) {p : Nat} (hp : p < (launch d).size) : ((launch d).ctx p).pid 0 = p := by
  rw [launch_size] at hp
  simp only [Launch.ctx, launch, Launch.grid1, Matrix.cons_val_zero]
  exact Nat.mod_eq_of_lt hp

/-- A scale lane's element lies inside the scale tensor. -/
theorem scale_lt {Rb Kb q t l3 l4 : Nat} (hq : q < Rb) (ht : t < Kb) (hl3 : l3 < 2) (hl4 : l4 < 256) :
    q * (Kb * 512) + t * 512 + l3 * 256 + l4 < Rb * Kb * 512 := by
  have h1 : q * (Kb * 512) + (t * 512 + l3 * 256 + l4) < q * (Kb * 512) + Kb * 512 := by omega
  have h2 : (q + 1) * (Kb * 512) ≤ Rb * (Kb * 512) := Nat.mul_le_mul_right _ hq
  nlinarith

/-- An element of a row-major `R × C` matrix lies inside it. -/
theorem mat_lt {R C r c : Nat} (hr : r < R) (hc : c < C) : r * C + c < R * C := by
  have h2 : (r + 1) * C ≤ R * C := Nat.mul_le_mul_right _ hr
  nlinarith

/-- **The value of a scale of `a`** (lines 223, 226): entry `(i, k / 32)` of the scale block of
step `t` (loaded at `[0, pm, t, 0, 0]`) is the byte at `sIdx K (pm·128 + i) (128 t + k)` of
`a_scale`. -/
theorem scaleA_val {d : Inst} {m : Mem} {vA : Tensor [1, 1, 1, 2, 256] U8} {o1 o2 : I32} {pmv t : Nat}
    (h1 : o1.toInt = ((pmv * 1 : Nat) : Int)) (h2 : o2.toInt = (t : Int))
    (hvA : ∀ l : Idx [1, 1, 1, 2, 256],
      (⟨d.a_scale.ptr 0, [1, d.M / 128, d.K / 128, 2, 256],
          [d.M / 128 * (d.K / 128) * 512, d.K / 128 * 512, 512, 256, 1], [1, 1, 1, 2, 256], .zero⟩ :
          TensorDesc .u8).inbAt
        (TensorDesc.coord ([(IntV.ofInt 0 : I32), o1, o2, IntV.ofInt 0, IntV.ofInt 0].map IntV.toInt) l) = true →
      m.read .u8 (d.a_scale.ptr ((⟨d.a_scale.ptr 0, [1, d.M / 128, d.K / 128, 2, 256],
          [d.M / 128 * (d.K / 128) * 512, d.K / 128 * 512, 512, 256, 1], [1, 1, 1, 2, 256], .zero⟩ :
          TensorDesc .u8).elemAt
        (TensorDesc.coord ([(IntV.ofInt 0 : I32), o1, o2, IntV.ofInt 0, IntV.ofInt 0].map IntV.toInt) l)).toNat).addr =
        some (vA.get l))
    (hpm : pmv < d.M / 128) (ht : t < d.K / 128) (h1' h2') (i k : Fin 128) {s : U8}
    (hs : m.read .u8 (d.a_scale.ptr ↑(sIdx d.K (pmv * 128 + i) (128 * t + k))).addr = some s) :
    (((vA.reshape [1, 1, 32, 4, 4] h1').permute [0, 3, 2, 1, 4]).reshape [128, 4] h2').get2 i
      ⟨k / 32, by omega⟩ = s := by
  refine (scaleA_get vA h1' h2' i ⟨k / 32, by omega⟩).trans ?_
  obtain ⟨e1, e2⟩ := lane5 (d.a_scale.ptr 0) 1 (d.M / 128) (d.K / 128) 2 256 (d.M / 128 * (d.K / 128) * 512)
    (d.K / 128 * 512) 512 256 1 [1, 1, 1, 2, 256] .zero i32_zero' h1 h2 i32_zero' i32_zero'
    (0 : Fin 1) (0 : Fin 1) (0 : Fin 1)
    (⟨(16 * (i % 32) + 4 * (i / 32) + k / 32) / 256, by omega⟩ : Fin 2)
    (⟨(16 * (i % 32) + 4 * (i / 32) + k / 32) % 256, by omega⟩ : Fin 256)
  have hr1 := hvA _ (e1.trans (decide_eq_true (by simp only [Fin.val_zero, Fin.isValue]; omega)))
  rw [e2] at hr1
  rw [sIdx_a d.K pmv t i k i.2 k.2] at hs
  have hq := Nat.div_add_mod (16 * (i % 32) + 4 * (i / 32) + k / 32) 256
  have e3 : (0 + ((0 : Fin 1) : Nat)) * (d.M / 128 * (d.K / 128) * 512) + (pmv * 1 + ((0 : Fin 1) : Nat)) * (d.K / 128 * 512) +
      (t + ((0 : Fin 1) : Nat)) * 512 + (0 + (16 * (i % 32) + 4 * (i / 32) + k / 32) / 256) * 256 +
        (0 + (16 * (i % 32) + 4 * (i / 32) + k / 32) % 256) * 1 =
      (pmv * (d.K / 128) + t) * 512 + (16 * (i % 32) + 4 * (i / 32) + k / 32) := by
    simp only [Fin.val_zero]; ring_nf; omega
  rw [e3] at hr1
  exact Option.some.inj (hr1.symm.trans hs)

/-- **The value of a scale of `b`** (lines 224, 227): entry `(j, k / 32)` of the scale block of
step `t` (loaded at `[0, 2 pn, t, 0, 0]`), for a column `pn·256 + j` inside `c`, is the byte at
`sIdx K (pn·256 + j) (128 t + k)` of `b_scale`. -/
theorem scaleB_val {d : Inst} (hN : d.N % 128 = 0) {m : Mem} {vB : Tensor [1, 2, 1, 2, 256] U8} {o1 o2 : I32}
    {pnv t : Nat} (h1 : o1.toInt = ((pnv * 2 : Nat) : Int)) (h2 : o2.toInt = (t : Int))
    (hvB : ∀ l : Idx [1, 2, 1, 2, 256],
      (⟨d.b_scale.ptr 0, [1, d.N / 128, d.K / 128, 2, 256],
          [d.N / 128 * (d.K / 128) * 512, d.K / 128 * 512, 512, 256, 1], [1, 2, 1, 2, 256], .zero⟩ :
          TensorDesc .u8).inbAt
        (TensorDesc.coord ([(IntV.ofInt 0 : I32), o1, o2, IntV.ofInt 0, IntV.ofInt 0].map IntV.toInt) l) = true →
      m.read .u8 (d.b_scale.ptr ((⟨d.b_scale.ptr 0, [1, d.N / 128, d.K / 128, 2, 256],
          [d.N / 128 * (d.K / 128) * 512, d.K / 128 * 512, 512, 256, 1], [1, 2, 1, 2, 256], .zero⟩ :
          TensorDesc .u8).elemAt
        (TensorDesc.coord ([(IntV.ofInt 0 : I32), o1, o2, IntV.ofInt 0, IntV.ofInt 0].map IntV.toInt) l)).toNat).addr =
        some (vB.get l))
    (ht : t < d.K / 128) (h1' h2') (j : Fin 256) (hj : pnv * 256 + j < d.N) (k : Fin 128) {s : U8}
    (hs : m.read .u8 (d.b_scale.ptr ↑(sIdx d.K (pnv * 256 + j) (128 * t + k))).addr = some s) :
    (((vB.reshape [2, 1, 32, 4, 4] h1').permute [0, 3, 2, 1, 4]).reshape [256, 4] h2').get2 j
      ⟨k / 32, by omega⟩ = s := by
  refine (scaleB_get vB h1' h2' j ⟨k / 32, by omega⟩).trans ?_
  obtain ⟨e1, e2⟩ := lane5 (d.b_scale.ptr 0) 1 (d.N / 128) (d.K / 128) 2 256 (d.N / 128 * (d.K / 128) * 512)
    (d.K / 128 * 512) 512 256 1 [1, 2, 1, 2, 256] .zero i32_zero' h1 h2 i32_zero' i32_zero'
    (0 : Fin 1) (⟨j / 128, by omega⟩ : Fin 2) (0 : Fin 1)
    (⟨(16 * (j % 32) + 4 * (j % 128 / 32) + k / 32) / 256, by omega⟩ : Fin 2)
    (⟨(16 * (j % 32) + 4 * (j % 128 / 32) + k / 32) % 256, by omega⟩ : Fin 256)
  have hr1 := hvB _ (e1.trans (decide_eq_true (by simp only [Fin.val_zero, Fin.isValue]; omega)))
  rw [e2] at hr1
  rw [sIdx_b d.K pnv t j k k.2] at hs
  have hq := Nat.div_add_mod (16 * (j % 32) + 4 * (j % 128 / 32) + k / 32) 256
  have e3 : (0 + ((0 : Fin 1) : Nat)) * (d.N / 128 * (d.K / 128) * 512) + (pnv * 2 + j / 128) * (d.K / 128 * 512) +
      (t + ((0 : Fin 1) : Nat)) * 512 + (0 + (16 * (j % 32) + 4 * (j % 128 / 32) + k / 32) / 256) * 256 +
        (0 + (16 * (j % 32) + 4 * (j % 128 / 32) + k / 32) % 256) * 1 =
      ((2 * pnv + j / 128) * (d.K / 128) + t) * 512 + (16 * (j % 32) + 4 * (j % 128 / 32) + k / 32) := by
    simp only [Fin.val_zero]; ring_nf; omega
  rw [e3] at hr1
  exact Option.some.inj (hr1.symm.trans hs)

/-! ## One program -/

/-- **One program.** Program `p` of the grid, run alone from `init`, is safe: it reads only the
inputs, writes only its tile, and leaves there what `OutSpec` says. -/
theorem safe_program {P : Platform} (hdot : Rules.Num.DotScaledE4m3 P)
    (htot : Rules.Num.DotScaledTotal P cfgMx) {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    {p : Nat} (hp : p < nM d * nN d) :
    Safe ((launch d).env P) (R d) (W d p) (fun _ m => Q d init p m) init
      ((model d).body ((launch d).ctx p)) := by
  have hpid : tl.program_id ((launch d).ctx p) 0 = IntV.ofNat p := by
    simp [tl.program_id, launch_pid d (by rw [launch_size]; exact hp)]
  obtain ⟨hM0, hM128, hMlt⟩ := hd.M_ok
  obtain ⟨hN0, hN128, hNlt⟩ := hd.N_ok
  obtain ⟨hK0, hK128, hKle⟩ := hd.K_ok
  have hgrid := hd.grid_lt
  obtain ⟨hal_a, hal_b, hal_as, hal_bs, hal_c⟩ := hd.base_al
  obtain ⟨hva, hvb, hvas, hvbs, hvc⟩ := hpre
  have hpm := pm_lt hd p
  have hpn := pn_lt hp
  have hnM := nM_eq hd
  have hrow : ∀ i, i < 128 → pm d p * 128 + i < d.M := fun i hi => row_lt hd p hi
  have hcol := col0_lt hd hp
  have hp31 : p < 2 ^ 31 := by unfold nM nN at hp; omega
  dsimp only [model, block_scaled_matmul_kernel, aDesc, bDesc, aScaleDesc, bScaleDesc, cDesc]
  rw [hpid]
  have hnMv : (IntV.ofNat ((d.M + (128 - 1)) / 128) : I32).toInt = (nM d : Int) := i32_nat (by omega)
  -- line 206
  refine safe_irem1 (by rw [i32_nat hp31]; omega) (by rw [hnMv]; exact_mod_cast nM_pos hd) fun r1 hr1 => ?_
  have h1 : r1.item.toInt = (pm d p : Int) := by
    rw [hr1, i32_nat hp31, hnMv]; unfold pm; push_cast; rfl
  -- line 207
  refine safe_idiv1 (by rw [i32_nat hp31]; omega) (by rw [hnMv]; exact_mod_cast nM_pos hd) fun r2 hr2 => ?_
  have h2 : r2.item.toInt = (pn d p : Int) := by
    rw [hr2, i32_nat hp31, hnMv]; unfold pn; push_cast; rfl
  -- lines 208–213
  have hoam : (r1.item * (IntV.ofNat 128 : I32)).toInt = ((pm d p * 128 : Nat) : Int) :=
    IntV.I32.mul_nat h1 (i32_nat (by norm_num)) (by have := hrow 0 (by norm_num); omega)
  have hobn : (r2.item * (IntV.ofNat 256 : I32)).toInt = ((pn d p * 256 : Nat) : Int) :=
    IntV.I32.mul_nat h2 (i32_nat (by norm_num)) (by omega)
  have hosm : (r1.item * (IntV.ofNat 1 : I32)).toInt = ((pm d p * 1 : Nat) : Int) :=
    IntV.I32.mul_nat h1 (i32_nat (by norm_num)) (by have := hrow 0 (by norm_num); omega)
  have hosn : (r2.item * (IntV.ofNat 2 : I32)).toInt = ((pn d p * 2 : Nat) : Int) :=
    IntV.I32.mul_nat h2 (i32_nat (by norm_num)) (by omega)
  -- lines 219–220: the loop's bounds
  have hTv : (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32).toInt = ((d.K / 128 : Nat) : Int) := by
    rw [i32_nat (by omega)]; congr 1; omega
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat ((d.K + (128 - 1)) / 128)) (IntV.ofInt 1) := by
    unfold tl.RangeOk Math.Float.Format.intRange
    rw [i32_zero, i32_one, hTv]; norm_num; omega
  have htc : tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat ((d.K + (128 - 1)) / 128) : I32).toInt
      (IntV.ofInt 1 : I32).toInt = d.K / 128 := by
    rw [i32_zero, i32_one, hTv]; unfold tl.tripCount
    simp only [sub_zero, add_sub_cancel_right, Int.ediv_one, Int.toNat_natCast]
  have hKT : 128 * (d.K / 128) = d.K := by omega
  have hb1 : DType.bytes .f8e4nv = 1 := rfl
  have hbu : DType.bytes .u8 = 1 := rfl
  have hb2 : DType.bytes .f16 = 2 := rfl
  refine Safe.mono (safe_forRange (Wk := W d p) hrange
    (fun t st m => m = init ∧ st.2.1.toInt = ((128 * t : Nat) : Int) ∧
      st.2.2.1.toInt = ((128 * t : Nat) : Int) ∧ st.2.2.2.toInt = (t : Int) ∧ AccInv d init p t st.1)
    (fun _ => ∅) (fun _ _ _ _ => Set.disjoint_empty _) (by rw [iterW_empty]; exact Set.empty_disjoint _)
    ⟨rfl, by rw [i32_zero]; rfl, by rw [i32_zero]; rfl, by rw [i32_zero]; rfl, acc0 d init p⟩ ?_ ?_)
    (fun _ _ h => h) (by rw [iterW_empty, Set.empty_union])
  · -- one step of the loop (lines 221–238)
    rw [htc]
    rintro t ht ⟨acc, oka, okb, osk⟩ m₂ ⟨rfl, hoka, hokb, hosk, hacc⟩
    dsimp only at hoka hokb hosk hacc ⊢
    have h128t : 128 * t + 128 ≤ d.K := by omega
    -- line 221
    refine safe_desc_load d.a rfl
      (access2 ⟨hM0, hMlt⟩ ⟨hK0, by omega⟩ (by simp [Buf.ptr_addr]; omega) (by rw [hb1]; omega) rfl
        (by rw [hoka, hb1]; push_cast; omega)) ?_ hva ?_ fun A hA1 _ => ?_
    · rintro ⟨i, k, ⟨⟩⟩ hin
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hoka i k
      rw [e1, decide_eq_true_iff] at hin
      rw [e2]
      exact lt_of_lt_of_le (mat_lt hin.1 hin.2) hd.a_len
    · refine Set.Subset.trans (d.a.elems_mono ?_) fun x hx => Or.inl (Or.inl (Or.inl hx))
      rintro _ ⟨⟨i, k, ⟨⟩⟩, he, rfl⟩
      beta_reduce at he ⊢
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hoka i k
      rw [e1, decide_eq_true_iff] at he
      rw [Set.mem_ofPred_eq]
      rw [e2]
      exact mat_lt he.1 he.2
    -- line 222
    refine safe_desc_load d.b rfl
      (access2 ⟨hN0, hNlt⟩ ⟨hK0, by omega⟩ (by simp [Buf.ptr_addr]; omega) (by rw [hb1]; omega) rfl
        (by rw [hokb, hb1]; push_cast; omega)) ?_ hvb ?_ fun Bm hB1 _ => ?_
    · rintro ⟨j, k, ⟨⟩⟩ hin
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hobn hokb j k
      rw [e1, decide_eq_true_iff] at hin
      rw [e2]
      exact lt_of_lt_of_le (mat_lt hin.1 hin.2) hd.b_len
    · refine Set.Subset.trans (d.b.elems_mono ?_) fun x hx => Or.inl (Or.inl (Or.inr hx))
      rintro _ ⟨⟨j, k, ⟨⟩⟩, he, rfl⟩
      beta_reduce at he ⊢
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hobn hokb j k
      rw [e1, decide_eq_true_iff] at he
      rw [Set.mem_ofPred_eq]
      rw [e2]
      exact mat_lt he.1 he.2
    -- line 223
    have hosk' : osk.toInt = ((t : Nat) : Int) := hosk
    have hsa5 : ∀ s ∈ [1, d.M / 128, d.K / 128, 2, 256], 0 < s ∧ s < 2 ^ 31 := by
      intro s hs; simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
      rcases hs with rfl | rfl | rfl | rfl | rfl <;> omega
    have hsb5 : ∀ s ∈ [1, d.N / 128, d.K / 128, 2, 256], 0 < s ∧ s < 2 ^ 31 := by
      intro s hs; simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
      rcases hs with rfl | rfl | rfl | rfl | rfl <;> omega
    refine safe_desc_load d.a_scale rfl
      (access5 hsa5 (by simp [Buf.ptr_addr]; omega) (by omega) (by omega) rfl) ?_ hvas ?_ fun vA hvA1 _ => ?_
    · rintro ⟨l0, l1, l2, l3, l4, ⟨⟩⟩ hin
      obtain ⟨e1, e2⟩ := lane5 _ _ _ _ _ _ _ _ _ _ _ _ _ i32_zero' hosm hosk' i32_zero' i32_zero' l0 l1 l2 l3 l4
      rw [e1, decide_eq_true_iff] at hin
      rw [e2]
      have hl0 : (l0 : Nat) = 0 := by omega
      have hl3 := l3.2
      have hl4 := l4.2
      simp only [hl0, zero_add, zero_mul, mul_one] at hin ⊢
      exact lt_of_lt_of_le (scale_lt hin.2.1 hin.2.2.1 hl3 hl4) hd.as_len
    · refine Set.Subset.trans (d.a_scale.elems_mono ?_) fun x hx => Or.inl (Or.inr hx)
      rintro _ ⟨⟨l0, l1, l2, l3, l4, ⟨⟩⟩, he, rfl⟩
      beta_reduce at he ⊢
      obtain ⟨e1, e2⟩ := lane5 _ _ _ _ _ _ _ _ _ _ _ _ _ i32_zero' hosm hosk' i32_zero' i32_zero' l0 l1 l2 l3 l4
      rw [e1, decide_eq_true_iff] at he
      rw [Set.mem_ofPred_eq, e2]
      have hl0 : (l0 : Nat) = 0 := by omega
      have hl3 := l3.2
      have hl4 := l4.2
      simp only [hl0, zero_add, zero_mul, mul_one] at he ⊢
      exact scale_lt he.2.1 he.2.2.1 hl3 hl4
    -- line 224
    refine safe_desc_load d.b_scale rfl
      (access5 hsb5 (by simp [Buf.ptr_addr]; omega) (by omega) (by omega) rfl) ?_ hvbs ?_ fun vB hvB1 _ => ?_
    · rintro ⟨l0, l1, l2, l3, l4, ⟨⟩⟩ hin
      obtain ⟨e1, e2⟩ := lane5 _ _ _ _ _ _ _ _ _ _ _ _ _ i32_zero' hosn hosk' i32_zero' i32_zero' l0 l1 l2 l3 l4
      rw [e1, decide_eq_true_iff] at hin
      rw [e2]
      have hl0 : (l0 : Nat) = 0 := by omega
      have hl3 := l3.2
      have hl4 := l4.2
      simp only [hl0, zero_add, zero_mul, mul_one] at hin ⊢
      exact lt_of_lt_of_le (scale_lt hin.2.1 hin.2.2.1 hl3 hl4) hd.bs_len
    · refine Set.Subset.trans (d.b_scale.elems_mono ?_) fun x hx => Or.inr hx
      rintro _ ⟨⟨l0, l1, l2, l3, l4, ⟨⟩⟩, he, rfl⟩
      beta_reduce at he ⊢
      obtain ⟨e1, e2⟩ := lane5 _ _ _ _ _ _ _ _ _ _ _ _ _ i32_zero' hosn hosk' i32_zero' i32_zero' l0 l1 l2 l3 l4
      rw [e1, decide_eq_true_iff] at he
      rw [Set.mem_ofPred_eq, e2]
      have hl0 : (l0 : Nat) = 0 := by omega
      have hl3 := l3.2
      have hl4 := l4.2
      simp only [hl0, zero_add, zero_mul, mul_one] at he ⊢
      exact scale_lt he.2.1 he.2.2.1 hl3 hl4
    -- lines 229–233
    refine safe_dot_scaled (by unfold ScaledCfg.ShapeOk ScaledCfg.StorageOk cfgMx; decide)
      ⟨Or.inl rfl, Or.inl rfl, Or.inl rfl, Or.inl rfl⟩ (htot _ _ _ _ _) fun r hr => ?_
    -- lines 234–238
    refine safe_ret ⟨rfl, ?_, ?_, ?_, ?_⟩
    · rw [IntV.I32.add_nat hoka (i32_nat (by norm_num)) (by omega)]; push_cast; ring
    · rw [IntV.I32.add_nat hokb (i32_nat (by norm_num)) (by omega)]; push_cast; ring
    · rw [IntV.I32.add_nat hosk' (i32_nat (by norm_num)) (by omega)]
    · intro i j hj xa xb sa sb hrda hrdb hdom hroom
      have hpm' : pm d p < d.M / 128 := hnM ▸ hpm
      have hkl : ∀ k : Fin 128, 128 * t + k < d.K := fun k => by omega
      refine acc_step hdot hr i j (T := d.K / 128) ht (fun k => ?_)
        (fun k => scaleA_val hosm hosk' hvA1 hpm' ht _ _ i k (hrda _ (hkl k)).2) (fun k => ?_)
        (fun k => scaleB_val hN128 hosn hosk' hvB1 ht _ _ j hj k (hrdb _ (hkl k)).2)
        (by rw [hKT]; exact hdom) (by rw [hKT]; exact hroom) (hacc i j hj xa xb sa sb hrda hrdb hdom hroom)
      · obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hoka i k
        have h := hA1 (i, k, ()) (by rw [e1]; exact decide_eq_true ⟨hrow i i.2, hkl k⟩)
        rw [e2] at h
        exact Option.some.inj (h.symm.trans (hrda _ (hkl k)).1)
      · obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hobn hokb j k
        have h := hB1 (j, k, ()) (by rw [e1]; exact decide_eq_true ⟨hj, hkl k⟩)
        rw [e2] at h
        exact Option.some.inj (h.symm.trans (hrdb _ (hkl k)).1)
  · -- lines 236–240
    rw [htc]
    rintro ⟨acc, oka, okb, osk⟩ m₂ ⟨rfl, -, -, -, hacc⟩
    dsimp only at hacc ⊢
    refine safe_cvt ⟨by unfold Math.Float.Format.Std; simp, by unfold Math.Float.Format.Std; simp, Or.inl rfl⟩
      fun c hc => ?_
    refine safe_last (safe_desc_store d.c rfl
      (access2 ⟨hM0, hMlt⟩ ⟨hN0, hNlt⟩ (by simp [Buf.ptr_addr]; omega) (by rw [hb2]; omega) rfl
        (by rw [hobn, hb2]; push_cast; omega)) ?_ ?_ hvc (d.c.elems_mono ?_) fun m' hm'1 _ _ => safe_ret ?_)
    · rintro ⟨i, j, ⟨⟩⟩ hin
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hobn i j
      rw [e1, decide_eq_true_iff] at hin
      rw [e2]
      exact lt_of_lt_of_le (mat_lt hin.1 hin.2) hd.c_len
    · rintro ⟨i, j, ⟨⟩⟩ ⟨i', j', ⟨⟩⟩ hin hin' he
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hobn i j
      obtain ⟨e1', e2'⟩ := lane2 _ _ _ _ _ _ hoam hobn i' j'
      rw [e1, decide_eq_true_iff] at hin
      rw [e1', decide_eq_true_iff] at hin'
      rw [e2, e2'] at he
      obtain ⟨hr, hc'⟩ := c_inj hin.2 hin'.2 he
      have hi : i = i' := Fin.ext (by omega)
      have hj : j = j' := Fin.ext (by omega)
      subst hi hj; rfl
    · rintro _ ⟨⟨i, j, ⟨⟩⟩, he, rfl⟩
      beta_reduce at he ⊢
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hobn i j
      rw [e1, decide_eq_true_iff] at he
      rw [e2]
      exact ⟨i, j, i.2, j.2, he.2, rfl⟩
    · intro i j hi hj hjN
      obtain ⟨e1, e2⟩ := lane2 _ _ _ _ _ _ hoam hobn (⟨i, hi⟩ : Fin 128) (⟨j, hj⟩ : Fin 256)
      have hr := hm'1 (⟨i, hi⟩, ⟨j, hj⟩, ()) (by rw [e1]; exact decide_eq_true ⟨hrow i hi, hjN⟩)
      rw [e2] at hr
      refine ⟨c.get (⟨i, hi⟩, ⟨j, hj⟩, ()), hr, fun xa xb sa sb hra hrb hdom hroom => ?_⟩
      obtain ⟨hfin, herr⟩ := hacc ⟨i, hi⟩ ⟨j, hj⟩ hjN xa xb sa sb hra hrb hdom hroom
      rw [partSum_end, hKT] at herr
      have hcv : FloatV.NaNEq (FloatV.cvt (acc.get2 ⟨i, hi⟩ ⟨j, hj⟩)) (c.get (⟨i, hi⟩, ⟨j, hj⟩, ())) :=
        hc (⟨i, hi⟩, ⟨j, hj⟩, ())
      exact ⟨acc.get2 ⟨i, hi⟩ ⟨j, hj⟩, hfin, herr, hcv, fun hroom' => out_f16 hfin herr hcv hroom'⟩

end Kernels.BlockScaledMxfp8
