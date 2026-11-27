import Claim
import TVA.Proof

/-!
# The model's body, the launch and the grid's coordinates

`body_eq` writes the model's body out line by line. Program `i` of the grid has
`program_id(0) = p0 i = i % gx` and `program_id(1) = p1 i = i / gx`; `p0` splits into the feature
block `iD i = p0 % nD` and the grid row `rG i = p0 / nD`, and `p1` into the batch row
`bOf i = p1 / H` and the head `hOf i = p1 % H`.
-/

namespace Claims.MeanPoolingFwd

open TVA

/-- The model's body, unfolded. -/
theorem body_eq (d : Inst) (c : Ctx) :
    (model d).body c =
      (do
        let pid := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id c 0))).item
        let i_d := (← tl.irem (Tensor.scalar pid)
          (Tensor.scalar (IntV.ofNat ((d.D + d.BD - 1) / d.BD) : I64))).item
        let i_t := (← tl.idiv (Tensor.scalar pid)
          (Tensor.scalar (IntV.ofNat ((d.D + d.BD - 1) / d.BD) : I64))).item
        let i_bh := (← tl.convert .i32 .i64 (Tensor.scalar (tl.program_id c 1))).item
        let i_b := (← tl.idiv (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat d.H : I64))).item
        let i_h := (← tl.irem (Tensor.scalar i_bh) (Tensor.scalar (IntV.ofNat d.H : I64))).item
        let (i_tg, i_t, bos, T64) ← if d.varlen then do
            let i_tg := i_t
            let i_n := (← tl.convert d.idxD .i32 (Tensor.scalar
              (← tl.load1 (d.ci.ptr 0 + i_t * (IntV.ofNat 2 : I64))))).item
            let i_t := (← tl.convert d.idxD .i64 (Tensor.scalar
              (← tl.load1 (d.ci.ptr 0 + i_t * (IntV.ofNat 2 : I64) + (IntV.ofNat 1 : I32))))).item
            let bos := (← tl.convert d.idxD .i64 (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n)))).item
            let eos := (← tl.convert d.idxD .i64
              (Tensor.scalar (← tl.load1 (d.cu.ptr 0 + i_n + (IntV.ofNat 1 : I32))))).item
            let T := eos - bos
            let _NT ← tl.cdiv (Tensor.scalar T) (Tensor.scalar (IntV.ofNat d.BT : I64))
            pure (i_tg, i_t, bos, T)
          else do
            let NT := (← tl.cdiv (Tensor.scalar (IntV.ofNat d.T : I32))
              (Tensor.scalar (IntV.ofNat d.BT : I32))).item
            let NT64 := (← tl.convert .i32 .i64 (Tensor.scalar NT)).item
            let i_tg := i_b * NT64 + i_t
            let T64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofNat d.T : I32))).item
            pure (i_tg, i_t, i_b * T64, T64)
        let o_t := bc (i_t * (IntV.ofNat d.BT : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.BT))
        let o_d := bc (i_d * (IntV.ofNat d.BD : I64)) + (← tl.convert .i32 .i64 (tl.arange 0 d.BD))
        let m_x := ((o_t.col d.BD).ltS T64).and ((o_d.row d.BT).ltS (IntV.ofNat d.D : I64))
        let p_x := d.x.ptr 0 + (bos * (IntV.ofNat d.H : I64) + i_h) * (IntV.ofNat d.D : I64)
          + o_t.col d.BD * (IntV.ofNat (d.H * d.D) : I64) + o_d.row d.BT
        let p_o := d.o.ptr 0 + (i_tg * (IntV.ofNat d.H : I64) + i_h) * (IntV.ofNat d.D : I64) + o_d
        let b_x : Tensor [d.BT, d.BD] F32 ← tl.cvt (← tl.load p_x (some m_x)
          (some (← tl.convert .f32 .bf16 (tl.full [d.BT, d.BD] (FloatV.litF32 (FloatV.ofPat 0))))))
        let b_s ← tl.sum0 b_x
        let L := (tl.iminimum (Tensor.scalar (IntV.ofNat d.BT : I64))
          (Tensor.scalar (T64 - i_t * (IntV.ofNat d.BT : I64)))).item
        let L32 : Tensor [] F32 ← tl.itof (Tensor.scalar L)
        let b_o ← tl.div b_s (bc L32.item)
        let out : Tensor [d.BD] BF16 ← tl.cvt b_o
        tl.store p_o out (some (o_d.ltS (IntV.ofNat d.D : I64)))) := rfl

/-! ## The grid -/

/-- The feature blocks, `cdiv(D, BD)`. -/
def nD (d : Inst) : Nat := (d.D + d.BD - 1) / d.BD

/-- The grid rows, `NT`: `len(chunk_indices)` or `cdiv(T, BT)`. -/
def ntG (d : Inst) : Nat := if d.varlen then d.NT else ntFix d

/-- Axis 0 of the grid. -/
def gx (d : Inst) : Nat := nD d * ntG d

/-- Axis 1 of the grid. -/
def gy (d : Inst) : Nat := d.B * d.H

/-- `gridX` is `nD · ntG`. -/
theorem gridX_eq (d : Inst) : gridX d = gx d := rfl

/-- Program `i`'s coordinate on axis 0. -/
def p0 (d : Inst) (i : Nat) : Nat := i % gx d

/-- Program `i`'s coordinate on axis 1. -/
def p1 (d : Inst) (i : Nat) : Nat := i / gx d % gy d

/-- Program `i`'s feature block, `i_d`. -/
def iD (d : Inst) (i : Nat) : Nat := p0 d i % nD d

/-- Program `i`'s grid row, the first `i_t`. -/
def rG (d : Inst) (i : Nat) : Nat := p0 d i / nD d

/-- Program `i`'s batch row, `i_b`. -/
def bOf (d : Inst) (i : Nat) : Nat := p1 d i / d.H

/-- Program `i`'s head, `i_h`. -/
def hOf (d : Inst) (i : Nat) : Nat := p1 d i % d.H

/-- The launch has `gx · gy` programs. -/
theorem launch_size (d : Inst) : (launch d).size = gx d * gy d := by
  simp [Launch.size, launch, gx, gy, gridX_eq]

/-- Program `i` has `program_id(0) = p0 i`. -/
theorem launch_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = p0 d i := by
  simp [Launch.ctx, launch, p0, gridX_eq]

/-- Program `i` has `program_id(1) = p1 i`. -/
theorem launch_pid1 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 1 = p1 d i := by
  simp [Launch.ctx, launch, p1, gx, gy, gridX_eq]

/-- `BT` is positive. -/
theorem bt_pos {d : Inst} (hd : d.ok) : 0 < d.BT := by
  obtain ⟨e, he⟩ := hd.bt_pow2; rw [he]; exact Nat.two_pow_pos e

/-- `BD` is positive. -/
theorem bd_pos {d : Inst} (hd : d.ok) : 0 < d.BD := by have := hd.bd_ge; omega

/-- `BT ≤ 2^16`. -/
theorem bt_le {d : Inst} (hd : d.ok) : d.BT ≤ 2 ^ 16 := by
  have h1 := hd.block_le
  have h2 := hd.bd_ge
  by_contra h
  have : 2 ^ 16 * 16 < d.BT * d.BD := by
    calc 2 ^ 16 * 16 < d.BT * 16 := Nat.mul_lt_mul_of_pos_right (by omega) (by norm_num)
      _ ≤ d.BT * d.BD := Nat.mul_le_mul_left _ h2
  omega

/-- `BT = 2^e` with `e = log₂ BT ≤ 16`. -/
theorem bt_log {d : Inst} (hd : d.ok) : d.BT = 2 ^ Nat.log 2 d.BT ∧ Nat.log 2 d.BT ≤ 16 := by
  obtain ⟨e, he⟩ := hd.bt_pow2
  have hl : Nat.log 2 d.BT = e := by rw [he]; exact Nat.log_pow (by norm_num) e
  rw [hl]
  refine ⟨he, ?_⟩
  have h := bt_le hd
  rw [he] at h
  exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h

/-- The depth bound of a sum over `BT` leaves is `log₂ BT`. -/
theorem clog_bt {d : Inst} (hd : d.ok) : Nat.clog 2 d.BT = Nat.log 2 d.BT := by
  obtain ⟨h, -⟩ := bt_log hd
  rw [h, Nat.clog_pow 2 _ (by norm_num), Nat.log_pow (by norm_num)]

/-- `nD · BD` covers `D`, and not by a whole block more. -/
theorem nD_mul (d : Inst) (hd : d.ok) : d.D ≤ nD d * d.BD ∧ nD d * d.BD < d.D + d.BD := by
  have hb := bd_pos hd
  unfold nD
  constructor
  · have := Nat.div_add_mod (d.D + d.BD - 1) d.BD
    have := Nat.mod_lt (d.D + d.BD - 1) hb
    rw [Nat.mul_comm]; omega
  · have := Nat.div_mul_le_self (d.D + d.BD - 1) d.BD
    omega

/-- `ntFix · BT` covers `T`, and not by a whole chunk more. -/
theorem ntFix_mul (d : Inst) (hd : d.ok) : d.T ≤ ntFix d * d.BT ∧ ntFix d * d.BT < d.T + d.BT := by
  have hb := bt_pos hd
  unfold ntFix
  constructor
  · have := Nat.div_add_mod (d.T + d.BT - 1) d.BT
    have := Nat.mod_lt (d.T + d.BT - 1) hb
    rw [Nat.mul_comm]; omega
  · have := Nat.div_mul_le_self (d.T + d.BT - 1) d.BT
    omega

/-- `nD ≥ 1`. -/
theorem nD_pos {d : Inst} (hd : d.ok) : 0 < nD d :=
  Nat.pos_of_ne_zero fun h => by
    have h1 := (nD_mul d hd).1
    rw [h, Nat.zero_mul] at h1
    have := hd.D_pos
    omega

/-- `nD ≤ 2^13`. -/
theorem nD_le {d : Inst} (hd : d.ok) : nD d ≤ 2 ^ 13 := by
  have h1 := (nD_mul d hd).2
  have h2 := hd.D_le
  have h3 := hd.bd_ge
  by_contra h
  have h4 : (2 ^ 13 + 1) * d.BD ≤ nD d * d.BD := Nat.mul_le_mul_right _ (by omega)
  have h5 : 2 ^ 13 * 16 ≤ 2 ^ 13 * d.BD := Nat.mul_le_mul_left _ h3
  rw [Nat.add_mul] at h4
  omega

/-- The grid has fewer than `2^31` programs. -/
theorem size_lt {d : Inst} (hd : d.ok) : gx d * gy d < 2 ^ 31 := hd.grid_lt

/-- A program of the grid has its coordinates inside the grid. -/
theorem coords_lt {d : Inst} {i : Nat} (hi : i < gx d * gy d) :
    p0 d i < gx d ∧ i / gx d < gy d ∧ p1 d i = i / gx d := by
  have hgx : 0 < gx d := by
    rcases Nat.eq_zero_or_pos (gx d) with h | h
    · rw [h, Nat.zero_mul] at hi; omega
    · exact h
  have h2 : i / gx d < gy d := (Nat.div_lt_iff_lt_mul hgx).mpr (by rw [Nat.mul_comm]; exact hi)
  exact ⟨Nat.mod_lt _ hgx, h2, Nat.mod_eq_of_lt h2⟩

/-- A program's grid row is below `ntG`, its feature block below `nD`. -/
theorem rG_lt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) :
    rG d i < ntG d ∧ iD d i < nD d := by
  have h := (coords_lt hi).1
  have hn := nD_pos hd
  refine ⟨?_, Nat.mod_lt _ hn⟩
  unfold rG
  exact (Nat.div_lt_iff_lt_mul hn).mpr (by unfold gx at h; rw [Nat.mul_comm]; exact h)

/-- A program's batch row is below `B`, its head below `H`. -/
theorem bOf_lt {d : Inst} (hd : d.ok) {i : Nat} (hi : i < gx d * gy d) :
    bOf d i < d.B ∧ hOf d i < d.H := by
  have h := (coords_lt hi).2.1
  have h' := (coords_lt hi).2.2
  refine ⟨?_, Nat.mod_lt _ hd.H_pos⟩
  unfold bOf; rw [h']
  exact (Nat.div_lt_iff_lt_mul hd.H_pos).mpr (by unfold gy at h; exact h)

/-! ## The launch is valid -/

theorem tritonOk_nil : Shape.TritonOk [] :=
  ⟨by intro k hk; exact absurd hk List.not_mem_nil, by decide⟩

/-- The shape `[1]` is accepted by Triton. -/
theorem tritonOk_one : Shape.TritonOk [1] :=
  ⟨by
    intro k hk
    rw [List.mem_singleton] at hk
    subst hk
    exact ⟨0, rfl⟩, by decide⟩

/-- A power-of-two dimension in front of an accepted shape, within `2^20` elements, is accepted. -/
theorem tritonOk_cons {a : Nat} {s : Shape} (ha : ∃ e, a = 2 ^ e) (hs : Shape.TritonOk s)
    (h : (a :: s).prod ≤ 2 ^ 20) : Shape.TritonOk (a :: s) :=
  ⟨fun k hk => by
    rcases List.mem_cons.mp hk with rfl | hk
    · exact ha
    · exact hs.1 k hk,
   h⟩

/-- **The launch is valid** on every covered instance. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hBT : d.BT ≤ 2 ^ 20 := (bt_le hd).trans (by norm_num)
  have hBD : d.BD ≤ 2 ^ 20 := hd.bd_le.trans (by norm_num)
  have hgy : 0 < gy d := Nat.mul_pos hd.B_pos hd.H_pos
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, ?_⟩
  · show gridX d < 2 ^ 31
    have h := hd.grid_lt
    have : gridX d ≤ gridX d * (d.B * d.H) := Nat.le_mul_of_pos_right _ hgy
    omega
  · show d.B * d.H < 2 ^ 16
    exact hd.BH_lt
  · show (1 : Nat) < 2 ^ 16
    norm_num
  · rw [launch_size]
    exact hd.grid_lt
  · intro s hs
    simp only [launch, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl
    · exact tritonOk_cons (s := []) hd.bt_pow2 tritonOk_nil (by simpa using hBT)
    · exact tritonOk_cons (s := []) hd.bd_pow2 tritonOk_nil (by simpa using hBD)
    · exact tritonOk_cons (s := [1]) hd.bt_pow2 tritonOk_one (by simpa using hBT)
    · exact tritonOk_cons (s := [d.BD]) ⟨0, rfl⟩
        (tritonOk_cons (s := []) hd.bd_pow2 tritonOk_nil (by simpa using hBD))
        (by simpa using hBD)
    · exact tritonOk_cons (s := [d.BD]) hd.bt_pow2
        (tritonOk_cons (s := []) hd.bd_pow2 tritonOk_nil (by simpa using hBD))
        (by simpa using hd.block_le)

end Claims.MeanPoolingFwd
