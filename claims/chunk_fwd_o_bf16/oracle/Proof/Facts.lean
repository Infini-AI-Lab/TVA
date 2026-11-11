import Claim
import TVA.Proof
import Proof.Defs

/-!
# Size facts of a covered instance

The bounds on `H`, `HV`, `K`, `V`, `B·T` that keep the kernel's `int32`/`int64` offsets from
wrapping, and `qh j < H`.
-/

namespace Claims.ChunkFwdO

open TVA

/-- For a valid instance, `HV < 2^16`. -/
theorem HV_lt16 {d : Inst} (hd : d.ok) : d.HV < 2 ^ 16 := by
  have h := hd.grid1_lt
  have hB := hd.B_pos
  have : d.HV ≤ d.B * d.HV := Nat.le_mul_of_pos_left _ hB
  omega

/-- For a valid instance, `H ≤ HV`. -/
theorem H_le_HV {d : Inst} (hd : d.ok) : d.H ≤ d.HV :=
  Nat.le_of_dvd hd.HV_pos hd.hv_dvd

/-- For a valid instance, `H < 2^16`. -/
theorem H_lt16 {d : Inst} (hd : d.ok) : d.H < 2 ^ 16 :=
  lt_of_le_of_lt (H_le_HV hd) (HV_lt16 hd)

/-- For a valid instance, `B * T ≤ 2^35`. -/
theorem BT_le {d : Inst} (hd : d.ok) : d.B * d.T ≤ 2 ^ 35 := by
  have h1 := hd.B_le
  have h2 := hd.T_le
  calc d.B * d.T ≤ 8192 * 2 ^ 22 := Nat.mul_le_mul h1 h2
    _ = 2 ^ 35 := by norm_num

/-- `qh j < H` for a value head `j < HV`. -/
theorem qh_lt {d : Inst} (hd : d.ok) {j : Nat} (hj : j < d.HV) : qh d j < d.H := by
  unfold qh
  obtain ⟨c, hc⟩ := hd.hv_dvd
  have hH := hd.H_pos
  have hc0 : 0 < c := by
    rcases Nat.eq_zero_or_pos c with h | h
    · have := hd.HV_pos; rw [hc, h] at this; simp at this
    · exact h
  have : d.HV / d.H = c := by rw [hc]; exact Nat.mul_div_cancel_left c hH
  rw [this]
  exact (Nat.div_lt_iff_lt_mul hc0).mpr (by rw [hc] at hj; linarith)

/-! ## The parts of `Pre` -/

theorem Pre.q_valid {d : Inst} {init : Mem} (h : Pre d init) : d.q.Valid init := h.1
/-- The precondition gives validity of the `k` buffer. -/
theorem Pre.k_valid {d : Inst} {init : Mem} (h : Pre d init) : d.k.Valid init := h.2.1
/-- The precondition gives validity of the `v` buffer. -/
theorem Pre.v_valid {d : Inst} {init : Mem} (h : Pre d init) : d.v.Valid init := h.2.2.1
/-- The precondition gives validity of the `h` buffer. -/
theorem Pre.h_valid {d : Inst} {init : Mem} (h : Pre d init) : d.h.Valid init := h.2.2.2.1
/-- The precondition gives validity of the `o` buffer. -/
theorem Pre.o_valid {d : Inst} {init : Mem} (h : Pre d init) : d.o.Valid init := h.2.2.2.2.1
/-- The precondition says the `q` buffer holds the values `qv`. -/
theorem Pre.q_holds {d : Inst} {init : Mem} (h : Pre d init) : d.q.Holds init d.qv :=
  h.2.2.2.2.2.1
/-- The precondition says the `k` buffer holds the values `kv`. -/
theorem Pre.k_holds {d : Inst} {init : Mem} (h : Pre d init) : d.k.Holds init d.kv :=
  h.2.2.2.2.2.2.1
/-- The precondition says the `v` buffer holds the values `vv`. -/
theorem Pre.v_holds {d : Inst} {init : Mem} (h : Pre d init) : d.v.Holds init d.vv :=
  h.2.2.2.2.2.2.2.1
/-- The precondition says the `h` buffer holds the values `hv`. -/
theorem Pre.h_holds {d : Inst} {init : Mem} (h : Pre d init) : d.h.Holds init d.hv :=
  h.2.2.2.2.2.2.2.2.1
/-- When `useG` is set, the precondition gives validity of `g` and that it holds `gv`. -/
theorem Pre.g_valid {d : Inst} {init : Mem} (h : Pre d init) (hg : d.useG = true) :
    d.g.Valid init ∧ d.g.Holds init d.gv := h.2.2.2.2.2.2.2.2.2.1 hg
/-- When `useGamma` is set, the precondition gives validity of `gg` and that it holds `ggv`. -/
theorem Pre.gg_valid {d : Inst} {init : Mem} (h : Pre d init) (hg : d.useGamma = true) :
    d.gg.Valid init ∧ d.gg.Holds init d.ggv := h.2.2.2.2.2.2.2.2.2.2.1 hg
/-- For variable-length instances, the precondition gives validity and contents of the `cu` and `ci` tables, and `TablesOk`. -/
theorem Pre.tables {d : Inst} {init : Mem} (h : Pre d init) (hv : d.varlen = true) :
    d.cu.Valid init ∧ d.ci.Valid init ∧ d.cu.Holds init d.cuVal ∧ d.ci.Holds init d.ciVal ∧
      TablesOk d := h.2.2.2.2.2.2.2.2.2.2.2 hv

end Claims.ChunkFwdO
