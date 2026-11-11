import Proof.Defs

/-!
# `attn_bwd_f16`: the launch and its programs

The grid, the program coordinates (`pidX`, `head`), and the launch is valid.
-/

namespace Claims.AttnBwd

open TVA

/-- Axis 0 of the grid: the number of blocks of `128` key rows. -/
def gx (d : Inst) : Nat := d.N_CTX / 128

/-- Program `i`'s block of key (and query) rows: its coordinate on axis 0. -/
def pidX (d : Inst) (i : Nat) : Nat := i % gx d

/-- Program `i`'s head `b = z·N_HEAD + h`: its coordinate on axis 2. -/
def head (d : Inst) (i : Nat) : Nat := i / gx d

/-- The launch has `gx · 1 · G` programs. -/
theorem launch_size (d : Inst) : (launch d).size = gx d * 1 * G d := by
  simp [Launch.size, launch, gx]

/-- Program `i` has `program_id(0) = pidX i`. -/
theorem launch_pid0 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 0 = pidX d i := by
  simp [Launch.ctx, launch, pidX, gx]

/-- Program `i` has `program_id(2) = head i`. -/
theorem launch_pid2 (d : Inst) (i : Nat) : ((launch d).ctx i).pid 2 = head d i := by
  simp [Launch.ctx, launch, head, gx]

/-- `Shape.TritonOk` from powers of two and a bound on the product. -/
theorem tritonOk_of {s : Shape} (h1 : ∀ k ∈ s, ∃ e, k = 2 ^ e) (h2 : s.prod ≤ 2 ^ 20) :
    s.TritonOk := ⟨h1, h2⟩

/-- **The launch is valid** on every covered instance. -/
theorem launch_valid {d : Inst} (hd : d.ok) : (launch d).Valid := by
  have hHD := hd.hd
  have hHDle : d.HEAD_DIM ≤ 256 := by omega
  have hp : ∃ e, d.HEAD_DIM = 2 ^ e := by
    rcases hHD with h | h | h | h | h
    · exact ⟨4, by rw [h]; rfl⟩
    · exact ⟨5, by rw [h]; rfl⟩
    · exact ⟨6, by rw [h]; rfl⟩
    · exact ⟨7, by rw [h]; rfl⟩
    · exact ⟨8, by rw [h]; rfl⟩
  have h1 : ∃ e, (1 : Nat) = 2 ^ e := ⟨0, rfl⟩
  have h16 : ∃ e, (16 : Nat) = 2 ^ e := ⟨4, rfl⟩
  have h32 : ∃ e, (32 : Nat) = 2 ^ e := ⟨5, rfl⟩
  have h128 : ∃ e, (128 : Nat) = 2 ^ e := ⟨7, rfl⟩
  refine ⟨?_, ?_, ?_, ?_, rfl, rfl, rfl, ?_⟩
  · show gx d < 2 ^ 31
    have := hd.n_le
    unfold gx; omega
  · show 1 < 2 ^ 16
    norm_num
  · show G d < 2 ^ 16
    exact hd.g_lt
  · rw [launch_size]
    have := hd.n_le
    have := hd.g_lt
    have : gx d ≤ 128 := by unfold gx; omega
    nlinarith
  · intro s hs
    simp only [launch, blocks, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl | rfl
    all_goals refine tritonOk_of ?_ ?_
    all_goals first
      | (intro k hk
         simp only [List.mem_cons, List.not_mem_nil, or_false] at hk
         rcases hk with rfl | rfl <;> assumption)
      | (simp only [List.prod_cons, List.prod_nil]; nlinarith)

end Claims.AttnBwd
