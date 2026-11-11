import Claim
import TVA.Proof
import Proof.Pre

/-!
# One program: the head, then the tail

`body_of_pre`: for the packed or padded head (lines 91–98), the rest of the body is a function of
the program's head `h`, chunk `it`, sequence start `s0` and length `len`; if it is safe for the
values of this program, the whole body is.
-/

namespace Claims.RecomputeWU

open TVA

/-- The launch has `g0 · (B·HV) · 1` programs. -/
theorem launch_size_eq (d : Inst) : (launch d).size = g0 d * (d.B * d.HV) * 1 := rfl

/-- Program `i`'s axis-0 coordinate is `p0Of`. -/
theorem ctx_pid0 (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 0 = p0Of d i := rfl

/-- Program `i`'s axis-1 coordinate is `bhOf`. -/
theorem ctx_pid1 (d : Inst) (i : Fin (launch d).size) : ((launch d).ctx i).pid 1 = bhOf d i := rfl

/-- `cu_seqlens[n]` is present, as the natural number `cuN`. -/
theorem idxAt_cu {d : Inst} {init : Mem} (hS : SeqOk d init) {n : Nat} (hn : n ≤ d.N) :
    idxAt d.cu init n = some ((cuN d init n : Nat) : Int) := by
  obtain ⟨x, hx, h0⟩ := hS.cu_some n hn
  unfold cuN; rw [hx]; simp [Int.toNat_of_nonneg h0]

/-- A `chunk_indices` entry is present and non-negative. -/
theorem idxAt_ci {d : Inst} {init : Mem} (hS : SeqOk d init) {j : Nat} (hj : j < 2 * d.NT) :
    idxAt d.ci init j = some (((idxAt d.ci init j).getD 0).toNat : Int) := by
  obtain ⟨x, hx, h0⟩ := hS.ci_some j hj
  rw [hx]; simp [Int.toNat_of_nonneg h0]

/-- `cu_seqlens` is among the inputs. -/
theorem cu_sub_R (d : Inst) : d.cu.bytes ⊆ Rall d := fun x hx => by simp [Rall, hx]
/-- `chunk_indices` is among the inputs. -/
theorem ci_sub_R (d : Inst) : d.ci.bytes ⊆ Rall d := fun x hx => by simp [Rall, hx]

/-- **The body is safe when its tail is**, whatever the head's dtype. -/
theorem body_of_pre {P : Platform} {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size)
    {rest : I64 → I64 → I64 → I64 → Prog Unit}
    (hrest : ∀ i_h i_t bos T_n : I64, i_h.toInt = (hOf d i : Nat) →
      i_t.toInt = (itOf d init i : Nat) → bos.toInt = (s0Of d init i : Nat) →
      T_n.toInt = (lenOf d init i : Nat) →
      Safe ((launch d).env P) (Rall d) (Wf d init i) (fun _ m' => Qf d init i m') init
        (rest i_h i_t bos T_n)) :
    Safe ((launch d).env P) (Rall d) (Wf d init i) (fun _ m' => Qf d init i m') init
      (preProg (d.cu.ptr 0) (d.ci.ptr 0) (IntV.ofNat d.T) d.HV d.isVarlen ((launch d).ctx i)
        rest) := by
  have hi : (i : Nat) < g0 d * (d.B * d.HV) := by
    have := Nat.lt_of_lt_of_eq i.isLt (launch_size_eq d); omega
  obtain ⟨hp, hb, hh, hbB⟩ := coords hd hi
  have hHV := HV_small hd
  have hgrid := hd.grid_bh
  have hg0 := hd.grid_t
  have hag0 : Ag d init init := fun _ _ => rfl
  by_cases hv : d.isVarlen = true
  · have hS := hpre.seqs hv
    have hNT : p0Of d i < d.NT := by rw [← g0_varlen hv]; exact hp
    obtain ⟨hn, hc⟩ := hS.ci_ok _ hNT
    have hN := hd.n_batch hv
    have hx1 := idxAt_ci hS (j := 2 * p0Of d i) (by omega)
    have hx2 := idxAt_ci hS (j := 2 * p0Of d i + 1) (by omega)
    have hx3 := idxAt_cu hS (n := chSeq d init (p0Of d i)) (by omega)
    have hx4 := idxAt_cu hS (n := chSeq d init (p0Of d i) + 1) (by omega)
    have hL1 := hd.ci_len hv
    have hL2 := hd.cu_len hv
    rw [hv]
    have hc1 := cuN_succ_le hS hn
    have hc01 := hS.cu_mono _ hn
    have hTlt := hd.t_lt
    unfold seqLen at hc
    refine pre_var (hag := hag0) (hsD := hd.idx_ok) (cu := d.cu) (ci := d.ci)
      (hcuR := cu_sub_R d) (hciR := ci_sub_R d) (hcuv := hS.cuv) (hciv := hS.civ)
      (T := IntV.ofNat d.T) (HV := d.HV) (rest := rest)
      (ctx := (launch d).ctx i) (p0 := p0Of d i) (bh := bhOf d i)
      (n := chSeq d init (p0Of d i)) (chunk := chChunk d init (p0Of d i))
      (c0 := cuN d init (chSeq d init (p0Of d i)))
      (c1 := cuN d init (chSeq d init (p0Of d i) + 1)) (hp0 := ctx_pid0 d i) (hbh := ctx_pid1 d i)
      (hp0lt := by omega) (hbhlt := by omega) (hHV0 := hd.hv_pos) (hHV := hHV)
      (hl1 := by omega) (hl3 := by omega) (hx1 := hx1) (hx2 := hx2) (hx3 := hx3) (hx4 := hx4)
      (hn := by omega) (hc1 := by omega) (hc01 := hc01) (hrest := ?_)
    intro i_h i_t bos T_n h1 h2 h3 h4
    refine hrest i_h i_t bos T_n h1 ?_ ?_ ?_
    · simp only [itOf, hv, ite_true]; exact h2
    · simp only [s0Of, hv, ite_true]; exact h3
    · simp only [lenOf, hv, ite_true, seqLen]; exact h4
  · have hv' : d.isVarlen = false := by simpa using hv
    rw [hv']
    refine pre_pad (cu := d.cu.ptr 0) (ci := d.ci.ptr 0) (T := IntV.ofNat d.T) (HV := d.HV)
      (rest := rest) (ctx := (launch d).ctx i)
      (p0 := p0Of d i) (bh := bhOf d i) (Tn := d.T) (hp0 := ctx_pid0 d i) (hbh := ctx_pid1 d i)
      (hp0lt := by omega) (hbhlt := by omega) (hHV0 := hd.hv_pos) (hHV := hHV)
      (hT := IntV.I32.toInt_ofNat hd.t_lt) (hTn := hd.t_lt) (hb := ?_) (hrest := ?_)
    · have hbB' : bOf d i < d.B := hbB
      have h1 : (bOf d i + 1) * d.T ≤ d.B * d.T := Nat.mul_le_mul_right _ hbB'
      have h2 : d.B * d.T ≤ 8192 * 2 ^ 22 :=
        Nat.mul_le_mul hd.b_batch hd.t_ctx
      unfold bOf at h1
      have : bhOf d i / d.HV * d.T + d.T ≤ 8192 * 2 ^ 22 := by nlinarith
      omega
    · intro i_h i_t bos T_n h1 h2 h3 h4
      refine hrest i_h i_t bos T_n h1 ?_ ?_ ?_
      · simp only [itOf, hv', Bool.false_eq_true, ite_false]; exact h2
      · simp only [s0Of, hv', Bool.false_eq_true, ite_false, bOf]; exact h3
      · simp only [lenOf, hv', Bool.false_eq_true, ite_false]; exact h4

end Claims.RecomputeWU
