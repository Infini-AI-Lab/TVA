import Proof.Walk

/-!
# One program is safe

Program `p` (row `pn p`, block `pi p`) converts its two program ids and `tl.arange(0, BD)` to
`int64`, loads its block of `x` (lane `l` reads element `n·D + i·BD + l` when `i·BD + l < D`), and
continues with `tail_noscale` or `tail_scale` by `HAS_SCALE`.
-/

set_option maxHeartbeats 1000000

namespace Claims.LogsumexpFwd

open TVA

/-- **One program.** -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (p : Fin (launch d).size) :
    Safe ((launch d).env P) (Rf d (pn d p) (pi d p)) (Wf d (pn d p) (pi d p))
      (fun _ m => Qf d init (pn d p) (pi d p) m) init ((model d).body ((launch d).ctx p)) := by
  have hp : (p : Nat) < d.N * ND d := lt_of_lt_of_eq p.isLt (launch_size d)
  have hn := pn_lt hp
  have hi := pi_lt hp
  have hval := launch_valid hd
  have hN31 : d.N < 2 ^ 31 := by simpa [launch] using hval.1
  have hnd := hd.nd_lt
  have hNDb := ND_bound hd hpre
  have hnD : pn d p * d.D ≤ 2 ^ 62 :=
    le_trans (Nat.mul_le_mul_right _ hn.le) hNDb
  simp only [model, logsumexp_fwd_kernel]
  refine safe_convert (fun _ => ⟨_, rfl⟩) fun r0 hr0 => ?_
  refine safe_convert (fun _ => ⟨_, rfl⟩) fun r1 hr1 => ?_
  refine safe_convert (fun _ => ⟨_, rfl⟩) fun o_a hoa => ?_
  have hin : r0.item.toInt = (pn d p : Int) :=
    conv_item (pid0_toInt hp hN31) ⟨by omega, by have := hn; omega⟩ hr0
  have hid : r1.item.toInt = (pi d p : Int) :=
    conv_item (pid1_toInt hp hd) ⟨by omega, by omega⟩ hr1
  have hoa' := oa_toInt hd.bd_le hoa
  have hod := od_toInt hd.bd_le (by omega) hid hoa'
  have hmd := md_get hd.d_lt hod
  refine safe_load_buf d.x (g := fun (l : Idx [d.BD]) => xIdx d (pn d p) (pi d p * d.BD + l.1.val))
    (en := fun (l : Idx [d.BD]) => decide (pi d p * d.BD + l.1.val < d.D))
    (Buf.addrs_of_get d.x hmd (fun l _ => px_get hin hnD hd.d_lt hod l))
    (fun l hl => lt_of_lt_of_le (xIdx_lt hn (of_decide_eq_true hl)) hd.x_len) hpre.1 ?_
    fun b_x hbx _ => ?_
  · refine Buf.elems_mono _ ?_
    rintro k ⟨l, hl, rfl⟩
    have hl' := of_decide_eq_true hl
    refine ⟨pi d p * d.BD + l.1.val, ?_, rfl⟩
    simp only [blk, Finset.mem_Ico]
    have := l.1.isLt
    have h2 : (pi d p + 1) * d.BD = pi d p * d.BD + d.BD := Nat.succ_mul _ _
    omega
  have hbx' : ∀ l : Idx [d.BD], pi d p * d.BD + l.1.val < d.D →
      b_x.get l = xsOf d init (pn d p) (pi d p * d.BD + l.1.val) := by
    intro l hl
    simp only [xsOf]
    rw [hbx l (decide_eq_true hl)]
    rfl
  have hpz := pz_eq hd hin hid hn hi
  cases hs : d.hasScale
  · simp only [hs, Bool.false_eq_true, ↓reduceIte]
    rw [hpz]
    exact tail_noscale hP hd hs hpre hn hi hmd hbx' rfl
  · simp only [hs, ↓reduceIte]
    rw [hpz]
    exact tail_scale hP hd hs hpre hn hi hmd hbx' rfl

end Claims.LogsumexpFwd
