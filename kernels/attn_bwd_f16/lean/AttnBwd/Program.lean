import AttnBwd.Out

/-!
# `attn_bwd_f16`: one program

`safe_program`: program `(p, b)` of the grid (block `p` of `128` rows, head `b`), run alone from
the initial memory, is safe (`TVA.Safe`): it reads only the six inputs (`inBytes`), writes only
its tiles of `dv`, `dk` and `dq` (`W`), each byte once, and leaves there what `OutSpec` says
(`Qp`). The proof walks the model line by line with the library's rules; the two helpers are
`safe_dkdv` and `safe_dq`.
-/

namespace Kernels.AttnBwd

open TVA

/-! ## The program's tiles -/

/-- Element `(r, c)` of program `(p, b)`'s tile: row `128p + r` of head `b`. -/
def tileG (d : Inst) (b p : Nat) : Idx [128, d.HEAD_DIM] → Nat :=
  fun ix => b * d.N_CTX * d.HEAD_DIM + (128 * p + (ix.1 : Nat)) * d.HEAD_DIM + (ix.2.1 : Nat)

/-- The elements of program `(p, b)`'s tile. -/
def tileSet (d : Inst) (b p : Nat) : Set Nat := Buf.img (tileG d b p) (fun _ => true)

/-- Program `(p, b)`'s write footprint: its tiles of `dv`, `dk` and `dq`. -/
def W (d : Inst) (b p : Nat) : Set Int :=
  d.DV.elems (tileSet d b p) ∪ d.DK.elems (tileSet d b p) ∪ d.DQ.elems (tileSet d b p)

/-- What program `(p, b)` leaves in memory `fin`: its rows of `dv`, `dk`, `dq` as `OutSpec` says. -/
def Qp (d : Inst) (init : Mem) (b p : Nat) (fin : Mem) : Prop :=
  ∀ (r : Fin 128) (j : Fin d.HEAD_DIM),
    OutSpec d init fin d.DV b (128 * p + r) j (fun hv => dVE d hv (128 * p + r) j)
      (fun hv => dVTol d hv (128 * p + r) j) ∧
    OutSpec d init fin d.DK b (128 * p + r) j (fun hv => dKE d hv (128 * p + r) j)
      (fun hv => dKTol d hv (128 * p + r) j) ∧
    OutSpec d init fin d.DQ b (128 * p + r) j (fun hv => dQE d hv (128 * p + r) j)
      (fun hv => dQTol d hv (128 * p + r) j)

theorem tileG_eq (d : Inst) (b p : Nat) (r : Fin 128) (c : Fin d.HEAD_DIM) :
    tileG d b p (r, c, ()) = eIdx d b (128 * p + r) c := by rw [eIdx_eq]; rfl

theorem tileSet_sub {d : Inst} {b p : Nat} (hb : b < G d) (hp : 128 * p + 128 ≤ d.N_CTX) :
    tileSet d b p ⊆ tSet d := by
  rintro _ ⟨⟨r, c, ⟨⟩⟩, -, rfl⟩
  exact elem_lt hb (show 128 * p + (r : Nat) < d.N_CTX by have := r.2; omega) c.2

theorem ptr_adj {dt : DType} (Bf : Buf dt) (base : Nat) {a : I64} (ha : a.toInt = (base : Int)) :
    Bf.ptr 0 + a = Bf.ptr ↑base := by
  rw [Ptr.add_int, Buf.ptr_offset, ha, zero_add]

theorem divmod_inj {K x x' y y' : Nat} (hy : y < K) (hy' : y' < K) (he : x * K + y = x' * K + y') :
    x = x' ∧ y = y' := by
  have e1 : (x * K + y) / K = x := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hy, zero_add]
  have e2 : (x' * K + y') / K = x' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hy', zero_add]
  have e3 : (x * K + y) % K = y := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hy]
  have e4 : (x' * K + y') % K = y' := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hy']
  rw [he] at e1 e3
  exact ⟨e1.symm.trans e2, e3.symm.trans e4⟩

/-- Tiles of different programs, and different lanes of one tile, are different elements. -/
theorem tileG_inj {d : Inst} {b p b' p' : Nat} (hp : 128 * p + 128 ≤ d.N_CTX)
    (hp' : 128 * p' + 128 ≤ d.N_CTX) (ix ix' : Idx [128, d.HEAD_DIM])
    (h : tileG d b p ix = tileG d b' p' ix') : b = b' ∧ p = p' ∧ ix = ix' := by
  obtain ⟨r, c, ⟨⟩⟩ := ix
  obtain ⟨r', c', ⟨⟩⟩ := ix'
  have hr := r.2
  have hr' := r'.2
  have e : (b * d.N_CTX + (128 * p + r)) * d.HEAD_DIM + c =
      (b' * d.N_CTX + (128 * p' + r')) * d.HEAD_DIM + c' := by
    have h' : b * d.N_CTX * d.HEAD_DIM + (128 * p + (r : Nat)) * d.HEAD_DIM + (c : Nat) =
        b' * d.N_CTX * d.HEAD_DIM + (128 * p' + (r' : Nat)) * d.HEAD_DIM + (c' : Nat) := h
    have e1 : (b * d.N_CTX + (128 * p + (r : Nat))) * d.HEAD_DIM =
        b * d.N_CTX * d.HEAD_DIM + (128 * p + (r : Nat)) * d.HEAD_DIM := by ring
    have e2 : (b' * d.N_CTX + (128 * p' + (r' : Nat))) * d.HEAD_DIM =
        b' * d.N_CTX * d.HEAD_DIM + (128 * p' + (r' : Nat)) * d.HEAD_DIM := by ring
    rw [e1, e2]; exact h'
  obtain ⟨h1, h2⟩ := divmod_inj c.2 c'.2 e
  obtain ⟨h3, h4⟩ := divmod_inj (K := d.N_CTX) (show 128 * p + (r : Nat) < d.N_CTX by omega)
    (show 128 * p' + (r' : Nat) < d.N_CTX by omega) h1
  obtain ⟨h5, h6⟩ := divmod_inj (K := 128) (x := p) (x' := p') r.2 r'.2 (by rw [Nat.mul_comm p, Nat.mul_comm p']; exact h4)
  refine ⟨h3, h5, ?_⟩
  have : r = r' := Fin.ext h6
  have : c = c' := Fin.ext h2
  subst_vars; rfl

/-! ## The two phases' rows -/

/-- `start_m` after the masked blocks of `dk`, `dv` (line 429): `128p + 128` with `CAUSAL`, else `0`. -/
def m1 (d : Inst) (p : Nat) : Nat := if d.CAUSAL then 128 * p + 128 else 0

/-- The query rows the masked blocks cover. -/
def C1 (d : Inst) (p : Nat) : Finset ℕ := if d.CAUSAL then Finset.Ico (128 * p) (128 * p + 128) else ∅

/-- The masked steps of `dk`, `dv`. -/
def T1 (d : Inst) : Nat := if d.CAUSAL then 8 else 0

/-- The key rows the masked blocks of `dq` cover. -/
def Cq1 (d : Inst) (p : Nat) : Finset ℕ := if d.CAUSAL then Finset.Ico (128 * p) (128 * p + 128) else ∅

/-- The unmasked steps of `dq` (lines 456, 486): `4p` with `CAUSAL`, else `N_CTX / 32`. -/
def Tq2 (d : Inst) (p : Nat) : Nat := if d.CAUSAL then 128 * p / 32 else d.N_CTX / 32

section Phases
variable {d : Inst} {p : Nat}

theorem m1_add (hd : d.ok) (hp : 128 * p + 128 ≤ d.N_CTX) :
    m1 d p ≤ d.N_CTX ∧ m1 d p + (d.N_CTX - m1 d p) / 32 * 32 = d.N_CTX := by
  have := hd.n_mod
  unfold m1; split_ifs <;> constructor <;> omega

theorem C1_lt (x : Nat) (hx : x ∈ C1 d p) : x < m1 d p := by
  unfold C1 m1 at *; split_ifs at * with h
  · rw [Finset.mem_Ico] at hx; omega
  · simp at hx

theorem C1_sub (hp : 128 * p + 128 ≤ d.N_CTX) : C1 d p ⊆ Finset.range d.N_CTX := by
  unfold C1; split_ifs
  · intro x hx; rw [Finset.mem_Ico] at hx; rw [Finset.mem_range]; omega
  · exact Finset.empty_subset _

theorem allowed_ge (r : Fin 128) {mm : Nat} (h : m1 d p ≤ mm) : Allowed d (128 * p + r) mm := by
  unfold m1 at h; unfold Allowed
  cases hc : d.CAUSAL
  · left; rfl
  · right; rw [hc] at h; simp at h; have := r.2; omega

theorem kv_steps (_hd : d.ok) (_hp : 128 * p + 128 ≤ d.N_CTX) (r : Fin 128) :
    T1 d + (d.N_CTX - m1 d p) / 32 = stepsKV d (128 * p + r) := by
  have hr := r.2
  have e : (128 * p + (r : Nat)) / 128 = p := by omega
  unfold T1 m1 stepsKV; rw [e]
  split_ifs <;> omega

theorem kv_vanish (hd : d.ok) (hp : 128 * p + 128 ≤ d.N_CTX) (r : Fin 128) {mm : Nat}
    (hm : mm < d.N_CTX)
    (hn : mm ∉ C1 d p ∪ Finset.Ico (m1 d p) (m1 d p + (d.N_CTX - m1 d p) / 32 * 32)) :
    ¬ Allowed d (128 * p + r) mm := by
  have h2 := (m1_add hd hp).2
  rw [h2] at hn
  unfold C1 m1 at hn; unfold Allowed
  cases hc : d.CAUSAL
  · simp only [hc] at hn; simp at hn; omega
  · simp only [hc, ite_true, Finset.mem_union, Finset.mem_Ico, not_or, not_and, not_lt] at hn
    simp only [Bool.true_eq_false, false_or, not_le]
    by_contra h'
    have := hn.1
    omega

theorem kv_cover_sub (hd : d.ok) (hp : 128 * p + 128 ≤ d.N_CTX) :
    C1 d p ∪ Finset.Ico (m1 d p) (m1 d p + (d.N_CTX - m1 d p) / 32 * 32) ⊆ Finset.range d.N_CTX :=
  cover_sub1 (C1_sub hp) (by rw [(m1_add hd hp).2])

theorem q_rows (hd : d.ok) (hp : 128 * p + 128 ≤ d.N_CTX) : Tq2 d p * 32 ≤ d.N_CTX := by
  have := hd.n_mod
  unfold Tq2; split_ifs <;> omega

theorem Cq1_apart (x : Nat) (hx : x ∈ Cq1 d p) : x < 0 ∨ 0 + Tq2 d p * 32 ≤ x := by
  unfold Cq1 Tq2 at *; split_ifs at * with h
  · rw [Finset.mem_Ico] at hx; right; omega
  · simp at hx

theorem Cq1_sub (hp : 128 * p + 128 ≤ d.N_CTX) : Cq1 d p ⊆ Finset.range d.N_CTX := by
  unfold Cq1; split_ifs
  · intro x hx; rw [Finset.mem_Ico] at hx; rw [Finset.mem_range]; omega
  · exact Finset.empty_subset _

theorem allowed_lt (r : Fin 128) {nn : Nat} (h : nn < 0 + Tq2 d p * 32) : Allowed d nn (128 * p + r) := by
  unfold Tq2 at h; unfold Allowed
  cases hc : d.CAUSAL
  · left; rfl
  · right; rw [hc] at h; simp at h; omega

theorem q_steps (r : Fin 128) : (if d.CAUSAL then 8 else 0) + Tq2 d p = stepsQ d (128 * p + r) := by
  have hr := r.2
  have e : (128 * p + (r : Nat)) / 128 = p := by omega
  unfold Tq2 stepsQ; rw [e]
  split_ifs <;> omega

theorem q_vanish (hd : d.ok) (_hp : 128 * p + 128 ≤ d.N_CTX) (r : Fin 128) {nn : Nat}
    (hn : nn < d.N_CTX) (hnc : nn ∉ Cq1 d p ∪ Finset.Ico 0 (0 + Tq2 d p * 32)) :
    ¬ Allowed d nn (128 * p + r) := by
  have := hd.n_mod
  unfold Cq1 Tq2 at hnc; unfold Allowed
  cases hc : d.CAUSAL
  · simp only [hc] at hnc; simp at hnc; omega
  · simp only [hc, ite_true, Finset.mem_union, Finset.mem_Ico, not_or, not_and, not_lt] at hnc
    simp only [Bool.true_eq_false, false_or, not_le]
    have := r.2
    omega

theorem q_cover_sub (hd : d.ok) (hp : 128 * p + 128 ≤ d.N_CTX) :
    Cq1 d p ∪ Finset.Ico 0 (0 + Tq2 d p * 32) ⊆ Finset.range d.N_CTX :=
  cover_sub1 (Cq1_sub hp) (by rw [zero_add]; exact q_rows hd hp)

end Phases

/-! ## The outputs of one program -/

section Outputs
variable {d : Inst}

theorem accI_range {a : F32} {C : Finset ℕ} {N T : ℕ} {f g h : ℕ → ℝ} (hC : C ⊆ Finset.range N)
    (hz : ∀ x, x < N → x ∉ C → f x = 0 ∧ g x = 0 ∧ h x = 0)
    (hacc : AccI a (∑ x ∈ C, f x) (∑ x ∈ C, g x) (∑ x ∈ C, h x) T) :
    AccI a (∑ x ∈ Finset.range N, f x) (∑ x ∈ Finset.range N, g x) (∑ x ∈ Finset.range N, h x) T := by
  rwa [sum_ext_range hC f (fun x h1 h2 => (hz x h1 h2).1), sum_ext_range hC g (fun x h1 h2 => (hz x h1 h2).2.1),
    sum_ext_range hC h (fun x h1 h2 => (hz x h1 h2).2.2)] at hacc

theorem stepsKV_le (hd : d.ok) (n : Nat) : stepsKV d n ≤ 1024 := by
  have := hd.n_le; unfold stepsKV; split_ifs <;> omega

theorem stepsQ_le (hd : d.ok) {m : Nat} (hm : m < d.N_CTX) : stepsQ d m ≤ 1024 := by
  have := hd.n_le; unfold stepsQ; split_ifs <;> omega

/-- The sizes the product by a scale needs: `|Σ dS·x| + tolerance ≤ 2^49`. -/
theorem scaled_size (hd : d.ok) {hv : HeadV} (_hok : hv.Ok d) {T : ℕ} (hT : T ≤ 1024) (x : ℕ → F16)
    (hx : ∀ m, m < d.N_CTX → (x m).isFinite = true) (e g h : ℕ → ℝ)
    (he : ∀ m, m < d.N_CTX → |e m| ≤ 2 ^ 15) (hg : ∀ m, m < d.N_CTX → 0 ≤ g m ∧ g m ≤ 2 ^ 17)
    (hh : ∀ m, m < d.N_CTX → 0 ≤ h m ∧ h m ≤ 2 ^ 17) :
    |∑ m ∈ Finset.range d.N_CTX, e m * (x m).toReal| +
      (∑ m ∈ Finset.range d.N_CTX, g m * |(x m).toReal| +
        accB T (∑ m ∈ Finset.range d.N_CTX, h m * |(x m).toReal|)) ≤ 2 ^ 49 ∧
    0 ≤ |∑ m ∈ Finset.range d.N_CTX, e m * (x m).toReal| +
      (∑ m ∈ Finset.range d.N_CTX, g m * |(x m).toReal| +
        accB T (∑ m ∈ Finset.range d.N_CTX, h m * |(x m).toReal|)) := by
  have hsub : Finset.range d.N_CTX ⊆ Finset.range d.N_CTX := subset_rfl
  have h1 : |∑ m ∈ Finset.range d.N_CTX, e m * (x m).toReal| ≤ 2 ^ 47 := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    have := sum_bd_le hd hsub (fun m => |e m|) x (fun m hm => ⟨abs_nonneg _, (he m hm).trans (by norm_num)⟩) hx
    refine le_trans (le_of_eq ?_) this
    exact Finset.sum_congr rfl fun m _ => abs_mul _ _
  have h2 := sum_bd_le hd hsub g x hg hx
  have h3 := sum_bd_le hd hsub h x hh hx
  have h30 : 0 ≤ ∑ m ∈ Finset.range d.N_CTX, h m * |(x m).toReal| :=
    Finset.sum_nonneg fun m hm => mul_nonneg (hh m (Finset.mem_range.mp hm)).1 (abs_nonneg _)
  have h4 := accB_le hT h30
  have : (2 : ℝ) ^ (-5 : ℤ) * ∑ m ∈ Finset.range d.N_CTX, h m * |(x m).toReal| ≤ 2 ^ (-5 : ℤ) * 2 ^ 47 :=
    mul_le_mul_of_nonneg_left h3 (by positivity)
  have : (2 : ℝ) ^ 47 + 2 ^ 47 + 2 ^ (-5 : ℤ) * 2 ^ 47 + 2 ^ (-85 : ℤ) ≤ 2 ^ 49 := by norm_num
  have h20 : 0 ≤ ∑ m ∈ Finset.range d.N_CTX, g m * |(x m).toReal| :=
    Finset.sum_nonneg fun m hm => mul_nonneg (hg m (Finset.mem_range.mp hm)).1 (abs_nonneg _)
  have h40 := accB_nn T h30
  have := abs_nonneg (∑ m ∈ Finset.range d.N_CTX, e m * (x m).toReal)
  constructor <;> linarith

theorem sm_abs (hd : d.ok) : |(smF d).toReal| ≤ 2 ^ 16 := by
  rw [toReal_def]
  have : ((|(smF d).toRat| : ℚ) : ℝ) ≤ ((2 ^ 16 : ℚ) : ℝ) := by exact_mod_cast hd.sm.2
  push_cast at this; exact this

end Outputs

/-- **What program `(p, b)` leaves**: from the accumulators' invariants after the last step, the
products by the scales, the roundings and the stored values, `Qp`. -/
theorem qp_final {d : Inst} (hd : d.ok) {init mC : Mem} {b p : Nat} (hp : 128 * p + 128 ≤ d.N_CTX)
    {dk2 dv2 dq dk' dq' : Tensor [128, d.HEAD_DIM] F32} {dv16 dk16 dq16 : Tensor [128, d.HEAD_DIM] F16}
    (hinv2 : KVInv d init b p (C1 d p ∪ Finset.Ico (m1 d p) (m1 d p + (d.N_CTX - m1 d p) / 32 * 32))
      (T1 d + (d.N_CTX - m1 d p) / 32) dk2 dv2)
    (hinvq : QInv d init b p (Cq1 d p ∪ Finset.Ico 0 (0 + Tq2 d p * 32))
      ((if d.CAUSAL then 8 else 0) + Tq2 d p) dq)
    (hdv16 : ∀ i, FloatV.NaNEq (FloatV.cvt (dv2.get i)) (dv16.get i))
    (hdk' : ∀ i, FloatV.NaNEq (FloatV.mul (dk2.get i) (smF d)) (dk'.get i))
    (hdk16 : ∀ i, FloatV.NaNEq (FloatV.cvt (dk'.get i)) (dk16.get i))
    (hdq' : ∀ i, FloatV.NaNEq (FloatV.mul (dq.get i) ln2F) (dq'.get i))
    (hdq16 : ∀ i, FloatV.NaNEq (FloatV.cvt (dq'.get i)) (dq16.get i))
    (hrdV : ∀ i, mC.read .f16 (d.DV.ptr ↑(tileG d b p i)).addr = some (dv16.get i))
    (hrdK : ∀ i, mC.read .f16 (d.DK.ptr ↑(tileG d b p i)).addr = some (dk16.get i))
    (hrdQ : ∀ i, mC.read .f16 (d.DQ.ptr ↑(tileG d b p i)).addr = some (dq16.get i)) :
    Qp d init b p mC := by
  intro r j
  have hn : 128 * p + (r : Nat) < d.N_CTX := by have := r.2; omega
  have hln2 : 0 < ln2F.toReal := by rw [ln2_real]; norm_num
  have hln2le : |ln2F.toReal| ≤ 1 := by rw [ln2_real]; norm_num [abs_of_pos]
  refine ⟨⟨dv16.get (r, j, ()), by rw [← tileG_eq]; exact hrdV _, fun hv hH hO => ?_⟩,
    ⟨dk16.get (r, j, ()), by rw [← tileG_eq]; exact hrdK _, fun hv hH hO => ?_⟩,
    ⟨dq16.get (r, j, ()), by rw [← tileG_eq]; exact hrdQ _, fun hv hH hO => ?_⟩⟩
  · have h := (hinv2 r j hv hH hO).1
    rw [kv_steps hd hp r] at h
    exact out_dv h (kv_cover_sub hd hp) (fun m hm hn' => kv_vanish hd hp r hm hn') (hdv16 (r, j, ()))
  · have h := (hinv2 r j hv hH hO).2
    rw [kv_steps hd hp r] at h
    have h' := accI_range (kv_cover_sub hd hp) (fun m hm hn' => by
      obtain ⟨-, -, -, h4, h5, h6⟩ := vanish (hv := hv) (kv_vanish hd hp r hm hn')
      rw [h4, h5, h6]; simp) h
    have hsz := scaled_size hd hO (stepsKV_le hd (128 * p + r)) (fun m => hv.q m j)
      (fun m hm => hO.q m j hm j.2) (fun m => dsE d hv (128 * p + r) m)
      (fun m => ds16Err d hv (128 * p + r) m) (fun m => ds16B d hv (128 * p + r) m)
      (fun m hm => dsE_abs hd hO hn hm) (fun m hm => ds16Err_bd hd hO hn hm)
      (fun m hm => ds16B_bd hd hO hn hm)
    have hs := sm_abs hd
    exact out_scaled h' hd.sm.1 (by
        have := mul_le_mul hs hsz.1 hsz.2 (by norm_num)
        refine this.trans ?_; norm_num)
      (hdk' (r, j, ())) (hdk16 (r, j, ())) rfl rfl
  · have h := hinvq r j hv hH hO
    rw [q_steps r] at h
    have h' := accI_range (q_cover_sub hd hp) (fun n hn' hnc => by
      obtain ⟨-, -, -, h4, h5, h6⟩ := vanish (hv := hv) (q_vanish hd hp r hn' hnc)
      rw [h4, h5, h6]; simp) h
    have hsz := scaled_size hd hO (stepsQ_le hd hn) (fun n => hv.k n j)
      (fun n hn' => hO.k n j hn' j.2) (fun n => dsE d hv n (128 * p + r))
      (fun n => ds16Err d hv n (128 * p + r)) (fun n => ds16B d hv n (128 * p + r))
      (fun n hn' => dsE_abs hd hO hn' hn) (fun n hn' => ds16Err_bd hd hO hn' hn)
      (fun n hn' => ds16B_bd hd hO hn' hn)
    exact out_scaled h' ln2_fin (by
        have := mul_le_mul hln2le hsz.1 hsz.2 (by norm_num)
        refine this.trans ?_; norm_num)
      (hdq' (r, j, ())) (hdq16 (r, j, ())) rfl (by rw [abs_of_pos hln2]; rfl)

/-! ## One program -/

/-- **One program is safe.** Program `(pid 0, pid 2) = (p, b)` reads only the inputs, writes only
its tiles `W d b p` of `DV`, `DK`, `DQ`, and leaves `Qp`: lines 375–500 in order, with the two
`dkdv` phases (lines 422–443) and the two `dq` phases (lines 468–496) by `safe_dkdv`/`safe_dq`. -/
theorem safe_program {E : Env} (hdot : Rules.Num.DotF16 E.P) (hdt : Rules.Num.DotTotal E.P dotCfg)
    (hexp : Rules.Num.Exp2F32 E.P) (hmt : Rules.Num.MathTotal E.P .exp2 .binary32)
    {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b p : Nat} (hb : b < G d)
    (hp : p < d.N_CTX / 128) (c : Ctx) (hc0 : c.pid 0 = p) (hc2 : c.pid 2 = b) :
    Safe E (inBytes d) (W d b p) (fun _ m => Qp d init b p m) init ((model d).body c) := by
  have hHD := hd_le hd
  have hHD0 := hd_pos hd
  have hNle := hd.n_le
  have hsize := hd.size_lt
  have hp' : 128 * p + 128 ≤ d.N_CTX := by have := hd.n_mod; omega
  have hG0 : 0 < G d := by omega
  have hH0 : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · unfold G at hb; rw [h] at hb; simp at hb
    · exact h
  have hNHD : d.N_CTX * d.HEAD_DIM < 2 ^ 31 := by
    have : d.N_CTX * d.HEAD_DIM ≤ G d * d.N_CTX * d.HEAD_DIM := by
      rw [Nat.mul_assoc]; exact Nat.le_mul_of_pos_left _ hG0
    omega
  have hbN : b * d.N_CTX < 2 ^ 31 := by
    have : b * d.N_CTX ≤ b * d.N_CTX * d.HEAD_DIM := Nat.le_mul_of_pos_right _ hHD0
    have : b * d.N_CTX * d.HEAD_DIM < G d * d.N_CTX * d.HEAD_DIM :=
      Nat.mul_lt_mul_of_pos_right (Nat.mul_lt_mul_of_pos_right hb (by omega)) hHD0
    omega
  obtain ⟨hQv, hKv, hVv, hDOv, hDQv, hDKv, hDVv, hMv, hDv⟩ := hpre
  simp only [model, attn_bwd]
  rw [show tl.program_id c 2 = IntV.ofNat b by simp [tl.program_id, hc2],
    show tl.program_id c 0 = IntV.ofNat p by simp [tl.program_id, hc0]]
  have hg := hd.g_lt
  have hHlt : d.H ≤ G d := by
    unfold G
    rcases Nat.eq_zero_or_pos d.Z with h | h
    · unfold G at hG0; rw [h] at hG0; simp at hG0
    · exact Nat.le_mul_of_pos_left _ h
  have hbv : (IntV.ofNat b : I32).toInt = (b : Int) := i32_nat (by omega)
  have hHv : (IntV.ofNat d.H : I32).toInt = (d.H : Int) := i32_nat (by omega)
  -- line 385
  refine safe_irem1 (by rw [hbv]; omega) (by rw [hHv]; omega) fun r1 hr1 => ?_
  refine safe_idiv1 (by rw [hbv]; omega) (by rw [hHv]; omega) fun r2 hr2 => ?_
  rw [hbv, hHv] at hr1 hr2
  have hr1' : r1.item.toInt = ((b % d.H : Nat) : Int) := by rw [hr1, Int.natCast_mod]
  have hr2' : r2.item.toInt = ((b / d.H : Nat) : Int) := by rw [hr2, Int.natCast_div]
  have hbNHD : b * d.N_CTX * d.HEAD_DIM < 2 ^ 31 := by
    have : b * d.N_CTX * d.HEAD_DIM < G d * d.N_CTX * d.HEAD_DIM :=
      Nat.mul_lt_mul_of_pos_right (Nat.mul_lt_mul_of_pos_right hb (by omega)) hHD0
    omega
  have hsum : d.N_CTX * d.HEAD_DIM * (b % d.H) + d.H * d.N_CTX * d.HEAD_DIM * (b / d.H) =
      b * d.N_CTX * d.HEAD_DIM := by
    have h := Nat.mod_add_div b d.H
    calc d.N_CTX * d.HEAD_DIM * (b % d.H) + d.H * d.N_CTX * d.HEAD_DIM * (b / d.H)
        = d.N_CTX * d.HEAD_DIM * (b % d.H + d.H * (b / d.H)) := by ring
      _ = b * d.N_CTX * d.HEAD_DIM := by rw [h]; ring
  have hadj : (IntV.cast (IntV.ofNat (d.N_CTX * d.HEAD_DIM) * r1.item +
      IntV.ofNat (d.H * d.N_CTX * d.HEAD_DIM) * r2.item) : I64).toInt = ((b * d.N_CTX * d.HEAD_DIM : Nat) : Int) := by
    rw [IntV.toInt_cast _ (by norm_num), IntV.I32.add_nat (IntV.I32.mul_nat (i32_nat (by omega)) hr1' (by omega))
      (IntV.I32.mul_nat (i32_nat (by nlinarith)) hr2' (by omega)) (by omega), hsum]
  generalize (IntV.cast (IntV.ofNat (d.N_CTX * d.HEAD_DIM) * r1.item +
      IntV.ofNat (d.H * d.N_CTX * d.HEAD_DIM) * r2.item) : I64) = adj at hadj ⊢
  have hoch : (IntV.cast (IntV.ofNat b * IntV.ofNat d.N_CTX : I32) : I64).toInt = ((b * d.N_CTX : Nat) : Int) := by
    rw [IntV.toInt_cast _ (by norm_num), IntV.I32.mul_nat hbv (i32_nat (by omega)) hbN]
  generalize (IntV.cast (IntV.ofNat b * IntV.ofNat d.N_CTX : I32) : I64) = och at hoch ⊢
  have hsn : (IntV.ofNat p * IntV.ofNat 128 : I32).toInt = ((128 * p : Nat) : Int) := by
    rw [IntV.I32.mul_nat (i32_nat (by omega)) (i32_nat (by norm_num)) (by omega)]; push_cast; ring
  generalize (IntV.ofNat p * IntV.ofNat 128 : I32) = sn at hsn ⊢
  rw [ptr_adj d.Q _ hadj, ptr_adj d.K _ hadj, ptr_adj d.V _ hadj, ptr_adj d.DO _ hadj,
    ptr_adj d.DQ _ hadj, ptr_adj d.DK _ hadj, ptr_adj d.DV _ hadj, ptr_adj d.M _ hoch, ptr_adj d.D _ hoch]
  -- the tiles of the program's rows
  have hrn : RowsAt (sn + tl.arange 0 128 : Tensor [128] I32) (128 * p) := rowsAt_add hsn (by omega)
  have hk0 : RowsAt (tl.arange 0 d.HEAD_DIM : Tensor [d.HEAD_DIM] I32) 0 := rowsAt_arange0 (by omega)
  have hfit : (128 * p + 128) * d.HEAD_DIM < 2 ^ 31 := lt_of_le_of_lt (Nat.mul_le_mul_right _ hp') hNHD
  have hAddr : ∀ {dt : DType} (Bf : Buf dt), Bf.Addrs (tl.addrs (Bf.ptr ↑(b * d.N_CTX * d.HEAD_DIM) +
      (sn + tl.arange 0 128 : Tensor [128] I32).col d.HEAD_DIM * (IntV.ofNat d.HEAD_DIM : I32) +
      (tl.arange 0 d.HEAD_DIM : Tensor [d.HEAD_DIM] I32).row 128 * (IntV.ofNat 1 : I32)) none)
      (tileG d b p) (fun _ => true) := fun Bf => Buf.addrs_of_get_all Bf fun ⟨r, c, ⟨⟩⟩ =>
    congrArg (fun (x : Tensor [128, d.HEAD_DIM] (Ptr _)) => x.get (r, c, ())) (tile_rc Bf _ hrn hk0 hfit)
  have hin : ∀ (len : Nat), G d * d.N_CTX * d.HEAD_DIM ≤ len → ∀ ix, (fun _ => true) ix = true →
      tileG d b p ix < len := by
    rintro len hlen ⟨r, c, ⟨⟩⟩ _
    have := elem_lt hb (show 128 * p + (r : Nat) < d.N_CTX by have := r.2; omega) c.2
    exact lt_of_lt_of_le this hlen
  have hsubT : tileSet d b p ⊆ tSet d := tileSet_sub hb hp'
  -- lines 412–413
  refine safe_load_buf d.K (hAddr d.K) (hin _ hd.k_len) hKv ((d.K.elems_mono hsubT).trans (k_sub d))
    fun k hk _ => ?_
  refine safe_load_buf d.V (hAddr d.V) (hin _ hd.v_len) hVv ((d.V.elems_mono hsubT).trans (v_sub d))
    fun v hv _ => ?_
  have hkv : ∀ hv' : HeadV, hv'.Holds d init b → ∀ (r : Fin 128) (c : Fin d.HEAD_DIM),
      k.get2 r c = hv'.k (128 * p + r) c ∧ v.get2 r c = hv'.v (128 * p + r) c := by
    intro hv' hH r c
    have hr : 128 * p + (r : Nat) < d.N_CTX := by have := r.2; omega
    have h1 := (hH (128 * p + r) hr).1 c c.2
    rw [eIdx_eq] at h1
    exact ⟨read_agree (fun _ _ => rfl) d.K (k_sub d) (elem_lt hb hr c.2) (hk (r, c, ()) rfl) h1.2.1,
      read_agree (fun _ _ => rfl) d.V (v_sub d) (elem_lt hb hr c.2) (hv (r, c, ()) rfl) h1.2.2.1⟩
  -- lines 415–429: the masked blocks
  refine Safe.bind_ro (Safe.mono (Q := fun (x : Tensor [128, d.HEAD_DIM] F32 × Tensor [128, d.HEAD_DIM] F32 × I32) m' =>
      m' = init ∧ x.2.2.toInt = ((m1 d p : Nat) : Int) ∧ KVInv d init b p (C1 d p) (T1 d) x.1 x.2.1)
      ?_ (fun x m' hx => ?_) subset_rfl)
  · cases hcz : d.CAUSAL
    · simp only [Bool.false_eq_true, ite_false]
      refine safe_ret ⟨rfl, ?_, ?_⟩
      · simp [m1, hcz, IntV.I32.toInt_ofInt]
      · simp only [C1, T1, hcz, Bool.false_eq_true, ite_false]; exact kvInv_zero d init b p
    · simp only [ite_true]
      refine Safe.bind_ro ((safe_dkdv hdot hdt hexp hmt hd (fun _ _ => rfl) hQv hDOv hMv hDv hb hp'
        (BM := 32 / 2) (Or.inl rfl) true hsn hsn (i32_nat (by norm_num)) (m0 := 128 * p) (T := 128 / (32 / 2))
        (by omega) (fun r mm _ _ => by simp [Allowed, hcz]) (C := ∅) (by simp) (by simp) (T0 := 0)
        (by norm_num) hkv (kvInv_zero d init b p) (smF d) _ _).mono (fun x m' hx => ?_) subset_rfl)
      obtain ⟨hm', hinv⟩ := hx
      subst m'
      refine safe_ret ⟨rfl, ?_, ?_⟩
      · show (sn + IntV.ofNat (128 / (32 / 2)) * IntV.ofNat (32 / 2)).toInt = _
        rw [IntV.I32.add_nat hsn (IntV.I32.mul_nat (i32_nat (by norm_num)) (i32_nat (by norm_num)) (by norm_num))
          (by norm_num; omega)]
        simp [m1, hcz]
      · have e1 : (∅ : Finset ℕ) ∪ Finset.Ico (128 * p) (128 * p + 128 / (32 / 2) * (32 / 2)) =
            Finset.Ico (128 * p) (128 * p + 128) := by simp
        rw [e1] at hinv
        simp only [C1, T1, hcz, ite_true]
        exact hinv
  obtain ⟨dk1, dv1, sm1⟩ := x
  obtain ⟨hm', hsm1, hinv1⟩ := hx
  subst m'
  simp only at hsm1 hinv1
  simp only []
  -- line 432
  obtain ⟨hm1N, hm1e⟩ := m1_add hd hp'
  have hsub1 : (IntV.ofNat d.N_CTX - sm1 : I32).toInt = ((d.N_CTX - m1 d p : Nat) : Int) :=
    IntV.I32.sub_nat (i32_nat (by omega)) hsm1 hm1N (by omega)
  refine safe_idiv1 (by rw [hsub1]; omega) (by rw [i32_nat (by norm_num)]; norm_num) fun r3 hr3 => ?_
  have hT2 : r3.item.toInt = (((d.N_CTX - m1 d p) / 32 : Nat) : Int) := by
    rw [hr3, hsub1, i32_nat (by norm_num)]; push_cast; rfl
  -- lines 433–443
  have hT2le : (d.N_CTX - m1 d p) / 32 ≤ 512 := by omega
  have hT1 : T1 d ≤ 8 := by unfold T1; split_ifs <;> omega
  refine Safe.bind_ro ((safe_dkdv hdot hdt hexp hmt hd (fun _ _ => rfl) hQv hDOv hMv hDv hb hp'
    (BM := 32) (Or.inr rfl) false hsn hsm1 hT2 (m0 := m1 d p) (T := (d.N_CTX - m1 d p) / 32)
    (by rw [hm1e]) (fun r mm h1 _ => by simp [allowed_ge (d := d) (p := p) r h1]) (C := C1 d p)
    (fun x hx => Or.inl (C1_lt x hx)) (C1_sub hp') (T0 := T1 d) (by omega) hkv hinv1 (smF d) _ _).mono
    (fun x m' hx => ?_) subset_rfl)
  obtain ⟨dk2, dv2⟩ := x
  obtain ⟨hm', hinv2⟩ := hx
  subst m'
  simp only at hinv2
  simp only []
  have hinj : ∀ ix ix' : Idx [128, d.HEAD_DIM], (fun _ => true) ix = true → (fun _ => true) ix' = true →
      tileG d b p ix = tileG d b p ix' → ix = ix' := fun ix ix' _ _ h => (tileG_inj hp' hp' ix ix' h).2.2
  have hDVt : d.DV.elems (tileSet d b p) ⊆ d.DV.elems (tSet d) := d.DV.elems_mono hsubT
  have hDKt : d.DK.elems (tileSet d b p) ⊆ d.DK.elems (tSet d) := d.DK.elems_mono hsubT
  have hDQt : d.DQ.elems (tileSet d b p) ⊆ d.DQ.elems (tSet d) := d.DQ.elems_mono hsubT
  have hout : ∀ x, x ∈ inBytes d → x ∉ d.DV.elems (tileSet d b p) ∧ x ∉ d.DK.elems (tileSet d b p) ∧
      x ∉ d.DQ.elems (tileSet d b p) := by
    intro x hx
    have h := Set.disjoint_right.mp hd.out_in hx
    unfold outBytes at h
    simp only [Set.mem_union, not_or] at h
    exact ⟨fun h' => h.2 (hDVt h'), fun h' => h.1.2 (hDKt h'), fun h' => h.1.1 (hDQt h')⟩
  -- line 446
  refine safe_cvt std32 fun dv16 hdv16 => ?_
  refine safe_store_buf d.DV (hAddr d.DV) (hin _ hd.dv_len) hinj hDVv
    (by unfold W tileSet; exact Set.subset_union_left.trans Set.subset_union_left)
    fun mA hrdA hfrA hpresA => ?_
  -- line 449
  refine safe_mul fun dk' hdk' => ?_
  refine safe_cvt std32 fun dk16 hdk16 => ?_
  -- line 451
  have hValDK : d.DK.Valid mA := Buf.valid_of_agree hDKv (fun x hx => hpresA x (hDKv.2.2.2 x hx))
  refine safe_store_buf d.DK (hAddr d.DK) (hin _ hd.dk_len) hinj hValDK (by
      intro x hx
      refine ⟨?_, fun h' => Set.disjoint_left.mp hd.dk_dv (hDKt hx) (hDVt h')⟩
      unfold W tileSet; exact Or.inl (Or.inr hx))
    fun mB hrdB hfrB hpresB => ?_
  have hmemB : ∀ x ∈ inBytes d, mB x = init x := fun x hx => by
    rw [hfrB x (hout x hx).2.1, hfrA x (hout x hx).1]
  have hvalB : ∀ {dt : DType} (Bf : Buf dt), Bf.Valid init → Bf.Valid mB := fun Bf hv =>
    Buf.valid_of_agree hv (fun x hx => hpresB x (hpresA x (hv.2.2.2 x hx)))
  -- line 456
  refine safe_idiv1 (by rw [i32_nat (by omega)]; omega) (by rw [i32_nat (by norm_num)]; norm_num)
    fun r5 hr5 => ?_
  rw [i32_nat (by omega), i32_nat (by norm_num)] at hr5
  -- lines 461–466
  refine safe_load_buf d.Q (hAddr d.Q) (hin _ hd.q_len) (hvalB d.Q hQv)
    ((d.Q.elems_mono hsubT).trans (q_sub d)) fun q hq _ => ?_
  refine safe_load_buf d.DO (hAddr d.DO) (hin _ hd.do_len) (hvalB d.DO hDOv)
    ((d.DO.elems_mono hsubT).trans (do_sub d)) fun dO hdO _ => ?_
  refine safe_load_buf d.M (g := fun (ix : Idx [128]) => b * d.N_CTX + (128 * p + (ix.1 : Nat)))
    (en := fun _ => true) (Buf.addrs_of_get_all d.M fun ⟨i, ⟨⟩⟩ =>
      congrArg (fun (x : Tensor [128] (Ptr .f32)) => x.get (i, ())) (vec_r d.M (b * d.N_CTX) hrn))
    (by
      rintro ⟨i, ⟨⟩⟩ _
      have := row_lt hb (show 128 * p + (i : Nat) < d.N_CTX by have := i.2; omega)
      have := hd.m_len
      simp only; omega) (hvalB d.M hMv)
    (by
      refine (d.M.elems_mono ?_).trans (m_sub d)
      rintro _ ⟨⟨i, ⟨⟩⟩, -, rfl⟩
      exact row_lt hb (show 128 * p + (i : Nat) < d.N_CTX by have := i.2; omega))
    fun mv hmv _ => ?_
  have hqv : ∀ hv' : HeadV, hv'.Holds d init b → ∀ (r : Fin 128) (c : Fin d.HEAD_DIM),
      q.get2 r c = hv'.q (128 * p + r) c ∧ dO.get2 r c = hv'.dO (128 * p + r) c := by
    intro hv' hH r c
    have hr : 128 * p + (r : Nat) < d.N_CTX := by have := r.2; omega
    have h1 := (hH (128 * p + r) hr).1 c c.2
    rw [eIdx_eq] at h1
    exact ⟨read_agree hmemB d.Q (q_sub d) (elem_lt hb hr c.2) (hq (r, c, ()) rfl) h1.1,
      read_agree hmemB d.DO (do_sub d) (elem_lt hb hr c.2) (hdO (r, c, ()) rfl) h1.2.2.2⟩
  have hmv' : ∀ hv' : HeadV, hv'.Holds d init b → ∀ r : Fin 128,
      mv.unsqueeze1.get2 r 0 = hv'.lse (128 * p + r) := by
    intro hv' hH r
    have hr : 128 * p + (r : Nat) < d.N_CTX := by have := r.2; omega
    have h1 := (hH (128 * p + r) hr).2.1
    rw [rIdx_eq] at h1
    exact read_agree hmemB d.M (m_sub d) (row_lt hb hr) (hmv (r, ()) rfl) h1
  have hKvB := hvalB d.K hKv
  have hVvB := hvalB d.V hVv
  have hDvB := hvalB d.D hDv
  -- lines 468–487: the masked blocks of `dq`
  refine Safe.bind_ro (Safe.mono (Q := fun (x : Tensor [128, d.HEAD_DIM] F32 × I32 × I32) m'' =>
      m'' = mB ∧ x.2.1.toInt = ((0 : Nat) : Int) ∧ x.2.2.toInt = ((Tq2 d p : Nat) : Int) ∧
        QInv d init b p (Cq1 d p) (if d.CAUSAL then 8 else 0) x.1)
      ?_ (fun x m'' hx => ?_) subset_rfl)
  · cases hcz : d.CAUSAL
    · simp only [Bool.false_eq_true, ite_false]
      refine safe_ret ⟨rfl, by simp [IntV.I32.toInt_ofInt], ?_, ?_⟩
      · show r5.item.toInt = _
        rw [hr5]; simp only [Tq2, hcz, Bool.false_eq_true, ite_false]; push_cast; rfl
      · simp only [Cq1, hcz, Bool.false_eq_true, ite_false]; exact qInv_zero d init b p
    · simp only [ite_true]
      have hen : (sn + IntV.ofNat 128 - IntV.ofNat (128 / (32 / 2)) * IntV.ofNat (32 / 2) : I32).toInt =
          ((128 * p : Nat) : Int) := by
        rw [IntV.I32.sub_nat (IntV.I32.add_nat hsn (i32_nat (by norm_num)) (by omega))
          (IntV.I32.mul_nat (i32_nat (by norm_num)) (i32_nat (by norm_num)) (by norm_num)) (by norm_num)
          (by omega)]
        norm_num
      refine Safe.bind_ro ((safe_dq hdot hdt hexp hmt hd hmemB hKvB hVvB hDvB hb hp'
        (BN := 32 / 2) (Or.inl rfl) true hsn hen (i32_nat (by norm_num)) (n0 := 128 * p) (T := 128 / (32 / 2))
        (by omega) (fun r nn _ _ => by simp [Allowed, hcz]) (C := ∅) (by simp) (by simp) (T0 := 0)
        (by norm_num) hqv hmv' (qInv_zero d init b p) _ _).mono (fun x m'' hx => ?_) subset_rfl)
      obtain ⟨hm'', hinv⟩ := hx
      subst m''
      refine safe_idiv1 (by rw [hen]; omega) (by rw [i32_nat (by norm_num)]; norm_num) fun r6 hr6 => ?_
      rw [hen, i32_nat (by norm_num)] at hr6
      have hr6' : r6.item.toInt = ((128 * p / 32 : Nat) : Int) := by rw [hr6]; push_cast; rfl
      refine safe_ret ⟨rfl, ?_, ?_, ?_⟩
      · show (sn + IntV.ofNat 128 - IntV.ofNat (128 / (32 / 2)) * IntV.ofNat (32 / 2) - r6.item * IntV.ofNat 32).toInt = _
        rw [IntV.I32.sub_nat hen (IntV.I32.mul_nat hr6' (i32_nat (by norm_num)) (by omega)) (by omega) (by omega)]
        have e : 128 * p - 128 * p / 32 * 32 = 0 := by omega
        rw [e]
      · show r6.item.toInt = _
        rw [hr6']; simp [Tq2, hcz]
      · have e1 : (∅ : Finset ℕ) ∪ Finset.Ico (128 * p) (128 * p + 128 / (32 / 2) * (32 / 2)) =
            Finset.Ico (128 * p) (128 * p + 128) := by simp
        rw [e1] at hinv
        simp only [Cq1, hcz, ite_true]
        exact hinv
  obtain ⟨dq1, sq1, nq1⟩ := x
  obtain ⟨hm'', hsq1, hnq1, hinvq1⟩ := hx
  subst m''
  simp only at hsq1 hnq1 hinvq1
  simp only []
  -- lines 489–496
  have hTq2 : Tq2 d p ≤ 512 := by unfold Tq2; split_ifs <;> omega
  have hTq1 : (if d.CAUSAL then 8 else 0) ≤ 8 := by split_ifs <;> omega
  refine Safe.bind_ro ((safe_dq hdot hdt hexp hmt hd hmemB hKvB hVvB hDvB hb hp'
    (BN := 32) (Or.inr rfl) false hsn hsq1 hnq1 (n0 := 0) (T := Tq2 d p)
    (by rw [zero_add]; exact q_rows hd hp') (fun r nn _ h2 => by simp [allowed_lt (d := d) (p := p) r h2])
    (C := Cq1 d p) (fun x hx => Cq1_apart x hx) (Cq1_sub hp') (T0 := if d.CAUSAL then 8 else 0)
    (by omega) hqv hmv' hinvq1 _ _).mono (fun x m'' hx => ?_) subset_rfl)
  obtain ⟨hm'', hinvq2⟩ := hx
  subst m''
  -- line 499
  refine safe_mul fun dq' hdq' => ?_
  refine safe_cvt std32 fun dq16 hdq16 => ?_
  -- line 500
  have hDQvB := hvalB d.DQ hDQv
  refine safe_last (safe_store_buf d.DQ (hAddr d.DQ) (hin _ hd.dq_len) hinj hDQvB (by
      intro x hx
      refine ⟨⟨?_, fun h' => Set.disjoint_left.mp hd.dq_dv (hDQt hx) (hDVt h')⟩,
        fun h' => Set.disjoint_left.mp hd.dq_dk (hDQt hx) (hDKt h')⟩
      unfold W tileSet; exact Or.inr hx)
    fun mC hrdC hfrC _ => safe_ret ?_)
  have hti : ∀ i : Idx [128, d.HEAD_DIM], tileG d b p i ∈ tileSet d b p := fun i => ⟨i, rfl, rfl⟩
  refine qp_final hd hp' hinv2 hinvq2 hdv16 hdk' hdk16 hdq' hdq16 (fun i => ?_) (fun i => ?_)
    (fun i => hrdC i rfl)
  · rw [← hrdA i rfl]
    refine Mem.read_congr fun x hx => ?_
    have hx' := hDVt (d.DV.span_sub_elems (hti i) hx)
    rw [hfrC x (fun h' => Set.disjoint_left.mp hd.dq_dv (hDQt h') hx'),
      hfrB x (fun h' => Set.disjoint_left.mp hd.dk_dv (hDKt h') hx')]
  · rw [← hrdB i rfl]
    refine Mem.read_congr fun x hx => ?_
    have hx' := hDKt (d.DK.span_sub_elems (hti i) hx)
    exact hfrC x (fun h' => Set.disjoint_left.mp hd.dq_dk (hDQt h') hx')

end Kernels.AttnBwd
