import Proof.Dkdv
import Proof.Dq
import Proof.Scalars
import Proof.Stages
import Proof.Outputs

/-!
# `attn_bwd_f16`: one program
-/

namespace Claims.AttnBwd

open TVA

/-- The elements of a 4-D tensor's storage that program `i` owns: its `128` rows of its head. -/
def tileSet (d : Inst) (i : Nat) : Set Nat := tileEl d (head d i) (128 * pidX d i)

/-- A memory that keeps every present byte keeps `Pre`. -/
theorem Pre.of_pres {d : Inst} {m0 m : Mem} (h : Pre d m0) (hp : ∀ x, m0 x ≠ none → m x ≠ none) :
    Pre d m := by
  obtain ⟨a1, a2, a3, a4, a5, a6, a7, a8, a9⟩ := h
  exact ⟨Buf.valid_of_agree a1 fun x hx => hp x (a1.2.2.2 x hx),
    Buf.valid_of_agree a2 fun x hx => hp x (a2.2.2.2 x hx),
    Buf.valid_of_agree a3 fun x hx => hp x (a3.2.2.2 x hx),
    Buf.valid_of_agree a4 fun x hx => hp x (a4.2.2.2 x hx),
    Buf.valid_of_agree a5 fun x hx => hp x (a5.2.2.2 x hx),
    Buf.valid_of_agree a6 fun x hx => hp x (a6.2.2.2 x hx),
    Buf.valid_of_agree a7 fun x hx => hp x (a7.2.2.2 x hx),
    Buf.valid_of_agree a8 fun x hx => hp x (a8.2.2.2 x hx),
    Buf.valid_of_agree a9 fun x hx => hp x (a9.2.2.2 x hx)⟩

/-- A read of an element of a tile is unchanged by a store whose bytes are disjoint from the
tile. -/
theorem read_keep {B : Buf .f16} {S : Set Nat} {X : Set Int} {m m' : Mem}
    (hdis : Disjoint (B.elems S) X) (hfr : ∀ x, x ∉ X → m' x = m x) {j : Nat} (hj : j ∈ S) :
    m'.read .f16 (B.ptr (j : Int)).addr = m.read .f16 (B.ptr (j : Int)).addr :=
  Mem.read_congr fun x hx => hfr x fun hxX => Set.disjoint_left.mp hdis (B.span_sub_elems hj hx) hxX

/-- The tiles of the three outputs of one program are pairwise disjoint, and disjoint from the
inputs. -/
theorem tile_disjoint {d : Inst} (hd : d.ok) {b n0 : Nat} (hb : b < G d) (hn0 : n0 + 128 ≤ d.N_CTX) :
    Disjoint (d.DK.elems (tileEl d b n0)) (d.DV.elems (tileEl d b n0)) ∧
    Disjoint (d.DQ.elems (tileEl d b n0)) (d.DK.elems (tileEl d b n0)) ∧
    Disjoint (d.DQ.elems (tileEl d b n0)) (d.DV.elems (tileEl d b n0)) ∧
    (∀ x ∈ inBytes d, x ∉ d.DV.elems (tileEl d b n0) ∧ x ∉ d.DK.elems (tileEl d b n0) ∧
      x ∉ d.DQ.elems (tileEl d b n0)) := by
  have hs := tileEl_sub d hb hn0
  refine ⟨hd.dk_dv.mono (d.DK.elems_mono hs) (d.DV.elems_mono hs),
    hd.dq_dk.mono (d.DQ.elems_mono hs) (d.DK.elems_mono hs),
    hd.dq_dv.mono (d.DQ.elems_mono hs) (d.DV.elems_mono hs), fun x hx => ?_⟩
  have h1 : x ∉ d.DV.elems (tSet d) := fun h =>
    Set.disjoint_left.mp hd.out_in (Or.inr h) hx
  have h2 : x ∉ d.DK.elems (tSet d) := fun h =>
    Set.disjoint_left.mp hd.out_in (Or.inl (Or.inr h)) hx
  have h3 : x ∉ d.DQ.elems (tSet d) := fun h =>
    Set.disjoint_left.mp hd.out_in (Or.inl (Or.inl h)) hx
  exact ⟨fun h => h1 (d.DV.elems_mono hs h), fun h => h2 (d.DK.elems_mono hs h),
    fun h => h3 (d.DQ.elems_mono hs h)⟩

/-- The bytes program `i` writes: its rows of `dq`, `dk` and `dv`. -/
def Wp (d : Inst) (i : Nat) : Set Int :=
  d.DQ.elems (tileSet d i) ∪ d.DK.elems (tileSet d i) ∪ d.DV.elems (tileSet d i)

/-- What program `i` leaves: `OutSpec` at each element of its rows of the three outputs. -/
def Qp (d : Inst) (init : Mem) (i : Nat) (m : Mem) : Prop :=
  ∀ r j, r < 128 → j < d.HEAD_DIM →
    OutSpec d init m d.DV (head d i) (128 * pidX d i + r) j (fun hv => dVE d hv (128 * pidX d i + r) j)
        (fun hv => dVTol d hv (128 * pidX d i + r) j) ∧
      OutSpec d init m d.DK (head d i) (128 * pidX d i + r) j
        (fun hv => dKE d hv (128 * pidX d i + r) j) (fun hv => dKTol d hv (128 * pidX d i + r) j) ∧
      OutSpec d init m d.DQ (head d i) (128 * pidX d i + r) j
        (fun hv => dQE d hv (128 * pidX d i + r) j) (fun hv => dQTol d hv (128 * pidX d i + r) j)

/-- **One program is safe** with the footprints `inBytes d` (reads), `Wp d i` (writes) and the
post-condition `Qp d init i`: its rows of `dV`, `dK` and `dQ` are as `OutSpec` says. -/
theorem safe_program {E : Env} (hP : Device E.P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    Safe E (inBytes d) (Wp d i) (fun _ m => Qp d init i m) init ((model d).body ((launch d).ctx i)) := by
  have hsz : (i : Nat) < gx d * 1 * G d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hgx : 0 < gx d := by
    rcases Nat.eq_zero_or_pos (gx d) with h | h
    · rw [h] at hsz; simp at hsz
    · exact h
  have hG : 0 < G d := by
    rcases Nat.eq_zero_or_pos (G d) with h | h
    · rw [h] at hsz; simp at hsz
    · exact h
  have hH : 0 < d.H := by
    rcases Nat.eq_zero_or_pos d.H with h | h
    · simp [G, h] at hG
    · exact h
  have hb : head d i < G d := head_lt hsz
  have hpx : pidX d i < gx d := pidX_lt hsz
  have hgx128 := mul_gx hd
  have hN := hd.n_le
  have hHD := hd.hd
  have hpid2 : (tl.program_id ((launch d).ctx i) 2).toInt = (head d i : Nat) := by
    rw [tl.program_id, launch_pid2]
    exact IntV.I32.toInt_ofNat (by have := hd.g_lt; omega)
  have hpid0 : (tl.program_id ((launch d).ctx i) 0).toInt = (pidX d i : Nat) := by
    rw [tl.program_id, launch_pid0]
    exact IntV.I32.toInt_ofNat (by omega)
  have hHpos : (IntV.ofNat d.H : I32).toInt = d.H := IntV.I32.toInt_ofNat (by
    have := hd.size_lt.2; have : d.H ≤ d.H * d.N_CTX * d.HEAD_DIM := by
      calc d.H = d.H * 1 * 1 := by ring
        _ ≤ d.H * d.N_CTX * d.HEAD_DIM := Nat.mul_le_mul (Nat.mul_le_mul_left _ (by omega)) (by omega)
    omega)
  simp only [model, attn_bwd]
  refine safe_irem1 (by rw [hpid2]; positivity) (by rw [hHpos]; exact_mod_cast hH) fun l0 hl0 => ?_
  refine safe_idiv1 (by rw [hpid2]; positivity) (by rw [hHpos]; exact_mod_cast hH) fun l1 hl1 => ?_
  rw [hpid2, hHpos] at hl0 hl1
  have hadj := adj_val hd hb hH hl0 hl1
  have hoff := off_val hd hb hpid2
  generalize (IntV.cast (IntV.ofNat (d.N_CTX * d.HEAD_DIM) * l0.item +
    IntV.ofNat (d.H * d.N_CTX * d.HEAD_DIM) * l1.item) : I64) = adj at hadj ⊢
  generalize (IntV.cast (tl.program_id ((launch d).ctx i) 2 * IntV.ofNat d.N_CTX) : I64) = off at hoff ⊢
  generalize tl.program_id ((launch d).ctx i) 0 = pid0 at hpid0 ⊢
  have hn0le : 128 * pidX d i + 128 ≤ d.N_CTX := by nlinarith
  have hKp := ptr_add_zero d.K adj hadj
  have hVp := ptr_add_zero d.V adj hadj
  have hQp := ptr_add_zero d.Q adj hadj
  have hDOp := ptr_add_zero d.DO adj hadj
  have hMp := ptr_add_zero d.M off hoff
  have hDp := ptr_add_zero d.D off hoff
  have hn0v : (pid0 * IntV.ofNat 128 : I32).toInt = ((128 * pidX d i : Nat) : Int) := by
    have := IntV.I32.mul_nat hpid0 (IntV.I32.toInt_ofNat (n := 128) (by norm_num)) (by omega)
    rw [this]; congr 1; ring
  have hHDle : d.HEAD_DIM ≤ 256 := by omega
  have hs1 : (IntV.ofNat d.HEAD_DIM : I32).toInt = d.HEAD_DIM := IntV.I32.toInt_ofNat (by omega)
  have hs2 : (IntV.ofNat 1 : I32).toInt = 1 := IntV.I32.toInt_ofNat (by omega)
  have hfit1 : (128 * pidX d i + 128 - 1) * d.HEAD_DIM < 2 ^ 31 :=
    lt_of_le_of_lt (Nat.mul_le_mul (by omega : 128 * pidX d i + 128 - 1 ≤ 2 ^ 14) hHDle) (by norm_num)
  have hfit2 : (d.HEAD_DIM - 1) * 1 < 2 ^ 31 := by omega
  refine safe_load_rc d d.K hd.k_len (K_sub_in d) hpre.2.1 hb
    (r := fun x : Fin 128 => 128 * pidX d i + x) (c := fun j : Fin d.HEAD_DIM => j.val)
    (fun x => by have := x.isLt; omega) (fun j => j.isLt) ?_ fun k hk => ?_
  · intro x j
    rw [ptrN_lane d.K hKp hn0v hs1 hs2 (by omega) hfit1 (by omega) hfit2 x j]
    congr 2; unfold eIdx; ring
  refine safe_load_rc d d.V hd.v_len (V_sub_in d) hpre.2.2.1 hb
    (r := fun x : Fin 128 => 128 * pidX d i + x) (c := fun j : Fin d.HEAD_DIM => j.val)
    (fun x => by have := x.isLt; omega) (fun j => j.isLt) ?_ fun v hv => ?_
  · intro x j
    rw [ptrN_lane d.V hVp hn0v hs1 hs2 (by omega) hfit1 (by omega) hfit2 x j]
    congr 2; unfold eIdx; ring
  refine Safe.bind_ro (Safe.mono (Q := Post1 d init (head d i) (128 * pidX d i)) ?_ ?_
    (Set.Subset.refl _))
  · by_cases hC : d.CAUSAL = true
    · simp only [hC, ite_true]
      have h8 : (IntV.ofNat (128 / (32 / 2)) : I32).toInt = 8 := by simp
      refine Safe.bind_ro ((dkdv_safe hP d hd hpre (fun x _ => rfl) hb (32 / 2)
        (Or.inl (by norm_num)) true (fun _ => hC) (128 * pidX d i) (128 * pidX d i) 8 0 ∅ hn0le
        (by norm_num; omega) (by norm_num) (by simp) (by simp) hQp hDOp hMp hDp hs1 hs2 hn0v hn0v
        (by rw [h8]; rfl) hk hv (DkdvInv.zero d init _ _)).mono ?_ le_rfl)
      rintro ⟨dk1, dv1⟩ m1 ⟨hm1, hinv1⟩
      refine Safe.ret ⟨hm1, ?_, ?_⟩
      · have h16 : (IntV.ofNat (32 / 2) : I32).toInt = 16 := by simp
        have hm := IntV.I32.mul_nat h8 h16 (by norm_num)
        have := IntV.I32.add_nat hn0v hm (by norm_num; omega)
        show (_ + _ : I32).toInt = _
        rw [this]; simp [stage1, hC]
      · simpa [stage1, hC] using hinv1
    · simp only [hC]
      refine Safe.ret ⟨rfl, ?_, ?_⟩
      · show (IntV.ofInt 0 : I32).toInt = _
        rw [IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩]; simp [stage1, hC]
      · simpa [stage1, hC] using DkdvInv.zero d init (head d i) (128 * pidX d i)
  · intro a m1 hpost
    obtain ⟨hm1, hs, hinv⟩ := hpost
    have hm1' := hm1.symm
    subst hm1'
    have hS1 := stage1_lt (d := d) hn0le
    have hs1le := stage1_le (d := d) hn0le
    have hAl1 := stage1_al d (128 * pidX d i)
    have ht1 := stage1_t d (128 * pidX d i)
    have hfin := stage_final hd (n0 := 128 * pidX d i) (by omega) hn0le
    generalize stage1 d (128 * pidX d i) = st at hS1 hs1le hAl1 ht1 hfin hs hinv
    obtain ⟨S1, t1, s1⟩ := st
    simp only [] at hS1 hs1le hAl1 ht1 hfin hs hinv
    have hNv : (IntV.ofNat d.N_CTX : I32).toInt = d.N_CTX := IntV.I32.toInt_ofNat (by omega)
    have hsub : (IntV.ofNat d.N_CTX - a.2.2 : I32).toInt = ((d.N_CTX - s1 : Nat) : Int) :=
      IntV.I32.sub_nat hNv hs hs1le (by omega)
    have h32 : (IntV.ofNat 32 : I32).toInt = 32 := IntV.I32.toInt_ofNat (by norm_num)
    refine safe_idiv1 (by rw [hsub]; positivity) (by rw [h32]; norm_num) fun l2 hl2 => ?_
    rw [hsub, h32] at hl2
    have hT2 : l2.item.toInt = (((d.N_CTX - s1) / 32 : Nat) : Int) := by
      rw [hl2]; push_cast; rfl
    refine Safe.bind_ro ((dkdv_safe hP d hd hpre (fun x _ => rfl) hb 32 (Or.inr rfl) false
      (fun h => absurd h (by simp)) (128 * pidX d i) s1 ((d.N_CTX - s1) / 32) t1 S1 hn0le
      (by have := Nat.div_mul_le_self (d.N_CTX - s1) 32; omega) (by omega)
      (fun s hs' => ⟨(hS1 s hs').1, Or.inl (hS1 s hs').2⟩) (fun _ => hAl1) hQp hDOp hMp hDp hs1 hs2
      hn0v hs hT2 hk hv hinv).mono ?_ le_rfl)
    intro r m1 hpost
    obtain ⟨hm1, hinvF⟩ := hpost
    have hm1' := hm1.symm
    subst hm1'
    rw [hfin.1, hfin.2] at hinvF
    have hDVp := ptr_add_zero d.DV adj hadj
    have hDKp := ptr_add_zero d.DK adj hadj
    have hDQp := ptr_add_zero d.DQ adj hadj
    obtain ⟨hdkdv, hdqdk, hdqdv, hin⟩ := tile_disjoint hd hb hn0le
    refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩
      fun dvc hdvc => ?_
    refine safe_store_rc d d.DV hd.dv_len hpre.2.2.2.2.2.2.1 hb (128 * pidX d i) hn0le ?_
      (fun x hx => Or.inr hx) fun m2 hrd2 hfr2 hpres2 => ?_
    · intro x j
      rw [ptrN_lane d.DV hDVp hn0v hs1 hs2 (by omega) hfit1 (by omega) hfit2 x j]
      congr 2; unfold eIdx; ring
    refine safe_mul fun dkm hdkm => ?_
    refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩
      fun dkc hdkc => ?_
    refine safe_store_rc d d.DK hd.dk_len (Buf.valid_of_agree hpre.2.2.2.2.2.1 fun x hx =>
        hpres2 x (hpre.2.2.2.2.2.1.2.2.2 x hx)) hb (128 * pidX d i) hn0le ?_
      (fun x hx => Set.mem_sdiff_of_mem (Or.inl (Or.inr hx)) (fun h => Set.disjoint_left.mp hdkdv hx h))
      fun m3 hrd3 hfr3 hpres3 => ?_
    · intro x j
      rw [ptrN_lane d.DK hDKp hn0v hs1 hs2 (by omega) hfit1 (by omega) hfit2 x j]
      congr 2; unfold eIdx; ring
    -- the memory `m3` after the two stores
    have hag3 : ∀ x ∈ inBytes d, m3 x = init x := by
      intro x hx
      obtain ⟨h1, h2, -⟩ := hin x hx
      rw [hfr3 x h2, hfr2 x h1]
    have hpres : ∀ x, init x ≠ none → m3 x ≠ none := fun x hx => hpres3 x (hpres2 x hx)
    have hpre3 : Pre d m3 := Pre.of_pres hpre hpres
    have h128 : (IntV.ofNat 32 : I32).toInt = 32 := IntV.I32.toInt_ofNat (by norm_num)
    refine safe_idiv1 (x := IntV.ofNat d.N_CTX) (y := IntV.ofNat 32)
      (by rw [IntV.I32.toInt_ofNat (by omega)]; positivity) (by rw [h128]; norm_num) fun l3 hl3 => ?_
    refine safe_load_rc d d.Q hd.q_len (Q_sub_in d) hpre3.1 hb
      (r := fun x : Fin 128 => 128 * pidX d i + x) (c := fun j : Fin d.HEAD_DIM => j.val)
      (fun x => by have := x.isLt; omega) (fun j => j.isLt) ?_ fun q hq => ?_
    · intro x j
      rw [ptrN_lane d.Q hQp hn0v hs1 hs2 (by omega) hfit1 (by omega) hfit2 x j]
      congr 2; unfold eIdx; ring
    refine safe_load_rc d d.DO hd.do_len (DO_sub_in d) hpre3.2.2.2.1 hb
      (r := fun x : Fin 128 => 128 * pidX d i + x) (c := fun j : Fin d.HEAD_DIM => j.val)
      (fun x => by have := x.isLt; omega) (fun j => j.isLt) ?_ fun dO hdO => ?_
    · intro x j
      rw [ptrN_lane d.DO hDOp hn0v hs1 hs2 (by omega) hfit1 (by omega) hfit2 x j]
      congr 2; unfold eIdx; ring
    refine safe_load_vec d d.M hd.m_len (M_sub_in d) hpre3.2.2.2.2.2.2.2.1 hb
      (r := fun x : Fin 128 => 128 * pidX d i + x) (fun x => by have := x.isLt; omega) ?_
      fun mv hmv => ?_
    · intro x
      have hl := IntV.I32.toInt_sadd_arange (n := 128) hn0v (by omega) x
      show (d.M.ptr 0 + off).offset ((pid0 * IntV.ofNat 128 + tl.arange 0 128).get1 x).toInt = _
      rw [hl, hMp, Buf.ptr_offset]; congr 1
    have hmm : ∀ x : Fin 128, m3.read .f32 (d.M.ptr ((rIdx d (head d i) (128 * pidX d i + x) : Nat) : Int)).addr =
        some (mv.unsqueeze1.get2 x 0) := hmv
    have h8 : (IntV.ofNat (128 / (32 / 2)) : I32).toInt = 8 := by simp
    have h16 : (IntV.ofNat (32 / 2) : I32).toInt = 16 := by simp
    have hm128 := IntV.I32.mul_nat h8 h16 (by norm_num)
    have hadd128 := IntV.I32.add_nat hn0v (IntV.I32.toInt_ofNat (n := 128) (by norm_num))
      (by omega)
    have hendn : (pid0 * IntV.ofNat 128 + IntV.ofNat 128 - IntV.ofNat (128 / (32 / 2)) *
        IntV.ofNat (32 / 2) : I32).toInt = ((128 * pidX d i : Nat) : Int) := by
      rw [IntV.I32.sub_nat hadd128 hm128 (by norm_num) (by omega)]
      have : 128 * pidX d i + 128 - 8 * 16 = 128 * pidX d i := by omega
      rw [this]
    refine Safe.bind_ro (Safe.mono (Q := Post2 d init m3 (head d i) (128 * pidX d i)) ?_ ?_
      (Set.Subset.refl _))
    · by_cases hC : d.CAUSAL = true
      · simp only [hC, ite_true]
        refine Safe.bind_ro ((dq_safe hP d hd hpre3 hag3 hb (32 / 2) (Or.inl (by norm_num)) true
          (fun _ => hC) (128 * pidX d i) (128 * pidX d i) 8 0 ∅ hn0le (by norm_num; omega)
          (by norm_num) (by simp) (fun h => absurd h (by simp)) hKp hVp hDp hs1 hs2 hn0v hendn h8
          hq hdO hmm (DqInv.zero d init _ _)).mono ?_ le_rfl)
        intro dq1 m4 ⟨hm4, hinv4⟩
        refine safe_idiv1 (x := pid0 * IntV.ofNat 128 + IntV.ofNat 128 -
          IntV.ofNat (128 / (32 / 2)) * IntV.ofNat (32 / 2)) (y := IntV.ofNat 32)
          (by rw [hendn]; positivity) (by rw [h128]; norm_num) fun l4 hl4 => ?_
        rw [hendn, h128] at hl4
        have hl4' : l4.item.toInt = ((128 * pidX d i / 32 : Nat) : Int) := by
          rw [hl4]; push_cast; rfl
        refine Safe.ret ⟨hm4, ?_, ?_, ?_⟩
        · show (_ - _ : I32).toInt = _
          have hq32 : (l4.item * IntV.ofNat 32 : I32).toInt = ((128 * pidX d i / 32 * 32 : Nat) : Int) :=
            IntV.I32.mul_nat hl4' (IntV.I32.toInt_ofNat (by norm_num)) (by omega)
          rw [IntV.I32.sub_nat hendn hq32 (Nat.div_mul_le_self _ _) (by omega)]
          simp only [stage2, hC, ite_true]
          congr 1; omega
        · simpa [stage2, hC] using hl4'
        · simpa [stage2, hC] using hinv4
      · simp only [hC]
        refine Safe.ret ⟨rfl, ?_, ?_, ?_⟩
        · show (IntV.ofInt 0 : I32).toInt = _
          rw [IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩]; simp [stage2, hC]
        · rw [hl3, IntV.I32.toInt_ofNat (by omega), h128]; simp [stage2, hC]
        · simpa [stage2, hC] using DqInv.zero d init (head d i) (128 * pidX d i)
    · intro a m4 hpost
      obtain ⟨hm4, hs4, hn4, hinv4⟩ := hpost
      have hm4' := hm4.symm
      subst hm4'
      have hS2 := stage2_lt (d := d) (q0 := 128 * pidX d i) hn0le
      have hAl2 := stage2_al d (q0 := 128 * pidX d i) (by omega)
      have ht2 := stage2_t d (128 * pidX d i)
      have hst2 := stage2_steps hd (128 * pidX d i) hn0le
      have hdis2 := stage2_disj d (q0 := 128 * pidX d i) (by omega)
      have hfin2 := stage2_final hd (q0 := 128 * pidX d i) (by omega)
      generalize stage2 d (128 * pidX d i) = st2 at hS2 hAl2 ht2 hst2 hdis2 hfin2 hs4 hn4 hinv4
      obtain ⟨S2, t2, n2, T2⟩ := st2
      simp only [] at hS2 hAl2 ht2 hst2 hdis2 hfin2 hs4 hn4 hinv4
      refine Safe.bind_ro ((dq_safe hP d hd hpre3 hag3 hb 32 (Or.inr rfl) false
        (fun h => absurd h (by simp)) (128 * pidX d i) n2 T2 t2 S2 hn0le hst2 (by omega)
        (fun s hs' => ⟨hS2 s hs', Or.inr (hdis2 s hs')⟩) (fun _ => hAl2) hKp hVp hDp hs1 hs2 hn0v hs4
        hn4 hq hdO hmm hinv4).mono ?_ le_rfl)
      intro rq m5 hpost5
      obtain ⟨hm5, hinvQ⟩ := hpost5
      have hm5' := hm5.symm
      subst hm5'
      rw [hfin2.1, hfin2.2] at hinvQ
      refine safe_mul fun dqm hdqm => ?_
      refine safe_cvt (g := .binary16) ⟨Or.inr (Or.inr (Or.inl rfl)), Or.inl rfl, Or.inl rfl⟩
        fun dqc hdqc => ?_
      refine safe_last (safe_store_rc d d.DQ hd.dq_len (Buf.valid_of_agree hpre.2.2.2.2.1 fun x hx =>
          hpres x (hpre.2.2.2.2.1.2.2.2 x hx)) hb (128 * pidX d i) hn0le ?_
        (fun x hx => Set.mem_sdiff_of_mem (Set.mem_sdiff_of_mem (Or.inl (Or.inl hx))
          (fun h => Set.disjoint_left.mp hdqdv hx h)) (fun h => Set.disjoint_left.mp hdqdk hx h))
        fun m6 hrd6 hfr6 hpres6 => ?_)
      · intro x j
        rw [ptrN_lane d.DQ hDQp hn0v hs1 hs2 (by omega) hfit1 (by omega) hfit2 x j]
        congr 2; unfold eIdx; ring
      refine Safe.ret ?_
      intro rr j hrr hj
      have hmem : eIdx d (head d i) (128 * pidX d i + rr) j ∈ tileEl d (head d i) (128 * pidX d i) :=
        ⟨rr, j, hrr, hj, rfl⟩
      refine ⟨⟨dvc.get2 ⟨rr, hrr⟩ ⟨j, hj⟩, ?_, fun hv hH hOk => ?_⟩,
        ⟨dkc.get2 ⟨rr, hrr⟩ ⟨j, hj⟩, ?_, fun hv hH hOk => ?_⟩,
        ⟨dqc.get2 ⟨rr, hrr⟩ ⟨j, hj⟩, ?_, fun hv hH hOk => ?_⟩⟩
      · rw [read_keep hdqdv.symm hfr6 hmem, read_keep hdkdv.symm hfr3 hmem]
        exact hrd2 ⟨rr, hrr⟩ ⟨j, hj⟩
      · exact dv_spec d (by omega) hinvF hdvc ⟨rr, hrr⟩ ⟨j, hj⟩ hv hH hOk
      · rw [read_keep hdqdk.symm hfr6 hmem]
        exact hrd3 ⟨rr, hrr⟩ ⟨j, hj⟩
      · exact dk_spec d hd hn0le (by omega) hinvF hdkm hdkc ⟨rr, hrr⟩ ⟨j, hj⟩ hv hH hOk
      · exact hrd6 ⟨rr, hrr⟩ ⟨j, hj⟩
      · exact dq_spec d hd hn0le (by omega) hinvQ hdqm hdqc ⟨rr, hrr⟩ ⟨j, hj⟩ hv hH hOk

end Claims.AttnBwd
