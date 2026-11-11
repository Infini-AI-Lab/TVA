import Proof.Defs
import Proof.Pid
import Proof.Wrap
import Proof.Grp
import Proof.I32
import Proof.Facts
import Proof.Loop
import Proof.Steps
import Proof.Lanes
import Proof.Epilogue
import Proof.Body
import Proof.Numeric

/-!
# `lora_shrink_bf16`: one program is `SafeA`

The walk of the kernel body for the program `i`: the program ids and the grouped order (lines
350-361), the adapter group and its early exits (366-376), the token indices of the tile (379-388),
the weights' address (226-233), the pointer tiles (236-252) and the loop (131-180, `LoopInv`), and
the end (`safe_epilogue`).
-/

namespace Claims.LoraShrink

open TVA

set_option maxHeartbeats 4000000 in
/-- **One program of a covered launch**, run alone, reads only `Rall`, writes nothing, updates
only the shared output cells with atomics, and leaves `Q`. -/
theorem safe_program {P : Platform} (hP : Device P) {d : Inst} (hd : d.ok) {init : Mem}
    (hpre : Pre d init) (i : Fin (launch d).size) :
    SafeA ((launch d).env P) (Rall d) ∅ (Cells d)
      (fun (_ : Unit) (log : List AtomicStep) (m : Mem) => Q d init i log m) [] init
      ((model d).body ((launch d).ctx i)) := by
  have hpre0 := hpre
  have hi : (i : Nat) < (launch d).size := i.isLt
  obtain ⟨hx0, hs, hg⟩ := decode_lt d hd i.isLt
  obtain ⟨hp0, hp1, hp2⟩ := ctx_pid d i
  have hR := hd.R_pos
  have hRm := hd.R_blk
  have hRK := hd.RK_lt
  have hK := hd.K_pos
  have hRlt : d.R < 2 ^ 31 := by nlinarith
  have hMlt : d.M ≤ 2 ^ 22 := hd.M_le
  have hcn_pos : 0 < cn d := by unfold cn; omega
  have hcn_lt : cn d < 2 ^ 31 := by unfold cn; omega
  have hcm_lt : cm d < 2 ^ 31 := by unfold cm; omega
  have hxlt : ax0 d i < 2 ^ 31 := by have := hd.grid0_lt; omega
  have hskpos : 0 < sk d := by unfold sk; split_ifs <;> norm_num
  have hsk64 : sk d ≤ 64 := by unfold sk; split_ifs <;> norm_num
  have hxg : ax0 d i < sk d * (cm d * cn d) := by
    have : grid0 d = sk d * (cm d * cn d) := by unfold grid0 cm cn; ring
    omega
  have hpmn : pidMN d (ax0 d i) < cm d * cn d := by
    unfold pidMN; exact Nat.div_lt_of_lt_mul (by linarith)
  have hpmn31 : pidMN d (ax0 d i) < 2 ^ 31 := lt_of_le_of_lt (Nat.div_le_self _ _) hxlt
  have hmod31 : pidMN d (ax0 d i) % (8 * cn d) < 2 ^ 31 := lt_of_le_of_lt (Nat.mod_le _ _) hpmn31
  have hdiv31 : pidMN d (ax0 d i) / (8 * cn d) < 2 ^ 31 := lt_of_le_of_lt (Nat.div_le_self _ _) hpmn31
  have hmod31' : ∀ b, pidMN d (ax0 d i) % (8 * cn d) % b < 2 ^ 31 := fun b =>
    lt_of_le_of_lt (Nat.mod_le _ _) hmod31
  have hdiv31' : ∀ b, pidMN d (ax0 d i) % (8 * cn d) / b < 2 ^ 31 := fun b =>
    lt_of_le_of_lt (Nat.div_le_self _ _) hmod31
  unfold model loraShrink
  dsimp only
  simp only [tl.program_id, hp0, hp1, hp2, IntV.ofNat_add_ofNat]
  refine SafeA.idiv1 (w := 32) ?_ ?_ fun r1 h1 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e1 : r1.item = IntV.ofNat (cn d) := by
    have := IntV.I32.item_idiv (a := d.R + (16 - 1)) (b := 16) (by omega) (by norm_num) h1
    simpa [cn, show d.R + 15 = d.R + (16 - 1) by omega] using this
  rw [e1]
  refine SafeA.idiv1 (w := 32) ?_ ?_ fun r2 h2 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e2 : r2.item = IntV.ofNat (cm d) := by
    have := IntV.I32.item_idiv (a := d.M + (32 - 1)) (b := 32) (by omega) (by norm_num) h2
    simpa [cm, show d.M + 31 = d.M + (32 - 1) by omega] using this
  rw [e2]
  refine SafeA.irem1 (w := 32) ?_ ?_ fun r3 h3 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e3 : r3.item = IntV.ofNat (pidSk d (ax0 d i)) :=
    IntV.I32.item_irem hxlt hskpos (by omega) h3
  rw [e3]
  refine SafeA.idiv1 (w := 32) ?_ ?_ fun r4 h4 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e4 : r4.item = IntV.ofNat (pidMN d (ax0 d i)) :=
    IntV.I32.item_idiv hxlt (by omega) h4
  rw [e4]
  simp only [IntV.ofNat_mul_ofNat]
  have h8cn : 8 * cn d < 2 ^ 31 := by unfold cn; omega
  have hfpm : gFpm (cn d) (pidMN d (ax0 d i)) < cm d := gFpm_lt hpmn
  have hgsz := gGsz_pos hpmn
  have hgsz8 : gGsz (cm d) (cn d) (pidMN d (ax0 d i)) ≤ 8 := gGsz_le
  refine SafeA.idiv1 (w := 32) ?_ ?_ fun r5 h5 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e5 : r5.item = IntV.ofNat (pidMN d (ax0 d i) / (8 * cn d)) :=
    IntV.I32.item_idiv (by omega) h8cn h5
  rw [e5]
  refine SafeA.irem1 (w := 32) ?_ ?_ fun r6 h6 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e6 : r6.item = IntV.ofNat (pidMN d (ax0 d i) % (8 * cn d)) :=
    IntV.I32.item_irem (by omega) (by omega) h8cn h6
  rw [e6]
  simp only [IntV.ofNat_mul_ofNat]
  rw [IntV.I32.item_iminimum_sub (b := pidMN d (ax0 d i) / (8 * cn d) * 8) (c := 8)
    (by have := hfpm; unfold gFpm at this; omega) (by omega) (by norm_num)]
  have hG : Min.min (cm d - pidMN d (ax0 d i) / (8 * cn d) * 8) 8 = gsm d (ax0 d i) := rfl
  rw [hG]
  have hgsm_pos : 0 < gsm d (ax0 d i) := hgsz
  have hgsm8 : gsm d (ax0 d i) ≤ 8 := hgsz8
  refine SafeA.irem1 (w := 32) ?_ ?_ fun r7 h7 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e7 : r7.item = IntV.ofNat (pidMN d (ax0 d i) % (8 * cn d) % gsm d (ax0 d i)) :=
    IntV.I32.item_irem (by omega) (by omega) (by omega) h7
  rw [e7]
  refine SafeA.irem1 (w := 32) ?_ ?_ fun r8 h8 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e8 : r8.item = IntV.ofNat (pidMN d (ax0 d i) % (8 * cn d)) :=
    IntV.I32.item_irem (by omega) (by omega) h8cn h8
  rw [e8]
  refine SafeA.idiv1 (w := 32) ?_ ?_ fun r9 h9 => ?_
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  · rw [IntV.I32.toInt_ofNat (by omega)]; positivity
  have e9 : r9.item = IntV.ofNat (pidMN d (ax0 d i) % (8 * cn d) / gsm d (ax0 d i)) :=
    IntV.I32.item_idiv (by omega) (by omega) h9
  rw [e9]
  simp only [IntV.ofNat_mul_ofNat, IntV.ofNat_add_ofNat]
  have hPM : pidMN d (ax0 d i) / (8 * cn d) * 8 + pidMN d (ax0 d i) % (8 * cn d) % gsm d (ax0 d i) = tileM d i := rfl
  have hPN : pidMN d (ax0 d i) % (8 * cn d) / gsm d (ax0 d i) = tileN d i := rfl
  have hPS : pidSk d (ax0 d i) = part d i := rfl
  rw [hPM, hPN, hPS]
  obtain ⟨hinpV, hloraV, houtV, htidV, hcntV, hstartV, hidV, htabH, hcntH, hstartH, hidH, htidH, houtH⟩ := hpre
  rw [buf_ptr_add_ofNat d.idBuf (grpOf d i) (by have := hd.G_lt; omega)]
  refine SafeA.load1_buf d.idBuf (hd.id_len _ hg) hidV (span_id_sub (hd.id_len _ hg)) fun v hv => ?_
  have ev : v = IntV.ofInt (idAt d (grpOf d i)) := by
    have := hidH (grpOf d i) (hd.id_len _ hg)
    rw [hv] at this
    exact Option.some.inj this
  rw [ev]
  by_cases hid : idAt d (grpOf d i) = -1
  · have : IntV.eq (IntV.ofInt (idAt d (grpOf d i))) (IntV.ofInt (-1) : I32) = true := by
      rw [hid]; simp [IntV.eq_eq]
    simp only [this, ite_true]
    exact SafeA.ret (Q_nil (fun h => h.1 hid))
  have hne : IntV.eq (IntV.ofInt (idAt d (grpOf d i))) (IntV.ofInt (-1) : I32) = false := by
    rw [IntV.eq_eq, IntV.I32.toInt_ofInt (by have := idAt_sfits hd (grpOf d i); exact this),
      IntV.I32.toInt_ofInt (by norm_num)]
    simp [hid]
  simp only [hne, Bool.false_eq_true, ite_false]
  -- the group's count
  rw [buf_ptr_add_ofNat d.cntBuf (grpOf d i) (by have := hd.G_lt; omega)]
  refine SafeA.load1_buf d.cntBuf (hd.cnt_len _ hg) hcntV (span_cnt_sub (hd.cnt_len _ hg)) fun v hv => ?_
  have ev2 : v = IntV.ofNat (cntAt d (grpOf d i)) := by
    have := hcntH (grpOf d i) (hd.cnt_len _ hg)
    rw [hv] at this
    exact Option.some.inj this
  rw [ev2]
  have hcntlt : cntAt d (grpOf d i) < 2 ^ 31 := by have := cntAt_le hd (grpOf d i); omega
  have hPMlt : tileM d i < cm d := gM_lt hpmn
  have hPM32 : tileM d i * 32 < 2 ^ 31 := by have : cm d ≤ 2 ^ 17 + 1 := by unfold cm; omega
                                             omega
  by_cases hact : tileM d i * 32 < cntAt d (grpOf d i)
  swap
  · have : IntV.le (IntV.ofNat (cntAt d (grpOf d i)) : I32) (IntV.ofNat (tileM d i * 32)) = true := by
      rw [IntV.le_eq, IntV.I32.toInt_ofNat hcntlt, IntV.I32.toInt_ofNat hPM32]
      simp; omega
    simp only [this, ite_true]
    exact SafeA.ret (Q_nil (fun h => hact h.2))
  have hne2 : IntV.le (IntV.ofNat (cntAt d (grpOf d i)) : I32) (IntV.ofNat (tileM d i * 32)) = false := by
    rw [IntV.le_eq, IntV.I32.toInt_ofNat hcntlt, IntV.I32.toInt_ofNat hPM32]
    simp; omega
  simp only [hne2, Bool.false_eq_true, ite_false]
  have hactive : Active d i := ⟨hid, hact⟩
  have hstart_lt := hactive.start_lt hd
  have hmlen_pos := hactive.mlen_pos
  have hmlen32 := mlen_le d i
  -- the start of the group
  rw [buf_ptr_add_ofNat d.startBuf (grpOf d i) (by have := hd.G_lt; omega)]
  refine SafeA.load1_buf d.startBuf (hd.start_len _ hg) hstartV (span_start_sub (hd.start_len _ hg))
    fun v hv => ?_
  have ev3 : v = IntV.ofNat (startAt d (grpOf d i)) := by
    have := hstartH (grpOf d i) (hd.start_len _ hg)
    rw [hv] at this
    exact Option.some.inj this
  rw [ev3]
  have hmlenv : (tl.iminimum (Tensor.scalar (IntV.ofNat 32 : I32))
      (Tensor.scalar (IntV.ofNat (cntAt d (grpOf d i)) - IntV.ofNat (tileM d i * 32)))).item =
      IntV.ofNat (mlen d i) := by
    rw [IntV.I32.item_iminimum_sub' (by omega) hcntlt (by norm_num)]
    rfl
  rw [hmlenv]
  have hx32 : ∀ r : Fin (32 - 0), ((tl.arange 0 32).get1 r).toInt = ((r.val : Nat) : Int) :=
    fun r => IntV.I32.toInt_tl_arange (by norm_num) r
  have hy32 : ∀ r : Fin (32 - 0),
      ((bc (IntV.ofNat (mlen d i) : I32) : Tensor [32 - 0] I32).get1 r).toInt = ((mlen d i : Nat) : Int) :=
    fun r => IntV.I32.toInt_ofNat (by omega)
  refine SafeA.irem (x := tl.arange 0 32) (y := bc (IntV.ofNat (mlen d i) : I32))
    (irem_ok (f := fun r : Fin (32 - 0) => r.val) (M := mlen d i) hx32 hy32 hmlen_pos) fun offm hoffm => ?_
  have hoffm_val : ∀ r : Fin (32 - 0), (offm.get1 r).toInt = ((r.val % mlen d i : Nat) : Int) :=
    irem_toInt (f := fun r : Fin (32 - 0) => r.val) (M := mlen d i) hx32 hy32 hmlen_pos hoffm
  have hpos_lt : ∀ r : Fin (32 - 0), startAt d (grpOf d i) + tileM d i * 32 + r.val % mlen d i < d.M :=
    fun r => (hactive.row hd (Nat.mod_lt _ hmlen_pos)).2.2
  rw [buf_ptr_add_ofNat d.tIdxBuf _ (by omega), buf_ptr_add_ofNat' d.tIdxBuf _ _ (by omega)]
  refine SafeA.load_buf d.tIdxBuf (g := fun idx : Idx [32 - 0] => startAt d (grpOf d i) + tileM d i * 32 + idx.1.val % mlen d i)
    (en := fun _ => true) (Buf.addrs_of_get_all _ ?_) ?_ htidV ?_ fun ram hramr _ => ?_
  · rintro ⟨r, u⟩
    cases u
    show (d.tIdxBuf.ptr _).offset (offm.get1 r).toInt = _
    rw [hoffm_val r, Buf.ptr_offset]
    push_cast; ring_nf
  · rintro ⟨r, u⟩ _
    exact hd.tid_len _ (hpos_lt r)
  · refine tid_elems_sub ?_
    rintro j ⟨⟨r, u⟩, _, rfl⟩
    exact hd.tid_len _ (hpos_lt r)
  have hram : ∀ r : Fin (32 - 0), ram.get1 r = IntV.ofNat (tokOf d i (r.val % mlen d i)) := by
    intro r
    have h1 := hramr (r, ()) rfl
    have h2 := htidH _ (hpos_lt r)
    rw [h2] at h1
    exact (Option.some.inj h1).symm
  have hlora : (d.lora (slice d i)).Valid init := hloraV _ hs
  have hS1' : slice d i * (d.M * d.R) < 2 ^ 31 := by
    by_cases h1 : 1 < d.S
    · exact lt_of_le_of_lt (Nat.mul_le_mul_right _ (by omega)) (hd.slice_wrap h1)
    · have : slice d i = 0 := by omega
      rw [this]; simp
  by_cases hS : d.S = 1
  have hsl : slice d i = 0 := by unfold slice; rw [hS]; exact Nat.mod_one _
  simp only [hS, ite_true, Prog.pure_eq]
  rw [show d.lora 0 = d.lora (slice d i) from by rw [hsl]]
  rw [show d.out.ptr 0 = d.out.ptr ((slice d i * (d.M * d.R) : Nat) : Int) from by rw [hsl]; simp]
  swap
  have hS1 : 1 < d.S := by have := hd.S_pos; omega
  simp only [hS, ite_false]
  obtain ⟨htabV, htabR⟩ := htabH hS1
  rw [buf_ptr_add_ofNat d.tab (slice d i) (by have := hd.S_lt; omega)]
  refine SafeA.load1_buf d.tab (hd.tab_len hS1 _ hs) htabV (span_tab_sub hS1 (hd.tab_len hS1 _ hs))
    fun v hv => ?_
  have ev4 : v = IntV.ofInt (d.lora (slice d i)).base := by
    have := htabR (slice d i) hs
    rw [hv] at this
    exact Option.some.inj this
  rw [ev4]
  simp only [Prog.pure_eq]
  have hptr : tl.intToPtr .bf16 (IntV.ofInt (d.lora (slice d i)).base : U64) =
      (d.lora (slice d i)).ptr 0 := by
    have hlen : 0 < (d.lora (slice d i)).len := by
      have := hd.lora_len (slice d i) (idAt d (grpOf d i)).toNat 0 0 hs (hactive.lora_idx hd).1 hR hK
      omega
    have hb2 : DType.bf16.bytes = 2 := rfl
    have hv1 := hlora.2.1
    rw [hb2] at hv1
    rw [intToPtr_ofInt hlora.1 (by omega)]
    simp [Buf.ptr]
  rw [hptr]
  rw [buf_ptr_add_ofNat d.out (slice d i * (d.M * d.R)) hS1']
  all_goals
    have hRlt' : d.R < 2 ^ 31 := hRlt
    have hx16 : ∀ n : Fin (16 - 0),
        (((tl.arange 0 16 + (IntV.ofNat (tileN d i * 16) : I32) : Tensor [16 - 0] I32)).get1 n).toInt =
        ((16 * tileN d i + n.val : Nat) : Int) := by
      intro n
      have h := tileN_lt hd hi (n := n.val) (by have := n.isLt; omega)
      rw [Tensor.get1_adds, IntV.I32.add_nat (IntV.I32.toInt_tl_arange (n := 16) (by norm_num) n)
        (IntV.I32.toInt_ofNat (by omega)) (by omega)]
      push_cast; ring
    have hy16 : ∀ n : Fin (16 - 0),
        ((bc (IntV.ofNat d.R : I32) : Tensor [16 - 0] I32).get1 n).toInt = (d.R : Int) :=
      fun n => IntV.I32.toInt_ofNat hRlt'
    refine SafeA.irem (x := tl.arange 0 16 + IntV.ofNat (tileN d i * 16))
      (y := bc (IntV.ofNat d.R : I32))
      (irem_ok (f := fun n : Fin (16 - 0) => 16 * tileN d i + n.val) (M := d.R) hx16 hy16 hR)
      fun rbn0 hrbn0 => ?_
    have hrbn : ∀ n : Fin (16 - 0), (rbn0.get1 n).toInt = ((16 * tileN d i + n.val : Nat) : Int) := by
      intro n
      rw [irem_toInt (f := fun n : Fin (16 - 0) => 16 * tileN d i + n.val) (M := d.R) hx16 hy16 hR
        hrbn0 n, Nat.mod_eq_of_lt (tileN_lt hd hi (by have := n.isLt; omega))]
    refine SafeA.max_contiguous_multiple_of (C' := 16) (v := 16)
      (contigArange (B := 16 * tileN d i) hrbn)
      (firstsArange (by norm_num) hrbn ⟨(tileN d i : Int), by push_cast; ring⟩) ?_
    refine SafeA.convert (d := .i32) (d' := .i64) (fun _ => ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩)
      fun ram64 hram64 => ?_
    have hr64 : ∀ r : Fin (32 - 0), (ram64.get1 r).toInt = ((tokOf d i (r.val % mlen d i) : Nat) : Int) := by
      intro r
      have hrow := hactive.row hd (Nat.mod_lt r.val hmlen_pos)
      rw [show ram64.get1 r = ram64.get (r, ()) from rfl, ConvRel.toInt_i32_i64 (hram64 (r, ()))]
      show (ram.get1 r).toInt = _
      rw [hram r, IntV.I32.toInt_ofNat (by have := hd.M_le; omega)]
    refine SafeA.convert (d := .i32) (d' := .i64) (x := Tensor.scalar (IntV.ofNat d.K : I32))
      (fun _ => ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩) fun d0r hd0r => ?_
    have hd0 : d0r.item.toInt = (d.K : Int) := by
      rw [show d0r.item = d0r.get () from rfl, ConvRel.toInt_i32_i64 (hd0r ())]
      exact IntV.I32.toInt_ofNat (by have := hd.K_top; omega)
    have hSTEP : bk d * sk d < 2 ^ 31 := by
      rcases bk_sk_cases d with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2] <;> norm_num
    have hSTEP2 : bk d * sk d ≤ 2 ^ 14 := by
      rcases bk_sk_cases d with ⟨h1, h2⟩ | ⟨h1, h2⟩ <;> rw [h1, h2] <;> norm_num
    have hSTEPpos : 0 < bk d * sk d := Nat.mul_pos (bk_pos d) (sk_pos d)
    have hKtop := hd.K_top
    have hxnum : (IntV.ofNat d.K + (IntV.ofNat (bk d * sk d) - 1 : I32)).toInt =
        ((d.K + (bk d * sk d - 1) : Nat) : Int) := by
      have h1 : (1 : I32).toInt = ((1 : Nat) : Int) := by
        rw [IntV.toInt_one (by norm_num)]; rfl
      have h2 := IntV.I32.sub_nat (IntV.I32.toInt_ofNat hSTEP) h1 hSTEPpos hSTEP
      exact IntV.I32.add_nat (IntV.I32.toInt_ofNat (by omega)) h2 (by omega)
    have hnum_lt : numIters d < 2 ^ 31 := by
      unfold numIters
      exact lt_of_le_of_lt (Nat.div_le_self _ _) (by omega)
    refine SafeA.idiv1 (w := 32) ?_ ?_ fun nr hnr => ?_
    · rw [hxnum]; positivity
    · rw [IntV.I32.toInt_ofNat hSTEP]; positivity
    have hnum : nr.item = IntV.ofNat (numIters d) := by
      rw [hxnum, IntV.I32.toInt_ofNat hSTEP] at hnr
      refine IntV.I32.eq_ofNat ?_ hnum_lt
      rw [hnr]
      exact (Int.natCast_div _ _).symm
    rw [hnum]
    have h0i : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)
    have h1i : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)
    have hnv : (IntV.ofNat (numIters d) : I32).toInt = (numIters d : Int) :=
      IntV.I32.toInt_ofNat hnum_lt
    have hTC : tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat (numIters d) : I32).toInt
        (IntV.ofInt 1 : I32).toInt = numIters d := by
      rw [h0i, h1i, hnv]
      show (((numIters d : Int) - 0 + 1 - 1) / 1).toNat = _
      simp
    refine SafeA.forRange (I := LoopInv d init i) ?_ ?_ ?_ ?_
    · unfold tl.RangeOk
      rw [h0i, h1i, hnv]
      have hmax : IntV.maxInt true 32 = 2 ^ 31 - 1 := by simp [IntV.maxInt]
      rw [hmax]
      refine ⟨by norm_num, ?_, ?_, ?_⟩ <;> omega
    · refine ⟨rfl, fun r kk => aPtr_zero hd hactive hr64 hd0 r kk,
        fun kk n => bPtr_zero hd hactive hi hrbn kk n, ?_⟩
      intro r n hr va vb hD
      show LaneInv d va vb (part d i) (min 0 (stepsOf d (part d i))) (FloatV.ofPat 0 : F32)
      rw [Nat.zero_min]
      exact laneInv_zero d va vb _
    · intro t ht st m1 hI
      rw [hTC] at ht
      have hkv : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) : I32) =
          IntV.ofNat t := by
        rw [h0i, h1i]; simp [IntV.ofInt_natCast]
      simp only [hkv]
      rw [hI.1]
      exact body_step hP hd hpre0 hi hactive hs ht ⟨rfl, hI.2⟩
    · intro st m1 hI
      obtain ⟨rfl, -, -, hacc⟩ := hI
      refine safe_epilogue hP hd hpre0 hi hactive hs rfl hram ?_
      intro r n hr va vb hD
      have := hacc r n hr va vb hD
      rwa [hTC, Nat.min_eq_right (stepsOf_le_numIters d _)] at this

end Claims.LoraShrink
