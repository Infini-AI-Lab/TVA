import Claim
import TVA.Proof
import Proof.Geometry
import Proof.Numeric

/-!
# The model's scalar operations (tile counts, `compute_pid`, loop bounds) evaluated at the instance.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-! ### Value bounds of the instance -/

/-- `nT d` is below `2^31`. -/
theorem nT_lt {d : Inst} (hd : d.ok) : nT d < 2 ^ 31 := by
  have h := hd.tiles_le
  have ht : tiles d = nT d := tiles_eq hd
  have hS := hd.sms.1
  omega

/-- The number of tile rows fits in `int32`. -/
theorem nPM_lt {d : Inst} (hd : d.ok) : nPM d < 2 ^ 31 := by
  have h := hd.M_le
  have hle : d.M / 128 ≤ d.M := Nat.div_le_self _ _
  simp only [nPM]
  omega

/-- The number of tile columns fits in `int32`. -/
theorem nPN_lt {d : Inst} (hd : d.ok) : nPN d < 2 ^ 31 := by
  have h := hd.N_le
  have hle : d.N / 256 ≤ d.N := Nat.div_le_self _ _
  simp only [nPN]
  omega

/-- `nPM d` is positive, for a tile index that exists. -/
theorem nPM_pos {d : Inst} (_hd : d.ok) {t : Nat} (ht : t < nT d) : 0 < nPM d := by
  have h : 0 < nT d := lt_of_le_of_lt (Nat.zero_le t) ht
  simp only [nT] at h
  exact Nat.pos_of_mul_pos_right h

/-- `d.M + 127` does not wrap. -/
theorem M127_lt {d : Inst} (hd : d.ok) : d.M + 127 < 2 ^ 31 := by
  have := hd.M_le; omega

/-- `N + 255` fits in `int32` (the model's `cdiv(N, 256)` numerator). -/
theorem N255_lt {d : Inst} (hd : d.ok) : d.N + 255 < 2 ^ 31 := by
  have := hd.N_le; omega

/-- `K + 63` fits in `int32` (the model's `cdiv(K, 64)` numerator). -/
theorem K63_lt {d : Inst} (hd : d.ok) : d.K + 63 < 2 ^ 31 := by
  have := hd.K_le; omega

/-- `cdiv(M, 128)` is `nPM d`. -/
theorem M127_div {d : Inst} (hd : d.ok) : (d.M + 127) / 128 = nPM d := by
  have h : d.M + 128 - 1 = d.M + 127 := by omega
  rw [← h, cdiv_eq (by norm_num : 0 < 128) hd.M_tile]
  rfl

/-- `cdiv(N, 256)` is `nPN d`. -/
theorem N255_div {d : Inst} (hd : d.ok) : (d.N + 255) / 256 = nPN d := by
  have h : d.N + 256 - 1 = d.N + 255 := by omega
  rw [← h, cdiv_eq (by norm_num : 0 < 256) hd.N_tile]
  rfl

/-- `cdiv(K, 64)` is `steps d.K`. -/
theorem K63_div {d : Inst} (_hd : d.ok) : (d.K + 63) / 64 = steps d.K := rfl

/-- `64` divides `d.K`. -/
theorem K_dvd {d : Inst} (hd : d.ok) : 64 ∣ d.K := by
  have h := hd.K_dim.2.1
  omega

/-- `steps d.K` is `d.K / 64`. -/
theorem steps_eq {d : Inst} (hd : d.ok) : steps d.K = d.K / 64 := by
  unfold steps
  have h : d.K + 64 - 1 = d.K + 63 := by omega
  have hdvd : d.K % 64 = 0 := by have := hd.K_dim.2.1; omega
  rw [← h, cdiv_eq (by norm_num : 0 < 64) hdvd]

/-- `steps d.K` is below `2^31`. -/
theorem steps_lt {d : Inst} (hd : d.ok) : steps d.K < 2 ^ 31 := by
  have := hd.K_le
  unfold steps
  omega

/-! ### The scalar operations of the model -/

/-- The `item` of a rank-0 `tl.iminimum` is `IntV.min`. -/
theorem iminimum_item (a b : I32) :
    (tl.iminimum (Tensor.scalar a) (Tensor.scalar b)).item = IntV.min a b := rfl

/-- `tl.iminimum` of two scalars is the smaller value. -/
theorem iminimum_toInt (a b : I32) :
    (tl.iminimum (Tensor.scalar a) (Tensor.scalar b)).item.toInt = min a.toInt b.toInt := by
  rw [iminimum_item, IntV.toInt_min]

/-- **`num_pid_m`, `num_pid_n`, `k_tiles`, `num_tiles` and `tile_id_c`, then the rest.** -/
theorem safe_setup {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {sp : I32} {i : Nat} (hsp : sp.toInt = (i : Int))
    (hi : i < d.NUM_SMS)
    {k : I32 → I32 → I32 → I32 → I32 → Prog α}
    (hk : ∀ npm npn kt nt tidc : I32,
      npm.toInt = ((nPM d : Nat) : Int) → npn.toInt = ((nPN d : Nat) : Int) →
      kt.toInt = ((steps d.K : Nat) : Int) → nt.toInt = ((nT d : Nat) : Int) →
      tidc.toInt = ((i : Int) - (d.NUM_SMS : Int)) →
      Safe E R W Q m (k npm npn kt nt tidc)) :
    Safe E R W Q m (do
      let num_pid_m := (← tl.idiv
          (Tensor.scalar ((IntV.ofNat d.M : I32) + (IntV.ofNat 127 : I32)))
          (Tensor.scalar (IntV.ofNat 128 : I32))).item
      let num_pid_n := (← tl.idiv
          (Tensor.scalar ((IntV.ofNat d.N : I32) + (IntV.ofNat 255 : I32)))
          (Tensor.scalar (IntV.ofNat 256 : I32))).item
      let k_tiles := (← tl.idiv
          (Tensor.scalar ((IntV.ofNat d.K : I32) + (IntV.ofNat 63 : I32)))
          (Tensor.scalar (IntV.ofNat 64 : I32))).item
      let num_tiles := num_pid_m * num_pid_n
      let tile_id_c := sp - (IntV.ofNat d.NUM_SMS : I32)
      k num_pid_m num_pid_n k_tiles num_tiles tile_id_c) := by
  have hM31 := M127_lt hd
  have hN31 := N255_lt hd
  have hK31 := K63_lt hd
  have hM : ((IntV.ofNat d.M : I32) + (IntV.ofNat 127 : I32)).toInt =
      ((d.M + 127 : Nat) : Int) :=
    IntV.I32.add_nat (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_ofNat (by norm_num)) hM31
  have hN : ((IntV.ofNat d.N : I32) + (IntV.ofNat 255 : I32)).toInt =
      ((d.N + 255 : Nat) : Int) :=
    IntV.I32.add_nat (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_ofNat (by norm_num)) hN31
  have hK : ((IntV.ofNat d.K : I32) + (IntV.ofNat 63 : I32)).toInt =
      ((d.K + 63 : Nat) : Int) :=
    IntV.I32.add_nat (IntV.I32.toInt_ofNat (by omega)) (IntV.I32.toInt_ofNat (by norm_num)) hK31
  have h128 : (IntV.ofNat 128 : I32).toInt = (128 : Int) := IntV.I32.toInt_ofNat (by norm_num)
  have h256 : (IntV.ofNat 256 : I32).toInt = (256 : Int) := IntV.I32.toInt_ofNat (by norm_num)
  have h64 : (IntV.ofNat 64 : I32).toInt = (64 : Int) := IntV.I32.toInt_ofNat (by norm_num)
  have hM8 : (0 : Int) ≤ ((d.M + 127 : Nat) : Int) := by exact_mod_cast Nat.zero_le _
  have hN8 : (0 : Int) ≤ ((d.N + 255 : Nat) : Int) := by exact_mod_cast Nat.zero_le _
  have hK8 : (0 : Int) ≤ ((d.K + 63 : Nat) : Int) := by exact_mod_cast Nat.zero_le _
  refine safe_idiv1 (x := (IntV.ofNat d.M : I32) + (IntV.ofNat 127 : I32))
    (y := (IntV.ofNat 128 : I32)) (by rw [hM]; exact hM8) (by rw [h128]; norm_num) ?_
  intro tm htm
  have hnpm' : tm.item.toInt = ((nPM d : Nat) : Int) := by
    have h1 : ((d.M + 127 : Nat) : Int) / ((128 : Nat) : Int) = ((nPM d : Nat) : Int) := by
      rw [← Int.natCast_ediv, M127_div hd]
    rw [htm, hM, h128]; exact h1
  refine safe_idiv1 (x := (IntV.ofNat d.N : I32) + (IntV.ofNat 255 : I32))
    (y := (IntV.ofNat 256 : I32)) (by rw [hN]; exact hN8) (by rw [h256]; norm_num) ?_
  intro tn htn
  have hnpn' : tn.item.toInt = ((nPN d : Nat) : Int) := by
    have h1 : ((d.N + 255 : Nat) : Int) / ((256 : Nat) : Int) = ((nPN d : Nat) : Int) := by
      rw [← Int.natCast_ediv, N255_div hd]
    rw [htn, hN, h256]; exact h1
  refine safe_idiv1 (x := (IntV.ofNat d.K : I32) + (IntV.ofNat 63 : I32))
    (y := (IntV.ofNat 64 : I32)) (by rw [hK]; exact hK8) (by rw [h64]; norm_num) ?_
  intro tk htk
  have hkt' : tk.item.toInt = ((steps d.K : Nat) : Int) := by
    have h1 : ((d.K + 63 : Nat) : Int) / ((64 : Nat) : Int) = ((steps d.K : Nat) : Int) := by
      rw [← Int.natCast_ediv, K63_div hd]
    rw [htk, hK, h64]; exact h1
  have hnt : (tm.item * tn.item).toInt = ((nT d : Nat) : Int) := by
    refine IntV.I32.mul_nat hnpm' hnpn' ?_
    exact nT_lt hd
  have hspS : (0 : Int) ≤ (i : Int) := by exact_mod_cast Nat.zero_le i
  have hSval : (IntV.ofNat d.NUM_SMS : I32).toInt = ((d.NUM_SMS : Nat) : Int) :=
    IntV.I32.toInt_ofNat hd.sms.2
  have hSfits : IntV.SFits 32 (sp.toInt - (IntV.ofNat d.NUM_SMS : I32).toInt) := by
    rw [hsp, hSval]
    have hi' : (i : Int) < (d.NUM_SMS : Int) := by exact_mod_cast hi
    have hS0 : (0 : Int) ≤ (d.NUM_SMS : Int) := by exact_mod_cast Nat.zero_le _
    have hS2 : (d.NUM_SMS : Int) < 2 ^ 31 := by
      have := hd.sms.2; exact_mod_cast this
    constructor <;> push_cast <;> omega
  have htidc : (sp - (IntV.ofNat d.NUM_SMS : I32)).toInt =
      ((i : Int) - (d.NUM_SMS : Int)) := by
    rw [IntV.toInt_sub rfl rfl hSfits, hsp, hSval]
  exact hk tm.item tn.item tk.item (tm.item * tn.item) (sp - (IntV.ofNat d.NUM_SMS : I32))
    hnpm' hnpn' hkt' (by simpa only [nT] using hnt) htidc

/-! ### `compute_pid` -/

/-- **`compute_pid`.** On a tile index `t < nT d` (`tid.toInt = t`, `npg.toInt = 8·nPN d`,
`npm.toInt = nPM d`), `compute_pid` returns `(pm d t, pn d t)`. -/
theorem safe_compute_pid {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {tid npg npm : I32} {t : Nat}
    (htid : tid.toInt = (t : Int)) (ht : t < nT d)
    (hnpg : npg.toInt = ((8 * nPN d : Nat) : Int))
    (hnpm : npm.toInt = ((nPM d : Nat) : Int))
    {k : I32 → I32 → Prog α}
    (hk : ∀ p q : I32, p.toInt = ((pm d t : Nat) : Int) → q.toInt = ((pn d t : Nat) : Int) →
      Safe E R W Q m (k p q)) :
    Safe E R W Q m (compute_pid tid npg npm 8 >>= fun pq => k pq.1 pq.2) := by
  have h8 : (IntV.ofNat (8 : Nat) : I32).toInt = (8 : Int) := IntV.I32.toInt_ofNat (by norm_num)
  have hS := nPN_pos hd
  have hgid := gid_lt hd ht
  have hzpos := gsz_pos hd ht
  have hpm := pm_lt hd ht
  have hpn := pn_lt hd ht
  have hm31 := nPM_lt hd
  have hnpgp : 0 < ((8 * nPN d : Nat) : Int) := by
    have : 0 < 8 * nPN d := by omega
    exact_mod_cast this
  have htid0 : (0 : Int) ≤ tid.toInt := by rw [htid]; exact_mod_cast Nat.zero_le t
  simp only [compute_pid]
  refine safe_idiv1 (x := tid) (y := npg) htid0 (by rw [hnpg]; exact hnpgp) ?_
  intro tg htg
  have hg' : tg.item.toInt = ((gid d t : Nat) : Int) := by
    rw [htg, htid, hnpg, ← Int.natCast_ediv]
    rfl
  have hgid8_lt : gid d t * 8 < 2 ^ 31 := by omega
  have hfpm : (tg.item * (IntV.ofNat (8 : Nat) : I32)).toInt = ((8 * gid d t : Nat) : Int) := by
    have h := IntV.I32.mul_nat (a := tg.item) (b := (IntV.ofNat (8 : Nat) : I32))
      hg' h8 hgid8_lt
    simpa only [Nat.mul_comm] using h
  have hfpm_le : (8 * gid d t : Nat) ≤ nPM d := by omega
  have hfpm31 : (8 * gid d t : Nat) < 2 ^ 31 := lt_of_le_of_lt hfpm_le hm31
  have hsub : (npm - tg.item * (IntV.ofNat (8 : Nat) : I32)).toInt =
      ((nPM d - 8 * gid d t : Nat) : Int) := by
    rw [IntV.I32.sub_nat hnpm hfpm hfpm_le hm31]
  have hgsm : (tl.iminimum (Tensor.scalar (npm - tg.item * (IntV.ofNat (8 : Nat) : I32)))
      (Tensor.scalar (IntV.ofNat (8 : Nat) : I32))).item.toInt =
      ((gsz d t : Nat) : Int) := by
    rw [iminimum_toInt, hsub, h8]
    show min ((nPM d - 8 * gid d t : Nat) : Int) ((8 : Nat) : Int) = ((gsz d t : Nat) : Int)
    rw [← Nat.cast_min]
    rfl
  have hgsmp : 0 < (tl.iminimum (Tensor.scalar (npm - tg.item * (IntV.ofNat (8 : Nat) : I32)))
      (Tensor.scalar (IntV.ofNat (8 : Nat) : I32))).item.toInt := by
    rw [hgsm]; exact_mod_cast hzpos
  refine safe_irem1 (x := tid)
    (y := (tl.iminimum (Tensor.scalar (npm - tg.item * (IntV.ofNat (8 : Nat) : I32)))
      (Tensor.scalar (IntV.ofNat (8 : Nat) : I32))).item) htid0 hgsmp ?_
  intro tm htm
  have hpmv' : tm.item.toInt = ((t % gsz d t : Nat) : Int) := by
    rw [htm, htid, hgsm, ← Int.natCast_emod]
  have hpmv31 : 8 * gid d t + t % gsz d t < 2 ^ 31 := by
    have := pm_lt hd ht
    simp only [pm] at this
    omega
  have hpidm : (tg.item * (IntV.ofNat (8 : Nat) : I32) + tm.item).toInt =
      ((pm d t : Nat) : Int) := by
    rw [IntV.I32.add_nat hfpm hpmv' hpmv31]
    simp only [pm]
  refine safe_irem1 (x := tid) (y := npg) htid0 (by rw [hnpg]; exact hnpgp) ?_
  intro tn htn
  have hnrem' : tn.item.toInt = ((t % (8 * nPN d) : Nat) : Int) := by
    rw [htn, htid, hnpg, ← Int.natCast_emod]
  have hnrem8 : 0 ≤ tn.item.toInt := by rw [hnrem']; exact_mod_cast Nat.zero_le _
  refine safe_idiv1 (x := tn.item)
    (y := (tl.iminimum (Tensor.scalar (npm - tg.item * (IntV.ofNat (8 : Nat) : I32)))
      (Tensor.scalar (IntV.ofNat (8 : Nat) : I32))).item) hnrem8 hgsmp ?_
  intro tp htp
  have hpnv' : tp.item.toInt = ((pn d t : Nat) : Int) := by
    rw [htp, hnrem', hgsm, ← Int.natCast_ediv]; rfl
  exact hk (tg.item * (IntV.ofNat (8 : Nat) : I32) + tm.item) tp.item hpidm hpnv'

/-! ### The loops' bounds -/

/-- The outer `tl.forRange` bounds are valid. -/
theorem rangeOk_outer {d : Inst} (hd : d.ok) {sp nt : I32} {i : Nat}
    (hsp : sp.toInt = (i : Int)) (hnt : nt.toInt = ((nT d : Nat) : Int))
    (hi : i < min d.NUM_SMS (tiles d)) :
    tl.RangeOk sp nt (IntV.ofNat d.NUM_SMS : I32) := by
  have hS : 0 < d.NUM_SMS := hd.sms.1
  have hSlt : d.NUM_SMS < 2 ^ 31 := hd.sms.2
  have htl : tiles d = nT d := tiles_eq hd
  have hle : nT d + d.NUM_SMS ≤ 2 ^ 31 := by
    have := hd.tiles_le
    omega
  have hstep : (IntV.ofNat d.NUM_SMS : I32).toInt = ((d.NUM_SMS : Nat) : Int) :=
    IntV.I32.toInt_ofNat hSlt
  have hlo : (i : Int) ≤ (2 ^ 31 : Int) - 1 := by
    have h1 : i < tiles d := by omega
    omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [hstep]; exact_mod_cast hS
  · rw [hnt, hstep, show IntV.maxInt true 32 = ((2 ^ 31 : Int) - 1) from by
      norm_num [IntV.maxInt]]
    push_cast
    omega
  · rw [hsp, show IntV.maxInt true 32 = ((2 ^ 31 : Int) - 1) from by
      norm_num [IntV.maxInt]]
    exact hlo
  · rw [hstep, show IntV.maxInt true 32 = ((2 ^ 31 : Int) - 1) from by
      norm_num [IntV.maxInt]]
    push_cast
    omega

/-- The inner `tl.forRange` bounds are valid. -/
theorem rangeOk_inner {d : Inst} (hd : d.ok) {kt : I32}
    (hkt : kt.toInt = ((steps d.K : Nat) : Int)) :
    tl.RangeOk (IntV.ofInt 0 : I32) kt (IntV.ofInt 1 : I32) := by
  have h0 : (IntV.ofInt 0 : I32).toInt = (0 : Int) :=
    IntV.I32.toInt_ofInt (by norm_num)
  have h1 : (IntV.ofInt 1 : I32).toInt = (1 : Int) :=
    IntV.I32.toInt_ofInt (by norm_num)
  have hst : steps d.K ≤ 128 := steps_le hd.K_le
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [h1]; norm_num
  · rw [hkt, h1, show IntV.maxInt true 32 = ((2 ^ 31 : Int) - 1) from by
      norm_num [IntV.maxInt]]
    push_cast
    omega
  · rw [h0, show IntV.maxInt true 32 = ((2 ^ 31 : Int) - 1) from by
      norm_num [IntV.maxInt]]
    norm_num
  · rw [h1, show IntV.maxInt true 32 = ((2 ^ 31 : Int) - 1) from by
      norm_num [IntV.maxInt]]
    norm_num

end Claims.PersistentMatmul
