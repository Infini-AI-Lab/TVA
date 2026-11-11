import Claim
import TVA.Proof
import Proof.Geometry
import Proof.Invariant
import Proof.Lanes
import Proof.Compute
import Proof.Numeric

/-!
# One outer trip (one tile): the inner loop, the epilogue and the store are safe.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-- A sum of two nonnegative `int32`s below `2^31` fits in `int32`. -/
theorem sfts32_add {x y : Int} (hx : 0 ≤ x) (hy : 0 ≤ y) (h : x + y < 2 ^ 31) :
    IntV.SFits 32 (x + y) := by
  unfold IntV.SFits
  have h2 : (2 : Int) ^ (32 - 1) = 2 ^ 31 := by norm_num
  rw [h2]
  constructor
  · linarith
  · exact h

/-- `safe_compute_pid` with the continuation taking the pair `(pid_m, pid_n)`, which is the
shape a destructuring `let (a, b) ← compute_pid …` has in a `do` block. -/
theorem safe_compute_pid_pair {d : Inst} (hd : d.ok) {E : Env} {R W : Set Int} {α : Type}
    {Q : α → Mem → Prop} {m : Mem} {tid npg npm : I32} {t : Nat}
    (htid : tid.toInt = (t : Int)) (ht : t < nT d)
    (hnpg : npg.toInt = ((8 * nPN d : Nat) : Int))
    (hnpm : npm.toInt = ((nPM d : Nat) : Int))
    {k : I32 × I32 → Prog α}
    (hk : ∀ p q : I32, p.toInt = ((pm d t : Nat) : Int) → q.toInt = ((pn d t : Nat) : Int) →
      Safe E R W Q m (k (p, q))) :
    Safe E R W Q m (compute_pid tid npg npm 8 >>= fun pq => k pq) := by
  refine safe_compute_pid (d := d) (t := t) (k := fun p q => k (p, q)) hd htid ht hnpg hnpm ?_
  intro p q hp hq
  exact hk p q hp hq

/-- **One trip of the outer loop.** The trip computes the tile `T = i + t * NUM_SMS` and stores
it; `CellInv` is carried by the inner loop (`hinner`), the store keeps the memory invariant and
adds the tile's `OutSpec` to `Done`. -/
theorem safe_trip {d : Inst} (hd : d.ok) {E : Env} {init m : Mem} (hpre : Pre d init)
    (hinner : ∀ {T t : Nat} {offs_am : Tensor [128] I32} {offs_bn : Tensor [256] I32}
      {offs_k_for_mask : Tensor [64] I32} {ki : I32} {acc : Tensor [128, 256] F32} {m₁ : Mem},
      T < nT d → t < steps d.K → MemInv d init m₁ →
      (∀ i : Fin 128, (offs_am.get1 i).toInt = ((pm d T * 128 + i.val : Nat) : Int)) →
      (∀ j : Fin 256, (offs_bn.get1 j).toInt = ((pn d T * 256 + j.val : Nat) : Int)) →
      (∀ j : Fin 64, (offs_k_for_mask.get1 j).toInt = (j.val : Int)) →
      ki.toInt = (t : Int) → CellInv d init T t acc →
      Safe E (abSet d) ∅
        (fun acc' (m' : Mem) => CellInv d init T (t + 1) acc' ∧ MemInv d init m' ∧ m' = m₁) m₁
        (innerLoop d offs_am offs_bn offs_k_for_mask ki acc))
    {i t T : Nat} (hTdef : T = i + t * d.NUM_SMS) (hTlt : T < nT d)
    (hdisj : ∀ u, u < t → Disjoint (tileSet d (i + u * d.NUM_SMS)) (tileSet d T))
    (hm : MemInv d init m) (hdone : Done d init i t m)
    {offs_k_for_mask : Tensor [64] I32} {npg npm kt tid tidc : I32}
    (hkm : ∀ j : Fin 64, (offs_k_for_mask.get1 j).toInt = (j.val : Int))
    (htid : tid.toInt = (T : Int))
    (htc : tidc.toInt = (T : Int) - (d.NUM_SMS : Int))
    (hnpg : npg.toInt = ((8 * nPN d : Nat) : Int))
    (hnpm : npm.toInt = ((nPM d : Nat) : Int))
    (hkt : kt.toInt = ((steps d.K : Nat) : Int)) :
    Safe E (abSet d) (tileSet d T)
      (fun tidc' (m' : Mem) =>
        tidc'.toInt = (T : Int) ∧ MemInv d init m' ∧ Done d init i (t + 1) m')
      m (trip d offs_k_for_mask npg npm kt tid tidc) := by
  have hSpos : 0 < d.NUM_SMS := hd.sms.1
  have hSlt : d.NUM_SMS < 2 ^ 31 := hd.sms.2
  have hTiles : tiles d = nT d := tiles_eq hd
  have hTle : T + d.NUM_SMS < 2 ^ 31 := by
    have h := hd.tiles_le
    have ht2 : T < tiles d := by rw [hTiles]; exact hTlt
    omega
  have hmulA : pm d T * 128 + 128 ≤ d.M := by
    have h := pm_mul_bound hd hTlt (i := 127) (by norm_num)
    omega
  have hcA : pm d T * 128 + 128 < 2 ^ 31 := by
    have h := hd.M_le
    omega
  have hmulB : pn d T * 256 + 256 ≤ d.N := by
    have h := pn_mul_bound hd hTlt (j := 255) (by norm_num)
    omega
  have hcB : pn d T * 256 + 256 < 2 ^ 31 := by
    have h := hd.N_le
    omega
  have hM31 : d.M < 2 ^ 31 := by have h := hd.M_le; omega
  have hN31 : d.N < 2 ^ 31 := by have h := hd.N_le; omega
  have hn128m : (128 : Nat) < 2 ^ 31 := by norm_num
  have hn256m : (256 : Nat) < 2 ^ 31 := by norm_num
  have hSval : (IntV.ofNat d.NUM_SMS : I32).toInt = ((d.NUM_SMS : Nat) : Int) :=
    IntV.I32.toInt_ofNat hSlt
  have hdvA : (128 : Int) ∣ ((pm d T * 128 : Nat) : Int) := by
    push_cast
    exact ⟨(pm d T : Int), by ring⟩
  have hdvB : (256 : Int) ∣ ((pn d T * 256 : Nat) : Int) := by
    push_cast
    exact ⟨(pn d T : Int), by ring⟩
  simp only [trip]
  refine safe_compute_pid_pair (d := d) (t := T) hd htid hTlt hnpg hnpm ?_
  intro pid_m pid_n hpm hpn
  dsimp only
  set oa : Tensor [128] I32 :=
    tl.where (((pid_m * (IntV.ofNat 128 : I32)) + tl.arange 0 128).ltS (IntV.ofNat d.M : I32))
      ((pid_m * (IntV.ofNat 128 : I32)) + tl.arange 0 128) (tl.full [128] (IntV.ofInt 0 : I32))
    with hoa
  set ob : Tensor [256] I32 :=
    tl.where (((pid_n * (IntV.ofNat 256 : I32)) + tl.arange 0 256).ltS (IntV.ofNat d.N : I32))
      ((pid_n * (IntV.ofNat 256 : I32)) + tl.arange 0 256) (tl.full [256] (IntV.ofInt 0 : I32))
    with hob
  have ham : ∀ i : Fin 128, (oa.get1 i).toInt = ((pm d T * 128 + i.val : Nat) : Int) := by
    intro i
    rw [hoa]
    exact start_arange_where_get1 (pm := pm d T) (pid := pid_m) (c := 128) (n := 128) (M := d.M)
      hpm hmulA hcA hn128m hM31 i
  have hbn : ∀ j : Fin 256, (ob.get1 j).toInt = ((pn d T * 256 + j.val : Nat) : Int) := by
    intro j
    rw [hob]
    exact start_arange_where_get1 (pm := pn d T) (pid := pid_n) (c := 256) (n := 256) (M := d.N)
      hpn hmulB hcB hn256m hN31 j
  refine safe_max_contiguous_multiple_of (x := oa) (C := 128) (v := 128)
    (contigArange (n := 128) (C := 128) (B := pm d T * 128) (x := oa) ham)
    (firstsArange (n := 128) (C := 128) (B := pm d T * 128) (x := oa) le_rfl ham hdvA) ?_
  refine safe_max_contiguous_multiple_of (x := ob) (C := 256) (v := 256)
    (contigArange (n := 256) (C := 256) (B := pn d T * 256) (x := ob) hbn)
    (firstsArange (n := 256) (C := 256) (B := pn d T * 256) (x := ob) le_rfl hbn hdvB) ?_
  have htripc : tl.tripCount (IntV.ofInt 0 : I32).toInt kt.toInt (IntV.ofInt 1 : I32).toInt
      = steps d.K := by
    have h0i : (IntV.ofInt 0 : I32).toInt = (0 : Int) := IntV.I32.toInt_ofInt (by norm_num)
    have h1i : (IntV.ofInt 1 : I32).toInt = (1 : Int) := IntV.I32.toInt_ofInt (by norm_num)
    rw [h0i, hkt, h1i]
    exact tripCount_steps d.K
  refine Safe.mono
    (W := iterW (fun _ => ∅)
      (tl.tripCount (IntV.ofInt 0 : I32).toInt kt.toInt (IntV.ofInt 1 : I32).toInt) ∪
        tileSet d T) ?_ (fun a m h => h) ?_
  · refine safe_forRange
      (I := fun (t' : Nat) (acc : Tensor [128, 256] F32) (m' : Mem) =>
        CellInv d init T t' acc ∧ MemInv d init m' ∧ m' = m)
      (Wt := fun _ => ∅) (Wk := tileSet d T) ?_ ?_ ?_ ?_ ?_ ?_
    · exact rangeOk_inner hd hkt
    · intro a b _ _
      simp
    · rw [iterW_empty]
      simp
    · refine ⟨?_, hm, rfl⟩
      intro iX nX va vb hva hvb
      rw [zeros_eq_full 128 256, full_get2_128_256]
      refine ⟨(FloatV.isFinite_ofPat_zero Format.binary32_regular), ?_⟩
      have hps : partSum (prodV va vb d.K) 0 = 0 := by simp [partSum]
      rw [hps, FloatV.toRat_ofPat_zero, partMag_zero, errB_zero']
      simp
    · intro t' ht' st m₁ hinv
      obtain ⟨hcell, hminv, hm₁⟩ := hinv
      have ht'K : t' < steps d.K := by rwa [htripc] at ht'
      have h0i : (IntV.ofInt 0 : I32).toInt = (0 : Int) := IntV.I32.toInt_ofInt (by norm_num)
      have h1i : (IntV.ofInt 1 : I32).toInt = (1 : Int) := IntV.I32.toInt_ofInt (by norm_num)
      have hkival :
          ((IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t' : Int) * (IntV.ofInt 1 : I32).toInt) :
            I32)).toInt = (t' : Int) := by
        rw [h0i, h1i]
        have hb : -(2 ^ 31 : Int) ≤ 0 + (t' : Int) * 1 ∧ 0 + (t' : Int) * 1 < 2 ^ 31 := by
          have h1 : t' < 2 ^ 31 := by
            have h2 := steps_lt hd
            omega
          constructor <;> omega
        rw [IntV.I32.toInt_ofInt hb]
        ring
      exact (hinner (m₁ := m₁) hTlt ht'K hminv ham hbn hkm hkival hcell).mono
        (fun a m' h => ⟨h.1, h.2.1, by rw [h.2.2, hm₁]⟩) le_rfl
    · intro st m₁ hinv
      rw [htripc] at hinv
      obtain ⟨hcell, hminv, hm₁⟩ := hinv
      have hT0 : (0 : Int) ≤ (T : Int) := by exact_mod_cast Nat.zero_le T
      have hS0' : (0 : Int) ≤ (d.NUM_SMS : Int) := by exact_mod_cast Nat.le_of_lt hSpos
      have hTS' : (T : Int) + (d.NUM_SMS : Int) < 2 ^ 31 := by
        push_cast
        exact_mod_cast hTle
      have h1 : IntV.toInt tidc = (T : Int) - (d.NUM_SMS : Int) := htc
      have h2 : IntV.toInt (IntV.ofNat d.NUM_SMS : I32) = (d.NUM_SMS : Int) := hSval
      have hsf : IntV.SFits 32 (IntV.toInt tidc + IntV.toInt (IntV.ofNat d.NUM_SMS : I32)) := by
        rw [h1, h2]
        unfold IntV.SFits
        have he : (2 : Int) ^ (32 - 1) = 2 ^ 31 := by norm_num
        rw [he]
        constructor
        · linarith
        · linarith
      have hnewtc : IntV.toInt (tidc + (IntV.ofNat d.NUM_SMS : I32)) = (T : Int) := by
        rw [IntV.toInt_add (a := tidc) (b := (IntV.ofNat d.NUM_SMS : I32)) rfl rfl hsf,
          h1, h2]
        ring
      refine safe_compute_pid_pair (d := d) (t := T) hd hnewtc hTlt hnpg hnpm ?_
      intro pid_m pid_n hpm hpn
      dsimp only
      set ocm : Tensor [128] I32 :=
        pid_m * (IntV.ofNat 128 : I32) + tl.arange 0 128 with hocm
      set ocn : Tensor [256] I32 :=
        pid_n * (IntV.ofNat 256 : I32) + tl.arange 0 256 with hocn
      have hcmO : ∀ i : Fin 128, (ocm.get1 i).toInt = ((pm d T * 128 + i.val : Nat) : Int) := by
        intro i
        rw [hocm]
        exact start_arange_get1 (pm := pm d T) (pid := pid_m) (c := 128) (n := 128)
          hpm hcA hn128m i
      have hcnO : ∀ j : Fin 256, (ocn.get1 j).toInt = ((pn d T * 256 + j.val : Nat) : Int) := by
        intro j
        rw [hocn]
        exact start_arange_get1 (pm := pn d T) (pid := pid_n) (c := 256) (n := 256)
          hpn hcB hn256m j
      refine safe_cvt (f := .binary32) (g := .binary16) (x := st) (rnd := .rn)
        (hok := ⟨?_, ?_, roundOk_f32_f16⟩) ?_
      · unfold Format.Std
        exact Or.inr (Or.inr (Or.inl rfl))
      · unfold Format.Std
        exact Or.inl rfl
      · intro c hc
        refine safe_store_buf d.c (g := cGs d T) (en := fun _ => true) (vals := c) ?_ ?_ ?_ ?_ ?_ ?_
        · intro x
          rw [tl.addrs]
          rw [ite_eq_left (cmask_get hd hTlt hcmO hcnO x)]
          exact congrArg some (cptrs_lane hd hTlt hcmO hcnO x)
        · intro x _
          exact cG_len hd hTlt x.1.isLt x.2.1.isLt
        · exact cGs_inj hd hTlt
        · exact Buf.valid_of_agree hpre.2.2 (fun x hx => hminv.2 x (hpre.2.2.2.2.2 x hx))
        · rw [cGs_img]
          intro x hx
          exact hx
        · intro m' hread hframe hpres
          have hframe' : ∀ x, x ∉ tileSet d T → m' x = m₁ x := by
            intro x hx
            exact hframe x (by rw [cGs_img]; exact hx)
          have hvals : ∀ (iX : Fin 128) (nX : Fin 256),
              FloatV.NaNEq (FloatV.cvt (st.get2 iX nX)) (c.get2 iX nX) := by
            intro iX nX
            have h := hc (⟨iX, nX, ()⟩ : Idx [128, 256])
            rw [FloatV.cvtWith_rn] at h
            simpa only [Tensor.get2] using h
          have hread' : ∀ (iX : Fin 128) (nX : Fin 256),
              m'.read .f16 (d.c.ptr (((pm d T * 128 + iX.val) * d.stride_cm +
                (pn d T * 256 + nX.val) * d.stride_cn : Nat) : Int)).addr = some (c.get2 iX nX) := by
            intro iX nX
            have h := hread (⟨iX, nX, ()⟩ : Idx [128, 256]) rfl
            simpa only [cGs, cG, Tensor.get2] using h
          have hpost : (tidc + (IntV.ofNat d.NUM_SMS : I32)).toInt = (T : Int) ∧
              MemInv d init m' ∧ Done d init i (t + 1) m' := by
            refine ⟨hnewtc, ?_, ?_⟩
            · refine memInv_store hd hTlt hminv hframe' hpres
            · have hframe2 : ∀ x, x ∉ tileSet d T → m' x = m x := by
                intro x hx
                rw [hframe' x hx, hm₁]
              refine done_succ hTdef hdone hdisj hframe2 ?_
              intro a b ha hb
              exact outspec_body hd hcell hvals hread' a b ha hb
          exact (safe_ret hpost).mono (W := ∅) (fun a m h => h) (Set.empty_subset _)
  · rw [iterW_empty]
    simp

end Claims.PersistentMatmul
