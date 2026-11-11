import TVA.Proof
import Claim
import Proof.Pid
import Proof.Foot
import Proof.Defs
import Proof.Store
import Proof.Inv
import Proof.Trip
import Proof.Tail

/-!
# One program is safe

`safe_program` walks the model's body for program `i`, in the pieces `Proof.Defs` cuts it into:
the grouped tile (`safe_pidPrefix`), the early return past the padded total (`safe_rest1`), the
block's entries and mask (`safe_rest2`, `safe_tokens_load`, `tokV_naive`), the expert (`safe_restTok`):
expert `-1` stores zeros (`safe_zero`), otherwise the tile pointers (`safe_rest3`) and the `K` loop
with its epilogue (`safe_main`, with `trip_safe` and `tail_safe`).
-/

namespace Claims.FusedMoe

open TVA

variable {P : Platform} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- The value a scalar load of the `num_tokens_post_padded` buffer gives is `ntpOf`. -/
theorem ntp_val {d : Inst} {init : Mem} (hpre : Pre d init) {v : I32}
    (hv : init.read .i32 (d.ntpb.ptr (0 : Nat)).addr = some v) : v.toInt = (ntpOf d init : Int) := by
  obtain ⟨w, hw, h0, h1⟩ := hpre.ntpEx
  have : w = v := Option.some.inj (hw.symm.trans (by simpa using hv))
  subst this
  omega

/-- With the naive assignment the block's entries are the block's index in the first lane and
`num_valid_tokens` in the others (lines 222-226). -/
theorem tokV_naive {d : Inst} (hd : d.ok) (init : Mem) (hn : d.naive = true)
    {pmN : Nat} {pm : I32} (hpm : pm.toInt = (pmN : Int)) (hnv : nv d < 2 ^ 31)
    {offs : Tensor [d.BM] I64} (hoffs : ∀ l : Fin d.BM, (offs.get1 l).toInt = (l : Nat)) :
    TokV d init pmN (tl.where (offs.eq (bc (IntV.ofInt 0 : I64))) (bc pm)
      (bc (IntV.ofNat (nv d) : I32))) := by
  intro l
  have hBM := (cfg_facts hd).1
  have hl := l.isLt
  have hz : ((IntV.ofInt 0 : I64)).toInt = 0 := IntV.toInt_ofInt (by norm_num) (by unfold IntV.SFits; norm_num)
  have hp : (pmN * d.BM + l) % d.BM = l := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]
  have hq : (pmN * d.BM + l) / d.BM = pmN := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hl, Nat.zero_add]
  simp only [entryOf, hn, ite_true, hp, hq]
  have e : (tl.where (offs.eq (bc (IntV.ofInt 0 : I64))) (bc pm) (bc (IntV.ofNat (nv d) : I32))).get1 l =
      if IntV.eq (offs.get1 l) (IntV.ofInt 0 : I64) = true then pm else (IntV.ofNat (nv d) : I32) := rfl
  rw [e]
  have h1 : (offs.get1 l).toInt = ((l : Nat) : Int) := hoffs l
  have hc : (IntV.eq (offs.get1 l) (IntV.ofInt 0 : I64) = true) ↔ (l : Nat) = 0 := by
    rw [IntV.eq_eq, decide_eq_true_iff, hz]
    constructor <;> intro h <;> omega
  by_cases h0 : (l : Nat) = 0
  · rw [ite_eq_left (hc.mpr h0), ite_eq_left h0]; exact hpm
  · rw [ite_eq_right (fun h => h0 (hc.mp h)), ite_eq_right h0]
    exact_mod_cast IntV.I32.toInt_ofNat hnv

/-- Without it, the block's entries are loaded from `sorted_token_ids` (lines 219-220). -/
theorem safe_tokens_load {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    {R W : Set Int} {Q : Unit → Mem → Prop} (hR : d.s.bytes ⊆ R) (hn : d.naive = false)
    {pmN : Nat} {pm : I32} (hpm : pm.toInt = (pmN : Int))
    (hmul : pmN * d.BM + d.BM ≤ ntpOf d init) (hpmBM : pmN * d.BM + d.BM < 2 ^ 31)
    {offs : Tensor [d.BM] I64} (hoffs : ∀ l : Fin d.BM, (offs.get1 l).toInt = (l : Nat))
    {k : Tensor [d.BM] I32 → Prog Unit}
    (hk : ∀ v : Tensor [d.BM] I32, TokV d init pmN v → Safe E R W Q init (k v)) :
    Safe E R W Q init (tl.convert .i32 .i64 (Tensor.scalar (pm * (IntV.ofNat d.BM : I32))) >>=
      fun c => tl.load (d.s.ptr 0 + (c.item + offs)) >>= k) := by
  have hBM := (cfg_facts hd).1
  have hs_len : d.EM ≤ d.s.len := hd.s_len hn
  have hEM := hpre.ntpEM
  have hBM0 : 0 < d.BM := by omega
  have hpmv : (pm * (IntV.ofNat d.BM : I32)).toInt = ((pmN * d.BM : Nat) : Int) := by
    rw [IntV.toInt_mul rfl rfl (by rw [hpm, i32_ofNat (by omega)]; exact sfits_mul_nat (by omega)),
      hpm, i32_ofNat (by omega)]
    push_cast; ring
  refine safe_conv32 ?_
  have hcv : ((Tensor.scalar (pm * (IntV.ofNat d.BM : I32))).castInt : Tensor [] I64).item.toInt =
      ((pmN * d.BM : Nat) : Int) := by
    show (IntV.cast (pm * (IntV.ofNat d.BM : I32)) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), hpmv]
  have hlane : ∀ l : Fin d.BM, ((((Tensor.scalar (pm * (IntV.ofNat d.BM : I32))).castInt :
      Tensor [] I64).item + offs).get1 l).toInt = ((pmN * d.BM + l : Nat) : Int) := by
    intro l
    rw [Tensor.get1_sadd]
    exact IntV.I64.add_nat hcv (hoffs l) (by omega)
  have hlane' : ∀ i : Idx [d.BM], ((((Tensor.scalar (pm * (IntV.ofNat d.BM : I32))).castInt :
      Tensor [] I64).item + offs).get i).toInt = ((pmN * d.BM + (i.1 : Nat) : Nat) : Int) :=
    fun i => hlane i.1
  refine safe_load_buf d.s (g := fun l : Idx [d.BM] => pmN * d.BM + (l.1 : Nat)) (en := fun _ => true)
    (Buf.addrs_of_get_all d.s ?_) ?_ hpre.vs (fun x hx => hR (d.s.elems_sub_bytes ?_ hx)) fun v hv _ => ?_
  · intro i
    show (d.s.ptr 0).offset (_ : I64).toInt = d.s.ptr ((pmN * d.BM + (i.1 : Nat) : Nat) : Int)
    rw [hlane' i, Buf.ptr_offset]; simp
  · rintro ⟨l, ⟨⟩⟩ _
    have := l.isLt
    show pmN * d.BM + l < d.s.len
    omega
  · rintro j ⟨⟨l, ⟨⟩⟩, _, rfl⟩
    have := l.isLt
    show pmN * d.BM + l < d.s.len
    omega
  · refine hk v fun l => ?_
    have h1 := hv (l, ()) rfl
    have hlt : pmN * d.BM + l < d.s.len := by have := l.isLt; omega
    unfold entryOf
    rw [ite_eq_right (by simp [hn]), ite_eq_left hlt]
    have h2 : init.read .i32 (d.s.ptr ((pmN * d.BM + l : Nat) : Int)).addr = some (v.get1 l) := h1
    rw [h2]

/-- The zeros branch (lines 234-250): expert `-1` stores `bfloat16` zeros to the block's rows. -/
theorem safe_zero {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {E : Env}
    {R W : Set Int} {pmN pnN : Nat} {pn : I32}
    (hpn : pn.toInt = (pnN : Int)) (hpnb : pnN * d.BN + d.BN ≤ d.N + d.BN - 1)
    (hact : pmN * d.BM + d.BM ≤ ntpOf d init)
    {ot : Tensor [d.BM] I64} {tm : Tensor [d.BM] Bool} (hl : TokLanes d init pmN ot tm)
    (hE : eOf d init pmN = -1) (hW : d.c.elems (Wset d init pmN pnN) ⊆ W) :
    Safe E R W (fun _ m => QT d init pmN pnN m) init
      (write_zeros_to_output (d.c.ptr 0) (IntV.ofNat (stride_cm d)) (IntV.ofNat 1) pn (IntV.ofNat d.N)
        (BLOCK_SIZE_M := d.BM) (BLOCK_SIZE_N := d.BN) ot tm >>= fun _ => pure ()) := by
  rw [zero_eq]
  refine safe_storeC hd hpre hpn hpnb hact hl _ hW fun m' hread hframe => ?_
  refine safe_ret ?_
  intro l j hl' hj' hpos hent hc
  have hdiv : (pmN * d.BM + l) / d.BM = pmN := by
    have := (cfg_facts hd).1
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hl', Nat.zero_add]
  refine ⟨_, hread ⟨l, hl'⟩ ⟨j, hj'⟩ hent hc, ?_⟩
  rw [hdiv, ite_eq_left hE]
  exact FloatV.NaNEq.refl _

/-- A block that starts below a multiple of its size lies wholly below it. -/
theorem block_le {pm BM n : Nat} (h : BM ∣ n) (hlt : pm * BM < n) : pm * BM + BM ≤ n := by
  obtain ⟨t, rfl⟩ := h
  have h0 : pm * BM < t * BM := by rwa [Nat.mul_comm t BM]
  have h1 : pm < t := Nat.lt_of_mul_lt_mul_right h0
  have h2 := Nat.mul_le_mul_right BM (Nat.succ_le_of_lt h1)
  rw [Nat.succ_mul] at h2
  rw [Nat.mul_comm BM t]
  exact h2

/-- The loop and the epilogue (lines 330-420) from the tile pointers and the zero accumulator. -/
theorem safe_main {d : Inst} (hd : d.ok) (hP : Device P) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) {pn : I32} (hpn : pn.toInt = (pidN d i : Int))
    (hact : pidM d i * d.BM + d.BM ≤ ntpOf d init)
    {ot : Tensor [d.BM] I64} {tm : Tensor [d.BM] Bool} (hl : TokLanes d init (pidM d i) ot tm)
    {e : Nat} (he : e < d.E) (hE : eOf d init (pidM d i) = (e : Int))
    {A : Tensor [d.BM, d.BK] (Ptr .bf16)} {B : Tensor [d.BK, d.BN] (Ptr .bf16)}
    (hA : ∀ (l : Fin d.BM) (kk : Fin d.BK),
      A.get2 l kk = d.a.ptr ((arow d init (pidM d i) l * d.K + kk : Nat) : Int))
    (hB : ∀ (kk : Fin d.BK) (j : Fin d.BN),
      B.get2 kk j = d.b.ptr ((e * d.stride_be + bcol d (pidN d i) j * d.stride_bn + kk : Nat) : Int)) :
    Safe ((launch d).env P) (Rfp d) (Wfp d init i) (fun _ m => Qfp d init i m) init
      (tl.idiv (Tensor.scalar (IntV.ofNat d.K + (IntV.ofNat (d.BK - 1) : I32)))
          (Tensor.scalar (IntV.ofNat d.BK : I32)) >>= fun r =>
        tl.forRange (IntV.ofInt 0 : I32) r.item (IntV.ofInt 1 : I32)
          (A, B, tl.zeros [d.BM, d.BN] .f32) (tripBody d tm) >>= fun x =>
        match x with | (_a_ptrs, _b_ptrs, accumulator) => tailOf d pn ot tm accumulator) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  obtain ⟨hBM, hBN, hBK, hGM⟩ := cfg_facts hd
  have hBK0 : 0 < d.BK := by omega
  have hsteps := steps_le hBK hd.K_le15
  have hKBK := hd.K_le
  refine safe_idiv1 (w := 32) ?_ ?_ ?_
  · rw [toInt_add_ofNat (a := d.K) (b := d.BK - 1) (by omega)]; positivity
  · rw [i32_ofNat (by omega)]; exact_mod_cast hBK0
  · intro r hr
    have hrt : r.item.toInt = (steps d.BK d.K : Int) := by
      have := cdiv_val hBK0 (by omega) (by omega) hr
      exact this
    have htz : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt (by norm_num)
    have hto : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt (by norm_num)
    have hT : tl.tripCount (IntV.ofInt 0 : I32).toInt r.item.toInt (IntV.ofInt 1 : I32).toInt =
        steps d.BK d.K := by
      show ((r.item.toInt - 0 + 1 - 1) / 1).toNat = steps d.BK d.K
      rw [hrt]
      simp
    rw [show Wfp d init i = (iterW (fun _ => (∅ : Set Int)) (tl.tripCount (IntV.ofInt 0 : I32).toInt
        r.item.toInt (IntV.ofInt 1 : I32).toInt) ∪ Wfp d init i) from by rw [hT]; simp [iterW]]
    refine safe_forRange (I := InInv d init (pidM d i) (pidN d i) e) (Wt := fun _ => (∅ : Set Int))
      (Wk := Wfp d init i) ?hrange ?hWt ?hWk ?h0 ?hstep ?hk
    case hrange =>
      unfold tl.RangeOk
      rw [htz, hrt, hto]
      have hbnd : IntV.maxInt true 32 = 2 ^ 31 - 1 := by simp [IntV.maxInt]
      rw [hbnd]
      refine ⟨by norm_num, ?_, ?_, ?_⟩ <;> omega
    case hWt => intro t u _ _; simp
    case hWk => rw [iterW_empty]; simp
    case h0 =>
      refine ⟨rfl, ?_, ?_, ?_⟩
      · intro l kk; rw [hA]; simp
      · intro kk j; rw [hB]; simp
      · intro l j _ va vb _ _ _
        refine ⟨0, ?_, ?_⟩
        · simp only [tl.zeros, Tensor.get2]
          exact DType.val?_zero_f32
        · simp [preSum, accErr_zero]
    case hstep =>
      intro t htc st m1 hI
      have ht : t < steps d.BK d.K := by rw [← hT]; exact htc
      refine trip_safe hd hP hpre he hact (fun l => hl.2 l) ht ?_ hI
      rw [IntV.I32.toInt_ofInt (by rw [htz, hto]; constructor <;> omega), htz, hto]
      ring
    case hk =>
      intro st m1 hI
      obtain ⟨A', B', C'⟩ := st
      rw [hT] at hI
      obtain ⟨hm1, -, -, hacc⟩ := hI
      subst hm1
      exact tail_safe hd hpre he hE hact hpn (by
        have := pidN_mul_le hd hi
        rw [Nat.add_mul, Nat.one_mul] at this
        exact this) hl hacc

/-- A flat index below the padded total is below `2^31`. -/
theorem entryOf_lt {d : Inst} (hd : d.ok) {init : Mem} {pos : Nat} (hpos : pos < ntpOf d init)
    (hpre : Pre d init) : entryOf d init pos < 2 ^ 31 := by
  have hEM := hpre.ntpEM
  have hEM2 := hd.EM_le
  have hnv : (nv d : Int) < 2 ^ 31 := by exact_mod_cast hd.nv_le
  have hBM := (cfg_facts hd).1
  have hq : ((pos / d.BM : Nat) : Int) < 2 ^ 31 := by
    have : pos / d.BM < 2 ^ 31 := by
      have := Nat.div_le_self pos d.BM
      omega
    exact_mod_cast this
  unfold entryOf
  by_cases h1 : d.naive = true
  · rw [ite_eq_left h1]
    by_cases h2 : pos % d.BM = 0
    · rw [ite_eq_left h2]; exact hq
    · rw [ite_eq_right h2]; exact hnv
  · rw [ite_eq_right h1]
    by_cases h2 : pos < d.s.len
    · rw [ite_eq_left h2]
      cases init.read .i32 (d.s.ptr (pos : Int)).addr with
      | none => exact hnv
      | some v =>
        have := (IntV.sfits_toInt v).2
        simpa using this
    · rw [ite_eq_right h2]; exact hnv

/-- A product of naturals below `2^63` fits `int64`. -/
theorem sfits64_mul_nat {a b : Nat} (h : a * b < 2 ^ 63) : IntV.SFits 64 ((a : Int) * (b : Int)) := by
  have e : (a : Int) * (b : Int) = ((a * b : Nat) : Int) := by push_cast; ring
  rw [e]; exact IntV.I64.sfits_natCast h

/-- The main branch: expert `e < E`. -/
theorem safe_rest3 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (hP : Device P)
    (i : Fin (launch d).size) {pn : I32} (hpn : pn.toInt = (pidN d i : Int))
    (hact : pidM d i * d.BM + d.BM ≤ ntpOf d init)
    {ot : Tensor [d.BM] I64} {tm : Tensor [d.BM] Bool} (hl : TokLanes d init (pidM d i) ot tm)
    {oe : I64} {e : Nat} (he : e < d.E) (hoe : oe.toInt = (e : Int))
    (hE : eOf d init (pidM d i) = (e : Int)) :
    Safe ((launch d).env P) (Rfp d) (Wfp d init i) (fun _ m => Qfp d init i m) init
      (rest3 d pn ot tm oe) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  obtain ⟨hBM, hBN, hBK, hGM⟩ := cfg_facts hd
  have hN := hd.N_pos
  have hN17 := hd.N_le17
  have hK15 := hd.K_le15
  have hK0 := hd.K_pos
  have hKBK := hd.K_le
  have hNle := hd.N_le
  have htop := hd.topk_pos
  have hpnb : pidN d i * d.BN + d.BN ≤ d.N + d.BN - 1 := by
    have h1 := pidN_mul_le hd hi
    rw [Nat.add_mul, Nat.one_mul] at h1
    exact h1
  have htopl := hd.topk_le
  unfold rest3
  dsimp only
  refine safe_conv32 ?_
  refine safe_conv32 ?_
  refine safe_conv32 ?_
  have hn64 : ((Tensor.scalar (IntV.ofNat d.N : I32)).castInt : Tensor [] I64).item.toInt = (d.N : Int) := by
    show (IntV.cast (IntV.ofNat d.N : I32) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), i32_ofNat (by omega)]
  have hpnBN : (pn * (IntV.ofNat d.BN : I32)).toInt = ((pidN d i * d.BN : Nat) : Int) := by
    rw [IntV.toInt_mul rfl rfl (by rw [hpn, i32_ofNat (by omega)]; exact sfits_mul_nat (by omega)),
      hpn, i32_ofNat (by omega)]
    push_cast; ring
  have harange : ∀ j : Fin d.BN, (((tl.arange 0 d.BN).castInt : Tensor [d.BN] I64).get1 j).toInt = ((j : Nat) : Int) := by
    intro j
    show (IntV.cast ((tl.arange 0 d.BN).get1 j) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num)]
    exact IntV.I32.toInt_tl_arange (n := d.BN) (by omega) j
  have hcn : ((Tensor.scalar (pn * (IntV.ofNat d.BN : I32))).castInt : Tensor [] I64).item.toInt =
      ((pidN d i * d.BN : Nat) : Int) := by
    show (IntV.cast (pn * (IntV.ofNat d.BN : I32)) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), hpnBN]
  refine Tile2.safe_irem_lanes (f := fun j : Fin d.BN => pidN d i * d.BN + (j : Nat)) (M := d.N)
    (fun j => Tile2.start_add_toInt hcn (fun j => harange j) (by omega) j)
    (fun j => hn64) hN fun offs_bn hbn => ?_
  have hent0 : ∀ l : Fin d.BM, 0 ≤ entryOf d init (pidM d i * d.BM + l) := fun l =>
    hpre.s1 (pidM d i * d.BM + l) (by have := l.isLt; omega)
  have hrow : ∀ l : Fin d.BM, (ot.get1 l).toInt =
      (((entryOf d init (pidM d i * d.BM + l)).toNat : Nat) : Int) :=
    fun l => by rw [hl.1 l]; exact (Int.toNat_of_nonneg (hent0 l)).symm
  have hentlt : ∀ l : Fin d.BM, (entryOf d init (pidM d i * d.BM + l)).toNat < 2 ^ 31 := fun l => by
    have h1 := entryOf_lt hd (pos := pidM d i * d.BM + l) (by have := l.isLt; omega) hpre
    have h2 := hent0 l
    omega
  refine safe_idiv (idiv_ok (f := fun l : Fin d.BM => (entryOf d init (pidM d i * d.BM + l)).toNat)
    hrow (M := d.top_k) (fun l => IntV.I64.toInt_ofNat (by omega)) htop) fun q hq => ?_
  refine safe_conv32 ?_
  refine safe_conv32 ?_
  refine safe_conv32 ?_
  refine safe_conv32 ?_
  refine safe_conv32 ?_
  have hqv : ∀ l : Fin d.BM, (q.get1 l).toInt = ((arow d init (pidM d i) l : Nat) : Int) := fun l =>
    idiv_toInt (f := fun l : Fin d.BM => (entryOf d init (pidM d i * d.BM + l)).toNat) (M := d.top_k)
      hrow (y := bc (IntV.ofNat d.top_k : I64)) (fun l => IntV.I64.toInt_ofNat (n := d.top_k) (by omega)) htop hq l
  have hK64 : ((Tensor.scalar (IntV.ofNat d.K : I32)).castInt : Tensor [] I64).item.toInt = (d.K : Int) := by
    show (IntV.cast (IntV.ofNat d.K : I32) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), i32_ofNat (by omega)]
  have harow : ∀ l : Fin d.BM, arow d init (pidM d i) l < 2 ^ 31 := fun l => by
    have := hentlt l
    unfold arow
    have := Nat.div_le_self (entryOf d init (pidM d i * d.BM + l)).toNat d.top_k
    omega
  have hk1 : (IntV.ofNat 1 : I32).toInt = 1 := i32_ofNat (by norm_num)
  have hsbe : ((Tensor.scalar (IntV.ofNat d.stride_be : I32)).castInt : Tensor [] I64).item.toInt =
      (d.stride_be : Int) := by
    show (IntV.cast (IntV.ofNat d.stride_be : I32) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), i32_ofNat hd.be]
  have hsbn : ((Tensor.scalar (IntV.ofNat d.stride_bn : I32)).castInt : Tensor [] I64).item.toInt =
      (d.stride_bn : Int) := by
    show (IntV.cast (IntV.ofNat d.stride_bn : I32) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), i32_ofNat hd.bn]
  refine safe_main hd hP hpre i hpn hact hl he hE ?_ ?_
  · intro l kk
    have hx : ∀ (l : Fin d.BM) (kk : Fin d.BK), arow d init (pidM d i) l < 2 ^ 31 →
        ((((q * ((Tensor.scalar (IntV.ofNat d.K : I32)).castInt : Tensor [] I64).item).col d.BK) :
          Tensor [d.BM, d.BK] I64).get2 l kk).toInt = ((arow d init (pidM d i) l * d.K : Nat) : Int) := by
      intro l kk hl'
      show ((q * ((Tensor.scalar (IntV.ofNat d.K : I32)).castInt : Tensor [] I64).item).get1 l).toInt = _
      rw [Tensor.toInt_get1_muls (c := (d.K : Int)) (f := fun l => ((arow d init (pidM d i) l : Nat) : Int))
        hK64 hqv (fun l => sfits64_mul_nat (by have h := Nat.mul_le_mul (le_of_lt (harow l)) hK15; norm_num at h ⊢; omega)) l]
      push_cast; ring
    have hy : ∀ (l : Fin d.BM) (kk : Fin d.BK), (kk : Nat) < d.BK →
        (((((tl.arange 0 d.BK).row d.BM * (IntV.ofNat 1 : I32)).castInt : Tensor [d.BM, d.BK] I64)).get2 l kk).toInt =
          (((kk : Nat) * 1 : Nat) : Int) := by
      intro l kk hk
      show (IntV.cast (((tl.arange 0 d.BK).row d.BM * (IntV.ofNat 1 : I32)).get2 l kk) : I64).toInt = _
      rw [IntV.toInt_cast _ (by norm_num)]
      exact Tile2.rowMul_toInt_on (κ := fun kk : Fin d.BK => (kk : Nat)) (N := d.BK)
        (fun j => IntV.I32.toInt_tl_arange (n := d.BK) (by omega) j) hk1 (by norm_num; omega) l kk hk
    have := Tile2.ptr_lane_sum_of_on d.a (e := 0) (base := d.a.ptr 0) (by simp [Buf.ptr])
      (ρ := fun l : Fin d.BM => arow d init (pidM d i) l) (κ := fun kk : Fin d.BK => (kk : Nat))
      (M := 2 ^ 31) (N := d.BK) (s0 := d.K) (s1 := 1)
      (fun l kk hl' => hx l kk hl') (fun l kk hk => hy l kk hk)
      (by have h := Nat.mul_le_mul (Nat.sub_le (2 ^ 31) 1) hK15; norm_num at h ⊢; omega) l kk (harow l) kk.isLt
    refine this.trans ?_
    congr 1
    push_cast; ring
  · intro kk j
    have hbase : d.b.ptr 0 + oe * ((Tensor.scalar (IntV.ofNat d.stride_be : I32)).castInt : Tensor [] I64).item =
        d.b.ptr ((e * d.stride_be : Nat) : Int) := by
      show (d.b.ptr 0).offset (oe * ((Tensor.scalar (IntV.ofNat d.stride_be : I32)).castInt : Tensor [] I64).item).toInt = _
      have h := hd.b_off e 0 0 he hN (by omega)
      rw [IntV.I64.mul_nat hoe hsbe (by omega)]
      simp [Buf.ptr, Ptr.offset]
    have hx : ∀ (kk : Fin d.BK) (j : Fin d.BN), (kk : Nat) < d.BK →
        ((((tl.arange 0 d.BK).col d.BN * (IntV.ofNat 1 : I32)).castInt : Tensor [d.BK, d.BN] I64).get2 kk j).toInt =
          (((kk : Nat) * 1 : Nat) : Int) := by
      intro kk j hk
      show (IntV.cast (((tl.arange 0 d.BK).col d.BN * (IntV.ofNat 1 : I32)).get2 kk j) : I64).toInt = _
      rw [IntV.toInt_cast _ (by norm_num)]
      exact Tile2.colMul_toInt_on (ρ := fun kk : Fin d.BK => (kk : Nat)) (M := d.BK)
        (fun j => IntV.I32.toInt_tl_arange (n := d.BK) (by omega) j) hk1 (by norm_num; omega) kk j hk
    have hy : ∀ (kk : Fin d.BK) (j : Fin d.BN), bcol d (pidN d i) j < d.N →
        ((offs_bn.row d.BK * ((Tensor.scalar (IntV.ofNat d.stride_bn : I32)).castInt : Tensor [] I64).item).get2 kk j).toInt =
          ((bcol d (pidN d i) j * d.stride_bn : Nat) : Int) := by
      intro kk j hj
      have h := hd.b_off 0 (d.N - 1) (d.BK - 1) hd.E_pos (by omega) (by omega)
      exact Tile2.rowMul_toInt_on (κ := fun j : Fin d.BN => bcol d (pidN d i) j) (N := d.N)
        hbn hsbn (by omega) kk j hj
    have := Tile2.ptr_lane_sum_of_on d.b (e := e * d.stride_be) (base := d.b.ptr 0 + oe * ((Tensor.scalar (IntV.ofNat d.stride_be : I32)).castInt : Tensor [] I64).item)
      hbase.symm.symm (ρ := fun kk : Fin d.BK => (kk : Nat)) (κ := fun j : Fin d.BN => bcol d (pidN d i) j)
      (M := d.BK) (N := d.N) (s0 := 1) (s1 := d.stride_bn)
      (fun kk j hk => hx kk j hk) (fun kk j hj => hy kk j hj)
      (by have h := hd.b_off 0 (d.N - 1) (d.BK - 1) hd.E_pos (by omega) (by omega); norm_num; omega)
      kk j kk.isLt (Nat.mod_lt _ hN)
    refine this.trans ?_
    congr 1
    push_cast; ring


/-- Lines 227-250: the widened entries, the mask, the expert; the zeros branch or `rest3`. -/
theorem safe_restTok {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (hP : Device P)
    (i : Fin (launch d).size) {pm pn : I32} (hpm : pm.toInt = (pidM d i : Int))
    (hpn : pn.toInt = (pidN d i : Int)) (hact : pidM d i * d.BM < ntpOf d init)
    {tok : Tensor [d.BM] I32} (htok : TokV d init (pidM d i) tok) :
    Safe ((launch d).env P) (Rfp d) (Wfp d init i) (fun _ m => Qfp d init i m) init
      (restTok d pm pn tok) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  obtain ⟨hBM, hBN, hBK, hGM⟩ := cfg_facts hd
  have hact' : pidM d i * d.BM + d.BM ≤ ntpOf d init := by
    exact block_le hpre.ntpBM hact
  have hnv := hd.nv_le
  unfold restTok
  refine safe_conv32 ?_
  refine safe_conv32 ?_
  have hnv64 : ((Tensor.scalar (IntV.ofNat (nv d) : I32)).castInt : Tensor [] I64).item.toInt =
      (nv d : Int) := by
    show (IntV.cast (IntV.ofNat (nv d) : I32) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num), i32_ofNat hnv]
  have hl : TokLanes d init (pidM d i) tok.castInt
      (tok.castInt.ltS ((Tensor.scalar (IntV.ofNat (nv d) : I32)).castInt : Tensor [] I64).item) := by
    refine ⟨fun l => ?_, fun l => ?_⟩
    · show (IntV.cast (tok.get1 l) : I64).toInt = _
      rw [IntV.toInt_cast _ (by norm_num), htok l]
    · show IntV.lt (IntV.cast (tok.get1 l) : I64) _ = _
      rw [IntV.lt_eq, IntV.toInt_cast _ (by norm_num), htok l, hnv64]
  have hptr : d.eid.ptr 0 + pm = d.eid.ptr (((pidM d i : Nat) : Nat) : Int) := by
    show (d.eid.ptr 0).offset pm.toInt = _
    rw [hpm, Buf.ptr_offset]; simp
  rw [hptr]
  have helen := hpre.elen (pidM d i) hact
  refine safe_load1_buf d.eid (i := pidM d i) helen hpre.ve
    (fun x hx => Rfp_eid d (d.eid.span_sub helen hx)) fun v hv => ?_
  have hev : eOf d init (pidM d i) = v.toInt := by unfold eOf; rw [hv]
  refine safe_conv32 ?_
  have hoe : ((Tensor.scalar v).castInt : Tensor [] I64).item.toInt = v.toInt := by
    show (IntV.cast v : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num)]
  by_cases hm1 : eOf d init (pidM d i) = -1
  · have hc : IntV.eq ((Tensor.scalar v).castInt : Tensor [] I64).item (IntV.ofInt (-1 : Int) : I64) = true := by
      rw [IntV.eq_eq, decide_eq_true_iff, hoe, ← hev, hm1, IntV.toInt_ofInt (by norm_num)
        (by unfold IntV.SFits; norm_num)]
    rw [ite_eq_left hc]
    refine Safe.mono (safe_zero hd hpre hpn (pnN := pidN d i) ?_ hact' hl hm1 le_rfl) (fun _ _ h => h) le_rfl
    · have := pidN_mul_le hd hi
      rw [Nat.add_mul, Nat.one_mul] at this
      exact this
  · obtain ⟨e, hee, hlt⟩ : ∃ e : Nat, eOf d init (pidM d i) = (e : Int) ∧ e < d.E := by
      rcases hpre.erange (pidM d i) hact with h | ⟨z, hz, hzl⟩
      · exact absurd h hm1
      · exact ⟨z, hz, hzl⟩
    have hc : IntV.eq ((Tensor.scalar v).castInt : Tensor [] I64).item (IntV.ofInt (-1 : Int) : I64) = false := by
      rw [IntV.eq_eq, decide_eq_false_iff_not, hoe, ← hev, hee, IntV.toInt_ofInt (by norm_num)
        (by unfold IntV.SFits; norm_num)]
      omega
    rw [ite_eq_right (by rw [hc]; simp)]
    exact safe_rest3 hd hpre hP i hpn hact' hl hlt (by rw [hoe, ← hev, hee]) hee

/-- Past the early return: the block `pm` is below the padded total. -/
theorem safe_rest2 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (hP : Device P)
    (i : Fin (launch d).size) {pm pn : I32} (hpm : pm.toInt = (pidM d i : Int))
    (hpn : pn.toInt = (pidN d i : Int)) (hact : pidM d i * d.BM < ntpOf d init)
    {offs : Tensor [d.BM] I64} (hoffs : ∀ l : Fin d.BM, (offs.get1 l).toInt = (l : Nat)) :
    Safe ((launch d).env P) (Rfp d) (Wfp d init i) (fun _ m => Qfp d init i m) init
      (rest2 d pm pn offs) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  have hnv := hd.nv_le
  have hmul := block_le hpre.ntpBM hact
  unfold rest2
  dsimp only
  by_cases hn : d.naive = true
  · simp only [hn, Bool.not_true, Bool.false_eq_true, ite_false, pure_bind]
    exact safe_restTok hd hpre hP i hpm hpn hact (tokV_naive hd init hn hpm hnv hoffs)
  · have hn' : d.naive = false := by simpa using hn
    simp only [hn', Bool.not_false, ite_true]
    refine safe_tokens_load hd hpre (Rfp_s d) hn' hpm hmul ?_ hoffs fun v hv => ?_
    · have h1 := pidM_mul_le hd hi
      have h2 := hd.EM_le
      have h3 := (cfg_facts hd).1
      rw [Nat.add_mul, Nat.one_mul] at h1
      omega
    · exact safe_restTok hd hpre hP i hpm hpn hact hv

/-- The early return and the block offsets. -/
theorem safe_rest1 {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (hP : Device P)
    (i : Fin (launch d).size) {pm pn : I32} (hpm : pm.toInt = (pidM d i : Int))
    (hpn : pn.toInt = (pidN d i : Int)) :
    Safe ((launch d).env P) (Rfp d) (Wfp d init i) (fun _ m => Qfp d init i m) init
      (rest1 d pm pn) := by
  have hi : (i : Nat) < grid d := lt_of_lt_of_eq i.isLt (launch_size d)
  obtain ⟨hBM, hBN, hBK, hGM⟩ := cfg_facts hd
  have hmul := pidM_mul_le hd hi
  unfold rest1
  refine safe_conv32 ?_
  refine safe_load1_buf d.ntpb (i := 0) hd.ntp_len hpre.vn
    (fun x hx => Rfp_ntp d (d.ntpb.span_sub hd.ntp_len hx)) fun ntp hntp => ?_
  have hnt : ntp.toInt = (ntpOf d init : Int) := ntp_val hpre hntp
  have hpmBM : (pm * (IntV.ofNat d.BM : I32)).toInt = ((pidM d i * d.BM : Nat) : Int) := by
    rw [IntV.toInt_mul rfl rfl (by
      rw [hpm, i32_ofNat (by omega)]
      refine sfits_mul_nat ?_
      have h1 := hmul
      have h2 := hd.EM_le
      rw [Nat.add_mul, Nat.one_mul] at h1
      omega), hpm, i32_ofNat (by omega)]
    push_cast; ring
  by_cases hle : IntV.le ntp (pm * (IntV.ofNat d.BM : I32)) = true
  · rw [ite_eq_left hle]
    refine safe_ret ?_
    rw [IntV.le_eq, decide_eq_true_iff, hnt, hpmBM] at hle
    intro l j _ _ hpos
    omega
  · rw [ite_eq_right hle]
    rw [IntV.le_eq, decide_eq_true_iff, hnt, hpmBM] at hle
    refine safe_rest2 hd hpre hP i hpm hpn (by omega) ?_
    intro l
    show (IntV.cast ((tl.arange 0 d.BM).get1 l) : I64).toInt = _
    rw [IntV.toInt_cast _ (by norm_num)]
    exact IntV.I32.toInt_tl_arange (n := d.BM) (by omega) l

/-- **One program is safe**: it reads the input buffers, writes its tile of `C`, and leaves
`outElem` in every element of the tile. -/
theorem safe_program {d : Inst} (hd : d.ok) (hP : Device P) {init : Mem} (hpre : Pre d init)
    (i : Fin (launch d).size) :
    Safe ((launch d).env P) (Rfp d) (Wfp d init i) (fun _ m => Qfp d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  rw [body_eq]
  exact safe_pidPrefix hd i fun pm pn hpm hpn => safe_rest1 hd hpre hP i hpm hpn

end Claims.FusedMoe
