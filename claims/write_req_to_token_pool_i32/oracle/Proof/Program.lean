import Proof.Copy

/-!
# `write_req_to_token_pool_i32`: one program

Program `b` loads its four scalars, runs the prefix copy loop (source lines 76-85), sums
`extend_lens[0 … b - 1]` (lines 88-90) and runs the copy loop of the new locations (lines
92-104). `body_eq` writes the model's body with the two copy loops as `copyProg`.
-/

namespace Claims.WriteReqToTokenPool

open TVA

/-- The program after the prefix loop: the running sum of `extend_lens` (source lines 88-90) and
the copy loop of the new locations (lines 92-104). -/
def restProg (d : Inst) (pid : I32) (req_pool_index pre_len seq_len : I64) : Prog Unit := do
  let cs : I64 := (← tl.convert .i32 .i64 (Tensor.scalar (IntV.ofInt 0 : I32))).item
  let cs ← tl.forRange (IntV.ofInt 0 : I32) pid (IntV.ofInt 1 : I32) cs fun i cs => do
    let e ← tl.load1 (d.extend_lens.ptr 0 + i)
    pure (cs + e)
  copyProg (seq_len - pre_len) (seq_len - pre_len) (fun o => d.out_cache_loc.ptr 0 + cs + o)
    (fun o => d.req_to_token.ptr 0 + req_pool_index * (IntV.ofNat d.max_context_len : I64) +
      o + (bc pre_len : Tensor [512] I64))
    fun _ => pure ()

/-- The model's body, with its two copy loops as `copyProg`. -/
theorem body_eq (d : Inst) (c : Ctx) : (model d).body c = (do
    let req_pool_index ← tl.load1 (d.req_pool_indices.ptr 0 + tl.program_id c 0)
    let pre_len ← tl.load1 (d.prefix_lens.ptr 0 + tl.program_id c 0)
    let seq_len ← tl.load1 (d.seq_lens.ptr 0 + tl.program_id c 0)
    let pp ← tl.load1 (d.prefix_pointers.ptr 0 + tl.program_id c 0)
    copyProg pre_len pre_len (fun o => tl.intToPtr .i64 pp + o)
      (fun o => d.req_to_token.ptr 0 + req_pool_index * (IntV.ofNat d.max_context_len : I64) + o)
      fun _ => restProg d (tl.program_id c 0) req_pool_index pre_len seq_len) := rfl

end Claims.WriteReqToTokenPool

namespace Claims.WriteReqToTokenPool

open TVA

/-! ## Values -/

/-- The running sums of `extend_lens` are in `[0, t·2^22]`. -/
theorem cum_bounds {d : Inst} {init : Mem} (hpre : Pre d init) :
    ∀ t, t ≤ d.bs → 0 ≤ cum d init t ∧ cum d init t ≤ (t : Int) * 2 ^ 22 := by
  intro t
  induction t with
  | zero => intro _; simp [cum]
  | succ t ih =>
    intro ht
    obtain ⟨h0, h1⟩ := ih (by omega)
    obtain ⟨e0, e1⟩ := hpre.ext t (by omega)
    have hs : cum d init (t + 1) = cum d init t + elen d init t := Finset.sum_range_succ _ _
    rw [hs]
    push_cast
    constructor <;> linarith

/-- The pieces of the read footprint. -/
theorem R_rpi (d : Inst) (init : Mem) (b : Nat) : d.req_pool_indices.elems {b} ⊆ R d init b :=
  fun z hz => by simp only [R, Set.mem_union]; tauto
theorem R_pp (d : Inst) (init : Mem) (b : Nat) : d.prefix_pointers.elems {b} ⊆ R d init b :=
  fun z hz => by simp only [R, Set.mem_union]; tauto
theorem R_pre (d : Inst) (init : Mem) (b : Nat) : d.prefix_lens.elems {b} ⊆ R d init b :=
  fun z hz => by simp only [R, Set.mem_union]; tauto
theorem R_seq (d : Inst) (init : Mem) (b : Nat) : d.seq_lens.elems {b} ⊆ R d init b :=
  fun z hz => by simp only [R, Set.mem_union]; tauto
theorem R_ext (d : Inst) (init : Mem) (b : Nat) : d.extend_lens.elems {i | i < b} ⊆ R d init b :=
  fun z hz => by simp only [R, Set.mem_union]; tauto
theorem R_prefix (d : Inst) (init : Mem) (b : Nat) :
    (d.prefixT b).elems (win 0 (pre d init b)) ⊆ R d init b :=
  fun z hz => by simp only [R, Set.mem_union]; tauto
theorem R_out (d : Inst) (init : Mem) (b : Nat) :
    d.out_cache_loc.elems (win (cum d init b) (seq d init b - pre d init b)) ⊆ R d init b :=
  fun z hz => by simp only [R, Set.mem_union]; tauto

/-- A scalar load of element `i` of a valid buffer at `ptr 0 + i`. -/
theorem load_at {dt : DType} {b : Buf dt} {m : Mem} (hv : b.Valid m) {i : Nat} (hi : i < b.len)
    (hi31 : i < 2 ^ 31) :
    ∃ v, m.read dt (b.ptr 0 + (IntV.ofNat i : I32)).addr = some v ∧
      m.read dt (b.ptr (i : Int)).addr = some v := by
  obtain ⟨v, hv'⟩ := b.read_some hv hi
  refine ⟨v, ?_, hv'⟩
  rw [ptr_zero_add, IntV.I32.toInt_ofNat hi31]
  exact hv'

/-- The span of `ptr 0 + i` is that of element `i`. -/
theorem span_at {dt : DType} (b : Buf dt) {i : Nat} (hi31 : i < 2 ^ 31) :
    Mem.span dt (b.ptr 0 + (IntV.ofNat i : I32)).addr ⊆ b.elems {i} := by
  rw [ptr_zero_add, IntV.I32.toInt_ofNat hi31]
  exact b.span_sub_elems rfl

/-- Two windows one after the other do not meet. -/
theorem win_disjoint_next (a p L : Int) : Disjoint (win a p) (win (a + p) L) := by
  rw [Set.disjoint_left]
  rintro e ⟨o, ho, he⟩ ⟨o', -, he'⟩
  have : (0 : Int) ≤ o' := by positivity
  linarith

/-- Two windows one after the other make the window of their total length. -/
theorem win_union_sub (a p L : Int) (hp : 0 ≤ p) (hL : 0 ≤ L) : win a p ∪ win (a + p) L ⊆ win a (p + L) := by
  rintro e (⟨o, ho, he⟩ | ⟨o', ho', he'⟩)
  · exact ⟨o, by linarith, he⟩
  · refine ⟨(p + o').toNat, ?_, ?_⟩
    · rw [Int.toNat_of_nonneg (by positivity)]; linarith
    · rw [Int.toNat_of_nonneg (by positivity)]; linarith

/-! ## The walk -/

/-- The table's length is at most `2^62` (its bytes lie in `[0, 2^64)`). -/
theorem table_len_lt {d : Inst} {m : Mem} (hv : d.req_to_token.Valid m) :
    (d.req_to_token.len : Int) ≤ 2 ^ 62 := by
  obtain ⟨h0, h1, -, -⟩ := hv
  have : (DType.bytes .i32 : Int) = 4 := by simp [DType.bytes]
  rw [this] at h1
  linarith

/-- `req_pool_indices[b] * max_context_len`, the row's first element, in `int64`. -/
theorem rowbase_toInt {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) {b : Nat}
    (hb : b < d.bs) {vr : I64} (hr : vr.toInt = rpi d init b) :
    (vr * (IntV.ofNat d.max_context_len : I64)).toInt = rpi d init b * (d.max_context_len : Int) := by
  obtain ⟨hp0, hps, hsm, hr0, hrl⟩ := hpre.req b hb
  have hlen := table_len_lt hpre.table_valid
  have hm : (IntV.ofNat d.max_context_len : I64).toInt = (d.max_context_len : Int) :=
    IntV.I64.toInt_ofNat (by have := hd.mcl_le; omega)
  have hnn : 0 ≤ rpi d init b * (d.max_context_len : Int) := by positivity
  exact IntV.toInt_mul hr hm (IntV.I64.sfits (by linarith) (by linarith))

/-- **The rest of the program** after the prefix loop: the running sum and the second copy loop,
from a memory `m'` that agrees with `init` outside the table and holds the prefix. -/
theorem safe_rest {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env) {b : Nat}
    (hb : b < d.bs) {vr vp vs : I64} (hr : vr.toInt = rpi d init b) (hp : vp.toInt = pre d init b)
    (hs : vs.toInt = seq d init b) {m' : Mem}
    (hfr : ∀ z, z ∉ d.req_to_token.bytes → m' z = init z) (hTv : d.req_to_token.Valid m')
    (hQ1 : ∀ o : Nat, (o : Int) < pre d init b →
      Copied d init m' (d.prefixT b) 0 (rpi d init b * (d.max_context_len : Int)) o) :
    Safe E (R d init b)
      (d.req_to_token.elems
        (win (rpi d init b * (d.max_context_len : Int) + pre d init b) (seq d init b - pre d init b)))
      (fun _ m => Q d init b m) m' (restProg d (IntV.ofNat b) vr vp vs) := by
  have hb31 : b < 2 ^ 31 := by have := hd.bs_le; omega
  obtain ⟨hp0, hps, hsm, hr0, hrl⟩ := hpre.req b hb
  have hmcl := hd.mcl_le
  unfold restProg
  refine safe_convert (fun _ => ⟨_, rfl⟩) fun c0 hc0 => ?_
  have hc0' : c0.item.toInt = 0 := by
    have h := ConvRel.toInt_i32_i64 (hc0 ())
    rw [show c0.item = c0.get () from rfl, h]
    exact IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have hlo : (IntV.ofInt 0 : I32).toInt = 0 := IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have hst : (IntV.ofInt 1 : I32).toInt = 1 := IntV.I32.toInt_ofInt ⟨by norm_num, by norm_num⟩
  have hbv : (IntV.ofNat b : I32).toInt = b := IntV.I32.toInt_ofNat hb31
  have hN : tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat b : I32).toInt
      (IntV.ofInt 1 : I32).toInt = b := by
    simp only [tl.tripCount, hlo, hst, hbv]
    simp
  have hrange : tl.RangeOk (IntV.ofInt 0 : I32) (IntV.ofNat b : I32) (IntV.ofInt 1 : I32) := by
    have hir : IntV.maxInt true 32 = 2 ^ 31 - 1 := by norm_num [IntV.maxInt]
    refine ⟨by rw [hst]; norm_num, ?_, ?_, ?_⟩
    · rw [hbv, hst, hir]; have := hd.bs_le; omega
    · rw [hlo, hir]; norm_num
    · rw [hst, hir]; norm_num
  have hcb := cum_bounds hpre
  let I : Nat → I64 → Mem → Prop := fun t cs m => cs.toInt = cum d init t ∧ m = m'
  rw [show d.req_to_token.elems (win (rpi d init b * (d.max_context_len : Int) + pre d init b)
      (seq d init b - pre d init b)) = iterW (fun _ => (∅ : Set Int))
        (tl.tripCount (IntV.ofInt 0 : I32).toInt (IntV.ofNat b : I32).toInt
          (IntV.ofInt 1 : I32).toInt) ∪
      d.req_to_token.elems (win (rpi d init b * (d.max_context_len : Int) + pre d init b)
        (seq d init b - pre d init b)) by simp]
  refine safe_forRange hrange I (fun _ => ∅) (by simp) (by simp) ⟨by
      rw [hc0']; simp [cum], rfl⟩ ?_ ?_
  · intro t ht cs m₁ ⟨hcs, hm₁⟩
    subst hm₁
    rw [hN] at ht
    have ht31 : t < 2 ^ 31 := by omega
    have hi : (IntV.ofInt ((IntV.ofInt 0 : I32).toInt + (t : Int) * (IntV.ofInt 1 : I32).toInt) :
        I32) = (IntV.ofNat t : I32) := by
      apply IntV.toInt_inj.mp
      rw [hlo, hst, IntV.I32.toInt_ofNat ht31, IntV.I32.toInt_ofInt ⟨by omega, by omega⟩]
      ring
    rw [hi]
    have htl : t < d.extend_lens.len := by have := hd.elen; omega
    obtain ⟨ve, hve, hve'⟩ := load_at hpre.ext_valid htl ht31
    have hdisj := hd.disj.2.2.2.2.1
    have hread : m₁.read .i64 (d.extend_lens.ptr 0 + (IntV.ofNat t : I32)).addr = some ve := by
      rw [ptr_zero_add, IntV.I32.toInt_ofNat ht31,
        read_init (d := d) d.extend_lens hdisj hfr (by positivity) (by exact_mod_cast htl)]
      exact hve'
    refine safe_load1 hread ((span_at d.extend_lens ht31).trans
      ((d.extend_lens.elems_mono (fun j (hj : j ∈ ({t} : Set Nat)) => by
        rw [Set.mem_singleton_iff.mp hj]; exact ht)).trans (R_ext d init b))) ?_
    refine safe_ret ⟨?_, rfl⟩
    have he : ve.toInt = elen d init t := (valOf_of_read hve').symm
    obtain ⟨c0', c1'⟩ := hcb t (by omega)
    obtain ⟨e0, e1⟩ := hpre.ext t (by omega)
    have hsum : cum d init (t + 1) = cum d init t + elen d init t := Finset.sum_range_succ _ _
    rw [hsum]
    have h8 : (t : Int) ≤ 8192 := by have := hd.bs_le; omega
    exact IntV.toInt_add hcs he (IntV.I64.sfits (by linarith) (by nlinarith))
  · intro cs m₁ ⟨hcs, hm₁⟩
    subst hm₁
    rw [hN] at hcs
    obtain ⟨c0', c1'⟩ := hcb b (by omega)
    have hb8 : (b : Int) ≤ 8192 := by have := hd.bs_le; omega
    have hL : (vs - vp).toInt = seq d init b - pre d init b :=
      IntV.toInt_sub hs hp (IntV.I64.sfits (by linarith) (by linarith))
    have hwin := hpre.window b hb
    have hlen := table_len_lt hpre.table_valid
    have hrb := rowbase_toInt hd hpre hb hr
    have hrow0 : 0 ≤ rpi d init b * (d.max_context_len : Int) := by positivity
    refine Safe.mono (Q := fun _ m => Q d init b m) (W := d.req_to_token.elems
        (win (rpi d init b * (d.max_context_len : Int) + pre d init b)
          (seq d init b - pre d init b)) ∪ ∅)
      (safe_copy (L := seq d init b - pre d init b) d.out_cache_loc hL (by linarith)
        (by linarith) ?_ ?_ ?_ ?_ hd.disj.2.2.2.2.2 (fun _ => hpre.out_valid) (R_out d init b)
        hfr hTv (Set.disjoint_empty _) ?_) (fun _ _ h => h) (by simp)
    · intro _ off l
      simp only [Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset, hcs, zero_add]
    · intro off l
      rw [Tensor.get_ptrAddT, Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset, Buf.ptr_offset,
        Buf.ptr_offset, hrb]
      simp only [bc, Tensor.get_full, hp]
      ring_nf
    · intro o ho
      constructor <;> linarith
    · intro o ho
      constructor <;> nlinarith
    · intro m'' hC hF hV
      refine safe_ret ⟨fun o ho => ?_, hC⟩
      obtain ⟨v, hv1, hv2⟩ := hQ1 o ho
      refine ⟨v, hv1, ?_⟩
      rw [read_frame hF ?_]
      · exact hv2
      · rintro j ⟨o', -, hj⟩ hje
        have : (0 : Int) ≤ o' := by positivity
        rw [hj] at hje
        linarith

/-- **One program** of the grid is safe from `init`: it reads its request's entries, its prefix
and its window of `out_cache_loc`, writes its row's window of the table, and ends with `Q`. -/
theorem safe_program {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    (i : Fin (launch d).size) :
    Safe E (R d init (i : Nat)) (W d init (i : Nat)) (fun _ m => Q d init (i : Nat) m) init
      ((model d).body ((launch d).ctx i)) := by
  rw [body_eq]
  simp only [tl.program_id, launch_pid]
  have hb : (i : Nat) < d.bs := lt_of_lt_of_eq i.isLt (launch_size d)
  generalize (i : Nat) = b at hb ⊢
  have hb31 : b < 2 ^ 31 := by have := hd.bs_le; omega
  obtain ⟨vr, hvr, hvr'⟩ := load_at hpre.rpi_valid (by have := hd.rplen; omega) hb31
  refine safe_load1 hvr ((span_at _ hb31).trans (R_rpi d init b)) ?_
  obtain ⟨vp, hvp, hvp'⟩ := load_at hpre.pre_valid (by have := hd.prelen; omega) hb31
  refine safe_load1 hvp ((span_at _ hb31).trans (R_pre d init b)) ?_
  obtain ⟨vs, hvs, hvs'⟩ := load_at hpre.seq_valid (by have := hd.slen; omega) hb31
  refine safe_load1 hvs ((span_at _ hb31).trans (R_seq d init b)) ?_
  obtain ⟨vq, hvq, hvq'⟩ := load_at hpre.pp_valid (by have := hd.pplen; omega) hb31
  refine safe_load1 hvq ((span_at _ hb31).trans (R_pp d init b)) ?_
  have hr : vr.toInt = rpi d init b := (valOf_of_read hvr').symm
  have hp : vp.toInt = pre d init b := (valOf_of_read hvp').symm
  have hs : vs.toInt = seq d init b := (valOf_of_read hvs').symm
  have hq : vq.toInt = uvalOf init d.prefix_pointers b := (uvalOf_of_read hvq').symm
  obtain ⟨hp0, hps, hsm, hr0, hrl⟩ := hpre.req b hb
  have hmcl := hd.mcl_le
  have hrb := rowbase_toInt hd hpre hb hr
  have hrow0 : 0 ≤ rpi d init b * (d.max_context_len : Int) := by positivity
  have hsub : d.req_to_token.elems (win (rpi d init b * (d.max_context_len : Int)) (pre d init b)) ∪
      d.req_to_token.elems (win (rpi d init b * (d.max_context_len : Int) + pre d init b)
        (seq d init b - pre d init b)) ⊆ W d init b := by
    intro z hz
    rw [← Buf.elems_union] at hz
    have h := d.req_to_token.elems_mono (win_union_sub _ _ _ hp0 (by linarith)) hz
    simp only [W, rowElems]
    rwa [add_sub_cancel] at h
  refine Safe.mono (safe_copy (L := pre d init b) (s0 := 0) (d.prefixT b) hp hp0 (by linarith)
    ?_ ?_ ?_ ?_ (hd.prefix_disj b hb) (fun h => (hpre.prefixes b hb h).2.2) (R_prefix d init b)
    (fun _ _ => rfl) hpre.table_valid
    (d.req_to_token.elems_disjoint (win_disjoint_next _ _ _)) ?_) (fun _ _ h => h) hsub
  · intro hpos off l
    obtain ⟨hbase, -, -⟩ := hpre.prefixes b hb hpos
    have e : ((vq.bits.toNat : Nat) : Int) = (d.prefixT b).base := by
      rw [← hbase, ← hq]
      rfl
    simp only [Tensor.get_ptrAdd, tl.intToPtr, Ptr.offset, Buf.ptr, e]
    congr 1
    ring
  · intro off l
    simp only [Tensor.get_ptrAdd, Ptr.add_int, Buf.ptr_offset, hrb, zero_add]
  · intro o ho
    have hpos : 0 < pre d init b := lt_of_le_of_lt (by positivity) ho
    obtain ⟨-, hlen, -⟩ := hpre.prefixes b hb hpos
    constructor <;> linarith
  · intro o ho
    constructor <;> linarith
  · intro m' hC hF hV
    have hwin : d.req_to_token.elems (win (rpi d init b * (d.max_context_len : Int)) (pre d init b))
        ⊆ d.req_to_token.bytes := d.req_to_token.elems_sub_bytes (win_lt fun o ho =>
          ⟨by linarith, by linarith⟩)
    exact safe_rest hd hpre E hb hr hp hs (fun z hz => hF z fun h => hz (hwin h)) hV hC

end Claims.WriteReqToTokenPool
