import Proof.Trip

/-!
# `slot_mappings_i64`: a request's program

Program `(g, b)`, `b < R` (lines 108-154): the group's scalars, the request's row and token range,
then the loop of `tripProg` over its tokens.
-/

namespace Claims.SlotMappings

open TVA

/-- The end of a request's program (lines 115-154), after the group's scalars. -/
def reqTail (d : Inst) (g b : Nat) (block_table_ptr : Ptr .i32) (block_table_stride : I64)
    (kv_block_size : I32) (mapping_enabled sharded : Bool) : Prog Unit := do
  let req_state_idx ← tl.load1 (d.idx_mapping.ptr 0 + (IntV.ofNat b : I32))
  let is_real_req : Bool := IntV.le (IntV.ofInt 0 : I64) req_state_idx
  let start_idx ← tl.load1 (d.query_start_loc.ptr 0 + (IntV.ofNat b : I32))
  let end_idx ← tl.load1 (d.query_start_loc.ptr 0 + (IntV.ofNat b : I32) + (IntV.ofInt 1 : I32))
  tl.forRange start_idx end_idx (IntV.ofNat 1024 : I32) () fun i _ =>
    tripProg d g block_table_ptr block_table_stride kv_block_size mapping_enabled sharded
      req_state_idx is_real_req end_idx i

/-- A request's program (lines 108-154), in group `g`. -/
def reqProg (d : Inst) (g b : Nat) : Prog Unit := do
  let block_table_ptr ← _load_ptr (d.block_table_ptrs.ptr 0 + (IntV.ofNat g : I32))
  let block_table_stride ← tl.load1 (d.block_table_strides.ptr 0 + (IntV.ofNat g : I32))
  let kv_block_size ← tl.load1 (d.block_sizes.ptr 0 + (IntV.ofNat g : I32))
  let mapping_enabled ← tl.load1 (d.slot_mapping_enabled.ptr 0 + (IntV.ofNat g : I32))
  let sharded : Bool ← if d.CP_SIZE ≠ 1 then
    tl.load1 (d.dcp_sharded.ptr 0 + (IntV.ofNat g : I32)) else pure false
  reqTail d g b block_table_ptr block_table_stride kv_block_size mapping_enabled sharded

/-- The block table's address, read back as a pointer, is the table. -/
theorem intToPtr_base {b : Buf .i32} (h0 : 0 ≤ b.base) (h1 : b.base < 2 ^ 64) :
    tl.intToPtr .i32 (IntV.ofInt b.base : U64) = b.ptr 0 := by
  simp only [tl.intToPtr, Buf.ptr, Int.zero_mul, add_zero]
  congr 1
  show ((BitVec.ofInt 64 b.base).toNat : Int) = b.base
  rw [BitVec.toNat_ofInt, Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num)),
    Int.emod_eq_of_lt h0 (by norm_num; omega)]

/-- The block table's address read back as a pointer is a multiple of 16 when the address is. -/
theorem hint_base {base : Int} (h16 : (16 : Int) ∣ base) :
    (16 : Int) ∣ HintVal.val (tl.intToPtr .i32 (IntV.ofInt base : U64)) := by
  show (16 : Int) ∣ ((BitVec.ofInt 64 base).toNat : Int)
  rw [BitVec.toNat_ofInt, Int.toNat_of_nonneg (Int.emod_nonneg _ (by norm_num))]
  rw [Int.emod_def]
  exact dvd_sub h16 (Dvd.dvd.mul_right (by norm_num) _)

/-- **The end of a request's program** is safe. -/
theorem safe_reqTail {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {g b : Nat} (hg : g < d.G) (hb : b < d.R) {P : Ptr .i32}
    (hP : 0 < (d.bt g).len → P = (d.bt g).ptr 0) {kv : I32} (hkv : kv.toInt = d.bsz g)
    {sharded : Bool} (hsh : d.CP_SIZE ≠ 1 → sharded = d.sh g) :
    Safe E (Rall d) (Wp d init g b) (fun _ m => Qp d init g b m) init
      (reqTail d g b P (IntV.ofNat (d.bstride g)) kv (d.en g) sharded) := by
  have hR := hd.R_le
  have hNi : (d.N : Int) ≤ 2 ^ 22 := by exact_mod_cast hd.N_le
  have hb32 : ((IntV.ofNat b : I32)).toInt = (b : Int) := toInt_ofNat32 (by omega)
  have hidxl : b < d.idx_mapping.len := by have := hd.idx_len; omega
  have hql : b < d.query_start_loc.len := by have := hd.qsl_len; omega
  have hql1 : b + 1 < d.query_start_loc.len := by have := hd.qsl_len; omega
  obtain ⟨vi, hvi⟩ := Buf.read_some d.idx_mapping hpre.1 hidxl
  obtain ⟨vs, hvs⟩ := Buf.read_some d.query_start_loc hpre.2.1 hql
  obtain ⟨ve, hve⟩ := Buf.read_some d.query_start_loc hpre.2.1 hql1
  simp only [reqTail]
  rw [ptr_zero_add _ hb32]
  refine safe_load1 hvi (fun z hz => rall_idx (span_bytes _ hidxl hz)) ?_
  rw [ptr_zero_add _ hb32]
  refine safe_load1 hvs (fun z hz => rall_qsl (span_bytes _ hql hz)) ?_
  rw [show d.query_start_loc.ptr (b : Int) + (IntV.ofInt 1 : I32) =
      d.query_start_loc.ptr ((b + 1 : Nat) : Int) by
    rw [Ptr.add_int, Buf.ptr_offset, IntV.I32.toInt_ofInt (by norm_num)]; push_cast; rfl]
  refine safe_load1 hve (fun z hz => rall_qsl (span_bytes _ hql1 hz)) ?_
  refine safe_last ?_
  have hreq : vi.toInt = idx d init b := (rd64_of hvi).symm
  have hL : vs.toInt = qsl d init b := (rd32_of hvs).symm
  have hH : ve.toInt = qsl d init (b + 1) := (rd32_of hve).symm
  obtain ⟨hL0, -⟩ := qsl_range hpre (show b ≤ d.R by omega)
  obtain ⟨-, hHN⟩ := qsl_range hpre (show b + 1 ≤ d.R by omega)
  have hW : Wp d init g b = d.slot_mappings.elems (tok d g (qsl d init b) (qsl d init (b + 1))) := by
    simp only [Wp, hiB, hb, ite_true]
  rw [hW]
  have hreal : IntV.le (IntV.ofInt 0 : I64) vi = decide (0 ≤ idx d init b) := by
    rw [IntV.le_eq, IntV.I64.toInt_ofInt (by norm_num), hreq]
  refine Safe.mono (safe_tokLoop hd hg hL hH hL0 hHN (fun x => IntV.ofInt (slot d init g b x))
    (fun t ht m hm => ?_) (inv_init hpre)) ?_ (le_refl _)
  · have hX := trip_var hL hL0 (le_trans hHN hNi) ht
    set X := (qsl d init b + 1024 * t).toNat with hXdef
    have hXc : (X : Int) = qsl d init b + 1024 * t := Int.toNat_of_nonneg (by omega)
    have htw : tripW d g (qsl d init b) (qsl d init (b + 1)) t =
        d.slot_mappings.elems (tok d g X (min (qsl d init (b + 1)) (X + 1024))) := by
      simp only [tripW, hXc]
    rw [htw]
    have := safe_tripProg hd hpre E hg hb hkv rfl hsh
      (toInt_ofNat64 (hd.bstride_lt g hg)) hreq hreal hH hX (by omega) (by omega) hm
      (P := P) (fun hr => hP (by
        have hidx : 0 ≤ idx d init b := by rw [hreal] at hr; exact of_decide_eq_true hr
        have := (row_fits hpre hg hb hidx (by omega)).2.1
        have h0 := (row_fits hpre hg hb hidx (by omega)).1
        exact_mod_cast (show (0 : Int) < (d.bt g).len by omega)))
    refine Safe.mono this ?_ (le_refl _)
    rintro _ m' ⟨hv, hfr, hpres⟩
    refine ⟨fun x h1 h2 => hv x (by omega) (by rw [hXc]; exact h2), hfr, hpres⟩
  · intro _ m hm t h1 h2
    simp only [hiB, hb, ite_true] at h2
    rw [hm t h1 h2]
    simp only [val, hb, ite_true]

/-- **A request's program** is safe: it stores each of its tokens' slots. -/
theorem safe_reqProg {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init) (E : Env)
    {g b : Nat} (hg : g < d.G) (hb : b < d.R) :
    Safe E (Rall d) (Wp d init g b) (fun _ m => Qp d init g b m) init (reqProg d g b) := by
  have hG := hd.G_lt
  have hR := hd.R_le
  have hN := hd.N_le
  have hNi : (d.N : Int) ≤ 2 ^ 22 := by exact_mod_cast hN
  have hg32 : ((IntV.ofNat g : I32)).toInt = (g : Int) := toInt_ofNat32 (by omega)
  have hb32 : ((IntV.ofNat b : I32)).toInt = (b : Int) := toInt_ofNat32 (by omega)
  have hpre' := hpre
  obtain ⟨-, -, -, hpt, hst, hsz, hen, hsh, -, hbtv, htab, -⟩ := hpre
  obtain ⟨hptr, h16, hstr, hbs, henr, hshr⟩ := htab g hg
  have hBC := hd.bsz_cp g hg
  have hbsz : d.bsz g < 2 ^ 31 := by
    have : d.bsz g ≤ d.bsz g * d.CP_SIZE := Nat.le_mul_of_pos_right _ hd.cp_pos
    omega
  obtain ⟨hb0, hb1, -, -⟩ := hbtv g hg
  simp only [reqProg, _load_ptr, bind_assoc]
  -- line 108
  rw [ptr_zero_add _ hg32]
  refine safe_load1 hptr (fun z hz => rall_ptrs (span_bytes _ (by have := hd.ptrs_len; omega) hz))
    ?_
  refine safe_multiple_of1 (hint_base h16) ?_
  -- lines 109-113
  rw [ptr_zero_add _ hg32]
  refine safe_load1 hstr (fun z hz => rall_strides (span_bytes _ (by have := hd.strides_len; omega) hz))
    ?_
  rw [ptr_zero_add _ hg32]
  refine safe_load1 hbs (fun z hz => rall_sizes (span_bytes _ (by have := hd.sizes_len; omega) hz))
    ?_
  rw [ptr_zero_add _ hg32]
  refine safe_load1 henr (fun z hz => rall_en (span_bytes _ (by have := hd.en_len; omega) hz)) ?_
  have hP : 0 < (d.bt g).len → tl.intToPtr .i32 (IntV.ofInt (d.bt g).base : U64) = (d.bt g).ptr 0 := by
    intro hlen
    refine intToPtr_base hb0 ?_
    have : (0 : Int) < ((d.bt g).len : Int) * (DType.i32.bytes : Int) := by
      have : (0 : Int) < (d.bt g).len := by exact_mod_cast hlen
      simp only [DType.bytes]; push_cast; omega
    omega
  have hkv : (IntV.ofNat (d.bsz g) : I32).toInt = d.bsz g := toInt_ofNat32 hbsz
  by_cases hC : d.CP_SIZE = 1
  · rw [ite_eq_right (not_not.mpr hC), pure_bind]
    exact safe_reqTail hd hpre' E hg hb hP hkv (fun h => absurd hC h)
  · rw [ite_eq_left hC, ptr_zero_add _ hg32]
    refine safe_load1 hshr (fun z hz => rall_sh (span_bytes _ (by have := hd.sh_len; omega) hz)) ?_
    exact safe_reqTail hd hpre' E hg hb hP hkv (fun _ => rfl)

end Claims.SlotMappings
