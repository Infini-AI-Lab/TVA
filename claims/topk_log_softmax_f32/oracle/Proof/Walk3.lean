import Proof.Walk2

/-!
# The gather loop

Trip `t` of the gather loop loads ids `t·TB + l` of row `p`, gathers their logits, computes
`(x − mv) − lse` and stores it to `out[p, t·TB + l]`. `GInv t m` says that the memory agrees with
the initial one on the program's reads, keeps every byte present, and holds, at each output
`j < t·TB` of the row, a value `OutRel` describes: the two roundings of `(x_id − mv) − lse` for the
row's id `j` as the initial memory holds it. `safe_gatherTrip` keeps it.
-/
namespace Claims.TopkLogSoftmax

open TVA

/-- The read footprint of program `p`: its row of logits and its row of ids. -/
def Rd (d : Inst) (p : Nat) : Set Int := d.logits.elems (rowIn d p) ∪ d.ids.elems (rowIds d p)

/-- The outputs of row `p` that trip `t` of the gather loop writes. -/
def outBlk (d : Inst) (p t : Nat) : Set Nat :=
  {e | ∃ l, l < d.TOPK_BLOCK_SIZE ∧ t * d.TOPK_BLOCK_SIZE + l < d.topk ∧
    e = p * d.topk + (t * d.TOPK_BLOCK_SIZE + l)}

/-- What output `j` of row `p` holds: `(x − mv) − lse` for the logit `x` of the row's id `j`, two
IEEE subtractions up to NaN payload. -/
def OutRel (d : Inst) (init : Mem) (p : Nat) (mv lse : F32) (j : Nat) (c : F32) : Prop :=
  ∃ dd : F32, FloatV.NaNEq (FloatV.sub (xsOf d init p (tidOf d init p j).toInt.toNat) mv) dd ∧
    FloatV.NaNEq (FloatV.sub dd lse) c

/-- The invariant of the gather loop after `t` trips. -/
def GInv (d : Inst) (init : Mem) (p : Nat) (mv lse : F32) (t : Nat) (m : Mem) : Prop :=
  (∀ a ∈ Rd d p, m a = init a) ∧ (∀ x, init x ≠ none → m x ≠ none) ∧
    ∀ j, j < d.topk → j < t * d.TOPK_BLOCK_SIZE → ∃ c : F32,
      m.read .f32 (d.out.ptr (outIdx d p j)).addr = some c ∧ OutRel d init p mv lse j c

/-! ## Footprints -/

lemma rowIn_sub {d : Inst} {p : Nat} (hp : p < d.batch) : rowIn d p ⊆ inSet d := by
  rintro e ⟨k, hk, rfl⟩; exact ⟨p, k, hp, hk, rfl⟩

lemma rowIds_sub {d : Inst} {p : Nat} (hp : p < d.batch) : rowIds d p ⊆ idSet d := by
  rintro e ⟨k, hk, rfl⟩; exact ⟨p, k, hp, hk, rfl⟩

lemma rowOut_sub {d : Inst} {p : Nat} (hp : p < d.batch) : rowOut d p ⊆ outSet d := by
  rintro e ⟨k, hk, rfl⟩; exact ⟨p, k, hp, hk, rfl⟩

lemma outBlk_sub {d : Inst} {p t : Nat} : outBlk d p t ⊆ rowOut d p := by
  rintro e ⟨l, -, hl, rfl⟩; exact ⟨_, hl, rfl⟩

/-- A row's outputs share no byte with any row's reads. -/
lemma out_Rd {d : Inst} (hd : d.ok) {p q : Nat} (hp : p < d.batch) (hq : q < d.batch) :
    Disjoint (d.out.elems (rowOut d p)) (Rd d q) := by
  refine Disjoint.union_right ?_ ?_
  · exact hd.out_logits.mono (Buf.elems_mono d.out (rowOut_sub hp))
      (Buf.elems_mono d.logits (rowIn_sub hq))
  · exact hd.out_ids.mono (Buf.elems_mono d.out (rowOut_sub hp))
      (Buf.elems_mono d.ids (rowIds_sub hq))

/-- Reads inside `Rd` agree with the initial memory. -/
lemma read_agree {d : Inst} {p : Nat} {init m : Mem} (h : ∀ a ∈ Rd d p, m a = init a)
    {dt : DType} {b : Buf dt} {S : Set Nat} (hS : b.elems S ⊆ Rd d p) {j : Nat} (hj : j ∈ S) :
    m.read dt (b.ptr (j : Int)).addr = init.read dt (b.ptr (j : Int)).addr :=
  Mem.read_congr fun x hx => h x (hS (Buf.span_sub_elems b hj hx))

set_option maxHeartbeats 1600000 in
/-- **One trip of the gather loop** reads ids and logits of row `p`, writes the outputs of
`outBlk p t`, and keeps `GInv`. -/
lemma safe_gatherTrip {d : Inst} (hd : d.ok) {E : Env} {init : Mem} (hpre : Pre d init)
    {p t : Nat} (hp : p < d.batch) (ht : t * d.TOPK_BLOCK_SIZE < d.topk) {c : I32}
    (hc : c.toInt = ((t * d.TOPK_BLOCK_SIZE : Nat) : Int)) {req : I64} (hreq : req.toInt = p)
    {mv lse : F32} {m₁ : Mem} (hI : GInv d init p mv lse t m₁) :
    Safe E (Rd d p) (d.out.elems (outBlk d p t)) (fun _ m₂ => GInv d init p mv lse (t + 1) m₂) m₁
      (gatherTrip (d.out.ptr 0) (d.logits.ptr ((p * d.logits_stride : Nat) : Int)) (d.ids.ptr 0)
        req (IntV.ofNat d.ids_stride) (IntV.ofNat d.topk) d.TOPK_BLOCK_SIZE mv lse c ()) := by
  obtain ⟨e, he, hTB⟩ := hd.tb
  have hTBle : d.TOPK_BLOCK_SIZE ≤ 1024 := by
    rw [hTB]; exact le_trans (Nat.pow_le_pow_right (by norm_num) he) (by norm_num)
  have hTBpos : 0 < d.TOPK_BLOCK_SIZE := by rw [hTB]; exact Nat.two_pow_pos e
  have hk := hd.topk_lt
  have hfit : t * d.TOPK_BLOCK_SIZE + d.TOPK_BLOCK_SIZE ≤ 2 ^ 31 := by omega
  have htopk31 : d.topk < 2 ^ 31 := by omega
  have hp31 : p < 2 ^ 31 := lt_of_lt_of_le hp (le_of_lt hd.batch_lt)
  obtain ⟨hagree, hpres, hdone⟩ := hI
  obtain ⟨hvl0, hvi0, hvo0, hids⟩ := hpre
  have hvl : d.logits.Valid m₁ := Buf.valid_of_agree hvl0 fun x hx => hpres x (hvl0.2.2.2 x hx)
  have hvi : d.ids.Valid m₁ := Buf.valid_of_agree hvi0 fun x hx => hpres x (hvi0.2.2.2 x hx)
  have hvo : d.out.Valid m₁ := Buf.valid_of_agree hvo0 fun x hx => hpres x (hvo0.2.2.2 x hx)
  have hmask : ∀ i : Idx [d.TOPK_BLOCK_SIZE - 0],
      ((c + tl.arange 0 d.TOPK_BLOCK_SIZE).ltS (IntV.ofNat d.topk : I32)).get i =
        decide (t * d.TOPK_BLOCK_SIZE + i.1.1 < d.topk) := blk_mask hc hfit htopk31
  unfold gatherTrip
  dsimp only
  -- `topk_ids_stride` to `int64`
  refine safe_convert (fun _ => ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩) fun is his => ?_
  have hIS : is.item.toInt = ((d.ids_stride : Nat) : Int) :=
    (cast64 (his ())).trans (IntV.I32.toInt_ofNat hd.ids_stride_lt)
  rw [ptr_add64 d.ids (mul64 hreq hIS hp31 hd.ids_stride_lt)]
  -- the ids
  refine safe_load_buf d.ids (blk_addrs d.ids (p * d.ids_stride) hc hfit htopk31) ?_ hvi ?_
    fun tids ht1 _ => ?_
  · intro i hi
    exact hd.ids_len p _ hp (of_decide_eq_true hi)
  · exact (Buf.elems_mono d.ids blk_img).trans Set.subset_union_right
  have hRids : d.ids.elems (rowIds d p) ⊆ Rd d p := Set.subset_union_right
  have hRlog : d.logits.elems (rowIn d p) ⊆ Rd d p := Set.subset_union_left
  have htid : ∀ i : Idx [d.TOPK_BLOCK_SIZE - 0], t * d.TOPK_BLOCK_SIZE + i.1.1 < d.topk →
      tids.get i = tidOf d init p (t * d.TOPK_BLOCK_SIZE + i.1.1) := by
    intro i hi
    have h := ht1 i (decide_eq_true hi)
    rw [read_agree hagree hRids ⟨_, hi, rfl⟩] at h
    exact (tidOf_of_read (d := d) (r := p) (j := t * d.TOPK_BLOCK_SIZE + i.1.1) h).symm
  have hidr : ∀ i : Idx [d.TOPK_BLOCK_SIZE - 0], t * d.TOPK_BLOCK_SIZE + i.1.1 < d.topk →
      0 ≤ (tids.get i).toInt ∧ (tids.get i).toInt < d.vocab := by
    intro i hi
    have h := ht1 i (decide_eq_true hi)
    rw [read_agree hagree hRids ⟨_, hi, rfl⟩] at h
    exact hids p _ hp hi _ h
  have hnn : ∀ i, ((c + tl.arange 0 d.TOPK_BLOCK_SIZE).ltS (IntV.ofNat d.topk : I32)).get i = true →
      0 ≤ (tids.get i).toInt := by
    intro i hi
    rw [hmask i] at hi
    exact (hidr i (of_decide_eq_true hi)).1
  -- the gather
  refine safe_load_buf d.logits (gather_addrs d.logits (p * d.logits_stride) hnn) ?_ hvl ?_
    fun w hw1 _ => ?_
  · intro i hi
    have hi' : ((c + tl.arange 0 d.TOPK_BLOCK_SIZE).ltS (IntV.ofNat d.topk : I32)).get i = true := hi
    rw [hmask i] at hi'
    have h2 := hidr i (of_decide_eq_true hi')
    have := hd.logits_len p hp
    show p * d.logits_stride + (tids.get i).toInt.toNat < d.logits.len
    omega
  · refine (Buf.elems_mono d.logits ?_).trans hRlog
    rintro e ⟨i, hi, rfl⟩
    have hi' : ((c + tl.arange 0 d.TOPK_BLOCK_SIZE).ltS (IntV.ofNat d.topk : I32)).get i = true := hi
    rw [hmask i] at hi'
    have h2 := hidr i (of_decide_eq_true hi')
    show p * d.logits_stride + (tids.get i).toInt.toNat ∈ rowIn d p
    exact ⟨_, by omega, rfl⟩
  refine safe_sub fun dd hdd => ?_
  refine safe_sub fun o ho => ?_
  -- `topk` to `int64`
  refine safe_convert (fun _ => ⟨_, ConvRel.iff_cast_i32_i64.mpr rfl⟩) fun tk htk => ?_
  have hTK : tk.item.toInt = ((d.topk : Nat) : Int) :=
    (cast64 (htk ())).trans (IntV.I32.toInt_ofNat htopk31)
  rw [ptr_add64 d.out (mul64 hreq hTK hp31 htopk31)]
  refine safe_last ?_
  have himg : Buf.img (s := [d.TOPK_BLOCK_SIZE - 0])
      (fun i => p * d.topk + (t * d.TOPK_BLOCK_SIZE + i.1.1))
      (fun i => decide (t * d.TOPK_BLOCK_SIZE + i.1.1 < d.topk)) ⊆ outBlk d p t := by
    rintro e ⟨⟨⟨l, hl⟩, u⟩, hi, rfl⟩
    exact ⟨l, by simpa using hl, of_decide_eq_true hi, rfl⟩
  refine safe_store_buf d.out (blk_addrs d.out (p * d.topk) hc hfit htopk31) ?_ ?_ hvo
    (Buf.elems_mono d.out himg) fun m' hst hfr hpres' => safe_ret ?_
  · intro i hi
    have h1 := of_decide_eq_true hi
    have h2 : (p + 1) * d.topk ≤ d.batch * d.topk := Nat.mul_le_mul_right _ hp
    have h3 := hd.out_len
    rw [Nat.add_mul, Nat.one_mul] at h2
    omega
  · rintro ⟨⟨a, ha⟩, ⟨⟩⟩ ⟨⟨b, hb⟩, ⟨⟩⟩ _ _ heq
    have : a = b := by simp only at heq; omega
    subst this; rfl
  -- the invariant after the trip
  have hout_not : ∀ x, x ∈ Rd d p → x ∉ d.out.elems (Buf.img (s := [d.TOPK_BLOCK_SIZE - 0])
      (fun i => p * d.topk + (t * d.TOPK_BLOCK_SIZE + i.1.1))
      (fun i => decide (t * d.TOPK_BLOCK_SIZE + i.1.1 < d.topk))) := by
    intro x hx hx'
    exact Set.disjoint_left.mp (out_Rd hd hp hp)
      (Buf.elems_mono d.out (himg.trans outBlk_sub) hx') hx
  refine ⟨fun a ha => (hfr a (hout_not a ha)).trans (hagree a ha), fun x hx => hpres' x (hpres x hx),
    fun j hj hjt => ?_⟩
  by_cases hjold : j < t * d.TOPK_BLOCK_SIZE
  · obtain ⟨c0, hc0, hrel⟩ := hdone j hj hjold
    refine ⟨c0, ?_, hrel⟩
    rw [← hc0]
    refine Mem.read_congr fun x hx => hfr x fun hx' => ?_
    obtain ⟨e', he', hxe⟩ := hx'
    obtain ⟨⟨⟨l, hl⟩, u⟩, hli, rfl⟩ := he'
    have hsame : outIdx d p j = p * d.topk + (t * d.TOPK_BLOCK_SIZE + l) := by
      by_contra hne
      have hdis := Buf.elems_disjoint d.out (S := {outIdx d p j})
        (T := {p * d.topk + (t * d.TOPK_BLOCK_SIZE + l)}) (by simpa using hne)
      exact Set.disjoint_left.mp hdis (Buf.span_sub_elems d.out (Set.mem_singleton _) hx)
        ⟨_, Set.mem_singleton _, hxe⟩
    simp only [outIdx] at hsame
    omega
  · obtain ⟨l, rfl⟩ : ∃ l, j = t * d.TOPK_BLOCK_SIZE + l := ⟨j - t * d.TOPK_BLOCK_SIZE, by omega⟩
    have hl : l < d.TOPK_BLOCK_SIZE := by
      have : (t + 1) * d.TOPK_BLOCK_SIZE = t * d.TOPK_BLOCK_SIZE + d.TOPK_BLOCK_SIZE := by ring
      omega
    have hl' : l < d.TOPK_BLOCK_SIZE - 0 := by simpa using hl
    let i : Idx [d.TOPK_BLOCK_SIZE - 0] := (⟨l, hl'⟩, ())
    have hen : decide (t * d.TOPK_BLOCK_SIZE + i.1.1 < d.topk) = true := decide_eq_true hj
    refine ⟨o.get i, hst i hen, dd.get i, ?_, ho i⟩
    have hmi : ((c + tl.arange 0 d.TOPK_BLOCK_SIZE).ltS (IntV.ofNat d.topk : I32)).get i = true := by
      rw [hmask i]; exact hen
    have hw := hw1 i hmi
    rw [read_agree hagree hRlog (S := rowIn d p) ⟨_, (by
      have := hidr i hj; omega), rfl⟩] at hw
    have hwx : w.get i = xsOf d init p (tids.get i).toInt.toNat :=
      (xsOf_of_read (d := d) (r := p) (k := (tids.get i).toInt.toNat) hw).symm
    have h1 := hdd i
    rw [hwx, htid i hj] at h1
    exact h1

end Claims.TopkLogSoftmax
