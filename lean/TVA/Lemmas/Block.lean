import TVA.Lemmas.Int
import TVA.Lemmas.Access

/-!
# One-dimensional blocks

The most common Triton access pattern:

```python
offsets = pid * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n
x = tl.load(x_ptr + offsets, mask=mask)
```

Program `p` handles elements `p * BLOCK + l` for `l < BLOCK`, those below `n` (`Tile p B n j`).
This file gives the lane addresses of such an access (`addrs_lane`) and its footprint
(`Buf.tile`). It also gives load and store rules that state the result per element
(`safe_load_block`, `safe_store_block`), and shows that the tiles of different programs and
of disjoint buffers do not overlap.

`int32` arithmetic does not wrap when the largest offset, `p * BLOCK + BLOCK - 1`, is below
`2^31`, that is `p * BLOCK + BLOCK ≤ 2^31` (`Fits`). Every program of the grid
`cdiv(n, BLOCK)` fits when `n + BLOCK ≤ 2^31` (`fits_of_grid`), and when `n < 2^31` and `BLOCK`
divides `2^31` (`fits_of_grid_dvd`), as every power of two up to `2^31` does
(`fits_of_grid_pow2`; Triton's block sizes are powers of two up to `2^20`).
-/
namespace TVA

namespace Block

/-- `pid * BLOCK + tl.arange(0, BLOCK)` for program `p`. -/
def offs (p B : Nat) : Tensor [B - 0] I32 := (IntV.ofNat p * IntV.ofNat B : I32) + tl.arange 0 B

/-- `offsets < n`. -/
def mask (p B n : Nat) : Tensor [B - 0] Bool := (offs p B).ltS (IntV.ofNat n)

/-- Element `j` belongs to program `p`'s tile and is below `n`. -/
def Tile (p B n j : Nat) : Prop := p * B ≤ j ∧ j < p * B + B ∧ j < n

/-- The block arithmetic of program `p` fits `int32`: its largest offset `p * B + B - 1` is below
`2^31`. -/
def Fits (p B : Nat) : Prop := p * B + B ≤ 2 ^ 31

theorem lane_toInt {p B : Nat} (hfit : Fits p B) (l : Fin (B - 0)) :
    ((offs p B).get (l, ())).toInt = p * B + l := by
  have hl := l.isLt
  simp only [Nat.sub_zero] at hl
  unfold Fits at hfit
  have hli : (IntV.ofInt ((0 : Nat) + (l : Nat)) : I32).toInt = l := by
    rw [IntV.I32.toInt_ofInt] <;> push_cast
    · ring
    · constructor <;> omega
  show ((IntV.ofNat p : I32) * IntV.ofNat B + IntV.ofInt ((0 : Nat) + (l : Nat))).toInt = _
  rcases Nat.eq_zero_or_pos p with rfl | hp0
  · -- program 0: the product is `0`, whatever `B` is as an `int32`
    have h0 : (IntV.ofNat 0 : I32).toInt = 0 := IntV.I32.toInt_ofNat (by norm_num)
    have hmul : ((IntV.ofNat 0 : I32) * IntV.ofNat B).toInt = 0 := by
      rw [IntV.toInt_mul (by norm_num), h0, zero_mul]
      rw [h0, zero_mul]; constructor <;> norm_num
    rw [IntV.toInt_add (by norm_num), hmul, hli]
    · simp
    · rw [hmul, hli]; constructor <;> omega
  · have hBB : B + B ≤ p * B + B := by
      have := Nat.le_mul_of_pos_left B hp0; omega
    have hp31 : p < 2 ^ 31 := by
      have : p ≤ p * B := Nat.le_mul_of_pos_right p (by omega); omega
    have hB31 : B < 2 ^ 31 := by omega
    have hpB : (p : Int) * B + B ≤ 2 ^ 31 := by exact_mod_cast hfit
    have hpi := IntV.I32.toInt_ofNat hp31
    have hBi := IntV.I32.toInt_ofNat hB31
    have hmul : ((IntV.ofNat p : I32) * IntV.ofNat B).toInt = (p : Int) * B := by
      rw [IntV.toInt_mul (by norm_num), hpi, hBi]
      rw [hpi, hBi]; constructor <;> norm_num <;> nlinarith
    rw [IntV.toInt_add (by norm_num), hmul, hli]
    rw [hmul, hli]; constructor <;> norm_num <;> nlinarith

theorem mask_lane {p B n : Nat} (hfit : Fits p B) (hn : n < 2 ^ 31) (l : Fin (B - 0)) :
    (mask p B n).get (l, ()) = decide (p * B + l < n) := by
  show IntV.lt ((offs p B).get (l, ())) (IntV.ofNat n) = _
  rw [IntV.lt_eq, lane_toInt hfit, IntV.I32.toInt_ofNat hn]
  simp only [decide_eq_decide]; norm_cast

/-- The address of lane `l` of `b.ptr 0 + offsets` under `mask`. -/
theorem addrs_lane {d : DType} (b : Buf d) {p B n : Nat} (hfit : Fits p B) (hn : n < 2 ^ 31)
    (l : Fin (B - 0)) :
    tl.addrs (b.ptr 0 + offs p B) (some (mask p B n)) (l, ()) =
      if p * B + l < n then some (b.ptr ((p * B + l : Nat) : Int)).addr else none := by
  have hget : ((b.ptr 0 + offs p B).get (l, ())).addr = (b.ptr ((p * B + l : Nat) : Int)).addr := by
    show ((b.ptr 0).offset ((offs p B).get (l, ())).toInt).addr = _
    rw [lane_toInt hfit]; simp [Buf.ptr]
  simp only [tl.addrs, mask_lane hfit hn, decide_eq_true_eq]
  split_ifs
  · rw [hget]
  · rfl

theorem lane_tile {p B n : Nat} (l : Fin (B - 0)) (h : p * B + l < n) : Tile p B n (p * B + l) := by
  have := l.isLt; simp only [Nat.sub_zero] at this
  exact ⟨by omega, by omega, h⟩

theorem tile_lane {p B n j : Nat} (h : Tile p B n j) : ∃ l : Fin (B - 0), j = p * B + l := by
  obtain ⟨h1, h2, -⟩ := h
  exact ⟨⟨j - p * B, by simp only [Nat.sub_zero]; omega⟩, by simp only; omega⟩

/-- Every element below `n` is in the tile of program `j / B`. -/
theorem tile_div {B n j : Nat} (hB : 0 < B) (hj : j < n) : Tile (j / B) B n j :=
  ⟨Nat.div_mul_le_self j B, by have := Nat.lt_div_mul_add (a := j) hB; linarith, hj⟩

theorem tile_unique {p q B n j : Nat} (hp : Tile p B n j) (hq : Tile q B n j) : p = q := by
  obtain ⟨h1, h2, -⟩ := hp
  obtain ⟨h3, h4, -⟩ := hq
  rcases lt_trichotomy p q with h | h | h
  · have : (p + 1) * B ≤ q * B := Nat.mul_le_mul_right B h
    nlinarith
  · exact h
  · have : (q + 1) * B ≤ p * B := Nat.mul_le_mul_right B h
    nlinarith

/-- Programs below the grid `cdiv(n, B)` have fitting arithmetic when `n + B ≤ 2^31`. -/
theorem fits_of_grid {p B n : Nat} (hnB : n + B ≤ 2 ^ 31) (hp : p < (n + B - 1) / B) :
    Fits p B := by
  have h1 : (p + 1) * B ≤ (n + B - 1) / B * B := Nat.mul_le_mul_right B hp
  have h2 : (n + B - 1) / B * B ≤ n + B - 1 := Nat.div_mul_le_self _ _
  unfold Fits; rw [Nat.succ_mul] at h1; omega

/-- Programs below the grid `cdiv(n, B)` have fitting arithmetic when `n < 2^31` and `B`
divides `2^31`: program `p`'s first offset `p * B` is a multiple of `B` below `n`, so at most
`2^31 - B`. -/
theorem fits_of_grid_dvd {p B n : Nat} (hB : B ∣ 2 ^ 31) (hn : n < 2 ^ 31)
    (hp : p < (n + B - 1) / B) : Fits p B := by
  have hB0 : 0 < B := Nat.pos_of_dvd_of_pos hB (by norm_num)
  have h1 : (p + 1) * B ≤ (n + B - 1) / B * B := Nat.mul_le_mul_right B hp
  have h2 : (n + B - 1) / B * B ≤ n + B - 1 := Nat.div_mul_le_self _ _
  rw [Nat.succ_mul] at h1
  have hlt : p * B < 2 ^ 31 := by omega
  obtain ⟨c, hc⟩ := hB
  have hpc : p < c := by
    rw [hc] at hlt
    rw [Nat.mul_comm B c] at hlt
    exact Nat.lt_of_mul_lt_mul_right hlt
  unfold Fits
  have : (p + 1) * B ≤ c * B := Nat.mul_le_mul_right B hpc
  rw [Nat.succ_mul] at this; rw [hc, Nat.mul_comm B c]; omega

/-- `fits_of_grid_dvd` for a power-of-two block size up to `2^31` (Triton's block sizes are powers
of two up to `2^20`): every program of the grid `cdiv(n, B)` fits when `n < 2^31`. -/
theorem fits_of_grid_pow2 {p B n e : Nat} (hB : B = 2 ^ e) (he : e ≤ 31) (hn : n < 2 ^ 31)
    (hp : p < (n + B - 1) / B) : Fits p B :=
  fits_of_grid_dvd (hB ▸ Nat.pow_dvd_pow 2 he) hn hp

end Block

open Block

namespace Buf

variable {d : DType}

/-- The bytes of the elements of `b` in program `p`'s tile. -/
def tile (b : Buf d) (p B n : Nat) : Set Int := b.elems {j | Tile p B n j}

theorem tile_sub_bytes (b : Buf d) {p B n : Nat} (hlen : n ≤ b.len) : b.tile p B n ⊆ b.bytes :=
  b.elems_sub_bytes fun _ hj => by have := hj.2.2; omega

theorem tile_disjoint (b : Buf d) {p q B n : Nat} (hpq : p ≠ q) :
    Disjoint (b.tile p B n) (b.tile q B n) :=
  b.elems_disjoint (Set.disjoint_left.mpr fun _ hj hj' => hpq (tile_unique hj hj'))

theorem tile_disjoint_of {d' : DType} (b : Buf d) (c : Buf d') (hbc : Disjoint b.bytes c.bytes)
    {p q B n : Nat} (hb : n ≤ b.len) (hc : n ≤ c.len) : Disjoint (b.tile p B n) (c.tile q B n) :=
  hbc.mono (b.tile_sub_bytes hb) (c.tile_sub_bytes hc)

theorem span_sub_tile (b : Buf d) {p B n j : Nat} (h : Tile p B n j) :
    Mem.span d (b.ptr (j : Int)).addr ⊆ b.tile p B n :=
  b.span_sub_elems h

end Buf

/-! ## Accesses -/

section Access
variable {d : DType} {p B n : Nat}

/-- The lanes of a block access, as buffer elements. -/
theorem addrs_block (b : Buf d) (hfit : Fits p B) (hn : n < 2 ^ 31) :
    b.Addrs (tl.addrs (b.ptr 0 + offs p B) (some (mask p B n)))
      (fun i => p * B + i.1) (fun i => decide (p * B + i.1 < n)) := by
  rintro ⟨l, ⟨⟩⟩
  rw [addrs_lane b hfit hn]
  simp only [decide_eq_true_eq]

theorem img_block : Buf.img (s := [B - 0]) (fun i => p * B + i.1) (fun i => decide (p * B + i.1 < n)) =
    {j | Tile p B n j} := by
  ext j; constructor
  · rintro ⟨⟨l, ⟨⟩⟩, h, rfl⟩; exact lane_tile l (by simpa using h)
  · intro hj
    obtain ⟨l, rfl⟩ := tile_lane hj
    exact ⟨(l, ()), by simpa using hj.2.2, rfl⟩

theorem accBytes_block (b : Buf d) (hfit : Fits p B) (hn : n < 2 ^ 31) :
    accBytes d (tl.addrs (b.ptr 0 + offs p B) (some (mask p B n))) = b.tile p B n := by
  rw [(addrs_block b hfit hn).accBytes, img_block]; rfl

variable {E : Env} {R W : Set Int} {α : Type} {Q : α → Mem → Prop} {m : Mem}

/-- **Block load.** The result holds the tile's elements; masked-off lanes are unspecified. -/
theorem safe_load_block (b : Buf d) (hfit : Fits p B) (hn : n < 2 ^ 31) (hlen : n ≤ b.len)
    (hv : b.Valid m) (hR : b.tile p B n ⊆ R) {k : Tensor [B - 0] d.Val → Prog α}
    (hk : ∀ v : Tensor [B - 0] d.Val, (∀ l : Fin (B - 0), p * B + l < n →
        m.read d (b.ptr ((p * B + l : Nat) : Int)).addr = some (v.get (l, ()))) →
      Safe E R W Q m (k v)) :
    Safe E R W Q m (tl.load (b.ptr 0 + offs p B) (some (mask p B n)) >>= k) :=
  safe_load_buf b (addrs_block b hfit hn) (fun _ h => by simp at h; omega) hv
    (by rw [img_block]; exact hR) fun v hv' _ => hk v fun l hl => hv' (l, ()) (by simpa using hl)

/-- **Block store.** Afterwards the tile holds the values, nothing else changed, and every
present byte is still present. -/
theorem safe_store_block (b : Buf d) (hfit : Fits p B) (hn : n < 2 ^ 31) (hlen : n ≤ b.len)
    (hv : b.Valid m) (hW : b.tile p B n ⊆ W) {vals : Tensor [B - 0] d.Val} {k : Unit → Prog α}
    (hk : ∀ m', (∀ l : Fin (B - 0), p * B + l < n →
        m'.read d (b.ptr ((p * B + l : Nat) : Int)).addr = some (vals.get (l, ()))) →
      (∀ x, x ∉ b.tile p B n → m' x = m x) → (∀ x, m x ≠ none → m' x ≠ none) →
      Safe E R W Q m' (k ())) :
    Safe E R W Q m (tl.store (b.ptr 0 + offs p B) vals (some (mask p B n)) >>= k) := by
  refine safe_store_buf b (addrs_block b hfit hn) (fun _ h => by simp at h; omega) ?_ hv
    (by rw [img_block]; exact hW) fun m' h1 h2 h3 => hk m' (fun l hl => h1 (l, ()) (by simpa using hl))
      (fun x hx => h2 x (by rw [img_block]; exact hx)) h3
  rintro ⟨l, ⟨⟩⟩ ⟨l', ⟨⟩⟩ - - h
  have : (l : Nat) = l' := by simp only at h; omega
  rw [Fin.ext this]

end Access

end TVA
