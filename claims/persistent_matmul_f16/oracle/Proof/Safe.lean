import Claim
import TVA.Proof
import Proof.Geometry
import Proof.Numeric
import Proof.Compute
import Proof.InnerSafe
import Proof.TripSafe

/-!
# Each program of the launch is safe: its setup, its outer loop of trips.
-/

set_option maxRecDepth 8000

namespace Claims.PersistentMatmul

open TVA

/-- **Each program of the launch is safe.** -/
theorem safe_body {d : Inst} (hd : d.ok) {init : Mem} (hpre : Pre d init)
    (E : Env)
    (hacc : Device.DotLiftAcc E.P ⟨.f16, .f16, .f32, .tf32, 0⟩ Device.relDot16 Device.absDot16)
    (htot : Device.DotTotal E.P ⟨.f16, .f16, .f32, .tf32, 0⟩)
    {i : Fin (launch d).size} :
    Safe E (Rd d init i) (Wr d init i) (fun _ m => Qd d init i m) init
      ((model d).body ((launch d).ctx i)) := by
  have hSlt : d.NUM_SMS < 2 ^ 31 := hd.sms.2
  have hSval : (IntV.ofNat d.NUM_SMS : I32).toInt = ((d.NUM_SMS : Nat) : Int) :=
    IntV.I32.toInt_ofNat hSlt
  have hisize : (i : Nat) < min d.NUM_SMS (tiles d) :=
    lt_of_lt_of_eq i.isLt (launch_size d)
  have hiS : (i : Nat) < d.NUM_SMS := lt_of_lt_of_le hisize (Nat.min_le_left _ _)
  have hit : (i : Nat) ≤ nT d := by
    have h1 : (i : Nat) < tiles d := lt_of_lt_of_le hisize (Nat.min_le_right _ _)
    rw [tiles_eq hd] at h1
    omega
  have hi31 : (i : Nat) < 2 ^ 31 := by
    have h1 : (i : Nat) < tiles d := lt_of_lt_of_le hisize (Nat.min_le_right _ _)
    have h2 := hd.tiles_le
    omega
  rw [body_trip d ((launch d).ctx i)]
  dsimp only
  set sp : I32 := tl.program_id ((launch d).ctx i) 0 with hspdef
  have hsp : sp.toInt = ((i : Nat) : Int) := by
    rw [hspdef]
    simp only [tl.program_id]
    rw [launch_pid]
    exact IntV.I32.toInt_ofNat hi31
  refine safe_setup (d := d) (m := init) (sp := sp) (i := (i : Nat)) hd hsp hiS
    (k := fun npm npn kt nt tidc =>
      tl.forRange sp nt (IntV.ofNat d.NUM_SMS : I32) tidc
        (fun tile_id tile_id_c =>
          trip d (tl.arange 0 64) ((IntV.ofNat 8 : I32) * npn) npm kt tile_id tile_id_c)
      >>= fun _ => pure ()) ?_
  intro npm npn kt nt tidc hnpm hnpn hkt hnt htidc
  refine (safe_forRange (lo := sp) (hi := nt) (step := (IntV.ofNat d.NUM_SMS : I32))
      (init := tidc)
      (body := fun tile_id tile_id_c =>
        trip d (tl.arange 0 64) ((IntV.ofNat 8 : I32) * npn) npm kt tile_id tile_id_c)
      (k := fun _ => pure ())
      (hrange := rangeOk_outer (d := d) hd hsp hnt hisize)
      (I := fun (t : Nat) (st : I32) (m' : Mem) =>
        st.toInt = (((i : Nat) + t * d.NUM_SMS : Nat) : Int) - (d.NUM_SMS : Int) ∧
        MemInv d init m' ∧ Done d init i t m')
      (Wt := fun t => tileSet d ((i : Nat) + t * d.NUM_SMS)) (Wk := ∅) ?_ ?_ ?_ ?_ ?_).mono
    (fun a m' h => h) ?_
  · -- trips write disjoint tiles
    intro t u htu hu
    have hu' : u < tl.tripCount ((i : Nat) : Int) (nT d : Int) (d.NUM_SMS : Int) := by
      rwa [hsp, hnt, hSval] at hu
    have ht' : t < tl.tripCount ((i : Nat) : Int) (nT d : Int) (d.NUM_SMS : Int) := by omega
    exact tileSet_disjoint hd (trip_lt hd hit ht') (trip_lt hd hit hu') (by
      intro h
      have := (tile_id_inj hd hiS hiS h).2
      omega)
  · -- the loop writes nothing outside its tiles
    exact Set.disjoint_empty _
  · -- the invariant at the start
    refine ⟨?_, ⟨fun x hx => rfl, fun x hx => hx⟩, ?_⟩
    · rw [htidc]
      push_cast
      ring
    · intro u hu
      omega
  · -- one trip
    intro t ht st m₁ hinv
    obtain ⟨hst, hminv, hdone⟩ := hinv
    have ht' : t < tl.tripCount ((i : Nat) : Int) (nT d : Int) (d.NUM_SMS : Int) := by
      rwa [hsp, hnt, hSval] at ht
    have hTlt : (i : Nat) + t * d.NUM_SMS < nT d := trip_lt hd hit ht'
    have hT31 : (i : Nat) + t * d.NUM_SMS < 2 ^ 31 := by
      have h1 := nT_lt hd
      omega
    set x : Int := sp.toInt + (t : Int) * (IntV.ofNat d.NUM_SMS : I32).toInt with hxdef
    have hxval : x = (((i : Nat) + t * d.NUM_SMS : Nat) : Int) := by
      rw [hxdef, hsp, hSval]
      push_cast
      ring
    have hsf : IntV.SFits 32 x := by
      rw [hxval]
      unfold IntV.SFits
      have he : (2 : Int) ^ (32 - 1) = 2 ^ 31 := by norm_num
      rw [he]
      constructor
      · have h0 : (0 : Int) ≤ (((i : Nat) + t * d.NUM_SMS : Nat) : Int) := by
          exact_mod_cast Nat.zero_le ((i : Nat) + t * d.NUM_SMS)
        have hc : (0 : Int) ≤ (2 : Int) ^ 31 := by positivity
        linarith
      · exact_mod_cast hT31
    have htid : (IntV.ofInt x : I32).toInt = (((i : Nat) + t * d.NUM_SMS : Nat) : Int) := by
      rw [IntV.I32.toInt_ofInt hsf, hxval]
    have h8n : 8 * nPN d < 2 ^ 31 := by
      have h1 : nPN d ≤ d.N := by
        simp only [nPN]
        exact Nat.div_le_self _ _
      have h2 : d.N ≤ 2 ^ 17 := hd.N_dim.2.2
      omega
    have hnpg : ((IntV.ofNat 8 : I32) * npn).toInt = ((8 * nPN d : Nat) : Int) :=
      IntV.I32.mul_nat (a := (IntV.ofNat 8 : I32)) (b := npn) (x := 8) (y := nPN d)
        (IntV.I32.toInt_ofNat (by norm_num)) hnpn h8n
    have hdisj : ∀ u, u < t → Disjoint (tileSet d ((i : Nat) + u * d.NUM_SMS))
        (tileSet d ((i : Nat) + t * d.NUM_SMS)) := by
      intro u hu
      have hu' : u < tl.tripCount ((i : Nat) : Int) (nT d : Int) (d.NUM_SMS : Int) := by omega
      exact tileSet_disjoint hd (trip_lt hd hit hu') (trip_lt hd hit ht') (by
        intro h
        have := (tile_id_inj hd hiS hiS h).2
        omega)
    exact (safe_trip (d := d) (E := E) (init := init) (m := m₁) hd hpre
      (hinner := fun hT htK hmi ham hbn hkm hki hcell =>
        safe_inner hd hpre hmi hacc htot hT htK ham hbn hkm hki hcell)
      (i := (i : Nat)) (t := t) (T := (i : Nat) + t * d.NUM_SMS) rfl hTlt hdisj hminv hdone
      (offs_k_for_mask := tl.arange 0 64)
      (npg := (IntV.ofNat 8 : I32) * npn) (npm := npm) (kt := kt)
      (tid := (IntV.ofInt x : I32)) (tidc := st)
      (fun j => arange_get1_toInt (by norm_num) j) htid hst hnpg hnpm hkt).mono
      (fun a m' h => ⟨by
        rw [h.1]
        push_cast
        ring, h.2.1, h.2.2⟩) le_rfl
  · -- the postcondition
    intro st m₁ hinv
    refine safe_ret ?_
    intro t ht a b ha hb
    refine hinv.2.2 t ?_ a b ha hb
    rwa [← hsp, ← hnt, ← hSval] at ht
  · -- the writes are the program's footprint
    rintro x (⟨t, ht, hx⟩ | hx)
    · exact ⟨t, by rwa [hsp, hnt, hSval] at ht, hx⟩
    · exact hx.elim

end Claims.PersistentMatmul
