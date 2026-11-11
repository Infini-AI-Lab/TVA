import TVA.Proof.Mem
import TVA.Proof.Buf
import TVA.Proof.Step
import TVA.Proof.Safe
import TVA.Proof.Rules.Core
import TVA.Proof.Rules.Memory
import TVA.Proof.Rules.Arith
import TVA.Proof.Rules.Convert
import TVA.Proof.Rules.Reduce
import TVA.Proof.Rules.ArgReduce
import TVA.Proof.Rules.Dot
import TVA.Proof.Rules.Loop
import TVA.Proof.Compose
import TVA.Proof.Correct

/-!
# The core of the proof kit

A claim's proof goes from the model to `Correct` in three steps:

1. **One program at a time.** For each program `i` of the grid, prove
   `Safe E (R i) (W i) (fun _ m => Q i m) init (k.body (L.ctx i))` (`TVA.Proof.Safe`): run alone
   from the initial memory, the program cannot reach undefined behaviour, always ends, reads only
   `R i`, writes only `W i` (each byte once), uses no atomic's result, and leaves a memory
   satisfying `Q i`. The proof walks the model's `do` block line by line with one rule per `tl.*`
   operation (`TVA.Proof.Rules.*`: `safe_load_buf`, `safe_store_buf`, `safe_add`, `safe_dot`,
   `safe_forRange`, …), composed by `Safe.bind`.
2. **Footprints.** No program writes a byte another program reads or writes, and no program reads
   a byte it writes; usually a few lines about buffer tiles (`TVA.Proof.Buf`, `Buf.elems_*`).
3. **The endpoint.** `correct_of_safe` (`TVA.Proof.Correct`) combines them: the launch is well
   behaved, every outcome is a normal end, and the final memory agrees with each program's own
   result on its footprint and with the initial memory elsewhere; the prover shows that this gives
   the claim's `Post`.

| file | what it gives |
|---|---|
| `Proof/Mem` | locality of reads and writes, read after write, `DistinctLanes`, `writeAll_read` |
| `Proof/Buf` | valid buffers can be read and written; `Buf.Addrs.accBytes`, `Buf.Addrs.distinct` |
| `Proof/Step` | one step in two memories: `Step.frame`, `Step.transfer`, `Mem.mix` |
| `Proof/Safe` | `Safe`, `Safe.bind`/`seq`/`bind_ro`/`mono`, and `stepT`, `no_ub`, `acc` |
| `Proof/Rules/Core` | `safe_ret`, `safe_last`, `safe_assume`, hints, `safe_reshape_any` |
| `Proof/Rules/Memory` | loads, stores, scalar accesses, padding, atomics on owned elements |
| `Proof/Rules/Arith` | float and integer arithmetic, platform `div` and `math` |
| `Proof/Rules/Convert` | `tl.convert`, `tl.cvt`, fp8, integer ↔ float |
| `Proof/Rules/Reduce`, `ArgReduce` | sums, max/min, argmax/argmin |
| `Proof/Rules/Dot` | `tl.dot`, `dot + c`, broadcast-multiply sums, `tl.dot_scaled` |
| `Proof/Rules/Loop` | `tl.forN`, `tl.forRange`, `tl.whileLoop` with per-iteration footprints |
| `Proof/Compose` | `independent`: the launch of `Safe` programs with disjoint footprints |
| `Proof/Correct` | `correct_of_safe`, `Launch.valid_grid1` |

Not here: access patterns (1-D blocks, 2-D tiles, block pointers, descriptors), integer width,
float error, `tl.dot` accumulation, scans, random numbers, and atomics shared between programs;
those kits build on this one.
-/
