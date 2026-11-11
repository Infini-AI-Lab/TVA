import TVA.Proof.Basic
import TVA.Proof.Int
import TVA.Proof.Float
import TVA.Proof.Softmax
import TVA.Proof.Recurrence
import TVA.Proof.Dot
import TVA.Proof.Pattern
import TVA.Proof.Shared
import TVA.Proof.Scan
import TVA.Proof.Random

/-!
# The proof kit

Lemmas for proving `claim.Holds`. Nothing here is trusted: every statement is proved from the
definitions in `TVA`.

* `TVA.Proof.Basic`: `Safe` (a program's footprint and the rule for each `tl.*` operation) and
  `correct_of_safe`, which turns per-program `Safe` facts into `Correct` for the launch;
* `TVA.Proof.Shared`: `SafeA` and `correct_of_safeA`, for programs that meet only through
  `tl.atomic_add` and similar atomics on shared cells;
* `TVA.Proof.Pattern`: 1-D blocks, 2-D tiles, block pointers and tensor descriptors;
* `TVA.Proof.Int`: integer widths, casts, division and remainder;
* `TVA.Proof.Float`: rounding, errors of operations, trees and sums;
* `TVA.Proof.Dot`: `acc += tl.dot(a, b)` loops and chunked accumulation;
* `TVA.Proof.Scan`, `TVA.Proof.Random`: scans, sorts, histograms, and `tl.rand`;
* `TVA.Proof.Softmax`, `TVA.Proof.Recurrence`: online softmax and decayed recurrences.
-/
