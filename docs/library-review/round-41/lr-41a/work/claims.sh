#!/bin/sh
# L189: elaborate every fixed claim (claims/<k>/Claim.lean) against the library as built, with `lake env lean`,
# from lean/ of the TVA root (run from that folder: sh <this file>). One line per claim: name, exit code, error lines.
for f in ../claims/*/Claim.lean; do echo "$f"; done | xargs -P 6 -I{} sh -c \
  'out=$(~/.elan/bin/lake env lean {} 2>&1); rc=$?; k=$(basename $(dirname {})); echo "$k exit=$rc errors=$(printf "%s\n" "$out" | grep -c "error")"'
