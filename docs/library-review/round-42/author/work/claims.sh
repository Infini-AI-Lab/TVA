#!/bin/sh
# Round 42 (author): elaborate every fixed claim (claims/<k>/Claim.lean) against the library as built
# (`lake build TVA` first), with `lake env lean`, from lean/ of the TVA root (run from that folder:
# sh <this file>). One line per claim: name, exit code, number of error lines; then whether claims/ differs
# from git's HEAD (`git diff --stat -- ../claims/`, empty when nothing changed).
for f in ../claims/*/Claim.lean; do echo "$f"; done | xargs -P 6 -I{} sh -c \
  'out=$(~/.elan/bin/lake env lean {} 2>&1); rc=$?; k=$(basename $(dirname {})); echo "$k exit=$rc errors=$(printf "%s\n" "$out" | grep -c "error")"' | sort
echo "git diff --stat -- claims/: [$(git diff --stat -- ../claims/)]"
