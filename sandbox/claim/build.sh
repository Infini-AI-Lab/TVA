#!/usr/bin/env bash
# Build tva3-claim:<tag>, the image every claim task starts from (Dockerfile), on tva3-base.
#   sandbox/claim/build.sh [BASE]
# BASE defaults to tva3-base:<content hash of lean/ now>, built first (../build_base.sh) when it is missing.
# <tag> is <the base's hash>-<a content hash of what this image adds>: tva-check (and replay.lean), the harness, docs/ as it is now,
# and the worked example (claims/vec_add_f32's own files with its proof, claims/vec_add_f32/oracle/). Prints the tag on its last line.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tva="$(cd "$here/../.." && pwd)"
base="${1:-tva3-base:$(python3 -I "$here/../lean_hash.py" "$tva/lean")}"
if ! docker image inspect "$base" >/dev/null 2>&1; then
  [ $# -eq 0 ] || { echo "no image $base" >&2; exit 1; }
  echo "$base is missing: building it" >&2
  base="$("$here/../build_base.sh" | tail -1)"
fi
# docs/index/ must describe this library: the image ships docs/ to provers. The check reads the built
# library, TVA and TVA.Proof (lean/: lake build && lake build TVA.Proof).
if [ ! -f "$tva/lean/.lake/build/lib/lean/TVA/Proof.olean" ]; then
  echo "the library is not built: cd $tva/lean && lake build && lake build TVA.Proof, then build again" >&2
  exit 1
fi
if ! PATH="$HOME/.elan/bin:$PATH" python3 -I "$tva/tools/make_index.py" --check >&2; then
  echo "docs/index/ is stale: run python3 -I tools/make_index.py, then build again" >&2
  exit 1
fi
ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
cp "$here/Dockerfile" "$here/tva-check" "$here/replay.lean" "$here/lake" "$here/settings.json" "$here/lean-fixer.md" "$here/CLAUDE.md" \
   "$here/process_safety.md" "$ctx/"
cp -a "$tva/docs" "$ctx/docs"
# the example: the claim's own files (claim_files.py's allow list, with Claim.lean's sha256 in claim.json) and its
# oracle proof. It is the only proof in the image: no other claim's oracle (claims/<k>/oracle/) enters it
python3 -I "$here/../claim_files.py" "$tva/claims/vec_add_f32" "$ctx/example"
if ! python3 -I "$here/oracle.py" vec_add_f32 "$ctx/example"; then
  echo "the worked example's proof is not available (oracle.py failed): the claim image needs it" >&2
  exit 1
fi
sub="$(cd "$ctx" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-8)"
tag="tva3-claim:${base#tva3-base:}-$sub"
DOCKER_BUILDKIT=1 docker build --build-arg "BASE=$base" -t "$tag" "$ctx" >&2
echo "built $tag (from $base)" >&2
echo "$tag"
