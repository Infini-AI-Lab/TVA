#!/usr/bin/env bash
# Build tva3-claim:<tag>, the image every claim task starts from (Dockerfile), on tva3-base.
#   sandbox/claim/build.sh [BASE]
# BASE defaults to tva3-base:<content hash of lean/ now>, built first (../build_base.sh) when it is missing.
# <tag> is <the base's hash>-<a content hash of what this image adds>: tva-check, the harness, docs/ as it is now,
# and the worked example (claims/vec_add_f32 with the proof oracle.py writes). Prints the tag on its last line.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tva="$(cd "$here/../.." && pwd)"
base="${1:-tva3-base:$(python3 -I "$here/../lean_hash.py" "$tva/lean")}"
if ! docker image inspect "$base" >/dev/null 2>&1; then
  [ $# -eq 0 ] || { echo "no image $base" >&2; exit 1; }
  echo "$base is missing: building it" >&2
  base="$("$here/../build_base.sh" | tail -1)"
fi
ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
cp "$here/Dockerfile" "$here/tva-check" "$here/lake" "$here/settings.json" "$here/lean-fixer.md" "$here/CLAUDE.md" \
   "$here/process_safety.md" "$ctx/"
cp -a "$tva/docs" "$ctx/docs"
# the example: the claim folder (without its GPU check: the containers have no GPU), Claim.lean's sha256 in
# claim.json, and its complete proof
mkdir "$ctx/example"
(cd "$tva/claims/vec_add_f32" && tar --exclude=__pycache__ --exclude=check_gpu.py --exclude='Proof*' -cf - .) \
  | tar -xf - -C "$ctx/example"
python3 -I - "$ctx/example" <<'EOF'
import hashlib, json, os, sys
d = sys.argv[1]
cj = json.load(open(os.path.join(d, "claim.json")))
cj["claim_sha256"] = hashlib.sha256(open(os.path.join(d, cj["lean"]), "rb").read()).hexdigest()
json.dump(cj, open(os.path.join(d, "claim.json"), "w"), indent=2)
EOF
if ! python3 -I "$here/oracle.py" vec_add_f32 "$ctx/example"; then
  echo "the worked example's proof is not available (oracle.py failed): the claim image needs it" >&2
  exit 1
fi
sub="$(cd "$ctx" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -c1-8)"
tag="tva3-claim:${base#tva3-base:}-$sub"
DOCKER_BUILDKIT=1 docker build --build-arg "BASE=$base" -t "$tag" "$ctx" >&2
echo "built $tag (from $base)" >&2
echo "$tag"
