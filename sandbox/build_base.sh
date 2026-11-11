#!/usr/bin/env bash
# Build tva3-base:<hash>: the Lean toolchain, Mathlib and the TVA library, built (base/Dockerfile).
#   sandbox/build_base.sh
# <hash> is a content hash of the library's source (lean/: lakefile.toml, lake-manifest.json, lean-toolchain,
# TVA.lean, TVA/**.lean), taken from a snapshot of it, so the image holds exactly the hashed files. Mathlib
# comes from lean/.lake/packages and the toolchain from the host's elan: nothing is downloaded.
# Prints the tag on its last line. The library must build: a module that fails stops the build.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tva="$(cd "$here/.." && pwd)"
lean="$tva/lean"
toolchain_name="$(tr -d '[:space:]' < "$lean/lean-toolchain")"          # leanprover/lean4:v4.35.0
toolchain="${ELAN_HOME:-$HOME/.elan}/toolchains/$(echo "$toolchain_name" | sed 's#/#--#; s#:#---#')"
[ -x "$toolchain/bin/lean" ] || { echo "no Lean toolchain $toolchain_name at $toolchain" >&2; exit 1; }
[ -d "$lean/.lake/packages/mathlib/.lake/build" ] || { echo "Mathlib is not built in $lean/.lake/packages" >&2; exit 1; }

ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/lean"
(cd "$lean" && cp -a lakefile.toml lake-manifest.json lean-toolchain TVA.lean TVA "$ctx/lean/")
find "$ctx/lean/TVA" -type f ! -name '*.lean' -delete
hash="$(python3 -I "$here/lean_hash.py" "$ctx/lean")"
tag="tva3-base:$hash"
cp "$here/base/Dockerfile" "$ctx/Dockerfile"
DOCKER_BUILDKIT=1 docker build \
  --build-context lean="$ctx/lean" \
  --build-context packages="$lean/.lake/packages" \
  --build-context toolchain="$toolchain" \
  -t "$tag" "$ctx" >&2
echo "built $tag" >&2
echo "$tag"
