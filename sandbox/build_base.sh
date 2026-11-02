#!/usr/bin/env bash
# Build tva-sandbox-base from TVA's tree at the last commit that changed anything outside sandbox/: the
# files TVA ships then (git archive), never its history. A commit that changes only the sandbox (which a task
# copies from the working tree) needs no new base image.
#   sandbox/build_base.sh [TAG]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
tva="$(cd "$here/.." && pwd)"
rev="$(git -C "$tva" log -1 --format=%h -- . ':(exclude)sandbox')"
tag="${1:-tva-sandbox-base:$rev}"
if [ -n "$(git -C "$tva" status --porcelain -- . ':(exclude)sandbox')" ]; then
  echo "TVA has uncommitted changes outside sandbox/; commit them first" >&2; exit 1
fi
ctx="$(mktemp -d)"
trap 'rm -rf "$ctx"' EXIT
mkdir -p "$ctx/tva"
# The tracked files at that commit, exactly (no build output, no .git).
git -C "$tva" archive --format=tar "$rev" | tar -x -C "$ctx/tva"
cp "$here/base/Dockerfile" "$ctx/Dockerfile"
echo "$rev" > "$ctx/tva/TVA_COMMIT"
docker build -t "$tag" "$ctx"
docker tag "$tag" tva-sandbox-base:latest
echo "built $tag (TVA $rev)"
