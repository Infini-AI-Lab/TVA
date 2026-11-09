#!/bin/bash
# Build the claim base image tva-claim-base:<commit> from tva-sandbox-base:<commit> (../build_base.sh), where
# <commit> is the last TVA commit that changed anything outside sandbox/ and claims/.
#   sandbox/claim/build_base.sh
# The example is claims/vec_add_f32 (without its GPU check) with the proof oracle.py derives from TVA's
# certified proof of vec_add_f32; the image build checks it with `tva-claim check`.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TVA=$(cd "$HERE/../.." && pwd)
REV=$(git -C "$TVA" log -1 --format=%h -- . ':(exclude)sandbox' ':(exclude)claims')
docker image inspect "tva-sandbox-base:$REV" >/dev/null || { echo "build tva-sandbox-base:$REV first (sandbox/build_base.sh)"; exit 1; }
CTX=$(mktemp -d)
trap 'rm -rf "${CTX:?}"' EXIT
cp "$HERE/Dockerfile" "$HERE/tva-claim" "$HERE/settings.json" "$HERE/lean-fixer.md" "$HERE/CLAUDE.md" \
   "$HERE/process_safety.md" "$CTX/"
cp "$HERE/../harness/lake" "$CTX/lake"
# the library's lakefile with only the library and the claim
python3 -I - "$TVA/lean/lakefile.toml" "$CTX/lakefile.toml" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
head = text.split('[[lean_lib]]\nname = "TVA"\n')[0]
head = re.sub(r'^defaultTargets = \[.*\]$', 'defaultTargets = ["TVA"]', head, flags=re.M)
open(sys.argv[2], "w").write(head + '[[lean_lib]]\nname = "TVA"\n\n'
    '# the claim (/app/claim): the fixed Claim.lean and the proof, Proof.lean and Proof/\n'
    '[[lean_lib]]\nname = "ClaimTask"\nsrcDir = "../claim"\nroots = ["Claim", "Proof"]\n')
EOF
# the example: the claim folder, its sha256 in claim.json, and its proof
mkdir "$CTX/example"
(cd "$TVA/claims/vec_add_f32" && tar --exclude=__pycache__ --exclude=check_gpu.py -cf - .) | tar -xf - -C "$CTX/example"
python3 -I - "$CTX/example" <<'EOF'
import hashlib, json, os, sys
d = sys.argv[1]
cj = json.load(open(os.path.join(d, "claim.json")))
cj.pop("check", None)
cj["claim_sha256"] = hashlib.sha256(open(os.path.join(d, "Claim.lean"), "rb").read()).hexdigest()
json.dump(cj, open(os.path.join(d, "claim.json"), "w"), indent=2)
EOF
python3 -I "$HERE/oracle.py" vec_add_f32 "$CTX/example"
docker build --build-arg "BASE=tva-sandbox-base:$REV" -t "tva-claim-base:$REV" "$CTX"
echo "built tva-claim-base:$REV"
