#!/bin/bash
# The verifier's entry point: verify.py writes the reward; a crash before it does leaves 0.
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.json /logs/verifier/reward.txt
python3 -I /tests/verify.py 2>&1 | tee /logs/verifier/verify.log
test -s /logs/verifier/reward.json || echo '{"reward": 0}' > /logs/verifier/reward.json
