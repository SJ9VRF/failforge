#!/usr/bin/env bash
set -euo pipefail
echo "FAILFORGE final package"
echo "This script validates the executable mock bridge and points to existing executed results."
echo
echo "1) Validate BrowserGym bridge core logic"
cd failforge_browsergym_bridge
PYTHONPATH=. python scripts/run_mock_bridge.py
PYTHONPATH=. python - <<'PY'
from failforge_bg.sampler import AdaptiveFailureSampler
from failforge_bg.verifier import TrajectoryVerifier
from failforge_bg.failure import FailureClassifier
from failforge_bg.types import Transition
s=AdaptiveFailureSampler(["a","b","c"])
assert abs(sum(s.probabilities().values())-1)<1e-8
v=TrajectoryVerifier()
assert v.score([Transition({}, "x", 1, True, False, {})], True)[0] > v.score([Transition({}, "x", 1, True, False, {"shortcut_used": True})], True)[0]
c=FailureClassifier()
assert "missing_verification" in {e.kind for e in c.classify("x", [Transition({}, "x", 0, True, False, {"verification_required": True})], False)}
print("core validation passed")
PY
cd ..
echo
echo "2) Executed result files are under failforge_visual/ and failforge_package/"
echo "3) Public MiniWoB run requires browsergym-miniwob + Playwright Chromium in a networked runtime."
