# Pick the lookback arm for the recency probe (folds 40-42).
# Rule (pre-committed, ledger 2026-08-28): lb200 only if the frozen round-1
# criterion says lb200 beats the lb90 baseline (99% CI all>0 AND >=3/4 folds
# same direction, i.e. verdict_round1 == true in compare_arms output).
# Any other outcome (tie, missing file, parse error) -> lb90, consistent with
# the pre-registered minimum-lookback tiebreak.
import json
from pathlib import Path

lb = "90"
try:
    r = json.loads(Path(r"outputs\screen_lookback_round1.json").read_text(encoding="utf-8"))
    if r.get("lb200", {}).get("verdict_round1") is True:
        lb = "200"
except Exception:
    pass
print(lb)
