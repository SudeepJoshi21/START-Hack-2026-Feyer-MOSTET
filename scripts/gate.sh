#!/bin/bash
# Full correctness gate for the committed code (judging-criteria.pdf, "Correctness"). Run on a compute node:
#
#   srun --partition=short --cpus-per-task=2 --mem=48G --time=00:44:00 bash scripts/gate.sh
#
# Env: REPEATS (official benchmark runs with the XLA options on, default 5), ORIGINAL_NPZ (--save-arrays output of
# the unmodified forked code, commit 02ec741; generated into .gate/ if not given), SCENARIOS=0 to skip the
# 16-scenario check. Exit status 0 only if every step passes.
#
#   1. code state: committed, benchmark.py and fixtures identical to the official repository blobs
#   2. benchmark.py (unchanged) with the XLA options off, then REPEATS times on; every run checked by
#      scripts/check_correctness.py against all four criteria + the original code's arrays
#   3. determinism: all runs bitwise identical (flags on vs off, and run to run)
#   4. tests/test_adjoint.py, tests/test_host.py, tests/test_robustness.py (generalisation / anti-overfitting)
#   5. scripts/scenarios.py check (criteria on 16 other inputs)
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT="$ROOT/.gate"
mkdir -p "$OUT"
REPEATS=${REPEATS:-5}
FAIL=0
step() { echo; echo "=== $*"; }
bad() { echo "GATE FAIL: $*"; FAIL=1; }

step "1. code state"
echo "commit $(git rev-parse HEAD)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || bad "uncommitted changes to tracked files"
[ "$(git rev-parse HEAD:benchmark.py)" = "39f1fd3053554d7894762c0595105ec47b322049" ] || bad "benchmark.py differs from the official blob"
echo "ff4ba353a2a03cdce7f3e61bad729c2e00888d842c28ba9cf699f95afb87e138  examples/data/uifo_800_3000.json
726f2cfe6986cb991fa48d5a28127b9330c3dc7e69d5bd193c8748acef03adc3  examples/data/uifo_800_3000_test_data.json" | sha256sum -c --quiet || bad "public fixtures changed"

if [ -z "${ORIGINAL_NPZ:-}" ]; then
  ORIGINAL_NPZ="$OUT/original-02ec741.npz"
  if [ ! -f "$ORIGINAL_NPZ" ]; then
    step "original code arrays (commit 02ec741, ~2 min)"
    rm -rf "$OUT/original"; git worktree add --detach "$OUT/original" 02ec741 >/dev/null
    (cd "$OUT/original" && unset XLA_FLAGS && python benchmark.py --cpus 1 --repeats 3 --output "$OUT/original.json" --save-arrays "$ORIGINAL_NPZ" | tail -1)
    git worktree remove --force "$OUT/original"
  fi
fi

step "2. official benchmark runs, each checked against all criteria"
run() {  # $1 label, $2 DIFFEROMETOR_XLA_OPT0
  env -u XLA_FLAGS DIFFEROMETOR_XLA_OPT0=$2 python benchmark.py --cpus 1 --repeats 3 --output "$OUT/$1.json" --save-arrays "$OUT/$1.npz" \
    | grep -E "first-run|^PASS" || bad "$1: benchmark.py did not PASS"
  python scripts/check_correctness.py "$OUT/$1.json" "$OUT/$1.npz" "$ORIGINAL_NPZ" | grep -E "FAIL|RESULT|\|ln S" || bad "$1: check_correctness"
  python scripts/check_correctness.py "$OUT/$1.json" "$OUT/$1.npz" "$ORIGINAL_NPZ" >/dev/null || bad "$1: criteria"
}
run flags-off 0
for i in $(seq 1 "$REPEATS"); do run flags-on-$i 1; done

step "3. determinism (bitwise identical outputs)"
python - "$OUT" "$REPEATS" <<'EOF' || bad "outputs not bitwise identical"
import sys, numpy as np
out, n = sys.argv[1], int(sys.argv[2])
ref = np.load(f"{out}/flags-on-1.npz")
ok = True
for name in ["flags-off"] + [f"flags-on-{i}" for i in range(2, n + 1)]:
    other = np.load(f"{out}/{name}.npz")
    same = all(np.array_equal(ref[k], other[k]) for k in ref.files)
    print(f"  flags-on-1 vs {name}: {'bitwise identical' if same else 'DIFFERENT'}")
    ok &= same
sys.exit(0 if ok else 1)
EOF

step "4. tests"
PYTHONPATH=. python tests/test_adjoint.py 2>&1 | tail -5 | tee "$OUT/test_adjoint.log" | grep -q "PASS test_signal_pattern_wraps_indices" || bad "test_adjoint"
cat "$OUT/test_adjoint.log"
PYTHONPATH=. timeout 1500 python -u tests/test_host.py > "$OUT/test_host.log" 2>&1; tail -2 "$OUT/test_host.log"
grep -q "ALL PASS" "$OUT/test_host.log" || bad "test_host"
PYTHONPATH=. timeout 1500 python -u tests/test_robustness.py > "$OUT/test_robustness.log" 2>&1; grep "^PASS\|ALL PASS\|Error" "$OUT/test_robustness.log" | cut -c1-120
grep -q "ALL PASS" "$OUT/test_robustness.log" || bad "test_robustness"

if [ "${SCENARIOS:-1}" != "0" ]; then
  step "5. scenarios (criteria on 16 other inputs)"
  python scripts/scenarios.py check | tee "$OUT/scenarios.log" | cut -c1-120
  grep -q "RESULT: ALL SCENARIOS PASS" "$OUT/scenarios.log" || bad "scenarios"
fi

echo
if [ $FAIL = 0 ]; then
  # Ledger read by scripts/pre-push (git hook): only simulator states that passed this gate may be pushed.
  LEDGER="$(git rev-parse --git-common-dir)/gate-passed"
  echo "$(git rev-parse HEAD:differometor) $(git rev-parse HEAD:benchmark.py) $(git rev-parse HEAD) $(date -u +%FT%TZ) REPEATS=$REPEATS SCENARIOS=${SCENARIOS:-1}" >> "$LEDGER"
  echo "GATE: ALL PASS (commit $(git rev-parse --short HEAD)); recorded in $LEDGER"
else
  echo "GATE: FAILED"
fi
exit $FAIL
