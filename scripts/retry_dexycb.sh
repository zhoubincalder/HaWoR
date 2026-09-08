#!/usr/bin/env bash
# Finish the DexYCB bronze conversion after the HOT3D recovery releases the link.
#
# The first attempt died at 1806 of 7200 clips with SIGSEGV inside
# pyarrow 25.0.1's Table.filter() -- 465,536 rows, 4 chunks of wide
# fixed_size_list columns. The crash killed the interpreter with no traceback,
# and because Python buffers stdout when it is not a TTY, the log showed the
# stage starting and then only "FAILED", with nothing in between. Diagnosing it
# needed the exit code (139), not the log.
#
# bronze_to_export.py now indexes clip_id -> rows once and uses Table.take,
# which does not crash and is also O(rows) instead of O(clips x rows). It also
# resumes, so the 1806 completed clips are reused rather than redone and only
# the shards not yet reached get fetched.
set -uo pipefail
cd "$(dirname "$0")/.."

WAIT_PID=${1:-}
LOG=datasets/dexycb_retry.log
: >"$LOG"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

if [ -n "$WAIT_PID" ]; then
  say "waiting for pid $WAIT_PID (HOT3D recovery) to release the network"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  say "pid $WAIT_PID gone"
fi

pre=$(ls -d datasets/dexycb_bronze_export/*/ 2>/dev/null | wc -l)
say "=== 1/3: convert dexycb (resuming from $pre clips) ==="
# -u so a crash cannot swallow the progress output again.
uv run python -u lib/datasets/bronze_to_export.py \
  --project dexycb --out_root datasets/dexycb_bronze_export \
  --bronze_root datasets/_bronze/dexycb --splits train valid >>"$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then say "FAILED: convert exited $rc"; exit 1; fi
n=$(ls -d datasets/dexycb_bronze_export/*/ 2>/dev/null | wc -l)
say "clips: $n (expected 7200)"
if [ "$n" -ne 7200 ]; then say "FAILED: expected 7200 clips"; exit 1; fi

say "=== 2/3: preprocess ==="
uv run python -u lib/datasets/hawor_preprocess_train.py \
  --video_root datasets/dexycb_bronze_export --set_file all.json >>"$LOG" 2>&1 \
  || { say "FAILED: preprocess"; exit 1; }
npz=$(ls datasets/dexycb_bronze_export/*/train_anno.npz 2>/dev/null | wc -l)
say "train_anno.npz: $npz"

say "=== 3/3: attach bronze's subject-disjoint split ==="
uv run python -u scripts/split_bronze_by_source.py \
  --export_root datasets/dexycb_bronze_export \
  --bronze_root datasets/_bronze/dexycb --write >>"$LOG" 2>&1 \
  || { say "FAILED: split"; exit 1; }
grep -aE "^train |^val |subject overlap" "$LOG" | tail -4 | tee -a "$LOG"

say "ALL DONE"
df -h /home/bzhou | tail -1 | tee -a "$LOG"
