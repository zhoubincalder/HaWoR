#!/usr/bin/env bash
# Finish the corpus publish, sending the two oversized archives as chunks.
#
# lakectl 1.80.0 / lakeFS 1.79.0 refuses an object once multipart needs more
# than 1000 parts. Observed: 17.15e9 and 19.46e9 bytes upload fine, 46.08e9
# does not, which puts the ceiling near 20e9 bytes -- so hot3d at 22.32e9 is
# over it too. (That is also why HOT3D uploaded fine before: at 1000 clips the
# archive was 14.8GB, and at 1516 clips it is 22.3GB.)
#
# ho3d (15.59e9) is under the ceiling and is left to the running job. This
# waits for it, then stops that job BEFORE it burns ~28 minutes failing on
# hot3d as a single object, and sends hot3d and dexycb through upload_chunked.sh
# instead.
set -uo pipefail
cd "$(dirname "$0")/.."

PUB=${1:-1666113}
LOG=datasets/publish_chunked.log
: >"$LOG"
say(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "waiting for ho3d_export upload to finish (publish pid $PUB)"
while kill -0 "$PUB" 2>/dev/null; do
  grep -aq '^\[ok  \] ho3d_export.tar.zst' datasets/publish.log && break
  grep -aq '^\[FAIL\] ho3d_export.tar.zst' datasets/publish.log && { say "ho3d FAILED; it needs chunking too"; break; }
  sleep 30
done

if kill -0 "$PUB" 2>/dev/null; then
  say "stopping publish job before its single-object hot3d attempt"
  pkill -P "$PUB" -f lakectl 2>/dev/null
  kill "$PUB" 2>/dev/null
  sleep 5
  kill -9 "$PUB" 2>/dev/null
  say "stopped"
else
  say "publish job already exited"
fi
grep -aE '^\[(ok|skip|FAIL)' datasets/publish.log | tail -8 | tee -a "$LOG"

for n in hot3d_clips_export dexycb_bronze_export; do
  a="datasets/_archives/$n.tar.zst"
  [ -f "$a" ] || { say "FAILED: $a missing"; exit 1; }
  say "=== chunked upload: $n ($(numfmt --to=iec "$(stat -c%s "$a")")) ==="
  bash scripts/upload_chunked.sh "$a" >>"$LOG" 2>&1 \
    || { say "FAILED: $n"; exit 1; }
  say "$n done"
done

# ho3d may have been mid-flight when the job was stopped; the uploader is
# idempotent by sha, so re-running it costs nothing if it already landed.
say "=== sweep: any single-object archives still missing ==="
bash scripts/upload_corpus_lakefs.sh >>"$LOG" 2>&1 || say "WARNING: sweep reported a failure"

say "=== branch contents ==="
lakectl fs ls lakefs://calder-dev/hawor-stage-a/datasets/hawor/stage-a/ 2>&1 \
  | grep -v sha256 | awk '{print $(NF-2), $(NF-1), $NF}' | tee -a "$LOG"
say "ALL DONE"
