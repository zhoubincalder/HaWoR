#!/usr/bin/env bash
# Re-acquire the 516 HOT3D clips lost to a stale LakeFS archive, after the
# bronze build finishes.
#
# WHY THIS EXISTS: the local hot3d_clips_export (1516 clips) was deleted on the
# strength of a LakeFS backup that turned out to be the Sep 7 snapshot -- 1000
# clips, taken before the expansion. The tarballs were verified sha256-equal to
# the local _archives copies, but both were the SAME stale artifact, so that
# check proved integrity and not content. The clips are re-downloadable; the
# lesson is that a restore must be verified by clip count and id range.
#
# Waits on the build's PID rather than a pgrep pattern: a `pgrep -f
# <script>.sh` loop deadlocked earlier today because the watcher's own command
# line matched the pattern it was searching for.
set -uo pipefail
cd "$(dirname "$0")/.."

BUILD_PID=${1:-1618215}
LOG=datasets/hot3d_redo.log
: >"$LOG"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "waiting for bronze build (pid $BUILD_PID) to finish"
while kill -0 "$BUILD_PID" 2>/dev/null; do sleep 60; done
if grep -aq "ALL DONE" datasets/bronze_build.log 2>/dev/null; then
  say "bronze build finished OK"
else
  say "WARNING: bronze build did not report ALL DONE -- continuing anyway,"
  say "         HOT3D recovery is independent of it"
  grep -aE "FAILED|Traceback" datasets/bronze_build.log 2>/dev/null | tail -3 | tee -a "$LOG"
fi

say "=== 1/4: download clips 2849-3364 (~52GB) ==="
uv run python scripts/download_hot3d_clips.py --lo 2849 --hi 3364 \
  >>"$LOG" 2>&1 || { say "FAILED: download"; exit 1; }
n=$(ls datasets/hot3d_clips/train_aria/*.tar 2>/dev/null | wc -l)
bad=$(find datasets/hot3d_clips/train_aria -name '*.tar' -size -10M 2>/dev/null | wc -l)
say "tars: $n (expected 516), undersized: $bad"
if [ "$n" -lt 516 ] || [ "$bad" -ne 0 ]; then
  say "FAILED: incomplete download"; exit 1
fi

say "=== 2/4: convert (the restored 1000 already have anno.pth and skip) ==="
uv run python lib/datasets/hot3d_clips_to_export.py \
  --clips_dir datasets/hot3d_clips/train_aria \
  --out_root datasets/hot3d_clips_export \
  --workers 8 --split all >>"$LOG" 2>&1 \
  || { say "FAILED: convert"; exit 1; }

say "=== 3/4: preprocess ==="
uv run python lib/datasets/hawor_preprocess_train.py \
  --video_root datasets/hot3d_clips_export --set_file all.json >>"$LOG" 2>&1 \
  || { say "FAILED: preprocess"; exit 1; }

clips=$(ls -d datasets/hot3d_clips_export/clip-* 2>/dev/null | wc -l)
npz=$(ls datasets/hot3d_clips_export/*/train_anno.npz 2>/dev/null | wc -l)
say "clips: $clips (expected 1516)   train_anno.npz: $npz"
if [ "$clips" -ne 1516 ] || [ "$npz" -ne 1516 ]; then
  say "FAILED: expected 1516 converted clips"; exit 1
fi

say "=== 4/4: subject-disjoint split (P0015 held out) ==="
uv run python scripts/split_hot3d_by_subject.py --write >>"$LOG" 2>&1 \
  || { say "FAILED: split"; exit 1; }
grep -aE "^train|^val|subject overlap|val fraction" "$LOG" | tail -5 | tee -a "$LOG"

say "ALL DONE"
say "NOT re-uploaded to LakeFS: the stale archive there still says 1000 clips."
say "Re-archive before deleting anything locally, and verify the new archive by"
say "clip count and id range, not just sha256."
df -h /home/bzhou | tail -1 | tee -a "$LOG"
