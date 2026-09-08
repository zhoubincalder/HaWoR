#!/usr/bin/env bash
# Wait for the HOT3D clip download, then convert + preprocess only what is new.
#
# Deliberately writes the manifest to all.json and does NOT touch train.json /
# val.json: those were split by SOURCE RECORDING to keep the folds
# recording-disjoint, and the new clips belong to recordings that may already
# sit on one side. Re-splitting is a separate, considered step.
set -uo pipefail

echo "waiting for the downloader..."
while pgrep -f "download_hot3d_clips.py" >/dev/null 2>&1; do sleep 60; done

n=$(ls datasets/hot3d_clips/train_aria/*.tar 2>/dev/null | wc -l)
echo "downloader exited with $n / 1516 clips"
grep -aE "ALL DONE|FAIL" datasets/hot3d_dl.log | tail -3
if [ "$n" -lt 1516 ]; then
  echo "WARNING: incomplete ($((1516-n)) missing). Converting what is present;"
  echo "re-run the downloader to fill the rest -- both steps are idempotent."
fi

echo "=== convert (skips the ~1000 already done) ==="
uv run python lib/datasets/hot3d_clips_to_export.py \
  --clips_dir datasets/hot3d_clips/train_aria \
  --out_root datasets/hot3d_clips_export \
  --workers 8 --split all || { echo "CONVERT FAILED"; exit 1; }

echo "=== preprocess (skips already-processed sequences) ==="
uv run python lib/datasets/hawor_preprocess_train.py \
  --video_root datasets/hot3d_clips_export --set_file all.json \
  || { echo "PREPROCESS FAILED"; exit 1; }

echo "ALL DONE"
ls -d datasets/hot3d_clips_export/*/ | wc -l | sed 's/^/converted sequences: /'
find datasets/hot3d_clips_export -name train_anno.npz | wc -l | sed 's/^/train_anno.npz:      /'
