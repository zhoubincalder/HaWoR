#!/usr/bin/env bash
# Restore HOT3D from LakeFS, then convert the two bronze datasets worth taking.
#
# Serialized deliberately: all three stages are network-bound on the same link,
# so running them together just makes each slower and the failures harder to
# read. Order is restore -> ho3d -> dexycb: cheapest first, so a broken
# assumption surfaces in 55 clips rather than 7200.
#
# HOT3D is RESTORED rather than converted from bronze. Bronze's hot3d_rect has
# 175,050 MANO-valid frames against the 227,400 already converted here, because
# its 93,000-frame valid split carries mano_valid=0 on both hands. See
# lib/datasets/bronze_to_export.py.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=datasets/bronze_build.log
LAKE=lakefs://calder-dev/hawor-stage-a/datasets/hawor/stage-a
: >"$LOG"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "=== stage 1/3: restore hot3d_clips_export from LakeFS ==="
if [ -d datasets/hot3d_clips_export ] && [ -f datasets/hot3d_clips_export/all.json ]; then
  say "already present, skipping"
else
  mkdir -p datasets/_restore
  if [ ! -f datasets/_restore/hot3d_clips_export.tar.zst ]; then
    lakectl fs download "$LAKE/hot3d_clips_export.tar.zst" \
      datasets/_restore/hot3d_clips_export.tar.zst >>"$LOG" 2>&1 \
      || { say "FAILED: download"; exit 1; }
  fi
  # Verify before unpacking; the .sha256 rides alongside the tarball on LakeFS.
  lakectl fs download "$LAKE/hot3d_clips_export.tar.zst.sha256" \
    datasets/_restore/expected.sha256 >>"$LOG" 2>&1
  want=$(awk '{print $1}' datasets/_restore/expected.sha256)
  got=$(sha256sum datasets/_restore/hot3d_clips_export.tar.zst | awk '{print $1}')
  if [ "$want" != "$got" ]; then
    say "FAILED: sha256 mismatch (want $want got $got)"; exit 1
  fi
  say "sha256 ok, extracting"
  tar -I zstd -xf datasets/_restore/hot3d_clips_export.tar.zst -C datasets/ \
    >>"$LOG" 2>&1 || { say "FAILED: extract"; exit 1; }
  rm -rf datasets/_restore
fi
say "hot3d clips: $(ls -d datasets/hot3d_clips_export/clip-* 2>/dev/null | wc -l)"

say "=== stage 2/3: convert ho3d ==="
uv run python lib/datasets/bronze_to_export.py \
  --project ho3d --out_root datasets/ho3d_export \
  --splits train valid >>"$LOG" 2>&1 || { say "FAILED: ho3d convert"; exit 1; }
uv run python lib/datasets/hawor_preprocess_train.py \
  --video_root datasets/ho3d_export --set_file all.json >>"$LOG" 2>&1 \
  || { say "FAILED: ho3d preprocess"; exit 1; }
say "ho3d sequences: $(ls datasets/ho3d_export/*/train_anno.npz 2>/dev/null | wc -l)"

say "=== stage 3/3: convert dexycb (7200 clips, ~46GB streamed) ==="
uv run python lib/datasets/bronze_to_export.py \
  --project dexycb --out_root datasets/dexycb_bronze_export \
  --splits train valid >>"$LOG" 2>&1 || { say "FAILED: dexycb convert"; exit 1; }
uv run python lib/datasets/hawor_preprocess_train.py \
  --video_root datasets/dexycb_bronze_export --set_file all.json >>"$LOG" 2>&1 \
  || { say "FAILED: dexycb preprocess"; exit 1; }
say "dexycb sequences: $(ls datasets/dexycb_bronze_export/*/train_anno.npz 2>/dev/null | wc -l)"

say "ALL DONE"
df -h /home/bzhou | tail -1 | tee -a "$LOG"
