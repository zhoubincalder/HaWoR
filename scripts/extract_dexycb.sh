#!/usr/bin/env bash
# Extract the DexYCB archives into the tree dexycb_to_export.py expects.
#
# Skips aligned_depth_to_color_*.png: roughly a third of every archive is depth
# we never read, and the converter only touches color_*.jpg, labels_*.npz,
# pose.npz and meta.yml.
set -uo pipefail
SRC="${1:-datasets/dexycb_download}"
DST="${2:-datasets/dexycb}"
mkdir -p "$DST"

# calibration/ holds the per-camera intrinsics, per-session extrinsics and
# per-subject MANO betas; the converter refuses to start without it.
if [ ! -d "$DST/calibration" ]; then
  tar xzf "$SRC/calibration.tar.gz" -C "$DST" && echo "[ok] calibration"
fi

for f in "$SRC"/*subject*.tar.gz; do
  [ -e "$f" ] || continue
  name=$(basename "$f" .tar.gz)
  if [ -d "$DST/$name" ] && [ -f "$DST/$name/.complete" ]; then
    echo "[skip] $name"
    continue
  fi
  echo "[get ] $name"
  tar xzf "$f" -C "$DST" \
      --wildcards --no-wildcards-match-slash \
      --exclude='*/aligned_depth_to_color_*.png' \
    && touch "$DST/$name/.complete" \
    && echo "[ok  ] $name  $(du -sh "$DST/$name" | cut -f1)"
done
echo "ALL DONE"
du -sh "$DST"
