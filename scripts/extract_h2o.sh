#!/usr/bin/env bash
# Extract the H2O egocentric archives into the tree h2o_to_export.py expects.
#
# Keeps rgb/ (1280x720 PNG), hand_pose_mano/, hand_pose/, cam_pose/ and
# cam_intrinsics.txt. Skips depth/, rgb256/ (a 256px duplicate of rgb),
# obj_pose*/, action_label/ and verb_label/ -- none are read by the converter,
# and together they are most of the bytes.
set -uo pipefail
SRC="${1:-datasets/h2o}"
DST="${2:-datasets/h2o_ego}"
mkdir -p "$DST"

for f in "$SRC"/subject*_ego_v1_1.tar.gz; do
  [ -e "$f" ] || continue
  name=$(basename "$f" _ego_v1_1.tar.gz)
  if [ -f "$DST/.${name}.complete" ]; then
    echo "[skip] $name"; continue
  fi
  echo "[get ] $name"
  tar xzf "$f" -C "$DST" \
      --wildcards \
      --exclude='*/depth/*' --exclude='*/rgb256/*' \
      --exclude='*/obj_pose/*' --exclude='*/obj_pose_rt/*' \
      --exclude='*/action_label/*' --exclude='*/verb_label/*' \
    && touch "$DST/.${name}.complete" \
    && echo "[ok  ] $name  $(du -sh "$DST/${name}_ego" 2>/dev/null | cut -f1)"
done
# the official pose_train / pose_val / pose_test frame lists
[ -d "$DST/label_split" ] || unzip -q -o "$SRC/label_split.zip" -d "$DST"
echo "ALL DONE"
du -sh "$DST"
