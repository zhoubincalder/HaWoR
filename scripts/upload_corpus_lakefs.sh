#!/usr/bin/env bash
# Upload the corpus archives to LakeFS with lakectl.
#
# NOT `calder data upload`: that has a short write timeout and dies ~32s into a
# 1.7 GB object. lakectl does multipart properly. Note also that `calder data
# upload` does not URL-encode branch names, so a branch containing a slash
# 404s on the object endpoint -- hence the flat name below.
#
# Idempotent, but keyed on the LOCAL sha256 recorded beside each archive rather
# than on object size. Size was the earlier test and it is not safe here: the
# HOT3D object already on the branch is a 1000-clip snapshot that has to be
# replaced by a 1516-clip archive, and two different corpora can coincide in
# size. The manifest is uploaded first so the branch never describes more than
# it holds.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-lakefs://calder-dev}"
BRANCH="${BRANCH:-hawor-stage-a}"
PREFIX="${PREFIX:-datasets/hawor/stage-a}"
SRC="${1:-datasets/_archives}"
MAN="$SRC/CORPUS_MANIFEST.json"

[ -f "$MAN" ] || { echo "no manifest at $MAN -- run archive_corpus.sh first"; exit 1; }

remote_sha() {  # object path -> the .sha256 sidecar already on the branch
  lakectl fs cat "$1.sha256" 2>/dev/null | tr -dc '0-9a-f' | head -c 64
}

echo "[put ] CORPUS_MANIFEST.json"
lakectl fs upload -s "$MAN" "$REPO/$BRANCH/$PREFIX/CORPUS_MANIFEST.json" >/dev/null \
  || { echo "[FAIL] manifest"; exit 1; }

fail=0
for a in "$SRC"/*.tar.zst; do
  [ -e "$a" ] || continue
  n=$(basename "$a")
  [ -f "$a.done" ] || { echo "[skip] $n not verified by archive_corpus.sh"; continue; }
  want=$(cat "$a.sha256" 2>/dev/null | tr -dc '0-9a-f' | head -c 64)
  have=$(remote_sha "$REPO/$BRANCH/$PREFIX/$n")
  if [ -n "$want" ] && [ "$want" = "$have" ]; then
    echo "[skip] $n already uploaded (sha matches)"; continue
  fi
  [ -n "$have" ] && echo "       replacing existing object (sha differs)"
  echo "[put ] $n $(numfmt --to=iec "$(stat -c%s "$a")")"
  if lakectl fs upload -s "$a" "$REPO/$BRANCH/$PREFIX/$n"; then
    # publish the checksum alongside, so a consumer can verify a pull and the
    # next run of this script can tell what is already there
    lakectl fs upload -s "$a.sha256" "$REPO/$BRANCH/$PREFIX/$n.sha256" >/dev/null
    lakectl fs upload -s "$a.count" "$REPO/$BRANCH/$PREFIX/$n.count" >/dev/null 2>&1
    echo "[ok  ] $n"
  else
    echo "[FAIL] $n"; fail=1
  fi
done

tot=$(uv run python -c "
import json;m=json.load(open('$MAN'))['totals_usable_frames']
print(f\"{m['train']} train / {m['val']} val / {m['test']} test usable frames\")" 2>/dev/null)
lakectl commit "$REPO/$BRANCH" -m "HaWoR stage-A corpus: 6 datasets, $tot" 2>&1 | tail -3

echo
lakectl fs ls "$REPO/$BRANCH/$PREFIX/" 2>&1 | grep -E "tar.zst$|MANIFEST"
[ "$fail" -eq 0 ] && echo "ALL DONE" || { echo "FAILED: one or more uploads"; exit 1; }
