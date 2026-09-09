#!/usr/bin/env bash
# Tar+zstd each export tree into one archive per dataset, for LakeFS upload.
#
# zstd -3 with all cores: the payload is already-compressed JPEG, so the ratio
# is only a few percent and the point of this step is producing ONE file per
# dataset, not saving space.
#
# VERIFICATION. The previous version accepted an archive if it listed at least
# 10 entries. That is how a 1000-clip HOT3D snapshot passed as the 1516-clip
# corpus: the tarball was sha256-identical to its local twin, both were stale,
# and 78,000 frames had to be re-downloaded. A checksum proves a transfer, not
# a content. So each archive is now listed back and its sequence-directory
# count compared against CORPUS_MANIFEST.json, which is measured from the live
# export. A mismatch fails the dataset instead of marking it done.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-datasets/_archives}"
MAN="$OUT/CORPUS_MANIFEST.json"
mkdir -p "$OUT"

# Superseded 4-of-10-subject DexYCB download is excluded on purpose: it is a
# convention reference only, and shipping it beside dexycb_bronze_export would
# invite training on both and duplicate subjects 01/02/03/06.
DATASETS="${DATASETS:-hot3d_clips_export dexycb_bronze_export arctic_export ho3d_export h2o_export h2o3d_export}"

echo "[man ] measuring exports -> $MAN"
uv run python scripts/corpus_manifest.py "$MAN" || { echo "[FAIL] manifest"; exit 1; }

expected_for() {  # dataset dir -> "<sequences> <content_fingerprint>"
  uv run python - "$MAN" "$1" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]))
for v in m['datasets'].values():
    if v['export_dir']==sys.argv[2]:
        print(v['sequences_on_disk'], v.get('content_fingerprint','')); break
else: print(-1, '')
PY
}

fail=0
for d in $DATASETS; do
  [ -d "datasets/$d" ] || { echo "[skip] $d absent"; continue; }
  a="$OUT/$d.tar.zst"
  read -r want fp <<<"$(expected_for "$d")"
  if [ -z "$want" ] || [ "$want" = "-1" ]; then
    echo "[FAIL] $d: not in manifest"; fail=1; continue
  fi
  if [ -f "$a.done" ]; then
    # Trusted only if BOTH the sequence count and the content fingerprint still
    # match. Count alone is not enough: ARCTIC's intrinsics fix rewrote every
    # train_anno.npz while leaving all 301 sequences in place, so a count check
    # would have skipped it and left the remote holding the wrong labels.
    prev=$(cat "$a.count" 2>/dev/null | tr -dc 0-9)
    prevfp=$(cat "$a.fingerprint" 2>/dev/null | tr -dc 0-9a-f)
    if [ "${prev:-0}" = "$want" ] && [ "${prevfp:-x}" = "$fp" ]; then
      echo "[skip] $d already archived ($want seq, fingerprint matches)"; continue
    fi
    if [ -z "$prevfp" ]; then
      # First run after content_fingerprint was introduced: absence is not
      # evidence of change, so do not claim the content changed.
      echo "[re  ] $d has no recorded fingerprint, re-archiving to establish one"
    elif [ "${prev:-0}" = "$want" ]; then
      echo "[re  ] $d same $want seq but CONTENT CHANGED, re-archiving"
    else
      echo "[re  ] $d changed ($prev -> $want seq), re-archiving"
    fi
    rm -f "$a" "$a.done" "$a.sha256" "$a.count" "$a.fingerprint"
  fi
  echo "[pack] $d ($(du -sh "datasets/$d" | cut -f1), $want seq)"
  tar -I 'zstd -3 -T0' -cf "$a" -C datasets "$d" \
    || { echo "[FAIL] $d: tar"; fail=1; continue; }
  # Count sequence directories inside the archive, not just entries.
  got=$(tar -I zstd -tf "$a" 2>/dev/null \
        | sed -n "s|^$d/\([^/]*\)/$|\1|p" | sort -u | wc -l)
  if [ "$got" -ne "$want" ]; then
    echo "[FAIL] $d: archive holds $got sequence dirs, export has $want"
    fail=1; continue
  fi
  sha256sum "$a" | awk '{print $1}' > "$a.sha256"
  echo "$want" > "$a.count"
  echo "$fp" > "$a.fingerprint"
  touch "$a.done"
  echo "[ok  ] $d -> $(du -h "$a" | cut -f1), $got seq verified, sha $(cut -c1-12 "$a.sha256")"
done

echo
du -ch "$OUT"/*.tar.zst 2>/dev/null | tail -1
[ "$fail" -eq 0 ] && echo "ALL DONE" || { echo "FAILED: one or more datasets"; exit 1; }
