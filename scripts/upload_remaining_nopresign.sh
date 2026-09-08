#!/usr/bin/env bash
# Upload the two remaining archives whole, via the non-presigned path.
#
# WHY NOT PRESIGNED. lakectl's default --pre-sign=true uploads through
# presigned part URLs, and that endpoint validates partNumber <= 1000:
#
#     parameter "partNumber" in path has an error: number must be at most 1000
#
# It refused dexycb (46.08e9 B, asked part 3107) and a 10e9 B chunk (asked part
# 1104). The implied part sizes -- 14.8MB and 9.06MB -- are not consistent with
# each other, nor with ho3d, which SUCCEEDED at 15.59e9 B using 2974 parts. So
# the cap is real but not a predictable function of size, and choosing a "safe"
# chunk size means guessing against behaviour that has already varied.
#
# --pre-sign=false proxies through the lakeFS server and does no part
# numbering: a 10e9 B object went through and came back with a single-part
# checksum (no "-N" suffix). That removes the constraint rather than working
# around it, so chunking is unnecessary and the archives go up whole.
#
# Slower than presigned (measured ~8.8 MB/s), which is the price of not
# splitting and reassembling 68GB.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-lakefs://calder-dev}"
BRANCH="${BRANCH:-hawor-stage-a}"
PREFIX="${PREFIX:-datasets/hawor/stage-a}"
LOG=datasets/upload_nopresign.log
: >"$LOG"
say(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

# The chunk experiment left one object behind; it is not part of the layout.
say "removing leftover chunk object from the abandoned chunked attempt"
lakectl fs rm "$REPO/$BRANCH/$PREFIX/hot3d_clips_export.tar.zst.part00" >>"$LOG" 2>&1 \
  && say "  removed .part00" || say "  .part00 not present"

fail=0
for n in hot3d_clips_export dexycb_bronze_export; do
  a="datasets/_archives/$n.tar.zst"
  [ -f "$a" ] || { say "FAILED: $a missing"; exit 1; }
  want=$(stat -c%s "$a")
  have=$(lakectl fs stat "$REPO/$BRANCH/$PREFIX/$n.tar.zst" 2>/dev/null \
         | awk -F': *' '/^Size/{print $2}' | tr -dc 0-9)
  if [ "${have:-0}" = "$want" ]; then
    say "[skip] $n already $want bytes"; continue
  fi
  say "[put ] $n $(numfmt --to=iec "$want")  (remote has ${have:-none})"
  if lakectl fs upload --pre-sign=false -s "$a" \
       "$REPO/$BRANCH/$PREFIX/$n.tar.zst" >>"$LOG" 2>&1; then
    lakectl fs upload --pre-sign=false -s "$a.sha256" \
      "$REPO/$BRANCH/$PREFIX/$n.tar.zst.sha256" >>"$LOG" 2>&1
    lakectl fs upload --pre-sign=false -s "$a.count" \
      "$REPO/$BRANCH/$PREFIX/$n.tar.zst.count" >>"$LOG" 2>&1
    got=$(lakectl fs stat "$REPO/$BRANCH/$PREFIX/$n.tar.zst" 2>/dev/null \
          | awk -F': *' '/^Size/{print $2}' | tr -dc 0-9)
    if [ "$got" = "$want" ]; then
      say "[ok  ] $n verified $got bytes"
    else
      say "[FAIL] $n uploaded but remote is $got, expected $want"; fail=1
    fi
  else
    say "[FAIL] $n upload"; fail=1
  fi
done

say "=== byte-for-byte check of every archive on the branch ==="
for f in arctic_export dexycb_bronze_export h2o_export h2o3d_export ho3d_export hot3d_clips_export; do
  r=$(lakectl fs stat "$REPO/$BRANCH/$PREFIX/$f.tar.zst" 2>/dev/null \
      | awk -F': *' '/^Size/{print $2}' | tr -dc 0-9)
  l=$(stat -c%s "datasets/_archives/$f.tar.zst" 2>/dev/null)
  st=$([ "$r" = "$l" ] && echo MATCH || echo DIFFER)
  [ "$st" = "DIFFER" ] && fail=1
  printf '%-26s remote=%-13s local=%-13s %s\n' "$f" "${r:-none}" "${l:-none}" "$st" | tee -a "$LOG"
done

if [ "$fail" -eq 0 ]; then
  say "committing"
  tot=$(uv run python -c "
import json;m=json.load(open('datasets/_archives/CORPUS_MANIFEST.json'))['totals_usable_frames']
print(f\"{m['train']} train / {m['val']} val / {m['test']} test usable frames\")" 2>/dev/null)
  lakectl commit "$REPO/$BRANCH" -m "HaWoR stage-A corpus: 6 datasets, $tot" >>"$LOG" 2>&1 \
    && say "committed" || say "WARNING: commit failed"
  say "ALL DONE"
else
  say "FAILED: not committing, some archive does not match local"
  exit 1
fi
