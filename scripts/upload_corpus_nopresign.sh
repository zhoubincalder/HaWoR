#!/usr/bin/env bash
# Upload corpus archives whole, via the non-presigned path, skipping by sha256.
#
# Generalises upload_remaining_nopresign.sh, which was hardcoded to hot3d and
# dexycb. --pre-sign=false because lakectl's default presigned multipart
# validates partNumber <= 1000 inconsistently: it refused 46.08e9 bytes (asked
# part 3107) and a 10e9-byte chunk (part 1104), while ho3d went through at
# 15.59e9 bytes using 2974 parts. The non-presigned path proxies through the
# lakeFS server and does no part numbering, and has carried 22.32e9 and 46.08e9
# whole. Slower (~7.5 MB/s measured), and worth it for not depending on a rule
# that has already varied.
#
# Skips an archive whose remote sha256 sidecar already matches the local one, so
# a re-run after re-packing only sends what actually changed. The manifest and
# the .count/.fingerprint sidecars are always refreshed: they are tiny, and a
# branch that describes itself wrongly is what this whole exercise is about.
set -uo pipefail
cd "$(dirname "$0")/.."

REPO="${REPO:-lakefs://calder-dev}"
BRANCH="${BRANCH:-hawor-stage-a}"
PREFIX="${PREFIX:-datasets/hawor/stage-a}"
SRC="${SRC:-datasets/_archives}"
WAIT_PID="${1:-}"
LOG=datasets/upload_arctic.log
: >"$LOG"
say(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

if [ -n "$WAIT_PID" ]; then
  say "waiting for packing (pid $WAIT_PID)"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  say "packing finished"
fi

MAN="$SRC/CORPUS_MANIFEST.json"
[ -f "$MAN" ] || { say "FAILED: no manifest"; exit 1; }

put(){ lakectl fs upload --pre-sign=false -s "$1" "$2" >>"$LOG" 2>&1; }

say "[put ] CORPUS_MANIFEST.json"
put "$MAN" "$REPO/$BRANCH/$PREFIX/CORPUS_MANIFEST.json" || { say "FAILED: manifest"; exit 1; }

fail=0
for a in "$SRC"/*.tar.zst; do
  [ -e "$a" ] || continue
  n=$(basename "$a")
  # Only archives archive_corpus.sh verified by sequence count are eligible.
  [ -f "$a.done" ] || { say "[skip] $n not verified by archive_corpus.sh"; continue; }
  want=$(cat "$a.sha256" 2>/dev/null | tr -dc '0-9a-f' | head -c 64)
  have=$(lakectl fs cat "$REPO/$BRANCH/$PREFIX/$n.sha256" 2>/dev/null | tr -dc '0-9a-f' | head -c 64)
  if [ -n "$want" ] && [ "$want" = "$have" ]; then
    say "[skip] $n unchanged (sha matches)"
  else
    [ -n "$have" ] && say "       $n differs from remote, replacing"
    say "[put ] $n $(numfmt --to=iec "$(stat -c%s "$a")")"
    if put "$a" "$REPO/$BRANCH/$PREFIX/$n"; then
      got=$(lakectl fs stat "$REPO/$BRANCH/$PREFIX/$n" 2>/dev/null \
            | awk -F': *' '/^Size/{print $2}' | tr -dc 0-9)
      if [ "$got" = "$(stat -c%s "$a")" ]; then say "[ok  ] $n verified $got bytes"
      else say "[FAIL] $n remote $got != local $(stat -c%s "$a")"; fail=1; continue; fi
    else
      say "[FAIL] $n upload"; fail=1; continue
    fi
  fi
  for sc in sha256 count fingerprint; do
    [ -f "$a.$sc" ] && put "$a.$sc" "$REPO/$BRANCH/$PREFIX/$n.$sc"
  done
done

say "=== byte-for-byte check of every archive on the branch ==="
for a in "$SRC"/*.tar.zst; do
  n=$(basename "$a")
  r=$(lakectl fs stat "$REPO/$BRANCH/$PREFIX/$n" 2>/dev/null \
      | awk -F': *' '/^Size/{print $2}' | tr -dc 0-9)
  l=$(stat -c%s "$a")
  st=$([ "$r" = "$l" ] && echo MATCH || echo DIFFER)
  [ "$st" = "DIFFER" ] && fail=1
  printf '%-30s remote=%-13s local=%-13s %s\n' "$n" "${r:-none}" "$l" "$st" | tee -a "$LOG"
done

if [ "$fail" -eq 0 ]; then
  tot=$(uv run python -c "
import json;m=json.load(open('$MAN'))['totals_usable_frames']
print(f\"{m['train']} train / {m['val']} val / {m['test']} test usable frames\")" 2>/dev/null)
  lakectl commit "$REPO/$BRANCH" -m "HaWoR stage-A corpus: 6 datasets, $tot (ARCTIC intrinsics fixed)" >>"$LOG" 2>&1 \
    && say "committed" || say "WARNING: commit failed"
  say "ALL DONE"
else
  say "FAILED: not committing, some archive does not match local"
  exit 1
fi
