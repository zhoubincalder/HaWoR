#!/usr/bin/env bash
# Archive every converted export, verify each by sequence count, then upload.
# Serialized: archiving is disk/CPU-bound and uploading is network-bound, but
# a half-written archive must never be uploaded, so the gate matters more than
# the overlap.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=datasets/publish.log
: >"$LOG"
say(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

say "=== 1/2: archive + verify ==="
bash scripts/archive_corpus.sh >>"$LOG" 2>&1 || { say "FAILED: archive"; exit 1; }
say "archives: $(ls datasets/_archives/*.tar.zst 2>/dev/null | wc -l), $(du -ch datasets/_archives/*.tar.zst 2>/dev/null | tail -1 | cut -f1)"

say "=== 2/2: upload to lakefs ==="
bash scripts/upload_corpus_lakefs.sh >>"$LOG" 2>&1 || { say "FAILED: upload"; exit 1; }
say "ALL DONE"
df -h /home/bzhou | tail -1 | tee -a "$LOG"
