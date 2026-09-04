#!/usr/bin/env bash
# Download DexYCB from the public Google Drive links on https://dex-ycb.github.io/
# Licence: CC BY-NC 4.0 (non-commercial). Resumable and idempotent -- re-run any time.
#
# Drive rate-limits individual files ("Too many users have viewed or downloaded
# this file recently"), and in practice the limit trips after a few tens of GB
# and then applies to whatever you ask for next. It clears on its own, but over
# hours -- not the 30s a naive retry waits. So this sweeps the outstanding list
# repeatedly with a long sleep between rounds rather than giving up on a file.
#
# gdown rather than curl: for files this large Drive serves a virus-scan
# interstitial whose confirm token is dynamic, so `&confirm=t` silently yields
# a 2 KB HTML page saved under a .tar.gz name.
#
# bop.tar.gz is skipped on purpose (BOP object-pose format, no hand labels).
# dex-ycb-20210415.tar.gz is skipped too: the per-frame labels_*.npz, per-
# sequence pose.npz and meta.yml all ship inside the subject archives, so that
# 127 GB archive is redundant for training.
set -uo pipefail
OUT="${1:-datasets/dexycb_download}"
ROUND_SLEEP="${ROUND_SLEEP:-1800}"     # between full sweeps, when work remains
MAX_ROUNDS="${MAX_ROUNDS:-48}"         # 48 x 30min = 24h
mkdir -p "$OUT"

FILES=(
  "calibration.tar.gz:1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ"
  "models.tar.gz:1cAzlQBpcTatI5ykYQ8ziQiHLUG_a_UpM"
  "20200709-subject-01.tar.gz:1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk"
  "20200813-subject-02.tar.gz:1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62"
  "20200820-subject-03.tar.gz:1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq"
  "20200903-subject-04.tar.gz:14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA"
  "20200908-subject-05.tar.gz:1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS"
  "20200918-subject-06.tar.gz:1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ"
  "20200928-subject-07.tar.gz:1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0"
  "20201002-subject-08.tar.gz:1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv"
  "20201015-subject-09.tar.gz:1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH"
  "20201022-subject-10.tar.gz:1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-"
)

# A subject archive is ~12 GB; anything under 1 GB that claims to be one is a
# truncated or interstitial download. calibration is legitimately tiny.
complete() {
  local f="$OUT/$1"
  [ -s "$f" ] || return 1
  file -b "$f" | grep -q gzip || return 1
  case "$1" in
    calibration.tar.gz) return 0 ;;
    models.tar.gz)      [ "$(stat -c%s "$f")" -gt 1400000000 ] ;;
    *)                  [ "$(stat -c%s "$f")" -gt 1000000000 ] ;;
  esac
}

for round in $(seq 1 "$MAX_ROUNDS"); do
  remaining=0 throttled=0
  for e in "${FILES[@]}"; do
    name="${e%%:*}"; id="${e#*:}"
    complete "$name" && continue
    remaining=$((remaining+1))
    echo "[get ] round $round: $name"
    # capture the WHOLE message: gdown prints "Too many users..." near the top
    # of a ~10 line error, so tailing it loses the one line worth matching on
    out=$(uv run gdown --continue "https://drive.google.com/uc?id=$id" -O "$OUT/$name" 2>&1)
    if echo "$out" | grep -q "Too many users"; then
      echo "  throttled by Drive"
      throttled=$((throttled+1))
      # gdown leaves nothing usable after this failure; a stray tiny file would
      # otherwise poison the next --continue
      [ -f "$OUT/$name" ] && [ "$(stat -c%s "$OUT/$name")" -lt 1000000 ] && rm -f "$OUT/$name"
      sleep 60
      continue
    fi
    if complete "$name"; then
      echo "[ok  ] $name $(du -h "$OUT/$name" | cut -f1)"
      remaining=$((remaining-1))
    else
      echo "  incomplete, will retry next round"
      echo "$out" | tail -2
    fi
  done
  if [ "$remaining" -eq 0 ]; then
    echo "ALL DONE after round $round"; break
  fi
  echo "--- round $round: $remaining outstanding ($throttled throttled), sleeping ${ROUND_SLEEP}s ---"
  sleep "$ROUND_SLEEP"
done

echo "=== final ==="
for e in "${FILES[@]}"; do
  name="${e%%:*}"
  complete "$name" && echo "  ok      $name" || echo "  MISSING $name"
done
du -sh "$OUT"
