#!/usr/bin/env bash
# Wall-clock slope measurement for ONE training configuration.
#
# Runs the same config at two step counts and takes the slope:
#
#     t = (wall_hi - wall_lo) / (steps_hi - steps_lo)
#
# Startup, weight loading, compilation and the sanity-val pass are constant
# between the two runs, so the subtraction cancels them and leaves the true
# per-step cost.
#
# This exists because tqdm's it/s is not good enough to compare configs: it
# reports two decimals, so 0.04 it/s spans 22-28.6 s/step, and the reading count
# is not the step count (a 40-step run emitted 83 readings), which invalidates
# fitting a rate curve against the reading index.
#
# Usage: NAME=x scripts/bench_one.sh KEY VALUE [KEY VALUE ...]
set -uo pipefail
cd "$(dirname "$0")/.."

NAME="${NAME:-bench}"
OUT="${OUT:-/tmp/claude-1001/-home-bzhou-ws-HaWoR/c87c54eb-645e-49f2-b7c5-227a139bc1a4/scratchpad/$NAME}"
LO="${LO:-20}"; HI="${HI:-60}"; BS="${BS:-4}"
# Frames per window. Hardcoding 16 here silently doubled the reported
# frames/s for a SEQ_LEN 8 run, so it is a parameter.
SEQ="${SEQ:-16}"
mkdir -p "$OUT"
echo "$NAME: batch $BS x seq_len $SEQ = $((BS*SEQ)) frames/step, "\
     "slope from $LO vs $HI steps, opts: $*"

# Warm the inductor cache FIRST. torch.compile persists its artifacts to disk,
# so an un-warmed pair has the first run paying full compilation and the second
# reusing it -- which broke the slope badly enough to produce a NEGATIVE s/step
# (20 steps 201.6s, 60 steps 136.0s). The slope only cancels startup if startup
# is genuinely equal between the two runs.
if [ ! -f "$OUT/.warm" ]; then
  echo "  warming compile cache (throwaway 2-step run)..."
  timeout 3600 uv run python train_full.py \
    --cfg hawor/configs/hawor_full_sapiens2_1024.yaml \
    --video_root datasets/hot3d_clips_export --exp_name "${NAME}_warm" \
    --out_dir "$OUT/run" --max_steps 2 --limit_val_batches 1 \
    --opts TRAIN.BATCH_SIZE "$BS" MODEL.WARM_START "" "$@" >"$OUT/warm.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "  warm-up FAILED rc=$rc"
    grep -aoE "torch.OutOfMemoryError[^.]*\.|Tried to allocate [0-9.]+ [MG]iB" "$OUT/warm.log" | head -2 | sed 's/^/      /'
    exit 1
  fi
  touch "$OUT/.warm"
fi

for n in "$LO" "$HI"; do
  log="$OUT/${n}.log"
  t0=$(date +%s.%N)
  timeout 3600 uv run python train_full.py \
    --cfg hawor/configs/hawor_full_sapiens2_1024.yaml \
    --video_root datasets/hot3d_clips_export --exp_name "${NAME}_${n}" \
    --out_dir "$OUT/run" --max_steps "$n" --limit_val_batches 1 \
    --opts TRAIN.BATCH_SIZE "$BS" MODEL.WARM_START "" "$@" >"$log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  w=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  if [ $rc -ne 0 ]; then
    echo "  @${n} steps: FAILED rc=$rc after ${w}s"
    grep -aoE "torch.OutOfMemoryError[^.]*\.|OutOfMemoryError|Tried to allocate [0-9.]+ [MG]iB|GPU 0 has a total capacity[^.]*\." "$log" | head -3 | sed 's/^/      /'
    grep -aE "Traceback|Error:" "$log" | tail -2 | sed 's/^/      /'
    exit 1
  fi
  echo "$w" > "$OUT/${n}.wall"
  echo "  @${n} steps: ${w}s"
done

wlo=$(cat "$OUT/${LO}.wall"); whi=$(cat "$OUT/${HI}.wall")
awk -v a="$wlo" -v b="$whi" -v lo="$LO" -v hi="$HI" -v bs="$BS" -v sq="$SEQ" -v nm="$NAME" \
  'BEGIN{t=(b-a)/(hi-lo);
   printf "\n%s: %.3f s/step   %.1f frames/s   (%d frames/step, startup+compile ~%.0fs)\n", \
          nm, t, bs*sq/t, bs*sq, a-lo*t}'
echo DONE
