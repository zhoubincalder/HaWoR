#!/usr/bin/env bash
# Short training run per dataset: does the full-frame model actually step on it.
#
# scripts/smoke_datasets.py checks what the dataset class hands the model.
# This checks the rest: collate, the backbone at 768x1024, the MANO head, the
# losses, and the optimizer step. A fold can pass every tensor check and still
# produce a non-finite loss the first time it reaches the 3D or reprojection
# term.
#
# One process per dataset, deliberately. Sharing an interpreter across configs
# gave a 3.2x wrong throughput measurement earlier in this project, and here it
# would also let one dataset's CUDA state mask another's failure.
#
# Not a benchmark: 12 steps is far too few to say anything about loss values.
# The question is only whether each dataset can be trained on at all.
set -uo pipefail
cd "$(dirname "$0")/.."

CFG="${CFG:-hawor/configs/hawor_full_sapiens2_1024.yaml}"
STEPS="${STEPS:-12}"
OUT="${OUT:-datasets/smoke_logs}"
# Appended as train_full.py --opts, which is argparse.REMAINDER and so must be
# last. e.g. OPTS="MODEL.NATIVE_RES True TRAIN.BATCH_SIZE 1"
OPTS="${OPTS:-}"
LOG=datasets/smoke_train.log
: >"$LOG"
mkdir -p "$OUT"
say(){ echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

declare -A ROOTS=(
  [hot3d]=datasets/hot3d_clips_export
  [dexycb]=datasets/dexycb_bronze_export
  [arctic]=datasets/arctic_export
  [ho3d]=datasets/ho3d_export
  [h2o]=datasets/h2o_export
  [h2o3d]=datasets/h2o3d_export
)
ORDER="${ORDER:-ho3d h2o3d h2o arctic hot3d dexycb}"

say "config $CFG, $STEPS steps per dataset${OPTS:+, opts: $OPTS}"
fail=0
for d in $ORDER; do
  r="${ROOTS[$d]}"
  [ -d "$r" ] || { say "[skip] $d: $r missing"; continue; }
  l="$OUT/$d.log"
  say "=== $d ==="
  # WARM_START is left as configured: starting from the frozen-backbone run is
  # what a real run does, and a checkpoint that fails to load is itself worth
  # catching here.
  timeout 2400 uv run python train_full.py \
    --cfg "$CFG" \
    --video_root "$r" \
    --exp_name "smoke_$d" \
    --out_dir "$OUT" \
    --max_steps "$STEPS" \
    --limit_val_batches 2 \
    ${OPTS:+--opts $OPTS} >"$l" 2>&1
  rc=$?
  # A crash that takes the interpreter down leaves no traceback and, with
  # buffered stdout, no output either -- so the exit code is checked first.
  if [ $rc -ne 0 ]; then
    say "[FAIL] $d exited $rc"
    grep -aE "Error|Traceback|assert|CUDA|out of memory" "$l" | tail -4 | sed 's/^/    /' | tee -a "$LOG"
    fail=1
    continue
  fi
  # Losses: Lightning prints them on the progress bar; pull the last one seen.
  last=$(grep -aoE "loss[^ ,]*=[0-9.eE+-]+" "$l" | tail -3 | tr '\n' ' ')
  if grep -aqiE "\bnan\b|\binf\b" <<<"$last"; then
    say "[FAIL] $d non-finite loss: $last"; fail=1; continue
  fi
  # `--` or grep reads the "->" pattern as an option bundle and dies.
  win=$(grep -aoE -- "-> [0-9]+ windows" "$l" | head -1)
  # Record the input mode and the observed step rate. A run that silently fell
  # back to the fixed canvas would otherwise look identical in this summary,
  # and the rate is the whole point of NATIVE_RES.
  sz=$(grep -aoE "frames at [^(]*" "$l" | head -1 | sed 's/frames at //;s/, *$//;s/ *$//')
  rate=$(grep -aoE "[0-9.]+(it/s|s/it)" "$l" | tail -1)
  say "[ok  ] $d  $win  [$sz]  ${rate:-no rate}  ${last:-no loss line captured}"
done

say "=== summary ==="
grep -aE "^\S+ \[(ok|FAIL|skip)" "$LOG" | sed 's/^[0-9:]* //' | tee -a "$LOG"
if [ $fail -eq 0 ]; then say "ALL DATASETS TRAINABLE"; else say "FAILED: see above"; exit 1; fi
