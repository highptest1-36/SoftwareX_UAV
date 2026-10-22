#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# GPU runbook of the experiments in docs/RESULTS.md (sections 2-4) for the four shipped profiles.
#   ASSETS=/path/assets IMAGES=/path/UAV-benchmark-M COCO=/path/coco OUT=experiments/runs \
#     setsid nohup bash experiments/run_experiments.sh > logs/experiments.log 2>&1 < /dev/null &
# ASSETS holds checkpoints/source_day_r50_1080/{model_final.pth,source_stats.pt}; COCO holds test_night.json and
# test_day.json (docs/INSTALL.md). Every step is resume-safe: completed runs are skipped, interrupted runs resume from
# their last checkpoint. JOBS runs share one GPU (each needs < 2 GB); time per frame measured under sharing is not a
# profiling result. Results were produced with a bf16-capable GPU (A100); other precisions give other numbers.
# Afterwards: python3 tools/summarize_runs.py --root $OUT --out results/runs_summary.json
#             python3 tools/results_report.py --summary results/runs_summary.json --out docs/RESULTS.md
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ASSETS:?set ASSETS}" "${IMAGES:?set IMAGES}" "${COCO:?set COCO}"
OUT=${OUT:-experiments/runs}; JOBS=${JOBS:-4}
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src
mkdir -p "$OUT" logs/runs
log() { echo "== $(date -u +%Y-%m-%dT%H:%M:%S) $*"; }
run() {  # name method seed out [extra uavsl-run args]
  local name=$1 m=$2 seed=$3 out=$4; shift 4
  log "start $name"
  if python3 -u -m uav_shiftlab.cli run --method "$m" --coco "$COCO/test_night.json" --images "$IMAGES" \
       --assets-root "$ASSETS" --seed "$seed" --out "$out" --checkpoint-every 250 --allow-test "$@" \
       > "logs/runs/$name.log" 2>&1
  then log "done  $name: $(grep '^\[done\]' "logs/runs/$name.log" | tail -1 | sed 's/.*: AP50/AP50/')"
  else log "!! FAILED $name (see logs/runs/$name.log)"; fi
}
pool() {  # reads "name|method seed out [args]" lines from stdin, runs JOBS at a time
  while IFS='|' read -r name args; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n || true; done
    # shellcheck disable=SC2086
    run "$name" $args &
    sleep 20   # stagger model loading
  done
  wait || true
}

# E1: main runs (3 seeds; source only is deterministic given the stream) with adapter snapshots for E5
{
  echo "e1_source_only_s0|source_only 0 $OUT/e1_seeds/source_only_s0 --snapshot-every 1000"
  for s in 0 1 2; do for m in mean_teacher tent whw; do
    echo "e1_${m}_s$s|$m $s $OUT/e1_seeds/${m}_s$s --snapshot-every 1000"
  done; done
} | pool

# E5: fixed-reference forgetting of the seed-0 runs (test_day, every 5th frame)
for m in source_only mean_teacher tent whw; do
  log "forgetting $m"
  python3 -u tools/forgetting_eval.py --run-dir "$OUT/e1_seeds/${m}_s0" --assets-root "$ASSETS" \
    --coco-dir "$COCO" --images "$IMAGES"
done

# E2: deliberate protocol violations; E4: other stream orders
{
  for m in mean_teacher tent; do
    echo "e2_abp_$m|$m 0 $OUT/e2_violations/viol_abp/$m --violation adapt_before_predict"
    echo "e2_reset_$m|$m 0 $OUT/e2_violations/viol_reset/$m --violation reset_each_sequence"
    echo "e2_ate_$m|$m 0 $OUT/e2_violations/viol_ate/$m --frozen --init-adapters $OUT/e1_seeds/${m}_s0/snapshots/teacher_005035.pth"
  done
  for o in reverse perm:1 perm:2 perm:3 perm:4; do for m in source_only mean_teacher tent whw; do
    echo "e4_${m}_${o/:/}|$m 0 $OUT/e4_order/${m}_${o/:/} --order $o"
  done; done
} | pool
log "ALL DONE"
