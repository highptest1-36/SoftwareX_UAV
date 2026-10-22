#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Resume check: interrupt + resume in SEPARATE processes (real process boundary); the resumed trace must equal the
# uninterrupted reference trace (tools/trace_capture.py --tag $REF) in every field. CPU only.
#   REF=ref setsid nohup bash tools/run_resume_check.sh > logs/resume_check.log 2>&1 < /dev/null &
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
REF=${REF:-ref}
chk() {  # level method every stop
  python3 -u tools/resume_check.py --phase interrupt --level $1 --method $2 --every $3 --stop $4
  python3 -u tools/resume_check.py --phase resume --level $1 --method $2 --every $3 --ref "$REF"
}
chk GC mean_teacher 5 12   # resume at frame 10, after the sequence boundary at frame 8 (Adam state, EMA teacher)
chk GC whw 5 12            # running feature statistics + SGD momentum
chk GC tent 5 12           # Adam state, teacher = student
echo "== RESUME CHECK DONE"
