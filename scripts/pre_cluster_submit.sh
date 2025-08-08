#!/usr/bin/env bash
# Pre-cluster helper: tidy artifacts, then (optionally) submit a job
# - Never deletes source code
# - Prints suggested git commands (no push by default)
# - With --submit, actually runs bsub on the specified job script
#
# Usage:
#   bash scripts/pre_cluster_submit.sh                             # tidy only (apply)
#   bash scripts/pre_cluster_submit.sh --keep 5                    # tidy, keep 5 backups
#   bash scripts/pre_cluster_submit.sh --submit jobs/jobs_cams_complete_rl.sh
#   bash scripts/pre_cluster_submit.sh --submit <job.sh> --keep 5

set -euo pipefail

KEEP=10
SUBMIT=false
JOB_SCRIPT="jobs/jobs_cams_complete_rl.sh"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep)
      KEEP="$2"; shift 2 ;;
    --submit)
      SUBMIT=true
      if [[ $# -ge 2 && "$2" != --* ]]; then
        JOB_SCRIPT="$2"; shift 2
      else
        shift 1
      fi ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# 1) Housekeeping (apply)
if [[ -x scripts/housekeeping.sh ]]; then
  echo "[1/3] Running housekeeping (apply, keep=$KEEP)..."
  bash scripts/housekeeping.sh --apply --keep "$KEEP"
else
  echo "[1/3] Running housekeeping (python fallback, apply, keep=$KEEP)..."
  python3 scripts/housekeeping.py --apply --keep "$KEEP"
fi

# 2) Git status and suggested commands (no push)
echo "[2/3] Git status and suggested commands:"
if command -v git >/dev/null 2>&1; then
  git status -s || true
  echo "Suggested (review before running):"
  echo "  git add -A"
  echo "  git commit -m 'housekeeping: archive artifacts before job submit'"
  echo "  git push origin HEAD"
else
  echo "git not available in PATH; skipping"
fi

# 3) Optional submission
if [[ "$SUBMIT" == true ]]; then
  if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Job script not found: $JOB_SCRIPT" >&2
    exit 1
  fi
  echo "[3/3] Submitting job: $JOB_SCRIPT"
  # Respect environment modules if present
  if command -v module >/dev/null 2>&1; then
    module load cuda/12.1 || true
  fi
  # Submit
  bsub < "$JOB_SCRIPT"
else
  echo "[3/3] Skipping submission. To submit:"
  echo "  bsub < $JOB_SCRIPT"
fi

echo "Done."

