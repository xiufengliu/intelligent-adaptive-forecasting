#!/usr/bin/env bash
# Safe housekeeping script: moves experiment artifacts into backup/<timestamp>
# Never deletes source code. Supports dry-run by default.
# Usage:
#   bash scripts/housekeeping.sh            # dry-run
#   bash scripts/housekeeping.sh --apply    # perform actions
#   bash scripts/housekeeping.sh --keep 5   # keep last 5 backups (prune older)
#   bash scripts/housekeeping.sh --apply --keep 5

set -euo pipefail

APPLY=false
KEEP=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=true
      shift
      ;;
    --keep)
      KEEP="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

timestamp() { date +%Y%m%d_%H%M%S; }

ROOT_DIR="$(pwd)"
BACKUP_DIR="${ROOT_DIR}/backup/$(timestamp)"

# Artifact globs (dirs or file patterns)
ARTIFACTS=(
  "logs"
  "results"
  "evaluation_results"
  "checkpoints"
  "wandb"
  "*.out"
  "*.err"
  "*.log"
)

mkdir -p "${BACKUP_DIR}"

echo "Housekeeping (dry-run=${APPLY,false})"
print_move() { echo "- move $1 -> ${BACKUP_DIR}/"; }

move_path() {
  local path="$1"
  if compgen -G "$path" > /dev/null; then
    for p in $path; do
      if [[ -e "$p" ]]; then
        print_move "$p"
        if [[ "$APPLY" == true ]]; then
          mv -v "$p" "${BACKUP_DIR}/" 2>/dev/null || mv -v "$p" "${BACKUP_DIR}/$(basename "$p")"
        fi
      fi
    done
  fi
}

for glob in "${ARTIFACTS[@]}"; do
  move_path "$glob"
  # handle nested under results/
  move_path "results/${glob}"
  # handle nested under logs/
  move_path "logs/${glob}"
  # handle any stray at repo root
  move_path "./${glob}"
done

# Prune old backups keeping most recent $KEEP
PRUNE_DIR="${ROOT_DIR}/backup"
if [[ -d "$PRUNE_DIR" ]]; then
  mapfile -t backups < <(ls -1dt "$PRUNE_DIR"/* 2>/dev/null || true)
  if (( ${#backups[@]} > KEEP )); then
    to_delete=("${backups[@]:KEEP}")
    echo "Pruning old backups (keep=$KEEP):"
    for d in "${to_delete[@]}"; do
      echo "- rm -rf $d"
      if [[ "$APPLY" == true ]]; then
        rm -rf "$d"
      fi
    done
  fi
fi

echo "Done. Backup directory: ${BACKUP_DIR}"

