#!/usr/bin/env bash
set -euo pipefail

backup_dir="backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

# Move experiment artifacts
for d in results logs wandb evaluation_results; do
  if [ -d "$d" ]; then
    mv "$d" "$backup_dir/" || true
  fi
done

# Move common scheduler outputs
shopt -s nullglob
mv ./*.out ./*.err "$backup_dir/" 2>/dev/null || true
shopt -u nullglob

# Move large JSON/logs from root
for f in *.json *.log; do
  [ -e "$f" ] && mv "$f" "$backup_dir/" || true
done

# Summary
echo "Moved artifacts to $backup_dir"

